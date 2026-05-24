from typing import Tuple

import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk

from relax.algorithm.mala_sampler import build_mala_sampler
from relax.algorithm.mgmd_types import (
    Diffv2OptStates,
    HParams,
    Diffv2TrainState,
    MGMDConfig,
)
from relax.algorithm.value_head import ValueHead
from relax.network.actor_critic import ActorCritic, ActorCriticParams
from relax.utils.experience import Experience
from relax.utils.jax_utils import (
    delayed_param_update,
    delayed_target_update,
    stack_trees,
    unstack_tree,
)
from relax.utils.typing_utils import Metric


def _split_info_vmap(info):
    scalar_keys = []
    scalar_vals = []
    array_info = {}
    for k, v in info.items():
        if jnp.ndim(v) == 1:
            scalar_keys.append(k)
            scalar_vals.append(v)
        else:
            array_info[k] = np.asarray(v)
    if scalar_keys:
        stacked = np.asarray(jnp.stack(scalar_vals))
        scalar_info = {k: np.asarray(stacked[i]) for i, k in enumerate(scalar_keys)}
    else:
        scalar_info = {}
    return scalar_info, array_info


class MGMD:

    def __init__(self, model: ActorCritic, params: ActorCriticParams, cfg: MGMDConfig,
                 *, obs_dim: int, hidden_dim: int):
        self.model = model
        self.cfg = cfg
        self._obs_dim = int(obs_dim)
        self._hidden_dim = int(hidden_dim)
        # Derived/exposed flags. Everything else lives on ``self.cfg``; the
        # two attributes below are also read off the algorithm by the
        # trainer (``algorithm.on_policy_ema`` / ``algorithm.one_step_dist_shift_beta``).
        self.on_policy_ema = (cfg.kl_budget is not None)
        self.one_step_dist_shift_beta = bool(cfg.one_step_dist_shift_beta)
        self.policy_loss_key = "losses/Policy_epsilon_MSE"

        # --- Optimizers: unscaled Adam; per-seed state.lr_{q,policy} is applied at update time. ---
        self.optim = optax.scale_by_adam()
        self.policy_optim = optax.scale_by_adam()

        # --- Optional V(s) network for KL-budget / on-policy-EMA beta adaptation and logging. ---
        value_params_init, value_opt_state_init = self._setup_value_network(params)

        self._timesteps = int(self.model.num_timesteps)

        # --- Initial vmap-stackable train state ---
        self.state = self._build_initial_state(params, value_params_init, value_opt_state_init)

        # --- Schedule cache for wandb SNR x-axis ---
        self._alphas_cumprod = np.asarray(self.model.schedule.alphas_cumprod)  # [T]
        self._snr = self._alphas_cumprod / np.maximum(1.0 - self._alphas_cumprod, 1e-8)

        # --- Stateless update / sampler closures (jit-able, vmap-able). ---
        # Build the MALA sampler; body lives in relax.algorithm.mala_sampler.
        _sampler_kw = dict(
            model=self.model, value_head=self.value_head, timesteps=self._timesteps,
            batch_independent_guidance=self.cfg.batch_independent_guidance,
            mala_guided_predictor=self.cfg.mala_guided_predictor,
            mala_no_predictor=self.cfg.mala_no_predictor,
        )
        sampler = build_mala_sampler(**_sampler_kw)
        # Both sampling paths (rollout + TD next-action) use --q_agg_sample aggregation.
        # The TD-backup target itself remains hardcoded to 'min' (clipped double-Q).
        agg_critic = lambda qm: _aggregate_q(qm, self.cfg.q_agg_sample)
        updater = self._stateless_update(build_mala_sampler(**_sampler_kw, compute_final_q=False), agg_critic)
        stateless_get_action = lambda key, state, obs: sampler(key, state, obs, agg_critic)
        self._jit_vmap_update = jax.jit(jax.vmap(updater))
        self._jit_vmap_get_action = jax.jit(jax.vmap(stateless_get_action))

    def update_vmap(self, key: jax.Array, data: Experience) -> Metric:
        self.state, info = self._jit_vmap_update(key, self.state, data)
        return _split_info_vmap(info)

    def warmup_vmap(self, data: Experience, N: int) -> None:
        key = jax.random.split(jax.random.key(0), N)
        obs = data.obs[:, 0]
        self._jit_vmap_update(key, self.state, data)
        self._jit_vmap_get_action(key, self.state, obs)

    def _huber_loss(self, td_err, reward_scale, q_td_huber_width):
        huber_delta = q_td_huber_width * reward_scale
        use_huber_loss = jnp.isfinite(huber_delta)
        huber_delta_safe = jnp.where(use_huber_loss, huber_delta, jnp.float32(1.0))
        abs_td_err = jnp.abs(td_err)
        quadratic = jnp.minimum(abs_td_err, huber_delta_safe)
        linear = abs_td_err - quadratic
        huber_per_elem = jnp.float32(0.5) * quadratic * quadratic + huber_delta_safe * linear
        per_elem_loss = jnp.where(use_huber_loss, huber_per_elem, td_err * td_err)
        return jnp.mean(per_elem_loss)

    def _stateless_update(self, sampler, agg_sample_fn):
        """Return the ``stateless_update(key, state, data)`` closure.

        Captures ``sampler`` (MALA sampler) and ``agg_sample_fn`` (Q aggregation
        used when sampling the TD next-action; distinct from the backup target
        which is hardcoded to 'min'). Called once from ``__init__``; the result
        is wrapped with ``jax.jit(jax.vmap(...))`` and stored as
        ``_jit_vmap_update``.
        """
        def stateless_update(
            key: jax.Array, state: Diffv2TrainState, data: Experience
        ) -> Tuple[Diffv2OptStates, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q_params = state.params.q          # tuple of N Q params
            target_q_params = state.params.target_q  # tuple of N target Q params
            policy_params = state.params.policy
            q_opt_states = state.opt_state.q   # tuple of N opt states
            policy_opt_state = state.opt_state.policy
            step = state.step
            num_q = len(q_params)
            next_eval_key, diffusion_time_key, diffusion_noise_key = jax.random.split(key, 3)

            reward *= state.hp.reward_scale

            # Sample a single tilted next-action
            mala_result = sampler(next_eval_key, state, next_obs, agg_sample_fn)
            tilted_action = mala_result.action

            # Clipped double Q-learning: all Qs bootstrap from min_i(target_Q_i).
            per_q_target_values = [self.model.q(tqp, next_obs, tilted_action) for tqp in target_q_params]
            q_target_min_for_backup = _aggregate_q(per_q_target_values, "min")
            shared_backup = reward + (1 - done) * state.hp.gamma * q_target_min_for_backup
            q_backup_per_q = [shared_backup] * num_q

            # One Adam step on each of the N Q critics against its TD target.
            q_params, q_opt_states, all_q_losses = self._train_q_ensemble(
                state, obs, action, q_params, q_opt_states, q_backup_per_q
            )

            # No-op when not doing adaptive beta
            value_params_updated, value_opt_state_updated, value_loss_log = \
                self._value_update_step(state, per_q_target_values, next_obs)

            target_q_params = tuple(delayed_target_update(q_params[i], target_q_params[i], state.hp.polyak_tau, step, self.cfg.delay_update) for i in range(num_q))

            def policy_loss_fn(policy_params) -> jax.Array:
                # Standard diffusion score-matching loss (eps-MSE)
                # against ``tilted_action`` (target action sampled above).
                # Uses optax.squared_error (== (x-y)**2), NOT optax.l2_loss (== 0.5*(x-y)**2)
                t = jax.random.randint(
                    diffusion_time_key,
                    (obs.shape[0],),
                    0,
                    self.model.num_timesteps,
                )
                noise = jax.random.normal(diffusion_noise_key, tilted_action.shape)
                tilted_action_noisy = self.model.q_sample(t, tilted_action, noise)
                noise_pred = self.model.eps_pred(policy_params, next_obs, tilted_action_noisy, t)
                return optax.squared_error(noise_pred, noise).mean()

            def _do(_):
                loss, grads = jax.value_and_grad(policy_loss_fn)(policy_params)
                return (loss,) + delayed_param_update(
                    self.policy_optim, policy_params, grads, policy_opt_state, state.hp.lr_policy, step, 1)
            total_loss, policy_params, policy_opt_state = jax.lax.cond(
                step % self.cfg.delay_update == 0, _do,
                lambda _: (state.policy_loss, policy_params, policy_opt_state), None)



            state = state._replace(
                params=ActorCriticParams(q_params, target_q_params, policy_params),
                opt_state=Diffv2OptStates(q=q_opt_states, policy=policy_opt_state, value=value_opt_state_updated),
                step=step + 1,
                log_eta_scales=mala_result.log_eta_scales,
                value_params=value_params_updated,
                policy_loss=total_loss,
            )

            # --- Losses ---
            # Q_MSE: average loss across ensemble members
            q_loss = jnp.mean(all_q_losses)

            info = {
                self.policy_loss_key: total_loss,
                "losses/Q_MSE": q_loss, # Actually Huber. needs to be renamed to just Q_loss
            }

            # V_MSE: only when V network exists
            if self.on_policy_ema and state.value_params is not None:
                info["losses/V_MSE"] = value_loss_log

            # --- MALA per-level arrays (logged as wandb.Table line plots) ---
            info["MALA/acceptance_rate"] = mala_result.per_level_acc
            info["MALA/clip_frac"] = mala_result.per_level_clip
            info["MALA/eta_scale"] = jnp.exp(mala_result.log_eta_scales)

            # --- Q section ---
            if self.on_policy_ema:
                info["Critic/inv_sqrt(E(Var(Q))_ema)"] = jnp.float32(1.0) / jnp.sqrt(jnp.maximum(state.advantage_second_moment_ema, jnp.float32(1e-6)))
            return state, info

        return stateless_update

    def _train_q_ensemble(self, state, obs, action, q_params, q_opt_states, q_backup_per_q):
        """One Adam step on each of the N Q critics against its TD target.

        The per-Q TD loss + Adam step is vmapped across the ensemble so XLA
        emits a single batched matmul per layer (and a single batched Adam
        update) instead of N independent unrolled ops.

        ``q_backup_per_q`` is a list of N TD targets (under clipped double
        Q-learning these are all equal to the min over target Qs). Returns
        ``(new_q_params_tuple, new_q_opt_states_tuple, all_q_losses)``.
        """
        num_q = len(q_params)

        def single_q_train_step(qp, opt_s, backup_qi):
            def q_loss_fn(p):
                q_pred_mean = self.model.q(p, obs, action)
                td_err = q_pred_mean - backup_qi
                return self._huber_loss(td_err, state.hp.reward_scale, state.hp.q_td_huber_width)

            qi_loss, qi_grads = jax.value_and_grad(q_loss_fn)(qp)
            update, new_opt = self.optim.update(qi_grads, opt_s, params=qp)
            update = jax.tree.map(lambda u: -state.hp.lr_q * u, update)
            new_qp = optax.apply_updates(qp, update)
            return new_qp, new_opt, qi_loss

        # Stack the N (params, opt-state, backup) tuples along a leading axis,
        # vmap one Adam step over the ensemble, then unstack back to N tuples.
        stacked_new_qp, stacked_new_q_opt_states, all_q_losses = jax.vmap(
            single_q_train_step
        )(stack_trees(q_params), stack_trees(q_opt_states), jnp.stack(q_backup_per_q))

        return unstack_tree(stacked_new_qp, num_q), unstack_tree(stacked_new_q_opt_states, num_q), all_q_losses

    def _value_update_step(self, state, per_q_target_values, next_obs):
        """Train V(s') against on-policy Q(s', a') targets (KL-budget mode).

        Returns ``(value_params_updated, value_opt_state_updated, value_loss_log)``.
        Outside KL-budget / EMA mode this is a pass-through that returns the
        existing V state unchanged and a zero loss.

        ``per_q_target_values`` is already computed at (next_obs, tilted_action)
        with a' ~ π_current. The legacy off-policy V branch (EMA mode without
        --kl_budget) was removed; ``validate_args`` enforces that combo.
        """
        if self.value_head is None or state.value_params is None:
            return state.value_params, state.opt_state.value, jnp.float32(0.0)

        q_for_v = _aggregate_q(per_q_target_values, self.cfg.q_agg_sample)
        return self.value_head.update_step(state, q_for_v, next_obs, state.hp.lr_q, self.optim)

    def get_action_vmap(self, key: jax.Array, obs: np.ndarray):
        """Vmapped counterpart of get_action.

        obs: numpy array of shape [N, num_envs, obs_dim].
        Returns (action [N, num_envs, act_dim], q_per_env [N, num_envs],
        v_per_env [N, num_envs]).
        """
        result = self._jit_vmap_get_action(key, self.state, obs)
        # log_eta_scales: shape [N, timesteps] — matches stacked state layout.
        self.state = self.state._replace(log_eta_scales=result.log_eta_scales)
        action_np = np.asarray(result.action)
        q_per_env = np.asarray(result.q)  # [N, num_envs]

        if not self.on_policy_ema:
            return action_np, q_per_env, None

        v = self.value_head.apply_vmap(self.state.value_params, jnp.asarray(obs))
        v_per_env = np.asarray(v)  # [N, num_envs]
        return action_np, q_per_env, v_per_env

    def _setup_value_network(self, params):
        """Construct the V(s) network used by KL-budget / on-policy-EMA mode.
        Returns (value_params, value_opt_state) or (None, None) when V is not
        used.

        The actual V-net plumbing (init, apply, vmap-apply, TD update step)
        lives in :class:`relax.algorithm.value_head.ValueHead`; we just
        construct it here.
        """
        if not self.on_policy_ema:
            self.value_head = None
            return None, None

        self.value_head = ValueHead.create(self._obs_dim, self._hidden_dim)
        value_params_init = self.value_head.init_params(jax.random.PRNGKey(42))
        value_opt_state_init = self.value_head.init_opt_state(value_params_init, self.optim)
        return value_params_init, value_opt_state_init

    def _build_initial_state(self, params, value_params_init, value_opt_state_init):
        """Construct a fresh Diffv2TrainState from a given set of network params.

        Factored out of __init__ so vmap-mode setup can call it N times with
        different init seeds and stack the results along a leading seed axis.
        """
        cfg = self.cfg
        # In-graph kl_budget sentinel: when --kl_budget is disabled (None),
        # store 1.0 so the value can still be a jnp.float32 in the vmappable
        # HParams; the host-side β cap reads ``cfg.kl_budget`` directly.
        kl_budget_val = 1.0 if cfg.kl_budget is None else cfg.kl_budget
        return Diffv2TrainState(
            params=params,
            opt_state=Diffv2OptStates(
                q=tuple(self.optim.init(qp) for qp in params.q),
                policy=self.policy_optim.init(params.policy),
                value=value_opt_state_init,
            ),
            step=jnp.int32(0),
            log_eta_scales=jnp.zeros((self._timesteps,), dtype=jnp.float32),
            beta=jnp.float32(cfg.beta),
            value_params=value_params_init,
            advantage_second_moment_ema=jnp.float32(cfg.initial_advantage_second_moment_ema),
            advantage_third_moment_ema=jnp.float32(0.0),
            dist_shift_covariance_ema=jnp.float32(0.0),
            dist_shift_shape_ema=jnp.float32(cfg.initial_dist_shift_shape_ema),
            hp=HParams(
                gamma=jnp.float32(cfg.gamma),
                polyak_tau=jnp.float32(cfg.polyak_tau),
                lr_q=jnp.float32(cfg.lr_q),
                lr_policy=jnp.float32(cfg.lr_policy),
                guidance_mult=jnp.float32(cfg.guidance_strength_multiplier),
                adv_ema_tau=jnp.float32(cfg.advantage_ema_tau),
                shape_ema_tau=jnp.float32(cfg.shape_ema_tau),
                kl_budget_val=jnp.float32(kl_budget_val),
                reward_scale=jnp.float32(cfg.reward_scale),
                x0_hat_clip_radius=jnp.float32(cfg.x0_hat_clip_radius),
                mala_adapt_rate=jnp.float32(cfg.mala_adapt_rate),
                q_td_huber_width=jnp.float32(cfg.q_td_huber_width),
                alpha=jnp.float32(cfg.alpha),
            ),
        )

    def make_vmapped_state(self, params_list, value_init_keys=None):
        """Build a vmapped train-state by stacking N independent initial states.

        params_list: list of N ActorCriticParams (one per seed).
        value_init_keys: optional list of N jax.random keys for value-net init.
                        Ignored when no V-network is used.
        Returns a Diffv2TrainState with a leading [N] axis on every leaf.
        """
        N = len(params_list)
        if self.value_head is not None:
            if value_init_keys is None:
                # Derive deterministic per-seed keys from the standard init seed.
                value_init_keys = [jax.random.PRNGKey(42) for _ in range(N)]
            vparams_list = [self.value_head.init_params(k) for k in value_init_keys]
            vopt_list = [self.value_head.init_opt_state(vp, self.optim) for vp in vparams_list]
        else:
            vparams_list = [None] * N
            vopt_list = [None] * N
        states = [
            self._build_initial_state(p, vp, vo)
            for p, vp, vo in zip(params_list, vparams_list, vopt_list)
        ]
        return stack_trees(states)

    def get_effective_hparams(self) -> dict:
        return {
            "lr_policy_effective": float(self.cfg.lr_policy),
            "lr_q_effective": float(self.cfg.lr_q),
        }


# ---------------------------------------------------------------------------
# MGMD-specific Q-ensemble aggregation. Lives at module scope so the closures
# inside ``MGMD._build_*`` can capture it cheaply; kept at the bottom of the
# file so the reader sees ``class MGMD`` first.
# ---------------------------------------------------------------------------
def _aggregate_q(q_means, mode: str):
    """Aggregate a list of N Q-network outputs (same shape) elementwise.

    mode='min':  pairwise jnp.minimum reduction across the list (used for the
                 TD-bootstrap target — clipped double-Q).
    mode='mean': sum(q_means) * (1/N) (selectable via ``--q_agg_sample`` for
                 the rollout / guidance path).

    Reduction order matches the legacy in-line implementations exactly so
    swapping callers to this helper is bit-identical.
    """
    if mode == "min":
        q = q_means[0]
        for m in q_means[1:]:
            q = jnp.minimum(q, m)
        return q
    if mode == "mean":
        return sum(q_means) * jnp.float32(1.0 / len(q_means))
    raise ValueError(f"_aggregate_q: unknown mode {mode!r}")
