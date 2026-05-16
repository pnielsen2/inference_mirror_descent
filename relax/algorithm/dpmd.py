from typing import Tuple

import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk

from relax.algorithm.base import Algorithm
from relax.algorithm.mala_sampler import build_mala_sampler
from relax.algorithm.dpmd_types import (
    Diffv2OptStates,
    HParams,
    Diffv2TrainState,
    DPMDConfig,
)
from relax.algorithm.value_head import ValueHead
from relax.network.diffv2 import Diffv2Net, Diffv2Params
from relax.utils.experience import Experience
from relax.utils.jax_utils import (
    delayed_param_update,
    delayed_target_update,
    stack_trees,
    unstack_tree,
)
from relax.utils.typing_utils import Metric


class DPMD(Algorithm):

    def __init__(self, agent: Diffv2Net, params: Diffv2Params, cfg: DPMDConfig,
                 *, obs_dim: int, hidden_dim: int):
        self.agent = agent
        self.cfg = cfg
        self._obs_dim = int(obs_dim)
        self._hidden_dim = int(hidden_dim)
        # Derived/exposed flags. Everything else lives on ``self.cfg``; the
        # two attributes below are also read off the algorithm by the
        # trainer (``algorithm.on_policy_ema`` / ``algorithm.one_step_dist_shift_eta``).
        self.on_policy_ema = (cfg.kl_budget is not None)
        self.one_step_dist_shift_eta = bool(cfg.one_step_dist_shift_eta)
        self.policy_loss_key = "losses/Policy_epsilon_MSE"

        # --- Optimizers: unscaled Adam; per-seed state.lr_{q,policy} is applied at update time. ---
        self.optim = optax.scale_by_adam()
        self.policy_optim = optax.scale_by_adam()

        # --- Optional V(s) network for normalized-advantage guidance (KL-budget / on-policy-EMA mode). ---
        value_params_init, value_opt_state_init = self._setup_value_network(params)

        self._timesteps = int(self.agent.num_timesteps)

        # --- Initial vmap-stackable train state ---
        self.state = self._build_initial_state(params, value_params_init, value_opt_state_init)

        # --- Schedule cache for wandb SNR x-axis ---
        B_sched = self.agent.diffusion.beta_schedule()
        self._alphas_cumprod = np.asarray(B_sched.alphas_cumprod)  # [T]
        self._snr = self._alphas_cumprod / np.maximum(1.0 - self._alphas_cumprod, 1e-8)

        # --- Stateless update / sampler closures (jit-able, vmap-able). ---
        # Each builder returns a single closure; the closures capture
        # ``self`` so all per-instance scalars / sub-networks are visible.
        # Bit-identical to the legacy in-line nested closures.
        sampler = self._build_mala_sampler()
        updater = self._build_update_step(sampler)
        # Rollout sampler: bind the configured ``--q_critic_agg`` aggregation
        # and let the JIT'd entry return the full ``MalaSampleResult``; the
        # host-side ``get_action_vmap`` then accesses ``.action`` / ``.q`` /
        # ``.log_eta_scales`` on the resulting namedtuple.
        agg_critic = lambda qm: _aggregate_q(qm, self.cfg.q_critic_agg)
        env_sampler = lambda key, state, obs: sampler(key, state, obs, agg_critic)
        self._implement_common_behavior(updater, env_sampler)

    def _build_update_step(self, sampler):
        """Return the jitted ``stateless_update(key, state, data)`` closure.

        ``sampler`` is the ``stateless_get_action_mala_full`` closure built by
        ``_build_mala_sampler``; it is invoked here to draw the next-state
        action used as the diffusion-loss target and the TD bootstrap point.
        """
        # The TD-bootstrap path always uses 'min' (clipped double-Q).
        agg_min = lambda qm: _aggregate_q(qm, "min")

        @jax.jit
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
            log_eta_scales = state.log_eta_scales
            num_q = len(q_params)
            # 8 fixed + num_q trailing slots; only 0, 3, 4 are consumed below.
            # The other slots are reserved to preserve PRNG layout against
            # earlier alpha-tuning / randomize-Q / shuffle paths.
            (next_eval_key, _, _, diffusion_time_key, diffusion_noise_key,
             _, _, _, *_) = jax.random.split(key, 8 + num_q)

            reward *= state.hp.reward_scale

            # Sample a single next-action and evaluate target-Q on it (td_actions=1).
            mala_result = sampler(
                next_eval_key, state, next_obs, agg_min,
            )
            next_action = mala_result.action
            log_eta_scales = mala_result.log_eta_scales
            q_target_per_q = [self.agent.q(tqp, next_obs, next_action) for tqp in target_q_params]

            not_done = (1 - done)
            # Clipped double Q-learning: all Qs bootstrap from min(target_Q_1..N).
            q_target_min_for_backup = _aggregate_q(q_target_per_q, "min")
            shared_backup = reward + not_done * state.hp.gamma * q_target_min_for_backup
            q_backup_per_q = [shared_backup] * num_q

            q_params, q_opt_states, all_q_losses = self._train_q_ensemble(
                state, obs, action, q_params, q_opt_states, q_backup_per_q
            )

            def policy_loss_fn(policy_params) -> jax.Array:
                # Constant-weight diffusion score-matching loss against
                # next_action (target action sampled above). The Q-based
                # reweighting path has been removed; this is bit-equivalent
                # to upstream `inference_mirror_descent`'s eps_mse path:
                #   weighted_p_loss(diffusion_noise_key, jnp.ones_like(weights),
                #                   denoiser, t, target_action, reduction="mean")
                # (see inference_mirror_descent/relax/algorithm/dpmd.py:1793).
                # The simplified `p_loss` uses optax.squared_error (matching
                # weighted_p_loss), NOT optax.l2_loss which would be a 0.5x
                # multiplier and silently produce a non-bit-identical run.
                def denoiser(t, x):
                    return self.agent.policy(policy_params, next_obs, x, t)

                t = jax.random.randint(
                    diffusion_time_key,
                    (obs.shape[0],),
                    0,
                    self.agent.num_timesteps,
                )
                return self.agent.diffusion.p_loss(
                    diffusion_noise_key,
                    denoiser,
                    t,
                    jax.lax.stop_gradient(next_action),
                )

            total_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_params)

            # Policy + target-Q updates, both gated by step % delay_update == 0.
            policy_params, policy_opt_state = delayed_param_update(
                self.policy_optim, policy_params, policy_grads, policy_opt_state,
                state.hp.lr_policy, step, self.cfg.delay_update,
            )
            target_q_params = tuple(
                delayed_target_update(
                    q_params[qi], target_q_params[qi], state.hp.polyak_tau,
                    step, self.cfg.delay_update,
                )
                for qi in range(num_q)
            )

            # Normalized advantage guidance: train V(s'). The advantage-EMA
            # itself is updated outside jit in VmapOffPolicyTrainer.sample()
            # (on-policy mode), so we pass state.advantage_second_moment_ema
            # through unchanged here.
            value_params_updated, value_opt_state_updated, value_loss_log = \
                self._value_update_step(state, q_target_per_q, next_obs)
            new_adv_second_moment_ema = state.advantage_second_moment_ema

            state = state._replace(
                params=Diffv2Params(q_params, target_q_params, policy_params),
                opt_state=Diffv2OptStates(q=q_opt_states, policy=policy_opt_state, value=value_opt_state_updated),
                step=step + 1,
                log_eta_scales=log_eta_scales,
                value_params=value_params_updated,
                advantage_second_moment_ema=new_adv_second_moment_ema,
            )

            # --- Losses ---
            # Q_MSE: average loss across ensemble members
            q_mse = jnp.mean(all_q_losses)

            info = {
                self.policy_loss_key: total_loss,
                "losses/Q_MSE": q_mse,
            }

            # V_MSE: only when V network exists
            if self.on_policy_ema and state.value_params is not None:
                info["losses/V_MSE"] = value_loss_log

            # --- MALA per-level arrays (logged as wandb.Table line plots) ---
            info["MALA/acceptance_rate"] = mala_result.per_level_acc
            info["MALA/clip_frac"] = mala_result.per_level_clip
            info["MALA/eta_scale"] = jnp.exp(log_eta_scales)

            # --- Q section ---
            if self.on_policy_ema:
                info["Critic/inv_sqrt(E(Var(Q))_ema)"] = jnp.float32(1.0) / jnp.sqrt(jnp.maximum(new_adv_second_moment_ema, jnp.float32(1e-6)))
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
        delta = state.hp.q_td_huber_width * state.hp.reward_scale
        use_huber = jnp.isfinite(delta)
        delta_safe = jnp.where(use_huber, delta, jnp.float32(1.0))

        def huber_loss(e):
            abs_e = jnp.abs(e)
            quad = jnp.minimum(abs_e, delta_safe)
            lin = abs_e - quad
            return jnp.float32(0.5) * quad * quad + delta_safe * lin

        def compute_td_loss(td_err):
            per_elem = jnp.where(use_huber, huber_loss(td_err), td_err * td_err)
            return jnp.mean(per_elem)

        def single_q_train_step(qp, opt_s, backup_qi):
            def q_loss_fn(p):
                q_pred_mean = self.agent.q(p, obs, action)
                td_err = q_pred_mean - backup_qi
                return compute_td_loss(td_err), q_pred_mean

            (qi_loss, qi_pred), qi_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(qp)
            update, new_opt = self.optim.update(qi_grads, opt_s, params=qp)
            update = jax.tree.map(lambda u: -state.hp.lr_q * u, update)
            new_qp = optax.apply_updates(qp, update)
            return new_qp, new_opt, qi_loss, qi_pred

        # Stack the N (params, opt-state, backup) tuples along a leading axis,
        # vmap one Adam step over the ensemble, then unstack back to N tuples.
        stacked_q_params = stack_trees(q_params)
        stacked_q_opt = stack_trees(q_opt_states)
        stacked_backup = jnp.stack(q_backup_per_q)

        stacked_new_qp, stacked_new_opt, all_q_losses, _all_q_preds = jax.vmap(
            single_q_train_step
        )(stacked_q_params, stacked_q_opt, stacked_backup)

        return unstack_tree(stacked_new_qp, num_q), unstack_tree(stacked_new_opt, num_q), all_q_losses

    def _value_update_step(self, state, q_target_per_q, next_obs):
        """Train V(s') against on-policy Q(s', a') targets (KL-budget mode).

        Returns ``(value_params_updated, value_opt_state_updated, value_loss_log)``.
        Outside KL-budget / EMA mode this is a pass-through that returns the
        existing V state unchanged and a zero loss.

        ``q_target_per_q`` is already computed at (next_obs, next_action)
        with a' ~ π_current. The legacy off-policy V branch (EMA mode without
        --kl_budget) was removed; ``validate_args`` enforces that combo.
        """
        if self.value_head is None or state.value_params is None:
            return state.value_params, state.opt_state.value, jnp.float32(0.0)

        q_for_v = _aggregate_q(q_target_per_q, self.cfg.q_critic_agg)
        return self.value_head.update_step(state, q_for_v, next_obs, state.hp.lr_q, self.optim)

    def _build_mala_sampler(self):
        """Build the MALA sampler closure. The actual sampler body lives in
        :mod:`relax.algorithm.mala_sampler` so ``dpmd.py`` reads top-to-bottom
        around the training step. Returns
        ``stateless_get_action_mala_full(key, state, obs, aggregate_q_fn)``.
        """
        return build_mala_sampler(
            agent=self.agent,
            value_head=self.value_head,
            timesteps=self._timesteps,
            energy_multiplier=self.cfg.energy_multiplier,
            batch_independent_guidance=self.cfg.batch_independent_guidance,
            mala_guided_predictor=self.cfg.mala_guided_predictor,
            mala_no_predictor=self.cfg.mala_no_predictor,
        )

    def get_action_vmap(self, key: jax.Array, obs: np.ndarray):
        """Vmapped counterpart of get_action.

        obs: numpy array of shape [N, num_envs, obs_dim].
        Returns (action [N, num_envs, act_dim], q_per_env [N, num_envs],
        v_per_env [N, num_envs]).
        """
        self._ensure_vmap_compiled()
        # ``stateless_*`` sampler/update fns take the full Diffv2TrainState;
        # JAX prunes unused leaves at trace time.
        result = self._get_action_vmap_fn(key, self.state, obs)
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
        # HParams; the host-side η cap reads ``cfg.kl_budget`` directly.
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
            tfg_eta=jnp.float32(cfg.tfg_eta),
            value_params=value_params_init,
            advantage_second_moment_ema=jnp.float32(cfg.initial_advantage_second_moment_ema),
            advantage_third_moment_ema=jnp.float32(0.0),
            dist_shift_covariance_ema=jnp.float32(0.0),
            dist_shift_shape_ema=jnp.float32(cfg.initial_dist_shift_shape_ema),
            hp=HParams(
                gamma=jnp.float32(cfg.gamma),
                polyak_tau=jnp.float32(cfg.tau),
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
            ),
        )

    def make_vmapped_state(self, params_list, value_init_keys=None):
        """Build a vmapped train-state by stacking N independent initial states.

        params_list: list of N Diffv2Params (one per seed).
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
# DPMD-specific Q-ensemble aggregation. Lives at module scope so the closures
# inside ``DPMD._build_*`` can capture it cheaply; kept at the bottom of the
# file so the reader sees ``class DPMD`` first.
# ---------------------------------------------------------------------------
def _aggregate_q(q_means, mode: str):
    """Aggregate a list of N Q-network outputs (same shape) elementwise.

    mode='min':  pairwise jnp.minimum reduction across the list (used for the
                 TD-bootstrap target — clipped double-Q).
    mode='mean': sum(q_means) * (1/N) (selectable via ``--q_critic_agg`` for
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
