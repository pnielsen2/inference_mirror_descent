from typing import Optional, Tuple

import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk

from relax.algorithm import distillation, noise_schedule, normalizers
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
from relax.utils.diffusion import NoiseLevel, cosine_log_snr_knots, log_snr_at
from relax.utils.jax_utils import (
    delayed_param_update,
    delayed_target_update,
    is_due,
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
        self.use_advantage_stats = bool(cfg.advantage_normalization or cfg.kl_budget is not None)
        self.on_policy_ema = self.use_advantage_stats
        self.one_step_dist_shift_beta = bool(cfg.one_step_dist_shift_beta)
        # V-free guidance-normalization knobs (see relax/cli/train_args.py).
        self.num_denoised_actions = int(cfg.num_denoised_actions)
        self.q_loss_normalization = bool(cfg.q_loss_normalization)
        self.batch_advantage_normalization = bool(cfg.batch_advantage_normalization)
        self.ema_advantage_normalization = bool(cfg.ema_advantage_normalization)
        self.ema_within_advantage_normalization = bool(cfg.ema_within_advantage_normalization)
        self.estimate_s_hat = bool(cfg.estimate_s_hat)
        self.adaptive_schedule = bool(model.adaptive_schedule)
        self.lr_anneal = bool(cfg.lr_anneal)
        # Training-time-only target-network usage; rollout sampling is unaffected.
        self.use_target_policy_training = bool(cfg.use_target_policy_training)
        self.use_target_q_sampling_training = bool(cfg.use_target_q_sampling_training)
        self.policy_loss_key = "losses/Policy_epsilon_MSE"

        # --- Optimizers: unscaled Adam; per-seed state.lr_{q,policy} is applied at update time. ---
        self.optim = optax.scale_by_adam()
        self.policy_optim = optax.scale_by_adam()

        # --- Optional V(s) network for KL-budget / on-policy-EMA beta adaptation and logging. ---
        value_params_init, value_opt_state_init = self._setup_value_network(params)

        self._timesteps = int(self.model.num_timesteps)

        # --- Initial vmap-stackable train state ---
        self.state = self._build_initial_state(params, value_params_init, value_opt_state_init)

        # --- Stateless update / sampler closures (jit-able, vmap-able). ---
        # Build the MALA sampler; body lives in relax.algorithm.mala_sampler.
        # _energy_kw is the subset that pins down the MH target energy, shared
        # with the schedule updater so both score the same density.
        _energy_kw = dict(
            model=self.model,
            ema_normalization=bool(self.cfg.advantage_normalization or self.cfg.q_loss_normalization),
            batch_advantage_normalization=self.batch_advantage_normalization,
            # Both EMA modes divide the guidance Q by state.q_running_std; only what
            # the update writes into that field differs (pooled batch std vs the
            # contraharmonic mean of the within-state sd), so the sampler is shared.
            ema_advantage_normalization=self.ema_advantage_normalization or self.ema_within_advantage_normalization,
            guidance_snr_anneal=self.cfg.guidance_snr_anneal,
        )
        # Kept on self as the single source of truth for what the chain samples:
        # get_eval_action_vmap rebuilds a sampler from it at its own N, so
        # evaluation can never drift onto a different density than rollout.
        self._sampler_kw = dict(
            **_energy_kw, value_head=self.value_head, timesteps=self._timesteps,
            batch_independent_guidance=self.cfg.batch_independent_guidance,
            denoising_predictor=self.cfg.denoising_predictor,
            guidance_gradient_space=self.cfg.guidance_gradient_space,
            num_denoised_actions=self.num_denoised_actions,
            latent_action=getattr(self.cfg, "latent_action", False),
        )
        sampler = build_mala_sampler(**self._sampler_kw)
        # Both sampling paths (rollout + TD next-action) use --q_agg_sample aggregation.
        # The TD-backup target itself remains hardcoded to 'min' (clipped double-Q).
        agg_critic = lambda qm: _aggregate_q(qm, self.cfg.q_agg_sample)
        updater = self._stateless_update(
            build_mala_sampler(**self._sampler_kw, compute_final_q=False,
                               schedule_cost=self.adaptive_schedule), agg_critic)
        stateless_get_action = lambda key, state, obs: sampler(key, state, obs, agg_critic)
        self._schedule_update = None
        if self.adaptive_schedule:
            self._schedule_update = noise_schedule.build_updater(timesteps=self._timesteps)
        self._jit_vmap_update = jax.jit(jax.vmap(updater))
        self._jit_vmap_get_action = jax.jit(jax.vmap(stateless_get_action))
        self._jit_vmap_eval_action = {}  # N -> compiled best-of-N eval sampler

    @property
    def _snr(self):
        """Per-seed SNR grid ``[N, T]`` for the wandb log2-SNR x-axis.

        A property rather than an ``__init__`` cache because the layout is a
        function of per-seed hp: ``s_hat`` may be overridden per vmap slot by
        ``--hp_pack_inline`` *after* construction, and both it and the adaptive
        knots move during training (so for ``adaptive`` this axis is where the
        levels sit now, not a fixed layout).
        """
        sched = jax.vmap(self.model.schedule_for)(self.state.hp, self.state.log_snr_levels)
        return np.asarray(sched.alphas_cumprod, np.float64) / np.maximum(
            np.asarray(sched.one_minus_alphas_cumprod, np.float64), 1e-8)

    def update_vmap(self, key: jax.Array, data: Experience, critic_weight: Optional[jax.Array] = None, env_step: Optional[float] = None):
        """Vmapped training update.

        ``critic_weight`` is an optional per-row weight (``[num_runs, batch]``) on
        the Q TD loss; ``None`` means all-ones (standard mean). The fused-denoising
        trainer passes zeros for the injected step-action rows so their backup
        (bogus at an episode's first state) is excluded. Always returns the
        index-0 tilted next-action per row so the fused path can reuse it to step
        the env; the normal path ignores it.
        """
        if critic_weight is None:
            critic_weight = jnp.ones_like(data.reward)
        # env_step (host-side env-step counter) drives --lr_anneal; broadcast the
        # shared scalar to a [num_runs] array so the default in_axes=0 vmap maps
        # one copy to each seed. None (e.g. warmup) => 0 => no annealing yet.
        num_runs = data.reward.shape[0]
        if env_step is None:
            env_step_arr = jnp.zeros((num_runs,), dtype=jnp.float32)
        else:
            env_step_arr = jnp.full((num_runs,), jnp.float32(env_step), dtype=jnp.float32)
        self.state, info, actions = self._jit_vmap_update(key, self.state, data, critic_weight, env_step_arr)
        scalar_info, array_info = _split_info_vmap(info)
        return scalar_info, array_info, actions

    def eval_q_v_vmap(self, obs: np.ndarray, action: np.ndarray):
        """Rollout-equivalent Q(obs, action) and V(obs) for the stepping action.

        Reproduces the ``(q_per_env, v_per_env)`` that ``get_action_vmap`` would
        have returned in the non-fused path: online-Q aggregated with
        ``--q_agg_sample`` and (when the on-policy-EMA V network exists) V(obs).
        obs/action: ``[num_runs, envs_per_run, dim]``.
        """
        if getattr(self, "_jit_vmap_eval_q", None) is None:
            def _eval_q(state, o, a):
                return _aggregate_q([self.model.q(qp, o, a) for qp in state.params.q], self.cfg.q_agg_sample)
            self._jit_vmap_eval_q = jax.jit(jax.vmap(_eval_q))
        q_per_env = np.asarray(self._jit_vmap_eval_q(self.state, jnp.asarray(obs), jnp.asarray(action)))
        if not self.on_policy_ema:
            return q_per_env, None
        v_per_env = np.asarray(self.value_head.apply_vmap(self.state.value_params, jnp.asarray(obs)))
        return q_per_env, v_per_env

    def warmup_vmap(self, data: Experience, N: int) -> None:
        key = jax.random.split(jax.random.key(0), N)
        obs = data.obs[:, 0]
        self._jit_vmap_update(key, self.state, data, jnp.ones_like(data.reward), jnp.zeros((N,), dtype=jnp.float32))
        self._jit_vmap_get_action(key, self.state, obs)

    def _huber_loss(self, td_err, reward_scale, q_td_huber_width, weight=None):
        huber_delta = q_td_huber_width * reward_scale
        use_huber_loss = jnp.isfinite(huber_delta)
        huber_delta_safe = jnp.where(use_huber_loss, huber_delta, jnp.float32(1.0))
        abs_td_err = jnp.abs(td_err)
        quadratic = jnp.minimum(abs_td_err, huber_delta_safe)
        linear = abs_td_err - quadratic
        huber_per_elem = jnp.float32(0.5) * quadratic * quadratic + huber_delta_safe * linear
        per_elem_loss = jnp.where(use_huber_loss, huber_per_elem, td_err * td_err)
        if weight is None:
            return jnp.mean(per_elem_loss)
        return jnp.sum(weight * per_elem_loss) / jnp.maximum(jnp.sum(weight), jnp.float32(1.0))

    def _stateless_update(self, sampler, agg_sample_fn):
        """Return the ``stateless_update(key, state, data, critic_weight)`` closure.

        Captures ``sampler`` (MALA sampler) and ``agg_sample_fn`` (Q aggregation
        used when sampling the TD next-action; distinct from the backup target
        which is hardcoded to 'min'). Called once from ``__init__``; the result
        is wrapped with ``jax.jit(jax.vmap(...))`` and stored as
        ``_jit_vmap_update``.
        """
        def stateless_update(
            key: jax.Array, state: Diffv2TrainState, data: Experience, critic_weight: jax.Array, env_step: jax.Array
        ) -> Tuple[Diffv2OptStates, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q_params = state.params.q          # tuple of N Q params
            target_q_params = state.params.target_q  # tuple of N target Q params
            policy_params = state.params.policy
            target_policy_params = state.params.target_policy  # None unless --use_target_policy_training
            q_opt_states = state.opt_state.q   # tuple of N opt states
            policy_opt_state = state.opt_state.policy
            step = state.step
            num_q = len(q_params)
            next_eval_key, diffusion_time_key, diffusion_noise_key = jax.random.split(key, 3)

            # LR annealing (ported from diffusion_policy_online_rl): scale BOTH the
            # Q and policy learning rates by the SAME linear factor of the env-step
            # count. factor = 1 for env_step <= transition_begin, decreasing linearly
            # to lr_anneal_end_factor over transition_steps env steps, then held. The
            # env-step count is threaded in from the trainer's host-side counter.
            if self.lr_anneal:
                _begin = jnp.float32(self.cfg.lr_anneal_transition_begin)
                _steps = jnp.maximum(jnp.float32(self.cfg.lr_anneal_transition_steps), jnp.float32(1.0))
                _end = jnp.float32(self.cfg.lr_anneal_end_factor)
                _frac = jnp.clip((env_step - _begin) / _steps, jnp.float32(0.0), jnp.float32(1.0))
                lr_factor = jnp.float32(1.0) + _frac * (_end - jnp.float32(1.0))
                lr_q_eff = state.hp.lr_q * lr_factor
                lr_policy_eff = state.hp.lr_policy * lr_factor
            else:
                lr_factor = jnp.float32(1.0)
                lr_q_eff = state.hp.lr_q
                lr_policy_eff = state.hp.lr_policy

            reward *= state.hp.reward_scale

            # Denoise K tilted next-actions per state (K = num_denoised_actions).
            # This training-time sampling may read the target policy and/or the
            # target critic; rollout sampling (get_action_vmap) always uses the
            # online pair, and the TD backup below always uses target_q.
            sample_params = state.params
            if self.use_target_policy_training:
                sample_params = sample_params._replace(policy=target_policy_params)
            if self.use_target_q_sampling_training:
                sample_params = sample_params._replace(q=target_q_params)
            sample_state = state._replace(params=sample_params)
            mala_result = sampler(next_eval_key, sample_state, next_obs, agg_sample_fn)
            tilted_actions = mala_result.action           # [K, batch, act_dim]
            K = self.num_denoised_actions
            next_obs_k = jnp.broadcast_to(next_obs, (K, *next_obs.shape))

            # Clipped double Q-learning: min_i(target_Q_i) per action, then the
            # TD backup averages that clipped double-Q over the K next-actions.
            per_q_target_values = [self.model.q(tqp, next_obs_k, tilted_actions) for tqp in target_q_params]  # each [K, batch]
            q_target_min_for_backup = jnp.mean(_aggregate_q(per_q_target_values, "min"), axis=0)  # [batch]
            shared_backup = reward + (1 - done) * state.hp.gamma * q_target_min_for_backup
            q_backup_per_q = [shared_backup] * num_q

            # Reuse this minibatch and its fixed targets for C sequential
            # critic/value optimizer steps. C is a static config value, so the
            # Python loop is unrolled once during JIT tracing.
            value_params_updated = state.value_params
            value_opt_state_updated = state.opt_state.value
            for _ in range(self.cfg.critic_update_steps):
                q_params, q_opt_states, q_losses = self._train_q_ensemble(
                    state, obs, action, q_params, q_opt_states, q_backup_per_q, critic_weight, lr_q=lr_q_eff
                )

                value_step_state = state._replace(
                    value_params=value_params_updated,
                    opt_state=state.opt_state._replace(value=value_opt_state_updated),
                )
                value_params_updated, value_opt_state_updated, value_loss = \
                    self._value_update_step(value_step_state, per_q_target_values, next_obs)

            q_loss = jnp.mean(q_losses)
            value_loss_log = value_loss

            # V-free q_loss normalization: track EMA(Q TD loss) in the
            # advantage_second_moment_ema slot (the sampler divides beta by
            # sqrt of it). Host-side EMA is skipped since on_policy_ema is False.
            new_adv_m2_ema = state.advantage_second_moment_ema
            if self.q_loss_normalization:
                tau = state.hp.adv_ema_tau
                new_adv_m2_ema = (jnp.float32(1.0) - tau) * new_adv_m2_ema + tau * q_loss

            # EMA advantage normalization (ported from diffusion_policy_online_rl):
            # track a slow EMA of the batch mean/std of the online (post-critic-
            # update), --q_agg_sample-aggregated Q at the sampled next-actions. The
            # sampler divides the guidance Q by q_running_std; q_running_mean is
            # tracked for logging (it cancels in the guidance gradient).
            new_q_running_mean = state.q_running_mean
            new_q_running_std = state.q_running_std
            if self.ema_advantage_normalization:
                q_norm_samples = _aggregate_q(
                    [self.model.q(qp, next_obs_k, tilted_actions) for qp in q_params],
                    self.cfg.q_agg_sample,
                )  # [K, batch] online Q at the tilted next-actions
                rate = state.hp.adv_norm_ema_rate
                new_q_running_mean = state.q_running_mean + rate * (jnp.mean(q_norm_samples) - state.q_running_mean)
                new_q_running_std = state.q_running_std + rate * (jnp.std(q_norm_samples) - state.q_running_std)

            target_q_params = tuple(delayed_target_update(q_params[i], target_q_params[i], state.hp.q_polyak_tau, step, state.hp.delay_target_q_update) for i in range(num_q))

            # Diffusion policy regresses toward all K tilted actions; flatten the
            # K axis into the batch so score-matching sees K*batch targets, and
            # ring-write that block into the distillation buffer, which is what
            # every policy step below draws its minibatch from (at
            # --distillation_buffer_size 1 the buffer IS this block).
            assert obs.shape[0] == self.cfg.batch_size, "MGMDConfig.batch_size must be the trainer's minibatch size; it sizes the distillation buffer"
            distill_buffer = distillation.push(
                state.distill_buffer,
                jnp.concatenate((next_obs_k, tilted_actions), -1).reshape(K * obs.shape[0], -1),
                step)

            def policy_loss_fn(policy_params, rows, time_key, noise_key) -> jax.Array:
                # Standard diffusion score-matching loss (eps-MSE) against the
                # tilted target actions in ``rows``, one [batch, obs_dim+act_dim]
                # slice of the distillation buffer. Uses optax.squared_error
                # (== (x-y)**2), NOT optax.l2_loss (== 0.5*(x-y)**2).
                policy_obs, policy_targets = rows[:, :obs.shape[-1]], rows[:, obs.shape[-1]:]
                n = policy_targets.shape[0]
                if self.adaptive_schedule:
                    # Train the continuous schedule the knots interpolate rather
                    # than the grid indices, which move: uniform in the schedule's
                    # own time coordinate u, with lambda read off between knots.
                    level = NoiseLevel.from_log_snr(log_snr_at(
                        state.log_snr_levels, jax.random.uniform(time_key, (n,))))
                else:
                    t = jax.random.randint(time_key, (n,), 0, self.model.num_timesteps)
                    level = NoiseLevel.at(self.model.schedule_for(state.hp), t)
                noise = jax.random.normal(noise_key, policy_targets.shape)
                tilted_action_noisy = self.model.q_sample(level, policy_targets, noise)
                noise_pred = self.model.eps_pred(policy_params, level, policy_obs, tilted_action_noisy)
                return optax.squared_error(noise_pred, noise).mean()

            def _do(_):
                updated_policy_params = policy_params
                updated_policy_opt_state = policy_opt_state
                # One optimizer step per minibatch of the reshuffled passes over
                # the distillation buffer. --policy_update_steps multiplies
                # --distillation_steps because both mean "another pass over the
                # current targets": at buffer size 1 a pass IS one step on the
                # fresh block, which is exactly what the former used to repeat.
                for policy_step_idx, rows in enumerate(distillation.epoch_batches(
                    distill_buffer, key, batch_size=self.cfg.batch_size,
                    epochs=self.cfg.policy_update_steps * self.cfg.distillation_steps,
                )):
                    # Preserve the exact old random stream for a single step;
                    # additional steps derive independent keys from the same base keys.
                    if policy_step_idx == 0:
                        time_key = diffusion_time_key
                        noise_key = diffusion_noise_key
                    else:
                        time_key = jax.random.fold_in(diffusion_time_key, policy_step_idx)
                        noise_key = jax.random.fold_in(diffusion_noise_key, policy_step_idx)
                    loss, grads = jax.value_and_grad(policy_loss_fn)(
                        updated_policy_params, rows, time_key, noise_key
                    )
                    updated_policy_params, updated_policy_opt_state = delayed_param_update(
                        self.policy_optim,
                        updated_policy_params,
                        grads,
                        updated_policy_opt_state,
                        lr_policy_eff,
                        step,
                        1,
                    )
                return loss, updated_policy_params, updated_policy_opt_state
            total_loss, policy_params, policy_opt_state = jax.lax.cond(
                is_due(step, state.hp.delay_policy_update), _do,
                lambda _: (state.policy_loss, policy_params, policy_opt_state), None)

            # Target policy tracks the just-updated online policy (mirrors target-Q).
            if self.use_target_policy_training:
                target_policy_params = delayed_target_update(
                    policy_params, target_policy_params, state.hp.policy_polyak_tau,
                    step, state.hp.delay_target_policy_update)

            state = state._replace(
                params=ActorCriticParams(q_params, target_q_params, policy_params, target_policy_params),
                opt_state=state.opt_state._replace(q=q_opt_states, policy=policy_opt_state, value=value_opt_state_updated),
                step=step + 1,
                log_eta_scales=mala_result.log_eta_scales,
                distill_buffer=distill_buffer,
                value_params=value_params_updated,
                policy_loss=total_loss,
                advantage_second_moment_ema=new_adv_m2_ema,
                q_running_mean=new_q_running_mean,
                q_running_std=new_q_running_std,
            )

            # --- Losses ---
            info = {
                self.policy_loss_key: total_loss,
                "losses/Q_loss": q_loss,
            }
            if self.lr_anneal:
                info["lr/anneal_factor"] = lr_factor
                info["lr/lr_q"] = lr_q_eff
                info["lr/lr_policy"] = lr_policy_eff

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
            if self.q_loss_normalization:
                info["Critic/q_loss_norm"] = jnp.sqrt(jnp.maximum(new_adv_m2_ema, jnp.float32(1e-6)))
            if self.ema_advantage_normalization:
                info["Critic/adv_norm_running_mean"] = new_q_running_mean
                info["Critic/adv_norm_running_std"] = new_q_running_std
                info["Critic/adv_norm_inv_std"] = jnp.float32(1.0) / jnp.maximum(new_q_running_std, jnp.float32(1e-6))
            if self.batch_advantage_normalization:
                # Representative batch-norm scale for monitoring: per-state Q
                # variance over the K next-actions (reusing per_q_target_values,
                # so ~free), batch-averaged and square-rooted. A proxy for the
                # per-noise-level denom the sampler applies during guidance.
                q_agg_k = _aggregate_q(per_q_target_values, self.cfg.q_agg_sample)  # [K, batch]
                info["Critic/batch_adv_norm"] = jnp.sqrt(jnp.maximum(jnp.mean(jnp.var(q_agg_k, axis=0, ddof=1)), jnp.float32(1e-6)))
            # Within-state normalizer EMAs over the K denoised actions, folded in
            # *after* the sampler ran, so nothing feeds back inside one graph and
            # the sampler + policy loss of a given step share one schedule. Always
            # accumulated when K>=2 so the diagnostics (CV^2, tau^2, and the two
            # floors they imply) are available at fixed --s_hat and under existing
            # normalization; they drive training only under the two flags passed in.
            # Reuses the target-Q already computed for the TD backup, so ~free.
            if self.num_denoised_actions >= 2:
                state, norm_info = normalizers.update(
                    state, _aggregate_q(per_q_target_values, self.cfg.q_agg_sample),
                    tilted_actions, step=step, K=self.num_denoised_actions,
                    use_for_advantage=self.ema_within_advantage_normalization,
                    use_for_s_hat=self.estimate_s_hat,
                )
                info.update(norm_info)
            # Move the knots a fraction gamma toward the equal-cost layout, off
            # the per-interval costs the sampler's own MH drifts produced -- so
            # the schedule is scored on exactly the density the chain equilibrated
            # to, with no second pass over it. Folded in here for the same reason
            # as the normalizer EMAs: the schedule this step's sampler and policy
            # loss ran on stays fixed.
            if self.adaptive_schedule:
                state, sched_info = self._schedule_update(state, mala_result.schedule_cost)
                info.update(sched_info)
                # Health check for the cost itself: a Tweedie-consistent denoiser
                # shrinks x0_hat toward the prior mean as the noise rises, so this
                # must FALL toward the noisy end. While it rises, d x0hat/dx still
                # carries 1/sqrt(abar) and the cost is not yet worth descending.
                info["Schedule/x0hat_clip_noisiest"] = mala_result.per_level_clip[-1]
                info["Schedule/x0hat_clip_cleanest"] = mala_result.per_level_clip[0]
            # Also return the index-0 (uniform-draw) tilted next-action per row so
            # the fused-denoising trainer can reuse it to step the env. XLA already
            # materializes tilted_actions for the policy loss, so this is ~free; the
            # normal update_vmap path discards it.
            return state, info, tilted_actions[0]

        return stateless_update

    def _train_q_ensemble(self, state, obs, action, q_params, q_opt_states, q_backup_per_q, critic_weight=None, lr_q=None):
        """One Adam step on each of the N Q critics against its TD target.

        The per-Q TD loss + Adam step is vmapped across the ensemble so XLA
        emits a single batched matmul per layer (and a single batched Adam
        update) instead of N independent unrolled ops.

        ``q_backup_per_q`` is a list of N TD targets (under clipped double
        Q-learning these are all equal to the min over target Qs). Returns
        ``(new_q_params_tuple, new_q_opt_states_tuple, all_q_losses)``.
        """
        num_q = len(q_params)
        lr_q_use = state.hp.lr_q if lr_q is None else lr_q

        def single_q_train_step(qp, opt_s, backup_qi):
            def q_loss_fn(p):
                q_pred_mean = self.model.q(p, obs, action)
                td_err = q_pred_mean - backup_qi
                return self._huber_loss(td_err, state.hp.reward_scale, state.hp.q_td_huber_width, critic_weight)

            qi_loss, qi_grads = jax.value_and_grad(q_loss_fn)(qp)
            update, new_opt = self.optim.update(qi_grads, opt_s, params=qp)
            update = jax.tree.map(lambda u: -lr_q_use * u, update)
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

        # per_q_target_values entries are [K, batch]; average over the K denoised
        # actions to get the on-policy V(s') regression target [batch].
        q_for_v = jnp.mean(_aggregate_q(per_q_target_values, self.cfg.q_agg_sample), axis=0)
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
        # Sampler returns K iid actions per state ([N, K, num_envs, ...]); index 0
        # is a uniform draw. Take it (and its raw Q) to step the environment.
        action_np = np.asarray(result.action[:, 0])  # [N, num_envs, act_dim]
        q_per_env = np.asarray(result.q[:, 0])       # [N, num_envs]

        if not self.on_policy_ema:
            return action_np, q_per_env, None

        v = self.value_head.apply_vmap(self.state.value_params, jnp.asarray(obs))
        v_per_env = np.asarray(v)  # [N, num_envs]
        return action_np, q_per_env, v_per_env

    def get_eval_action_vmap(self, key: jax.Array, obs: np.ndarray, num_actions: int) -> np.ndarray:
        """Best-of-N action for separate evaluation: [N_runs, envs, act_dim].

        Denoises ``num_actions`` iid candidates per state from the same chain
        rollout uses (``self._sampler_kw`` at a different K) and executes the
        highest --q_agg_sample-aggregated-Q one. Unlike ``get_action_vmap`` this
        writes nothing back to ``self.state``, so the MALA step-size adaptation
        the training rollout is tuning is left alone and evaluation cannot
        perturb the run. Compiled once per ``num_actions``.
        """
        if num_actions not in self._jit_vmap_eval_action:
            sampler = build_mala_sampler(**{**self._sampler_kw, "num_denoised_actions": int(num_actions)})
            agg_critic = lambda qm: _aggregate_q(qm, self.cfg.q_agg_sample)

            def eval_action(key, state, obs):
                result = sampler(key, state, obs, agg_critic)  # action [N, envs, act], q [N, envs]
                best = jnp.argmax(result.q, axis=0)            # [envs]
                return jnp.take_along_axis(result.action, best[None, :, None], axis=0)[0]

            self._jit_vmap_eval_action[num_actions] = jax.jit(jax.vmap(eval_action))
        return np.asarray(self._jit_vmap_eval_action[num_actions](key, self.state, obs))

    def _setup_value_network(self, params):
        """Construct the V(s) network used by KL-budget / on-policy-EMA mode.
        Returns (value_params, value_opt_state) or (None, None) when V is not
        used.

        The actual V-net plumbing (init, apply, vmap-apply, TD update step)
        lives in :class:`relax.algorithm.value_head.ValueHead`; we just
        construct it here.
        """
        if not self.use_advantage_stats:
            self.value_head = None
            return None, None

        self.value_head = ValueHead.create(self._obs_dim, self._hidden_dim, orthogonal_init=self.cfg.orthogonal_init)
        value_params_init = self.value_head.init_params(jax.random.PRNGKey(42))
        value_opt_state_init = self.value_head.init_opt_state(value_params_init, self.optim)
        return value_params_init, value_opt_state_init

    def _build_initial_state(self, params, value_params_init, value_opt_state_init):
        """Construct a fresh Diffv2TrainState from a given set of network params.

        Factored out of __init__ so vmap-mode setup can call it N times with
        different init seeds and stack the results along a leading seed axis.
        """
        cfg = self.cfg
        # The target policy exists only under --use_target_policy_training, and
        # starts as an exact copy of the online policy (mirrors target-Q init).
        if self.use_target_policy_training and params.target_policy is None:
            params = params._replace(target_policy=params.policy)
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
            # Cosine's shape, cut to the allowed log-SNR range: the layout the
            # fixed families already use, from which the equal-cost rule departs.
            log_snr_levels=(jnp.asarray(cosine_log_snr_knots(
                self._timesteps, cfg.noise_schedule_log_snr_min, cfg.noise_schedule_log_snr_max))
                if self.adaptive_schedule else None),
            distill_buffer=jnp.zeros(
                (cfg.batch_size * self.num_denoised_actions * cfg.distillation_buffer_size,
                 self._obs_dim + self.model.act_dim), jnp.float32),
            value_params=value_params_init,
            advantage_second_moment_ema=jnp.float32(cfg.initial_advantage_second_moment_ema),
            advantage_third_moment_ema=jnp.float32(0.0),
            dist_shift_covariance_ema=jnp.float32(0.0),
            dist_shift_shape_ema=jnp.float32(cfg.initial_dist_shift_shape_ema),
            q_running_mean=jnp.float32(0.0),
            q_running_std=jnp.float32(1.0),
            sigma_q_within_ema=jnp.float32(0.0), sigma_q_within_sq_ema=jnp.float32(0.0),
            log_s_ema=jnp.float32(0.0), log_s_sq_ema=jnp.float32(0.0),
            hp=HParams(
                gamma=jnp.float32(cfg.gamma),
                q_polyak_tau=jnp.float32(cfg.q_polyak_tau),
                policy_polyak_tau=jnp.float32(cfg.policy_polyak_tau),
                delay_target_q_update=jnp.float32(cfg.delay_target_q_update),
                delay_policy_update=jnp.float32(cfg.delay_policy_update),
                delay_target_policy_update=jnp.float32(cfg.delay_target_policy_update),
                lr_q=jnp.float32(cfg.lr_q),
                lr_policy=jnp.float32(cfg.lr_policy),
                guidance_mult=jnp.float32(cfg.guidance_strength_multiplier),
                guidance_mult_increasing=jnp.float32(cfg.guidance_strength_schedule == "increasing"),
                adv_ema_tau=jnp.float32(cfg.advantage_ema_tau),
                shape_ema_tau=jnp.float32(cfg.shape_ema_tau),
                adv_norm_ema_rate=jnp.float32(cfg.advantage_norm_ema_rate),
                s_hat_ema_rate=jnp.float32(cfg.s_hat_ema_rate),
                kl_budget_val=jnp.float32(kl_budget_val),
                reward_scale=jnp.float32(cfg.reward_scale),
                x0_hat_clip_radius=jnp.float32(cfg.x0_hat_clip_radius),
                mala_adapt_rate=jnp.float32(cfg.mala_adapt_rate),
                q_td_huber_width=jnp.float32(cfg.q_td_huber_width),
                alpha=jnp.float32(cfg.alpha),
                T=jnp.float32(cfg.T),
                eta=jnp.float32(cfg.eta),
                s_hat=jnp.float32(cfg.s_hat),
                noise_schedule_gamma=jnp.float32(cfg.noise_schedule_gamma),
                noise_schedule_warmup=jnp.float32(cfg.noise_schedule_warmup),
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
