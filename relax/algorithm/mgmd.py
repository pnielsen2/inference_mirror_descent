import math
from typing import Optional, Tuple

import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk

from relax.algorithm.diffusion_sampler import build_diffusion_sampler
from relax.algorithm.mala_sampler import build_mala_sampler
from relax.algorithm.mgmd_types import (
    Diffv2OptStates,
    HParams,
    Diffv2TrainState,
    MalaSampleResult,
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
        self.use_advantage_stats = bool(cfg.advantage_normalization or cfg.kl_budget is not None)
        self.on_policy_ema = self.use_advantage_stats
        self.one_step_dist_shift_beta = bool(cfg.one_step_dist_shift_beta)
        self.mgmd_variant = cfg.mgmd_variant
        self.rsm_variant = self.mgmd_variant == "rsm"
        self.soft_resample_variant = self.mgmd_variant == "soft_resample"
        # V-free guidance-normalization knobs (see relax/cli/train_args.py).
        self.num_denoised_actions = int(cfg.num_denoised_actions)
        self.soft_resample_actions = int(cfg.soft_resample_actions)
        self.best_of_n_actions = int(cfg.best_of_n_actions)
        self.best_of_n_td_action_sampling = bool(cfg.best_of_n_td_action_sampling)
        self.best_of_n_td_actions = int(
            cfg.best_of_n_td_actions if cfg.best_of_n_td_actions is not None else cfg.best_of_n_actions
        )
        self.uses_best_of_n_noise = (
            self.best_of_n_actions > 1
            or (self.best_of_n_td_action_sampling and self.best_of_n_td_actions > 1)
        )
        self.q_loss_normalization = bool(cfg.q_loss_normalization)
        self.batch_advantage_normalization = bool(cfg.batch_advantage_normalization)
        self.ema_advantage_normalization = bool(cfg.ema_advantage_normalization)
        self.lr_anneal = bool(cfg.lr_anneal)
        self.policy_loss_key = (
            "losses/Policy_RSM_weighted_epsilon_MSE"
            if self.rsm_variant else
            "losses/Policy_soft_resample_weighted_epsilon_MSE"
            if self.soft_resample_variant else
            "losses/Policy_epsilon_MSE"
        )

        # --- Optimizers: unscaled Adam; per-seed state.lr_{q,policy} is applied at update time. ---
        self.optim = optax.scale_by_adam()
        self.policy_optim = optax.scale_by_adam()
        self.best_of_n_noise_optim = optax.adam(self.cfg.best_of_n_noise_lr)

        # --- Optional V(s) network for KL-budget / on-policy-EMA beta adaptation and logging. ---
        value_params_init, value_opt_state_init = self._setup_value_network(params)

        self._timesteps = int(self.model.num_timesteps)

        # --- Initial vmap-stackable train state ---
        self.state = self._build_initial_state(params, value_params_init, value_opt_state_init)

        # --- Schedule cache for wandb SNR x-axis ---
        self._alphas_cumprod = np.asarray(self.model.schedule.alphas_cumprod)  # [T]
        self._snr = self._alphas_cumprod / np.maximum(1.0 - self._alphas_cumprod, 1e-8)

        # --- Stateless update / sampler closures (jit-able, vmap-able). ---
        # Default MGMD uses MALA-guided sampling. The RSM variant binds to a
        # plain DDPM sampler for both rollout and TD next-action sampling.
        _sampler_kw = dict(
            model=self.model, value_head=self.value_head, timesteps=self._timesteps,
            batch_independent_guidance=self.cfg.batch_independent_guidance,
            ema_normalization=bool(self.cfg.advantage_normalization or self.cfg.q_loss_normalization),
            denoising_predictor=self.cfg.denoising_predictor,
            guidance_gradient_space=self.cfg.guidance_gradient_space,
            num_denoised_actions=self.num_denoised_actions,
            batch_advantage_normalization=self.batch_advantage_normalization,
            ema_advantage_normalization=self.ema_advantage_normalization,
        )
        # Both sampling paths (rollout + TD next-action) use --q_agg_sample aggregation.
        # The TD-backup target itself remains hardcoded to 'min' (clipped double-Q).
        agg_critic = lambda qm: _aggregate_q(qm, self.cfg.q_agg_sample)
        if self.rsm_variant:
            rollout_sampler = build_diffusion_sampler(
                model=self.model,
                timesteps=self._timesteps,
                num_denoised_actions=1,
            )
            td_sampler = build_diffusion_sampler(
                model=self.model,
                timesteps=self._timesteps,
                num_denoised_actions=self.num_denoised_actions,
                compute_final_q=False,
            )
            training_num_actions = self.num_denoised_actions
        elif self.soft_resample_variant:
            proposal_sampler = build_diffusion_sampler(
                model=self.model,
                timesteps=self._timesteps,
                num_denoised_actions=self.soft_resample_actions,
                compute_final_q=True,
            )
            rollout_sampler = self._stateless_soft_resample_sampler(proposal_sampler)
            td_sampler = self._stateless_soft_resample_sampler(proposal_sampler)
            training_num_actions = 1
        else:
            rollout_sampler = build_mala_sampler(
                **dict(_sampler_kw, num_denoised_actions=self.best_of_n_actions)
            )
            if self.best_of_n_td_action_sampling:
                td_sampler = self._stateless_best_of_n_training_sampler(
                    build_mala_sampler(
                        **dict(_sampler_kw, num_denoised_actions=self.best_of_n_td_actions)
                    )
                )
                training_num_actions = 1
            else:
                td_sampler = build_mala_sampler(**_sampler_kw, compute_final_q=False)
                training_num_actions = self.num_denoised_actions
        updater = self._stateless_update(td_sampler, agg_critic, training_num_actions)
        stateless_get_action = self._stateless_get_action(rollout_sampler, agg_critic)
        self._jit_vmap_update = jax.jit(jax.vmap(updater))
        self._jit_vmap_get_action = jax.jit(jax.vmap(stateless_get_action))
        self._jit_vmap_eval_action_by_n = {}
        self._last_rollout_soft_resample_ess = None
        self._last_rollout_soft_resample_pmax = None
        self._last_soft_resample_array_log_step = -float("inf")

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
        if self.soft_resample_variant:
            dump_arrays = self._soft_resample_array_dump_due(env_step)
            self._augment_soft_resample_rollout_info(scalar_info, array_info, dump_arrays)
            self._filter_soft_resample_array_dumps(array_info, dump_arrays)
        return scalar_info, array_info, actions

    def _soft_resample_array_dump_due(self, env_step: Optional[float]) -> bool:
        interval = int(self.cfg.soft_resample_ess_dump_interval)
        if interval <= 0 or env_step is None:
            return False
        env_step_f = float(env_step)
        if env_step_f - self._last_soft_resample_array_log_step < interval:
            return False
        self._last_soft_resample_array_log_step = env_step_f
        return True

    def _augment_soft_resample_rollout_info(self, scalar_info: dict, array_info: dict, dump_arrays: bool) -> None:
        ess = self._last_rollout_soft_resample_ess
        pmax = self._last_rollout_soft_resample_pmax
        if ess is None or pmax is None:
            return
        _add_np_summary_info(scalar_info, "SoftResample/Rollout/ess", ess)
        _add_np_summary_info(scalar_info, "SoftResample/Rollout/pmax", pmax)
        for slot in range(ess.shape[1]):
            scalar_info[f"SoftResample/Rollout/ess_slot_{slot}"] = ess[:, slot]
            scalar_info[f"SoftResample/Rollout/pmax_slot_{slot}"] = pmax[:, slot]
        if dump_arrays:
            array_info["SoftResample/Rollout/ess_hist"] = ess
            array_info["SoftResample/Rollout/pmax_hist"] = pmax

    def _filter_soft_resample_array_dumps(self, array_info: dict, dump_arrays: bool) -> None:
        if dump_arrays:
            return
        for key in list(array_info.keys()):
            if key.startswith("SoftResample/TD/") and key.endswith("_hist"):
                array_info.pop(key)

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

    def _stateless_get_action(self, sampler, agg_sample_fn):
        """Return rollout sampler closure.

        ``best_of_n_actions=1`` preserves the existing single-sample rollout.
        ``best_of_n_actions>1`` selects the highest final online-Q candidate and
        then adds DPMD-style learned Gaussian execution noise. The selected Q is
        used only for choosing the pre-noise candidate; the returned Q is
        recomputed at the actually executed noisy action.
        """
        def stateless_get_action(key: jax.Array, state: Diffv2TrainState, obs: jax.Array):
            sample_key, noise_key = jax.random.split(key)
            result = sampler(sample_key, state, obs, agg_sample_fn)
            if self.soft_resample_variant:
                return MalaSampleResult(
                    action=result.action[0],
                    q=result.q[0],
                    log_eta_scales=result.log_eta_scales,
                    per_level_acc=result.per_level_acc,
                    per_level_clip=result.per_level_clip,
                    candidate_action=result.candidate_action,
                    weights=result.weights,
                    ess=result.ess,
                    pmax=result.pmax,
                    selected_idx=result.selected_idx,
                )
            if self.best_of_n_actions == 1:
                return MalaSampleResult(
                    action=result.action[0],
                    q=result.q[0],
                    log_eta_scales=result.log_eta_scales,
                    per_level_acc=result.per_level_acc,
                    per_level_clip=result.per_level_clip,
                )

            best_idx = jnp.argmax(result.q, axis=0)
            gather_idx = best_idx[None, ...]
            while gather_idx.ndim < result.action.ndim:
                gather_idx = gather_idx[..., None]
            best_action = jnp.take_along_axis(result.action, gather_idx, axis=0).squeeze(axis=0)
            noise_scale = jnp.exp(state.log_best_of_n_noise_scale)
            exec_action = best_action + jax.random.normal(noise_key, best_action.shape) * noise_scale
            exec_q = _aggregate_q(
                [self.model.q(qp, obs, exec_action) for qp in state.params.q],
                self.cfg.q_agg_sample,
            )
            return MalaSampleResult(
                action=exec_action,
                q=exec_q,
                log_eta_scales=result.log_eta_scales,
                per_level_acc=result.per_level_acc,
                per_level_clip=result.per_level_clip,
            )

        return stateless_get_action

    def _stateless_eval_action(self, sampler, agg_sample_fn):
        """Eval-only best-of-N action selector.

        Evaluation intentionally differs from rollout best-of-N in one way:
        after selecting the highest-Q candidate, it executes that candidate
        directly, without the learned post-selection Gaussian exploration noise.
        """
        def stateless_eval_action(key: jax.Array, state: Diffv2TrainState, obs: jax.Array):
            result = sampler(key, state, obs, agg_sample_fn)
            best_idx = jnp.argmax(result.q, axis=0)
            gather_idx = best_idx[None, ...]
            while gather_idx.ndim < result.action.ndim:
                gather_idx = gather_idx[..., None]
            action = jnp.take_along_axis(result.action, gather_idx, axis=0).squeeze(axis=0)
            q = jnp.take_along_axis(result.q, best_idx[None, ...], axis=0).squeeze(axis=0)
            return action, q

        return stateless_eval_action

    def _build_eval_action_vmap(self, num_actions: int):
        num_actions = int(num_actions)
        agg_critic = lambda qm: _aggregate_q(qm, self.cfg.q_agg_sample)
        if self.rsm_variant or self.soft_resample_variant:
            sampler = build_diffusion_sampler(
                model=self.model,
                timesteps=self._timesteps,
                num_denoised_actions=num_actions,
                compute_final_q=True,
            )
        else:
            sampler = build_mala_sampler(
                model=self.model,
                value_head=self.value_head,
                timesteps=self._timesteps,
                batch_independent_guidance=self.cfg.batch_independent_guidance,
                ema_normalization=bool(self.cfg.advantage_normalization or self.cfg.q_loss_normalization),
                denoising_predictor=self.cfg.denoising_predictor,
                guidance_gradient_space=self.cfg.guidance_gradient_space,
                num_denoised_actions=num_actions,
                batch_advantage_normalization=self.batch_advantage_normalization,
                ema_advantage_normalization=self.ema_advantage_normalization,
            )
        return jax.jit(jax.vmap(self._stateless_eval_action(sampler, agg_critic)))

    def _stateless_best_of_n_training_sampler(self, sampler):
        """Return a TD next-action sampler with DPMD-style best-of-N selection.

        The wrapped sampler emits N candidate actions per replay state. This
        closure selects the highest final online-Q candidate, adds the same
        learned Gaussian best-of-N noise used by rollout, recomputes online Q at
        the executed action, and restores a leading singleton action axis so the
        training update can keep its [K, batch, ...] tensor convention.
        """
        def stateless_best_of_n_training_sampler(
            key: jax.Array,
            state: Diffv2TrainState,
            obs: jax.Array,
            aggregate_q_fn,
        ):
            sample_key, noise_key = jax.random.split(key)
            result = sampler(sample_key, state, obs, aggregate_q_fn)
            if self.best_of_n_td_actions == 1:
                return result

            best_idx = jnp.argmax(result.q, axis=0)
            gather_idx = best_idx[None, ...]
            while gather_idx.ndim < result.action.ndim:
                gather_idx = gather_idx[..., None]
            best_action = jnp.take_along_axis(result.action, gather_idx, axis=0).squeeze(axis=0)

            noise_scale = jnp.exp(state.log_best_of_n_noise_scale)
            exec_action = best_action + jax.random.normal(noise_key, best_action.shape) * noise_scale
            exec_q = aggregate_q_fn(
                [self.model.q(qp, obs, exec_action) for qp in state.params.q],
            )
            return MalaSampleResult(
                action=exec_action[None, ...],
                q=exec_q[None, ...],
                log_eta_scales=result.log_eta_scales,
                per_level_acc=result.per_level_acc,
                per_level_clip=result.per_level_clip,
            )

        return stateless_best_of_n_training_sampler

    def _stateless_soft_resample_sampler(self, sampler):
        """Return a sampler targeting pi_old(a|s) * exp(beta * processed_Q).

        The wrapped plain-diffusion sampler emits N proposal actions per state.
        This closure computes per-state Boltzmann weights, samples one proposal
        for rollout / TD backup, and keeps the full candidate set + weights for
        policy distillation and ESS logging.
        """
        def stateless_soft_resample_sampler(
            key: jax.Array,
            state: Diffv2TrainState,
            obs: jax.Array,
            aggregate_q_fn,
        ):
            sample_key, choice_key = jax.random.split(key)
            result = sampler(sample_key, state, obs, aggregate_q_fn)
            q_processed = _processed_q_for_resampling(
                result.q,
                state,
                self.batch_advantage_normalization,
                self.ema_advantage_normalization,
            )
            beta_resample = _soft_resample_beta(
                state,
                self.cfg.advantage_normalization or self.q_loss_normalization,
            )
            weights, ess, pmax = _soft_resample_weights_from_q(q_processed, beta_resample)
            logits = beta_resample * q_processed
            selected_idx = jax.random.categorical(choice_key, logits, axis=0)

            gather_idx = selected_idx[None, ...]
            while gather_idx.ndim < result.action.ndim:
                gather_idx = gather_idx[..., None]
            selected_action = jnp.take_along_axis(result.action, gather_idx, axis=0).squeeze(axis=0)
            selected_q = jnp.take_along_axis(result.q, selected_idx[None, ...], axis=0).squeeze(axis=0)

            return MalaSampleResult(
                action=selected_action[None, ...],
                q=selected_q[None, ...],
                log_eta_scales=result.log_eta_scales,
                per_level_acc=result.per_level_acc,
                per_level_clip=result.per_level_clip,
                candidate_action=result.action,
                weights=weights,
                ess=ess,
                pmax=pmax,
                selected_idx=selected_idx,
            )

        return stateless_soft_resample_sampler

    def _best_of_n_noise_loss(self, log_noise_scale):
        noise_scale = jnp.exp(log_noise_scale)
        entropy_approx = jnp.float32(0.5 * self.model.act_dim) * jnp.log(
            jnp.float32(2.0 * math.pi * math.e) * noise_scale * noise_scale
        )
        target_entropy = jnp.float32(-self.cfg.best_of_n_noise_target_entropy_scale * self.model.act_dim)
        loss = log_noise_scale * jax.lax.stop_gradient(entropy_approx - target_entropy)
        return loss, entropy_approx

    def _stateless_update(self, sampler, agg_sample_fn, training_num_actions: int):
        """Return the ``stateless_update(key, state, data, critic_weight)`` closure.

        Captures ``sampler`` (MALA or plain diffusion sampler) and ``agg_sample_fn`` (Q aggregation
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
            q_opt_states = state.opt_state.q   # tuple of N opt states
            policy_opt_state = state.opt_state.policy
            best_of_n_noise_opt_state = state.opt_state.best_of_n_noise
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

            # Denoise K tilted next-actions per replay state. By default
            # K = num_denoised_actions. With --best_of_n_td_action_sampling,
            # the sampler internally denoises N=best_of_n_td_actions candidates,
            # selects one DPMD-style action, and returns it as K=1.
            mala_result = sampler(next_eval_key, state, next_obs, agg_sample_fn)
            tilted_actions = mala_result.action           # [K, batch, act_dim]
            K = training_num_actions
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
            q_norm_samples = None
            if self.ema_advantage_normalization or self.rsm_variant:
                q_norm_samples = _aggregate_q(
                    [self.model.q(qp, next_obs_k, tilted_actions) for qp in q_params],
                    self.cfg.q_agg_sample,
                )  # [K, batch] online Q at the tilted next-actions
            if self.soft_resample_variant:
                q_norm_samples = _aggregate_q(
                    [self.model.q(qp, next_obs_k, tilted_actions) for qp in q_params],
                    self.cfg.q_agg_sample,
                )
                if mala_result.candidate_action is not None:
                    candidate_next_obs = jnp.broadcast_to(
                        next_obs,
                        (self.soft_resample_actions, *next_obs.shape),
                    )
                    q_norm_samples = _aggregate_q(
                        [self.model.q(qp, candidate_next_obs, mala_result.candidate_action) for qp in q_params],
                        self.cfg.q_agg_sample,
                    )  # [N, batch] online Q at all soft-resample candidates
            if self.ema_advantage_normalization:
                rate = state.hp.adv_norm_ema_rate
                new_q_running_mean = state.q_running_mean + rate * (jnp.mean(q_norm_samples) - state.q_running_mean)
                new_q_running_std = state.q_running_std + rate * (jnp.std(q_norm_samples) - state.q_running_std)

            target_q_params = tuple(delayed_target_update(q_params[i], target_q_params[i], state.hp.polyak_tau, step, self.cfg.delay_update) for i in range(num_q))

            # Diffusion policy regresses toward all K tilted actions; flatten the
            # K axis into the batch so score-matching sees K*batch targets (for
            # K=1 these are exactly the previous [batch, ...] tensors).
            if self.soft_resample_variant:
                policy_K = self.soft_resample_actions
                policy_action_block = mala_result.candidate_action
                policy_obs_block = jnp.broadcast_to(next_obs, (policy_K, *next_obs.shape))
                soft_policy_weights = jax.lax.stop_gradient(mala_result.weights)
            else:
                policy_K = K
                policy_action_block = tilted_actions
                policy_obs_block = next_obs_k
                soft_policy_weights = None
            policy_targets = policy_action_block.reshape(policy_K * obs.shape[0], -1)   # [policy_K*batch, act_dim]
            policy_obs = policy_obs_block.reshape(policy_K * obs.shape[0], -1)           # [policy_K*batch, obs_dim]
            rsm_q_used = jnp.zeros((policy_targets.shape[0],), dtype=policy_targets.dtype)
            rsm_scaled_q = jnp.zeros((policy_targets.shape[0],), dtype=policy_targets.dtype)
            rsm_weights = jnp.ones((policy_targets.shape[0], 1), dtype=policy_targets.dtype)
            if self.rsm_variant:
                q_for_rsm = q_norm_samples.reshape(K * obs.shape[0])
                if self.ema_advantage_normalization:
                    q_for_rsm = (q_for_rsm - state.q_running_mean) / jnp.maximum(state.q_running_std, jnp.float32(1e-6))
                rsm_q_used = q_for_rsm
                rsm_scaled_q = jnp.clip(state.beta * q_for_rsm, jnp.float32(-3.0), jnp.float32(3.0))
                rsm_weights = jax.lax.stop_gradient(jnp.exp(rsm_scaled_q))[..., None]

            def policy_loss_fn(policy_params, time_key, noise_key) -> jax.Array:
                # Default MGMD: unweighted eps-MSE distillation. RSM variant:
                # same diffusion target actions, weighted by exp(beta * Q_used).
                t = jax.random.randint(
                    time_key,
                    (policy_targets.shape[0],),
                    0,
                    self.model.num_timesteps,
                )
                noise = jax.random.normal(noise_key, policy_targets.shape)
                tilted_action_noisy = self.model.q_sample(t, policy_targets, noise)
                noise_pred = self.model.eps_pred(policy_params, policy_obs, tilted_action_noisy, t)
                per_dim_loss = optax.squared_error(noise_pred, noise)
                if self.rsm_variant:
                    return (rsm_weights * per_dim_loss).mean()
                if self.soft_resample_variant:
                    per_action_loss = jnp.mean(per_dim_loss, axis=-1).reshape(policy_K, obs.shape[0])
                    return jnp.mean(jnp.sum(soft_policy_weights * per_action_loss, axis=0))
                return per_dim_loss.mean()

            def _do(_):
                updated_policy_params = policy_params
                updated_policy_opt_state = policy_opt_state
                for policy_step_idx in range(self.cfg.policy_update_steps):
                    # Preserve the exact old random stream for P=1; additional
                    # steps derive independent keys from the same base keys.
                    if policy_step_idx == 0:
                        time_key = diffusion_time_key
                        noise_key = diffusion_noise_key
                    else:
                        time_key = jax.random.fold_in(diffusion_time_key, policy_step_idx)
                        noise_key = jax.random.fold_in(diffusion_noise_key, policy_step_idx)
                    loss, grads = jax.value_and_grad(policy_loss_fn)(
                        updated_policy_params, time_key, noise_key
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
                step % self.cfg.delay_update == 0, _do,
                lambda _: (state.policy_loss, policy_params, policy_opt_state), None)

            log_best_of_n_noise_scale = state.log_best_of_n_noise_scale
            best_of_n_noise_loss = jnp.float32(0.0)
            best_of_n_noise_entropy = jnp.float32(0.0)
            if self.uses_best_of_n_noise:
                def _update_noise(_):
                    (loss, entropy_approx), grads = jax.value_and_grad(
                        self._best_of_n_noise_loss, has_aux=True
                    )(log_best_of_n_noise_scale)
                    updates, new_opt_state = self.best_of_n_noise_optim.update(
                        grads, best_of_n_noise_opt_state, params=log_best_of_n_noise_scale
                    )
                    new_log_noise = optax.apply_updates(log_best_of_n_noise_scale, updates)
                    return new_log_noise, new_opt_state, loss, entropy_approx

                def _skip_noise(_):
                    loss, entropy_approx = self._best_of_n_noise_loss(log_best_of_n_noise_scale)
                    return log_best_of_n_noise_scale, best_of_n_noise_opt_state, loss, entropy_approx

                log_best_of_n_noise_scale, best_of_n_noise_opt_state, best_of_n_noise_loss, best_of_n_noise_entropy = jax.lax.cond(
                    step % self.cfg.delay_best_of_n_noise_update == 0,
                    _update_noise,
                    _skip_noise,
                    None,
                )

            state = state._replace(
                params=ActorCriticParams(q_params, target_q_params, policy_params),
                opt_state=Diffv2OptStates(
                    q=q_opt_states,
                    policy=policy_opt_state,
                    best_of_n_noise=best_of_n_noise_opt_state,
                    value=value_opt_state_updated,
                ),
                step=step + 1,
                log_eta_scales=mala_result.log_eta_scales,
                value_params=value_params_updated,
                log_best_of_n_noise_scale=log_best_of_n_noise_scale,
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
            if self.uses_best_of_n_noise:
                info["BestOfN/noise_scale"] = jnp.exp(log_best_of_n_noise_scale)
                info["BestOfN/noise_entropy_approx"] = best_of_n_noise_entropy
                info["BestOfN/noise_loss"] = best_of_n_noise_loss
                info["BestOfN/num_actions"] = jnp.float32(self.best_of_n_actions)
                info["BestOfN/rollout_num_actions"] = jnp.float32(self.best_of_n_actions)
                info["BestOfN/td_num_actions"] = jnp.float32(self.best_of_n_td_actions)
                info["BestOfN/td_action_sampling"] = jnp.float32(self.best_of_n_td_action_sampling)
            if self.rsm_variant:
                info["RSM/q_used_mean"] = jnp.mean(rsm_q_used)
                info["RSM/q_used_std"] = jnp.std(rsm_q_used)
                info["RSM/scaled_q_mean"] = jnp.mean(rsm_scaled_q)
                info["RSM/scaled_q_std"] = jnp.std(rsm_scaled_q)
                info["RSM/weights_mean"] = jnp.mean(rsm_weights)
                info["RSM/weights_std"] = jnp.std(rsm_weights)
                info["RSM/weights_min"] = jnp.min(rsm_weights)
                info["RSM/weights_max"] = jnp.max(rsm_weights)
                info["RSM/num_denoised_actions"] = jnp.float32(self.num_denoised_actions)
            if self.soft_resample_variant:
                td_ess = mala_result.ess.reshape(-1)
                td_pmax = mala_result.pmax.reshape(-1)
                td_selected_idx = mala_result.selected_idx.reshape(-1).astype(jnp.float32)
                info.update(_summary_info("SoftResample/TD/ess", td_ess))
                info.update(_summary_info("SoftResample/TD/pmax", td_pmax))
                info.update(_summary_info("SoftResample/TD/selected_idx", td_selected_idx))
                info["SoftResample/actions"] = jnp.float32(self.soft_resample_actions)
                info["SoftResample/beta_resample"] = _soft_resample_beta(
                    state,
                    self.cfg.advantage_normalization or self.q_loss_normalization,
                )
                info["SoftResample/TD/ess_hist"] = td_ess
                info["SoftResample/TD/pmax_hist"] = td_pmax

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
        if self.soft_resample_variant:
            self._last_rollout_soft_resample_ess = np.asarray(result.ess)
            self._last_rollout_soft_resample_pmax = np.asarray(result.pmax)
        # _stateless_get_action already returns the selected rollout action:
        # index-0 for best_of_n_actions=1, or best-of-N plus learned noise for N>1.
        action_np = np.asarray(result.action)  # [N, num_envs, act_dim]
        q_per_env = np.asarray(result.q)       # [N, num_envs]

        if not self.on_policy_ema:
            return action_np, q_per_env, None

        v = self.value_head.apply_vmap(self.state.value_params, jnp.asarray(obs))
        v_per_env = np.asarray(v)  # [N, num_envs]
        return action_np, q_per_env, v_per_env

    def get_eval_action_vmap(self, key: jax.Array, obs: np.ndarray, num_actions: int):
        """Vmapped separate-eval action.

        ``num_actions`` is the evaluation-only best-of-N count. The returned
        action has shape [num_runs, eval_envs_per_run, act_dim]. This method
        does not update ``self.state`` or MALA adaptation statistics.
        """
        num_actions = int(num_actions)
        if num_actions not in self._jit_vmap_eval_action_by_n:
            self._jit_vmap_eval_action_by_n[num_actions] = self._build_eval_action_vmap(num_actions)
        action, q = self._jit_vmap_eval_action_by_n[num_actions](key, self.state, obs)
        return np.asarray(action), np.asarray(q)

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
        # In-graph kl_budget sentinel: when --kl_budget is disabled (None),
        # store 1.0 so the value can still be a jnp.float32 in the vmappable
        # HParams; the host-side β cap reads ``cfg.kl_budget`` directly.
        kl_budget_val = 1.0 if cfg.kl_budget is None else cfg.kl_budget
        return Diffv2TrainState(
            params=params,
            opt_state=Diffv2OptStates(
                q=tuple(self.optim.init(qp) for qp in params.q),
                policy=self.policy_optim.init(params.policy),
                best_of_n_noise=self.best_of_n_noise_optim.init(
                    jnp.float32(math.log(cfg.best_of_n_noise_scale_init))
                ),
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
            q_running_mean=jnp.float32(0.0),
            q_running_std=jnp.float32(1.0),
            log_best_of_n_noise_scale=jnp.float32(math.log(cfg.best_of_n_noise_scale_init)),
            hp=HParams(
                gamma=jnp.float32(cfg.gamma),
                polyak_tau=jnp.float32(cfg.polyak_tau),
                lr_q=jnp.float32(cfg.lr_q),
                lr_policy=jnp.float32(cfg.lr_policy),
                guidance_mult=jnp.float32(cfg.guidance_strength_multiplier),
                guidance_mult_increasing=jnp.float32(cfg.guidance_strength_schedule == "increasing"),
                adv_ema_tau=jnp.float32(cfg.advantage_ema_tau),
                shape_ema_tau=jnp.float32(cfg.shape_ema_tau),
                adv_norm_ema_rate=jnp.float32(cfg.advantage_norm_ema_rate),
                kl_budget_val=jnp.float32(kl_budget_val),
                reward_scale=jnp.float32(cfg.reward_scale),
                x0_hat_clip_radius=jnp.float32(cfg.x0_hat_clip_radius),
                mala_adapt_rate=jnp.float32(cfg.mala_adapt_rate),
                q_td_huber_width=jnp.float32(cfg.q_td_huber_width),
                alpha=jnp.float32(cfg.alpha),
                T=jnp.float32(cfg.T),
                eta=jnp.float32(cfg.eta),
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
            "mgmd_variant_effective": self.cfg.mgmd_variant,
            "lr_policy_effective": float(self.cfg.lr_policy),
            "lr_q_effective": float(self.cfg.lr_q),
            "best_of_n_noise_scale_init_effective": float(self.cfg.best_of_n_noise_scale_init),
            "best_of_n_td_action_sampling_effective": bool(self.cfg.best_of_n_td_action_sampling),
            "best_of_n_td_actions_effective": int(self.best_of_n_td_actions),
            "soft_resample_actions_effective": int(self.soft_resample_actions),
            "soft_resample_ess_dump_interval_effective": int(self.cfg.soft_resample_ess_dump_interval),
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


def _processed_q_for_resampling(q: jax.Array, state: Diffv2TrainState,
                                batch_advantage_normalization: bool,
                                ema_advantage_normalization: bool) -> jax.Array:
    """Apply the same scale conventions used by MGMD's Q-guided sampler."""
    out = q
    if batch_advantage_normalization:
        denom = jnp.sqrt(jnp.maximum(jnp.mean(jnp.var(out, axis=0, ddof=1)), jnp.float32(1e-6)))
        out = out / jax.lax.stop_gradient(denom)
    if ema_advantage_normalization:
        mean = jax.lax.stop_gradient(state.q_running_mean)
        std = jnp.maximum(jax.lax.stop_gradient(state.q_running_std), jnp.float32(1e-6))
        out = (out - mean) / std
    return out


def _soft_resample_beta(state: Diffv2TrainState, q_scale_ema_normalization: bool) -> jax.Array:
    if not q_scale_ema_normalization:
        return state.beta
    return state.beta / jnp.sqrt(jnp.maximum(state.advantage_second_moment_ema, jnp.float32(1e-6)))


def _soft_resample_weights_from_q(q: jax.Array, beta: jax.Array):
    """Return per-state softmax weights, ESS, and max probability.

    ``q`` is shaped ``[N, ...]``. The leading axis is the candidate axis and
    every remaining position is one independent state. Normalization is over
    the candidate axis only.
    """
    logits = beta * q
    weights = jax.nn.softmax(logits, axis=0)
    ess = jnp.float32(1.0) / jnp.maximum(jnp.sum(weights * weights, axis=0), jnp.float32(1e-12))
    pmax = jnp.max(weights, axis=0)
    return weights, ess, pmax


def _summary_info(prefix: str, values: jax.Array) -> dict:
    flat = jnp.ravel(values)
    return {
        f"{prefix}_mean": jnp.mean(flat),
        f"{prefix}_std": jnp.std(flat),
        f"{prefix}_min": jnp.min(flat),
        f"{prefix}_max": jnp.max(flat),
        f"{prefix}_p10": jnp.quantile(flat, jnp.float32(0.10)),
        f"{prefix}_p25": jnp.quantile(flat, jnp.float32(0.25)),
        f"{prefix}_p50": jnp.quantile(flat, jnp.float32(0.50)),
        f"{prefix}_p75": jnp.quantile(flat, jnp.float32(0.75)),
        f"{prefix}_p90": jnp.quantile(flat, jnp.float32(0.90)),
    }


def _add_np_summary_info(info: dict, prefix: str, values: np.ndarray) -> None:
    arr = np.asarray(values, dtype=np.float32)
    info[f"{prefix}_mean"] = np.mean(arr, axis=1)
    info[f"{prefix}_std"] = np.std(arr, axis=1)
    info[f"{prefix}_min"] = np.min(arr, axis=1)
    info[f"{prefix}_max"] = np.max(arr, axis=1)
    for q, name in [(0.10, "p10"), (0.25, "p25"), (0.50, "p50"), (0.75, "p75"), (0.90, "p90")]:
        info[f"{prefix}_{name}"] = np.quantile(arr, q, axis=1)
