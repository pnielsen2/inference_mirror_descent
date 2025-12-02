from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle

from relax.algorithm.base import Algorithm
from relax.network.dacer import DACERNet, DACERParams
from relax.network.diffv2 import Diffv2Net, Diffv2Params
from relax.network.transition import TransitionNet, TransitionParams
from relax.utils.experience import Experience
from relax.utils.typing_utils import Metric


class Diffv2OptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState


class Diffv2TrainState(NamedTuple):
    params: Diffv2Params
    opt_state: Diffv2OptStates
    step: int
    entropy: float
    running_mean: float
    running_std: float


class TransitionOptStates(NamedTuple):
    dyn: optax.OptState | None
    reward: optax.OptState | None


class DPMDTrainState(NamedTuple):
    params: Diffv2Params
    opt_state: Diffv2OptStates
    step: int
    entropy: float
    running_mean: float
    running_std: float
    transition_params: TransitionParams | None
    transition_opt_state: TransitionOptStates | None

class DPMD(Algorithm):

    def __init__(
        self,
        agent: Diffv2Net,
        params: Diffv2Params,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        alpha_lr: float = 3e-2,
        lr_schedule_end: float = 5e-5,
        tau: float = 0.005,
        delay_alpha_update: int = 250,
        delay_update: int = 2,
        reward_scale: float = 0.2,
        num_samples: int = 200,
        use_ema: bool = True,
        use_reweighting: bool = True,
        q_critic_agg: str = "min",
        fix_q_norm_bug: bool = False,
        tfg_lambda: float = 0.0,
        tfg_lambda_schedule: str = "constant",
        tfg_recur_steps: int = 0,
        particle_selection_lambda: float = np.inf,
        critic_type: str = "q",
        state_population_size: int = 16,
        transition_refresh_type: str = "recur",
        transition_refresh_L: int = 3,
        bprop_refresh: int = 0,
        transition_use_crn: bool = False,
        transition_net: TransitionNet | None = None,
        transition_params: TransitionParams | None = None,
        td_guided_targets: bool = False,
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.delay_alpha_update = delay_alpha_update
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.num_samples = num_samples
        self.optim = optax.adam(lr)
        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr,
            end_value=lr_schedule_end,
            transition_steps=int(5e4),
            transition_begin=int(2.5e4),
        )
        self.policy_optim = optax.adam(learning_rate=lr_schedule)
        self.alpha_optim = optax.adam(alpha_lr)
        self.entropy = 0.0
        self.use_ema = use_ema
        self.use_reweighting = use_reweighting
        self.q_critic_agg = q_critic_agg
        self.fix_q_norm_bug = fix_q_norm_bug
        self.tfg_lambda = tfg_lambda
        self.tfg_lambda_schedule = tfg_lambda_schedule
        self.tfg_recur_steps = tfg_recur_steps
        self.particle_selection_lambda = particle_selection_lambda

        if critic_type not in ("q", "reward1step", "transition_reward"):
            raise ValueError(
                f"Invalid critic_type: {critic_type}. Expected 'q', 'reward1step', or 'transition_reward'."
            )

        # For now we only implement recurrence-style refresh; disallow other types explicitly.
        if transition_refresh_type != "recur":
            raise ValueError(
                f"Unsupported transition_refresh_type: {transition_refresh_type}. Only 'recur' is currently supported."
            )

        # Enforce that transition-only options are only used with the transition_reward critic.
        default_state_pop = 16
        default_refresh_type = "recur"
        default_refresh_L = 3
        default_bprop_refresh = 0
        default_use_crn = False

        if critic_type != "transition_reward":
            if (
                state_population_size != default_state_pop
                or transition_refresh_type != default_refresh_type
                or transition_refresh_L != default_refresh_L
                or bprop_refresh != default_bprop_refresh
                or transition_use_crn != default_use_crn
            ):
                raise ValueError(
                    "Transition-related options (state_population_size, transition_refresh_type, "
                    "transition_refresh_L, bprop_refresh, transition_use_crn) require critic_type='transition_reward'."
                )
        else:
            if transition_refresh_L < bprop_refresh:
                raise ValueError("transition_refresh_L must be >= bprop_refresh for transition_reward critic.")

        if critic_type == "transition_reward":
            if transition_net is None or transition_params is None:
                raise ValueError("critic_type='transition_reward' requires transition_net and transition_params.")
            self.transition_net = transition_net
            dyn_opt_state = self.optim.init(transition_params.dyn_params)
            reward_opt_state = self.optim.init(transition_params.reward_params)
            transition_opt_state = TransitionOptStates(dyn=dyn_opt_state, reward=reward_opt_state)
        else:
            self.transition_net = None
            transition_params = None
            transition_opt_state = TransitionOptStates(dyn=None, reward=None)

        self.critic_type = critic_type
        self.state_population_size = state_population_size
        self.transition_refresh_type = transition_refresh_type
        self.transition_refresh_L = transition_refresh_L
        self.bprop_refresh = bprop_refresh
        self.transition_use_crn = transition_use_crn
        self.td_guided_targets = td_guided_targets

        self.state = DPMDTrainState(
            params=params,
            opt_state=Diffv2OptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
            ),
            step=jnp.int32(0),
            entropy=jnp.float32(0.0),
            running_mean=jnp.float32(0.0),
            running_std=jnp.float32(1.0),
            transition_params=transition_params,
            transition_opt_state=transition_opt_state,
        )

        timesteps = self.agent.num_timesteps
        if self.tfg_lambda_schedule == "linear":
            idx = jnp.arange(timesteps, dtype=jnp.float32)
            denom = jnp.maximum(timesteps - 1, 1)
            t_levels = 1.0 - idx / denom
            lambda_levels = self.tfg_lambda * t_levels
        else:
            lambda_levels = self.tfg_lambda * jnp.ones((timesteps,), dtype=jnp.float32)

        def lambda_for_step(t_idx: jax.Array) -> jax.Array:
            t_next = jnp.maximum(t_idx - 1, 0)
            return lambda_levels[t_next]

        # Aggregation of twin Qs for *signals* (reweighting, tilt, logging, etc.).
        # TD targets below always use hard min(q1, q2) regardless of this setting.

        def hard_min_q(q1: jax.Array, q2: jax.Array) -> jax.Array:
            return jnp.minimum(q1, q2)

        if q_critic_agg == "min":
            def aggregate_q(q1: jax.Array, q2: jax.Array) -> jax.Array:
                return hard_min_q(q1, q2)
        elif q_critic_agg == "mean":
            def aggregate_q(q1: jax.Array, q2: jax.Array) -> jax.Array:
                return 0.5 * (q1 + q2)
        elif q_critic_agg == "max":
            def aggregate_q(q1: jax.Array, q2: jax.Array) -> jax.Array:
                return jnp.maximum(q1, q2)
        else:
            raise ValueError(f"Invalid q_critic_agg: {q_critic_agg}. Expected 'min', 'mean', or 'max'.")

        def critic_value_from_x0_td(params, obs_batch: jax.Array, actions_x0: jax.Array) -> jax.Array:
            policy_params, log_alpha, q1_params, q2_params, transition_params = params
            q1 = self.agent.q(q1_params, obs_batch, actions_x0)
            q2 = self.agent.q(q2_params, obs_batch, actions_x0)
            return hard_min_q(q1, q2)

        def critic_value_from_x0_env(params, obs_batch: jax.Array, actions_x0: jax.Array) -> jax.Array:
            policy_params, log_alpha, q1_params, q2_params, transition_params = params
            q1 = self.agent.q(q1_params, obs_batch, actions_x0)
            q2 = self.agent.q(q2_params, obs_batch, actions_x0)
            return aggregate_q(q1, q2)

        def init_sprime(obs_batch: jax.Array, key: jax.Array) -> jax.Array:
            batch_size = obs_batch.shape[0]
            obs_dim = obs_batch.shape[-1]
            if self.critic_type == "transition_reward":
                return jax.random.normal(
                    key,
                    (batch_size, self.state_population_size, obs_dim),
                    dtype=obs_batch.dtype,
                )
            return jnp.zeros(
                (batch_size, self.state_population_size, obs_dim),
                dtype=obs_batch.dtype,
            )

        def critic_step_env(
            params,
            obs_batch: jax.Array,
            actions_x0: jax.Array,
            sprime_t: jax.Array,
            t: int,
            key: jax.Array,
        ) -> Tuple[jax.Array, jax.Array]:
            if self.critic_type != "transition_reward":
                q_vals = critic_value_from_x0_env(params, obs_batch, actions_x0)
                return q_vals, sprime_t

            policy_params, log_alpha, q1_params, q2_params, transition_params = params
            dyn_params = transition_params.dyn_params
            reward_params = transition_params.reward_params

            B = self.transition_net.diffusion.beta_schedule()
            sqrt_recip = B.sqrt_recip_alphas_cumprod[t]
            sqrt_recipm1 = B.sqrt_recipm1_alphas_cumprod[t]
            sqrt_alpha_bar = B.sqrt_alphas_cumprod[t]
            sqrt_one_minus_alpha_bar = B.sqrt_one_minus_alphas_cumprod[t]

            batch_size = obs_batch.shape[0]
            K = self.state_population_size

            obs_exp = jnp.repeat(obs_batch[:, None, :], K, axis=1)
            act_exp = jnp.repeat(actions_x0[:, None, :], K, axis=1)
            t_feat = jnp.full((batch_size, K, 1), t, dtype=obs_batch.dtype)

            refresh_L = self.transition_refresh_L
            bprop_L = self.bprop_refresh

            def single_refresh(sprime, key_step):
                noise_pred = self.transition_net.dyn_policy(
                    dyn_params,
                    obs_exp,
                    act_exp,
                    sprime,
                    t_feat,
                )
                s0_hat = sprime * sqrt_recip - noise_pred * sqrt_recipm1
                s0_hat = jnp.clip(s0_hat, -1.0, 1.0)
                z = jax.random.normal(key_step, sprime.shape, dtype=sprime.dtype)
                sprime_next = sqrt_alpha_bar * s0_hat + sqrt_one_minus_alpha_bar * z
                return sprime_next, s0_hat

            if refresh_L <= 0:
                sprime_final, s0_hat_last = single_refresh(sprime_t, key)
            else:
                # Compute prefix and tail lengths as plain Python integers to keep
                # control flow outside of JAX tracing.
                prefix_L = max(refresh_L - bprop_L, 0)
                tail_L = min(refresh_L, bprop_L)

                sprime_mid = sprime_t
                key_mid = key

                def single_refresh_nograd(sprime, key_step):
                    sprime_next, _ = single_refresh(jax.lax.stop_gradient(sprime), key_step)
                    return sprime_next, None

                if prefix_L > 0:
                    keys_prefix = jax.random.split(key_mid, prefix_L)
                    sprime_mid, _ = jax.lax.scan(single_refresh_nograd, sprime_mid, keys_prefix)
                    key_mid = jax.random.split(key_mid, 1)[0]

                if tail_L > 0:
                    keys_tail = jax.random.split(key_mid, tail_L)
                    sprime_final, s0_hats = jax.lax.scan(single_refresh, sprime_mid, keys_tail)
                    s0_hat_last = s0_hats[-1]
                else:
                    sprime_final, s0_hat_last = single_refresh(sprime_mid, key_mid)

            r_hat = self.transition_net.reward(
                reward_params,
                obs_exp,
                act_exp,
                s0_hat_last,
            )

            r_mean = jnp.mean(r_hat, axis=1)
            q_vals = self.reward_scale * r_mean / (1.0 - self.gamma)

            return q_vals, sprime_final

        def critic_step_td(
            params,
            obs_batch: jax.Array,
            actions_x0: jax.Array,
            sprime_t: jax.Array,
            t: int,
            key: jax.Array,
        ) -> Tuple[jax.Array, jax.Array]:
            q_vals = critic_value_from_x0_td(params, obs_batch, actions_x0)
            return q_vals, sprime_t

        def transition_q_from_sprime(
            params,
            obs_batch: jax.Array,
            actions_x0: jax.Array,
            sprime_t: jax.Array,
            t: int,
        ) -> jax.Array:
            policy_params, log_alpha, q1_params, q2_params, transition_params = params
            dyn_params = transition_params.dyn_params
            reward_params = transition_params.reward_params

            B = self.transition_net.diffusion.beta_schedule()
            sqrt_recip = B.sqrt_recip_alphas_cumprod[t]
            sqrt_recipm1 = B.sqrt_recipm1_alphas_cumprod[t]

            batch_size = obs_batch.shape[0]
            K = self.state_population_size

            obs_exp = jnp.repeat(obs_batch[:, None, :], K, axis=1)
            act_exp = jnp.repeat(actions_x0[:, None, :], K, axis=1)
            t_feat = jnp.full((batch_size, K, 1), t, dtype=obs_batch.dtype)

            noise_pred = self.transition_net.dyn_policy(
                dyn_params,
                obs_exp,
                act_exp,
                sprime_t,
                t_feat,
            )

            s0_hat = sprime_t * sqrt_recip - noise_pred * sqrt_recipm1
            s0_hat = jnp.clip(s0_hat, -1.0, 1.0)

            r_hat = self.transition_net.reward(
                reward_params,
                obs_exp,
                act_exp,
                s0_hat,
            )

            r_mean = jnp.mean(r_hat, axis=1)
            q_vals = self.reward_scale * r_mean / (1.0 - self.gamma)
            return q_vals

        if np.isinf(particle_selection_lambda):
            def select_action_from_particles(acts: jax.Array, qs: jax.Array, key: jax.Array) -> jax.Array:
                q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
                act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
                return act
        else:
            lambda_sel = jnp.array(particle_selection_lambda, dtype=jnp.float32)

            def select_action_from_particles(acts: jax.Array, qs: jax.Array, key: jax.Array) -> jax.Array:
                logits = lambda_sel * qs
                logits = logits - jnp.max(logits, axis=0, keepdims=True)
                idx = jax.random.categorical(key, logits, axis=0)
                idx = idx[None, :]
                act = jnp.take_along_axis(acts, idx[..., None], axis=0).squeeze(axis=0)
                return act

        def sample_with_particles(
            key: jax.Array,
            log_alpha: jax.Array,
            single: bool,
            single_sampler,
        ) -> jax.Array:
            key_sample, key_select, noise_key = jax.random.split(key, 3)
            if self.agent.num_particles == 1:
                act, _ = single_sampler(key_sample)
            else:
                keys = jax.random.split(key_sample, self.agent.num_particles)
                acts, qs = jax.vmap(single_sampler)(keys)
                act = select_action_from_particles(acts, qs, key_select)

            act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.agent.noise_scale

            if single:
                return act[0]
            else:
                return act

        def sample_action_with_critic(
            key: jax.Array,
            params,
            obs: jax.Array,
            critic_value_fn,
        ) -> jax.Array:
            if self.agent.energy_mode and self.agent.mala_steps > 0:
                return stateless_get_action_mala(key, params, obs, critic_value_fn)
            elif self.tfg_recur_steps > 0 or self.tfg_lambda != 0.0:
                return stateless_get_action_tfg_recur(key, params, obs, critic_value_fn)
            else:
                return stateless_get_action_base(key, params, obs, critic_value_fn)

        @jax.jit
        def stateless_update(
            key: jax.Array, state: DPMDTrainState, data: Experience
        ) -> Tuple[DPMDTrainState, Metric]:
            obs, action, raw_reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state = state.opt_state
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std
            transition_params_state = state.transition_params
            next_eval_key, new_eval_key, new_q1_eval_key, new_q2_eval_key, log_alpha_key, diffusion_time_key, diffusion_noise_key = jax.random.split(
                key, 7)

            # Helper functions used in Q-based modes
            def get_min_q(s, a):
                q1 = self.agent.q(q1_params, s, a)
                q2 = self.agent.q(q2_params, s, a)
                q = aggregate_q(q1, q2)
                return q

            def get_min_taret_q(s, a):
                q1 = self.agent.q(target_q1_params, s, a)
                q2 = self.agent.q(target_q2_params, s, a)
                q = jnp.minimum(q1, q2)
                return q

            # Shared helpers for optimizer updates
            def param_update(optim, params, grads, opt_state):
                update, new_opt_state = optim.update(grads, opt_state)
                new_params = optax.apply_updates(params, update)
                return new_params, new_opt_state

            def delay_param_update(optim, params, grads, opt_state):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda p, o: param_update(optim, p, grads, o),
                    lambda p, o: (p, o),
                    params,
                    opt_state,
                )

            def delay_alpha_param_update(optim, params, opt_state):
                return jax.lax.cond(
                    step % self.delay_alpha_update == 0,
                    lambda p, o: param_update(optim, p, jax.grad(log_alpha_loss_fn)(p), o),
                    lambda p, o: (p, o),
                    params,
                    opt_state,
                )

            def delay_target_update(params, target_params, tau):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda tgt: optax.incremental_update(params, tgt, tau),
                    lambda tgt: tgt,
                    target_params,
                )

            # Q-based modes: 'q' and 'reward1step'
            if self.critic_type != "transition_reward":
                reward = raw_reward * self.reward_scale

                if self.td_guided_targets:
                    # Use the full guidance / MALA sampler for TD targets, as in
                    # the refactored version. This is more expensive but can
                    # produce stronger early improvements.
                    td_params = (policy_params, log_alpha, q1_params, q2_params, transition_params_state)
                    next_action = sample_action_with_critic(
                        next_eval_key,
                        td_params,
                        next_obs,
                        critic_value_from_x0_td,
                    )
                else:
                    # Cheaper original DP-MD behavior: use the base Diffv2Net
                    # get_action interface for TD target actions.
                    td_policy_params = (policy_params, log_alpha, q1_params, q2_params)
                    next_action = self.agent.get_action(next_eval_key, td_policy_params, next_obs)

                if self.critic_type == "q":
                    q1_target = self.agent.q(target_q1_params, next_obs, next_action)
                    q2_target = self.agent.q(target_q2_params, next_obs, next_action)
                    q_target = jnp.minimum(q1_target, q2_target)
                    q_backup = reward + (1 - done) * self.gamma * q_target
                else:
                    # reward1step: approximate Q using scaled immediate reward only
                    q_backup = reward / (1.0 - self.gamma)

                def q_loss_fn(q_params: hk.Params) -> jax.Array:
                    q = self.agent.q(q_params, obs, action)
                    q_loss = jnp.mean((q - q_backup) ** 2)
                    return q_loss, q

                (q1_loss, q1), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
                (q2_loss, q2), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)

                def policy_loss_fn(policy_params) -> jax.Array:
                    q_min = get_min_q(next_obs, next_action)
                    q_mean, q_std = q_min.mean(), q_min.std()
                    if self.fix_q_norm_bug:
                        norm_q = (q_min - running_mean) / running_std
                    else:
                        norm_q = q_min - running_mean / running_std
                    scaled_q = norm_q.clip(-3.0, 3.0) / jnp.exp(log_alpha)
                    base_q_weights = jnp.exp(scaled_q)
                    if self.use_reweighting:
                        loss_weights = base_q_weights
                    else:
                        loss_weights = jnp.ones_like(base_q_weights)

                    def denoiser(t, x):
                        return self.agent.policy(policy_params, next_obs, x, t)

                    t = jax.random.randint(
                        diffusion_time_key,
                        (next_obs.shape[0],),
                        0,
                        self.agent.num_timesteps,
                    )
                    loss = self.agent.diffusion.weighted_p_loss(
                        diffusion_noise_key,
                        loss_weights,
                        denoiser,
                        t,
                        jax.lax.stop_gradient(next_action),
                    )

                    return loss, (base_q_weights, scaled_q, q_mean, q_std)

                (total_loss, (q_weights, scaled_q, q_mean, q_std)), policy_grads = jax.value_and_grad(
                    policy_loss_fn, has_aux=True
                )(policy_params)

                # update alpha
                def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                    approx_entropy = 0.5 * self.agent.act_dim * jnp.log(
                        2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha)) ** 2
                    )
                    log_alpha_loss = -1 * log_alpha * (
                        -1 * jax.lax.stop_gradient(approx_entropy) + self.agent.target_entropy
                    )
                    return log_alpha_loss

                q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
                q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)
                policy_params, policy_opt_state = delay_param_update(
                    self.policy_optim, policy_params, policy_grads, policy_opt_state
                )
                log_alpha, log_alpha_opt_state = delay_alpha_param_update(
                    self.alpha_optim, log_alpha, log_alpha_opt_state
                )

                target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
                target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)
                target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)

                new_running_mean = running_mean + 0.001 * (q_mean - running_mean)
                new_running_std = running_std + 0.001 * (q_std - running_std)

                state = DPMDTrainState(
                    params=Diffv2Params(
                        q1_params,
                        q2_params,
                        target_q1_params,
                        target_q2_params,
                        policy_params,
                        target_policy_params,
                        log_alpha,
                    ),
                    opt_state=Diffv2OptStates(
                        q1=q1_opt_state,
                        q2=q2_opt_state,
                        policy=policy_opt_state,
                        log_alpha=log_alpha_opt_state,
                    ),
                    step=step + 1,
                    entropy=jnp.float32(0.0),
                    running_mean=new_running_mean,
                    running_std=new_running_std,
                    transition_params=state.transition_params,
                    transition_opt_state=state.transition_opt_state,
                )
                info = {
                    "q1_loss": q1_loss,
                    "q1_mean": jnp.mean(q1),
                    "q1_max": jnp.max(q1),
                    "q1_min": jnp.min(q1),
                    "q2_loss": q2_loss,
                    "policy_loss": total_loss,
                    "alpha": jnp.exp(log_alpha),
                    "q_weights_std": jnp.std(q_weights),
                    "q_weights_mean": jnp.mean(q_weights),
                    "q_weights_min": jnp.min(q_weights),
                    "q_weights_max": jnp.max(q_weights),
                    "scale_q_mean": jnp.mean(scaled_q),
                    "scale_q_std": jnp.std(scaled_q),
                    "running_q_mean": new_running_mean,
                    "running_q_std": new_running_std,
                    "entropy_approx": 0.5
                    * self.agent.act_dim
                    * jnp.log(2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha)) ** 2),
                }
                return state, info

            # Transition-reward critic mode: train dynamics diffusion and reward MLP, 
            # and use constant-weight diffusion loss for the policy (behavior cloning).
            assert self.critic_type == "transition_reward"

            transition_params = state.transition_params
            transition_opt_state = state.transition_opt_state

            dyn_params = transition_params.dyn_params
            reward_params = transition_params.reward_params
            dyn_opt_state = transition_opt_state.dyn
            reward_opt_state = transition_opt_state.reward

            # Split diffusion_noise_key further for dynamics and reward losses.
            diff_key_policy, diff_key_dyn = jax.random.split(diffusion_noise_key)
            dyn_time_key, dyn_noise_key, reward_key = jax.random.split(diff_key_dyn, 3)

            # Dynamics diffusion loss: conditional on (obs, action), predict eps for next_obs.
            t_dyn = jax.random.randint(
                dyn_time_key,
                (obs.shape[0],),
                0,
                self.agent.num_timesteps,
            )

            def dyn_model(t_vec: jax.Array, x_noisy: jax.Array) -> jax.Array:
                t_feat = t_vec[:, None].astype(jnp.float32)
                return self.transition_net.dyn_policy(dyn_params, obs, action, x_noisy, t_feat)

            dyn_loss = self.transition_net.diffusion.p_loss(
                dyn_noise_key,
                dyn_model,
                t_dyn,
                next_obs,
            )

            # Reward loss: MSE on raw environment rewards.
            def reward_loss_fn(r_params: hk.Params) -> jax.Array:
                r_hat = self.transition_net.reward(r_params, obs, action, next_obs)
                return jnp.mean((r_hat - raw_reward) ** 2)

            reward_loss, reward_grads = jax.value_and_grad(reward_loss_fn)(reward_params)
            reward_params, reward_opt_state = param_update(self.optim, reward_params, reward_grads, reward_opt_state)

            # Update dynamics params and opt_state.
            dyn_grads = jax.grad(lambda p: self.transition_net.diffusion.p_loss(dyn_noise_key, lambda t, x: self.transition_net.dyn_policy(p, obs, action, x, t[:, None].astype(jnp.float32)), t_dyn, next_obs))(dyn_params)
            dyn_params, dyn_opt_state = param_update(self.optim, dyn_params, dyn_grads, dyn_opt_state)

            new_transition_params = TransitionParams(dyn_params=dyn_params, reward_params=reward_params)
            new_transition_opt_state = TransitionOptStates(dyn=dyn_opt_state, reward=reward_opt_state)

            # Constant-weight diffusion loss for the policy on buffer actions (behavior cloning).
            def policy_loss_fn(policy_params) -> jax.Array:
                def denoiser(t, x):
                    return self.agent.policy(policy_params, obs, x, t)

                t = jax.random.randint(
                    diffusion_time_key,
                    (obs.shape[0],),
                    0,
                    self.agent.num_timesteps,
                )
                loss = self.agent.diffusion.p_loss(
                    diff_key_policy,
                    denoiser,
                    t,
                    jax.lax.stop_gradient(action),
                )
                return loss, ()

            (policy_loss, _), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)

            # Alpha loss identical to other modes.
            def log_alpha_loss_fn(log_alpha_val: jax.Array) -> jax.Array:
                approx_entropy = 0.5 * self.agent.act_dim * jnp.log(
                    2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha_val)) ** 2
                )
                log_alpha_loss = -1.0 * log_alpha_val * (
                    -1.0 * jax.lax.stop_gradient(approx_entropy) + self.agent.target_entropy
                )
                return log_alpha_loss

            policy_params, policy_opt_state = delay_param_update(
                self.policy_optim, policy_params, policy_grads, policy_opt_state
            )
            log_alpha, log_alpha_opt_state = delay_alpha_param_update(
                self.alpha_optim, log_alpha, log_alpha_opt_state
            )

            # Keep Q params and targets fixed; only update policy target.
            target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)

            new_running_mean = running_mean
            new_running_std = running_std

            state = DPMDTrainState(
                params=Diffv2Params(
                    q1_params,
                    q2_params,
                    target_q1_params,
                    target_q2_params,
                    policy_params,
                    target_policy_params,
                    log_alpha,
                ),
                opt_state=Diffv2OptStates(
                    q1=q1_opt_state,
                    q2=q2_opt_state,
                    policy=policy_opt_state,
                    log_alpha=log_alpha_opt_state,
                ),
                step=step + 1,
                entropy=jnp.float32(0.0),
                running_mean=new_running_mean,
                running_std=new_running_std,
                transition_params=new_transition_params,
                transition_opt_state=new_transition_opt_state,
            )
            info = {
                "dyn_loss": dyn_loss,
                "reward_loss": reward_loss,
                "policy_loss": policy_loss,
                "alpha": jnp.exp(log_alpha),
            }
            return state, info

        def stateless_get_action_tfg_recur(
            key: jax.Array,
            params,
            obs: jax.Array,
            critic_value_fn,
        ) -> jax.Array:
            policy_params, log_alpha, q1_params, q2_params, transition_params = params

            single = obs.ndim == 1
            if single:
                obs_batch = obs[None, :]
            else:
                obs_batch = obs

            shape = (*obs_batch.shape[:-1], self.agent.act_dim)
            B = self.agent.diffusion.beta_schedule()
            timesteps = self.agent.num_timesteps

            def model_fn(t, x):
                return self.agent.policy(policy_params, obs_batch, x, t)

            def q_mean_from_x(x_in, t_idx):
                noise_pred = model_fn(t_idx, x_in)
                x0_hat = (
                    x_in * B.sqrt_recip_alphas_cumprod[t_idx]
                    - noise_pred * B.sqrt_recipm1_alphas_cumprod[t_idx]
                )
                x0_hat = jnp.clip(x0_hat, -1.0, 1.0)
                q = critic_value_fn(params, obs_batch, x0_hat)
                return jnp.mean(q)

            def grad_guidance(x_in, t_idx):
                # Use Q-gradient guidance only if lambda_for_step(t_idx) > 0; otherwise no guidance
                def guided(x):
                    return jax.grad(lambda xx: q_mean_from_x(xx, t_idx))(x)

                def unguided(x):
                    return jnp.zeros_like(x)

                lam = lambda_for_step(t_idx)
                return jax.lax.cond(lam > 0.0, guided, unguided, x_in)

            def tfg_sample(single_key: jax.Array):
                x_key, sprime_key, loop_key = jax.random.split(single_key, 3)
                x = 0.5 * jax.random.normal(x_key, shape)
                sprime = init_sprime(obs_batch, sprime_key)
                t_seq = jnp.arange(timesteps)[::-1]

                def body_fn(carry, t):
                    x_t, sprime_t, key_t = carry

                    def q_mean_from_x_given_sprime(x_in: jax.Array) -> jax.Array:
                        noise_pred = model_fn(t, x_in)
                        x0_hat = (
                            x_in * B.sqrt_recip_alphas_cumprod[t]
                            - noise_pred * B.sqrt_recipm1_alphas_cumprod[t]
                        )
                        x0_hat = jnp.clip(x0_hat, -1.0, 1.0)
                        q_vals, _ = critic_step_env(
                            params,
                            obs_batch,
                            x0_hat,
                            sprime_t,
                            t,
                            key_t,
                        )
                        return jnp.mean(q_vals)

                    def grad_guidance_local(x_in: jax.Array) -> jax.Array:
                        def guided(x):
                            return jax.grad(q_mean_from_x_given_sprime)(x)

                        def unguided(x):
                            return jnp.zeros_like(x)

                        lam = lambda_for_step(t)
                        return jax.lax.cond(lam > 0.0, guided, unguided, x_in)

                    def recur_step(carry_recur, _):
                        x_cur, key_cur = carry_recur
                        key_cur, key_down, key_up = jax.random.split(key_cur, 3)
                        noise_down = jax.random.normal(key_down, x_cur.shape)
                        noise_up = jax.random.normal(key_up, x_cur.shape)

                        noise_pred = model_fn(t, x_cur)
                        grad_q = grad_guidance_local(x_cur)
                        sigma_t = B.sqrt_one_minus_alphas_cumprod[t]
                        lambda_t = lambda_for_step(t)
                        eps_guided = noise_pred - lambda_t * sigma_t * grad_q
                        model_mean, model_log_variance = self.agent.diffusion.p_mean_variance(
                            t, x_cur, eps_guided
                        )
                        x_tm1 = model_mean + (t > 0) * jnp.exp(0.5 * model_log_variance) * noise_down

                        alpha_t = B.alphas[t]
                        sqrt_alpha_t = jnp.sqrt(alpha_t)
                        sqrt_one_minus_alpha_t = jnp.sqrt(1.0 - alpha_t)
                        x_back = sqrt_alpha_t * x_tm1 + sqrt_one_minus_alpha_t * noise_up

                        return (x_back, key_cur), None

                    if self.tfg_recur_steps > 0:
                        (x_t, key_t), _ = jax.lax.scan(
                            recur_step,
                            (x_t, key_t),
                            jnp.arange(self.tfg_recur_steps),
                        )

                    key_t, key_down = jax.random.split(key_t)
                    noise_down = jax.random.normal(key_down, x_t.shape)
                    noise_pred = model_fn(t, x_t)
                    grad_q = grad_guidance_local(x_t)
                    sigma_t = B.sqrt_one_minus_alphas_cumprod[t]
                    lambda_t = lambda_for_step(t)
                    eps_guided = noise_pred - lambda_t * sigma_t * grad_q
                    model_mean, model_log_variance = self.agent.diffusion.p_mean_variance(
                        t, x_t, eps_guided
                    )
                    x_next = model_mean + (t > 0) * jnp.exp(0.5 * model_log_variance) * noise_down
                    return (x_next, sprime_t, key_t), None

                (final_carry, _) = jax.lax.scan(
                    body_fn,
                    (x, sprime, loop_key),
                    t_seq,
                )
                x_final, sprime_final, key_final = final_carry
                act_final = jnp.clip(x_final, -1.0, 1.0)
                q, _ = critic_step_env(
                    params,
                    obs_batch,
                    act_final,
                    sprime_final,
                    0,
                    key_final,
                )
                return act_final, q

            def single_sampler(single_key: jax.Array):
                return tfg_sample(single_key)

            return sample_with_particles(key, log_alpha, single, single_sampler)

        def stateless_get_action_mala(
            key: jax.Array,
            params,
            obs: jax.Array,
            critic_value_fn,
        ) -> jax.Array:
            policy_params, log_alpha, q1_params, q2_params, transition_params = params

            single = obs.ndim == 1
            if single:
                obs_batch = obs[None, :]
            else:
                obs_batch = obs

            shape = (*obs_batch.shape[:-1], self.agent.act_dim)
            B = self.agent.diffusion.beta_schedule()
            timesteps = self.agent.num_timesteps

            def energy_model(t, x):
                E = self.agent.energy_fn(policy_params, obs_batch, x, t)
                return E

            def energy_total(t, x, sprime_t):
                def only_model(x_in):
                    return energy_model(t, x_in)

                def with_q(x_in):
                    E_mod = energy_model(t, x_in)
                    noise_pred = self.agent.policy(policy_params, obs_batch, x_in, t)
                    x0_hat = (
                        x_in * B.sqrt_recip_alphas_cumprod[t]
                        - noise_pred * B.sqrt_recipm1_alphas_cumprod[t]
                    )
                    x0_hat_clipped = jnp.clip(x0_hat, -1.0, 1.0)
                    lambda_t = lambda_for_step(t)
                    def q_from_backend():
                        if self.critic_type == "transition_reward":
                            return transition_q_from_sprime(
                                params,
                                obs_batch,
                                x0_hat_clipped,
                                sprime_t,
                                t,
                            )
                        else:
                            return critic_value_fn(params, obs_batch, x0_hat_clipped)
                    q_vals = q_from_backend()
                    return E_mod - lambda_t * q_vals

                lambda_t = lambda_for_step(t)
                return jax.lax.cond(lambda_t > 0.0, with_q, only_model, x)

            def mala_chain(single_key: jax.Array):
                key_x, sprime_key, loop_key = jax.random.split(single_key, 3)
                x0 = jax.random.normal(key_x, shape)
                sprime0 = init_sprime(obs_batch, sprime_key)

                def level_body(i, carry):
                    x_curr, sprime_curr, k = carry
                    t = timesteps - 1 - i

                    eta_base_t = jnp.maximum(B.betas[t], jnp.float32(1e-8))
                    log_eta_scale0 = jnp.float32(0.0)

                    def mala_body(_, state):
                        x_in, sprime_in, k_in, log_eta_scale = state

                        # Compute energy gradient wrt x using current sprime_in
                        E_x_grad, vjp_x = jax.vjp(
                            lambda xx: energy_total(t, xx, sprime_in), x_in
                        )
                        grad_E_x = vjp_x(jnp.ones_like(E_x_grad))[0]

                        k_in, noise_key, u_key, key_qx, key_qprop = jax.random.split(k_in, 5)
                        eta_k = jnp.clip(
                            jnp.exp(log_eta_scale) * eta_base_t,
                            jnp.float32(1e-8),
                            jnp.float32(0.5),
                        )
                        z = jax.random.normal(noise_key, x_in.shape)
                        sd = jnp.sqrt(jnp.float32(2.0) * eta_k)
                        x_prop = x_in - eta_k * grad_E_x + sd * z

                        # Energy model values for current and proposal
                        E_mod_x = energy_model(t, x_in)
                        E_mod_prop = energy_model(t, x_prop)

                        # Denoised actions for critic
                        noise_pred_x = self.agent.policy(
                            policy_params, obs_batch, x_in, t
                        )
                        x0_hat_x = (
                            x_in * B.sqrt_recip_alphas_cumprod[t]
                            - noise_pred_x * B.sqrt_recipm1_alphas_cumprod[t]
                        )
                        x0_hat_x = jnp.clip(x0_hat_x, -1.0, 1.0)

                        noise_pred_prop = self.agent.policy(
                            policy_params, obs_batch, x_prop, t
                        )
                        x0_hat_prop = (
                            x_prop * B.sqrt_recip_alphas_cumprod[t]
                            - noise_pred_prop * B.sqrt_recipm1_alphas_cumprod[t]
                        )
                        x0_hat_prop = jnp.clip(x0_hat_prop, -1.0, 1.0)

                        lambda_t = lambda_for_step(t)

                        def q_and_sprime_x():
                            if self.critic_type == "transition_reward":
                                q_x, sprime_x = critic_step_env(
                                    params,
                                    obs_batch,
                                    x0_hat_x,
                                    sprime_in,
                                    t,
                                    key_qx,
                                )
                            else:
                                q_x = critic_value_fn(
                                    params,
                                    obs_batch,
                                    x0_hat_x,
                                )
                                sprime_x = sprime_in
                            return q_x, sprime_x

                        def q_and_sprime_prop():
                            if self.critic_type == "transition_reward":
                                q_prop, sprime_prop = critic_step_env(
                                    params,
                                    obs_batch,
                                    x0_hat_prop,
                                    sprime_in,
                                    t,
                                    key_qprop,
                                )
                            else:
                                q_prop = critic_value_fn(
                                    params,
                                    obs_batch,
                                    x0_hat_prop,
                                )
                                sprime_prop = sprime_in
                            return q_prop, sprime_prop

                        q_x, sprime_x = q_and_sprime_x()
                        q_prop, sprime_prop = q_and_sprime_prop()

                        E_x_acc = E_mod_x - lambda_t * q_x
                        E_x_prop_acc = E_mod_prop - lambda_t * q_prop

                        # Proposal densities as before
                        # Use grad_E_x as gradient at x_in and reuse it for mean_f
                        # For mean_r, we need gradient at x_prop; reuse E_x_grad's structure
                        # by recomputing gradient at x_prop through energy_total
                        E_x_prop_grad, vjp_x_prop = jax.vjp(
                            lambda xx: energy_total(t, xx, sprime_in), x_prop
                        )
                        grad_E_x_prop = vjp_x_prop(jnp.ones_like(E_x_prop_grad))[0]

                        mean_f = x_in - eta_k * grad_E_x
                        mean_r = x_prop - eta_k * grad_E_x_prop

                        def log_gauss(xv, meanv):
                            diff = xv - meanv
                            return -jnp.sum(diff * diff, axis=-1) / (jnp.float32(4.0) * eta_k)

                        log_q_prop_given_x = log_gauss(x_prop, mean_f)
                        log_q_x_given_prop = log_gauss(x_in, mean_r)

                        log_alpha = (-E_x_prop_acc + E_x_acc) + (
                            log_q_x_given_prop - log_q_prop_given_x
                        )
                        u = jax.random.uniform(u_key, E_x_acc.shape)
                        accept = jnp.log(u) < jnp.minimum(jnp.float32(0.0), log_alpha)

                        x_new = jnp.where(accept[..., None], x_prop, x_in)
                        sprime_new = jnp.where(
                            accept[..., None, None], sprime_prop, sprime_x
                        )

                        acc_rate = jnp.mean(accept.astype(jnp.float32))
                        target = jnp.float32(0.574)
                        adapt_rate = jnp.float32(0.05)
                        log_eta_scale = log_eta_scale + adapt_rate * (acc_rate - target)

                        return x_new, sprime_new, k_in, log_eta_scale

                    x_curr, sprime_curr, k, _ = jax.lax.fori_loop(
                        0,
                        self.agent.mala_steps,
                        mala_body,
                        (x_curr, sprime_curr, k, log_eta_scale0),
                    )

                    noise_pred = self.agent.policy(policy_params, obs_batch, x_curr, t)
                    model_mean, model_log_variance = self.agent.diffusion.p_mean_variance(
                        t, x_curr, noise_pred
                    )
                    k, z_key = jax.random.split(k)
                    z = jax.random.normal(z_key, x_curr.shape)
                    x_next = model_mean + (t > 0) * jnp.exp(0.5 * model_log_variance) * z
                    return x_next, sprime_curr, k

                x_final, sprime_final, _ = jax.lax.fori_loop(
                    0, timesteps, level_body, (x0, sprime0, loop_key)
                )
                act_final = jnp.clip(x_final, -1.0, 1.0)
                q, _ = critic_step_env(
                    params,
                    obs_batch,
                    act_final,
                    sprime_final,
                    0,
                    loop_key,
                )
                return act_final, q

            def single_sampler(single_key: jax.Array):
                return mala_chain(single_key)

            return sample_with_particles(key, log_alpha, single, single_sampler)

        def stateless_get_action_base(
            key: jax.Array,
            params,
            obs: jax.Array,
            critic_value_fn,
        ) -> jax.Array:
            policy_params, log_alpha, q1_params, q2_params, transition_params = params

            single = obs.ndim == 1
            if single:
                obs_batch = obs[None, :]
            else:
                obs_batch = obs

            shape = (*obs_batch.shape[:-1], self.agent.act_dim)

            def model_fn(t, x):
                return self.agent.policy(policy_params, obs_batch, x, t)

            def energy_model_fn(t, x):
                return self.agent.energy_fn(policy_params, obs_batch, x, t)

            def base_sample(single_key: jax.Array):
                if self.agent.energy_mode:
                    act = self.agent.mala_sample(single_key, model_fn, energy_model_fn, shape)
                else:
                    act = self.agent.diffusion.p_sample(single_key, model_fn, shape)
                act = jnp.clip(act, -1.0, 1.0)

                if self.critic_type == "transition_reward":
                    # Use a fresh sprime population per env rollout for scoring.
                    key_sprime = jax.random.PRNGKey(0)
                    sprime = init_sprime(obs_batch, key_sprime)
                    q = transition_q_from_sprime(
                        params,
                        obs_batch,
                        act,
                        sprime,
                        0,
                    )
                else:
                    q = critic_value_fn(params, obs_batch, act)

                return act, q

            def single_sampler(single_key: jax.Array):
                return base_sample(single_key)

            return sample_with_particles(key, log_alpha, single, single_sampler)

        def stateless_get_action_env(
            key: jax.Array,
            params,
            obs: jax.Array,
        ) -> jax.Array:
            if self.critic_type != "transition_reward":
                return sample_action_with_critic(key, params, obs, critic_value_from_x0_env)

            policy_params, log_alpha, q1_params, q2_params, transition_params = params

            def critic_value_env_transition(params_inner, obs_batch, actions_x0):
                # For env-time scoring, we use t=0 and a fresh sprime population per observation.
                key_sprime = jax.random.PRNGKey(0)
                sprime = init_sprime(obs_batch, key_sprime)
                q_vals = transition_q_from_sprime(
                    params_inner,
                    obs_batch,
                    actions_x0,
                    sprime,
                    0,
                )
                return q_vals

            return sample_action_with_critic(key, params, obs, critic_value_env_transition)

        self._implement_common_behavior(stateless_update, stateless_get_action_env, self.agent.get_deterministic_action)

    def get_policy_params(self):
        # Used by env-time samplers which expect the extended 5-tuple including
        # transition_params.
        return (
            self.state.params.policy,
            self.state.params.log_alpha,
            self.state.params.q1,
            self.state.params.q2,
            self.state.transition_params,
        )

    def get_policy_params_to_save(self):
        # Used for env-time sampling and saving, where samplers expect
        # the extended 5-tuple including transition_params.
        return (
            self.state.params.target_poicy,
            self.state.params.log_alpha,
            self.state.params.q1,
            self.state.params.q2,
            self.state.transition_params,
        )

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)