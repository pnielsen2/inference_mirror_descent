from typing import NamedTuple, Tuple

import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle

from relax.algorithm.base import Algorithm
from relax.network.dacer import DACERNet, DACERParams
from relax.network.diffv2 import Diffv2Net, Diffv2Params
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

        self.state = Diffv2TrainState(
            params=params,
            opt_state=Diffv2OptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                # policy=self.optim.init(params.policy),
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
            ),
            step=jnp.int32(0),
            entropy=jnp.float32(0.0),
            running_mean=jnp.float32(0.0),
            running_std=jnp.float32(1.0)
        )
        self.use_ema = use_ema
        self.use_reweighting = use_reweighting
        self.q_critic_agg = q_critic_agg
        self.fix_q_norm_bug = fix_q_norm_bug
        self.tfg_lambda = tfg_lambda
        self.tfg_lambda_schedule = tfg_lambda_schedule
        self.tfg_recur_steps = tfg_recur_steps
        self.particle_selection_lambda = particle_selection_lambda
        if critic_type not in ("q", "reward1step"):
            raise ValueError(f"Invalid critic_type: {critic_type}. Expected 'q' or 'reward1step'.")
        self.critic_type = critic_type

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

        def sample_action_with_agg(
            key: jax.Array,
            params,
            obs: jax.Array,
            aggregate_q_fn,
        ) -> jax.Array:
            if self.agent.energy_mode and self.agent.mala_steps > 0:
                return stateless_get_action_mala(key, params, obs, aggregate_q_fn)
            elif self.tfg_recur_steps > 0 or self.tfg_lambda != 0.0:
                return stateless_get_action_tfg_recur(key, params, obs, aggregate_q_fn)
            else:
                return stateless_get_action_base(key, params, obs, aggregate_q_fn)

        @jax.jit
        def stateless_update(
            key: jax.Array, state: Diffv2TrainState, data: Experience
        ) -> Tuple[Diffv2OptStates, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state = state.opt_state
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std
            next_eval_key, new_eval_key, new_q1_eval_key, new_q2_eval_key, log_alpha_key, diffusion_time_key, diffusion_noise_key = jax.random.split(
                key, 7)

            reward *= self.reward_scale

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

            td_params = (policy_params, log_alpha, q1_params, q2_params)
            next_action = sample_action_with_agg(next_eval_key, td_params, next_obs, hard_min_q)

            if self.critic_type == "q":
                q1_target = self.agent.q(target_q1_params, next_obs, next_action)
                q2_target = self.agent.q(target_q2_params, next_obs, next_action)
                q_target = jnp.minimum(q1_target, q2_target)  # - jnp.exp(log_alpha) * next_logp
                q_backup = reward + (1 - done) * self.gamma * q_target
            else:
                q_backup = reward / (1.0 - self.gamma)

            def q_loss_fn(q_params: hk.Params) -> jax.Array:
                q = self.agent.q(q_params, obs, action)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss, q

            (q1_loss, q1), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params)
            (q2_loss, q2), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params)
            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)


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

                # Always log the Q-based weights (base_q_weights), even if
                # constant-weight training is enabled, so that statistics
                # reflect the reweighting that would be applied.
                return loss, (base_q_weights, scaled_q, q_mean, q_std)

            (total_loss, (q_weights, scaled_q, q_mean, q_std)), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)

            # update alpha
            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                approx_entropy = 0.5 * self.agent.act_dim * jnp.log( 2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha)) ** 2)
                log_alpha_loss = -1 * log_alpha * (-1 * jax.lax.stop_gradient(approx_entropy) + self.agent.target_entropy)
                return log_alpha_loss

            # update networks
            def param_update(optim, params, grads, opt_state):
                update, new_opt_state = optim.update(grads, opt_state)
                new_params = optax.apply_updates(params, update)
                return new_params, new_opt_state

            def delay_param_update(optim, params, grads, opt_state):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda params, opt_state: param_update(optim, params, grads, opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            def delay_alpha_param_update(optim, params, opt_state):
                return jax.lax.cond(
                    step % self.delay_alpha_update == 0,
                    lambda params, opt_state: param_update(optim, params, jax.grad(log_alpha_loss_fn)(params), opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            def delay_target_update(params, target_params, tau):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda target_params: optax.incremental_update(params, target_params, tau),
                    lambda target_params: target_params,
                    target_params
                )

            q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)
            policy_params, policy_opt_state = delay_param_update(self.policy_optim, policy_params, policy_grads, policy_opt_state)
            log_alpha, log_alpha_opt_state = delay_alpha_param_update(self.alpha_optim, log_alpha, log_alpha_opt_state)

            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)
            target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)

            new_running_mean = running_mean + 0.001 * (q_mean - running_mean)
            new_running_std = running_std + 0.001 * (q_std - running_std)

            state = Diffv2TrainState(
                params=Diffv2Params(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha),
                opt_state=Diffv2OptStates(q1=q1_opt_state, q2=q2_opt_state, policy=policy_opt_state, log_alpha=log_alpha_opt_state),
                step=step + 1,
                entropy=jnp.float32(0.0),
                running_mean=new_running_mean,
                running_std=new_running_std
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
                "entropy_approx": 0.5 * self.agent.act_dim * jnp.log( 2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha)) ** 2),
            }
            return state, info

        def stateless_get_action_tfg_recur(
            key: jax.Array,
            params,
            obs: jax.Array,
            aggregate_q_fn,
        ) -> jax.Array:
            policy_params, log_alpha, q1_params, q2_params = params

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
                q1 = self.agent.q(q1_params, obs_batch, x0_hat)
                q2 = self.agent.q(q2_params, obs_batch, x0_hat)
                q = aggregate_q_fn(q1, q2)
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
                x_key, loop_key = jax.random.split(single_key)
                x = 0.5 * jax.random.normal(x_key, shape)
                t_seq = jnp.arange(timesteps)[::-1]

                def body_fn(carry, t):
                    x_t, key_t = carry

                    def recur_step(carry_recur, _):
                        x_cur, key_cur = carry_recur
                        key_cur, key_down, key_up = jax.random.split(key_cur, 3)
                        noise_down = jax.random.normal(key_down, x_cur.shape)
                        noise_up = jax.random.normal(key_up, x_cur.shape)

                        noise_pred = model_fn(t, x_cur)
                        grad_q = grad_guidance(x_cur, t)
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
                    grad_q = grad_guidance(x_t, t)
                    sigma_t = B.sqrt_one_minus_alphas_cumprod[t]
                    lambda_t = lambda_for_step(t)
                    eps_guided = noise_pred - lambda_t * sigma_t * grad_q
                    model_mean, model_log_variance = self.agent.diffusion.p_mean_variance(
                        t, x_t, eps_guided
                    )
                    x_next = model_mean + (t > 0) * jnp.exp(0.5 * model_log_variance) * noise_down
                    return (x_next, key_t), None

                (final_carry, _) = jax.lax.scan(
                    body_fn,
                    (x, loop_key),
                    t_seq,
                )
                x_final, _ = final_carry
                act_final = jnp.clip(x_final, -1.0, 1.0)
                q1 = self.agent.q(q1_params, obs_batch, act_final)
                q2 = self.agent.q(q2_params, obs_batch, act_final)
                q = aggregate_q_fn(q1, q2)
                return act_final, q

            def single_sampler(single_key: jax.Array):
                return tfg_sample(single_key)

            return sample_with_particles(key, log_alpha, single, single_sampler)

        def stateless_get_action_mala(
            key: jax.Array,
            params,
            obs: jax.Array,
            aggregate_q_fn,
        ) -> jax.Array:
            policy_params, log_alpha, q1_params, q2_params = params

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

            def q_min_from_x0(x0_in):
                q1 = self.agent.q(q1_params, obs_batch, x0_in)
                q2 = self.agent.q(q2_params, obs_batch, x0_in)
                return aggregate_q_fn(q1, q2)

            def energy_total(t, x):
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
                    q_min = q_min_from_x0(x0_hat_clipped)
                    lambda_t = lambda_for_step(t)
                    return E_mod - lambda_t * q_min

                lambda_t = lambda_for_step(t)
                return jax.lax.cond(lambda_t > 0.0, with_q, only_model, x)

            def mala_chain(single_key: jax.Array):
                key_x, loop_key = jax.random.split(single_key)
                x0 = jax.random.normal(key_x, shape)

                def level_body(i, carry):
                    x_curr, k = carry
                    t = timesteps - 1 - i

                    eta_base_t = jnp.maximum(B.betas[t], jnp.float32(1e-8))
                    log_eta_scale0 = jnp.float32(0.0)

                    def mala_body(_, state):
                        x_in, k_in, log_eta_scale = state

                        # Compute energy and its gradient in a single forward/backward pass
                        E_x, vjp_x = jax.vjp(lambda xx: energy_total(t, xx), x_in)
                        grad_E_x = vjp_x(jnp.ones_like(E_x))[0]

                        k_in, noise_key, u_key = jax.random.split(k_in, 3)
                        eta_k = jnp.clip(
                            jnp.exp(log_eta_scale) * eta_base_t,
                            jnp.float32(1e-8),
                            jnp.float32(0.5),
                        )
                        z = jax.random.normal(noise_key, x_in.shape)
                        sd = jnp.sqrt(jnp.float32(2.0) * eta_k)
                        x_prop = x_in - eta_k * grad_E_x + sd * z

                        # Same for the proposal state
                        E_x_prop, vjp_x_prop = jax.vjp(lambda xx: energy_total(t, xx), x_prop)
                        grad_E_x_prop = vjp_x_prop(jnp.ones_like(E_x_prop))[0]

                        mean_f = x_in - eta_k * grad_E_x
                        mean_r = x_prop - eta_k * grad_E_x_prop

                        def log_gauss(xv, meanv):
                            diff = xv - meanv
                            return -jnp.sum(diff * diff, axis=-1) / (jnp.float32(4.0) * eta_k)

                        log_q_prop_given_x = log_gauss(x_prop, mean_f)
                        log_q_x_given_prop = log_gauss(x_in, mean_r)

                        log_alpha = (-E_x_prop + E_x) + (log_q_x_given_prop - log_q_prop_given_x)
                        u = jax.random.uniform(u_key, E_x.shape)
                        accept = jnp.log(u) < jnp.minimum(jnp.float32(0.0), log_alpha)

                        x_new = jnp.where(accept[..., None], x_prop, x_in)

                        acc_rate = jnp.mean(accept.astype(jnp.float32))
                        target = jnp.float32(0.574)
                        adapt_rate = jnp.float32(0.05)
                        log_eta_scale = log_eta_scale + adapt_rate * (acc_rate - target)

                        return x_new, k_in, log_eta_scale

                    x_curr, k, _ = jax.lax.fori_loop(
                        0,
                        self.agent.mala_steps,
                        mala_body,
                        (x_curr, k, log_eta_scale0),
                    )

                    noise_pred = self.agent.policy(policy_params, obs_batch, x_curr, t)
                    model_mean, model_log_variance = self.agent.diffusion.p_mean_variance(
                        t, x_curr, noise_pred
                    )
                    k, z_key = jax.random.split(k)
                    z = jax.random.normal(z_key, x_curr.shape)
                    x_next = model_mean + (t > 0) * jnp.exp(0.5 * model_log_variance) * z
                    return x_next, k

                x_final, _ = jax.lax.fori_loop(0, timesteps, level_body, (x0, loop_key))
                act_final = jnp.clip(x_final, -1.0, 1.0)
                q1 = self.agent.q(q1_params, obs_batch, act_final)
                q2 = self.agent.q(q2_params, obs_batch, act_final)
                q = aggregate_q_fn(q1, q2)
                return act_final, q

            def single_sampler(single_key: jax.Array):
                return mala_chain(single_key)

            return sample_with_particles(key, log_alpha, single, single_sampler)

        def stateless_get_action_base(
            key: jax.Array,
            params,
            obs: jax.Array,
            aggregate_q_fn,
        ) -> jax.Array:
            policy_params, log_alpha, q1_params, q2_params = params

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
                q1 = self.agent.q(q1_params, obs_batch, act)
                q2 = self.agent.q(q2_params, obs_batch, act)
                q = aggregate_q_fn(q1, q2)
                return act, q

            def single_sampler(single_key: jax.Array):
                return base_sample(single_key)

            return sample_with_particles(key, log_alpha, single, single_sampler)

        def stateless_get_action_env(
            key: jax.Array,
            params,
            obs: jax.Array,
        ) -> jax.Array:
            return sample_action_with_agg(key, params, obs, aggregate_q)

        self._implement_common_behavior(stateless_update, stateless_get_action_env, self.agent.get_deterministic_action)

    def get_policy_params(self):
        return (self.state.params.policy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2 )

    def get_policy_params_to_save(self):
        return (self.state.params.target_poicy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)