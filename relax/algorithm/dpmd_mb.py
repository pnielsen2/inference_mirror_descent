from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import haiku as hk

from relax.algorithm.base import Algorithm
from relax.network.model_based import ModelBasedNet, ModelBasedParams
from relax.utils.experience import Experience
from relax.utils.typing_utils import Metric


class ModelBasedOptStates(NamedTuple):
    policy: optax.OptState
    dynamics: optax.OptState
    reward: optax.OptState
    value: optax.OptState
    log_alpha: optax.OptState


class ModelBasedTrainState(NamedTuple):
    params: ModelBasedParams
    opt_state: ModelBasedOptStates
    step: int
    running_mean: float
    running_std: float


class DPMDMB(Algorithm):

    def __init__(
        self,
        agent: ModelBasedNet,
        params: ModelBasedParams,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        alpha_lr: float = 3e-2,
        lr_schedule_end: float = 5e-5,
        tau: float = 0.005,
        delay_alpha_update: int = 250,
        delay_update: int = 2,
        reward_scale: float = 0.2,
        num_mc_samples: int = 8,
        lr_policy=None,
        lr_dyn=None,
        lr_reward=None,
        lr_value=None,
        lr_schedule_steps: int = int(5e4),
        lr_schedule_begin: int = int(2.5e4),
    ):
        self.agent = agent
        self.gamma = gamma
        self.tau = tau
        self.delay_alpha_update = delay_alpha_update
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.num_mc_samples = num_mc_samples

        if lr_policy is None:
            lr_policy_init = lr
        else:
            lr_policy_init = float(lr_policy)

        lr_schedule = optax.schedules.linear_schedule(
            init_value=lr_policy_init,
            end_value=lr_schedule_end,
            transition_steps=int(lr_schedule_steps),
            transition_begin=int(lr_schedule_begin),
        )

        dyn_lr = lr if lr_dyn is None else float(lr_dyn)
        rew_lr = lr if lr_reward is None else float(lr_reward)
        value_lr = lr if lr_value is None else float(lr_value)

        self.policy_optim = optax.adam(learning_rate=lr_schedule)
        self.dyn_optim = optax.adam(dyn_lr)
        self.rew_optim = optax.adam(rew_lr)
        self.value_optim = optax.adam(value_lr)
        self.alpha_optim = optax.adam(alpha_lr)

        self.state = ModelBasedTrainState(
            params=params,
            opt_state=ModelBasedOptStates(
                policy=self.policy_optim.init(params.policy),
                dynamics=self.dyn_optim.init(params.dynamics),
                reward=self.rew_optim.init(params.reward),
                value=self.value_optim.init(params.value),
                log_alpha=self.alpha_optim.init(params.log_alpha),
            ),
            step=jnp.int32(0),
            running_mean=jnp.float32(0.0),
            running_std=jnp.float32(1.0),
        )
        def q_model(
            dyn_params: hk.Params,
            reward_params: hk.Params,
            value_targ_params: hk.Params,
            key: jax.Array,
            s: jax.Array,
            a: jax.Array,
        ) -> jax.Array:
            batch_size = s.shape[0]

            def single_q(s_i: jax.Array, a_i: jax.Array, key_i: jax.Array) -> jax.Array:
                def dyn_model(t, x_t):
                    return self.agent.dynamics(dyn_params, s_i[None, ...], a_i[None, ...], x_t, t)

                keys = jax.random.split(key_i, self.num_mc_samples)

                def sample_next_state(key_s: jax.Array) -> jax.Array:
                    return self.agent.dyn_diffusion.p_sample(
                        key_s,
                        dyn_model,
                        (1, self.agent.obs_dim),
                    )[0]

                s_next = jax.vmap(sample_next_state)(keys)
                s_rep = jnp.repeat(s_i[None, :], self.num_mc_samples, axis=0)
                a_rep = jnp.repeat(a_i[None, :], self.num_mc_samples, axis=0)
                r_hat = self.agent.reward(reward_params, s_rep, a_rep, s_next)
                v_hat = self.agent.value(value_targ_params, s_next)
                returns = r_hat + self.gamma * v_hat
                return jnp.mean(returns)

            keys_sa = jax.random.split(key, batch_size)
            return jax.vmap(single_q)(s, a, keys_sa)


        @jax.jit
        def stateless_update(
            key: jax.Array,
            state: ModelBasedTrainState,
            data: Experience,
        ) -> Tuple[ModelBasedTrainState, Metric]:
            obs, action, reward, next_obs, done = (
                data.obs,
                data.action,
                data.reward,
                data.next_obs,
                data.done,
            )

            (
                policy_params,
                target_policy_params,
                dyn_params,
                reward_params,
                value_params,
                target_value_params,
                log_alpha,
            ) = state.params

            (
                policy_opt_state,
                dyn_opt_state,
                rew_opt_state,
                value_opt_state,
                log_alpha_opt_state,
            ) = state.opt_state

            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std

            model_key, policy_key, q_key = jax.random.split(key, 3)

            reward_scaled = reward * self.reward_scale

            def dynamics_loss_fn(dyn_p: hk.Params) -> jax.Array:
                def dyn_model(t, x_t):
                    return self.agent.dynamics(dyn_p, obs, action, x_t, t)

                t_key, noise_key = jax.random.split(model_key)
                t = jax.random.randint(t_key, (obs.shape[0],), 0, self.agent.num_timesteps)
                return self.agent.dyn_diffusion.p_loss(noise_key, dyn_model, t, next_obs)

            dyn_loss, dyn_grads = jax.value_and_grad(dynamics_loss_fn)(dyn_params)

            def reward_loss_fn(rew_p: hk.Params) -> jax.Array:
                r_pred = self.agent.reward(rew_p, obs, action, next_obs)
                return jnp.mean((r_pred - reward_scaled) ** 2)

            rew_loss, rew_grads = jax.value_and_grad(reward_loss_fn)(reward_params)

            def value_loss_fn(val_p: hk.Params) -> jax.Array:
                v = self.agent.value(val_p, obs)
                v_next = self.agent.value(target_value_params, next_obs)
                v_target = reward_scaled + (1.0 - done) * self.gamma * v_next
                return jnp.mean((v - jax.lax.stop_gradient(v_target)) ** 2)

            val_loss, val_grads = jax.value_and_grad(value_loss_fn)(value_params)

            def policy_loss_fn(policy_p: hk.Params) -> jax.Array:
                def denoiser(t, x):
                    return self.agent.policy(policy_p, obs, x, t)

                q_vals = jax.lax.stop_gradient(
                    q_model(dyn_params, reward_params, target_value_params, q_key, obs, action)
                )
                q_mean = q_vals.mean()
                q_std = q_vals.std()

                norm_q = (q_vals - running_mean) / (running_std + 1e-6)
                scaled_q = norm_q.clip(-3.0, 3.0) / jnp.exp(log_alpha)
                q_weights = jnp.exp(scaled_q)

                t_key, noise_key = jax.random.split(policy_key)
                t = jax.random.randint(
                    t_key,
                    (obs.shape[0],),
                    0,
                    self.agent.num_timesteps,
                )
                loss = self.agent.diffusion.weighted_p_loss(
                    noise_key,
                    q_weights,
                    denoiser,
                    t,
                    jax.lax.stop_gradient(action),
                )

                return loss, (q_weights, scaled_q, q_mean, q_std)

            (policy_loss, (q_weights, scaled_q, q_mean, q_std)), policy_grads = jax.value_and_grad(
                policy_loss_fn, has_aux=True
            )(policy_params)

            def log_alpha_loss_fn(log_alpha_val: jax.Array) -> jax.Array:
                approx_entropy = 0.5 * self.agent.act_dim * jnp.log(
                    2
                    * jnp.pi
                    * jnp.e
                    * (0.1 * jnp.exp(log_alpha_val)) ** 2
                )
                return -1.0 * log_alpha_val * (
                    -1.0 * jax.lax.stop_gradient(approx_entropy)
                    + self.agent.target_entropy
                )

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
                    lambda p, o: param_update(
                        optim, p, jax.grad(log_alpha_loss_fn)(p), o
                    ),
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

            dyn_params, dyn_opt_state = param_update(
                self.dyn_optim, dyn_params, dyn_grads, dyn_opt_state
            )
            reward_params, rew_opt_state = param_update(
                self.rew_optim, reward_params, rew_grads, rew_opt_state
            )
            value_params, value_opt_state = param_update(
                self.value_optim, value_params, val_grads, value_opt_state
            )
            policy_params, policy_opt_state = delay_param_update(
                self.policy_optim, policy_params, policy_grads, policy_opt_state
            )
            log_alpha, log_alpha_opt_state = delay_alpha_param_update(
                self.alpha_optim, log_alpha, log_alpha_opt_state
            )

            target_value_params = delay_target_update(
                value_params, target_value_params, self.tau
            )
            target_policy_params = delay_target_update(
                policy_params, target_policy_params, self.tau
            )

            new_running_mean = running_mean + 0.001 * (q_mean - running_mean)
            new_running_std = running_std + 0.001 * (q_std - running_std)

            new_state = ModelBasedTrainState(
                params=ModelBasedParams(
                    policy=policy_params,
                    target_policy=target_policy_params,
                    dynamics=dyn_params,
                    reward=reward_params,
                    value=value_params,
                    target_value=target_value_params,
                    log_alpha=log_alpha,
                ),
                opt_state=ModelBasedOptStates(
                    policy=policy_opt_state,
                    dynamics=dyn_opt_state,
                    reward=rew_opt_state,
                    value=value_opt_state,
                    log_alpha=log_alpha_opt_state,
                ),
                step=step + 1,
                running_mean=new_running_mean,
                running_std=new_running_std,
            )

            info: Metric = {
                "dyn_loss": dyn_loss,
                "reward_loss": rew_loss,
                "value_loss": val_loss,
                "policy_loss": policy_loss,
                "alpha": jnp.exp(log_alpha),
                "q_weights_std": jnp.std(q_weights),
                "q_weights_mean": jnp.mean(q_weights),
                "q_weights_min": jnp.min(q_weights),
                "q_weights_max": jnp.max(q_weights),
                "scale_q_mean": jnp.mean(scaled_q),
                "scale_q_std": jnp.std(scaled_q),
                "running_q_mean": new_running_mean,
                "running_q_std": new_running_std,
            }

            return new_state, info

        def stateless_get_action(key: jax.Array, params: ModelBasedParams, obs: jax.Array) -> jax.Array:
            (
                policy_params,
                _,
                dyn_params,
                reward_params,
                value_params,
                target_value_params,
                log_alpha,
            ) = params

            # Support both unbatched obs (obs_dim,) and batched obs (B, obs_dim)
            if obs.ndim == 1:
                obs = obs[None, :]

            key_act, key_q, key_noise = jax.random.split(key, 3)

            def sample_particle(k: jax.Array) -> jax.Array:
                def model_fn(t, x):
                    return self.agent.policy(policy_params, obs, x, t)

                return self.agent.diffusion.p_sample(
                    k, model_fn, (*obs.shape[:-1], self.agent.act_dim)
                )

            keys = jax.random.split(key_act, self.agent.num_particles)
            acts = jax.vmap(sample_particle)(keys)

            batch_size = obs.shape[0]
            s_flat = jnp.repeat(obs[None, ...], self.agent.num_particles, axis=0).reshape(
                -1, self.agent.obs_dim
            )
            a_flat = acts.reshape(-1, self.agent.act_dim)
            q_flat = q_model(dyn_params, reward_params, target_value_params, key_q, s_flat, a_flat)
            qs = q_flat.reshape(self.agent.num_particles, batch_size)

            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            best_act = jnp.take_along_axis(
                acts, q_best_ind[..., None], axis=0
            ).squeeze(axis=0)

            best_act = best_act + jax.random.normal(key_noise, best_act.shape) * jnp.exp(
                log_alpha
            ) * self.agent.noise_scale
            return best_act

        def stateless_get_deterministic_action(params: ModelBasedParams, obs: jax.Array) -> jax.Array:
            key = jax.random.key(0)
            (
                policy_params,
                _,
                dyn_params,
                reward_params,
                value_params,
                _,
                _,
            ) = params

            log_alpha = -jnp.inf
            params_det = ModelBasedParams(
                policy=policy_params,
                target_policy=policy_params,
                dynamics=dyn_params,
                reward=reward_params,
                value=value_params,
                target_value=value_params,
                log_alpha=log_alpha,
            )

            act = stateless_get_action(key, params_det, obs)
            # For single-environment evaluation, obs is (obs_dim,) and we want
            # an action of shape (act_dim,), not (1, act_dim).
            if act.ndim == 2 and act.shape[0] == 1:
                act = act[0]
            return act

        self._implement_common_behavior(stateless_update, stateless_get_action, stateless_get_deterministic_action)

    def get_policy_params(self):
        return self.state.params
