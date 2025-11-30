from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle

from relax.algorithm.base import Algorithm
from relax.algorithm.dpmd import Diffv2OptStates, Diffv2TrainState
from relax.network.diffv2 import Diffv2Net, Diffv2Params
from relax.utils.experience import Experience
from relax.utils.typing_utils import Metric


class DPMDBC(Algorithm):
    """DPMD variant with behavior-cloning diffusion and Q tilt only at inference.

    This algorithm is intentionally as close as possible to DPMD:
      - Same Diffv2Net architecture and hyperparameters.
      - Same Q training and TD target.
      - Same alpha/log_alpha training.
      - Same diffusion sampler and multi-particle best-of-Q selection at inference.

    The only change is in the policy loss:
      - DPMD: diffusion policy is trained with Q-weighted epsilon loss on next_action.
      - DPMDBC: diffusion policy is trained with *unweighted* diffusion loss (behavior cloning)
        on the replay buffer actions, without any Q-dependent weighting.

    As a result, all Q influence on the policy happens at inference time via the
    existing multi-particle best-of-Q selection in Diffv2Net.get_action, not via
    the training loss.
    """

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
        train_q_on_noisy_actions: bool = False,
        guided_sampling: bool = False,
        tfg_recurrence: bool = False,
        tfg_recur_steps: int = 3,
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
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
            ),
            step=jnp.int32(0),
            entropy=jnp.float32(0.0),
            running_mean=jnp.float32(0.0),
            running_std=jnp.float32(1.0),
        )
        self.use_ema = use_ema
        self.train_q_on_noisy_actions = train_q_on_noisy_actions
        self.guided_sampling = guided_sampling
        self.tfg_recurrence = tfg_recurrence
        if self.tfg_recurrence and self.train_q_on_noisy_actions:
            raise ValueError(
                "tfg_recurrence requires Q trained on clean actions; disable train_q_on_noisy_actions when using this flag."
            )
        # Hyperparameters for recurrence-style guidance in action space
        self.tfg_recur_steps = tfg_recur_steps
        self.tfg_step_size = 0.05

        @jax.jit
        def stateless_update(
            key: jax.Array,
            state: Diffv2TrainState,
            data: Experience,
        ) -> Tuple[Diffv2TrainState, Metric]:
            obs, action, reward, next_obs, done = (
                data.obs,
                data.action,
                data.reward,
                data.next_obs,
                data.done,
            )
            (
                q1_params,
                q2_params,
                target_q1_params,
                target_q2_params,
                policy_params,
                target_policy_params,
                log_alpha,
            ) = state.params
            (
                q1_opt_state,
                q2_opt_state,
                policy_opt_state,
                log_alpha_opt_state,
            ) = state.opt_state
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std

            (
                next_eval_key,
                new_eval_key,
                new_q1_eval_key,
                new_q2_eval_key,
                log_alpha_key,
                diffusion_time_key,
                diffusion_noise_key,
                q_diffusion_time_key,
                q_diffusion_noise_key,
            ) = jax.random.split(key, 9)

            reward = reward * self.reward_scale

            if self.train_q_on_noisy_actions:
                t_q = jax.random.randint(
                    q_diffusion_time_key,
                    (obs.shape[0],),
                    0,
                    self.agent.num_timesteps,
                )
                noise_q = jax.random.normal(q_diffusion_noise_key, action.shape)
                action_for_q = jax.vmap(self.agent.diffusion.q_sample)(
                    t_q, action, noise_q
                )
            else:
                action_for_q = action

            def get_min_q(s, a):
                q1 = self.agent.q(q1_params, s, a)
                q2 = self.agent.q(q2_params, s, a)
                return jnp.minimum(q1, q2)

            def get_min_target_q(s, a):
                q1 = self.agent.q(target_q1_params, s, a)
                q2 = self.agent.q(target_q2_params, s, a)
                return jnp.minimum(q1, q2)

            # Target Q uses the same next_action construction as in DPMD,
            # optionally replaced by a guided sampler that evaluates Q on
            # an intermediate noisy diffusion state.
            if self.guided_sampling:
                next_action = self.agent.get_action_guided(
                    next_eval_key,
                    (policy_params, log_alpha, q1_params, q2_params),
                    next_obs,
                )
            else:
                next_action = self.agent.get_action(
                    next_eval_key,
                    (policy_params, log_alpha, q1_params, q2_params),
                    next_obs,
                )
            q1_target = self.agent.q(target_q1_params, next_obs, next_action)
            q2_target = self.agent.q(target_q2_params, next_obs, next_action)
            q_target = jnp.minimum(q1_target, q2_target)
            q_backup = reward + (1.0 - done) * self.gamma * q_target

            def q_loss_fn(q_params: hk.Params) -> Tuple[jax.Array, jax.Array]:
                q = self.agent.q(q_params, obs, action_for_q)
                q_loss = jnp.mean((q - q_backup) ** 2)
                return q_loss, q

            (q1_loss, q1), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(
                q1_params
            )
            (q2_loss, q2), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(
                q2_params
            )

            q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
            q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
            q1_params = optax.apply_updates(q1_params, q1_update)
            q2_params = optax.apply_updates(q2_params, q2_update)

            # Unweighted diffusion behavior cloning for the policy: match buffer actions.
            def policy_loss_fn(policy_params: hk.Params) -> jax.Array:
                def denoiser(t, x):
                    return self.agent.policy(policy_params, obs, x, t)

                t = jax.random.randint(
                    diffusion_time_key,
                    (obs.shape[0],),
                    0,
                    self.agent.num_timesteps,
                )
                loss = self.agent.diffusion.p_loss(
                    diffusion_noise_key,
                    denoiser,
                    t,
                    jax.lax.stop_gradient(action),
                )
                return loss, ()

            (policy_loss, _), policy_grads = jax.value_and_grad(
                policy_loss_fn, has_aux=True
            )(policy_params)

            # log_alpha update is identical to DPMD.
            def log_alpha_loss_fn(log_alpha_val: jax.Array) -> jax.Array:
                approx_entropy = 0.5 * self.agent.act_dim * jnp.log(
                    2
                    * jnp.pi
                    * jnp.e
                    * (0.1 * jnp.exp(log_alpha_val)) ** 2
                )
                log_alpha_loss = -1.0 * log_alpha_val * (
                    -1.0 * jax.lax.stop_gradient(approx_entropy)
                    + self.agent.target_entropy
                )
                return log_alpha_loss

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

            q1_params, q1_opt_state = param_update(
                self.optim, q1_params, q1_grads, q1_opt_state
            )
            q2_params, q2_opt_state = param_update(
                self.optim, q2_params, q2_grads, q2_opt_state
            )
            policy_params, policy_opt_state = delay_param_update(
                self.policy_optim, policy_params, policy_grads, policy_opt_state
            )
            log_alpha, log_alpha_opt_state = delay_alpha_param_update(
                self.alpha_optim, log_alpha, log_alpha_opt_state
            )

            target_q1_params = delay_target_update(
                q1_params, target_q1_params, self.tau
            )
            target_q2_params = delay_target_update(
                q2_params, target_q2_params, self.tau
            )
            target_policy_params = delay_target_update(
                policy_params, target_policy_params, self.tau
            )

            # Track running statistics of Q for logging (as in DPMD).
            q_min = get_min_q(next_obs, next_action)
            q_mean, q_std = q_min.mean(), q_min.std()
            new_running_mean = running_mean + 0.001 * (q_mean - running_mean)
            new_running_std = running_std + 0.001 * (q_std - running_std)

            new_state = Diffv2TrainState(
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
            )

            info: Metric = {
                "q1_loss": q1_loss,
                "q1_mean": jnp.mean(q1),
                "q1_max": jnp.max(q1),
                "q1_min": jnp.min(q1),
                "q2_loss": q2_loss,
                "policy_loss": policy_loss,
                "alpha": jnp.exp(log_alpha),
                "running_q_mean": new_running_mean,
                "running_q_std": new_running_std,
            }
            return new_state, info

        def stateless_get_action_tfg(
            key: jax.Array,
            params: Diffv2Params,
            obs: jax.Array,
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

            def tfg_sample(single_key: jax.Array):
                x_key, noise_key = jax.random.split(single_key)
                x = 0.5 * jax.random.normal(x_key, shape)
                noise = jax.random.normal(noise_key, (timesteps, *shape))
                t_seq = jnp.arange(timesteps)[::-1]

                def body_fn(x_t, inputs):
                    t, noise_t = inputs

                    def loss_for_grad(x_in: jax.Array) -> jax.Array:
                        noise_pred = model_fn(t, x_in)
                        x0_hat = (
                            x_in * B.sqrt_recip_alphas_cumprod[t]
                            - noise_pred * B.sqrt_recipm1_alphas_cumprod[t]
                        )
                        x0_hat = jnp.clip(x0_hat, -1.0, 1.0)
                        q1 = self.agent.q(q1_params, obs_batch, x0_hat)
                        q2 = self.agent.q(q2_params, obs_batch, x0_hat)
                        q = jnp.minimum(q1, q2)
                        return -jnp.mean(q)

                    def recur_body(x_cur, _):
                        grad_x = jax.grad(loss_for_grad)(x_cur)
                        x_new = x_cur - self.tfg_step_size * grad_x
                        x_new = jnp.clip(x_new, -1.0, 1.0)
                        return x_new, None

                    if self.tfg_recur_steps > 0:
                        x_t, _ = jax.lax.scan(
                            recur_body, x_t, jnp.arange(self.tfg_recur_steps)
                        )

                    noise_pred = model_fn(t, x_t)
                    model_mean, model_log_variance = self.agent.diffusion.p_mean_variance(
                        t, x_t, noise_pred
                    )
                    x_next = model_mean + (t > 0) * jnp.exp(
                        0.5 * model_log_variance
                    ) * noise_t
                    return x_next, None

                x_final, _ = jax.lax.scan(body_fn, x, (t_seq, noise))
                act_final = jnp.clip(x_final, -1.0, 1.0)
                q1 = self.agent.q(q1_params, obs_batch, act_final)
                q2 = self.agent.q(q2_params, obs_batch, act_final)
                q = jnp.minimum(q1, q2)
                return act_final, q

            key, noise_key = jax.random.split(key)
            if self.agent.num_particles == 1:
                act, _ = tfg_sample(key)
            else:
                keys = jax.random.split(key, self.agent.num_particles)
                acts, qs = jax.vmap(tfg_sample)(keys)
                q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
                act = jnp.take_along_axis(
                    acts, q_best_ind[..., None], axis=0
                ).squeeze(axis=0)

            act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(
                log_alpha
            ) * self.agent.noise_scale

            if single:
                return act[0]
            else:
                return act

        if self.tfg_recurrence:
            stateless_get_action = stateless_get_action_tfg
        elif self.guided_sampling:
            def stateless_get_action(key: jax.Array, params: Diffv2Params, obs: jax.Array) -> jax.Array:
                return self.agent.get_action_guided(key, params, obs)
        else:
            stateless_get_action = self.agent.get_action

        self._implement_common_behavior(
            stateless_update,
            stateless_get_action,
            self.agent.get_deterministic_action,
        )

    def get_policy_params(self):
        return (
            self.state.params.policy,
            self.state.params.log_alpha,
            self.state.params.q1,
            self.state.params.q2,
        )

    def get_policy_params_to_save(self):
        return (
            self.state.params.target_poicy,
            self.state.params.log_alpha,
            self.state.params.q1,
            self.state.params.q2,
        )

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params_to_save(), obs)
        return np.asarray(action)
