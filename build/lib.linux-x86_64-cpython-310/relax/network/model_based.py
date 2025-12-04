from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
import haiku as hk
import math

from relax.network.blocks import Activation, Identity, mlp, scaled_sinusoidal_encoding
from relax.utils.diffusion import GaussianDiffusion


class ModelBasedParams(NamedTuple):
    policy: hk.Params
    target_policy: hk.Params
    dynamics: hk.Params
    reward: hk.Params
    value: hk.Params
    target_value: hk.Params
    log_alpha: jax.Array


@dataclass
class ModelBasedNet:
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    dynamics: Callable[[hk.Params, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]
    reward: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    value: Callable[[hk.Params, jax.Array], jax.Array]
    num_timesteps: int
    obs_dim: int
    act_dim: int
    num_particles: int
    target_entropy: float
    noise_scale: float
    beta_schedule_scale: float
    beta_schedule_type: str = "linear"

    @property
    def diffusion(self) -> GaussianDiffusion:
        return GaussianDiffusion(self.num_timesteps, self.beta_schedule_scale, self.beta_schedule_type)

    @property
    def dyn_diffusion(self) -> GaussianDiffusion:
        return GaussianDiffusion(self.num_timesteps, self.beta_schedule_scale, self.beta_schedule_type)

    def get_action(self, key: jax.Array, policy_params: ModelBasedParams, obs: jax.Array, q_func: Callable[[jax.Array, jax.Array], jax.Array]) -> jax.Array:
        policy_params, log_alpha = (policy_params.policy, policy_params.log_alpha)

        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)

        def sample(key: jax.Array) -> Tuple[jax.Array, jax.Array]:
            act = self.diffusion.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            q = q_func(obs, act)
            return act.clip(-1, 1), q

        key, noise_key = jax.random.split(key)
        if self.num_particles == 1:
            act, _ = sample(key)
        else:
            keys = jax.random.split(key, self.num_particles)
            acts, qs = jax.vmap(sample)(keys)
            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
        act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale
        return act


def _dyn_forward(hidden_sizes: Sequence[int], activation: Activation, time_dim: int) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
    def forward(s: jax.Array, a: jax.Array, s_t: jax.Array, t: jax.Array) -> jax.Array:
        batch_shape = s.shape[:-1]
        te = scaled_sinusoidal_encoding(t, dim=time_dim, batch_shape=batch_shape)
        x = jnp.concatenate([s, a, s_t, te], axis=-1)
        return mlp(hidden_sizes, s.shape[-1], activation, Identity)(x)

    return forward


def _reward_forward(hidden_sizes: Sequence[int], activation: Activation) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    def forward(s: jax.Array, a: jax.Array, s_next: jax.Array) -> jax.Array:
        x = jnp.concatenate([s, a, s_next], axis=-1)
        return mlp(hidden_sizes, 1, activation, Identity, squeeze_output=True)(x)

    return forward


def _value_forward(hidden_sizes: Sequence[int], activation: Activation) -> Callable[[jax.Array], jax.Array]:
    def forward(s: jax.Array) -> jax.Array:
        return mlp(hidden_sizes, 1, activation, Identity, squeeze_output=True)(s)

    return forward


def create_model_based_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    diffusion_hidden_sizes: Sequence[int],
    *,
    activation: Activation,
    num_timesteps: int,
    num_particles: int,
    noise_scale: float,
    beta_schedule_scale: float,
    beta_schedule_type: str = "linear",
) -> Tuple[ModelBasedNet, ModelBasedParams]:
    from relax.network.blocks import DACERPolicyNet

    policy = hk.without_apply_rng(hk.transform(lambda obs, act, t: DACERPolicyNet(diffusion_hidden_sizes, activation)(obs, act, t)))
    dynamics = hk.without_apply_rng(hk.transform(_dyn_forward(hidden_sizes, activation, time_dim=16)))
    reward = hk.without_apply_rng(hk.transform(_reward_forward(hidden_sizes, activation)))
    value = hk.without_apply_rng(hk.transform(_value_forward(hidden_sizes, activation)))

    @jax.jit
    def init(key, obs, act):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        sample_s = jnp.zeros((1, obs_dim))
        sample_a = jnp.zeros((1, act_dim))
        sample_t = jnp.zeros((1,))

        p_policy = policy.init(k1, sample_s, sample_a, sample_t)
        p_dyn = dynamics.init(k2, sample_s, sample_a, sample_s, sample_t)
        p_rew = reward.init(k3, sample_s, sample_a, sample_s)
        p_val = value.init(k4, sample_s)
        p_val_targ = p_val

        log_alpha = jnp.array(math.log(5.0), dtype=jnp.float32)

        return ModelBasedParams(
            policy=p_policy,
            target_policy=p_policy,
            dynamics=p_dyn,
            reward=p_rew,
            value=p_val,
            target_value=p_val_targ,
            log_alpha=log_alpha,
        )

    params = init(key, jnp.zeros((1, obs_dim)), jnp.zeros((1, act_dim)))

    net = ModelBasedNet(
        policy=policy.apply,
        dynamics=dynamics.apply,
        reward=reward.apply,
        value=value.apply,
        num_timesteps=num_timesteps,
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_particles=num_particles,
        target_entropy=-act_dim * 0.9,
        noise_scale=noise_scale,
        beta_schedule_scale=beta_schedule_scale,
        beta_schedule_type=beta_schedule_type,
    )

    return net, params
