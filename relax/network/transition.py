from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
import haiku as hk

from relax.utils.diffusion import GaussianDiffusion


class TransitionParams(NamedTuple):
    dyn_params: hk.Params
    reward_params: hk.Params


@dataclass
class TransitionNet:
    dyn_policy: Callable[..., jax.Array]
    reward: Callable[..., jax.Array]
    diffusion: GaussianDiffusion
    num_timesteps: int


def create_transition_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    diffusion_hidden_sizes: Sequence[int],
    num_timesteps: int,
    beta_schedule_scale: float,
    beta_schedule_type: str,
) -> Tuple[TransitionNet, TransitionParams]:
    def dyn_mlp(obs, act, sprime, t):
        x = jnp.concatenate([obs, act, sprime, t], axis=-1)
        mlp = hk.nets.MLP(list(hidden_sizes) + [sprime.shape[-1]])
        return mlp(x)

    def reward_mlp(obs, act, sprime):
        x = jnp.concatenate([obs, act, sprime], axis=-1)
        mlp = hk.nets.MLP(list(hidden_sizes) + [1])
        out = mlp(x)
        return jnp.squeeze(out, axis=-1)

    dyn = hk.without_apply_rng(hk.transform(lambda obs, act, sprime, t: dyn_mlp(obs, act, sprime, t)))
    reward = hk.without_apply_rng(hk.transform(lambda obs, act, sprime: reward_mlp(obs, act, sprime)))

    @jax.jit
    def init_params(key, obs, act, sprime, t):
        k1, k2 = jax.random.split(key)
        dyn_params = dyn.init(k1, obs, act, sprime, t)
        reward_params = reward.init(k2, obs, act, sprime)
        return TransitionParams(dyn_params=dyn_params, reward_params=reward_params)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    sample_sprime = jnp.zeros((1, obs_dim))
    sample_t = jnp.zeros((1, 1))
    params = init_params(key, sample_obs, sample_act, sample_sprime, sample_t)

    diffusion = GaussianDiffusion(num_timesteps, beta_schedule_scale, beta_schedule_type)

    net = TransitionNet(
        dyn_policy=dyn.apply,
        reward=reward.apply,
        diffusion=diffusion,
        num_timesteps=num_timesteps,
    )
    return net, params
