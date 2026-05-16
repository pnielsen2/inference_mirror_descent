from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple, Optional

import jax, jax.numpy as jnp
import haiku as hk

from relax.network.blocks import Activation, QNet, EnergyPolicyNet
from relax.utils.diffusion import GaussianDiffusion

class Diffv2Params(NamedTuple):
    q: Tuple  # tuple of N Q network params
    target_q: Tuple  # tuple of N target Q network params
    policy: hk.Params


@dataclass
class Diffv2Net:
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    num_timesteps: int
    act_dim: int
    beta_schedule_scale: float
    beta_schedule_type: str = 'linear'
    x_recon_clip_radius: Optional[float] = 1.0
    snr_max: float = 124.0
    energy_fn: Optional[Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]] = None
    mala_steps: int = 1

    @property
    def diffusion(self) -> GaussianDiffusion:
        return GaussianDiffusion(self.num_timesteps, 
                                 self.beta_schedule_scale,
                                 self.beta_schedule_type,
                                 x_recon_clip_radius=self.x_recon_clip_radius,
                                 snr_max=self.snr_max)


def create_diffv2_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    diffusion_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
    beta_schedule_scale: float = 0.3,
    beta_schedule_type: str = "linear",
    mala_steps: int = 1,
    x_recon_clip_radius: Optional[float] = 1.0,
    snr_max: float = 124.0,
    num_q_networks: int = 2,
) -> Tuple[Diffv2Net, Diffv2Params]:
    q_net = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))

    policy = hk.without_apply_rng(
        hk.transform(
            lambda obs, act, t, h: EnergyPolicyNet(diffusion_hidden_sizes, activation)(
                obs,
                act,
                t,
                h,
            )
        )
    )

    def policy_apply(params, obs, act, t):
        h_zeros = jnp.zeros(obs.shape[:-1], dtype=jnp.float32)
        return jax.grad(lambda a: policy.apply(params, obs, a, t, h_zeros).sum())(act)
    def energy_apply(params, obs, act, t):
        h_zeros = jnp.zeros(obs.shape[:-1], dtype=jnp.float32)
        return policy.apply(params, obs, act, t, h_zeros)

    num_q = int(num_q_networks)

    @jax.jit
    def init(key, obs, act):
        keys = jax.random.split(key, num_q + 1)
        policy_key = keys[-1]
        q_params_list = []
        for i in range(num_q):
            q_params_list.append(q_net.init(keys[i], obs, act))
        q_params = tuple(q_params_list)
        target_q_params = tuple(jax.tree.map(lambda x: x, qp) for qp in q_params)
        policy_params = policy.init(policy_key, obs, act, 0, jnp.zeros((1,), dtype=jnp.float32))
        return Diffv2Params(q_params, target_q_params, policy_params)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = Diffv2Net(
        q=q_net.apply,
        policy=policy_apply,
        num_timesteps=num_timesteps,
        act_dim=act_dim,
        beta_schedule_scale=beta_schedule_scale,
        beta_schedule_type=beta_schedule_type,
        x_recon_clip_radius=x_recon_clip_radius,
        snr_max=snr_max,
        energy_fn=energy_apply,
        mala_steps=mala_steps,
    )
    return net, params
