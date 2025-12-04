from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
import haiku as hk

from relax.network.blocks import Activation, Identity, mlp, scaled_sinusoidal_encoding


class PcParams(NamedTuple):
    """Parameters for the four per-transition PC-MD networks.

    - policy: energy over (s0, a_t, t)
    - dynamics: energy over (s0, a_t, s_t, t)
    - reward: amortized reward over (s0, a_t, s_t, t)
    - value: amortized value over (s_t, t)
    - value_targ: EMA target parameters for value
    """
    policy: hk.Params
    dynamics: hk.Params
    reward: hk.Params
    value: hk.Params
    value_targ: hk.Params


@dataclass
class PcNet:
    """Container for PC-MD transition networks.

    These functions have the following signatures (batched):
      policy(params, s0, a_t, t) -> (B,)
      dynamics(params, s0, a_t, s_t, t) -> (B,)
      reward(params, s0, a_t, s_t, t) -> (B,)
      value(params, s_t, t) -> (B,)
    """

    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    dynamics: Callable[[hk.Params, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]
    reward: Callable[[hk.Params, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]
    value: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    obs_dim: int
    act_dim: int
    time_dim: int


def _policy_forward(
    hidden_sizes: Sequence[int],
    activation: Activation,
    time_dim: int,
) -> Callable[[jax.Array, jax.Array, jax.Array], jax.Array]:
    """Builds a Haiku forward fn for the policy energy E_pi(s0, a_t, t).

    We mirror other diffusion networks by encoding t with scaled_sinusoidal_encoding
    followed by an MLP, and using an MLP over [s0, a_t, te].
    """

    def forward(s0: jax.Array, a_t: jax.Array, t: jax.Array) -> jax.Array:
        # s0: (B, Ds), a_t: (B, Da), t: (B,) or broadcastable
        batch_shape = s0.shape[:-1]
        te = scaled_sinusoidal_encoding(t, dim=time_dim, batch_shape=batch_shape)
        x = jnp.concatenate([s0, a_t, te], axis=-1)
        return mlp(hidden_sizes, 1, activation, Identity, squeeze_output=True)(x)

    return forward


def _dynamics_forward(
    hidden_sizes: Sequence[int],
    activation: Activation,
    time_dim: int,
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
    """Builds dynamics energy E_dyn(s0, a_t, s_t, t)."""

    def forward(s0: jax.Array, a_t: jax.Array, s_t: jax.Array, t: jax.Array) -> jax.Array:
        batch_shape = s0.shape[:-1]
        te = scaled_sinusoidal_encoding(t, dim=time_dim, batch_shape=batch_shape)
        x = jnp.concatenate([s0, a_t, s_t, te], axis=-1)
        return mlp(hidden_sizes, 1, activation, Identity, squeeze_output=True)(x)

    return forward


def _reward_forward(
    hidden_sizes: Sequence[int],
    activation: Activation,
    time_dim: int,
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
    """Builds amortized reward network R_hat(s0, a_t, s_t, t)."""

    def forward(s0: jax.Array, a_t: jax.Array, s_t: jax.Array, t: jax.Array) -> jax.Array:
        batch_shape = s0.shape[:-1]
        te = scaled_sinusoidal_encoding(t, dim=time_dim, batch_shape=batch_shape)
        x = jnp.concatenate([s0, a_t, s_t, te], axis=-1)
        return mlp(hidden_sizes, 1, activation, Identity, squeeze_output=True)(x)

    return forward


def _value_forward(
    hidden_sizes: Sequence[int],
    activation: Activation,
    time_dim: int,
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Builds amortized value network V_hat(s_t, t)."""

    def forward(s_t: jax.Array, t: jax.Array) -> jax.Array:
        batch_shape = s_t.shape[:-1]
        te = scaled_sinusoidal_encoding(t, dim=time_dim, batch_shape=batch_shape)
        x = jnp.concatenate([s_t, te], axis=-1)
        return mlp(hidden_sizes, 1, activation, Identity, squeeze_output=True)(x)

    return forward


def create_pcmd_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    *,
    activation: Activation,
    time_dim: int = 16,
) -> Tuple[PcNet, PcParams]:
    """Create PC-MD transition networks and initial parameters.

    The architecture mirrors other Haiku networks in this repo: MLPs with
    sinusoidal time embeddings, with sizes controlled by hidden_sizes and
    activation.
    """

    policy_forward = _policy_forward(hidden_sizes, activation, time_dim)
    dynamics_forward = _dynamics_forward(hidden_sizes, activation, time_dim)
    reward_forward = _reward_forward(hidden_sizes, activation, time_dim)
    value_forward = _value_forward(hidden_sizes, activation, time_dim)

    policy = hk.without_apply_rng(hk.transform(policy_forward))
    dynamics = hk.without_apply_rng(hk.transform(dynamics_forward))
    reward = hk.without_apply_rng(hk.transform(reward_forward))
    value = hk.without_apply_rng(hk.transform(value_forward))

    @jax.jit
    def init(key: jax.Array) -> PcParams:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        sample_s = jnp.zeros((1, obs_dim), dtype=jnp.float32)
        sample_a = jnp.zeros((1, act_dim), dtype=jnp.float32)
        sample_t = jnp.zeros((1,), dtype=jnp.float32)

        p_policy = policy.init(k1, sample_s, sample_a, sample_t)
        p_dyn = dynamics.init(k2, sample_s, sample_a, sample_s, sample_t)
        p_rew = reward.init(k3, sample_s, sample_a, sample_s, sample_t)
        p_val = value.init(k4, sample_s, sample_t)
        p_val_targ = p_val
        return PcParams(policy=p_policy, dynamics=p_dyn, reward=p_rew, value=p_val, value_targ=p_val_targ)

    params = init(key)
    net = PcNet(
        policy=policy.apply,
        dynamics=dynamics.apply,
        reward=reward.apply,
        value=value.apply,
        obs_dim=obs_dim,
        act_dim=act_dim,
        time_dim=time_dim,
    )
    return net, params
