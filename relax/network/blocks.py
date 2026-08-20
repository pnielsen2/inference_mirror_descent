from dataclasses import dataclass
from typing import Any, Callable, Sequence

import jax, jax.numpy as jnp
import haiku as hk

from relax.utils.jax_utils import fix_repr, is_broadcastable

Activation = Callable[[jax.Array], jax.Array]
Identity: Activation = lambda x: x


@dataclass
@fix_repr
class ValueNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    w_init: Any = None
    name: str = None

    def __call__(self, obs: jax.Array) -> jax.Array:
        return mlp(self.hidden_sizes, 1, self.activation, self.output_activation, w_init=self.w_init)(obs)[..., 0]


@dataclass
@fix_repr
class QNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    w_init: Any = None
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array) -> jax.Array:
        input = jnp.concatenate((obs, act), axis=-1)
        return mlp(self.hidden_sizes, 1, self.activation, self.output_activation, w_init=self.w_init)(input)[..., 0]


def mlp(hidden_sizes: Sequence[int], output_size: int, activation: Activation, output_activation: Activation, w_init: Any = None) -> Callable[[jax.Array], jax.Array]:
    layers = []
    for hidden_size in hidden_sizes:
        layers += [hk.Linear(hidden_size, w_init=w_init), activation]
    layers += [hk.Linear(output_size, w_init=w_init), output_activation]
    return hk.Sequential(layers)


def scaled_sinusoidal_encoding(t: jax.Array, *, dim: int, theta: int = 1000, batch_shape = None) -> jax.Array:
    """Sinusoidal encoding of a noise level; ``theta`` sets the range it resolves.

    The ``dim/2`` angular frequencies sit geometrically between ``1`` and
    ``theta**(-(dim/2-1)/(dim/2))`` radians per unit of ``t``. Two things follow:
    the top channel gives ~1 rad per unit whatever ``theta`` is, and the bottom
    sweeps ``range * theta**(-(dim/2-1)/(dim/2))`` radians end to end. A channel
    that sweeps far under a radian is a near-constant, so ``theta`` decides how
    many of the ``dim/2`` channels carry anything over a given input range.

    ``theta = 10000`` over ``[0, 1000)`` is what DDPM inherited from Transformer
    positional encodings; that is an observed pairing, not a derived law, but it
    does leave 7 of 8 channels informative and the last monotone. The default
    here reproduces that shape for a log-SNR range of ``[-50, 50]``: at ``theta =
    1000, dim = 16`` the frequencies are ``[1, .42, .18, .075, .032, .013,
    .0056, .0024]``, sweeping ``[100, 42, 18, 7.5, 3.2, 1.3, .56, .24]`` radians
    -- 6 informative, 1 monotone. Under ``theta = 10000`` five of the eight
    would instead vary by under a radian end to end, i.e. be wasted.

    Note what this does NOT change: the top frequency is 1 rad per unit either
    way, so the separation between two *adjacent* noise levels is untouched.
    This is about how many embedding dimensions carry signal, not about
    sensitivity to nearby levels; only rescaling the input itself moves that.
    """
    assert dim % 2 == 0
    if batch_shape is not None:
        assert is_broadcastable(jnp.shape(t), batch_shape)

    scale = 1 / dim ** 0.5
    half_dim = dim // 2
    freq_seq = jnp.arange(half_dim) / half_dim
    inv_freq = theta ** -freq_seq

    emb = jnp.einsum('..., j -> ... j', t, inv_freq)
    emb = jnp.concatenate((
        jnp.sin(emb),
        jnp.cos(emb),
    ), axis=-1)
    emb *= scale

    if batch_shape is not None:
        emb = jnp.broadcast_to(emb, (*batch_shape, dim))

    return emb
