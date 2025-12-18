from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Sequence, Tuple, Union

import jax, jax.numpy as jnp
import haiku as hk
from haiku.initializers import Constant

from relax.utils.jax_utils import fix_repr, is_broadcastable

Activation = Callable[[jax.Array], jax.Array]
Identity: Activation = lambda x: x
Tanh: Activation = lambda x: jnp.tanh(x)


@dataclass
@fix_repr
class ValueNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array) -> jax.Array:
        return mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True)(obs)


@dataclass
@fix_repr
class QNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array) -> jax.Array:
        input = jnp.concatenate((obs, act), axis=-1)
        return mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True)(input)


@dataclass
@fix_repr
class DistributionalQNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    min_log_std: float = -0.1
    max_log_std: float = 4.0
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array) -> Tuple[jax.Array, jax.Array]:
        input = jnp.concatenate((obs, act), axis=-1)
        value_mean = mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True)(input)
        value_log_std = mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True)(input)
        denominator = max(abs(self.min_log_std), abs(self.max_log_std))
        value_log_std = (
            jnp.maximum( self.max_log_std * jnp.tanh(value_log_std / denominator), 0.0) +
            jnp.minimum(-self.min_log_std * jnp.tanh(value_log_std / denominator), 0.0)
        )
        return value_mean, value_log_std

@dataclass
@fix_repr
class DistributionalQNet2(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array) -> Tuple[jax.Array, jax.Array]:
        input = jnp.concatenate((obs, act), axis=-1)
        output = mlp(self.hidden_sizes, 2, self.activation, self.output_activation)(input)
        value_mean = output[..., 0]
        value_std = jax.nn.softplus(output[..., 1])
        return value_mean, value_std


@dataclass
@fix_repr
class PolicyNet(hk.Module):
    act_dim: int
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    min_log_std: float = -20.0
    max_log_std: float = 0.5
    log_std_mode: Union[str, float] = 'shared'  # shared, separate, global (provide initial value)
    name: str = None

    def __call__(self, obs: jax.Array, *, return_log_std: bool = False) -> jax.Array:
        if self.log_std_mode == 'shared':
            output = mlp(self.hidden_sizes, self.act_dim * 2, self.activation, self.output_activation)(obs)
            mean, log_std = jnp.split(output, 2, axis=-1)
        elif self.log_std_mode == 'separate':
            mean = mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)
            log_std = mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)
        else:
            initial_log_std = float(self.log_std_mode)
            mean = mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)
            log_std = hk.get_parameter('log_std', shape=(self.act_dim,), init=Constant(initial_log_std))
            log_std = jnp.broadcast_to(log_std, mean.shape)
        if not (self.min_log_std is None and self.max_log_std is None):
            log_std = jnp.clip(log_std, self.min_log_std, self.max_log_std)
        if return_log_std:
            return mean, log_std
        else:
            return mean, jnp.exp(log_std)

@dataclass
@fix_repr
class PolicyStdNet(hk.Module):
    act_dim: int
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Tanh
    min_log_std: float = -5.0
    max_log_std: float = 2.0
    name: str = None

    def __call__(self, obs: jax.Array) -> jax.Array:
        log_std = mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)
        return self.min_log_std + (log_std + 1) / 2 * (self.max_log_std - self.min_log_std)


@dataclass
@fix_repr
class DeterministicPolicyNet(hk.Module):
    act_dim: int
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array) -> jax.Array:
        return mlp(self.hidden_sizes, self.act_dim, self.activation, self.output_activation)(obs)


@dataclass
@fix_repr
class ModelNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array) -> jax.Array:
        obs_dim = obs.shape[-1]
        input = jnp.concatenate((obs, act), axis=-1)
        return mlp(self.hidden_sizes, obs_dim, self.activation, self.output_activation)(input)


@dataclass
@fix_repr
class QScoreNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array) -> jax.Array:
        act_dim = act.shape[-1]
        input = jnp.concatenate((obs, act), axis=-1)
        return mlp(self.hidden_sizes, act_dim, self.activation, self.output_activation)(input)


@dataclass
@fix_repr
class DiffusionPolicyNet(hk.Module):
    time_dim: int
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    name: str = None

    def __call__(self, obs: jax.Array, act: jax.Array, t: jax.Array) -> jax.Array:
        act_dim = act.shape[-1]
        te = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=obs.shape[:-1])
        input = jnp.concatenate((obs, act, te), axis=-1)
        return mlp(self.hidden_sizes, act_dim, self.activation, self.output_activation)(input)

@dataclass
@fix_repr
class DACERPolicyNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    time_dim: int = 16
    horizon_dim: int = 8
    name: str = None

    def __call__(
        self,
        obs: jax.Array,
        act: jax.Array,
        t: jax.Array,
        h: jax.Array,
    ) -> jax.Array:
        """Forward pass with required horizon index h.
        
        Args:
            obs: Observation, shape [..., obs_dim]
            act: Action, shape [..., act_dim]
            t: Diffusion timestep
            h: Horizon step index (use 0 for single-step mode)
        """
        act_dim = act.shape[-1]
        te = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=obs.shape[:-1])
        te = hk.Linear(self.time_dim * 2)(te)
        te = self.activation(te)
        te = hk.Linear(self.time_dim)(te)
        # Always use horizon embedding for consistent parameter structure
        he = scaled_sinusoidal_encoding(h, dim=self.horizon_dim, batch_shape=obs.shape[:-1])
        he = hk.Linear(self.horizon_dim * 2)(he)
        he = self.activation(he)
        he = hk.Linear(self.horizon_dim)(he)
        input = jnp.concatenate((obs, act, te, he), axis=-1)
        return mlp(self.hidden_sizes, act_dim, self.activation, self.output_activation)(input)


@dataclass
@fix_repr
class EnergyPolicyNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    time_dim: int = 16
    horizon_dim: int = 8
    name: str = None

    def __call__(
        self,
        obs: jax.Array,
        act: jax.Array,
        t: jax.Array,
        h: jax.Array,
    ) -> jax.Array:
        """Forward pass with required horizon index h.
        
        Args:
            obs: Observation, shape [..., obs_dim]
            act: Action, shape [..., act_dim]
            t: Diffusion timestep
            h: Horizon step index (use 0 for single-step mode)
        """
        te = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=obs.shape[:-1])
        te = hk.Linear(self.time_dim * 2)(te)
        te = self.activation(te)
        te = hk.Linear(self.time_dim)(te)
        # Always use horizon embedding for consistent parameter structure
        he = scaled_sinusoidal_encoding(h, dim=self.horizon_dim, batch_shape=obs.shape[:-1])
        he = hk.Linear(self.horizon_dim * 2)(he)
        he = self.activation(he)
        he = hk.Linear(self.horizon_dim)(he)
        input = jnp.concatenate((obs, act, te, he), axis=-1)
        # Output scalar energy (squeeze last dim)
        return mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True)(input)

@dataclass
@fix_repr
class SequencePolicyNet(hk.Module):
    """Joint sequence denoiser for H-step action sequences.
    
    Takes the full action sequence and outputs noise for all H actions jointly,
    allowing the model to capture correlations across the sequence.
    """
    hidden_sizes: Sequence[int]
    horizon: int
    activation: Activation
    output_activation: Activation = Identity
    time_dim: int = 16
    name: str = None

    def __call__(
        self,
        obs: jax.Array,
        seq_act: jax.Array,
        t: jax.Array,
    ) -> jax.Array:
        """Forward pass for joint sequence denoising.
        
        Args:
            obs: Observation, shape [B, obs_dim]
            seq_act: Action sequence, shape [B, H, act_dim]
            t: Diffusion timestep, shape [B] or [B, 1]
            
        Returns:
            Noise prediction, shape [B, H, act_dim]
        """
        # Normalize obs to rank-2 [B, obs_dim]. Under nested grad/scan,
        # JAX may introduce an extra diffusion-step dimension, e.g. [B, T, obs_dim].
        # For the sequence prior, obs is conceptually constant across T, so we
        # safely collapse that extra axis.
        if obs.ndim > 2:
            obs = obs[:, 0, :]

        B = obs.shape[0]
        H = self.horizon
        act_dim = seq_act.shape[-1]
        
        # Flatten action sequence: [B, H, act_dim] -> [B, H * act_dim]
        seq_flat = seq_act.reshape(B, -1)
        
        # Time embedding
        te = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=(B,))
        te = hk.Linear(self.time_dim * 2)(te)
        te = self.activation(te)
        te = hk.Linear(self.time_dim)(te)
        
        # Concatenate obs, flattened sequence, and time embedding
        input = jnp.concatenate((obs, seq_flat, te), axis=-1)
        
        # MLP outputs full sequence noise
        output = mlp(self.hidden_sizes, H * act_dim, self.activation, self.output_activation)(input)
        
        # Reshape back to [B, H, act_dim]
        return output.reshape(B, H, act_dim)


@dataclass
@fix_repr
class EnergySequencePolicyNet(hk.Module):
    """Energy-based joint sequence denoiser for H-step action sequences.
    
    Outputs a scalar energy for the full sequence; the denoiser is the gradient
    of this energy w.r.t. the action sequence.
    """
    hidden_sizes: Sequence[int]
    horizon: int
    activation: Activation
    output_activation: Activation = Identity
    time_dim: int = 16
    name: str = None

    def __call__(
        self,
        obs: jax.Array,
        seq_act: jax.Array,
        t: jax.Array,
    ) -> jax.Array:
        """Forward pass for joint sequence energy.
        
        Args:
            obs: Observation, shape [B, obs_dim]
            seq_act: Action sequence, shape [B, H, act_dim]
            t: Diffusion timestep, shape [B] or [B, 1]
            
        Returns:
            Energy, shape [B] (scalar per batch element)
        """
        # See SequencePolicyNet.__call__ for discussion: normalize obs to [B, obs_dim]
        # in case JAX has introduced an extra diffusion-step axis.
        if obs.ndim > 2:
            obs = obs[:, 0, :]

        B = obs.shape[0]
        
        # Flatten action sequence: [B, H, act_dim] -> [B, H * act_dim]
        seq_flat = seq_act.reshape(B, -1)
        
        # Time embedding
        te = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=(B,))
        te = hk.Linear(self.time_dim * 2)(te)
        te = self.activation(te)
        te = hk.Linear(self.time_dim)(te)
        
        # Concatenate obs, flattened sequence, and time embedding
        input = jnp.concatenate((obs, seq_flat, te), axis=-1)
        
        # Output scalar energy
        return mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True)(input)


def mlp(hidden_sizes: Sequence[int], output_size: int, activation: Activation, output_activation: Activation, *, squeeze_output: bool = False) -> Callable[[jax.Array], jax.Array]:
    layers = []
    for hidden_size in hidden_sizes:
        layers += [hk.Linear(hidden_size), activation]
    layers += [hk.Linear(output_size), output_activation]
    if squeeze_output:
        layers.append(partial(jnp.squeeze, axis=-1))
    return hk.Sequential(layers)


def scaled_sinusoidal_encoding(t: jax.Array, *, dim: int, theta: int = 10000, batch_shape = None) -> jax.Array:
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
