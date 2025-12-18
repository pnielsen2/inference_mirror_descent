from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import haiku as hk
import math

from relax.network.blocks import (
    Activation,
    Identity,
    mlp,
    scaled_sinusoidal_encoding,
    DACERPolicyNet,
    EnergyPolicyNet,
    SequencePolicyNet,
    EnergySequencePolicyNet,
)
from relax.utils.diffusion import GaussianDiffusion


class ModelBasedParams(NamedTuple):
    policy: hk.Params
    target_policy: hk.Params
    dynamics: hk.Params
    reward: hk.Params
    value: hk.Params
    target_value: hk.Params
    log_alpha: jax.Array
    # Optional: joint sequence policy params (when joint_seq=True and H>1)
    seq_policy: Optional[hk.Params] = None


@dataclass
class ModelBasedNet:
    # policy(params, obs, act, t, h) -> noise prediction (per-step)
    policy: Callable[..., jax.Array]
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
    energy_mode: bool = False
    # energy_fn(params, obs, act, t, h) -> scalar energy (optional, per-step)
    energy_fn: Optional[Callable[..., jax.Array]] = None
    # Joint sequence mode fields (when joint_seq=True and H>1)
    joint_seq: bool = False
    # seq_policy(params, obs, seq_act, t) -> noise [B, H, act_dim] (joint sequence)
    seq_policy: Optional[Callable[..., jax.Array]] = None
    # seq_energy_fn(params, obs, seq_act, t) -> scalar energy [B] (joint sequence)
    seq_energy_fn: Optional[Callable[..., jax.Array]] = None

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
        """Deterministic dynamics model.

        Treats the leading dimensions of ``s`` as an arbitrary batch shape and
        broadcasts ``a``, ``s_t``, and ``t`` to match. This makes the
        transition model robust to extra batch axes introduced by nested
        ``scan``/``grad`` transformations (e.g., diffusion time or MALA
        iteration indices), while preserving the standard behavior when inputs
        are simple 2D batches.
        """

        # Batch shape: all leading dims except the feature dim.
        batch_shape = s.shape[:-1]

        # Broadcast action, target state, and timesteps to match ``s``'s batch
        # shape. This is a no-op in the common 2D case, but makes the
        # transition model robust to extra batch axes introduced by nested
        # transformations.
        a = jnp.broadcast_to(a, batch_shape + (a.shape[-1],))
        s_t = jnp.broadcast_to(s_t, s.shape)
        t = jnp.broadcast_to(t, batch_shape)

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
    energy_param: bool = False,
    H_train: int = 1,
    joint_seq: bool = False,
) -> Tuple[ModelBasedNet, ModelBasedParams]:

    # Determine if we should use joint sequence mode
    use_joint_seq = joint_seq and H_train > 1

    if energy_param:
        # Energy parameterization for the policy: network outputs scalar energy E(s, a_t, t, h)
        # and the diffusion denoiser is the gradient of this energy w.r.t. action, as in
        # Diffv2Net.energy_mode.
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

        def policy_apply(
            params: hk.Params,
            obs: jax.Array,
            act: jax.Array,
            t: jax.Array,
            h: jax.Array,
        ) -> jax.Array:
            """Apply policy with required horizon index h (use 0 for single-step)."""
            return jax.grad(lambda a: policy.apply(params, obs, a, t, h).sum())(act)

        def energy_apply_fn(
            params: hk.Params,
            obs: jax.Array,
            act: jax.Array,
            t: jax.Array,
            h: jax.Array,
        ) -> jax.Array:
            """Apply energy function with required horizon index h (use 0 for single-step)."""
            return policy.apply(params, obs, act, t, h)

        energy_apply: Optional[
            Callable[[hk.Params, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]
        ] = energy_apply_fn
    else:
        # Standard DACER-style diffusion denoiser.
        policy = hk.without_apply_rng(
            hk.transform(
                lambda obs, act, t, h: DACERPolicyNet(diffusion_hidden_sizes, activation)(
                    obs,
                    act,
                    t,
                    h,
                )
            )
        )

        def policy_apply(
            params: hk.Params,
            obs: jax.Array,
            act: jax.Array,
            t: jax.Array,
            h: jax.Array,
        ) -> jax.Array:
            """Apply policy with required horizon index h (use 0 for single-step)."""
            return policy.apply(params, obs, act, t, h)

        energy_apply = None

    # Joint sequence policy (when joint_seq=True and H_train > 1)
    if use_joint_seq:
        if energy_param:
            # Energy-based joint sequence denoiser
            seq_policy_net = hk.without_apply_rng(
                hk.transform(
                    lambda obs, seq_act, t: EnergySequencePolicyNet(
                        diffusion_hidden_sizes, H_train, activation
                    )(obs, seq_act, t)
                )
            )

            def seq_policy_apply(
                params: hk.Params,
                obs: jax.Array,
                seq_act: jax.Array,
                t: jax.Array,
            ) -> jax.Array:
                """Apply joint sequence policy; returns gradient of energy."""
                return jax.grad(lambda a: seq_policy_net.apply(params, obs, a, t).sum())(seq_act)

            def seq_energy_apply(
                params: hk.Params,
                obs: jax.Array,
                seq_act: jax.Array,
                t: jax.Array,
            ) -> jax.Array:
                """Apply joint sequence energy function."""
                return seq_policy_net.apply(params, obs, seq_act, t)
        else:
            # Standard joint sequence denoiser
            seq_policy_net = hk.without_apply_rng(
                hk.transform(
                    lambda obs, seq_act, t: SequencePolicyNet(
                        diffusion_hidden_sizes, H_train, activation
                    )(obs, seq_act, t)
                )
            )

            def seq_policy_apply(
                params: hk.Params,
                obs: jax.Array,
                seq_act: jax.Array,
                t: jax.Array,
            ) -> jax.Array:
                """Apply joint sequence policy."""
                return seq_policy_net.apply(params, obs, seq_act, t)

            seq_energy_apply = None
    else:
        seq_policy_net = None
        seq_policy_apply = None
        seq_energy_apply = None

    dynamics = hk.without_apply_rng(hk.transform(_dyn_forward(hidden_sizes, activation, time_dim=16)))
    reward = hk.without_apply_rng(hk.transform(_reward_forward(hidden_sizes, activation)))
    value = hk.without_apply_rng(hk.transform(_value_forward(hidden_sizes, activation)))

    @jax.jit
    def init(key, obs, act):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        sample_s = jnp.zeros((1, obs_dim))
        sample_a = jnp.zeros((1, act_dim))
        sample_t = jnp.zeros((1,))
        sample_h = jnp.zeros((1,))  # Always use h for consistent structure

        p_policy = policy.init(k1, sample_s, sample_a, sample_t, sample_h)
        p_dyn = dynamics.init(k2, sample_s, sample_a, sample_s, sample_t)
        p_rew = reward.init(k3, sample_s, sample_a, sample_s)
        p_val = value.init(k4, sample_s)
        p_val_targ = p_val

        # Initialize sequence policy if using joint sequence mode
        if use_joint_seq:
            sample_seq_a = jnp.zeros((1, H_train, act_dim))
            p_seq_policy = seq_policy_net.init(k5, sample_s, sample_seq_a, sample_t)
        else:
            p_seq_policy = None

        log_alpha = jnp.array(math.log(5.0), dtype=jnp.float32)

        return ModelBasedParams(
            policy=p_policy,
            target_policy=p_policy,
            dynamics=p_dyn,
            reward=p_rew,
            value=p_val,
            target_value=p_val_targ,
            log_alpha=log_alpha,
            seq_policy=p_seq_policy,
        )

    params = init(key, jnp.zeros((1, obs_dim)), jnp.zeros((1, act_dim)))

    net = ModelBasedNet(
        policy=policy_apply,
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
        energy_mode=energy_param,
        energy_fn=energy_apply,
        joint_seq=use_joint_seq,
        seq_policy=seq_policy_apply,
        seq_energy_fn=seq_energy_apply,
    )

    return net, params
