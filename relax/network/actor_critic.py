"""Actor-critic bundle used by DPMD.

Holds the haiku-applied callables for:

* the Q ensemble (``q``),
* the energy-based diffusion policy, exposed both as a noise/score predictor
  ``policy(params, s, a, t) = grad_a E(s, a, t)`` and as the raw scalar
  energy ``energy_fn(params, s, a, t) = E(s, a, t)``,

plus the precomputed DDPM noise schedule (``schedule``) and a handful of
scalar config fields (``num_timesteps``, ``act_dim``, ``x_recon_clip_radius``,
``mala_steps``) that downstream code reads directly.

The classmethod :meth:`ActorCritic.create` is the single entry point;
it initialises haiku params, eagerly builds the beta schedule, and wraps
the energy MLP with a :func:`jax.grad` so the :meth:`policy` method
returns the noise prediction used in the diffusion loss.
"""
from dataclasses import dataclass
from typing import NamedTuple, Sequence, Tuple, Optional, Any

import jax, jax.numpy as jnp
import haiku as hk

from relax.network.blocks import Activation, Identity, QNet, mlp, scaled_sinusoidal_encoding
from relax.utils.diffusion import BetaScheduleCoefficients, build_beta_schedule
from relax.utils.jax_utils import fix_repr


# ---------------------------------------------------------------------------
# Energy MLP. Outputs a scalar E(s, a, t, h); the noise / score prediction
# used everywhere in this codebase is grad_a E(s, a, t, h=0). See
# ``ActorCritic.policy`` below for the gradient wrapper.
# ---------------------------------------------------------------------------
@dataclass
@fix_repr
class EnergyPolicyNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    time_dim: int = 16
    horizon_dim: int = 8
    zero_init_final: bool = False
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
        return mlp(self.hidden_sizes, 1, self.activation, self.output_activation, squeeze_output=True, zero_init_final=self.zero_init_final)(input)


class ActorCriticParams(NamedTuple):
    q: Tuple  # tuple of N Q network params
    target_q: Tuple  # tuple of N target Q network params
    policy: hk.Params


@dataclass
class ActorCritic:
    """Actor-critic bundle: Q ensemble + energy-based diffusion policy + DDPM schedule.

    The two haiku-transformed networks (``_q_net`` and ``_energy_net``) are
    stored as private fields. Public access goes through the methods
    :meth:`q`, :meth:`policy`, :meth:`energy_fn`, and :meth:`q_sample`,
    so a reader can ``grep "def policy"`` and land on the actual
    implementation rather than chasing a Callable field through a closure.

    Constructed via :meth:`ActorCritic.create` (architecture-only,
    seed-independent); fresh haiku params are sampled per-seed via
    :meth:`init_params`. One ``ActorCritic`` instance can serve any
    number of seeds since the haiku transforms are pure functions of
    their params.
    """
    _q_net: Any                     # hk.Transformed for the Q ensemble member
    _energy_net: Any                # hk.Transformed for the energy MLP
    schedule: BetaScheduleCoefficients
    num_timesteps: int
    obs_dim: int
    act_dim: int
    num_q_networks: int
    x_recon_clip_radius: Optional[float] = 1.0
    mala_steps: int = 1

    def q(self, params: hk.Params, obs: jax.Array, act: jax.Array) -> jax.Array:
        """Q(s, a) for a single ensemble member's params."""
        return self._q_net.apply(params, obs, act)

    def energy_fn(self, params: hk.Params, obs: jax.Array, act: jax.Array, t: jax.Array) -> jax.Array:
        """Scalar energy E(s, a, t, h=0) of the diffusion policy."""
        h_zeros = jnp.zeros(obs.shape[:-1], dtype=jnp.float32)
        return self._energy_net.apply(params, obs, act, t, h_zeros)

    def policy(self, params: hk.Params, obs: jax.Array, act: jax.Array, t: jax.Array) -> jax.Array:
        """Noise / score prediction ε̂(s, a, t) = grad_a E(s, a, t, h=0).

        This is the eps-prediction head used by the diffusion loss in
        ``DPMD.policy_loss_fn`` and by the MALA sampler's predictor step.
        Implemented as the gradient of :meth:`energy_fn` w.r.t. ``act`` so
        the energy MLP is the single source of truth for the policy.
        """
        return jax.grad(lambda a: self.energy_fn(params, obs, a, t).sum())(act)

    def q_sample(self, t: jax.Array, x_0: jax.Array, noise: jax.Array) -> jax.Array:
        """Forward diffusion q(x_t | x_0): x_t = sqrt(ṱ_t)·x_0 + sqrt(1-ṱ_t)·ε.

        Shapes: ``t`` is ``(B,)`` and ``x_0`` / ``noise`` are ``(B, act_dim)``.
        """
        sqrt_ac = self.schedule.sqrt_alphas_cumprod[t][:, None]
        sqrt_omac = self.schedule.sqrt_one_minus_alphas_cumprod[t][:, None]
        return sqrt_ac * x_0 + sqrt_omac * noise

    @staticmethod
    def create(
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int],
        diffusion_hidden_sizes: Sequence[int],
        activation: Activation = jax.nn.relu,
        num_timesteps: int = 20,
        beta_schedule_type: str = "linear",
        mala_steps: int = 1,
        x_recon_clip_radius: Optional[float] = 1.0,
        snr_max: float = 124.0,
        num_q_networks: int = 2,
    ) -> "ActorCritic":
        """Build the architecture: haiku transforms + DDPM schedule.

        Seed-independent. One :class:`ActorCritic` instance can be reused
        across any number of seeds; fresh per-seed params are sampled
        with :meth:`init_params`.
        """
        q_net = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))

        energy_net = hk.without_apply_rng(
            hk.transform(
                lambda obs, act, t, h: EnergyPolicyNet(diffusion_hidden_sizes, activation)(
                    obs,
                    act,
                    t,
                    h,
                )
            )
        )

        schedule = build_beta_schedule(
            num_timesteps=num_timesteps,
            beta_schedule_type=beta_schedule_type,
            snr_max=snr_max,
        )

        return ActorCritic(
            _q_net=q_net,
            _energy_net=energy_net,
            schedule=schedule,
            num_timesteps=num_timesteps,
            obs_dim=obs_dim,
            act_dim=act_dim,
            num_q_networks=int(num_q_networks),
            x_recon_clip_radius=x_recon_clip_radius,
            mala_steps=mala_steps,
        )

    def init_params(self, key: jax.Array) -> "ActorCriticParams":
        """Sample fresh haiku params for one seed.

        Builds the Q-ensemble params (and a deepcopy as the target-Q
        params) plus the energy-policy params. The PRNG split is
        ``num_q_networks + 1`` keys: one per Q net, plus one for the
        policy. Mirrors the original ``create_actor_critic`` init layout
        for bit-exact PRNG compatibility.
        """
        @jax.jit
        def _init(k):
            sample_obs = jnp.zeros((1, self.obs_dim))
            sample_act = jnp.zeros((1, self.act_dim))
            keys = jax.random.split(k, self.num_q_networks + 1)
            policy_key = keys[-1]
            q_params = tuple(
                self._q_net.init(keys[i], sample_obs, sample_act)
                for i in range(self.num_q_networks)
            )
            target_q_params = tuple(jax.tree.map(lambda x: x, qp) for qp in q_params)
            policy_params = self._energy_net.init(
                policy_key, sample_obs, sample_act, 0, jnp.zeros((1,), dtype=jnp.float32)
            )
            return ActorCriticParams(q_params, target_q_params, policy_params)
        return _init(key)
