"""Actor-critic bundle used by MGMD.

Holds the haiku-applied callables for:

* the Q ensemble (``q``),
* the energy-based diffusion policy, exposed as a noise predictor
  ``eps_pred(params, s, a, t) = ε̂(s, a, t)`` and as the presentation
  scalar energy ``energy_fn(params, s, a, t) = E(s, a, t)``,

plus the precomputed DDPM noise schedule (``schedule``) and a handful of
scalar config fields (``num_timesteps``, ``act_dim``, ``x_recon_clip_radius``,
``mala_steps``) that downstream code reads directly.

The classmethod :meth:`ActorCritic.create` is the single entry point;
it builds the beta schedule and constructs :attr:`energy_fn` and
:attr:`eps_pred` callables for the chosen ``policy_parameterization``.
"""
from dataclasses import dataclass
from typing import NamedTuple, Sequence, Tuple, Optional, Any

import jax, jax.numpy as jnp
import haiku as hk

from relax.network.blocks import Activation, Identity, QNet, mlp, scaled_sinusoidal_encoding
from relax.utils.diffusion import BetaScheduleCoefficients, build_beta_schedule
from relax.utils.jax_utils import fix_repr


# ---------------------------------------------------------------------------
# Scalar policy network. Outputs a raw scalar field interpreted as either
# E_theta or f_theta depending on policy_parameterization; ActorCritic.energy_fn
# and ActorCritic.eps_pred (constructed at create() time) are the public interface.
# ---------------------------------------------------------------------------
@dataclass
@fix_repr
class ScalarPolicyNet(hk.Module):
    hidden_sizes: Sequence[int]
    activation: Activation
    output_activation: Activation = Identity
    time_dim: int = 16
    policy_final_layer: str = "default"
    name: str = None

    def __call__(
        self,
        obs: jax.Array,
        act: jax.Array,
        t: jax.Array,
    ) -> jax.Array:
        """Forward pass.

        Args:
            obs: Observation, shape [..., obs_dim]
            act: Action, shape [..., act_dim]
            t: Diffusion timestep
        """
        te = scaled_sinusoidal_encoding(t, dim=self.time_dim, batch_shape=obs.shape[:-1])
        te = hk.Linear(self.time_dim * 2)(te)
        te = self.activation(te)
        te = hk.Linear(self.time_dim)(te)
        input = jnp.concatenate((obs, act, te), axis=-1)
        if self.policy_final_layer == "default":
            return mlp(self.hidden_sizes, 1, self.activation, self.output_activation)(input)[..., 0]
        v = mlp(self.hidden_sizes, act.shape[-1], self.activation, self.output_activation)(input)
        if self.policy_final_layer == "ff":
            return hk.Linear(1)(v)[..., 0]
        if self.policy_final_layer == "L2":
            return -0.5 * jnp.sum(v ** 2, axis=-1)
        return jnp.sum(v * act, axis=-1)  # "IP"


class ActorCriticParams(NamedTuple):
    q: Tuple  # tuple of N Q network params
    target_q: Tuple  # tuple of N target Q network params
    policy: hk.Params


@dataclass
class ActorCritic:
    """Actor-critic bundle: Q ensemble + energy-based diffusion policy + DDPM schedule.

    The two haiku-transformed networks (``_q_net`` and ``_policy_scalar``) are
    stored as private fields. Public method access goes through :meth:`q`
    and :meth:`q_sample`; so a reader can ``grep "def q"`` and land on the
    actual implementation.
    :attr:`energy_fn` and :attr:`eps_pred` are callable fields constructed
    at :meth:`create` time.

    Constructed via :meth:`ActorCritic.create` (architecture-only,
    seed-independent); fresh haiku params are sampled per-seed via
    :meth:`init_params`. One ``ActorCritic`` instance can serve any
    number of seeds since the haiku transforms are pure functions of
    their params.
    """
    _q_net: Any                     # hk.Transformed for the Q ensemble member
    _policy_scalar: Any             # hk.Transformed raw scalar output network; used by init_params
    energy_fn: Any                  # (params, obs, act, t) -> scalar E_theta; set at create() time
    eps_pred: Any                   # (params, obs, act, t) -> action-shaped ε̂; set at create() time
    schedule: BetaScheduleCoefficients
    num_timesteps: int
    obs_dim: int
    act_dim: int
    num_q_networks: int
    policy_parameterization: str = "E"
    policy_final_layer: str = "default"
    x_recon_clip_radius: Optional[float] = 1.0
    mala_steps: int = 1

    def q(self, params: hk.Params, obs: jax.Array, act: jax.Array) -> jax.Array:
        """Q(s, a) for a single ensemble member's params."""
        return self._q_net.apply(params, obs, act)

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
        policy_parameterization: str = "E",
        policy_final_layer: str = "default",
    ) -> "ActorCritic":
        """Build the architecture: haiku transforms + DDPM schedule.

        Seed-independent. One :class:`ActorCritic` instance can be reused
        across any number of seeds; fresh per-seed params are sampled
        with :meth:`init_params`.
        """
        q_net = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))

        policy_scalar = hk.without_apply_rng(
            hk.transform(
                lambda obs, act, t: ScalarPolicyNet(diffusion_hidden_sizes, activation, policy_final_layer=policy_final_layer)(
                    obs,
                    act,
                    t,
                )
            )
        )

        schedule = build_beta_schedule(
            num_timesteps=num_timesteps,
            beta_schedule_type=beta_schedule_type,
            snr_max=snr_max,
        )

        def raw_scalar(params, obs, act, t):
            return policy_scalar.apply(params, obs, act, t)

        if policy_parameterization == "E":
            energy_fn = raw_scalar
            def eps_pred(params, obs, act, t):
                energy_grad = jax.grad(lambda a: raw_scalar(params, obs, a, t).sum())(act)
                sigma_t = jnp.expand_dims(schedule.sqrt_one_minus_alphas_cumprod[t], axis=-1)
                return sigma_t * energy_grad
        else:  # "f"
            def energy_fn(params, obs, act, t):
                return raw_scalar(params, obs, act, t) / schedule.sqrt_one_minus_alphas_cumprod[t]
            def eps_pred(params, obs, act, t):
                return jax.grad(lambda a: raw_scalar(params, obs, a, t).sum())(act)

        return ActorCritic(
            _q_net=q_net,
            _policy_scalar=policy_scalar,
            energy_fn=energy_fn,
            eps_pred=eps_pred,
            schedule=schedule,
            num_timesteps=num_timesteps,
            obs_dim=obs_dim,
            act_dim=act_dim,
            num_q_networks=int(num_q_networks),
            policy_parameterization=policy_parameterization,
            policy_final_layer=policy_final_layer,
            x_recon_clip_radius=x_recon_clip_radius,
            mala_steps=mala_steps,
        )

    def init_params(self, key: jax.Array) -> "ActorCriticParams":
        """Sample fresh haiku params for one seed.

        Builds the Q-ensemble params (and a deepcopy as the target-Q
        params) plus the scalar policy params. The PRNG split is
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
            policy_params = self._policy_scalar.init(
                policy_key, sample_obs, sample_act, 0
            )
            return ActorCriticParams(q_params, target_q_params, policy_params)
        return _init(key)
