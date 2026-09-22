"""Unguided samplers for the *base* diffusion policy.

The conventional ways to draw an action from a diffusion policy: sample
``x_T ~ N(0, I)`` and take ``timesteps`` transitions driven by the policy's own
eps prediction, with no MCMC and no Q guidance. These are the samplers MGMD's
MALA chain replaces, and the reference an eta = 0 (beta = 0) MALA chain has to be
compared against when the question is *how* the base policy is sampled rather
than *which* density is sampled.

Two transitions, selected by ``transition``:

* ``"ddim"`` -- the deterministic DDIM update, no x0 clip.
* ``"ddpm_mean"`` -- the DDPM posterior mean with the Tweedie x0 clipped to
  ``model.x_recon_clip_radius`` (1 for a normalized action box). This is DIPO's
  evaluation sampler: ``Diffusion.sample(state, eval=True)`` sets
  ``noise_ratio = 0``, which drops the ancestral noise from every ``p_sample``
  and leaves exactly this mean, over ``clip_denoised=True`` x0 predictions, with
  one action per state and no critic involvement.

Each is exactly what :func:`relax.algorithm.mala_sampler.build_mala_sampler`
produces at the matching ``denoising_predictor`` with ``mala_steps=0``,
``beta=0``, ``alpha=1`` (asserted bit-for-bit in
``scripts/test_base_policy_samplers.py``), but at a fraction of the cost: that
build still evaluates the guidance gradient -- a backward pass through the Q net
*and* the policy net -- at every level before multiplying it by a traced beta of
zero, which XLA cannot fold away. Here the only work per level is the single eps
prediction the transition consumes, which is what makes a best-of-N sweep to
N = 1024 affordable.

Returns the same :class:`MalaSampleResult` bundle as the MALA sampler so all of
them can sit behind one best-of-N read-out (``MGMD.get_eval_action_vmap``), with
the MALA-only diagnostics left at their inputs: nothing here adapts a step size
and no level runs an accept/reject.
"""
import jax
import jax.numpy as jnp

from relax.algorithm.mala_sampler import ddim_from_eps, ddpm_mean_from_eps
from relax.algorithm.mgmd_types import Diffv2TrainState, MalaSampleResult
from relax.network.actor_critic import ActorCritic
from relax.utils.diffusion import NoiseLevel

TRANSITIONS = ("ddim", "ddpm_mean")


def build_base_policy_sampler(
    *,
    model: ActorCritic,
    timesteps: int,
    transition: str = "ddim",
    num_denoised_actions: int = 1,
    latent_action: bool = False,
    compute_final_q: bool = True,
):
    """Return ``stateless_get_action_base(key, state, obs, aggregate_q_fn)``.

    Mirrors the MALA sampler's signature and jit/vmap-ability. ``state`` is read
    for the policy params, the Q params and the per-seed noise schedule only:
    ``hp.alpha`` / ``hp.eta`` / ``state.beta`` do not enter an unguided chain, so
    the same state can be handed to either sampler.
    """
    if transition not in TRANSITIONS:
        raise ValueError(f"Unknown transition: {transition!r}; expected one of {TRANSITIONS}")

    def stateless_get_action_base(
        key: jax.Array,
        state: Diffv2TrainState,
        obs: jax.Array,
        aggregate_q_fn,
    ) -> MalaSampleResult:
        policy_params = state.params.policy
        q_params_tuple = state.params.q
        # Denoise K actions per state, iid: the leading [K] axis on obs is what
        # the best-of-N read-out ranks over. Same layout as the MALA sampler.
        K = num_denoised_actions
        obs = jnp.broadcast_to(obs, (K, *obs.shape))
        action_shape = (*obs.shape[:-1], model.act_dim)
        # Split even though both loops are deterministic, so x_T is the same draw
        # the MALA sampler would have started from at this key.
        key_x, _ = jax.random.split(key, 2)
        schedule = model.schedule_for(state.hp, state.log_snr_levels)

        def step(i, x):
            t_idx = timesteps - 1 - i
            level = NoiseLevel.at(schedule, t_idx)
            eps_pred = model.eps_pred(policy_params, level, obs, x)
            if transition == "ddim":
                return ddim_from_eps(schedule, t_idx, x, eps_pred)
            return ddpm_mean_from_eps(schedule, level, t_idx, x, eps_pred,
                                      model.x_recon_clip_radius)

        x_0 = jax.lax.fori_loop(0, timesteps, step, jax.random.normal(key_x, action_shape))
        # In latent-action mode the endpoint is an unbounded latent; the env-side
        # Gaussian-CDF squash bounds it. Otherwise clip to the normalized box.
        act_final = x_0 if latent_action else jnp.clip(x_0, -1.0, 1.0)
        if compute_final_q:
            final_q = aggregate_q_fn([model.q(qp, obs, act_final) for qp in q_params_tuple])
        else:
            final_q = jnp.zeros(obs.shape[:-1])
        zeros_per_level = jnp.zeros((timesteps,), dtype=jnp.float32)
        return MalaSampleResult(
            action=act_final, q=final_q, log_eta_scales=state.log_eta_scales,
            per_level_acc=zeros_per_level, per_level_clip=zeros_per_level,
        )

    return stateless_get_action_base
