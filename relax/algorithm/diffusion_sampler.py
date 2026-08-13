"""Unguided DDPM reverse sampler used by the MGMD RSM variant."""
from typing import Callable

import jax
import jax.numpy as jnp

from relax.algorithm.mgmd_types import Diffv2TrainState, MalaSampleResult
from relax.network.actor_critic import ActorCritic
from relax.utils.diffusion import p_mean_variance


def build_diffusion_sampler(
    *,
    model: ActorCritic,
    timesteps: int,
    num_denoised_actions: int = 1,
    compute_final_q: bool = True,
) -> Callable:
    """Return a jit-able DDPM sampler with the same interface as MALA sampler.

    This intentionally mirrors diffusion_policy_online_rl's plain diffusion
    sampler: start from ``0.5 * N(0, I)``, run stochastic DDPM reverse steps,
    and clip the reconstructed clean action inside ``p_mean_variance`` using
    the model's diffusion reconstruction clip radius.
    """
    def stateless_get_action_diffusion(
        key: jax.Array,
        state: Diffv2TrainState,
        obs: jax.Array,
        aggregate_q_fn,
    ) -> MalaSampleResult:
        policy_params = state.params.policy
        q_params_tuple = state.params.q

        K = num_denoised_actions
        obs = jnp.broadcast_to(obs, (K, *obs.shape))
        action_shape = (*obs.shape[:-1], model.act_dim)

        x_key, noise_key = jax.random.split(key)
        x = jnp.float32(0.5) * jax.random.normal(x_key, action_shape)
        reverse_noise = jax.random.normal(noise_key, (timesteps, *action_shape))

        def body_fn(x_curr, inputs):
            t_idx, noise_t = inputs
            eps_pred = model.eps_pred(policy_params, obs, x_curr, t_idx)
            model_mean, model_log_variance = p_mean_variance(
                model.schedule,
                t_idx,
                x_curr,
                eps_pred,
                model.x_recon_clip_radius,
            )
            x_next = model_mean + (t_idx > 0) * jnp.exp(jnp.float32(0.5) * model_log_variance) * noise_t
            return x_next, None

        t = jnp.arange(timesteps)[::-1]
        action, _ = jax.lax.scan(body_fn, x, (t, reverse_noise))

        if compute_final_q:
            q = aggregate_q_fn([model.q(qp, obs, action) for qp in q_params_tuple])
        else:
            q = jnp.zeros(action.shape[:-1], dtype=action.dtype)

        zeros = jnp.zeros((timesteps,), dtype=action.dtype)
        return MalaSampleResult(
            action=action,
            q=q,
            log_eta_scales=state.log_eta_scales,
            per_level_acc=zeros,
            per_level_clip=zeros,
        )

    return stateless_get_action_diffusion
