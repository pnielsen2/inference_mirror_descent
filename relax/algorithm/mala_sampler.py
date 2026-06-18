"""MALA-corrected reverse-diffusion sampler used by MGMD.

One full pass of the sampler:

* Sample x_T ~ N(0, I), then for t = T-1 .. 0:
    1. ``mala_steps`` MALA correction steps targeting
       ``E_total(t, x) = alpha * E_θ(s, x, t) - beta * Q_agg(s, x_0_hat)``.
       (Q_agg is omitted when ``beta == 0`` so zero-guidance
       runs are unbiased.)
    2. The selected denoising predictor step.
* Return ``MalaSampleResult(action, q, log_eta_scales, per_level_acc,
  per_level_clip)`` with the per-level acceptance / clip arrays the trainer
  logs to wandb.
"""
from typing import Callable

import jax
import jax.numpy as jnp

from relax.algorithm.mgmd_types import Diffv2TrainState, MalaSampleResult
from relax.network.actor_critic import ActorCritic


def build_mala_sampler(
    *,
    model: ActorCritic,
    value_head,                    # ValueHead | None — supplied iff on-policy-EMA / KL-budget mode
    timesteps: int,
    batch_independent_guidance: bool,
    compute_final_q: bool = True,
    advantage_normalization: bool,
    denoising_predictor: str,
    guidance_gradient_space: str,
) -> Callable:
    """Return ``stateless_get_action_mala_full(key, state, obs, aggregate_q_fn)``.

    The returned closure is jit-able and vmap-able. ``aggregate_q_fn`` is
    supplied per-call; both the rollout path and the TD next-action sampling
    path pass ``--q_agg_sample``-aggregation, letting the same sampler serve
    both call sites.
    """
    def stateless_get_action_mala_full(
        key: jax.Array,
        state: Diffv2TrainState,
        obs: jax.Array,
        aggregate_q_fn,
    ) -> MalaSampleResult:
        policy_params = state.params.policy
        q_params_tuple = state.params.q
        log_eta_scales_in = state.log_eta_scales
        beta_cmd = state.beta
        value_params = state.value_params
        x0_hat_clip_radius = state.hp.x0_hat_clip_radius
        mala_adapt_rate = state.hp.mala_adapt_rate
        guidance_multiplier = state.hp.guidance_mult

        action_shape = (*obs.shape[:-1], model.act_dim)

        key_x, loop_key = jax.random.split(key, 2)
        schedule = model.schedule
        x_recon_clip_radius = model.x_recon_clip_radius
        reduce_over_batch = jnp.sum if batch_independent_guidance else jnp.mean
        if advantage_normalization:
            beta_current = beta_cmd / jnp.sqrt(jnp.maximum(state.advantage_second_moment_ema, jnp.float32(1e-6)))
        else:
            beta_current = beta_cmd

        # ---- Shared Tweedie building blocks -------------------------
        def reconstruct_x0_from_noise(x_in, t_idx, noise_pred):
            return (
                x_in * schedule.sqrt_recip_alphas_cumprod[t_idx]
                - noise_pred * schedule.sqrt_recipm1_alphas_cumprod[t_idx]
            )

        def q_aggregated_at_clipped_x0_hat(x0_hat):
            x0_clipped = jnp.clip(x0_hat, -x0_hat_clip_radius, x0_hat_clip_radius)
            return aggregate_q_fn([model.q(qp, obs, x0_clipped) for qp in q_params_tuple])

        # beta_current = β: fixed constant, sqrt(2δ/M) (KL-budget), or
        # min(β_KL, β*) (one-step dist-shift), depending on run config.
        def energy_total(t, x):
            E_vals, vjp_fn = jax.vjp(lambda a: model.energy_fn(policy_params, obs, a, t), x)
            (e_grad,) = vjp_fn(jnp.ones_like(E_vals))
            noise_pred = schedule.sqrt_one_minus_alphas_cumprod[t] * e_grad  # ε̂ = σ_t ∇_x E_θ
            x0_hat = reconstruct_x0_from_noise(x, t, noise_pred)
            clip_frac = jnp.mean((jnp.abs(x0_hat) > x0_hat_clip_radius).astype(jnp.float32))
            return state.hp.alpha * E_vals - beta_current * q_aggregated_at_clipped_x0_hat(x0_hat), clip_frac

        def guidance_value_from_x(x_in, t_idx):
            # Tweedie-clean prediction with alpha-scaled base
            # score; guidance component is NOT scaled.
            eps_pred = model.eps_pred(policy_params, obs, x_in, t_idx)
            x0_hat = reconstruct_x0_from_noise(x_in, t_idx, eps_pred)
            q = q_aggregated_at_clipped_x0_hat(x0_hat)

            # KL-budget mode adapts beta from advantage-moment statistics,
            # but the sampling objective here still uses the same clipped-Q
            # guidance term as the fixed-beta path.
            return guidance_multiplier * reduce_over_batch(q)

        def compute_guidance_gradient(x_in, t_idx):
            if guidance_gradient_space == "xt":
                return jax.grad(lambda x: guidance_value_from_x(x, t_idx))(x_in)

            eps_pred = model.eps_pred(policy_params, obs, x_in, t_idx)
            x0_hat = reconstruct_x0_from_noise(x_in, t_idx, eps_pred)
            if guidance_gradient_space == "x0hatclipped":
                x0_hat = jnp.clip(x0_hat, -x0_hat_clip_radius, x0_hat_clip_radius)
            return jax.grad(
                lambda action: guidance_multiplier * reduce_over_batch(
                    aggregate_q_fn([model.q(qp, obs, action) for qp in q_params_tuple])
                )
            )(jax.lax.stop_gradient(x0_hat))

        def jacobian_free_energy_and_drift(t_idx, x):
            E_vals, vjp_fn = jax.vjp(lambda a: model.energy_fn(policy_params, obs, a, t_idx), x)
            (e_grad,) = vjp_fn(jnp.ones_like(E_vals))
            noise_pred = schedule.sqrt_one_minus_alphas_cumprod[t_idx] * e_grad
            x0_hat = reconstruct_x0_from_noise(x, t_idx, noise_pred)
            x0_eval = (
                jnp.clip(x0_hat, -x0_hat_clip_radius, x0_hat_clip_radius)
                if guidance_gradient_space == "x0hatclipped"
                else x0_hat
            )
            q = aggregate_q_fn([model.q(qp, obs, x0_eval) for qp in q_params_tuple])
            grad_q = jax.grad(
                lambda action: jnp.sum(
                    aggregate_q_fn([model.q(qp, obs, action) for qp in q_params_tuple])
                )
            )(jax.lax.stop_gradient(x0_eval))
            energy = state.hp.alpha * E_vals - beta_current * q
            grad_energy = state.hp.alpha * e_grad - beta_current * grad_q
            clip_frac = jnp.mean((jnp.abs(x0_hat) > x0_hat_clip_radius).astype(jnp.float32))
            return energy, grad_energy, clip_frac

        # ---- Denoising predictor step (chosen at build time) -----------------
        def guided_eps_pred(t_idx, x_in):
            noise_pred_scaled = state.hp.alpha * model.eps_pred(policy_params, obs, x_in, t_idx)
            grad_q = compute_guidance_gradient(x_in, t_idx)
            sigma_t = schedule.sqrt_one_minus_alphas_cumprod[t_idx]
            return noise_pred_scaled - beta_current * sigma_t * grad_q

        def ddpm_mean_step(t_idx, x_in):
            eps_pred = guided_eps_pred(t_idx, x_in)
            x0_hat = jnp.clip(reconstruct_x0_from_noise(x_in, t_idx, eps_pred),
                               -x_recon_clip_radius, x_recon_clip_radius)
            return x0_hat * schedule.posterior_mean_coef1[t_idx] + x_in * schedule.posterior_mean_coef2[t_idx]

        def ddim_step(t_idx, x_in):
            eps_pred = guided_eps_pred(t_idx, x_in)
            sqrt_ab_t = schedule.sqrt_alphas_cumprod[t_idx]
            sqrt_one_minus_ab_t = schedule.sqrt_one_minus_alphas_cumprod[t_idx]
            sqrt_ab_prev = jnp.sqrt(schedule.alphas_cumprod_prev[t_idx])
            sqrt_one_minus_ab_prev = jnp.sqrt(1.0 - schedule.alphas_cumprod_prev[t_idx])
            return (
                (sqrt_ab_prev / sqrt_ab_t) * x_in
                + (sqrt_one_minus_ab_prev - (sqrt_ab_prev / sqrt_ab_t) * sqrt_one_minus_ab_t) * eps_pred
            )

        if denoising_predictor == "Identity":
            def denoising_step(t_idx, x_curr):
                return x_curr
        elif denoising_predictor == "DDPM_mean":
            denoising_step = ddpm_mean_step
        elif denoising_predictor == "DDIM":
            denoising_step = ddim_step
        else:
            raise ValueError(f"Unknown denoising_predictor: {denoising_predictor}")

        # ---- MALA step-size scale clamp range (shared across all levels) -
        log_eta_min = jnp.log(jnp.float32(1e-8) / jnp.maximum(jnp.max(schedule.betas), jnp.float32(1e-8)))
        log_eta_max = jnp.log(jnp.float32(0.5) / jnp.maximum(jnp.min(schedule.betas), jnp.float32(1e-8)))

        # ---- Per-diffusion-level MALA correction + denoising predictor ----
        def run_mala_chain_at_level(t_idx, x_t, rng, log_eta_scales):
            eta_base_t = jnp.maximum(schedule.betas[t_idx], jnp.float32(1e-8))
            eta_upper = jnp.float32(0.5)

            def mala_body(_, state):
                x_current, rng_step, log_eta_scale, accept_rate_sum, clip_frac_sum = state
                # Langevin Dynamics transition for Metropolis-Hastings forward proposal
                # compute energy, vector-jacobian product, and clip_frac in one pass
                if guidance_gradient_space == "xt":
                    E_x, vjp_x, clip_x = jax.vjp(lambda x: energy_total(t_idx, x), x_current, has_aux=True)
                    grad_E_x = vjp_x(jnp.ones_like(E_x))[0]
                else:
                    E_x, grad_E_x, clip_x = jacobian_free_energy_and_drift(t_idx, x_current)
                step_size = jnp.clip(jnp.exp(log_eta_scale) * eta_base_t, jnp.float32(1e-8), eta_upper)

                proposal_mean = x_current - step_size * grad_E_x
                proposal_std = jnp.sqrt(jnp.float32(2.0) * step_size)

                rng_step, noise_key, u_key = jax.random.split(rng_step, 3)
                z = jax.random.normal(noise_key, x_current.shape)
                
                x_prop = proposal_mean + proposal_std * z
                # Langevin Dynamics reverse transition for Metropolis-Hastings acceptance
                # compute energy, vector-jacobian product, and clip_frac in one pass
                if guidance_gradient_space == "xt":
                    E_x_prop, vjp_x_prop, _clip_prop = jax.vjp(lambda xx: energy_total(t_idx, xx), x_prop, has_aux=True)
                    grad_E_x_prop = vjp_x_prop(jnp.ones_like(E_x_prop))[0]
                else:
                    E_x_prop, grad_E_x_prop, _clip_prop = jacobian_free_energy_and_drift(t_idx, x_prop)

                reverse_mean = x_prop - step_size * grad_E_x_prop

                def gaussian_log_density(x, mean):
                    diff = x - mean
                    return -jnp.sum(diff * diff, axis=-1) / (jnp.float32(4.0) * step_size)

                proposal_log_prob = gaussian_log_density(x_prop, proposal_mean)
                reverse_log_prob  = gaussian_log_density(x_current, reverse_mean)

                # MH log acceptance: log[π(x')q(x|x') / π(x)q(x'|x)]
                log_acceptance_ratio = (-E_x_prop + E_x) + (reverse_log_prob - proposal_log_prob)
                accept = jnp.log(jax.random.uniform(u_key, E_x.shape)) < jnp.minimum(jnp.float32(0.0), log_acceptance_ratio)

                x_next = jnp.where(accept[..., None], x_prop, x_current)

                acc_rate = jnp.mean(accept.astype(jnp.float32).reshape(-1))
                target = jnp.float32(0.574)  # optimal MALA acceptance rate in high dimensions (Roberts et al. 1997)
                log_eta_scale = log_eta_scale + mala_adapt_rate * (acc_rate - target)
                log_eta_scale = jnp.clip(log_eta_scale, log_eta_min, log_eta_max)

                return (
                    x_next,
                    rng_step,
                    log_eta_scale,
                    accept_rate_sum + acc_rate,
                    clip_frac_sum + clip_x,
                )

            mala_corrected_x_t, rng_out, log_eta_scale_new, acc_sum, clip_sum = jax.lax.fori_loop(
                0,
                model.mala_steps,
                mala_body,
                (x_t, rng, log_eta_scales[t_idx], jnp.float32(0.0), jnp.float32(0.0)),
            )
            log_eta_scales = log_eta_scales.at[t_idx].set(log_eta_scale_new)
            return mala_corrected_x_t, rng_out, log_eta_scales, acc_sum, clip_sum

        def process_one_noise_level(i, carry):
            # At noise level t: run MALA correction chain, then denoise to noise level t-1.
            x_t, rng, log_eta_scales, per_level_acc, per_level_clip_frac = carry
            t_idx = timesteps - 1 - i
        
            mala_corrected_x_t, rng, log_eta_scales, acc_sum, clip_sum = run_mala_chain_at_level(
                t_idx, x_t, rng, log_eta_scales,
            )
            x_t_minus_1 = denoising_step(t_idx, mala_corrected_x_t)

            # ---- per-level diagnostics ----
            mala_steps_f = jnp.float32(model.mala_steps)
            per_level_acc     = per_level_acc.at[t_idx].set(acc_sum / jnp.maximum(mala_steps_f, jnp.float32(1.0)))
            per_level_clip_frac = per_level_clip_frac.at[t_idx].set(clip_sum / jnp.maximum(mala_steps_f, jnp.float32(1.0)))
            return x_t_minus_1, rng, log_eta_scales, per_level_acc, per_level_clip_frac

        # ---- Drive the reverse diffusion: T → 0 over the level loop -----
        x_T = jax.random.normal(key_x, action_shape)
        init_per_level_acc = jnp.zeros((timesteps,), dtype=jnp.float32)
        init_per_level_clip_frac = jnp.zeros((timesteps,), dtype=jnp.float32)
        x_0, _, log_eta_scales_out, per_level_acc_out, per_level_clip_frac_out = jax.lax.fori_loop(
            0,
            timesteps,
            process_one_noise_level,
            (x_T, loop_key, log_eta_scales_in, init_per_level_acc, init_per_level_clip_frac),
        )

        act_final = jnp.clip(x_0, -1.0, 1.0)
        if compute_final_q:
            final_q = aggregate_q_fn([model.q(qp, obs, act_final) for qp in q_params_tuple])
        else:
            final_q = jnp.zeros(obs.shape[:-1])
        return MalaSampleResult(
            action=act_final, q=final_q, log_eta_scales=log_eta_scales_out,
            per_level_acc=per_level_acc_out, per_level_clip=per_level_clip_frac_out,
        )

    return stateless_get_action_mala_full
