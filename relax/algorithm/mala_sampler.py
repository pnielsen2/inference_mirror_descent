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

        def reconstruct_x0_from_xt(x_in, t_idx):
            eps_pred = model.eps_pred(policy_params, obs, x_in, t_idx)
            return reconstruct_x0_from_noise(x_in, t_idx, eps_pred)

        def clip_x0_hat(x0_hat):
            return jnp.clip(x0_hat, -x0_hat_clip_radius, x0_hat_clip_radius)

        def q_aggregated_at_action(action):
            return aggregate_q_fn([model.q(qp, obs, action) for qp in q_params_tuple])

        def q_aggregated_at_clipped_x0_hat(x0_hat):
            return q_aggregated_at_action(clip_x0_hat(x0_hat))

        # beta_current = β: fixed constant, sqrt(2δ/M) (KL-budget), or
        # min(β_KL, β*) (one-step dist-shift), depending on run config.
        def base_energy_value_and_grad(t_idx, x):
            E_vals, vjp_fn = jax.vjp(
                lambda a: model.energy_fn(policy_params, obs, a, t_idx),
                x,
            )
            grad_E = vjp_fn(jnp.ones_like(E_vals))[0]
            return E_vals, grad_E

        def reconstruct_mala_x0_hat(x, t_idx, grad_E):
            noise_pred = schedule.sqrt_one_minus_alphas_cumprod[t_idx] * grad_E
            return reconstruct_x0_from_noise(x, t_idx, noise_pred)

        def compute_clip_frac(x0_hat):
            return jnp.mean(
                (jnp.abs(x0_hat) > x0_hat_clip_radius).astype(jnp.float32)
            )

        def mala_q_gradient_at_action(action):
            action_eval = jax.lax.stop_gradient(action)
            return jax.grad(
                lambda a: jnp.sum(q_aggregated_at_action(a))
            )(action_eval)

        def xt_target_energy(t_idx, x):
            E_vals, grad_E = base_energy_value_and_grad(t_idx, x)
            x0_hat = reconstruct_mala_x0_hat(x, t_idx, grad_E)
            q = q_aggregated_at_clipped_x0_hat(x0_hat)
            energy = state.hp.alpha * E_vals - beta_current * q
            return energy, compute_clip_frac(x0_hat)

        def xt_energy_and_drift(t_idx, x):
            energy, vjp_fn, clip_frac = jax.vjp(
                lambda z: xt_target_energy(t_idx, z),
                x,
                has_aux=True,
            )
            grad_energy = vjp_fn(jnp.ones_like(energy))[0]
            return energy, grad_energy, clip_frac

        def x0hat_energy_and_drift(t_idx, x):
            E_vals, grad_E = base_energy_value_and_grad(t_idx, x)
            x0_hat = reconstruct_mala_x0_hat(x, t_idx, grad_E)
            q = q_aggregated_at_action(x0_hat)
            grad_q = mala_q_gradient_at_action(x0_hat)
            energy = state.hp.alpha * E_vals - beta_current * q
            grad_energy = state.hp.alpha * grad_E - beta_current * grad_q
            return energy, grad_energy, compute_clip_frac(x0_hat)

        def x0hatclipped_energy_and_drift(t_idx, x):
            E_vals, grad_E = base_energy_value_and_grad(t_idx, x)
            x0_hat = reconstruct_mala_x0_hat(x, t_idx, grad_E)
            x0_eval = clip_x0_hat(x0_hat)
            q = q_aggregated_at_action(x0_eval)
            grad_q = mala_q_gradient_at_action(x0_eval)
            energy = state.hp.alpha * E_vals - beta_current * q
            grad_energy = state.hp.alpha * grad_E - beta_current * grad_q
            return energy, grad_energy, compute_clip_frac(x0_hat)

        if guidance_gradient_space == "xt":
            mala_energy_and_drift = xt_energy_and_drift
        elif guidance_gradient_space == "x0hat":
            mala_energy_and_drift = x0hat_energy_and_drift
        else:
            mala_energy_and_drift = x0hatclipped_energy_and_drift

        def predictor_q_scalar(action):
            q = q_aggregated_at_action(action)
            return guidance_multiplier * reduce_over_batch(q)

        def compute_xt_guidance_gradient(x_in, t_idx):
            def guidance_value_from_xt(x):
                x0_hat = reconstruct_x0_from_xt(x, t_idx)
                return predictor_q_scalar(clip_x0_hat(x0_hat))

            return jax.grad(guidance_value_from_xt)(x_in)

        def compute_x0hat_guidance_gradient(x_in, t_idx):
            x0_hat = jax.lax.stop_gradient(reconstruct_x0_from_xt(x_in, t_idx))
            return jax.grad(predictor_q_scalar)(x0_hat)

        def compute_x0hatclipped_guidance_gradient(x_in, t_idx):
            x0_hat = reconstruct_x0_from_xt(x_in, t_idx)
            x0_eval = jax.lax.stop_gradient(clip_x0_hat(x0_hat))
            return jax.grad(predictor_q_scalar)(x0_eval)

        if guidance_gradient_space == "xt":
            compute_guidance_gradient = compute_xt_guidance_gradient
        elif guidance_gradient_space == "x0hat":
            compute_guidance_gradient = compute_x0hat_guidance_gradient
        else:
            compute_guidance_gradient = compute_x0hatclipped_guidance_gradient

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
                # Langevin Dynamics transition for Metropolis-Hastings forward proposal.
                E_x, grad_E_x, clip_x = mala_energy_and_drift(t_idx, x_current)
                step_size = jnp.clip(jnp.exp(log_eta_scale) * eta_base_t, jnp.float32(1e-8), eta_upper)

                proposal_mean = x_current - step_size * grad_E_x
                proposal_std = jnp.sqrt(jnp.float32(2.0) * step_size)

                rng_step, noise_key, u_key = jax.random.split(rng_step, 3)
                z = jax.random.normal(noise_key, x_current.shape)
                
                x_prop = proposal_mean + proposal_std * z
                # Reverse proposal uses the same gradient mode at the proposed point.
                E_x_prop, grad_E_x_prop, _clip_prop = mala_energy_and_drift(t_idx, x_prop)

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
