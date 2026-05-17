"""MALA-corrected reverse-diffusion sampler used by DPMD.

One full pass of the sampler:

* Sample x_T ~ N(0, I), then for t = T-1 .. 0:
    1. ``mala_steps`` MALA correction steps targeting
       ``E_total(t, x) = energy_multiplier * E_θ(s, x, t) - tfg_eta * Q_agg(s, x_0_hat)``.
       (Q_agg is omitted when ``tfg_eta == 0`` so the eta-sweep
       no-budget runs are unbiased.)
    2. A deterministic DDIM-style predictor step (``--ddim_predictor`` is
       hardcoded on). Skipped when ``--mala_no_predictor`` is set.
* Return ``MalaSampleResult(action, q, log_eta_scales, per_level_acc,
  per_level_clip)`` with the per-level acceptance / clip arrays the trainer
  logs to wandb.

Behavior and PRNG layout are byte-identical to the previous in-DPMD
implementation; this is a pure refactor that hoists the closure to module
scope so ``dpmd.py`` reads top-to-bottom around the training step.
"""
from typing import Callable

import jax
import jax.numpy as jnp

from relax.algorithm.dpmd_types import Diffv2TrainState, MalaSampleResult
from relax.network.actor_critic import ActorCritic


def build_mala_sampler(
    *,
    model: ActorCritic,
    value_head,                    # ValueHead | None — supplied iff on-policy-EMA / KL-budget mode
    timesteps: int,
    energy_multiplier: float,
    batch_independent_guidance: bool,
    mala_guided_predictor: bool,
    mala_no_predictor: bool,
) -> Callable:
    """Return ``stateless_get_action_mala_full(key, state, obs, aggregate_q_fn)``.

    The returned closure is jit-able and vmap-able. ``aggregate_q_fn`` is
    supplied per-call (the TD-update path passes ``min``-aggregation; the
    rollout path passes ``--q_critic_agg``-aggregation), letting the same
    sampler serve both call sites.
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
        tfg_eta_current = state.tfg_eta
        value_params = state.value_params
        adv_second_moment_ema = state.advantage_second_moment_ema
        x0_hat_clip_radius = state.hp.x0_hat_clip_radius
        mala_adapt_rate = state.hp.mala_adapt_rate
        guidance_multiplier = state.hp.guidance_mult

        action_shape = (*obs.shape[:-1], model.act_dim)

        # 3-way split kept verbatim to preserve PRNG layout from a deleted
        # multi-particle / particle-select sampling path; the two unused
        # keys must continue to be split off here for byte-exact PRNG match.
        key_sample, _key_select, _noise_key = jax.random.split(key, 3)
        key_x, loop_key = jax.random.split(key_sample)
        schedule = model.schedule
        x_recon_clip_radius = model.x_recon_clip_radius
        reduce_over_batch = jnp.sum if batch_independent_guidance else jnp.mean

        # ---- Shared Tweedie building blocks -------------------------
        def reconstruct_x0_from_noise(x_in, t_idx, noise_pred):
            return (
                x_in * schedule.sqrt_recip_alphas_cumprod[t_idx]
                - noise_pred * schedule.sqrt_recipm1_alphas_cumprod[t_idx]
            )

        def q_aggregated_at_clipped_x0_hat(x0_hat):
            x0_clipped = jnp.clip(x0_hat, -x0_hat_clip_radius, x0_hat_clip_radius)
            return aggregate_q_fn([model.q(qp, obs, x0_clipped) for qp in q_params_tuple])

        # ---- Energy target (drives MALA correction steps) -----------
        def energy_model(t, x):
            E = model.energy_fn(policy_params, obs, x, t)
            # Scale base energy by energy_multiplier (tempers the base distribution)
            return energy_multiplier * E

        def energy_total(t, x):
            # NOTE: keep the ``lax.cond(tfg_eta_current > 0.0, ...)`` wrapper
            # for the same XLA-fusion / bit-exactness reason as
            # ``compute_guidance_gradient``
            # above. Both branches are mathematically equivalent when
            # ``tfg_eta_current > 0`` (the only regime exercised today), but
            # collapsing the cond perturbs floating-point reduction order in
            # the surrounding fused region.
            def with_q(x_in):
                E_mod = energy_model(t, x_in)
                # Use raw (unscaled) noise pred for Tweedie reconstruction;
                # energy_model already applies energy_multiplier.
                noise_pred = model.policy(policy_params, obs, x_in, t)
                x0_hat = reconstruct_x0_from_noise(x_in, t, noise_pred)
                clip_frac = jnp.mean((jnp.abs(x0_hat) > x0_hat_clip_radius).astype(jnp.float32))
                q_agg = q_aggregated_at_clipped_x0_hat(x0_hat)
                return E_mod - tfg_eta_current * q_agg, clip_frac

            def only_model(x_in):
                return energy_model(t, x_in), jnp.float32(0.0)

            return jax.lax.cond(tfg_eta_current > 0.0, with_q, only_model, x)

        # ---- Q-guidance gradient (drives predictor step) ------------
        def scaled_policy_noise(t_idx, x_in):
            noise_pred = model.policy(policy_params, obs, x_in, t_idx)
            return energy_multiplier * noise_pred

        def guidance_value_from_x(x_in, t_idx):
            # Tweedie-clean prediction with energy_multiplier-tempered base
            # score; guidance component is NOT scaled.
            noise_pred_scaled = scaled_policy_noise(t_idx, x_in)
            x0_hat = reconstruct_x0_from_noise(x_in, t_idx, noise_pred_scaled)
            q = q_aggregated_at_clipped_x0_hat(x0_hat)

            # On-policy-EMA / KL-budget mode: normalize advantage (Q - V) / std.
            if value_head is not None and value_params is not None:
                v = value_head.apply(value_params, obs)
                advantage = q - v
                adv_std = jnp.sqrt(jnp.maximum(adv_second_moment_ema, jnp.float32(1e-6)))
                return guidance_multiplier * reduce_over_batch(advantage / adv_std)
            return guidance_multiplier * reduce_over_batch(q)

        def compute_guidance_gradient(x_in, t_idx):
            # NOTE: keep the ``lax.cond(tfg_eta_current > 0.0, ...)`` wrapper.
            # When ``tfg_eta_current == 0`` the unguided branch is a constant
            # zero, but with ``tfg_eta_current > 0`` (the case for all current
            # KL-budget / eta-sweep runs) the cond is still load-bearing for
            # bit-exactness: removing it lets XLA fuse ``jax.grad(guidance_value_from_x)``
            # the surrounding fused region differently, which flips the last
            # ULPs of MALA-energy reductions and breaks the bit-exact baseline.
            def guided(_):
                return jax.grad(lambda xx: guidance_value_from_x(xx, t_idx))(x_in)

            def unguided(_):
                return jnp.zeros_like(x_in)

            return jax.lax.cond(tfg_eta_current > 0.0, guided, unguided, operand=None)

        # ---- DDIM predictor step (guided or unguided, chosen at build time) ----
        # The PRNG split is retained -- and the resulting ``_z_key``
        # intentionally unused -- to keep ``k_out`` bit-identical with
        # the legacy stochastic DDPM predictor path.
        if mala_guided_predictor:
            def ddim_step(t_idx, x_in, k_in):
                noise_pred_scaled = scaled_policy_noise(t_idx, x_in)  # base score; guidance below is NOT scaled.
                k_out, _z_key = jax.random.split(k_in)
                grad_q = compute_guidance_gradient(x_in, t_idx)
                sigma_t = schedule.sqrt_one_minus_alphas_cumprod[t_idx]
                eps_pred = noise_pred_scaled - tfg_eta_current * sigma_t * grad_q
                x0_hat = jnp.clip(reconstruct_x0_from_noise(x_in, t_idx, eps_pred),
                                   -x_recon_clip_radius, x_recon_clip_radius)
                model_mean = x0_hat * schedule.posterior_mean_coef1[t_idx] + x_in * schedule.posterior_mean_coef2[t_idx]
                return model_mean, k_out
        else:
            def ddim_step(t_idx, x_in, k_in):
                noise_pred_scaled = scaled_policy_noise(t_idx, x_in)
                k_out, _z_key = jax.random.split(k_in)
                x0_hat = jnp.clip(reconstruct_x0_from_noise(x_in, t_idx, noise_pred_scaled),
                                   -x_recon_clip_radius, x_recon_clip_radius)
                model_mean = x0_hat * schedule.posterior_mean_coef1[t_idx] + x_in * schedule.posterior_mean_coef2[t_idx]
                return model_mean, k_out

        if mala_no_predictor:
            def denoising_step(t_idx, x_curr, k):
                return x_curr, k
        else:
            denoising_step = ddim_step

        # ---- MALA step-size scale clamp range (shared across all levels) -
        log_eta_min = jnp.log(jnp.float32(1e-8) / jnp.maximum(jnp.max(schedule.betas), jnp.float32(1e-8)))
        log_eta_max = jnp.log(jnp.float32(0.5) / jnp.maximum(jnp.min(schedule.betas), jnp.float32(1e-8)))

        # ---- Per-diffusion-level MALA correction + DDIM predictor --------
        def run_mala_chain_at_level(t_idx, x_t, rng, log_eta_scales):
            eta_base_t = jnp.maximum(schedule.betas[t_idx], jnp.float32(1e-8))
            eta_upper = jnp.float32(0.5)

            def mala_body(_, state):
                x_current, rng_step, log_eta_scale, accept_rate_sum, clip_frac_sum = state
                # Langevin Dynamics transition for Metropolis-Hastings forward proposal
                # compute energy, vector-jacobian product, and clip_frac in one pass
                E_x, vjp_x, clip_x = jax.vjp(lambda x: energy_total(t_idx, x), x_current, has_aux=True)
                grad_E_x = vjp_x(jnp.ones_like(E_x))[0]
                step_size = jnp.clip(jnp.exp(log_eta_scale) * eta_base_t, jnp.float32(1e-8), eta_upper)

                proposal_mean = x_current - step_size * grad_E_x
                proposal_std = jnp.sqrt(jnp.float32(2.0) * step_size)

                rng_step, noise_key, u_key = jax.random.split(rng_step, 3)
                z = jax.random.normal(noise_key, x_current.shape)
                
                x_prop = proposal_mean + proposal_std * z
                # Langevin Dynamics reverse transition for Metropolis-Hastings acceptance
                # compute energy, vector-jacobian product, and clip_frac in one pass
                E_x_prop, vjp_x_prop, _clip_prop = jax.vjp(lambda xx: energy_total(t_idx, xx), x_prop, has_aux=True)
                grad_E_x_prop = vjp_x_prop(jnp.ones_like(E_x_prop))[0]

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
            x_t_minus_1, rng = denoising_step(t_idx, mala_corrected_x_t, rng)

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
        final_q_values = [model.q(qp, obs, act_final) for qp in q_params_tuple]
        final_q = aggregate_q_fn(final_q_values)
        return MalaSampleResult(
            action=act_final, q=final_q, log_eta_scales=log_eta_scales_out,
            per_level_acc=per_level_acc_out, per_level_clip=per_level_clip_frac_out,
        )

    return stateless_get_action_mala_full
