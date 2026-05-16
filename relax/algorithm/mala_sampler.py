"""MALA-corrected reverse-diffusion sampler used by DPMD.

One full pass of the sampler:

* Sample x_T ~ N(0, I), then for t = T-1 .. 0:
    1. ``mala_steps`` MALA correction steps targeting
       ``E_total(t, x) = energy_multiplier * E_θ(s, x, t) - tfg_eta * Q_agg(s, x_0_hat)``.
       (Q_agg is omitted when ``tfg_eta == 0`` so the eta-sweep
       ``critic_normalization='none'`` runs are unbiased.)
    2. A deterministic DDIM-style predictor step (``--ddim_predictor`` is
       hardcoded on; the original stochastic-DDPM noise term was removed
       because MALA already injects noise). Skipped when
       ``--mala_no_predictor`` is set.
* Return ``MalaSampleResult(action, q, log_eta_scales, per_level_acc,
  per_level_clip)`` with the per-level acceptance / clip arrays the trainer
  logs to wandb.

Behaviour and PRNG layout are byte-identical to the previous in-DPMD
implementation; this is a pure refactor that hoists the closure to module
scope so ``dpmd.py`` reads top-to-bottom around the training step.
"""
from typing import Callable

import jax
import jax.numpy as jnp

from relax.algorithm.dpmd_types import Diffv2TrainState, MalaSampleResult
from relax.network.diffv2 import Diffv2Net


def build_mala_sampler(
    *,
    agent: Diffv2Net,
    value_head,                    # ValueHead | None — used iff critic_normalization == "ema"
    timesteps: int,
    energy_multiplier: float,
    batch_independent_guidance: bool,
    critic_normalization: str,
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
        x0_hat_clip_radius_hp = state.hp.x0_hat_clip_radius
        mala_adapt_rate_hp = state.hp.mala_adapt_rate
        guidance_mult_hp = state.hp.guidance_mult

        obs_batch = obs
        shape = (*obs_batch.shape[:-1], agent.act_dim)

        # 3-way split kept verbatim to preserve PRNG layout from a deleted
        # multi-particle / particle-select sampling path; the two unused
        # keys must continue to be split off here for byte-exact PRNG match.
        key_sample, _key_select, _noise_key = jax.random.split(key, 3)
        key_x, loop_key = jax.random.split(key_sample)
        B = agent.diffusion.beta_schedule()

        # ---- Sampler-local helpers (capture `state`, `obs_batch`, `B`) ---
        def energy_model(t, x):
            E = agent.energy_fn(policy_params, obs_batch, x, t)
            # Scale base energy by energy_multiplier (tempers the base distribution)
            return energy_multiplier * E

        def q_aggregated_at_x0(x0_in):
            q_means = [agent.q(qp, obs_batch, x0_in) for qp in q_params_tuple]
            return aggregate_q_fn(q_means)

        def q_mean_from_x(x_in, t_idx):
            # Tweedie-clean prediction with energy_multiplier-tempered base
            # score; guidance component is NOT scaled.
            noise_pred = agent.policy(policy_params, obs_batch, x_in, t_idx)
            noise_pred_scaled = energy_multiplier * noise_pred
            x0_hat = (
                x_in * B.sqrt_recip_alphas_cumprod[t_idx]
                - noise_pred_scaled * B.sqrt_recipm1_alphas_cumprod[t_idx]
            )
            x0_q = jnp.clip(x0_hat, -x0_hat_clip_radius_hp, x0_hat_clip_radius_hp)
            q = q_aggregated_at_x0(x0_q)

            agg_fn = jnp.sum if batch_independent_guidance else jnp.mean
            mult = guidance_mult_hp

            # Critic normalization: use (Q - V) / std instead of Q
            if critic_normalization == "ema" and value_params is not None:
                v = value_head.apply(value_params, obs_batch)
                advantage = q - v
                adv_std = jnp.sqrt(jnp.maximum(adv_second_moment_ema, jnp.float32(1e-6)))
                return mult * agg_fn(advantage / adv_std)
            return mult * agg_fn(q)

        def grad_guidance(x_in, t_idx):
            def guided(_):
                return jax.grad(lambda xx: q_mean_from_x(xx, t_idx))(x_in)

            def unguided(_):
                return jnp.zeros_like(x_in)

            return jax.lax.cond(tfg_eta_current > 0.0, guided, unguided, operand=None)

        def energy_total(t, x, sample_key):
            def only_model(x_in):
                return energy_model(t, x_in), jnp.float32(0.0)

            def with_q(x_in):
                E_mod = energy_model(t, x_in)
                noise_pred = agent.policy(policy_params, obs_batch, x_in, t)
                x0_hat = (
                    x_in * B.sqrt_recip_alphas_cumprod[t]
                    - noise_pred * B.sqrt_recipm1_alphas_cumprod[t]
                )
                x0_q = jnp.clip(x0_hat, -x0_hat_clip_radius_hp, x0_hat_clip_radius_hp)
                clip_frac = jnp.mean((jnp.abs(x0_hat) > x0_hat_clip_radius_hp).astype(jnp.float32))
                q_agg = q_aggregated_at_x0(x0_q)
                return E_mod - tfg_eta_current * q_agg, clip_frac

            return jax.lax.cond(tfg_eta_current > 0.0, with_q, only_model, x)

        # ---- DDIM-style deterministic predictor (variant chosen at build time) ----
        # The PRNG split is retained -- and the resulting ``_z_key``
        # intentionally unused -- to keep ``k_out`` bit-identical with
        # the legacy stochastic DDPM predictor path.
        if mala_guided_predictor:
            def predictor_step(t_idx, x_in, k_in):
                noise_pred = agent.policy(policy_params, obs_batch, x_in, t_idx)
                noise_pred_scaled = energy_multiplier * noise_pred  # base score; guidance below is NOT scaled
                k_out, _z_key = jax.random.split(k_in)
                grad_q = grad_guidance(x_in, t_idx)
                sigma_t = B.sqrt_one_minus_alphas_cumprod[t_idx]
                eps_pred = noise_pred_scaled - tfg_eta_current * sigma_t * grad_q
                model_mean, _ = agent.diffusion.p_mean_variance(t_idx, x_in, eps_pred)
                return model_mean, k_out
        else:
            def predictor_step(t_idx, x_in, k_in):
                noise_pred = agent.policy(policy_params, obs_batch, x_in, t_idx)
                noise_pred_scaled = energy_multiplier * noise_pred
                k_out, _z_key = jax.random.split(k_in)
                model_mean, _ = agent.diffusion.p_mean_variance(t_idx, x_in, noise_pred_scaled)
                return model_mean, k_out

        # ---- Post-MALA transition: either DDIM step or skip (build-time choice) ----
        if mala_no_predictor:
            def after_mala_correction(t_idx, x_curr, k):
                return x_curr, k
        else:
            def after_mala_correction(t_idx, x_curr, k):
                return predictor_step(t_idx, x_curr, k)

        # ---- MALA step-size scale clamp range (shared across all levels) -
        eta_base_min = jnp.maximum(jnp.min(B.betas), jnp.float32(1e-8))
        eta_base_max = jnp.maximum(jnp.max(B.betas), jnp.float32(1e-8))
        log_eta_min = jnp.log(jnp.float32(1e-8) / eta_base_max)
        # Fixed step-size cap of 0.5 (per-level recurrence cap removed).
        log_eta_max = jnp.log(jnp.float32(0.5) / eta_base_min)

        # ---- Per-diffusion-level MALA correction + DDIM predictor --------
        def level_body(i, carry):
            x_curr, k, log_eta_scales, acc_sum, acc_count, per_level_acc, per_level_clip_frac = carry
            t = timesteps - 1 - i
            eta_base_t = jnp.maximum(B.betas[t], jnp.float32(1e-8))
            log_eta_scale0 = log_eta_scales[t]
            eta_upper = jnp.float32(0.5)

            def mala_body(_, state):
                x_step, k_step, log_eta_scale, acc_sum_step, acc_count_step, clip_sum_step = state

                E_x, vjp_x, clip_x = jax.vjp(lambda xx: energy_total(t, xx, k_step), x_step, has_aux=True)
                grad_E_x = vjp_x(jnp.ones_like(E_x))[0]

                k_step, noise_key, u_key = jax.random.split(k_step, 3)
                eta_k = jnp.clip(
                    jnp.exp(log_eta_scale) * eta_base_t,
                    jnp.float32(1e-8),
                    eta_upper,
                )
                z = jax.random.normal(noise_key, x_step.shape)
                sd = jnp.sqrt(jnp.float32(2.0) * eta_k)
                x_prop = x_step - eta_k * grad_E_x + sd * z

                E_x_prop, vjp_x_prop, _clip_prop = jax.vjp(lambda xx: energy_total(t, xx, k_step), x_prop, has_aux=True)
                grad_E_x_prop = vjp_x_prop(jnp.ones_like(E_x_prop))[0]

                mean_f = x_step - eta_k * grad_E_x
                mean_r = x_prop - eta_k * grad_E_x_prop

                def log_gauss(xv, meanv):
                    diff = xv - meanv
                    return -jnp.sum(diff * diff, axis=-1) / (jnp.float32(4.0) * eta_k)

                log_q_prop_given_x = log_gauss(x_prop, mean_f)
                log_q_x_given_prop = log_gauss(x_step, mean_r)

                log_alpha = (-E_x_prop + E_x) + (log_q_x_given_prop - log_q_prop_given_x)
                u = jax.random.uniform(u_key, E_x.shape)
                accept = jnp.log(u) < jnp.minimum(jnp.float32(0.0), log_alpha)

                x_new = jnp.where(accept[..., None], x_prop, x_step)

                acc_rate = jnp.mean(accept.astype(jnp.float32).reshape(-1))
                target = jnp.float32(0.574)
                log_eta_scale = log_eta_scale + mala_adapt_rate_hp * (acc_rate - target)
                log_eta_scale = jnp.clip(log_eta_scale, log_eta_min, log_eta_max)

                return (
                    x_new,
                    k_step,
                    log_eta_scale,
                    acc_sum_step + acc_rate,
                    acc_count_step + jnp.float32(1.0),
                    clip_sum_step + clip_x,
                )

            acc_sum_before = acc_sum
            acc_count_before = acc_count
            x_curr, k, log_eta_scale_final, acc_sum_level, acc_count_level, clip_sum_level = jax.lax.fori_loop(
                0,
                agent.mala_steps,
                mala_body,
                (x_curr, k, log_eta_scale0, jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0)),
            )
            log_eta_scales = log_eta_scales.at[t].set(log_eta_scale_final)
            acc_sum = acc_sum_before + acc_sum_level
            acc_count = acc_count_before + acc_count_level

            # Per-level mean acceptance: arithmetic preserved verbatim from
            # the legacy `do_mala_level` -> outer subtract path so XLA
            # lowering stays bit-identical with the pre-flatten code.
            level_acc_sum = acc_sum - acc_sum_before
            level_acc_count = acc_count - acc_count_before
            level_acc = level_acc_sum / jnp.maximum(level_acc_count, jnp.float32(1.0))
            per_level_acc = per_level_acc.at[t].set(level_acc)
            mala_steps_f = jnp.float32(agent.mala_steps)
            per_level_clip_frac = per_level_clip_frac.at[t].set(
                clip_sum_level / jnp.maximum(mala_steps_f, jnp.float32(1.0))
            )

            x_next, k = after_mala_correction(t, x_curr, k)
            return x_next, k, log_eta_scales, acc_sum, acc_count, per_level_acc, per_level_clip_frac

        # ---- Drive the reverse diffusion: T → 0 over the level loop -----
        x0 = jax.random.normal(key_x, shape)
        init_per_level_acc = jnp.zeros((timesteps,), dtype=jnp.float32)
        init_per_level_clip_frac = jnp.zeros((timesteps,), dtype=jnp.float32)
        x_final, _, log_eta_scales_out, _acc_sum_final, _acc_count_final, per_level_acc_out, per_level_clip_frac_out = jax.lax.fori_loop(
            0,
            timesteps,
            level_body,
            (x0, loop_key, log_eta_scales_in, jnp.float32(0.0), jnp.float32(0.0),
             init_per_level_acc, init_per_level_clip_frac),
        )

        act_final = jnp.clip(x_final, -1.0, 1.0)
        q_means_f = [agent.q(qp, obs_batch, act_final) for qp in q_params_tuple]
        q = aggregate_q_fn(q_means_f)
        return MalaSampleResult(
            action=act_final, q=q, log_eta_scales=log_eta_scales_out,
            per_level_acc=per_level_acc_out, per_level_clip=per_level_clip_frac_out,
        )

    return stateless_get_action_mala_full
