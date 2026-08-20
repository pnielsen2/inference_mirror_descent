"""MALA-corrected reverse-diffusion sampler used by MGMD.

One full pass of the sampler:

* Sample x_T ~ N(0, I), then for t = T-1 .. 0:
    1. ``mala_steps`` MALA correction steps targeting
       ``E_total(t, x) = alpha * E_θ(s, x, t) - beta * Q_agg(s, x_0_hat)``,
       built by :func:`build_target_energy` (module level, so that the one
       definition of the chain's target is also available off-grid).
    2. The selected denoising predictor step -- the one out of the cleanest
       level (t = 0) taken separately, since ``Identity_then_DDPM_mean`` is the
       identity everywhere else.
* Return ``MalaSampleResult(action, q, log_eta_scales, per_level_acc,
  per_level_clip)`` with the per-level acceptance / clip arrays the trainer
  logs to wandb.
"""
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from relax.algorithm.mgmd_types import Diffv2TrainState, MalaSampleResult
from relax.network.actor_critic import ActorCritic
from relax.utils.diffusion import NoiseLevel, tweedie_x0


GUIDANCE_SNR_ANNEAL = ("none", "sqrt_abar", "abar")


def guidance_snr_anneal_factor(level: NoiseLevel, mode: str):
    """Level-dependent multiplier on beta, damping the guidance as SNR -> 0.

    The guided term is ``Q`` evaluated at the Tweedie estimate
    ``x0_hat = (x - omac grad E) / sqrt(abar)``, so its gradient in x carries a
    ``1 / sqrt(abar)`` factor. For an *exact* score that factor is cancelled by
    the bracket, which is itself O(sqrt(abar)) at low SNR -- the exact posterior
    mean moves with x only as ``(sqrt(abar)/omac) Cov[x_0|x]``, which vanishes as
    the posterior widens to the prior. An inexact score does not cancel it, and
    whatever error it has is amplified by that same ``1 / sqrt(abar)``: at
    lambda = -15 that is 1808x. Left alone this makes the low-SNR end of the
    tilted path enormously expensive to traverse (measured: 2e16 dynamic range in
    the score-optimal cost, which collapses an adaptive schedule onto lambda_min).

    ``abar`` restores the exact asymptotics -- it is 1 at the clean end, where the
    Tweedie approximation is good and the target must be left alone, and supplies
    the missing factor at the noisy end, so the guidance gradient decays like
    ``sqrt(abar)`` and ``rho_lambda -> N(0, I)`` as the paper's geometry assumes.
    (With a Gaussian prior of scale s the exact form is
    ``sigmoid(lambda + 2 log s)``; ``abar = sigmoid(lambda)`` is the s = 1 case.)
    ``sqrt_abar`` only holds the guidance gradient *constant* in lambda rather
    than decaying, which stops the collapse but leaves an O(beta) score
    perturbation at the noisy end.
    """
    if mode == "none":
        return jnp.float32(1.0)
    if mode == "sqrt_abar":
        return level.sqrt_abar
    if mode == "abar":
        return level.sqrt_abar * level.sqrt_abar
    raise ValueError(f"Unknown guidance_snr_anneal: {mode!r}; expected one of {GUIDANCE_SNR_ANNEAL}")


def build_target_energy(
    state: Diffv2TrainState,
    obs: jax.Array,
    aggregate_q_fn,
    *,
    model: ActorCritic,
    ema_normalization: bool,
    batch_advantage_normalization: bool,
    ema_advantage_normalization: bool,
    guidance_snr_anneal: str = "none",
):
    """The MALA/MH target ``U_level(x) = alpha E_theta - beta_eff Q(clip(x0_hat))``.

    Module-level, and parameterized by a :class:`NoiseLevel` rather than by a
    schedule index, so that the *one* definition of what the chain is sampling
    from also serves off-grid levels -- which is what
    :mod:`relax.algorithm.noise_schedule` needs to score a schedule. It is
    deliberately not the guided-predictor expression: no
    ``--guidance_strength_multiplier``, no gradient-space choice, since neither
    appears in the density the MH ratio corrects towards.

    ``obs`` already carries the leading ``[K]`` axis. Returns
    ``(energy_total, agg_q_at_action, beta_current)``; the first also returns
    ``(clip_frac, e_grad)`` as aux, both of which its callers reuse.
    """
    policy_params = state.params.policy
    q_params_tuple = state.params.q
    x0_hat_clip_radius = state.hp.x0_hat_clip_radius
    if ema_normalization:
        beta_current = state.beta / jnp.sqrt(jnp.maximum(state.advantage_second_moment_ema, jnp.float32(1e-6)))
    else:
        beta_current = state.beta

    # --batch_advantage_normalization: divide Q by sqrt(mean_s Var_K(Q)) --
    # the batch-mean of the per-state sample variance (ddof=1) over the K
    # denoised actions. stop-gradient'd, so it only rescales the guidance
    # magnitude (grad flows through the numerator Q).
    def maybe_batch_normalize(q):
        if not batch_advantage_normalization:
            return q
        denom = jnp.sqrt(jnp.maximum(jnp.mean(jnp.var(q, axis=0, ddof=1)), jnp.float32(1e-6)))
        return q / jax.lax.stop_gradient(denom)

    # --ema_advantage_normalization (ported from diffusion_policy_online_rl):
    # z-score Q by a slow EMA of the batch mean/std of the online agg-Q at the
    # sampled next-actions (tracked in the trainer's q_running_mean/std). Both
    # stats are stop-gradient'd, so the mean subtraction is a gradient /
    # energy-difference no-op (kept to mirror (Q - mu)/sigma) and only the
    # 1/sigma factor rescales the guidance magnitude.
    def maybe_ema_normalize(q):
        if not ema_advantage_normalization:
            return q
        mean = jax.lax.stop_gradient(state.q_running_mean)
        std = jnp.maximum(jax.lax.stop_gradient(state.q_running_std), jnp.float32(1e-6))
        return (q - mean) / std

    def agg_q_at_action(action):
        return maybe_ema_normalize(maybe_batch_normalize(aggregate_q_fn([model.q(qp, obs, action) for qp in q_params_tuple])))

    def q_aggregated_at_clipped_x0_hat(x0_hat):
        x0_clipped = jnp.clip(x0_hat, -x0_hat_clip_radius, x0_hat_clip_radius)
        return agg_q_at_action(x0_clipped)

    def energy_total(level, x):
        # beta_eff carries the SNR anneal, so it is a function of the LEVEL and
        # cannot be folded into the scalar beta_current returned below.
        E_vals, vjp_fn = jax.vjp(lambda a: model.energy_fn(policy_params, level, obs, a), x)
        (e_grad,) = vjp_fn(jnp.ones_like(E_vals))
        noise_pred = level.sqrt_omac[..., None] * e_grad  # ε̂ = σ_t ∇_x E_θ
        x0_hat = tweedie_x0(level, x, noise_pred)
        clip_frac = jnp.mean((jnp.abs(x0_hat) > x0_hat_clip_radius).astype(jnp.float32))
        beta_eff = beta_current * guidance_snr_anneal_factor(level, guidance_snr_anneal)
        return state.hp.alpha * E_vals - beta_eff * q_aggregated_at_clipped_x0_hat(x0_hat), (clip_frac, e_grad)

    return energy_total, agg_q_at_action, beta_current


def build_mala_sampler(
    *,
    model: ActorCritic,
    value_head,                    # ValueHead | None — supplied iff on-policy-EMA / KL-budget mode
    timesteps: int,
    batch_independent_guidance: bool,
    compute_final_q: bool = True,
    ema_normalization: bool,
    denoising_predictor: str,
    guidance_gradient_space: str,
    num_denoised_actions: int = 1,
    batch_advantage_normalization: bool = False,
    ema_advantage_normalization: bool = False,
    latent_action: bool = False,
    schedule_cost: bool = False,
    collect_levels: bool = False,
    guidance_snr_anneal: str = "none",
) -> Callable:
    """Return ``stateless_get_action_mala_full(key, state, obs, aggregate_q_fn)``.

    The returned closure is jit-able and vmap-able. ``aggregate_q_fn`` is
    supplied per-call; both the rollout path and the TD next-action sampling
    path pass ``--q_agg_sample``-aggregation, letting the same sampler serve
    both call sites.

    ``schedule_cost`` records the score-optimal cost of the interval between each
    adjacent pair of knots (unweighted, i.e. corrector speed v = 1; see
    :mod:`relax.algorithm.noise_schedule`), which that module turns into the
    schedule update. The cost differences two adjacent levels' drifts at a
    shared state, and MH needed both drifts anyway, so the schedule is scored on
    the chain the sampler was already running -- no extra network pass. It
    requires ``guidance_gradient_space == "xt"``, whose drift is the exact score
    the cost is defined on.

    ``collect_levels`` additionally returns one sample of every level's law,
    which nothing in training needs; it is what lets the tests re-derive the
    cost from scratch.
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
        value_params = state.value_params
        x0_hat_clip_radius = state.hp.x0_hat_clip_radius
        mala_adapt_rate = state.hp.mala_adapt_rate
        guidance_multiplier = state.hp.guidance_mult
        guidance_mult_increasing = state.hp.guidance_mult_increasing

        # Denoise K actions per state: broadcast a leading [K] axis onto obs so
        # every downstream Q / energy / eps call yields [K, batch, ...]. The K
        # slices are iid (independent x_T + MALA noise), so index 0 is a uniform
        # draw and the K per-state samples feed --batch_advantage_normalization.
        K = num_denoised_actions
        obs = jnp.broadcast_to(obs, (K, *obs.shape))
        action_shape = (*obs.shape[:-1], model.act_dim)

        key_x, loop_key = jax.random.split(key, 2)
        # Per-seed noise schedule: laid out from traced hp, so it can differ by
        # vmap slot and move over training. Cost is ~30 flops x T against
        # T x mala_steps network passes.
        schedule = model.schedule_for(state.hp, state.log_snr_levels)
        level = lambda t_idx: NoiseLevel.at(schedule, t_idx)
        x_recon_clip_radius = model.x_recon_clip_radius
        increasing_scheduler = schedule.alphas_cumprod / schedule.alphas_cumprod_prev
        increasing_scheduler = increasing_scheduler / jnp.maximum(increasing_scheduler[0], jnp.float32(1e-8))
        # beta_current = β: fixed constant, sqrt(2δ/M) (KL-budget), or
        # min(β_KL, β*) (one-step dist-shift), depending on run config.
        energy_total, agg_q_at_action, beta_current = build_target_energy(
            state, obs, aggregate_q_fn, model=model, ema_normalization=ema_normalization,
            batch_advantage_normalization=batch_advantage_normalization,
            ema_advantage_normalization=ema_advantage_normalization,
            guidance_snr_anneal=guidance_snr_anneal,
        )

        # Reduce a per-sample q [K, batch] to the scalar jax.grad differentiates:
        # the K denoised actions are independent samples (summed over axis 0),
        # while the batch axis is summed (batch-independent) or meaned (1/B).
        # For K=1 this matches the previous jnp.sum / jnp.mean exactly.
        def reduce_over_batch(q):
            q = jnp.sum(q, axis=0)
            return jnp.sum(q) if batch_independent_guidance else jnp.mean(q)

        def guidance_multiplier_at_t(t_idx):
            # The SNR anneal multiplies the predictor's guidance too, so the
            # proposal keeps pushing towards the same density MH corrects to.
            return guidance_multiplier * guidance_snr_anneal_factor(
                level(t_idx), guidance_snr_anneal
            ) * (
                (jnp.float32(1.0) - guidance_mult_increasing)
                + guidance_mult_increasing * increasing_scheduler[t_idx]
            )

        def q_aggregated_at_clipped_x0_hat(x0_hat):
            x0_clipped = jnp.clip(x0_hat, -x0_hat_clip_radius, x0_hat_clip_radius)
            return agg_q_at_action(x0_clipped)

        def guidance_value_from_x(x_in, t_idx):
            # Tweedie-clean prediction with alpha-scaled base
            # score; guidance component is NOT scaled.
            lvl = level(t_idx)
            eps_pred = model.eps_pred(policy_params, lvl, obs, x_in)
            x0_hat = tweedie_x0(lvl, x_in, eps_pred)
            q = q_aggregated_at_clipped_x0_hat(x0_hat)
            guidance_multiplier_t = guidance_multiplier_at_t(t_idx)

            # KL-budget mode adapts beta from advantage-moment statistics,
            # but the sampling objective here still uses the same clipped-Q
            # guidance term as the fixed-beta path.
            return guidance_multiplier_t * reduce_over_batch(q)

        def compute_guidance_gradient(x_in, t_idx):
            guidance_multiplier_t = guidance_multiplier_at_t(t_idx)
            if guidance_gradient_space == "xt":
                return jax.grad(lambda x: guidance_value_from_x(x, t_idx))(x_in)

            lvl = level(t_idx)
            eps_pred = model.eps_pred(policy_params, lvl, obs, x_in)
            x0_hat = tweedie_x0(lvl, x_in, eps_pred)
            if guidance_gradient_space == "x0hat":
                return jax.grad(
                    lambda x0: guidance_multiplier_t * reduce_over_batch(
                        q_aggregated_at_clipped_x0_hat(x0)
                    )
                )(jax.lax.stop_gradient(x0_hat))

            x0_clipped = jnp.clip(x0_hat, -x0_hat_clip_radius, x0_hat_clip_radius)
            return jax.grad(
                lambda action: guidance_multiplier_t * reduce_over_batch(
                    agg_q_at_action(action)
                )
            )(jax.lax.stop_gradient(x0_clipped))

        def jacobian_free_energy_and_drift(t_idx, x):
            lvl = level(t_idx)
            E_vals, vjp_fn = jax.vjp(lambda a: model.energy_fn(policy_params, lvl, obs, a), x)
            (e_grad,) = vjp_fn(jnp.ones_like(E_vals))
            noise_pred = lvl.sqrt_omac * e_grad
            x0_hat = tweedie_x0(lvl, x, noise_pred)
            x0_clipped = jnp.clip(x0_hat, -x0_hat_clip_radius, x0_hat_clip_radius)
            q = agg_q_at_action(x0_clipped)
            if guidance_gradient_space == "x0hat":
                grad_q = jax.grad(
                    lambda x0: jnp.sum(q_aggregated_at_clipped_x0_hat(x0))
                )(jax.lax.stop_gradient(x0_hat))
            else:
                grad_q = jax.grad(
                    lambda action: jnp.sum(agg_q_at_action(action))
                )(jax.lax.stop_gradient(x0_clipped))
            beta_eff = beta_current * guidance_snr_anneal_factor(lvl, guidance_snr_anneal)
            energy = state.hp.alpha * E_vals - beta_eff * q
            grad_energy = state.hp.alpha * e_grad - beta_eff * grad_q
            clip_frac = jnp.mean((jnp.abs(x0_hat) > x0_hat_clip_radius).astype(jnp.float32))
            return energy, grad_energy, clip_frac, e_grad

        # ---- Denoising predictor step (chosen at build time) -----------------
        def guided_eps_pred(t_idx, x_in):
            lvl = level(t_idx)
            noise_pred_scaled = state.hp.alpha * model.eps_pred(policy_params, lvl, obs, x_in)
            grad_q = compute_guidance_gradient(x_in, t_idx)
            return noise_pred_scaled - beta_current * lvl.sqrt_omac * grad_q

        def ddpm_mean_step(t_idx, x_in, _e_grad):
            eps_pred = guided_eps_pred(t_idx, x_in)
            x0_hat = jnp.clip(tweedie_x0(level(t_idx), x_in, eps_pred),
                               -x_recon_clip_radius, x_recon_clip_radius)
            return x0_hat * schedule.posterior_mean_coef1[t_idx] + x_in * schedule.posterior_mean_coef2[t_idx]

        def ddim_from_eps(t_idx, x_in, eps_pred):
            sqrt_ab_t = schedule.sqrt_alphas_cumprod[t_idx]
            sqrt_one_minus_ab_t = schedule.sqrt_one_minus_alphas_cumprod[t_idx]
            sqrt_ab_prev = jnp.sqrt(schedule.alphas_cumprod_prev[t_idx])
            # Stored rather than 1 - abar_prev: that subtraction loses most of its
            # relative precision in float32 once abar_prev -> 1, which is exactly
            # where small s_hat puts the clean end.
            sqrt_one_minus_ab_prev = jnp.sqrt(schedule.one_minus_alphas_cumprod_prev[t_idx])
            return (
                (sqrt_ab_prev / sqrt_ab_t) * x_in
                + (sqrt_one_minus_ab_prev - (sqrt_ab_prev / sqrt_ab_t) * sqrt_one_minus_ab_t) * eps_pred
            )

        def ddim_step(t_idx, x_in, _e_grad):
            return ddim_from_eps(t_idx, x_in, guided_eps_pred(t_idx, x_in))

        def ddim_unguided_step(t_idx, x_in, e_grad):
            # ε̂ = α σ_t ∇_x E_θ at the MALA-accepted state: the last MALA step
            # already computed this gradient, so the transport is free. No
            # guidance and no x0 clip -- all tilting stays inside the MH loop.
            # The α factor matches the base component of the chain's target
            # exp(−αE + βQ), whose level-t factor is p_t^α with score α∇log p_t;
            # ``guided_eps_pred`` applies the same α to its base term.
            eps = state.hp.alpha * schedule.sqrt_one_minus_alphas_cumprod[t_idx] * e_grad
            return ddim_from_eps(t_idx, x_in, eps)

        # ---- The MH energy: value and drift. Both variants return
        # ``(U, grad_x U, clip_frac, e_grad)`` so the MALA body does not branch.
        def mh_energy_xt(t_idx, x):
            E, vjp, (clip, e_grad) = jax.vjp(
                lambda xx: energy_total(level(t_idx), xx), x, has_aux=True)
            return E, vjp(jnp.ones_like(E))[0], clip, e_grad

        mh_energy = mh_energy_xt if guidance_gradient_space == "xt" else jacobian_free_energy_and_drift

        def identity_step(_t_idx, x_curr, _e_grad):
            return x_curr

        # ``denoising_step`` transitions between adjacent noise levels;
        # ``final_denoising_step`` is the t = 0 -> clean one, which the two
        # differ on only for Identity_then_DDPM_mean.
        if denoising_predictor in ("Identity", "Identity_then_DDPM_mean"):
            denoising_step = identity_step
        elif denoising_predictor == "DDPM_mean":
            denoising_step = ddpm_mean_step
        elif denoising_predictor == "DDIM":
            denoising_step = ddim_step
        elif denoising_predictor == "DDIM_unguided":
            if model.mala_steps < 1:
                raise ValueError("DDIM_unguided reuses the gradient from the last MALA "
                                 "step and therefore requires --mala_steps >= 1")
            denoising_step = ddim_unguided_step
        else:
            raise ValueError(f"Unknown denoising_predictor: {denoising_predictor}")
        # Identity_then_DDPM_mean leaves the chain itself exactly the identity
        # predictor's -- every level's MALA target, acceptance rate and schedule
        # cost is untouched, since nothing moves the sample between levels -- and
        # changes only what is read off the cleanest level: the guided posterior
        # mean rather than the (still noisy) level-0 sample. At t = 0 that mean is
        # exactly the guided Tweedie estimate, since abar_prev = 1 there makes
        # posterior_mean_coef1[0] = 1 and posterior_mean_coef2[0] = 0.
        final_denoising_step = (ddpm_mean_step if denoising_predictor == "Identity_then_DDPM_mean"
                                else denoising_step)

        if schedule_cost and guidance_gradient_space != "xt":
            raise ValueError("schedule_cost is a divergence between the exact level-wise "
                             "scores, which only guidance_gradient_space 'xt' computes; "
                             "the Jacobian-free drift is an approximation to them")

        # ---- MALA step-size scale clamp range (shared across all levels) -
        log_eta_min = jnp.log(jnp.float32(1e-8) / jnp.maximum(jnp.max(schedule.betas), jnp.float32(1e-8)))
        log_eta_max = jnp.log(jnp.float32(0.5) / jnp.maximum(jnp.min(schedule.betas), jnp.float32(1e-8)))

        # ---- Per-diffusion-level MALA correction + denoising predictor ----
        def run_mala_chain_at_level(t_idx, x_t, rng, log_eta_scales, drift_in):
            eta_base_t = jnp.maximum(schedule.betas[t_idx], jnp.float32(1e-8))
            eta_upper = jnp.float32(0.5)

            def mala_body(m, state):
                x_current, rng_step, log_eta_scale, accept_rate_sum, clip_frac_sum, _e_grad_at_x, cost, drift = state
                # Langevin Dynamics transition for Metropolis-Hastings forward proposal
                # compute energy, vector-jacobian product, and clip_frac in one pass
                E_x, grad_E_x, clip_x, e_grad_x = mh_energy(t_idx, x_current)
                step_size = jnp.clip(jnp.exp(log_eta_scale) * eta_base_t, jnp.float32(1e-8), eta_upper)
                if schedule_cost:
                    # First evaluation of this level is at the state it inherited,
                    # where the previous level's drift was also taken -- so the two
                    # scores of the score-optimal cost meet at the same x, and both
                    # were already needed by MH. v(t') is this level's MALA step
                    # size: matching the paper's corrector dZ = v grad log p dtau +
                    # sqrt(2v) dW against the MALA proposal x + h grad log p +
                    # sqrt(2h) z gives v = h exactly, so the velocity weight is the
                    # step the corrector actually takes. See
                    # :mod:`relax.algorithm.noise_schedule`.
                    gap = grad_E_x - drift_in
                    cost = jnp.where(
                        m == 0,
                        step_size * step_size * jnp.mean(jnp.sum(gap * gap, axis=-1)),
                        cost)

                proposal_mean = x_current - step_size * grad_E_x
                proposal_std = jnp.sqrt(jnp.float32(2.0) * step_size)

                rng_step, noise_key, u_key = jax.random.split(rng_step, 3)
                z = jax.random.normal(noise_key, x_current.shape)
                
                x_prop = proposal_mean + proposal_std * z
                # Langevin Dynamics reverse transition for Metropolis-Hastings acceptance
                # compute energy, vector-jacobian product, and clip_frac in one pass
                E_x_prop, grad_E_x_prop, _clip_prop, e_grad_prop = mh_energy(t_idx, x_prop)

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

                if schedule_cost:
                    # The accepted drift is what the identity predictor hands to
                    # the next level, to be differenced there against its own.
                    drift = jnp.where(accept[..., None], grad_E_x_prop, grad_E_x)
                return (
                    x_next,
                    rng_step,
                    log_eta_scale,
                    accept_rate_sum + acc_rate,
                    clip_frac_sum + clip_x,
                    jnp.where(accept[..., None], e_grad_prop, e_grad_x),
                    cost,
                    drift,
                )

            init_cost = jnp.float32(0.0) if schedule_cost else None
            mala_corrected_x_t, rng_out, log_eta_scale_new, acc_sum, clip_sum, e_grad_out, cost_out, drift_out = jax.lax.fori_loop(
                0,
                model.mala_steps,
                mala_body,
                (x_t, rng, log_eta_scales[t_idx], jnp.float32(0.0), jnp.float32(0.0),
                 jnp.zeros_like(x_t), init_cost, drift_in),
            )
            log_eta_scales = log_eta_scales.at[t_idx].set(log_eta_scale_new)
            return (mala_corrected_x_t, rng_out, log_eta_scales, acc_sum, clip_sum,
                    e_grad_out, cost_out, drift_out)

        def process_one_noise_level(step_fn, i, carry):
            # At noise level t: run MALA correction chain, then denoise to noise level t-1.
            x_t, rng, log_eta_scales, per_level_acc, per_level_clip_frac, x_levels, cost, drift = carry
            t_idx = timesteps - 1 - i

            mala_corrected_x_t, rng, log_eta_scales, acc_sum, clip_sum, e_grad, level_cost, drift = run_mala_chain_at_level(
                t_idx, x_t, rng, log_eta_scales, drift,
            )
            x_t_minus_1 = step_fn(t_idx, mala_corrected_x_t, e_grad)

            # ---- per-level diagnostics ----
            mala_steps_f = jnp.float32(model.mala_steps)
            per_level_acc     = per_level_acc.at[t_idx].set(acc_sum / jnp.maximum(mala_steps_f, jnp.float32(1.0)))
            per_level_clip_frac = per_level_clip_frac.at[t_idx].set(clip_sum / jnp.maximum(mala_steps_f, jnp.float32(1.0)))
            if collect_levels:
                # Post-correction, so this is the level's own equilibrium sample
                # x ~ rho_lambda_t -- what scoring the schedule needs.
                x_levels = x_levels.at[t_idx].set(mala_corrected_x_t)
            if schedule_cost:
                cost = cost.at[t_idx].set(level_cost)
            return x_t_minus_1, rng, log_eta_scales, per_level_acc, per_level_clip_frac, x_levels, cost, drift

        # ---- Drive the reverse diffusion: T → 0 over the level loop -----
        x_T = jax.random.normal(key_x, action_shape)
        init_per_level_acc = jnp.zeros((timesteps,), dtype=jnp.float32)
        init_per_level_clip_frac = jnp.zeros((timesteps,), dtype=jnp.float32)
        # x_levels[j] is a sample of the level-j law, with the synthetic level
        # ``timesteps`` (lambda = -inf) being the N(0, I) the chain starts from.
        # None is an empty pytree node, so not collecting costs nothing.
        init_x_levels = (jnp.zeros((timesteps + 1, *action_shape)).at[timesteps].set(x_T)
                         if collect_levels else None)
        # cost[j] scores the interval between knots j and j+1. The chain enters its
        # noisiest knot from N(0, I), whose own drift is x -- so cost[timesteps-1]
        # measures the initialization mismatch, which the schedule update discards:
        # that knot is pinned at the reference end and has no interval below it.
        init_cost = jnp.zeros((timesteps,), dtype=jnp.float32) if schedule_cost else None
        init_drift = x_T if schedule_cost else None
        carry = (x_T, loop_key, log_eta_scales_in, init_per_level_acc,
                 init_per_level_clip_frac, init_x_levels, init_cost, init_drift)
        # When the last transition differs, peel its level out of the loop rather
        # than select on the traced t_idx: a jnp.where would evaluate both
        # predictors at every level, which for Identity_then_DDPM_mean means T
        # guided predictor passes instead of one.
        peel_final_level = final_denoising_step is not denoising_step
        carry = jax.lax.fori_loop(
            0, timesteps - peel_final_level,
            partial(process_one_noise_level, denoising_step), carry,
        )
        if peel_final_level:
            carry = process_one_noise_level(final_denoising_step, timesteps - 1, carry)
        x_0, _, log_eta_scales_out, per_level_acc_out, per_level_clip_frac_out, x_levels_out, cost_out, _ = carry

        # In latent-action mode the endpoint is an unbounded latent; the env-side
        # Gaussian-CDF squash bounds it. Otherwise clip to the normalized box.
        act_final = x_0 if latent_action else jnp.clip(x_0, -1.0, 1.0)
        if compute_final_q:
            final_q = aggregate_q_fn([model.q(qp, obs, act_final) for qp in q_params_tuple])
        else:
            final_q = jnp.zeros(obs.shape[:-1])
        return MalaSampleResult(
            action=act_final, q=final_q, log_eta_scales=log_eta_scales_out,
            per_level_acc=per_level_acc_out, per_level_clip=per_level_clip_frac_out,
            schedule_cost=cost_out, x_levels=x_levels_out,
        )

    return stateless_get_action_mala_full
