from typing import NamedTuple, Optional

import numpy as np
import jax, jax.numpy as jnp

class NoiseLevel(NamedTuple):
    """One noise level, as a value rather than as a ``(schedule, index)`` pair.

    Everything the policy network and the Tweedie reconstruction need about a
    single level, and nothing else. Passing the level itself is what lets a
    level exist *off* the grid, which is what an adaptive schedule needs: the
    schedule loss differentiates through where a level sits, and distillation
    draws its levels continuously between them. ``cond`` is what the network
    conditions on -- the integer index for the fixed schedule families, the
    continuous log-SNR for the adaptive one, whose indices carry no fixed
    physical meaning (see :func:`schedule_from_log_snr`).
    """
    cond: jax.Array
    sqrt_abar: jax.Array
    sqrt_omac: jax.Array
    sqrt_recip_abar: jax.Array
    sqrt_recipm1_abar: jax.Array

    @staticmethod
    def at(sched: "BetaScheduleCoefficients", t) -> "NoiseLevel":
        """Level ``t`` of a precomputed schedule; ``t`` is a scalar or ``[B]``."""
        return NoiseLevel(sched.t_cond[t], sched.sqrt_alphas_cumprod[t],
                          sched.sqrt_one_minus_alphas_cumprod[t],
                          sched.sqrt_recip_alphas_cumprod[t],
                          sched.sqrt_recipm1_alphas_cumprod[t])

    @staticmethod
    def from_log_snr(lam) -> "NoiseLevel":
        """Level at log-SNR ``lam``, off any grid: ``abar = sigmoid(lam)``.

        Everything goes through ``log_sigmoid``, so no expression ever forms
        ``1 - abar`` and none of the four coefficients overflows or cancels at
        either end -- which matters here because ``lam`` is drawn from an
        unbounded Gaussian rather than read off a bounded grid.
        """
        half_log_abar = jnp.float32(0.5) * jax.nn.log_sigmoid(lam)
        half_log_omac = jnp.float32(0.5) * jax.nn.log_sigmoid(-lam)
        return NoiseLevel(lam, jnp.exp(half_log_abar), jnp.exp(half_log_omac),
                          jnp.exp(-half_log_abar), jnp.exp(-lam / jnp.float32(2.0)))


def tweedie_x0(level: NoiseLevel, x: jax.Array, noise_pred: jax.Array) -> jax.Array:
    """Tweedie clean estimate ``x_0_hat = (x - sqrt(1-abar) eps) / sqrt(abar)``.

    The trailing ``[..., None]`` is what lets one level (sampler) and one level
    per sample (distillation) share this code: it is a no-op broadcast on a
    scalar coefficient and the action-axis expansion on a ``[B]`` one.
    """
    return (x * level.sqrt_recip_abar[..., None]
            - noise_pred * level.sqrt_recipm1_abar[..., None])


class BetaScheduleCoefficients(NamedTuple):
    """Precomputed DDPM schedule arrays, indexed cleanest-first.

    A ``NamedTuple`` (not a frozen dataclass) so that it is a jax pytree: that
    is what lets a *per-seed* schedule ride inside the vmapped train state,
    which in turn is what makes ``--s_hat`` a per-seed ablation axis instead of
    a separate job. ``one_minus_alphas_cumprod`` is stored rather than
    recomputed as ``1 - alphas_cumprod`` because that subtraction cancels
    badly at the clean end (``abar -> 1``) in float32 -- exactly where a small
    ``s_hat`` pushes the schedule.
    """
    betas: jax.Array
    alphas: jax.Array
    alphas_cumprod: jax.Array
    alphas_cumprod_prev: jax.Array
    one_minus_alphas_cumprod: jax.Array
    one_minus_alphas_cumprod_prev: jax.Array
    sqrt_alphas_cumprod: jax.Array
    sqrt_one_minus_alphas_cumprod: jax.Array
    log_one_minus_alphas_cumprod: jax.Array
    sqrt_recip_alphas_cumprod: jax.Array
    sqrt_recipm1_alphas_cumprod: jax.Array
    posterior_variance: jax.Array
    posterior_log_variance_clipped: jax.Array
    posterior_mean_coef1: jax.Array
    posterior_mean_coef2: jax.Array
    # What the policy network conditions on at each level. The level index for
    # every fixed family (its physical meaning is pinned by the family), the
    # log-SNR itself for the adaptive one (its levels move).
    t_cond: jax.Array

    @staticmethod
    def from_parts(betas, alphas_cumprod, one_minus_alphas_cumprod):
        """Derive every array from ``(beta, abar, 1-abar)``.

        Written in ``jnp`` and free of any ``1 - abar`` or ``1/abar - 1``
        subtraction, so it is both jit/vmap-traceable and safe in float32.
        Accepting ``1-abar`` as an argument is what buys the latter: every
        expression below is a product or quotient of positive, well-scaled
        quantities.
        """
        abar, omac = alphas_cumprod, one_minus_alphas_cumprod
        abar_prev = jnp.concatenate([jnp.ones_like(abar[:1]), abar[:-1]])
        omac_prev = jnp.concatenate([jnp.zeros_like(omac[:1]), omac[:-1]])
        alphas = 1. - betas
        posterior_variance = betas * omac_prev / omac
        return BetaScheduleCoefficients(
            betas=betas, alphas=alphas,
            alphas_cumprod=abar, alphas_cumprod_prev=abar_prev,
            one_minus_alphas_cumprod=omac, one_minus_alphas_cumprod_prev=omac_prev,
            sqrt_alphas_cumprod=jnp.sqrt(abar),
            sqrt_one_minus_alphas_cumprod=jnp.sqrt(omac),
            # log(omac) is accurate where omac is small (clean end) but amplifies
            # relative error as omac -> 1; log1p(-abar) is accurate exactly there.
            log_one_minus_alphas_cumprod=jnp.where(omac > 0.5, jnp.log1p(-abar), jnp.log(omac)),
            sqrt_recip_alphas_cumprod=jnp.sqrt(1. / abar),
            sqrt_recipm1_alphas_cumprod=jnp.sqrt(omac / abar),
            posterior_variance=posterior_variance,
            posterior_log_variance_clipped=jnp.log(jnp.maximum(posterior_variance, 1e-20)),
            posterior_mean_coef1=betas * jnp.sqrt(abar_prev) / omac,
            posterior_mean_coef2=omac_prev * jnp.sqrt(alphas) / omac,
            t_cond=jnp.arange(abar.shape[0], dtype=jnp.float32),
        )

    @staticmethod
    def from_beta(betas: np.ndarray):
        """Eager entry point: ``abar`` and ``1-abar`` in float64, then delegate.

        The ``cumprod`` and the single ``1 - abar`` happen in host float64, so
        the clean end is exact before anything narrows to float32.
        """
        abar = np.cumprod(1. - np.asarray(betas, np.float64), axis=0)
        return BetaScheduleCoefficients.from_parts(betas, abar, 1. - abar)

    @staticmethod
    def cosine_beta_schedule(timesteps: int):
        s = 0.008
        t = np.arange(0, timesteps + 1) / timesteps
        alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        alphas_cumprod /= alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = np.clip(betas, 0, 0.999)
        return betas
    
    @staticmethod
    def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=0.999):
        return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)

    @staticmethod
    def constant_kl_beta_schedule(timesteps: int, snr_max=1000.0):
        """Constant-KL schedule: noise levels equally spaced in log(1 + SNR).

        Derived from requiring constant mutual-information loss per step:
            I(x_0; x_{k-1}) - I(x_0; x_k) = const  for all k
        where I(x_0; x_t) = (d/2) log(1 + SNR(t)).

        This gives:
            alpha_bar_k = 1 - R^{-(T-k)/T},  R = 1 + SNR_max
        with k=0 cleanest (alpha_bar ~ 1) and k=T-1 noisiest (alpha_bar > 0).
        Pure noise (alpha_bar = 0) is an implicit endpoint outside the schedule;
        the reverse process starts from N(0,I) and the model at t=T-1 provides
        the first denoising step.
        """
        R = 1.0 + snr_max
        T = timesteps
        k = np.arange(T, dtype=np.float64)
        exponent = (T - k) / T
        alphas_cumprod = 1.0 - R ** (-exponent)
        # Derive betas: beta_k = 1 - alpha_bar_k / alpha_bar_{k-1}
        alphas_cumprod_with_1 = np.concatenate([[1.0], alphas_cumprod])
        betas = 1.0 - alphas_cumprod_with_1[1:] / alphas_cumprod_with_1[:-1]
        betas = np.clip(betas, 1e-8, 0.999)
        return betas

def shift_schedule(base: BetaScheduleCoefficients, s_hat) -> BetaScheduleCoefficients:
    """Retarget a schedule from unit-variance data onto data of scale ``s_hat``.

    Every criterion for laying out diffusion steps -- information delivered per
    step, or variance transported by the sampler -- depends on the schedule and
    the data only through the *effective* log-SNR
    ``lambda_eff = lambda + 2 log s``, where ``s`` is the standard deviation of
    the clean data. A schedule only writes down ``lambda``, so one tuned for
    unit-variance data is uniformly off by ``2 log s``; using it unchanged
    silently asserts ``s = 1``. Undoing the offset is a rigid translation of the
    log-SNR curve, ``lambda_k -> lambda_k - 2 log s_hat``, which since
    ``exp(lambda) = abar / (1 - abar)`` reads

        abar_k  ->  abar_k / d_k,   d_k = abar_k + s_hat**2 * (1 - abar_k)

    and is exactly equivalent to standardizing the actions by ``s_hat`` before
    diffusing them. Note this *relabels* which physical noise level each
    ``lambda`` denotes rather than reordering the levels, so no amount of
    reshaping within a schedule family reproduces it.

    The induced map on the betas is elementwise, with no cumprod round trip and
    no cancellation:

        beta'_k = s_hat**2 * beta_k / d_k

    since ``d_k * abar_{k-1} - abar_k * d_{k-1} = s_hat**2 * abar_{k-1} beta_k``.
    ``s_hat`` may therefore be a *traced* per-seed scalar, which is what makes
    it an "easy" (vmapped) ablation axis and what will let it vary during
    training. ``d`` is a sum of positive terms, so it never cancels.

    ``s_hat < 1`` (actions narrower than unit scale, the usual case for a
    ``tanh``-bounded policy) raises every ``abar``: less noise is needed to
    destroy a narrower distribution, so the schedule spends fewer steps on the
    noise-dominated end and more where signal and noise are comparable.
    """
    s2 = s_hat ** 2
    d = base.alphas_cumprod + s2 * base.one_minus_alphas_cumprod
    return BetaScheduleCoefficients.from_parts(
        s2 * base.betas / d,
        base.alphas_cumprod / d,
        s2 * base.one_minus_alphas_cumprod / d,
    )


def adaptive_time_grid(num_timesteps: int) -> jax.Array:
    """The ``u`` coordinate the adaptive knots sit on: ``u_i = i / (N-1)``.

    The knots are ``N`` free log-SNR values, cleanest first, pinned at their two
    ends. Placing them at evenly spaced ``u`` makes ``u`` the schedule's own time
    coordinate, so ``lambda(u)`` interpolated between the knots is the continuous
    schedule they discretize -- what distillation draws from, and what the mean
    and standard deviation logged by :mod:`relax.algorithm.noise_schedule`
    describe.
    """
    return jnp.arange(num_timesteps, dtype=jnp.float32) / (num_timesteps - 1)


def log_snr_at(levels: jax.Array, u) -> jax.Array:
    """The continuous schedule: ``lambda`` linearly interpolated between knots."""
    return jnp.interp(u, adaptive_time_grid(levels.shape[0]), levels)


def cosine_log_snr_knots(num_timesteps: int, log_snr_min: float, log_snr_max: float) -> np.ndarray:
    """Initial knots: the cosine log-SNR trajectory truncated to the bounds.

    Cosine sweeps ``lambda`` from ``+inf`` at ``t = 0`` to ``-inf`` at ``t = 1``,
    so it can be inverted for the two times at which it crosses the bounds and
    resampled evenly between them. The result starts at ``log_snr_max``, ends at
    ``log_snr_min``, and follows cosine's shape in between -- i.e. the schedule
    the fixed families already use, cut to the range the adaptive levels are
    allowed to occupy. Eager (init only); the knots are state from then on.
    """
    s = 0.008
    # abar(t) = cos^2(u)/cos^2(u_0), u = (t+s)/(1+s) pi/2, so t(lambda) inverts
    # through abar = sigmoid(lambda).
    u0 = (s / (1 + s)) * (np.pi / 2)
    c = np.cos(u0) ** 2
    def t_of(lam):
        abar = 1.0 / (1.0 + np.exp(-lam))
        return np.arccos(np.sqrt(np.clip(c * abar, 0.0, 1.0))) * 2 * (1 + s) / np.pi - s
    t = np.linspace(t_of(log_snr_max), t_of(log_snr_min), num_timesteps)
    abar = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2 / c
    abar = np.clip(abar, 1e-30, 1 - 1e-16)
    lam = np.log(abar / (1 - abar))
    # Pin the ends exactly; the inversion is only accurate to float precision.
    lam[0], lam[-1] = log_snr_max, log_snr_min
    return lam.astype(np.float32)


def schedule_from_log_snr(levels: jax.Array) -> BetaScheduleCoefficients:
    """Coefficients for an arbitrary decreasing log-SNR ladder, ``abar = sigmoid(lambda)``.

    The network conditions on ``lambda`` (``t_cond``) rather than on the level
    index: the index of a moving knot means nothing, while a log-SNR means the
    same physical noise level however the schedule is rearranged, so the policy
    stays in distribution as the knots migrate.
    """
    log_abar = jax.nn.log_sigmoid(levels)
    # beta_k = 1 - abar_k / abar_{k-1} (abar_{-1} = 1, the clean end), taken
    # through expm1 so the clean end keeps its relative precision.
    log_abar_prev = jnp.concatenate([jnp.zeros_like(log_abar[:1]), log_abar[:-1]])
    return BetaScheduleCoefficients.from_parts(
        -jnp.expm1(log_abar - log_abar_prev), jnp.exp(log_abar), jax.nn.sigmoid(-levels),
    )._replace(t_cond=levels)


def build_beta_schedule(
    num_timesteps: int,
    beta_schedule_type: str,
    snr_max: float,
    s_hat: float = 1.0,
) -> BetaScheduleCoefficients:
    """Eager builder for the DDPM noise schedule.

    Returns a :class:`BetaScheduleCoefficients` holding all precomputed arrays.
    Called once at ``ActorCritic.create`` time to fix the base layout; training
    reads its schedule through ``ActorCritic.schedule_for(hp)``, which derives
    the per-seed one from this base.

    ``cosine`` and ``linear`` are used **as-is**: they are already defined on a
    ``num_timesteps`` grid, so their cleanest level automatically gets cleaner as
    steps are added (cosine SNR_0 = 124 / 401 / 1155 at T = 20 / 40 / 80). They
    previously went through a ``(1 - snr_max/(1+snr_max)) / betas[0]`` rescale
    that pinned SNR_0 to ``snr_max`` for *every* T, which

      * made ``--diffusion_steps 40/80`` pointless at the clean end -- the extra
        steps were spent subdividing the noisy end (rescale factors 3.2x, 9.2x), and
      * broke ``linear`` outright: betas[0]=1e-4 forced an 80x rescale that
        clipped 19 of 20 levels to 0.999, leaving only ~2 usable noise levels.

    At the shipped T=20 cosine default the rescale was a 1.0009x no-op, so
    removing it reproduces prior cosine runs to within 0.09%.

    ``snr_max`` is still required by ``constant_kl``, which is *defined* by it
    (alpha_bar_k = 1 - (1+snr_max)^{-(T-k)/T}); it is ignored otherwise.

    ``s_hat`` eagerly shifts whichever family was chosen onto data of that scale
    (see :func:`shift_schedule`); it is orthogonal to the family and to
    ``snr_max``, and defaults to the no-op ``1.0``. Training does **not** use
    this argument -- ``ActorCritic.create`` builds the unshifted base and the
    shift is applied per-seed from ``state.hp.s_hat`` inside the traced code, so
    it can vary by vmap slot and over training. It is kept here for offline
    schedule analysis, where a single static schedule is what you want.
    ``adaptive`` has no fixed layout to shift and ignores it.
    """
    if beta_schedule_type == 'adaptive':
        raise ValueError(
            "'adaptive' has no fixed layout to precompute -- its levels are free knots "
            "carried in the train state. Use ActorCritic.schedule_for(hp, levels), or "
            "schedule_from_log_snr(levels) directly."
        )
    if beta_schedule_type == 'constant_kl':
        betas = BetaScheduleCoefficients.constant_kl_beta_schedule(
            num_timesteps, snr_max=snr_max)
    elif beta_schedule_type == 'cosine':
        betas = BetaScheduleCoefficients.cosine_beta_schedule(num_timesteps)
    elif beta_schedule_type == 'linear':
        betas = BetaScheduleCoefficients.linear_beta_schedule(num_timesteps)
    else:
        raise ValueError(f"Unknown beta_schedule_type: {beta_schedule_type}")
    base = BetaScheduleCoefficients.from_beta(np.clip(betas, 0, 0.999))
    return base if s_hat == 1.0 else shift_schedule(base, s_hat)


def p_mean_variance(
    schedule: BetaScheduleCoefficients,
    t,
    x: jax.Array,
    noise_pred: jax.Array,
    x_recon_clip_radius: Optional[float] = None,
):
    """DDPM reverse-process posterior mean / log-variance.

    ``t`` is the diffusion timestep index (scalar). Returns
    ``(model_mean, model_log_variance)`` where ``model_log_variance`` is
    just ``schedule.posterior_log_variance_clipped[t]``.
    """
    x_recon = x * schedule.sqrt_recip_alphas_cumprod[t] - noise_pred * schedule.sqrt_recipm1_alphas_cumprod[t]
    if x_recon_clip_radius is not None:
        r = jnp.float32(x_recon_clip_radius)
        x_recon = jnp.clip(x_recon, -r, r)
    model_mean = x_recon * schedule.posterior_mean_coef1[t] + x * schedule.posterior_mean_coef2[t]
    model_log_variance = schedule.posterior_log_variance_clipped[t]
    return model_mean, model_log_variance
