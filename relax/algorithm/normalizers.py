"""Global normalizer scalars estimated on the fly from the K denoised actions.

Both scalars below summarize a *heterogeneous* population (one true value per
state) by a single number, but their loss geometries differ, and the geometry --
not convenience -- dictates the accumulator. See ``notes/snr_schedule_shift``.

* **Advantage scale** ``sigma_hat``, the divisor of the guidance Q. The tilt
  ``pi(a|o) exp(beta Q / sigma_hat)`` is an exponential family in
  ``eta = beta / sigma_hat``, so the KL to the ideally-normalized tilt is a
  Bregman divergence of the Q cumulant generating function; for Gaussian Q that
  is exactly ``(beta^2/2) (sigma_Q/sigma_hat - 1)^2``. Dividing by the *within
  state* sd ``sigma_Q(o)`` is what makes the per-state KL spend equal to
  ``beta^2/2`` at every state. The cost is asymmetric -- under-tilting saturates
  at ``beta^2/2`` (fall back to the untilted policy) while over-tilting diverges
  like ``e^{2u}`` -- and the population minimizer is the *contraharmonic* mean
  ``E[sigma_Q^2] / E[sigma_Q]``, which sits above the arithmetic mean by exactly
  the factor ``1 + CV^2``. Hence two accumulators in LINEAR space.

* **Schedule shift** ``s_hat``, the assumed clean-action scale. Its cost is
  ``cosh(log(s/s_hat))``, symmetric in the log, so the target is the geometric
  mean and the accumulators live in LOG space.

Only *unbiasedness* matters for either: an EMA at rate ``r`` averages over
~``1/r`` states, which kills estimator variance but passes estimator bias
straight through. So the first moment of the sample sd is divided by ``c4`` and
the log accumulator has ``log_bias`` subtracted, both exact. (A *per-state*
shift would want the opposite -- the Bayes inflation and shrinkage of
``notes/snr_schedule_shift`` -- but that is a different question.)
"""
import math

import jax.numpy as jnp
from scipy.special import digamma, polygamma

_EPS = jnp.float32(1e-8)
# Safety rail only. cosh is flat, so a wrong s_hat is cheap, but a runaway
# estimate would move the whole noise schedule. The upper end has to clear an
# untrained policy (a fresh --latent_action net already sits near 2) without
# admitting a divergence; the lower end a very sharp converged policy.
S_HAT_BOUNDS = (0.02, 4.0)


# Host-side, not jnp: these are static functions of K, and a jnp call inside a jit
# trace is staged into the graph even on literal inputs, so it could not be a float.
def c4(nu: int) -> float:
    """``E[s_nu] = c4(nu) * sigma`` for a sample sd on ``nu`` degrees of freedom."""
    return math.sqrt(2.0 / nu) * math.exp(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2))


def log_bias(nu: int) -> float:
    """``E[log s_nu] = log sigma + log_bias(nu)``, exact and additive."""
    return float(0.5 * (digamma(nu / 2) - math.log(nu / 2)))


def log_var(nu: int) -> float:
    """``Var[log s_nu]``; subtract from ``Var(log s_K)`` to recover ``tau^2``."""
    return float(0.25 * polygamma(1, nu / 2))


def update(state, q, actions, *, step, K, use_for_advantage, use_for_s_hat,
           s_hat_bounds=S_HAT_BOUNDS):
    """Fold one batch of ``[K, batch]`` per-state samples into the normalizer EMAs.

    ``q`` is the aggregated Q at the K tilted actions (raw, *not* already
    normalized) and ``actions`` is ``[K, batch, act_dim]``. The derived scalars
    are written into the fields the sampler already reads -- ``q_running_std``
    for the guidance-Q divisor and ``hp.s_hat`` for the schedule shift -- so the
    sampler needs no knowledge of any of this, and since the EMAs are folded in
    *after* the sampler ran there is no feedback loop inside one graph.

    The EMAs are seeded with the first observation rather than a constant, so no
    initial value has to be guessed. Returns ``(state, info)``.
    """
    nu_q = K - 1
    # Pooling the act_dim coordinates buys degrees of freedom for free: d(K-1)
    # rather than K-1, so K=2 already gives a usable s_hat. It does conflate
    # genuine state dependence with anisotropy across coordinates, which is the
    # right trade when the target is a single scalar shift.
    nu_s = actions.shape[-1] * (K - 1)
    sigma_q = jnp.sqrt(jnp.maximum(jnp.var(q, axis=0, ddof=1), _EPS))
    log_s = jnp.log(jnp.maximum(jnp.mean(jnp.var(actions, axis=0, ddof=1), axis=-1), _EPS)) / 2

    ema = lambda prev, x, rate: jnp.where(step == 0, x, prev + rate * (x - prev))
    r_a, r_s = state.hp.adv_norm_ema_rate, state.hp.s_hat_ema_rate
    m1 = ema(state.sigma_q_within_ema, jnp.mean(sigma_q), r_a)
    m2 = ema(state.sigma_q_within_sq_ema, jnp.mean(sigma_q ** 2), r_a)
    l1 = ema(state.log_s_ema, jnp.mean(log_s), r_s)
    l2 = ema(state.log_s_sq_ema, jnp.mean(log_s ** 2), r_s)

    # E[sigma^2] = m2 (Bessel, already unbiased); E[sigma] = m1 / c4.
    adv_scale = c4(nu_q) * m2 / jnp.maximum(m1, _EPS)
    s_hat = jnp.clip(jnp.exp(l1 - log_bias(nu_s)), *s_hat_bounds)
    # Heterogeneity of each population, and the residual cost each floor implies:
    # E[KL]/(beta^2/2) -> CV^2/(1+CV^2) for the tilt, transport cost -> e^{tau^2/2}.
    cv2 = jnp.maximum(m2 * c4(nu_q) ** 2 / jnp.maximum(m1 ** 2, _EPS) - 1, 0.0)
    tau2 = jnp.maximum(l2 - l1 ** 2 - log_var(nu_s), 0.0)

    state = state._replace(sigma_q_within_ema=m1, sigma_q_within_sq_ema=m2,
                           log_s_ema=l1, log_s_sq_ema=l2)
    if use_for_advantage:
        state = state._replace(q_running_std=adv_scale)
    if use_for_s_hat:
        state = state._replace(hp=state.hp._replace(s_hat=s_hat))
    return state, {
        "Normalizer/adv_scale": adv_scale,
        # sqrt(E[sigma^2]) is what --batch_advantage_normalization uses; it is
        # smaller than the optimum by exactly sqrt(1 + CV^2), i.e. it over-tilts.
        "Normalizer/adv_scale_rms": jnp.sqrt(m2),
        "Normalizer/sigma_q_cv2": cv2,
        "Normalizer/adv_kl_floor_frac": cv2 / (1 + cv2),
        "Normalizer/s_hat_estimate": s_hat,
        "Normalizer/s_hat_tau2": tau2,
        "Normalizer/s_hat_cosh_floor": jnp.exp(tau2 / 2),
    }
