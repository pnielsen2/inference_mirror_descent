"""Place the noise levels by equalizing a score-optimal cost.

``--beta_schedule_type adaptive`` implements the schedule of Williams, Campbell,
Doucet and Syed, *Score-Optimal Diffusion Schedules* (NeurIPS 2024). The levels
are not a schedule family with a few parameters: they are ``N`` free knots in
log-SNR, cleanest first, pinned at both ends, and each update moves the interior
ones toward the layout that spends the same effort on every step.

THE COST. The paper derives what an update from ``p_t`` to ``p_t'`` costs by
imagining a predictor that moves the sample and a Langevin corrector that fixes
up whatever the predictor got wrong, and measuring the work the corrector does:
the squared distance, at ``tau -> 0``, between the corrector's velocity and the
velocity it would have had if the predictor had landed exactly on ``p_t'``. That
work is a velocity-weighted Fisher divergence (their Eq. 13-15),

    L(t -> t') = v(t')^2 E_{x ~ p_t} || grad log p_t'(x) - grad log p_t(x) ||^2 ,

where ``v(t)`` is the speed of that hypothetical corrector. Our sampler is exactly
the case this specializes to: with ``--denoising_predictor Identity`` (or
``Identity_then_DDPM_mean``, which is the identity between every pair of levels
and only reads the clean action off the last one) the predictor is the identity
map, which is the paper's Example 2.1 (annealed Langevin), so the corrector does
all the work and the cost is their ``L_c``. That is the branch with
no Jacobian term -- nothing here needs a Hessian or a Hutchinson estimator. Both
scores are evaluated at the *same* x, drawn from the level the chain is coming
FROM, which is precisely the state a level inherits from the one above it.

WE TAKE v(t) = THE MALA STEP SIZE, WHERE THE PAPER TAKES sigma(t). Their
criterion (Sec. 3.3) is that Langevin should "explore the same proportion of our
distribution" at every t, i.e. that ``v`` track the scale on which the corrector
moves; they then set ``v = sigma`` on the grounds that for normalised data the
scale of ``p_t`` is of order ``sigma(t)``. That holds in the variance-EXPLODING
convention their image results use, where ``s(t) = 1`` and ``sigma`` runs to 80
against a data scale of 0.5. It does not hold here: we are variance-preserving,
so the marginal variance ``abar s_data^2 + (1 - abar)`` moves by 2.3x-4.9x at the
action scales these policies reach, while ``sigma^2 = 1 - abar`` moves by 3.3e6.

We do not need the proxy, because our corrector is MALA and we know its step
size. Matching the paper's corrector, ``dZ = v grad log p dtau + sqrt(2 v) dW``,
against the MALA proposal ``x + h grad log p + sqrt(2 h) z`` gives ``v = h`` with
``dtau = 1``: the step size IS the velocity, not a stand-in for it. So the weight
is ``h_t^2``, with ``h_t = clip(exp(log_eta_scale_t) beta_t, 1e-8, 0.5)`` read off
the level being entered -- the same number the proposal uses on that step, after
Robbins-Monro has tuned it to 0.574 acceptance.

WHAT THIS ASSUMES, AND WHEN IT IS WRONG. The paper's cost measures the corrector's
instantaneous *effort*: a faster corrector displaces more per unit tau, so it
scores higher. That is the right notion under their Assumption 3.1, where the
corrector runs to stationarity and no error survives, so the cost is pure path
geometry. With ``--mala_steps 1`` we are far from that, and the quantity that
governs sample quality is instead the discrepancy that SURVIVES correction,
``D_t (1 - kappa_t)^(2 mala_steps)`` with ``kappa_t ~ m_t h_t`` the per-step
contraction. Those two disagree exactly where the corrector is impotent: if the
step collapses while the target's curvature saturates (the clean end, where
``m -> 1/s^2`` rather than ``1/(1-abar)``), effort-based weighting scores that
region as cheap while survival-based weighting scores it as expensive. The
observable that separates them is the per-level acceptance rate: flat at 0.574
means the adaptation is holding ``m h = O(1)`` and the two agree up to a constant;
acceptance drifting to 1 at the clean end means the corrector is idle there and
this weight is understating that region.

NOTHING IS RE-EVALUATED. ``grad log p = -grad U`` for the exact MALA target ``U =
alpha E_theta - beta_eff Q(clip(x0_hat))``, and MH already computes that drift at
every step. The drift a level accepts and hands on, differenced against the first
drift the next level takes at that same inherited state, *is* the integrand -- so
the sampler returns one scalar per interval and this module is pure algebra over
them. There is no separate estimator to drift out of sync with the density MH
corrects towards. Unlike a thermodynamic-length cost there is also no within-state
variance, so a single denoised action per state suffices.

THE UPDATE (their Algorithms 1 and 2). Cost is locally quadratic in the step,
``L ~ delta(t) dt^2``, so ``sqrt(L)`` is a length element and the total
``Lambda = sum_i sqrt(L_i)`` is the path's intrinsic length -- an invariant of
the path, not of how it is discretized. Theorem 3.1: total cost is minimized by
the generator travelling that path at constant speed, i.e. knot ``i`` sits where
the cumulative length first reaches ``Lambda * i / (N-1)``. So accumulate the
lengths, invert the cumulative-length map by interpolation, and resample it
evenly. Because the inversion returns the two endpoint knots unchanged, the
pinned ends stay pinned for free.

Per-batch cost estimates are noisy, so the new layout is not adopted outright but
blended in, ``levels <- gamma * optimal + (1 - gamma) * levels``, with ``gamma``
from ``state.hp.noise_schedule_gamma`` (per-seed, packable like any other hp).
Both sequences are decreasing, so their convex combination is too: monotonicity
of the ladder needs no enforcing.

ON THE RATE. ``gamma`` is not comparable to the paper's stated 0.05-0.1 at face
value. In their released code the blend fires per schedule *resample*, and a
resample waits until every level has been visited ``n_l_min`` times by the random
per-sample timestep draw -- ``n_l_min = 24`` over T = 1000 levels at batch 384 for
their CIFAR run, so roughly every 63 batches, making their 0.01 about 1.6e-4 of
schedule motion per batch. Here the MALA chain sweeps *every* level on every
update, with ``batch_size`` samples per level rather than 24, so one blend happens
per training step and a given gamma buys ~60x more motion per batch than the same
number does there. Hence the 1e-3 default. Their 1D experiments, which do resample
nearly every batch, used 0.01-0.1, so that is the right range to compare a
per-step gamma against.

The paper interpolates with a monotone cubic (Fritsch-Carlson); we use a linear
interpolant, which is monotone as well and is what ``jnp.interp`` gives under jit.
The knots move by a fraction ``gamma`` of the gap per update, so the interpolant's
smoothness is not what limits accuracy here.
"""
import jax.numpy as jnp

from relax.utils.diffusion import adaptive_time_grid


def _mean_std(levels):
    """Mean and standard deviation of the schedule read as a distribution.

    The knots define a continuous schedule ``lambda(u)``, linear between them on
    an even ``u`` grid, and drawing ``u ~ U(0,1)`` (what distillation does) pushes
    that into a distribution over log-SNR. Its moments are exact for a piecewise
    linear map: a segment from ``a`` to ``b`` contributes ``(a + b)/2`` to the
    first moment and ``(a^2 + ab + b^2)/3`` to the second. This is the honest
    summary of where the levels have ended up -- the layout itself is free, so a
    mean and a spread describe it rather than parameterize it.
    """
    a, b = levels[:-1], levels[1:]
    w = jnp.float32(1.0) / (levels.shape[0] - 1)
    mean = jnp.sum(w * (a + b) / jnp.float32(2.0))
    second = jnp.sum(w * (a * a + a * b + b * b) / jnp.float32(3.0))
    return mean, jnp.sqrt(jnp.maximum(second - mean * mean, jnp.float32(0.0)))


def build_updater(*, timesteps: int):
    """Return ``update(state, interval_cost)``: one blended step of Algorithm 1.

    ``interval_cost[j]`` is the cost of the interval between knots ``j`` and
    ``j + 1``. The last entry is dropped: the chain enters its noisiest knot from
    ``N(0, I)``, so that entry measures the initialization mismatch rather than an
    interval, and that knot is pinned at the reference end anyway.
    """
    u_grid = adaptive_time_grid(timesteps)

    def update(state, interval_cost):
        levels = state.log_snr_levels
        # The floor is what keeps the cumulative length STRICTLY increasing, so
        # its inverse stays well defined (and free of the 0/0 a tie would give
        # jnp.interp) where a stretch of the path costs nothing to cross. Each
        # floored interval contributes 1e-6 of phantom length, negligible against
        # the totals these costs produce.
        # The floor is RELATIVE to the largest interval, not absolute: the cost's
        # overall scale moves with beta, the reward scale, the action dimension and
        # the v(t) convention, so an absolute floor silently becomes binding when
        # the scale drops -- and it binds first on the CHEAPEST interval, which is
        # the clean end, where the surviving-error argument says accuracy matters
        # most. Relative keeps it a regulariser for the inverse rather than a
        # hidden reparameterisation of the clean tail.
        c = interval_cost[:-1]
        length = jnp.sqrt(jnp.maximum(c, jnp.float32(1e-12) * jnp.max(c)))
        cumulative = jnp.concatenate([jnp.zeros((1,), jnp.float32), jnp.cumsum(length)])
        total = cumulative[-1]
        optimal = jnp.interp(total * u_grid, cumulative, levels)

        gamma = state.hp.noise_schedule_gamma
        new_levels = gamma * optimal + (jnp.float32(1.0) - gamma) * levels
        mean, std = _mean_std(levels)
        return state._replace(log_snr_levels=new_levels), {
            # The schedule as of this batch, i.e. the one the cost was measured on.
            "Schedule/path_length": total,
            "Schedule/total_cost": jnp.sum(interval_cost[:-1]),
            "Schedule/noise_schedule_mean": mean,
            "Schedule/noise_schedule_std": std,
        }

    return update
