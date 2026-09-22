"""Place adaptive log-SNR levels by equalizing one-step MALA difficulty.

The construction follows the equal-path-length schedule of Williams, Campbell,
Doucet and Syed, *Score-Optimal Diffusion Schedules* (NeurIPS 2024), but uses a
local metric specialized to this sampler's finite corrector budget. For an
interval entered at sample ``x``, let ``gap`` be the difference between the two
adjacent target scores at that same ``x`` and ``h`` the receiving level's MALA
step size. Changing the score shifts the proposal mean by ``h * gap`` against
proposal variance ``2h``, so the dimensionless one-step difficulty is

    C = h * mean(gap ** 2).

The sampler already computes both scores and ``h`` for Metropolis-Hastings, so
recording this scalar adds no network evaluations. Cost is locally quadratic in
the log-SNR interval width, making ``sqrt(C)`` a length element. The optimal
interior knots divide cumulative length evenly; each update moves the current
knots a small fraction ``gamma`` toward that layout. Both endpoints remain
pinned and a convex combination of decreasing layouts stays decreasing.
"""
import jax.numpy as jnp

from relax.utils.diffusion import adaptive_time_grid


def _mean_std(levels):
    """Mean and standard deviation of the schedule read as a distribution.

    The knots define a continuous schedule ``lambda(u)``, linear between them on
    an even ``u`` grid, and drawing ``u ~ U(0,1)`` (what distillation does) pushes
    that into a distribution over log-SNR. Its moments are exact for a piecewise
    linear map: a segment from ``a`` to ``b`` contributes ``(a + b)/2`` to the
    first moment and ``(a^2 + ab + b^2)/3`` to the second.
    """
    a, b = levels[:-1], levels[1:]
    w = jnp.float32(1.0) / (levels.shape[0] - 1)
    mean = jnp.sum(w * (a + b) / jnp.float32(2.0))
    second = jnp.sum(w * (a * a + a * b + b * b) / jnp.float32(3.0))
    return mean, jnp.sqrt(jnp.maximum(second - mean * mean, jnp.float32(0.0)))


def build_updater(*, timesteps: int):
    """Return ``update(state, interval_cost)`` for one slow knot update.

    ``interval_cost[j]`` is the cost between knots ``j`` and ``j + 1``. The
    final entry is discarded because it measures initialization from ``N(0,I)``
    into the noisiest knot rather than an interval between two knots.
    """
    u_grid = adaptive_time_grid(timesteps)

    def update(state, interval_cost):
        levels = state.log_snr_levels
        c = interval_cost[:-1]
        # A relative floor keeps the cumulative length strictly increasing
        # without changing the layout when the overall cost scale changes.
        length = jnp.sqrt(jnp.maximum(c, jnp.float32(1e-12) * jnp.max(c)))
        cumulative = jnp.concatenate([jnp.zeros((1,), jnp.float32), jnp.cumsum(length)])
        total = cumulative[-1]
        optimal = jnp.interp(total * u_grid, cumulative, levels)
        gamma = state.hp.noise_schedule_gamma
        new_levels = gamma * optimal + (jnp.float32(1.0) - gamma) * levels
        mean, std = _mean_std(levels)
        return state._replace(log_snr_levels=new_levels), {
            "Schedule/path_length": total,
            "Schedule/total_cost": jnp.sum(c),
            "Schedule/noise_schedule_mean": mean,
            "Schedule/noise_schedule_std": std,
        }

    return update
