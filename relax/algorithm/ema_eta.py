"""Host-side EMA + adaptive guidance-strength η update for the KL-budget /
on-policy path of DPMD.

Two pure functions, one per sampler dispatch mode in ``VmapOffPolicyTrainer``:

* :func:`update_state_kl_only` -- KL-budget-only η selection.
  η = sqrt(2δ / E[A^2])  (then η_state := η * sqrt(E[A^2]) so the in-graph
  guidance scale matches the formula used by ``q_mean_from_x`` which divides
  the advantage by sqrt(E[A^2])).

* :func:`update_state_one_step` -- second-order one-step distribution-shift
  η*: takes the KL ceiling AND a quadratic-shape correction estimated from
  consecutive (A_t, A_{t+1}) advantage pairs, returns η = min(η_KL, η*).

Each function returns ``(new_state, adv_per_env)``: the trainer replaces
``algorithm.state`` with ``new_state`` and uses ``adv_per_env`` to roll the
one-step covariance buffer for the next call.

The numerical sequence (float64 promotion, the 1e-8 clamps, the order of
EMA updates) is preserved verbatim from the previous in-trainer
implementation so this is a pure code move.
"""
from typing import Tuple

import jax.numpy as jnp
import numpy as np


def _broadcast_state_scalar(value, N: int) -> np.ndarray:
    """Promote a state scalar/[N] field to a [N] float64 ndarray."""
    arr = np.asarray(value).astype(np.float64)
    if arr.ndim == 0:
        arr = np.broadcast_to(arr, (N,)).astype(np.float64)
    return arr


def update_state_kl_only(state, q_per_env: np.ndarray, v_per_env: np.ndarray,
                         N: int) -> Tuple[object, np.ndarray]:
    """KL-budget-only path: update E[A^2] EMA and set η = sqrt(2δ/E[A^2])."""
    adv_per_env = q_per_env - v_per_env                # [N, M]
    m2_batch = np.mean(adv_per_env ** 2, axis=1)        # [N]

    tau_v = _broadcast_state_scalar(state.hp.adv_ema_tau, N)
    cur_m2 = np.asarray(state.advantage_second_moment_ema)
    new_m2 = (1 - tau_v) * cur_m2 + tau_v * m2_batch

    kl_budget = _broadcast_state_scalar(state.hp.kl_budget_val, N)
    m2_safe = np.maximum(new_m2, 1e-8)
    sqrt_v = np.sqrt(m2_safe)
    eta_kl_raw = np.sqrt(2.0 * kl_budget / m2_safe)
    new_eta = eta_kl_raw * sqrt_v

    new_state = state._replace(
        advantage_second_moment_ema=jnp.asarray(new_m2.astype(np.float32)),
        tfg_eta=jnp.asarray(new_eta.astype(np.float32)),
    )
    return new_state, adv_per_env


def update_state_one_step(state, q_per_env: np.ndarray, v_per_env: np.ndarray,
                          prev_adv_per_env, prev_valid,
                          N: int) -> Tuple[object, np.ndarray]:
    """One-step distribution-shift path: update {E[A^2], E[A^3], cov, shape}
    EMAs and pick η = min(η_KL, η*) where η* comes from the second-order
    expansion using the one-step covariance estimate."""
    adv_per_env = q_per_env - v_per_env                # [N, M]
    m2_batch = np.mean(adv_per_env ** 2, axis=1)        # [N]
    m3_batch = np.mean(adv_per_env ** 3, axis=1)        # [N]

    tau_v = _broadcast_state_scalar(state.hp.adv_ema_tau, N)
    cur_m2 = np.asarray(state.advantage_second_moment_ema)
    cur_m3 = np.asarray(state.advantage_third_moment_ema)
    cur_c = np.asarray(state.dist_shift_covariance_ema)
    cur_shape = np.asarray(state.dist_shift_shape_ema)

    new_m2 = (1 - tau_v) * cur_m2 + tau_v * m2_batch
    new_m3 = (1 - tau_v) * cur_m3 + tau_v * m3_batch
    new_c = cur_c
    new_shape = cur_shape

    # One-step covariance c_batch is only valid for env-steps where the
    # previous step did not terminate; seeds with no valid samples this
    # batch keep the prior EMA values unchanged.
    if prev_adv_per_env is not None and prev_valid is not None:
        valid = prev_valid                               # [N, M] bool
        valid_count = np.sum(valid, axis=1)              # [N]
        prod = valid.astype(np.float64) * (adv_per_env ** 2) * prev_adv_per_env
        sums = np.sum(prod, axis=1)
        c_batch = np.where(valid_count > 0, sums / np.maximum(valid_count, 1), 0.0)
        c_batch_valid = valid_count > 0

        new_c_candidate = (1 - tau_v) * cur_c + tau_v * c_batch
        new_c = np.where(c_batch_valid, new_c_candidate, cur_c)

        gamma = _broadcast_state_scalar(state.hp.gamma, N)
        tau_s = _broadcast_state_scalar(state.hp.shape_ema_tau, N)
        v_raw_safe = np.maximum(m2_batch, 1e-8)
        b_batch = 2.0 * gamma * c_batch + m3_batch
        s_batch = b_batch / v_raw_safe ** 1.5
        new_shape_candidate = (1 - tau_s) * cur_shape + tau_s * s_batch
        new_shape = np.where(c_batch_valid, new_shape_candidate, cur_shape)

    # η = min(η_KL, η*); η* is +inf when shape is non-negative.
    kl_budget = _broadcast_state_scalar(state.hp.kl_budget_val, N)
    m2_safe = np.maximum(new_m2, 1e-8)
    sqrt_v = np.sqrt(m2_safe)
    eta_kl_raw = np.sqrt(2.0 * kl_budget / m2_safe)
    eta_star_raw = np.where(new_shape < -1e-8, -1.0 / (sqrt_v * new_shape), np.inf)
    eta_raw = np.minimum(eta_star_raw, eta_kl_raw)
    new_eta = eta_raw * sqrt_v

    new_state = state._replace(
        advantage_second_moment_ema=jnp.asarray(new_m2.astype(np.float32)),
        advantage_third_moment_ema=jnp.asarray(new_m3.astype(np.float32)),
        dist_shift_covariance_ema=jnp.asarray(new_c.astype(np.float32)),
        dist_shift_shape_ema=jnp.asarray(new_shape.astype(np.float32)),
        tfg_eta=jnp.asarray(new_eta.astype(np.float32)),
    )
    return new_state, adv_per_env
