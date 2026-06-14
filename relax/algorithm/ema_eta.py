"""Host-side EMA + adaptive guidance-strength β update for the KL-budget /
on-policy path of MGMD.

Two pure functions, one per sampler dispatch mode in ``VmapOffPolicyTrainer``:

* :func:`update_state_m2_only` -- advantage-normalization path.
  Updates M = EMA(E[A²]) only, leaving β unchanged.

* :func:`update_state_kl_only` -- KL-budget-only β selection.
  β = sqrt(2δ/M) where M = EMA(E[A²]).  (Paper §5.1, eq. 2.)

* :func:`update_state_one_step` -- second-order one-step distribution-shift.
  β = min(β_KL, β*) where β_KL = sqrt(2δ/M) and β* = -1/(sqrt(M)·s).
  s = EMA((2γc + κ₃)/v^{3/2}) is the dimensionless shape.

Each function returns ``(new_state, adv_per_env)``: the trainer replaces
``algorithm.state`` with ``new_state`` and uses ``adv_per_env`` to roll the
one-step covariance buffer for the next call.
"""
from typing import Tuple

import numpy as np


def _ema(cur, batch, tau):
    return (1 - tau) * cur + tau * batch


def update_state_m2_only(state, q_per_env: np.ndarray, v_per_env: np.ndarray) -> Tuple[object, np.ndarray]:
    """Advantage-normalization path: update M = EMA(E[A²]) only."""
    adv_per_env = q_per_env - v_per_env
    m2_hat = np.mean(adv_per_env ** 2, axis=1)
    new_m2_ema = _ema(state.advantage_second_moment_ema, m2_hat, state.hp.adv_ema_tau)
    return state._replace(advantage_second_moment_ema=new_m2_ema), adv_per_env


def update_state_kl_only(state, q_per_env: np.ndarray, v_per_env: np.ndarray) -> Tuple[object, np.ndarray]:
    """KL-budget path: update M = EMA(E[A²]) and set β = sqrt(2δ/M)."""
    adv_per_env = q_per_env - v_per_env
    m2_hat = np.mean(adv_per_env ** 2, axis=1)

    new_m2_ema = _ema(state.advantage_second_moment_ema, m2_hat, state.hp.adv_ema_tau)
    new_beta = np.sqrt(2.0 * state.hp.kl_budget_val / np.maximum(new_m2_ema, 1e-6))

    return state._replace(advantage_second_moment_ema=new_m2_ema, beta=new_beta), adv_per_env


def update_state_one_step(state, q_per_env: np.ndarray, v_per_env: np.ndarray,
                          prev_adv_per_env: np.ndarray, prev_valid: np.ndarray) -> Tuple[object, np.ndarray]:
    """One-step distribution-shift path: update moment EMAs and set
    β = min(β_KL, β*) where β_KL = sqrt(2δ/M) and β* = -1/(sqrt(M)·s).  (Paper §5.2.)

    s = EMA((2γc + κ₃)/v^{3/2}) is the dimensionless shape; β* is finite
    only when s < 0 (sign condition); otherwise falls back to β_KL.
    """
    adv_per_env = q_per_env - v_per_env
    m2_hat = np.mean(adv_per_env ** 2, axis=1)
    m3_hat = np.mean(adv_per_env ** 3, axis=1)

    # s_hat construction
    # One-step covariance estimator ĉ = mean((A')²·A); only valid for
    # non-terminal transitions (A' from next step, A from current step).
    # On step 1, prev_valid is all-False so cont_count=0 everywhere and
    # any_continued=False, leaving c/shape EMAs unchanged.
    cont_count    = np.sum(prev_valid, axis=1, dtype=np.float32) # [num_runs]: # envs per run whose episode continued
    any_continued = cont_count > 0                               # [num_runs]: True if ≥1 env contributed
    c_hat = np.sum(prev_valid * (adv_per_env ** 2) * prev_adv_per_env, axis=1) / np.maximum(cont_count, 1)
    s_hat = (2.0 * state.hp.gamma * c_hat + m3_hat) / np.maximum(m2_hat, 1e-8) ** 1.5

    # EMA updates
    new_m2_ema = _ema(state.advantage_second_moment_ema, m2_hat, state.hp.adv_ema_tau)
    new_shape_ema = np.where(any_continued, _ema(state.dist_shift_shape_ema, s_hat, state.hp.shape_ema_tau), state.dist_shift_shape_ema)

    # new_beta calculation: β = min(β_KL, β*);  β* = -1/(sqrt(M)·s),  +∞ when s ≥ 0.
    sqrt_m2  = np.sqrt(np.maximum(new_m2_ema, 1e-6))
    beta_kl   = np.sqrt(2.0 * state.hp.kl_budget_val) / sqrt_m2
    beta_star = np.where(new_shape_ema < -1e-8, -1.0 / (sqrt_m2 * new_shape_ema), np.inf)
    new_beta  = np.minimum(beta_star, beta_kl)

    # EMA updates for logging
    new_m3_ema = _ema(state.advantage_third_moment_ema, m3_hat, state.hp.adv_ema_tau)
    new_c_ema     = np.where(any_continued, _ema(state.dist_shift_covariance_ema, c_hat,  state.hp.adv_ema_tau), state.dist_shift_covariance_ema)
    return state._replace(
        advantage_second_moment_ema=new_m2_ema,
        dist_shift_shape_ema       =new_shape_ema,
        beta                       =new_beta,
        advantage_third_moment_ema =new_m3_ema,
        dist_shift_covariance_ema  =new_c_ema,
    ), adv_per_env
