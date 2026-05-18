"""Host-side EMA + adaptive guidance-strength η update for the KL-budget /
on-policy path of DPMD.

Two pure functions, one per sampler dispatch mode in ``VmapOffPolicyTrainer``:

* :func:`update_state_kl_only` -- KL-budget-only η selection.
  η = sqrt(2δ/M) where M = EMA(E[A²]).  (Paper §5.1, eq. 2.)

* :func:`update_state_one_step` -- second-order one-step distribution-shift.
  η = min(η_KL, η*) where η_KL = sqrt(2δ/M) and η* = -1/(sqrt(M)·s).
  s = EMA((2γc + κ₃)/v^{3/2}) is the dimensionless shape.

Each function returns ``(new_state, adv_per_env)``: the trainer replaces
``algorithm.state`` with ``new_state`` and uses ``adv_per_env`` to roll the
one-step covariance buffer for the next call.
"""
from typing import Tuple

import numpy as np


def update_state_kl_only(state, q_per_env: np.ndarray, v_per_env: np.ndarray) -> Tuple[object, np.ndarray]:
    """KL-budget path: update M = EMA(E[A²]) and set η = sqrt(2δ/M)."""
    adv_per_env = q_per_env - v_per_env
    m2_batch    = np.mean(adv_per_env ** 2, axis=1)

    tau_v  = state.hp.adv_ema_tau
    new_m2 = (1 - tau_v) * state.advantage_second_moment_ema + tau_v * m2_batch
    new_eta = np.sqrt(2.0 * state.hp.kl_budget_val / np.maximum(new_m2, 1e-6))

    return state._replace(
        advantage_second_moment_ema=new_m2,
        tfg_eta                    =new_eta,
    ), adv_per_env


def update_state_one_step(state, q_per_env: np.ndarray, v_per_env: np.ndarray,
                          prev_adv_per_env, prev_valid) -> Tuple[object, np.ndarray]:
    """One-step distribution-shift path: update moment EMAs and set
    η = min(η_KL, η*) where η_KL = sqrt(2δ/M) and η* = -1/(sqrt(M)·s).  (Paper §5.2.)

    s = EMA((2γc + κ₃)/v^{3/2}) is the dimensionless shape; η* is finite
    only when s < 0 (sign condition); otherwise falls back to η_KL.
    """
    adv_per_env = q_per_env - v_per_env
    m2_batch    = np.mean(adv_per_env ** 2, axis=1)
    m3_batch    = np.mean(adv_per_env ** 3, axis=1)

    tau_v     = state.hp.adv_ema_tau
    new_m2    = (1 - tau_v) * state.advantage_second_moment_ema + tau_v * m2_batch
    new_m3    = (1 - tau_v) * state.advantage_third_moment_ema  + tau_v * m3_batch
    new_c     = state.dist_shift_covariance_ema
    new_shape = state.dist_shift_shape_ema

    # One-step covariance estimator ĉ = mean((A')²·A); only valid for
    # non-terminal transitions (A' from next step, A from current step).
    if prev_adv_per_env is not None and prev_valid is not None:
        valid_count   = np.sum(prev_valid, axis=1)
        c_batch_valid = valid_count > 0
        c_batch = np.where(
            c_batch_valid,
            np.sum(prev_valid.astype(np.float64) * (adv_per_env ** 2) * prev_adv_per_env, axis=1)
            / np.maximum(valid_count, 1),
            0.0,
        ).astype(np.float32)
        tau_s   = state.hp.shape_ema_tau
        gamma   = state.hp.gamma
        s_batch = (2.0 * gamma * c_batch + m3_batch) / np.maximum(m2_batch, 1e-8) ** 1.5

        new_c     = np.where(c_batch_valid, (1 - tau_v) * new_c     + tau_v * c_batch,  new_c)
        new_shape = np.where(c_batch_valid, (1 - tau_s) * new_shape + tau_s * s_batch, new_shape)

    # η = min(η_KL, η*);  η* = -1/(sqrt(M)·s),  +∞ when s ≥ 0.
    sqrt_m2  = np.sqrt(np.maximum(new_m2, 1e-6))
    eta_kl   = np.sqrt(2.0 * state.hp.kl_budget_val) / sqrt_m2
    eta_star = np.where(new_shape < -1e-8, -1.0 / (sqrt_m2 * new_shape), np.inf)
    new_eta  = np.minimum(eta_star, eta_kl)

    return state._replace(
        advantage_second_moment_ema=new_m2,
        advantage_third_moment_ema =new_m3,
        dist_shift_covariance_ema  =new_c,
        dist_shift_shape_ema       =new_shape,
        tfg_eta                    =new_eta,
    ), adv_per_env
