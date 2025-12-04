from dataclasses import dataclass
from typing import Dict

import numpy as np
import jax.numpy as jnp


@dataclass
class PcLevelsConfig:
    K: int = 40
    alpha_bar_min: float = 0.05
    alpha_bar_max: float = 1.0
    beta_schedule: str = "linear"
    lambda_scale: float = 1.5
    sigma_p: float = 2.0


def _alpha_bar_schedule(cfg: PcLevelsConfig) -> np.ndarray:
    K = int(cfg.K)
    lo = float(cfg.alpha_bar_min)
    hi = float(cfg.alpha_bar_max)
    if not (0.0 <= lo <= hi <= 1.0):
        raise ValueError("alpha_bar_min/max must satisfy 0 <= min <= max <= 1")
    return np.linspace(lo, hi, K, dtype=np.float64)


def _beta_schedule(cfg: PcLevelsConfig, alpha_bar: np.ndarray) -> np.ndarray:
    name = cfg.beta_schedule
    lam = float(cfg.lambda_scale)
    K = len(alpha_bar)
    if name == "linear":
        sched = np.linspace(0.0, lam, K, dtype=np.float64)
    elif name == "late_constant":
        frac = 0.7
        cut = int(np.floor(frac * (K - 1)))
        sched = np.zeros(K, dtype=np.float64)
        sched[cut:] = lam
    elif name == "late_ramp":
        frac = 0.7
        cut = int(np.floor(frac * (K - 1)))
        ramp_len = max(K - cut, 1)
        sched = np.zeros(K, dtype=np.float64)
        sched[cut:] = np.linspace(0.0, lam, ramp_len, dtype=np.float64)
    elif name == "sigma_power":
        p = float(cfg.sigma_p)
        sigma2 = np.maximum(1.0 - alpha_bar, 0.0)
        sched = lam * sigma2 ** p
    else:
        sched = np.linspace(0.0, lam, K, dtype=np.float64)
    return sched


def _eta_a_schedule(alpha_bar: np.ndarray) -> np.ndarray:
    eta = np.maximum(1.0 - alpha_bar, 1e-6)
    return eta.astype(np.float64)


def build_levels_jax(cfg: PcLevelsConfig) -> Dict[str, jnp.ndarray]:
    alpha_bar = _alpha_bar_schedule(cfg)
    beta_t = _beta_schedule(cfg, alpha_bar)
    eta_a = _eta_a_schedule(alpha_bar)
    out = {
        "K": int(cfg.K),
        "alpha_bar": jnp.asarray(alpha_bar, dtype=jnp.float32),
        "beta_t": jnp.asarray(beta_t, dtype=jnp.float32),
        "eta_a": jnp.asarray(eta_a, dtype=jnp.float32),
    }
    return out
