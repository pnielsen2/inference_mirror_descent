#!/usr/bin/env python3
"""Sweep 80 analysis: training curve ablation plots, pairwise matrix plots,
best-config vs baselines, and per-config matrix plots.

Uses netscratch offline wandb run dirs for HP values and local CSV files for
episode returns. No internet connection required.

Outputs in --out-dir (default: figures/sweep80/):
  sweep80_ablation_curves_{axis}.png      one per HP axis (3 files)
  sweep80_pair_matrix_{metric}.png        pairwise ablation matrices (2 files)
  sweep80_best_vs_baselines_{metric}.png  best config vs LSAC baselines (2 files)
  sweep80_config_matrix_{metric}.png      tfg_eta x q_agg_sample (2 files)

Usage:
    python scripts/plot_sweep80_analysis.py
    python scripts/plot_sweep80_analysis.py --min-steps 900000 --out-dir /path/to/figs
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import plot_sweep45_kl1024_vs_baselines_vs_dpmd as sweep45_plot
from scipy import stats

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_DIR = SCRIPT_DIR.parent
FIG_DIR = REPO_DIR / "figures"
LSAC_DATA_DIR = Path.home() / "LSAC" / "data"
NETSCRATCH_WANDB_ROOT = Path("/n/netscratch/kdbrantley_lab/Lab/pnielsen/wandb")

SWEEP_ID = 80
ENVS = ["Ant-v3", "HalfCheetah-v3", "Hopper-v3", "Humanoid-v3", "Swimmer-v3", "Walker2d-v3"]
HP_AXES = ["q_agg_sample", "q_td_huber_width", "tfg_eta"]

CORRECTED_SWEEP_FILTERS = {
    "alg": "dpmd",
    "num_vec_envs": 5,
    "reward_scale": 1.0,
    "dpmd_constant_weight": True,
    "num_particles": 1,
    "mala_steps": 2,
    "beta_schedule_type": "cosine",
    "beta_schedule_scale": 1.0,
    "dpmd_no_entropy_tuning": True,
    "buffer_size": 400000,
    "x0_hat_clip_radius": 3.0,
    "mala_adapt_rate": 0.2,
    "mala_per_level_eta": True,
    "update_per_iteration": 4,
    "lr_q": 0.00015,
    "lr_policy": 0.0003,
    "mala_guided_predictor": True,
    "ddim_predictor": True,
    "tau": 0.005,
    "advantage_ema_tau": 0.001,
    "shape_ema_tau": 0.0002,
    "initial_advantage_second_moment_ema": 1.0,
    "gamma": 0.99,
    "parallel_runs": 12,
    "config_tag_keys": "q_agg_sample,q_td_huber_width,tfg_eta",
}

BIN_SIZE = 10_000
MAX_STEPS = 1_000_000
TAIL_WINDOW_STEPS = 50_000
CI_LEVEL = 0.90
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 0

BASELINE_ALGOS = ["SAC", "TD3", "DIPO", "PPO", "TRPO"]
METHOD_COLORS = {"SAC": "C1", "TD3": "C2", "DIPO": "C3", "PPO": "C4", "TRPO": "C5"}
LSAC_N_SEEDS = 10
SWEEP80_COLOR = "C0"
ABLATION_PALETTE = plt.cm.tab10.colors
BEST_VS_MGMD_COLOR = "C6"

# ---------------------------------------------------------------------------
# Netscratch offline wandb run loading
# ---------------------------------------------------------------------------


def _parse_config_yaml(config_path: Path) -> dict | None:
    """Parse wandb offline run files/config.yaml; return HP dict or None."""
    import yaml

    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None

    def _val(key):
        entry = cfg.get(key)
        return entry.get("value") if isinstance(entry, dict) else entry

    cfg["_value"] = _val

    try:
        return {
            "alg": str(_val("alg")),
            "env": str(_val("env")),
            "num_vec_envs": int(_val("num_vec_envs")),
            "reward_scale": float(_val("reward_scale")),
            "dpmd_constant_weight": bool(_val("dpmd_constant_weight")),
            "num_particles": int(_val("num_particles")),
            "mala_steps": int(_val("mala_steps")),
            "beta_schedule_type": str(_val("beta_schedule_type")),
            "beta_schedule_scale": float(_val("beta_schedule_scale")),
            "dpmd_no_entropy_tuning": bool(_val("dpmd_no_entropy_tuning")),
            "buffer_size": int(_val("buffer_size")),
            "x0_hat_clip_radius": float(_val("x0_hat_clip_radius")),
            "mala_adapt_rate": float(_val("mala_adapt_rate")),
            "mala_per_level_eta": bool(_val("mala_per_level_eta")),
            "update_per_iteration": int(_val("update_per_iteration")),
            "lr_q": float(_val("lr_q")),
            "lr_policy": float(_val("lr_policy")),
            "mala_guided_predictor": bool(_val("mala_guided_predictor")),
            "ddim_predictor": bool(_val("ddim_predictor")),
            "tau": float(_val("tau")),
            "advantage_ema_tau": float(_val("advantage_ema_tau")),
            "shape_ema_tau": float(_val("shape_ema_tau")),
            "initial_advantage_second_moment_ema": float(_val("initial_advantage_second_moment_ema")),
            "gamma": float(_val("gamma")),
            "parallel_runs": int(_val("parallel_runs")),
            "config_tag_keys": str(_val("config_tag_keys")),
            "sweep_id": int(_val("sweep_id")),
            "q_agg_sample": str(_val("q_agg_sample")),
            "q_td_huber_width": float(_val("q_td_huber_width")),
            "tfg_eta": float(_val("tfg_eta")),
            "seed": int(_val("seed")),
            "seed_index": int(_val("seed_index")),
        }
    except (TypeError, ValueError):
        return None


def _matches_corrected_launch(hp: dict, sweep_id: int) -> bool:
    if hp.get("sweep_id") != sweep_id:
        return False
    if hp.get("env") not in ENVS:
        return False
    if hp.get("q_agg_sample") not in {"min", "mean"}:
        return False
    if hp.get("tfg_eta") not in {22.5, 45.0, 90.0}:
        return False
    q_hw = hp.get("q_td_huber_width")
    if not (q_hw == 30.0 or math.isinf(q_hw)):
        return False
    for key, expected in CORRECTED_SWEEP_FILTERS.items():
        actual = hp.get(key)
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                return False
        else:
            if actual != expected:
                return False
    return True


def _history_item_key(item) -> str | None:
    if item.key:
        return item.key
    if item.nested_key:
        return "/".join(item.nested_key)
    return None


def _load_wandb_episode_history(wandb_file: Path, env: str) -> pd.DataFrame:
    from wandb.sdk.internal import datastore
    from wandb.proto import wandb_internal_pb2 as pb

    target_key = f"episode_return/{env}"
    rows: list[tuple[int, float]] = []

    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_file))
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = pb.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue

        step = None
        ret = None
        for item in rec.history.item:
            key = _history_item_key(item)
            if key is None or item.value_json == "":
                continue
            if key == "_step":
                try:
                    step = int(float(json.loads(item.value_json)))
                except Exception:
                    step = None
            elif key == target_key:
                try:
                    ret = float(json.loads(item.value_json))
                except Exception:
                    ret = None
        if step is not None and ret is not None and math.isfinite(ret):
            rows.append((step, ret))

    if not rows:
        return pd.DataFrame(columns=["_step", "return"])

    hist = pd.DataFrame(rows, columns=["_step", "return"]).sort_values("_step")
    hist = hist.drop_duplicates(subset=["_step", "return"]).reset_index(drop=True)
    return hist


def load_sweep_runs(sweep_id: int, min_steps: float = 900_000) -> list[dict]:
    """Load corrected sweep runs from local offline wandb directories.

    Uses only per-run local wandb metadata and local wandb history records,
    filtering to runs that exactly match the corrected sweep launch template.
    Runs with fewer than min_steps env steps are filtered out.
    """
    sweep_dir = NETSCRATCH_WANDB_ROOT / f"sweep_{sweep_id}"
    if not sweep_dir.exists():
        print(f"  Sweep dir not found: {sweep_dir}", flush=True)
        return []

    runs_by_key: dict[tuple, dict] = {}

    for job_dir in sorted(sweep_dir.iterdir()):
        if not job_dir.is_dir() or not job_dir.name.startswith("job_"):
            continue
        wandb_dir = job_dir / "wandb"
        if not wandb_dir.exists():
            continue

        for run_dir in sorted(wandb_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("offline-run-"):
                continue

            wandb_files = [
                f for f in run_dir.iterdir()
                if f.suffix == ".wandb" and not f.name.endswith(".wandb.synced")
            ]
            if not wandb_files:
                continue
            wandb_file = wandb_files[0]

            config_path = run_dir / "files" / "config.yaml"
            if not config_path.exists():
                continue
            hp = _parse_config_yaml(config_path)
            if hp is None or not _matches_corrected_launch(hp, sweep_id):
                continue
            env = hp["env"]
            hist = _load_wandb_episode_history(wandb_file, env)
            if len(hist) == 0:
                continue
            max_step = float(hist["_step"].max())
            if max_step < min_steps:
                print(
                    f"  Skip ({int(max_step)} steps): env={env} seed={hp['seed']}"
                    f" agg={hp['q_agg_sample']} hw={_level_label(hp['q_td_huber_width'])} eta={hp['tfg_eta']:g}",
                    flush=True,
                )
                continue

            run_key = (env, hp["seed"], hp["q_agg_sample"], hp["q_td_huber_width"], hp["tfg_eta"])
            run = dict(
                env=env,
                seed=hp["seed"],
                q_agg_sample=hp["q_agg_sample"],
                q_td_huber_width=hp["q_td_huber_width"],
                tfg_eta=hp["tfg_eta"],
                hist=hist,
            )
            prev = runs_by_key.get(run_key)
            if prev is None or float(prev["hist"]["_step"].max()) < max_step:
                runs_by_key[run_key] = run

    runs = list(runs_by_key.values())
    print(f"Loaded {len(runs)} runs for sweep {sweep_id}", flush=True)
    for env in ENVS:
        n = sum(1 for r in runs if r["env"] == env)
        print(f"  {env}: {n} runs", flush=True)
    return runs


# ---------------------------------------------------------------------------
# Curve helpers
# ---------------------------------------------------------------------------


def windowed_mean_curve(hist: pd.DataFrame, bins: np.ndarray) -> np.ndarray:
    """Mean return of episodes ending in each BIN_SIZE-step window."""
    steps = hist["_step"].to_numpy(dtype=float)
    returns = hist["return"].to_numpy(dtype=float)
    curve = np.full(len(bins), np.nan)
    for i, t in enumerate(bins):
        mask = (steps > float(t) - BIN_SIZE) & (steps <= float(t))
        if mask.any():
            curve[i] = float(np.nanmean(returns[mask]))
    return curve


def aggregate_t_ci(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean ± half-width of 90% t-CI across a list of curves."""
    arr = np.stack(curves)
    counts = np.sum(~np.isnan(arr), axis=0)
    mean = np.nanmean(arr, axis=0)
    mean[counts == 0] = np.nan
    ci = np.full(arr.shape[1], np.nan)
    ci[counts == 1] = 0.0
    for n in sorted({int(v) for v in counts if v > 1}):
        mask = counts == n
        std = np.nanstd(arr[:, mask], axis=0, ddof=1)
        t_crit = stats.t.ppf(1.0 - (1.0 - CI_LEVEL) / 2.0, df=n - 1)
        ci[mask] = t_crit * std / math.sqrt(float(n))
    return mean, ci


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def tail_mean(hist: pd.DataFrame) -> float:
    steps = hist["_step"].to_numpy(dtype=float)
    returns = hist["return"].to_numpy(dtype=float)
    if len(steps) == 0:
        return math.nan
    mask = steps >= steps.max() - TAIL_WINDOW_STEPS
    return float(np.mean(returns[mask])) if mask.any() else math.nan


def exp_log_score(score: float, benchmark: float) -> float:
    """clip(score / benchmark, 0.01, 1.0) — the linear form of log_score."""
    if not math.isfinite(score) or not math.isfinite(benchmark) or benchmark <= 0:
        return math.nan
    return float(np.clip(score / benchmark, 0.01, 1.0))


def geomean(xs) -> float:
    pos = [x for x in xs if x is not None and math.isfinite(float(x)) and float(x) > 0]
    if not pos:
        return math.nan
    return float(np.exp(np.mean(np.log(pos))))


def bootstrap_geomean_ci(xs, rng) -> tuple[float, float]:
    pos = np.array(
        [x for x in xs if x is not None and math.isfinite(float(x)) and float(x) > 0],
        dtype=float,
    )
    if len(pos) < 2:
        return math.nan, math.nan
    logs = np.log(pos)
    idx = rng.integers(0, len(logs), size=(BOOTSTRAP_SAMPLES, len(logs)))
    boot = np.exp(logs[idx].mean(axis=1))
    lo, hi = np.quantile(boot, [(1 - CI_LEVEL) / 2, 1 - (1 - CI_LEVEL) / 2])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# LSAC baseline helpers
# ---------------------------------------------------------------------------


def load_lsac_df(env: str) -> pd.DataFrame | None:
    key = env.split("-")[0].lower()
    pkl = LSAC_DATA_DIR / f"all_data_{key}.pkl"
    if not pkl.exists():
        return None
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    return pd.read_csv(StringIO(data))


def benchmark_score(df: pd.DataFrame) -> float | None:
    t_crit = stats.t.ppf(1 - (1 - CI_LEVEL) / 2, df=LSAC_N_SEEDS - 1)
    best = None
    for algo in BASELINE_ALGOS:
        ad = df[df["algo"] == algo].sort_values("steps")
        if ad.empty:
            continue
        steps = ad["steps"].to_numpy(float)
        means = ad["rew_mean"].to_numpy(float)
        stds = ad["rew_std"].to_numpy(float)
        upper = means + t_crit * stds / math.sqrt(LSAC_N_SEEDS)
        tail = upper[steps > steps.max() - TAIL_WINDOW_STEPS]
        if tail.size == 0:
            tail = upper[-1:]
        s = float(np.mean(tail))
        if best is None or s > best:
            best = s
    return best


def load_all_benchmarks(envs) -> dict[str, float]:
    out: dict[str, float] = {}
    for env in envs:
        df = load_lsac_df(env)
        if df is None:
            print(f"  Warning: no LSAC data for {env}", flush=True)
            continue
        s = benchmark_score(df)
        if s is not None and math.isfinite(s):
            out[env] = s
    return out


def _plot_baselines(ax, df: pd.DataFrame):
    t_crit = stats.t.ppf(1 - (1 - CI_LEVEL) / 2, df=LSAC_N_SEEDS - 1)
    for algo in BASELINE_ALGOS:
        if algo not in df["algo"].values:
            continue
        ad = df[df["algo"] == algo].sort_values("steps")
        x = ad["steps"].values / 1e6
        means = ad["rew_mean"].values
        cis = t_crit * ad["rew_std"].values / math.sqrt(LSAC_N_SEEDS)
        color = METHOD_COLORS.get(algo, "gray")
        ax.plot(x, means, label=algo, linewidth=1.2, color=color)
        ax.fill_between(x, means - cis, means + cis, alpha=0.15, color=color)


def _baseline_y_bounds(df: pd.DataFrame) -> list[np.ndarray]:
    t_crit = stats.t.ppf(1 - (1 - CI_LEVEL) / 2, df=LSAC_N_SEEDS - 1)
    parts: list[np.ndarray] = []
    for algo in BASELINE_ALGOS:
        if algo not in df["algo"].values:
            continue
        ad = df[df["algo"] == algo].sort_values("steps")
        means = ad["rew_mean"].values
        cis = t_crit * ad["rew_std"].values / math.sqrt(LSAC_N_SEEDS)
        parts.extend([means - cis, means + cis])
    return parts


def _fit_y_axis(ax, y_parts):
    arrs = []
    for p in y_parts:
        a = np.asarray(p, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            arrs.append(a)
    if not arrs:
        return
    vals = np.concatenate(arrs)
    ymin, ymax = float(vals.min()), float(vals.max())
    if ymin == ymax:
        pad = max(1.0, 0.05 * max(abs(ymin), 1.0))
    else:
        pad = 0.05 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)


# ---------------------------------------------------------------------------
# Part 1: Training curve ablation plots
# ---------------------------------------------------------------------------


def _level_label(val) -> str:
    if isinstance(val, float):
        return "inf" if math.isinf(val) else f"{val:g}"
    return str(val)


def _level_sort_key(val):
    if isinstance(val, float):
        return (0, 1 if math.isinf(val) else 0, val)
    return (1, 0, str(val))


def plot_ablation_curves(runs: list[dict], bins: np.ndarray, out_dir: Path):
    """Plot one 2x3-grid PNG per HP axis showing average training curve per level."""
    for axis in HP_AXES:
        levels = sorted(
            {r[axis] for r in runs if r[axis] is not None},
            key=_level_sort_key,
        )

        fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
        axes_flat = axes.flatten()

        for ei, env in enumerate(ENVS):
            ax = axes_flat[ei]
            y_parts: list = []
            for li, level in enumerate(levels):
                group = [r for r in runs if r["env"] == env and r[axis] == level]
                if not group:
                    continue
                curves = [windowed_mean_curve(r["hist"], bins) for r in group]
                mean, ci = aggregate_t_ci(curves)
                x = bins / 1e6
                valid = ~np.isnan(mean)
                color = ABLATION_PALETTE[li % len(ABLATION_PALETTE)]
                label = f"{_level_label(level)} (n={len(group)})"
                ax.plot(x[valid], mean[valid], color=color, linewidth=1.6, label=label)
                band = valid & ~np.isnan(ci)
                if band.any():
                    ax.fill_between(
                        x[band], mean[band] - ci[band], mean[band] + ci[band],
                        color=color, alpha=0.20,
                    )
                if valid.any():
                    y_parts.extend([mean[valid]])
                    if not np.all(np.isnan(ci[valid])):
                        y_parts.extend([mean[valid] - ci[valid], mean[valid] + ci[valid]])

            ax.set_title(env.replace("-v3", ""), fontsize=12)
            ax.set_xlabel("Env steps (M)", fontsize=10)
            if ei % 3 == 0:
                ax.set_ylabel("Episode return", fontsize=10)
            ax.set_xlim(0, MAX_STEPS / 1e6)
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(labelsize=9)
            if y_parts:
                _fit_y_axis(ax, y_parts)

        handles, labels = [], []
        for ax in axes_flat:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        fig.legend(handles, labels, loc="lower center",
                   ncol=max(1, len(handles)), fontsize=10,
                   bbox_to_anchor=(0.5, 0.0))
        fig.suptitle(f"Sweep 80 — ablation axis: {axis}", fontsize=13, y=1.01)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        out = out_dir / f"sweep80_ablation_curves_{axis}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}", flush=True)


# ---------------------------------------------------------------------------
# Part 2: Compute run-level and config-level metrics
# ---------------------------------------------------------------------------


def compute_run_metrics(runs: list[dict], benchmarks: dict[str, float]):
    """Attach tail_score and els (exp log score) to each run dict in-place."""
    for r in runs:
        r["tail_score"] = tail_mean(r["hist"])
        bench = benchmarks.get(r["env"], 0.0)
        r["els"] = exp_log_score(r["tail_score"], bench)


def _config_key(r: dict) -> tuple:
    return (r["q_agg_sample"], r["q_td_huber_width"], r["tfg_eta"])


def compute_config_scores(runs: list[dict]) -> dict[tuple, dict]:
    """Return config_key -> {exp_log_score, geomean_score, hp, n_runs}."""
    groups: dict[tuple, list] = defaultdict(list)
    for r in runs:
        groups[_config_key(r)].append(r)

    out: dict[tuple, dict] = {}
    for cfg, cfg_runs in groups.items():
        els_vals = [r["els"] for r in cfg_runs if math.isfinite(r.get("els", math.nan))]
        env_tail: dict[str, list] = defaultdict(list)
        for r in cfg_runs:
            ts = r["tail_score"]
            if math.isfinite(ts) and ts > 0:
                env_tail[r["env"]].append(ts)
        per_env_means = [float(np.mean(v)) for v in env_tail.values() if v]
        out[cfg] = {
            "exp_log_score": geomean(els_vals),
            "geomean_score": geomean(per_env_means),
            "hp": {k: cfg_runs[0][k] for k in HP_AXES},
            "n_runs": len(cfg_runs),
        }
    return out


def _best_config_entry(config_scores: dict, metric: str):
    valid = {k: v for k, v in config_scores.items() if math.isfinite(v[metric])}
    if not valid:
        return None, None
    best_key = max(valid, key=lambda k: valid[k][metric])
    return best_key, valid[best_key]


def _prepare_smoothed_runs(hists: list[pd.DataFrame]) -> list[pd.DataFrame]:
    out = []
    for hist in hists:
        canon = sweep45_plot.canonicalize_hist(hist)
        if canon is not None:
            out.append(sweep45_plot.smooth_hist(canon))
    return out


def _load_local_mgmd_histories() -> dict[str, list[pd.DataFrame]]:
    root = NETSCRATCH_WANDB_ROOT / "sweep_45"
    histories: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for env, run_ids in sweep45_plot.SWEEP45_RUN_IDS.items():
        for run_id in run_ids:
            matches = sorted(root.glob(f"job_*/wandb/offline-run-*-{run_id}"))
            if not matches:
                print(f"  Warning: local MGMD run not found: {run_id}", flush=True)
                continue
            run_dir = matches[-1]
            wandb_files = [
                f for f in run_dir.iterdir()
                if f.suffix == ".wandb" and not f.name.endswith(".wandb.synced")
            ]
            if not wandb_files:
                continue
            hist = _load_wandb_episode_history(wandb_files[0], env)
            prepared = _prepare_smoothed_runs([hist])
            if prepared:
                histories[env].extend(prepared)
    if "Humanoid-v3" in histories:
        histories["Humanoid-v3"] = sweep45_plot.rescale_to_target_max(
            histories["Humanoid-v3"],
            float(MAX_STEPS),
        )
    return dict(histories)


def _best_config_histories_for_interp(runs: list[dict], best_key: tuple) -> dict[str, list[pd.DataFrame]]:
    q_agg, q_hw, tfg = best_key
    by_env: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for r in runs:
        if r["q_agg_sample"] == q_agg and r["q_td_huber_width"] == q_hw and r["tfg_eta"] == tfg:
            prepared = _prepare_smoothed_runs([r["hist"]])
            if prepared:
                by_env[r["env"]].extend(prepared)
    if "Humanoid-v3" in by_env:
        by_env["Humanoid-v3"] = sweep45_plot.rescale_to_target_max(
            by_env["Humanoid-v3"],
            float(MAX_STEPS),
        )
    return dict(by_env)


# ---------------------------------------------------------------------------
# Part 3: Pairwise matrix plots
# ---------------------------------------------------------------------------

_MATRIX_AXES = ["env", "q_agg_sample", "q_td_huber_width", "tfg_eta"]
_MATRIX_PAIRS = [
    (a, b)
    for i, a in enumerate(_MATRIX_AXES)
    for b in _MATRIX_AXES[i + 1:]
]


def _getval(r: dict, axis: str):
    return r["env"] if axis == "env" else r[axis]


def _axis_levels(runs: list[dict], axis: str) -> list:
    return sorted({_getval(r, axis) for r in runs}, key=_level_sort_key)


def _cell_metric(
    runs: list[dict], metric: str, rng
) -> tuple[float, float, float]:
    """Return (point, ci_lo, ci_hi) geometric mean for the metric over these runs."""
    if metric == "exp_log_score":
        vals = [r["els"] for r in runs if math.isfinite(r.get("els", math.nan))]
    else:
        env_tail: dict[str, list] = defaultdict(list)
        for r in runs:
            ts = r["tail_score"]
            if math.isfinite(ts) and ts > 0:
                env_tail[r["env"]].append(ts)
        vals = [float(np.mean(v)) for v in env_tail.values() if v]
    pt = geomean(vals)
    lo, hi = bootstrap_geomean_ci(vals, rng)
    return pt, lo, hi


def _draw_matrix(ax, rows_lvl, cols_lvl, mat, norm, cmap, ax_row: str, ax_col: str):
    for i, rv in enumerate(rows_lvl):
        for j, cv in enumerate(cols_lvl):
            pt, lo, hi = mat[(rv, cv)]
            color = cmap(norm(pt)) if math.isfinite(pt) else (0.8, 0.8, 0.8, 1.0)
            ax.add_patch(plt.Rectangle([j - 0.5, i - 0.5], 1, 1, color=color, zorder=0))
            text = f"{pt:.3g}\n[{lo:.3g},{hi:.3g}]" if math.isfinite(pt) else "N/A"
            r, g, b = color[0], color[1], color[2]
            tc = "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white"
            ax.text(j, i, text, ha="center", va="center", fontsize=6.5, color=tc)
    ax.set_xlim(-0.5, len(cols_lvl) - 0.5)
    ax.set_ylim(-0.5, len(rows_lvl) - 0.5)
    ax.set_xticks(range(len(cols_lvl)))
    ax.set_yticks(range(len(rows_lvl)))
    ax.set_xticklabels([_level_label(v) for v in cols_lvl], fontsize=7.5, rotation=30, ha="right")
    ax.set_yticklabels([_level_label(v) for v in rows_lvl], fontsize=7.5)
    ax.set_xlabel(ax_col, fontsize=8.5)
    ax.set_ylabel(ax_row, fontsize=8.5)
    ax.set_title(f"{ax_row} × {ax_col}", fontsize=9)


def plot_pair_matrices(
    runs: list[dict], metric: str, metric_label: str, out_path: Path, rng
):
    ncols = 3
    nrows = math.ceil(len(_MATRIX_PAIRS) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    all_pts: list[float] = []
    all_mats = []
    for ax_row, ax_col in _MATRIX_PAIRS:
        lvl_rows = _axis_levels(runs, ax_row)
        lvl_cols = _axis_levels(runs, ax_col)
        mat: dict = {}
        for rv in lvl_rows:
            for cv in lvl_cols:
                group = [r for r in runs if _getval(r, ax_row) == rv and _getval(r, ax_col) == cv]
                mat[(rv, cv)] = _cell_metric(group, metric, rng) if group else (math.nan, math.nan, math.nan)
                if math.isfinite(mat[(rv, cv)][0]):
                    all_pts.append(mat[(rv, cv)][0])
        all_mats.append((ax_row, ax_col, lvl_rows, lvl_cols, mat))

    vmin = min(all_pts) if all_pts else 0.0
    vmax = max(all_pts) if all_pts else 1.0
    if vmin == vmax:
        vmin -= 0.01
        vmax += 0.01
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn

    for idx, (ax_row, ax_col, lvl_rows, lvl_cols, mat) in enumerate(all_mats):
        row, col = divmod(idx, ncols)
        _draw_matrix(axes[row, col], lvl_rows, lvl_cols, mat, norm, cmap, ax_row, ax_col)

    for idx in range(len(all_mats), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, label=metric_label)
    fig.suptitle(
        f"Sweep 80 — {metric_label}: pairwise ablation matrices\n"
        f"(90% bootstrap CI, {BOOTSTRAP_SAMPLES} samples)",
        fontsize=11,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Part 4: Best-config vs baselines
# ---------------------------------------------------------------------------


def plot_best_vs_baselines(
    runs: list[dict],
    config_scores: dict,
    metric: str,
    metric_label: str,
    bins: np.ndarray,
    out_dir: Path,
):
    valid = {k: v for k, v in config_scores.items() if math.isfinite(v[metric])}
    if not valid:
        print(f"  No valid configs for metric {metric}", flush=True)
        return
    best_key = max(valid, key=lambda k: valid[k][metric])
    best_hp = valid[best_key]["hp"]
    best_score = valid[best_key][metric]
    q_agg, q_hw, tfg = best_key
    hw_label = "inf" if isinstance(q_hw, float) and math.isinf(q_hw) else f"{q_hw:g}"
    series_label = f"Sweep80 (agg={q_agg}, hw={hw_label}, η={tfg:g})"
    print(
        f"  Best config ({metric_label}): score={best_score:.4f}  {series_label}",
        flush=True,
    )

    best_runs = [
        r for r in runs
        if r["q_agg_sample"] == q_agg and r["q_td_huber_width"] == q_hw and r["tfg_eta"] == tfg
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    axes_flat = axes.flatten()

    for ei, env in enumerate(ENVS):
        ax = axes_flat[ei]
        y_parts: list = []

        env_runs = [r for r in best_runs if r["env"] == env]
        if env_runs:
            curves = [windowed_mean_curve(r["hist"], bins) for r in env_runs]
            mean, ci = aggregate_t_ci(curves)
            x = bins / 1e6
            valid_mask = ~np.isnan(mean)
            ax.plot(x[valid_mask], mean[valid_mask], color=SWEEP80_COLOR,
                    linewidth=2.0, label=series_label, zorder=3)
            band = valid_mask & ~np.isnan(ci)
            if band.any():
                ax.fill_between(
                    x[band], mean[band] - ci[band], mean[band] + ci[band],
                    color=SWEEP80_COLOR, alpha=0.2, zorder=2,
                )
            y_parts.extend([mean[valid_mask]])
            if not np.all(np.isnan(ci[valid_mask])):
                y_parts.extend([mean[valid_mask] - ci[valid_mask],
                                 mean[valid_mask] + ci[valid_mask]])

        bl_df = load_lsac_df(env)
        if bl_df is not None:
            _plot_baselines(ax, bl_df)
            y_parts.extend(_baseline_y_bounds(bl_df))

        ax.set_title(env.replace("-v3", ""), fontsize=12)
        ax.set_xlabel("Env steps (M)", fontsize=10)
        if ei % 3 == 0:
            ax.set_ylabel("Episode return", fontsize=10)
        ax.set_xlim(0, MAX_STEPS / 1e6)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=9)
        if y_parts:
            _fit_y_axis(ax, y_parts)

    handles, labels = [], []
    for ax in axes_flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(handles), 6), fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"Sweep 80 best config ({metric_label}) vs baselines\n{series_label}",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    out = out_dir / f"sweep80_best_vs_baselines_{metric}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}", flush=True)


def plot_best_vs_mgmd(
    runs: list[dict],
    config_scores: dict,
    metric: str,
    metric_label: str,
    out_dir: Path,
):
    best_key, best_entry = _best_config_entry(config_scores, metric)
    if best_key is None or best_entry is None:
        print(f"  No valid configs for metric {metric}", flush=True)
        return

    q_agg, q_hw, tfg = best_key
    hw_label = "inf" if isinstance(q_hw, float) and math.isinf(q_hw) else f"{q_hw:g}"
    best_label = f"Sweep80 (agg={q_agg}, hw={hw_label}, η={tfg:g})"
    mgmd_histories = _load_local_mgmd_histories()
    best_histories = _best_config_histories_for_interp(runs, best_key)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    axes_flat = axes.flatten()

    for ei, env in enumerate(ENVS):
        ax = axes_flat[ei]
        y_parts: list[np.ndarray] = []

        mgmd_runs = mgmd_histories.get(env, [])
        if mgmd_runs:
            sweep45_plot.plot_interpolated_series(
                ax,
                env,
                mgmd_runs,
                sweep45_plot.SWEEP45_LABEL,
                sweep45_plot.SWEEP45_COLOR,
                200,
                CI_LEVEL,
                y_parts,
            )

        best_runs_env = best_histories.get(env, [])
        if best_runs_env:
            sweep45_plot.plot_interpolated_series(
                ax,
                env,
                best_runs_env,
                best_label,
                BEST_VS_MGMD_COLOR,
                200,
                CI_LEVEL,
                y_parts,
            )

        ax.set_xlim(0, MAX_STEPS / 1e6)
        ax.set_title(env.replace("-v3", ""), fontsize=12)
        ax.set_xlabel("Env steps (M)", fontsize=10)
        if ei % 3 == 0:
            ax.set_ylabel("Episode return", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=9)
        if y_parts:
            _fit_y_axis(ax, y_parts)

    handles, labels = [], []
    for ax in axes_flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(handles), 6), fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        f"Sweep 80 best config ({metric_label}) vs sweep45 MGMD\n{best_label}",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    out = out_dir / f"sweep80_best_vs_mgmd_{metric}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}", flush=True)


# ---------------------------------------------------------------------------
# Part 5: Per-config matrix (tfg_eta × q_agg_sample, split by q_td_huber_width)
# ---------------------------------------------------------------------------


def _config_cell_metric(runs: list[dict], metric: str, rng) -> tuple[float, float, float]:
    """Metric for a single (q_td_huber_width, tfg_eta, q_agg_sample) cell."""
    if metric == "exp_log_score":
        vals = [r["els"] for r in runs if math.isfinite(r.get("els", math.nan))]
    else:
        env_tail: dict[str, list] = defaultdict(list)
        for r in runs:
            ts = r["tail_score"]
            if math.isfinite(ts) and ts > 0:
                env_tail[r["env"]].append(ts)
        vals = [float(np.mean(v)) for v in env_tail.values() if v]
    pt = geomean(vals)
    lo, hi = bootstrap_geomean_ci(vals, rng)
    return pt, lo, hi


def plot_config_matrices(
    runs: list[dict], metric: str, metric_label: str, out_path: Path, rng
):
    hw_vals = sorted(
        {r["q_td_huber_width"] for r in runs},
        key=lambda v: (1 if isinstance(v, float) and math.isinf(v) else 0, v),
    )
    eta_vals = sorted({r["tfg_eta"] for r in runs}, key=_level_sort_key)
    agg_vals = sorted({r["q_agg_sample"] for r in runs})

    all_pts: list[float] = []
    hw_mats = []
    for hw in hw_vals:
        hw_runs = [r for r in runs if r["q_td_huber_width"] == hw]
        mat: dict = {}
        for eta in eta_vals:
            for agg in agg_vals:
                group = [r for r in hw_runs if r["tfg_eta"] == eta and r["q_agg_sample"] == agg]
                mat[(eta, agg)] = _config_cell_metric(group, metric, rng) if group else (math.nan, math.nan, math.nan)
                if math.isfinite(mat[(eta, agg)][0]):
                    all_pts.append(mat[(eta, agg)][0])
        hw_mats.append((hw, mat))

    vmin = min(all_pts) if all_pts else 0.0
    vmax = max(all_pts) if all_pts else 1.0
    if vmin == vmax:
        vmin -= 0.01
        vmax += 0.01
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.RdYlGn

    n = len(hw_vals)
    fig, axes = plt.subplots(1, n, figsize=(5 * n + 1.5, 5 + 0.5 * len(eta_vals)))
    if n == 1:
        axes = [axes]

    for k, (hw, mat) in enumerate(hw_mats):
        ax = axes[k]
        hw_label = "inf" if isinstance(hw, float) and math.isinf(hw) else f"{hw:g}"
        for i, eta in enumerate(eta_vals):
            for j, agg in enumerate(agg_vals):
                pt, lo, hi = mat[(eta, agg)]
                color = cmap(norm(pt)) if math.isfinite(pt) else (0.8, 0.8, 0.8, 1.0)
                ax.add_patch(
                    plt.Rectangle([j - 0.5, i - 0.5], 1, 1, color=color, zorder=0)
                )
                text = f"{pt:.3g}\n[{lo:.3g},{hi:.3g}]" if math.isfinite(pt) else "N/A"
                r, g, b = color[0], color[1], color[2]
                tc = "black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white"
                ax.text(j, i, text, ha="center", va="center", fontsize=9, color=tc)
        ax.set_xlim(-0.5, len(agg_vals) - 0.5)
        ax.set_ylim(-0.5, len(eta_vals) - 0.5)
        ax.set_xticks(range(len(agg_vals)))
        ax.set_yticks(range(len(eta_vals)))
        ax.set_xticklabels(agg_vals, fontsize=10)
        ax.set_yticklabels([_level_label(v) for v in eta_vals], fontsize=10)
        ax.set_xlabel("q_agg_sample", fontsize=11)
        ax.set_ylabel("tfg_eta", fontsize=11)
        ax.set_title(f"q_td_huber_width = {hw_label}", fontsize=11)

    fig.suptitle(
        f"Sweep 80 — {metric_label}: per-config matrix\n"
        f"(geomean across all envs & seeds; 90% bootstrap CI)",
        fontsize=12,
        y=1.04,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep-id", type=int, default=SWEEP_ID)
    ap.add_argument(
        "--min-steps", type=float, default=900_000,
        help="Filter runs with fewer than this many env steps (default: 900000)",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=FIG_DIR / "sweep80",
        help="Output directory for figures",
    )
    args = ap.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Loading sweep {args.sweep_id} runs ===", flush=True)
    runs = load_sweep_runs(args.sweep_id, min_steps=args.min_steps)
    if not runs:
        sys.exit(f"No runs found. Check netscratch path: {NETSCRATCH_WANDB_ROOT}")

    print(f"\n=== Loading LSAC benchmarks ===", flush=True)
    benchmarks = load_all_benchmarks(ENVS)
    for env, s in benchmarks.items():
        print(f"  {env}: benchmark={s:.2f}", flush=True)

    bins = np.arange(BIN_SIZE, MAX_STEPS + BIN_SIZE, BIN_SIZE, dtype=int)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    compute_run_metrics(runs, benchmarks)
    config_scores = compute_config_scores(runs)

    print(f"\nConfig scores (sorted by exp_log_score):", flush=True)
    for cfg, s in sorted(config_scores.items(), key=lambda kv: -kv[1]["exp_log_score"]):
        print(
            f"  {cfg}: els={s['exp_log_score']:.4f}"
            f"  gm={s['geomean_score']:.2f}  n={s['n_runs']}",
            flush=True,
        )

    print(f"\n=== Part 1: Training curve ablation plots ===", flush=True)
    plot_ablation_curves(runs, bins, out_dir)

    print(f"\n=== Part 3: Pairwise matrix plots ===", flush=True)
    plot_pair_matrices(
        runs, "exp_log_score", "Geomean exp(log_score)",
        out_dir / "sweep80_pair_matrix_exp_log_score.png", rng,
    )
    plot_pair_matrices(
        runs, "geomean", "Geomean tail score",
        out_dir / "sweep80_pair_matrix_geomean.png", rng,
    )

    print(f"\n=== Part 4: Best-config vs baselines ===", flush=True)
    plot_best_vs_baselines(
        runs, config_scores, "exp_log_score", "Geomean exp(log_score)", bins, out_dir,
    )
    plot_best_vs_baselines(
        runs, config_scores, "geomean_score", "Geomean tail score", bins, out_dir,
    )

    print(f"\n=== Part 4b: Best-config vs MGMD ===", flush=True)
    plot_best_vs_mgmd(
        runs, config_scores, "exp_log_score", "Geomean exp(log_score)", out_dir,
    )
    plot_best_vs_mgmd(
        runs, config_scores, "geomean_score", "Geomean tail score", out_dir,
    )

    print(f"\n=== Part 5: Per-config matrix plots ===", flush=True)
    plot_config_matrices(
        runs, "exp_log_score", "Geomean exp(log_score)",
        out_dir / "sweep80_config_matrix_exp_log_score.png", rng,
    )
    plot_config_matrices(
        runs, "geomean", "Geomean tail score",
        out_dir / "sweep80_config_matrix_geomean.png", rng,
    )

    print(f"\nAll done. Output dir: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
