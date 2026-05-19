#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import wandb

import plot_sweep_4env_training_curves as base
import plot_sweep_6env_training_curves as six_env

ENVS = list(six_env.ENVS)
SWEEP45_LABEL = "MGMD"
DPMD_LABEL = "DPMD"
SWEEP45_COLOR = base.METHOD_COLORS["MGMD"]
DPMD_COLOR = "C6"
DEFAULT_DPMD_ROOT = Path("/n/netscratch/kdbrantley_lab/Lab/pnielsen/dpmd_full_sweep/20260512_004424_individual/logs")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-id", "--sweep_id", type=int, default=45)
    ap.add_argument("--config-tag", "--config_tag", type=str, default="sweep45_kl_budget=1024")
    ap.add_argument("--dpmd-root", type=Path, default=DEFAULT_DPMD_ROOT)
    ap.add_argument("--num-points", type=int, default=200)
    ap.add_argument("--ci-level", type=float, default=0.90)
    ap.add_argument("--max-steps", type=int, default=1_000_000)
    ap.add_argument("--out", type=Path, default=None)
    return ap.parse_args()


ROLLING_WINDOW = 11


def smooth_hist(hist: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Apply a backward-looking rolling mean of `window` episodes to returns."""
    hist = hist.copy()
    hist["return"] = hist["return"].rolling(window, min_periods=1).mean()
    return hist


def rescale_to_target_max(hists: list[pd.DataFrame], target_max: float) -> list[pd.DataFrame]:
    """Rescale each seed's _step so its maximum maps to target_max."""
    result = []
    for hist in hists:
        if hist is None or len(hist) == 0:
            result.append(hist)
            continue
        hist = hist.copy()
        s_max = float(hist["_step"].max())
        if s_max > 0 and s_max != target_max:
            hist["_step"] = hist["_step"] * (target_max / s_max)
        result.append(hist)
    return result


def canonicalize_hist(hist: pd.DataFrame | None) -> pd.DataFrame | None:
    if hist is None or len(hist) == 0:
        return None
    required = {"_step", "return"}
    if not required.issubset(hist.columns):
        return None
    out = hist[["_step", "return"]].copy()
    out = out.dropna(subset=["_step", "return"])
    if len(out) == 0:
        return None
    out["_step"] = out["_step"].astype(float)
    out["return"] = out["return"].astype(float)
    out = out.sort_values("_step")
    out = out.groupby("_step", as_index=False, sort=True)["return"].mean()
    if len(out) < 2:
        return None
    return out.reset_index(drop=True)


# Hardcoded wandb run IDs for sweep45 kl_budget=1024 (8 seeds per env).
SWEEP45_RUN_IDS: dict[str, list[str]] = {
    "Ant-v3":         ["fepyisin", "5vcxwghc", "r6od1b70", "7lce7wj7", "yxkffn2c", "r9xyr8bz", "yfvjmok6", "d5ehks9f"],
    "HalfCheetah-v3": ["o7etiyl2", "xxnn144m", "vb9ljmta", "hucmlk2i", "h74hqdbv", "g75dmb1h", "9rnei5ej", "9jff3s2r"],
    "Hopper-v3":      ["3r7qyuq6", "1o9ypcv8", "bll9d49g", "39s804k2", "u7x0c695", "8tn5316z", "hs39yp3h", "1yn1xaor"],
    "Humanoid-v3":    ["9fk6skn6", "qlq8lwqm", "mecu931a", "t3m41biu", "uyae7y0v", "5xz6p4i5", "6ai20bf1", "msadbg7g"],
    "Swimmer-v3":     ["cuk37n3e", "3gp14sy4", "tcnnwg4x", "zhflkz2g", "lyhs9luy", "yfsl5es5", "taxqev3e", "poc6hjzv"],
    "Walker2d-v3":    ["9dpeefa6", "xmge3230", "6q6xxtty", "k0msvm6m", "9bfrdrmr", "iprvrm3u", "xacaqcky", "6pjfcysz"],
}


def load_sweep45_histories():
    api = wandb.Api()
    histories: dict[str, list[pd.DataFrame]] = defaultdict(list)
    for env, run_ids in SWEEP45_RUN_IDS.items():
        ep_key = f"episode_return/{env}"
        for run_id in run_ids:
            try:
                run = api.run(f"{base.WANDB_PROJECT}/{run_id}")
                rows = run.history(keys=[ep_key], x_axis="_step", pandas=True)
            except Exception as e:
                print(f"  warning: could not fetch {run_id}: {e}", flush=True)
                continue
            if rows is None or len(rows) == 0 or ep_key not in rows.columns:
                continue
            hist = rows[["_step", ep_key]].rename(columns={ep_key: "return"})
            canon = canonicalize_hist(hist)
            if canon is not None:
                histories[env].append(smooth_hist(canon))
    return dict(histories)


def load_dpmd_histories(root: Path):
    histories = defaultdict(list)
    for csv_path in sorted(root.rglob("episode_returns.csv")):
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "step" not in df.columns:
            continue
        value_cols = [col for col in df.columns if col.startswith("episode_return/")]
        if len(value_cols) != 1:
            continue
        value_col = value_cols[0]
        env = value_col.split("/", 1)[1]
        hist = df[["step", value_col]].rename(columns={"step": "_step", value_col: "return"})
        canon = canonicalize_hist(hist)
        if canon is not None:
            histories[env].append(smooth_hist(canon))
    return dict(histories)


MIN_COVERAGE_FRAC = 0.5  # drop seeds that reached < 50% of the longest seed's max step


def interp_runs(runs: list[pd.DataFrame], num_points: int):
    usable = [hist for hist in runs if hist is not None and len(hist) >= 2]
    if not usable:
        return np.array([]), np.empty((0, 0), dtype=float)
    global_max = max(float(hist["_step"].max()) for hist in usable)
    usable = [hist for hist in usable if float(hist["_step"].max()) >= MIN_COVERAGE_FRAC * global_max]
    if not usable:
        return np.array([]), np.empty((0, 0), dtype=float)
    min_step = max(float(hist["_step"].min()) for hist in usable)
    max_step = min(float(hist["_step"].max()) for hist in usable)
    if not np.isfinite(min_step) or not np.isfinite(max_step) or min_step >= max_step:
        return np.array([]), np.empty((0, 0), dtype=float)
    grid = np.linspace(min_step, max_step, num_points, dtype=float)
    values = []
    for hist in usable:
        x = hist["_step"].to_numpy(dtype=float)
        y = hist["return"].to_numpy(dtype=float)
        values.append(np.interp(grid, x, y))
    return grid, np.vstack(values)


def aggregate_interpolated(values: np.ndarray, ci_level: float):
    if values.size == 0:
        return np.array([]), np.array([])
    mean = values.mean(axis=0)
    if values.shape[0] <= 1:
        return mean, np.zeros_like(mean)
    std = values.std(axis=0, ddof=1)
    sem = std / np.sqrt(float(values.shape[0]))
    t_crit = stats.t.ppf(1.0 - (1.0 - ci_level) / 2.0, df=values.shape[0] - 1)
    ci = t_crit * sem
    return mean, ci


def plot_interpolated_series(ax, env: str, runs: list[pd.DataFrame], label: str, color: str, num_points: int, ci_level: float, y_parts: list[np.ndarray]):
    grid, values = interp_runs(runs, num_points)
    if values.size == 0:
        print(f"{label:<15s} {env:<16s}: no usable histories", flush=True)
        return None
    mean, ci = aggregate_interpolated(values, ci_level)
    x = grid / 1e6
    ax.plot(x, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.2)
    y_parts.extend([mean, mean - ci, mean + ci])
    print(
        f"{label:<15s} {env:<16s}: n={values.shape[0]} step_range=[{int(grid[0])}, {int(grid[-1])}]",
        flush=True,
    )
    return grid, mean, ci, values.shape[0]


def build_output_path(args: argparse.Namespace) -> Path:
    if args.out is not None:
        return args.out
    return base.FIG_DIR / "eta45_vs_baselines.png"


def verify_env_coverage(name: str, histories: dict[str, list[pd.DataFrame]]):
    missing = [env for env in ENVS if env not in histories or not histories[env]]
    if missing:
        raise SystemExit(f"{name} is missing histories for envs: {missing}")


def main() -> None:
    args = parse_args()
    six_env.configure_plot_style()

    sweep45_histories = load_sweep45_histories()
    dpmd_histories = load_dpmd_histories(args.dpmd_root)

    if "Humanoid-v3" in sweep45_histories:
        sweep45_histories["Humanoid-v3"] = rescale_to_target_max(
            sweep45_histories["Humanoid-v3"], float(args.max_steps)
        )

    verify_env_coverage(args.config_tag, sweep45_histories)
    verify_env_coverage(DPMD_LABEL, dpmd_histories)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    axes = axes.flatten()

    for idx, env in enumerate(ENVS):
        ax = axes[idx]
        y_parts: list[np.ndarray] = []
        plot_interpolated_series(
            ax,
            env,
            sweep45_histories[env],
            SWEEP45_LABEL,
            SWEEP45_COLOR,
            args.num_points,
            args.ci_level,
            y_parts,
        )
        baseline_df = base.load_lsac_baseline(env)
        if baseline_df is not None:
            base.plot_baselines(ax, baseline_df)
            y_parts.extend(six_env.baseline_y_bounds(baseline_df))
        plot_interpolated_series(
            ax,
            env,
            dpmd_histories[env],
            DPMD_LABEL,
            DPMD_COLOR,
            args.num_points,
            args.ci_level,
            y_parts,
        )
        ax.set_xlim(0, args.max_steps / 1e6)
        ax.set_title(env.replace("-v3", ""), fontsize=12)
        ax.set_xlabel("Steps (M)", fontsize=10)
        ax.set_ylabel("Episode Return", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=9)
        six_env.fit_y_axis(ax, y_parts)

    handles = []
    labels = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=max(1, len(handles)), fontsize=10, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)

    base.FIG_DIR.mkdir(exist_ok=True)
    out = build_output_path(args)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}", flush=True)
    plt.close(fig)


if __name__ == "__main__":
    main()
