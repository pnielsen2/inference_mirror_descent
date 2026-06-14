#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tensorboard.backend.event_processing import event_accumulator

from fit_kalman_runs import Kalman1D, fit_params

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONDITIONS = ["cleanrl_sac_baseline", "policy_gradient_mean_Q"]
DEFAULT_ENVS = ["HalfCheetah-v4", "Walker2d-v4", "Hopper-v4", "Ant-v4", "Humanoid-v4", "Swimmer-v4"]
COLORS = {
    "cleanrl_sac_baseline": "tab:blue",
    "policy_gradient_mean_Q": "tab:orange",
}
LABELS = {
    "cleanrl_sac_baseline": "CleanRL SAC baseline",
    "policy_gradient_mean_Q": "Policy-gradient mean Q",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--metric", default="charts/episodic_return")
    parser.add_argument("--interval", type=float, default=5000.0)
    parser.add_argument("--ci-level", type=float, default=0.90)
    parser.add_argument("--conditions", nargs=2, default=DEFAULT_CONDITIONS)
    parser.add_argument("--env-ids", nargs=6, default=DEFAULT_ENVS)
    parser.add_argument("--smoothed-root", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def fit_y_axis(ax, y_parts: list[np.ndarray]) -> None:
    arrays = []
    for part in y_parts:
        arr = np.asarray(part, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            arrays.append(arr)
    if not arrays:
        return
    values = np.concatenate(arrays)
    ymin = float(values.min())
    ymax = float(values.max())
    if ymin == ymax:
        pad = max(1.0, 0.05 * max(abs(ymin), 1.0))
    else:
        pad = 0.05 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)


def discover_event_files(run_root: Path, metric: str) -> list[dict]:
    print(f"discovering event files under {run_root}", flush=True)
    pattern = "*/*/seed_*/runs/*/events.out.tfevents.*"
    records = []
    for event_file in sorted(run_root.glob(pattern)):
        rel = event_file.relative_to(run_root)
        parts = rel.parts
        if len(parts) < 5:
            continue
        condition = parts[0]
        env_id = parts[1]
        seed = int(parts[2].split("_")[-1])
        records.append({
            "condition": condition,
            "env_id": env_id,
            "seed": seed,
            "event_file": event_file,
            "metric": metric,
        })
    print(f"found {len(records)} event files", flush=True)
    return records


def load_curve(event_file: Path, metric_name: str):
    accumulator = event_accumulator.EventAccumulator(str(event_file))
    accumulator.Reload()
    events = accumulator.Scalars(metric_name)
    if not events:
        return None
    steps = np.array([event.step for event in events], dtype=np.float64)
    values = np.array([event.value for event in events], dtype=np.float64)
    order = np.argsort(steps)
    steps = steps[order]
    values = values[order]
    unique_steps, unique_indices = np.unique(steps, return_index=True)
    return unique_steps, values[unique_indices]


def smooth_curve(steps: np.ndarray, values: np.ndarray, interval: float) -> pd.DataFrame | None:
    if len(steps) < 2:
        return None
    times = np.asarray(steps, dtype=float)
    y = np.asarray(values, dtype=float)
    params = fit_params(times, y)
    model = Kalman1D(params)
    r = float(params.r)
    max_step = float(times[-1])
    max_grid = math.ceil(max_step / interval) * interval
    regular_grid = np.arange(interval, max_grid + 0.5, interval, dtype=float)
    smoothed_grid = np.concatenate([[0.0], regular_grid])
    full_df, _ = model.filter_timeline(y, observation_times=times, output_times=smoothed_grid)
    smoothed_rows = model.smooth_timeline(full_df, output_times=smoothed_grid)
    return pd.DataFrame({
        "_step": smoothed_rows["time"].astype(int),
        "mean": smoothed_rows["smoothed_mean"],
        "mean_var": smoothed_rows["smoothed_var"],
        "obs_var": smoothed_rows["smoothed_var"] + r,
    })


def aggregate_group(run_dfs: list[pd.DataFrame], ci_level: float) -> pd.DataFrame:
    merged = None
    for idx, run_df in enumerate(run_dfs):
        frame = run_df[["_step", "mean", "mean_var"]].rename(
            columns={"mean": f"mean_{idx}", "mean_var": f"mean_var_{idx}"}
        )
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on="_step", how="outer")
    if merged is None:
        return pd.DataFrame(columns=["_step", "mean", "ci_lo", "ci_hi", "n"])
    merged = merged.sort_values("_step").reset_index(drop=True)
    mean_cols = [c for c in merged.columns if c.startswith("mean_") and not c.startswith("mean_var_")]
    var_cols = [c for c in merged.columns if c.startswith("mean_var_")]
    means = merged[mean_cols].to_numpy(dtype=float)
    variances = merged[var_cols].to_numpy(dtype=float)
    counts = np.sum(~np.isnan(means), axis=1)
    point_mean = np.nanmean(means, axis=1)
    point_mean[counts == 0] = np.nan
    total_var = np.full(len(merged), np.nan)
    one_mask = counts == 1
    if np.any(one_mask):
        total_var[one_mask] = np.nansum(variances[one_mask], axis=1)
    many_mask = counts > 1
    if np.any(many_mask):
        between = np.nanvar(means[many_mask], axis=1, ddof=1) / counts[many_mask]
        within = np.nansum(variances[many_mask], axis=1) / (counts[many_mask] ** 2)
        total_var[many_mask] = between + within
    se = np.sqrt(np.clip(total_var, 0.0, None))
    half_width = np.full(len(merged), np.nan)
    z_crit = stats.norm.ppf(1.0 - (1.0 - ci_level) / 2.0)
    if np.any(one_mask):
        half_width[one_mask] = z_crit * se[one_mask]
    for n in sorted({int(v) for v in counts if v > 1}):
        mask = counts == n
        t_crit = stats.t.ppf(1.0 - (1.0 - ci_level) / 2.0, df=n - 1)
        half_width[mask] = t_crit * se[mask]
    return pd.DataFrame({
        "_step": merged["_step"].to_numpy(dtype=float),
        "mean": point_mean,
        "ci_lo": point_mean - half_width,
        "ci_hi": point_mean + half_width,
        "n": counts.astype(int),
    })


def plot_comparison(smoothed_runs: dict, env_ids: list[str], conditions: list[str], output_file: Path, ci_level: float) -> None:
    print(f"plotting comparison figure to {output_file}", flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True)
    axes = axes.flatten()
    for ax, env_id in zip(axes, env_ids):
        print(f"  plotting env {env_id}", flush=True)
        y_parts: list[np.ndarray] = []
        for condition in conditions:
            env_runs = smoothed_runs[condition][env_id]
            if not env_runs:
                print(f"    skipping {condition} for {env_id} (no runs)", flush=True)
                continue
            print(f"    aggregating {condition} for {env_id} with {len(env_runs)} runs", flush=True)
            agg = aggregate_group(list(env_runs.values()), ci_level)
            valid = np.isfinite(agg["mean"].to_numpy(dtype=float))
            if not np.any(valid):
                print(f"    no valid aggregated points for {condition} {env_id}", flush=True)
                continue
            x = agg["_step"].to_numpy(dtype=float) / 1e6
            mean = agg["mean"].to_numpy(dtype=float)
            ci_lo = agg["ci_lo"].to_numpy(dtype=float)
            ci_hi = agg["ci_hi"].to_numpy(dtype=float)
            ax.plot(
                x[valid],
                mean[valid],
                label=f"{LABELS.get(condition, condition)} (n={len(env_runs)})",
                color=COLORS.get(condition),
                linewidth=1.8,
            )
            band = valid & np.isfinite(ci_lo) & np.isfinite(ci_hi)
            if np.any(band):
                ax.fill_between(x[band], ci_lo[band], ci_hi[band], color=COLORS.get(condition), alpha=0.20)
                y_parts.extend([ci_lo[band], ci_hi[band]])
            y_parts.append(mean[valid])
        ax.set_title(env_id)
        ax.set_xlabel("Env steps (M)")
        ax.set_ylabel("Smoothed episodic return")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=9)
        ax.set_xlim(0.0, 1.0)
        if y_parts:
            fit_y_axis(ax, y_parts)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.0))
    ci_pct = int(round(100 * ci_level))
    fig.suptitle(
        f"CleanRL SAC comparison: baseline vs policy-gradient mean Q\nKalman-smoothed training curves with {ci_pct}% CI",
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.12)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure to {output_file}", flush=True)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    run_name = run_root.name
    smoothed_root = args.smoothed_root or (REPO_ROOT / "data" / "plotting_data" / "cleanrl_sac_compare" / run_name / "smoothed_runs")
    output_file = args.output_file or (REPO_ROOT / "figures" / f"{run_name}_smoothed.png")

    print(f"starting cleanrl SAC smoothing for run root {run_root}", flush=True)
    print(f"smoothed run output root: {smoothed_root}", flush=True)
    print(f"plot output file: {output_file}", flush=True)

    records = discover_event_files(run_root, args.metric)
    filtered = [r for r in records if r["condition"] in args.conditions and r["env_id"] in args.env_ids]
    print(f"retained {len(filtered)} event files after condition/env filtering", flush=True)

    smoothed_runs = defaultdict(lambda: defaultdict(dict))
    total = len(filtered)
    for idx, record in enumerate(filtered, start=1):
        condition = record["condition"]
        env_id = record["env_id"]
        seed = record["seed"]
        out_path = smoothed_root / condition / env_id / f"seed_{seed}.csv"
        print(f"[{idx}/{total}] processing {condition} {env_id} seed={seed}", flush=True)
        if out_path.exists() and not args.overwrite:
            print(f"[{idx}/{total}] loading cached smoothed run from {out_path}", flush=True)
            smoothed_df = pd.read_csv(out_path)
            smoothed_runs[condition][env_id][seed] = smoothed_df
            continue
        print(f"[{idx}/{total}] reading tensorboard scalars from {record['event_file']}", flush=True)
        curve = load_curve(record["event_file"], record["metric"])
        if curve is None:
            print(f"[{idx}/{total}] no curve found; skipping", flush=True)
            continue
        steps, values = curve
        print(f"[{idx}/{total}] fitting Kalman smoother on {len(steps)} points", flush=True)
        smoothed_df = smooth_curve(steps, values, args.interval)
        if smoothed_df is None:
            print(f"[{idx}/{total}] insufficient points to smooth; skipping", flush=True)
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{idx}/{total}] storing smoothed run to {out_path}", flush=True)
        smoothed_df.to_csv(out_path, index=False)
        smoothed_runs[condition][env_id][seed] = smoothed_df
        print(f"[{idx}/{total}] stored smoothed run", flush=True)

    print("finished smoothing individual runs", flush=True)
    for condition in args.conditions:
        for env_id in args.env_ids:
            print(
                f"available smoothed runs for {condition} {env_id}: {len(smoothed_runs[condition][env_id])}",
                flush=True,
            )
    plot_comparison(smoothed_runs, args.env_ids, args.conditions, output_file, args.ci_level)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
