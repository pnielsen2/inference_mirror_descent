#!/usr/bin/env python3
"""Plot DDPM_mean vs DDIM training curves for sweep 86, one subplot per env.

Pulls every run with ``config.sweep_id == 86`` from wandb, splits them by
``config.denoising_predictor`` (DDPM_mean vs DDIM), interpolates each run's
``episode_return/{env}`` stream onto a common env-step grid, and plots the
across-run mean with a 90% confidence band. Output is a single PNG with a
2x3 grid of subplots (one per environment).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb
from scipy import stats

ENTITY = "pnielsen2-harvard"
PROJECT = "diffusion_online_rl"
ENVS = ["Ant-v3", "HalfCheetah-v3", "Hopper-v3", "Humanoid-v3", "Swimmer-v3", "Walker2d-v3"]

METHODS = ["DDPM_mean", "DDIM"]
METHOD_COLORS = {"DDPM_mean": "#1f77b4", "DDIM": "#d62728"}
METHOD_LABELS = {"DDPM_mean": "DDPM mean", "DDIM": "DDIM"}

CI_LEVEL = 0.90
N_GRID = 300
BIN_SIZE = 5000  # env steps per bin for per-curve preprocessing


def fetch_full_runs(api, sweep_id):
    """Return fully-hydrated Run objects for config.sweep_id == sweep_id.

    ``api.runs(filters=...)`` returns paginated stubs whose ``.config`` is
    empty, so we re-fetch each run by id to read denoising_predictor/env.
    """
    stubs = list(api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"config.sweep_id": int(sweep_id)},
        per_page=500,
    ))
    print(f"  {len(stubs)} runs in sweep {sweep_id}; fetching full configs (8-way parallel)...")

    def _full(rid):
        return api.run(f"{ENTITY}/{PROJECT}/{rid}")

    with ThreadPoolExecutor(max_workers=8) as ex:
        runs = list(ex.map(_full, [r.id for r in stubs]))
    return runs


def _bin_curve(steps, values, bin_size=BIN_SIZE):
    """Bin episodes into ``bin_size``-step windows.

    Each window is identified by its right edge ``k * bin_size`` and holds the
    mean of every episode return whose end-step falls in ``((k-1)*bin_size,
    k*bin_size]``. Returns (edges, means) with one point per non-empty window.
    """
    edges = np.maximum(np.ceil(steps / bin_size), 1).astype(np.int64) * int(bin_size)
    uniq = np.unique(edges)
    means = np.array([values[edges == e].mean() for e in uniq], dtype=float)
    return uniq.astype(float), means


def fetch_curve(run):
    """Return (env, steps, values) for this run's episode-return stream, or None.

    The raw per-episode stream is preprocessed into one point per
    ``BIN_SIZE``-step window (mean of episode returns ending in that window).
    """
    env = run.group if (run.group in ENVS) else run.config.get("env")
    if env not in ENVS:
        return None
    ep_key = f"episode_return/{env}"
    hist = run.history(keys=[ep_key, "_step"], samples=10000, pandas=True)
    if hist is None or hist.empty or ep_key not in hist.columns:
        return None
    eps = hist[["_step", ep_key]].dropna(subset=[ep_key]).sort_values("_step")
    if eps.empty:
        return None
    steps = eps["_step"].to_numpy(dtype=float)
    values = eps[ep_key].to_numpy(dtype=float)
    binned_steps, binned_values = _bin_curve(steps, values)
    return env, binned_steps, binned_values


def collect_curves(runs):
    """Return {(env, method): [(steps, values), ...]} and the global max step."""
    by_cell = defaultdict(list)

    def _work(run):
        method = run.config.get("denoising_predictor")
        if method not in METHODS:
            return None
        res = fetch_curve(run)
        if res is None:
            return None
        env, steps, values = res
        return env, method, steps, values

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_work, runs))

    global_max = 0.0
    for r in results:
        if r is None:
            continue
        env, method, steps, values = r
        by_cell[(env, method)].append((steps, values))
        global_max = max(global_max, float(steps.max()))
    return by_cell, global_max


def interpolate(curves, grid):
    """Interpolate each (steps, values) curve onto grid; NaN outside its range."""
    rows = []
    for steps, values in curves:
        rows.append(np.interp(grid, steps, values, left=np.nan, right=np.nan))
    return np.array(rows)


def mean_ci(arr, ci_level=CI_LEVEL):
    n_per_step = np.sum(~np.isnan(arr), axis=0)
    mean = np.nanmean(arr, axis=0)
    sem = np.asarray(stats.sem(arr, axis=0, nan_policy="omit"), dtype=float)
    ci = np.zeros_like(mean)
    valid = n_per_step > 1
    if np.any(valid):
        dof = np.clip(n_per_step - 1, 1, None)
        t_crit = stats.t.ppf(1 - (1 - ci_level) / 2, df=dof)
        ci = np.where(valid, t_crit * sem, 0.0)
    return mean, ci


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep-id", "--sweep_id", type=int, default=86)
    ap.add_argument("--out", type=str,
                    default="/n/home09/pnielsen/inference_mirror_descent/scripts/"
                            "topsis_out/sweep_86/ddpm_vs_ddim_training_curves.png")
    args = ap.parse_args()

    api = wandb.Api(timeout=60)
    print(f"Querying wandb for config.sweep_id == {args.sweep_id} ...")
    runs = fetch_full_runs(api, args.sweep_id)

    print("Fetching episode-return histories ...")
    by_cell, global_max = collect_curves(runs)
    if global_max <= 0:
        print("No usable run histories found; aborting.")
        return
    grid = np.linspace(0.0, global_max, N_GRID)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False)
    axes = axes.flatten()

    handles = {}
    for ax, env in zip(axes, ENVS):
        for method in METHODS:
            curves = by_cell.get((env, method), [])
            if not curves:
                continue
            arr = interpolate(curves, grid)
            mean, ci = mean_ci(arr)
            color = METHOD_COLORS[method]
            line, = ax.plot(grid / 1e6, mean, color=color, linewidth=1.8,
                            label=METHOD_LABELS[method])
            ax.fill_between(grid / 1e6, mean - ci, mean + ci, color=color, alpha=0.2)
            handles.setdefault(method, line)
        ax.set_title(env.replace("-v3", ""), fontsize=12)
        ax.set_xlabel("Env steps (M)", fontsize=10)
        ax.set_ylabel("Episode return", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

    ordered = [(handles[m], METHOD_LABELS[m]) for m in METHODS if m in handles]
    if ordered:
        legend_handles, legend_labels = zip(*ordered)
        fig.legend(legend_handles, legend_labels, loc="lower center",
                   ncol=len(legend_handles), fontsize=11, frameon=True)

    fig.suptitle(f"DDPM mean vs DDIM "
                 f"(mean +/- {int(CI_LEVEL * 100)}% CI across runs)", fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(args.out, dpi=150)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
