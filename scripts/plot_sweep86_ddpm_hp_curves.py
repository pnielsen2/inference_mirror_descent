#!/usr/bin/env python3
"""DDPM_mean training curves split by an hp axis, one subplot per env.

Filters sweep-86 runs to ``denoising_predictor == 'DDPM_mean'`` and produces
one PNG per split variable (``guidance_strength_multiplier`` and
``guidance_gradient_space``). In each PNG, every level of the split variable
gets its own across-run mean curve with a 90% confidence band, in a 2x3 grid
(one subplot per environment) with a single shared legend.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_sweep86_ddpm_vs_ddim import (
    CI_LEVEL,
    ENVS,
    N_GRID,
    fetch_curve,
    fetch_full_runs,
    interpolate,
    mean_ci,
)
import wandb

SPLIT_VARS = ["guidance_strength_multiplier", "guidance_gradient_space"]
DDPM = "DDPM_mean"


def _level_sort_key(level):
    try:
        return (0, float(level))
    except (TypeError, ValueError):
        return (1, str(level))


def collect_curves(runs, split_var):
    """Return {(env, level): [(steps, values), ...]} and global max step.

    Only runs with denoising_predictor == DDPM_mean are kept; ``level`` is the
    value of ``split_var`` read from each run's config.
    """
    by_cell = defaultdict(list)

    def _work(run):
        if run.config.get("denoising_predictor") != DDPM:
            return None
        if split_var not in run.config:
            return None
        level = run.config.get(split_var)
        res = fetch_curve(run)
        if res is None:
            return None
        env, steps, values = res
        return env, level, steps, values

    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_work, runs))

    global_max = 0.0
    for r in results:
        if r is None:
            continue
        env, level, steps, values = r
        by_cell[(env, level)].append((steps, values))
        global_max = max(global_max, float(steps.max()))
    return by_cell, global_max


def make_figure(by_cell, global_max, split_var, out_path):
    levels = sorted({lvl for (_env, lvl) in by_cell.keys()}, key=_level_sort_key)
    cmap = plt.get_cmap("tab10")
    colors = {lvl: cmap(i % 10) for i, lvl in enumerate(levels)}
    grid = np.linspace(0.0, global_max, N_GRID)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=False)
    axes = axes.flatten()

    handles = {}
    for ax, env in zip(axes, ENVS):
        for lvl in levels:
            curves = by_cell.get((env, lvl), [])
            if not curves:
                continue
            arr = interpolate(curves, grid)
            mean, ci = mean_ci(arr)
            color = colors[lvl]
            line, = ax.plot(grid / 1e6, mean, color=color, linewidth=1.8, label=str(lvl))
            ax.fill_between(grid / 1e6, mean - ci, mean + ci, color=color, alpha=0.2)
            handles.setdefault(lvl, line)
        ax.set_title(env.replace("-v3", ""), fontsize=12)
        ax.set_xlabel("Env steps (M)", fontsize=10)
        ax.set_ylabel("Episode return", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")

    ordered = [(handles[lvl], str(lvl)) for lvl in levels if lvl in handles]
    if ordered:
        legend_handles, legend_labels = zip(*ordered)
        fig.legend(legend_handles, legend_labels, loc="lower center",
                   ncol=len(legend_handles), fontsize=11, frameon=True,
                   title=split_var)

    fig.suptitle(f"DDPM mean by {split_var} "
                 f"(mean +/- {int(CI_LEVEL * 100)}% CI across runs)", fontsize=14)
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep-id", "--sweep_id", type=int, default=86)
    ap.add_argument("--out-dir", type=str,
                    default="/n/home09/pnielsen/inference_mirror_descent/scripts/"
                            "topsis_out/sweep_86")
    args = ap.parse_args()

    api = wandb.Api(timeout=60)
    print(f"Querying wandb for config.sweep_id == {args.sweep_id} ...")
    runs = fetch_full_runs(api, args.sweep_id)

    for split_var in SPLIT_VARS:
        print(f"\nCollecting DDPM_mean curves split by {split_var} ...")
        by_cell, global_max = collect_curves(runs, split_var)
        if global_max <= 0:
            print(f"  No usable DDPM_mean runs for {split_var}; skipping.")
            continue
        out_path = f"{args.out_dir}/ddpm_{split_var}_training_curves.png"
        make_figure(by_cell, global_max, split_var, out_path)


if __name__ == "__main__":
    main()
