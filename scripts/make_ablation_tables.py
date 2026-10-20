#!/usr/bin/env python
"""Recompute the Section 6 ablation numbers for the ICLR submission.

Three ablations, all scored with the estimator of the paper's main table
(``trailing average over the last W evaluation checkpoints of the across-seed
median``, 80% bootstrap CI over seeds -- this reproduces
``figures/algorithm_comparison_trailing_of_median_ci80.csv`` exactly):

* step size -- fixed ``h_t = c (1 - alpha_t)`` (zzhong sweeps 245/248) against
  the Robbins-Monro rule (sweep 244, ``rollout_alpha = 1``);
* TD backup action -- unguided DDIM next-action (sweep 246) against tilted;
* annealing length -- ``T = 50`` (sweep 247) against ``T = 100``.

Our own sweeps are read from the ``episode_returns.csv`` / ``eval_episode_returns.csv``
mirrors under each sweep's codebase snapshot (see AGENTS.md). The collaborator's
sweeps live under another user's holylabs wandb dir with the CSV mirrors in an
unreadable home, so their eval curve is recovered from the offline ``.wandb``
datastore instead. Two traps there: ``open_for_scan`` opens ``r+b`` (hence the
copy to a scratch dir), and history keys containing ``/`` arrive as
``nested_key``, not ``key``.

Usage:
    python scripts/make_ablation_tables.py                  # all three tables
    python scripts/make_ablation_tables.py --at 1000000     # pin the step
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import json
import os
import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ENVS = ["Ant-v3", "HalfCheetah-v3", "Hopper-v3", "Humanoid-v3", "Swimmer-v3", "Walker2d-v3"]
PLOT_ENVS = ["Ant-v3", "HalfCheetah-v3", "Humanoid-v3", "Walker2d-v3"]

# Codebase snapshots of our own sweeps, and the per-env base --seed that
# identifies the rollout_alpha=1 arm of sweep 244 (the main-results arm).
CODEBASE = "/n/home09/pnielsen/inference_mirror_descent/logs/slurm"
SWEEP244 = "20260919_115738"
SWEEP246 = "20260923_193738"   # --td_next_action ddim
SWEEP247 = "20260923_204649"   # --diffusion_steps 50
MAIN_ARM_SEED = {"Walker2d-v3": "60", "Hopper-v3": "70", "Humanoid-v3": "80",
                 "Ant-v3": "90", "HalfCheetah-v3": "100", "Swimmer-v3": "110"}
ZZHONG = "/n/holylabs/kdbrantley_lab/Lab/zzhong/wandb"


def load_mirror(snapshot: str, arm_seed: dict | None = None):
    """{env: {seed_index: {env_step: mean eval return}}} from the CSV mirrors."""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    pattern = f"{CODEBASE}/{snapshot}/codebase/logs/*/*/eval_episode_returns.csv"
    for path in glob.glob(pattern):
        env = path.split("/logs/")[-1].split("/")[0]
        run = os.path.basename(os.path.dirname(path))
        if arm_seed and f"_s{arm_seed[env]}_" not in run:
            continue
        episodes = collections.defaultdict(list)
        with open(path) as fh:
            for row in csv.DictReader(fh):
                episodes[(int(row["seed_index"]), int(row["step"]))].append(float(row["episode_return"]))
        for (seed, step), returns in episodes.items():
            out[env][seed][step] = float(np.mean(returns))
    return out


def load_datastore(sweeps=(245, 248)):
    """{(env, multiplier): {seed: {env_step: eval return}}} from offline .wandb files."""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    scratch = tempfile.mkdtemp(prefix="mgmd_ds_")
    try:
        for sweep in sweeps:
            for run_dir in sorted(glob.glob(f"{ZZHONG}/sweep_{sweep}/job_*/wandb/offline-run-*")):
                blobs = glob.glob(os.path.join(run_dir, "run-*.wandb"))
                if not blobs:
                    continue
                cfg = yaml.safe_load(open(os.path.join(run_dir, "files", "config.yaml")))
                val = lambda k: (cfg.get(k) or {}).get("value")
                key = (val("env"), float(val("mala_step_size_beta_multiplier")))
                seed = int(val("seed"))
                local = os.path.join(scratch, "run.wandb")
                shutil.copyfile(blobs[0], local)          # open_for_scan needs r+b
                store = datastore.DataStore()
                store.open_for_scan(local)
                while True:
                    blob = store.scan_data()
                    if blob is None:
                        break
                    rec = pb.Record()
                    rec.ParseFromString(blob)
                    if rec.WhichOneof("record_type") != "history":
                        continue
                    item = {(i.key or "/".join(i.nested_key)): i.value_json for i in rec.history.item}
                    if "eval/episode_return_mean" in item:
                        out[key][seed][int(json.loads(item["_step"]))] = \
                            float(json.loads(item["eval/episode_return_mean"]))
                os.remove(local)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return out


def score(curves, at, window=10, every=10000, boot=4000, rng=None):
    """Trailing average of the across-seed median, with an 80% bootstrap CI."""
    rng = rng or np.random.default_rng(0)
    steps = [at - i * every for i in range(window)][::-1]
    seeds = [s for s, c in curves.items() if all(x in c for x in steps)]
    if not seeds:
        return None
    mat = np.array([[curves[s][x] for x in steps] for s in seeds])
    center = float(np.mean(np.median(mat, axis=0)))
    idx = rng.integers(0, len(seeds), (boot, len(seeds)))
    draws = np.array([np.mean(np.median(mat[i], axis=0)) for i in idx])
    lo, hi = np.percentile(draws, [10, 90])
    return center, float(lo), float(hi), len(seeds)


def last_common_step(curves, every=10000):
    return (min(max(c) for c in curves.values()) // every) * every if curves else 0


def cell(r):
    return "--" if r is None else f"{r[0]:.0f} [{r[1]:.0f}, {r[2]:.0f}] (n={r[3]})"


def curve_stats(curves, window, level, n_boot, rng):
    seeds = sorted(curves)
    if len(seeds) != 10:
        raise ValueError(f"expected 10 seeds, found {len(seeds)}")
    steps = sorted(set.intersection(*(set(curves[s]) for s in seeds)))
    if len(steps) < window:
        raise ValueError(f"only {len(steps)} common checkpoints for a window of {window}")
    raw = np.array([[curves[s][step] for step in steps] for s in seeds])
    center = np.lib.stride_tricks.sliding_window_view(
        np.median(raw, axis=0), window).mean(axis=-1)
    draws = np.empty((n_boot, center.size))
    for start in range(0, n_boot, 512):
        count = min(512, n_boot - start)
        idx = rng.integers(0, len(seeds), size=(count, len(seeds)))
        medians = np.median(raw[idx], axis=1)
        draws[start:start + count] = np.lib.stride_tricks.sliding_window_view(
            medians, window, axis=-1).mean(axis=-1)
    lo, hi = np.percentile(draws, [50 * (1 - level), 50 * (1 + level)], axis=0)
    return pd.DataFrame({
        "step": steps[window - 1:],
        "statistic": "median",
        "center": center,
        "ci_lo": lo,
        "ci_hi": hi,
        "n_seeds": len(seeds),
        "ci_coverage": level,
    })


def plot_ablation(series, output, window, level, n_boot, rng, envs=PLOT_ENVS,
                  ncols=None):
    matplotlib.rcParams["pdf.fonttype"] = 42
    rows = []
    ncols = ncols or (3 if len(envs) == 6 else len(envs))
    nrows = (len(envs) + ncols - 1) // ncols
    figsize = (16, 8.5) if len(envs) == 6 else (21.3 / nrows, 4.25 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for ax, env in zip(axes.ravel(), envs):
        for label, arm, color, sweep, config in series:
            if env not in arm or not arm[env]:
                continue
            stats = curve_stats(arm[env], window, level, n_boot, rng)
            x = stats["step"].to_numpy() / 1e6
            ax.plot(x, stats["center"].to_numpy(), color=color, label=label,
                    linewidth=2.2 if color == "black" else 1.6)
            ax.fill_between(x, stats["ci_lo"].to_numpy(), stats["ci_hi"].to_numpy(),
                            color=color, alpha=0.15, linewidth=0)
            stats.insert(0, "aggregation", "trailing_of_median")
            stats.insert(0, "config", config)
            stats.insert(0, "sweep", sweep)
            stats.insert(0, "env", env)
            stats.insert(0, "series", label)
            rows.append(stats)
        ax.set_title(env)
        ax.set_xlabel("Environment steps (millions)")
        ax.set_ylabel("Evaluation return")
        ax.grid(alpha=0.3)
    for ax in axes.ravel()[len(envs):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(series),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.045 if nrows > 1 else 0.11, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    pd.concat(rows, ignore_index=True).to_csv(output.with_suffix(".csv"), index=False)
    print(f"Wrote {output}")
    print(f"Wrote {output.with_suffix('.csv')}")


def make_plots(main_arm, ddim_arm, fixed, figure_dir, window, level, n_boot, boot_seed):
    mults = sorted({m for env, m in fixed if env in PLOT_ENVS})
    fixed_series = [("MGMD (ours)", main_arm, "black", "244", "adaptive")]
    for i, mult in enumerate(mults):
        arm = {env: fixed.get((env, mult), {}) for env in ENVS}
        fixed_series.append((f"Fixed c={mult:g}", arm, f"C{i}",
                             "245" if mult <= 2 else "248", f"c={mult:g}"))
    plot_ablation(
        fixed_series,
        figure_dir / "fixed_step_size_sweeps245_248_vs_mgmd_trailing_of_median_ci80.pdf",
        window, level, n_boot, np.random.default_rng(boot_seed), ncols=2)
    plot_ablation(
        fixed_series,
        figure_dir / "fixed_step_size_sweeps245_248_vs_mgmd_trailing_of_median_ci80_6env.pdf",
        window, level, n_boot, np.random.default_rng(boot_seed), ENVS)
    # Same figure without the largest step size, whose collapse sets the y range
    # and flattens the remaining curves. Colours are kept per multiplier, so the
    # retained curves match the full version.
    plot_ablation(
        [s for s in fixed_series if s[4] != "c=8"],
        figure_dir / "fixed_step_size_sweeps245_248_vs_mgmd_trailing_of_median_ci80_no_c8.pdf",
        window, level, n_boot, np.random.default_rng(boot_seed), ncols=2)
    plot_ablation(
        [("MGMD (ours)", main_arm, "black", "244", "tilted_td_action"),
         ("DDIM TD backup", ddim_arm, "C0", "246", "ddim_td_action")],
        figure_dir / "td_next_action_sweep246_vs_mgmd_trailing_of_median_ci80.pdf",
        window, level, n_boot, np.random.default_rng(boot_seed), ncols=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", type=int, default=1000000, help="env step for the completed comparisons")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--figure-dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "figures")
    ap.add_argument("--n-boot", type=int, default=20_000)
    ap.add_argument("--boot-seed", type=int, default=0)
    args = ap.parse_args()

    main_arm = load_mirror(SWEEP244, MAIN_ARM_SEED)
    ddim_arm = load_mirror(SWEEP246)
    t50_arm = load_mirror(SWEEP247)
    fixed = load_datastore()
    sc = lambda c, at: score(c, at, window=args.window)

    print(f"\n=== (i) MALA step size: fixed h_t = c (1-alpha_t) vs adaptive, @ {args.at} ===")
    mults = sorted({m for _, m in fixed})
    print(f"{'env':16s} {'adaptive':>26s} " + " ".join(f"{'c='+str(m):>26s}" for m in mults))
    for env in ENVS:
        row = [cell(sc(main_arm[env], args.at))] + [cell(sc(fixed[(env, m)], args.at)) for m in mults]
        print(f"{env:16s} " + " ".join(f"{x:>26s}" for x in row))
    print("\n  (configs still running are '--' here; compare them at a matched step:)")
    for env in ENVS:
        for m in mults:
            if fixed[(env, m)] and sc(fixed[(env, m)], args.at) is None:
                s = last_common_step(fixed[(env, m)])
                print(f"    {env:16s} c={m:<4} @{s/1e6:.2f}M  fixed {cell(sc(fixed[(env, m)], s))}"
                      f"   adaptive {cell(sc(main_arm[env], s))}")

    for name, arm, base in [("(ii) TD backup from the unguided base policy", ddim_arm, "tilted"),
                            ("(iii) annealing path T = 50", t50_arm, "T = 100")]:
        print(f"\n=== {name} (matched step; ablation arms may still be running) ===")
        for env in ENVS:
            s = last_common_step(arm[env])
            if s < args.window * 10000:
                print(f"{env:16s} no data")
                continue
            print(f"{env:16s} @{s/1e6:.2f}M  ablation {cell(sc(arm[env], s)):>28s}   {base} {cell(sc(main_arm[env], s))}")

    if args.plots:
        make_plots(main_arm, ddim_arm, fixed, args.figure_dir, args.window,
                   0.8, args.n_boot, args.boot_seed)


if __name__ == "__main__":
    main()
