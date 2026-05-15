#!/usr/bin/env python
"""Training curves for MGMD sourced from rd_sweep top-3 runs per env.

For each MuJoCo env:
  1. Enumerate all runs across every batch in sweeps/rd_sweep/batches/.
  2. Pick the most recent finished wandb run for each suffix (same policy as
     render_rd_regression.py -- duplicates resolve to the latest by
     created_at).
  3. Score each run with the same log2 normalized score used in
     render_rd_regression.py:
        score = log2(max(0.5 * (max_smoothed_11 + mean_final_50) / norm, 0.01))
  4. Keep the top-3 runs by score for that env.
  5. Re-bin each top-3 run every 5000 env steps: at each bin boundary t,
     take the mean of raw episode returns whose _step falls in (t-5000, t].
  6. Across the 3 binned curves, report mean and 50% t CI at every bin.

The MGMD curve is plotted against the same baselines as
figures/model_free_training_curves_all_envs.png (SAC, TD3, DIPO, PPO, TRPO)
using solid lines and the color scheme from generate_paper_figures.py.
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import yaml
from scipy import stats

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_DIR = SCRIPT_DIR.parent
RD_SWEEP_DIR = REPO_DIR / "sweeps" / "rd_sweep"
BATCHES_DIR = RD_SWEEP_DIR / "batches"

sys.path.insert(0, str(SCRIPT_DIR))
import plot_new_training_curves as pntc  # noqa: E402

WANDB_PROJECT = "pnielsen2-harvard/diffusion_online_rl"

# Same normalizers as render_rd_regression.py. Duplicated with -v4 keys so the
# scoring works regardless of whether a batch uses -v3 or -v4 env names.
_NORMS_BASE = {
    "HalfCheetah": 11000.0,
    "Swimmer": 110.0,
    "Walker2d": 4800.0,
    "Ant": 6000.0,
    "Hopper": 3500.0,
    "Humanoid": 6500.0,
}
NORMALIZERS = {}
for _stem, _n in _NORMS_BASE.items():
    NORMALIZERS[f"{_stem}-v3"] = _n
    NORMALIZERS[f"{_stem}-v4"] = _n

SUFFIX_PAT = re.compile(r"rdsweep_b\d+_r\d+")

# Baselines and styling match model_free_training_curves_all_envs.png
# (scripts/generate_paper_figures.py): SAC, TD3, DIPO, PPO, TRPO -- no OURS,
# no DSAC, solid lines.
BASELINE_ALGOS = ["SAC", "TD3", "DIPO", "PPO", "TRPO"]
METHOD_COLORS = {
    "MGMD": "C0",
    "SAC": "C1",
    "TD3": "C2",
    "DIPO": "C3",
    "PPO": "C4",
    "TRPO": "C5",
}
LSAC_ENV_KEYS = {
    "HalfCheetah-v4": "halfcheetah",
    "Hopper-v4": "hopper",
    "Walker2d-v4": "walker2d",
    "Ant-v4": "ant",
    "Humanoid-v4": "humanoid",
    "Swimmer-v4": "swimmer",
}
ENV_ORDER = ["HalfCheetah-v4", "Ant-v4", "Swimmer-v4",
             "Walker2d-v4", "Hopper-v4", "Humanoid-v4"]
LSAC_N_SEEDS = 10
LSAC_CI_LEVEL = 0.90  # matches generate_paper_figures.py
MGMD_CI_LEVEL = 0.50


def canonical_env(env):
    """Map env name to its -v4 form so MGMD -v3 sweep runs line up with LSAC."""
    for stem in _NORMS_BASE:
        if env.startswith(stem):
            return f"{stem}-v4"
    return env


def path_get(run_path, ab_name):
    for p in run_path:
        if p.get("ablation") == ab_name:
            return p.get("level")
    return None


def collect_manifest_runs(batches_dir):
    """Walk every batch's manifest.yaml. Return {suffix: env_name_raw}."""
    info = {}
    if not batches_dir.exists():
        sys.exit(f"No batches dir at {batches_dir}")
    for batch_dir in sorted(batches_dir.iterdir(),
                            key=lambda p: int(p.name) if p.name.isdigit() else 1e9):
        if not batch_dir.is_dir():
            continue
        mf = batch_dir / "manifest.yaml"
        if not mf.exists():
            continue
        with open(mf) as f:
            manifest = yaml.safe_load(f)
        for r in manifest.get("runs", []):
            env = path_get(r.get("path", []), "Environment")
            if env:
                info[r["suffix"]] = env
    return info


def group_runs_by_suffix(all_runs):
    """{suffix: [finished runs]} using the finished filter + suffix regex."""
    by_suffix = defaultdict(list)
    for run in all_runs:
        if run.state != "finished":
            continue
        m = SUFFIX_PAT.search(run.name)
        if m:
            by_suffix[m.group(0)].append(run)
    return by_suffix


def fetch_history(run, env_name):
    """Return DataFrame with columns _step, return, or None."""
    for key in (f"episode_return/{env_name}", "sample/episode_return"):
        try:
            h = run.history(keys=[key, "_step"], samples=20000)
        except Exception:
            continue
        if h is None or len(h) == 0 or key not in h.columns:
            continue
        h = h.dropna(subset=[key])
        if len(h) > 0:
            return h[["_step", key]].rename(columns={key: "return"}).reset_index(drop=True)
    return None


def compute_score(hist, env_name):
    """log2 normalized score matching render_rd_regression.py."""
    vals = hist["return"].values
    if len(vals) < 11:
        return None
    norm = NORMALIZERS.get(env_name)
    if norm is None:
        return None
    smoothed = pd.Series(vals).rolling(window=11, min_periods=11).mean()
    max_smoothed = float(smoothed.max())
    final = float(np.mean(vals[-50:]))
    normalized = 0.5 * (max_smoothed + final) / norm
    return float(np.log2(max(normalized, 0.01)))


def bin_curve(hist, bins, bin_size):
    """At each t in bins, mean of returns with _step in (t - bin_size, t]."""
    steps = hist["_step"].values
    returns = hist["return"].values
    out = np.full(len(bins), np.nan)
    for i, t in enumerate(bins):
        mask = (steps > t - bin_size) & (steps <= t)
        if mask.any():
            out[i] = returns[mask].mean()
    return out


def aggregate_top_curves(curves, ci_level=MGMD_CI_LEVEL):
    arr = np.stack(curves)  # (n_runs, n_bins)
    n = arr.shape[0]
    mean = np.nanmean(arr, axis=0)
    if n > 1:
        sem = stats.sem(arr, axis=0, nan_policy="omit")
        t_crit = stats.t.ppf(1 - (1 - ci_level) / 2, df=n - 1)
        ci = t_crit * sem
    else:
        ci = np.zeros_like(mean)
    return mean, ci, n


def plot_baselines(ax, df, ci_level=LSAC_CI_LEVEL, n_seeds=LSAC_N_SEEDS):
    """Plot SAC/TD3/DIPO/PPO/TRPO baselines with solid lines."""
    t_crit = stats.t.ppf(1 - (1 - ci_level) / 2, df=n_seeds - 1)
    for algo in BASELINE_ALGOS:
        if algo not in df["algo"].values:
            continue
        algo_df = df[df["algo"] == algo].sort_values("steps")
        steps = algo_df["steps"].values / 1e6
        means = algo_df["rew_mean"].values
        stds = algo_df["rew_std"].values
        cis = t_crit * stds / np.sqrt(n_seeds)
        color = METHOD_COLORS[algo]
        ax.plot(steps, means, label=algo, linewidth=1.2, color=color)
        ax.fill_between(steps, means - cis, means + cis, alpha=0.15, color=color)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--bin-size", type=int, default=5000)
    ap.add_argument("--max-steps", type=int, default=1_000_000)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--batches-dir", type=Path, default=BATCHES_DIR)
    ap.add_argument("--wandb-project", type=str, default=WANDB_PROJECT)
    args = ap.parse_args()

    bins = np.arange(args.bin_size, args.max_steps + args.bin_size, args.bin_size)

    os.environ.setdefault(
        "WANDB_API_KEY",
        "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",
    )
    api = wandb.Api(timeout=120)

    print("Collecting rd_sweep manifest suffixes...")
    run_info = collect_manifest_runs(args.batches_dir)
    print(f"  {len(run_info)} total suffixes across batches")

    print("Fetching rd_sweep wandb runs (one query)...")
    all_runs = list(api.runs(
        args.wandb_project,
        filters={"display_name": {"$regex": "rdsweep"}},
    ))
    print(f"  {len(all_runs)} rdsweep runs found")

    by_suffix = group_runs_by_suffix(all_runs)
    print(f"  {len(by_suffix)} suffixes have at least one finished run")

    # For each suffix: most recent finished run, fetch history, score.
    scored = []  # (env_canon, env_raw, score, hist, suffix)
    for suffix, runs in by_suffix.items():
        if suffix not in run_info:
            continue
        env_raw = run_info[suffix]
        run = max(runs, key=lambda r: r.created_at)
        hist = fetch_history(run, env_raw)
        if hist is None:
            print(f"  {suffix}  {env_raw}  NO_HISTORY")
            continue
        score = compute_score(hist, env_raw)
        if score is None:
            print(f"  {suffix}  {env_raw}  NO_SCORE")
            continue
        scored.append((canonical_env(env_raw), env_raw, score, hist, suffix))

    print(f"Scored {len(scored)} runs")

    # Group by canonical env, pick top-k by score.
    by_env = defaultdict(list)
    for env_c, env_raw, score, hist, suffix in scored:
        by_env[env_c].append((score, hist, suffix, env_raw))

    top_by_env = {}
    for env_c, entries in by_env.items():
        entries.sort(key=lambda x: -x[0])
        top = entries[: args.top_k]
        top_by_env[env_c] = top
        picks = ", ".join(f"{s:.3f}:{suf}" for s, _, suf, _ in top)
        print(f"  {env_c}: top {len(top)} of {len(entries)}  [{picks}]")

    # Bin each top-k run and aggregate.
    mgmd_per_env = {}
    for env_c, top in top_by_env.items():
        curves = [bin_curve(h, bins, args.bin_size) for _, h, _, _ in top]
        mean, ci, n = aggregate_top_curves(curves)
        mgmd_per_env[env_c] = (mean, ci, n)

    # Load baselines (reuse the LSAC loader, but only plot SAC/TD3/DIPO/PPO/TRPO).
    print("\nLoading baselines...")
    lsac = pntc.load_lsac_baselines()
    print(f"  Baseline data available for: {sorted(lsac.keys()) if lsac else 'none'}")

    env_order = [e for e in ENV_ORDER if e in mgmd_per_env]
    env_order += [e for e in sorted(mgmd_per_env) if e not in env_order]
    n_envs = len(env_order)
    n_cols = 3
    n_rows = (n_envs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    mgmd_color = METHOD_COLORS["MGMD"]
    for idx, env in enumerate(env_order):
        ax = axes[idx]
        mean, ci, n = mgmd_per_env[env]
        ax.plot(bins / 1e6, mean,
                label="MGMD", color=mgmd_color, linewidth=1.5)
        ax.fill_between(bins / 1e6, mean - ci, mean + ci,
                        alpha=0.2, color=mgmd_color)
        if env in lsac:
            plot_baselines(ax, lsac[env])
        ax.set_title(env.replace("-v4", ""), fontsize=11)
        ax.set_xlabel("Steps (M)", fontsize=9)
        ax.set_ylabel("Episode Return", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=8)
        ax.set_xlim(0, args.max_steps / 1e6)

    for idx in range(n_envs, len(axes)):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(handles), 5), fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    figures_dir = REPO_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)
    out_path = args.out or (figures_dir / "training_curves_rd_sweep_top3.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
