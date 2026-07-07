#!/usr/bin/env python
"""
Plot training curves for MGMD variants: baseline, KL-Budget, and dist_shift_eta.
Generates:
  1. HalfCheetah-specific plot: MGMD baseline vs MGMD (KL-Budget)
  2. All-environments plot: KL-Budget + dist_shift_eta (+ LSAC baselines if available)
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

import wandb

# ── wandb setup ──────────────────────────────────────────────────────────────
from relax.utils.fs import wandb_entity_project
WANDB_PROJECT = wandb_entity_project()

# ── Run IDs ──────────────────────────────────────────────────────────────────

# Original MGMD baseline (HalfCheetah, seeds 0-4, Jan 15, tfg_lambda=16)
MGMD_BASELINE_IDS = {
    "HalfCheetah-v4": {
        0: "9pdbizha",
        1: "5egkso5r",
        2: "tn9w6owf",
        3: "xgg06ldb",
        4: "yz2wehd0",
    },
}

# MGMD KL-Budget: March 25 HalfCheetah (10 seeds) + March 27 all envs (1 seed)
MGMD_KL_BUDGET_IDS = {
    "HalfCheetah-v4": {
        5: "bscdb6k6",
        6: "s77ywsoz",
        7: "hurp23vk",
        8: "8xcmblzz",
        9: "qlcbvy0i",
        10: "8dw3rm98",
        11: "j74xbjv4",
        12: "5bu21100",
        13: "jb8f66ci",
        14: "6ljcurqt",
    },
    "Ant-v4": {0: "99q9yepa"},
    "Walker2d-v4": {0: "ae1mp0y6"},
    "Swimmer-v4": {0: "rvxnwnnv"},
    "Humanoid-v4": {0: "o97g3a8w"},
    "Hopper-v4": {0: "wwxlh0z8"},
}

# March 27 HalfCheetah run (same variant as March 25, just different seed)
MGMD_KL_BUDGET_EXTRA_HC = {0: "fptp2r7r"}

# MGMD dist_shift_eta: March 30 (1 seed per env)
MGMD_DIST_SHIFT_ETA_IDS = {
    "HalfCheetah-v4": {1: "w0nsyotu"},
    "Hopper-v4": {1: "bk8pkd0j"},
    "Walker2d-v4": {1: "xe358pb3"},
    "Ant-v4": {1: "6mdw2hrs"},
    "Swimmer-v4": {1: "mx6j3c41"},
    "Humanoid-v4": {1: "xbxyxj3p"},
}

# Environments for the all-envs plot
ALL_ENVS = [
    "HalfCheetah-v4",
    "Hopper-v4",
    "Walker2d-v4",
    "Ant-v4",
    "Humanoid-v4",
    "Swimmer-v4",
]

# LSAC baseline data directory (from cluster; may not exist locally)
LSAC_DATA_DIR = os.path.expanduser("~/LSAC/data")

# LSAC methods to include
LSAC_ALGO_MAP = {
    "OURS": "LSAC",
    "DSAC": "DSAC",
    "SAC": "SAC",
    "TD3": "TD3",
    "DIPO": "DIPO",
    "PPO": "PPO",
    "TRPO": "TRPO",
}

LSAC_ENV_KEYS = {
    "HalfCheetah-v4": "halfcheetah",
    "Hopper-v4": "hopper",
    "Walker2d-v4": "walker2d",
    "Ant-v4": "ant",
    "Humanoid-v4": "humanoid",
    "Swimmer-v4": "swimmer",
}

# ── Color scheme ─────────────────────────────────────────────────────────────
METHOD_COLORS = {
    "MGMD": "C0",
    "MGMD (KL-Budget)": "C1",
    "MGMD (dist_shift_eta)": "C2",
    "LSAC": "C3",
    "DSAC": "C4",
    "SAC": "C5",
    "TD3": "C6",
    "DIPO": "C7",
    "PPO": "C8",
    "TRPO": "C9",
}

# ── Data fetching ────────────────────────────────────────────────────────────

MAX_STEPS = 1_000_000
STEP_INTERVAL = 10_000
TARGET_STEPS = np.arange(0, MAX_STEPS + STEP_INTERVAL, STEP_INTERVAL)


def get_metric_key(run):
    """Determine the correct metric key for a run."""
    env = run.config.get("env", "")
    # Try the new format first
    new_key = f"episode_return/{env}"
    old_key = "sample/episode_return"

    # Sample a few rows to check which key exists
    h = run.history(samples=3)
    if new_key in h.columns:
        return new_key
    if old_key in h.columns:
        return old_key
    # Fallback: check all columns
    for col in h.columns:
        if "episode_return" in col:
            return col
    return old_key


def fetch_curve(run, metric_key=None):
    """Fetch and interpolate a training curve from a wandb run."""
    if metric_key is None:
        metric_key = get_metric_key(run)

    history = run.history(keys=[metric_key, "_step"], samples=10000)
    if history.empty or metric_key not in history.columns:
        print(f"    WARNING: no data for metric {metric_key} in run {run.id}")
        return None
    history = history.dropna(subset=[metric_key])
    if len(history) == 0:
        return None

    steps = history["_step"].values
    values = history[metric_key].values
    interp = np.interp(TARGET_STEPS, steps, values, left=np.nan, right=values[-1])
    return interp


def fetch_runs(api, id_dict):
    """Fetch curves for a dict of {env: {seed: run_id}}. Returns {env: [curves]}."""
    result = defaultdict(list)
    for env, seeds in id_dict.items():
        print(f"  {env} ({len(seeds)} seeds)...")
        for seed, run_id in sorted(seeds.items()):
            run = api.run(f"{WANDB_PROJECT}/{run_id}")
            curve = fetch_curve(run)
            if curve is not None:
                result[env].append(curve)
                print(f"    seed {seed}: OK")
            else:
                print(f"    seed {seed}: FAILED")
    return dict(result)


def load_lsac_baselines():
    """Load LSAC baseline data from pickle files if available."""
    if not os.path.isdir(LSAC_DATA_DIR):
        print(f"  LSAC data directory not found: {LSAC_DATA_DIR}")
        return {}

    import pandas as pd
    from io import StringIO

    baselines = {}
    for env, env_key in LSAC_ENV_KEYS.items():
        pkl_path = os.path.join(LSAC_DATA_DIR, f"all_data_{env_key}.pkl")
        if not os.path.exists(pkl_path):
            continue
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        df = pd.read_csv(StringIO(data))
        baselines[env] = df
    return baselines


# ── Plotting helpers ─────────────────────────────────────────────────────────

def compute_mean_ci(curves, ci_level=0.95):
    """Compute mean and CI from a list of curves."""
    arr = np.array(curves)
    n = arr.shape[0]
    mean = np.nanmean(arr, axis=0)
    if n > 1:
        sem = stats.sem(arr, axis=0, nan_policy="omit")
        t_crit = stats.t.ppf(1 - (1 - ci_level) / 2, df=n - 1)
        ci = t_crit * sem
    else:
        ci = np.zeros_like(mean)
    return mean, ci, n


def plot_method(ax, curves, label, color, ci_level=0.95, linestyle="-", linewidth=1.5):
    """Plot a method's mean curve with CI band."""
    if not curves:
        return
    mean, ci, n = compute_mean_ci(curves, ci_level)
    ax.plot(
        TARGET_STEPS / 1e6, mean,
        label=label,
        color=color, linewidth=linewidth, linestyle=linestyle,
    )
    ax.fill_between(TARGET_STEPS / 1e6, mean - ci, mean + ci, alpha=0.2, color=color)


def plot_lsac_baselines(ax, df, ci_level=0.90, n_seeds=10):
    """Plot LSAC baseline methods on an axis."""
    t_crit = stats.t.ppf(1 - (1 - ci_level) / 2, df=n_seeds - 1)
    for lsac_name, display_name in LSAC_ALGO_MAP.items():
        if lsac_name not in df["algo"].values:
            continue
        algo_df = df[df["algo"] == lsac_name].sort_values("steps")
        steps = algo_df["steps"].values / 1e6
        means = algo_df["rew_mean"].values
        stds = algo_df["rew_std"].values
        cis = t_crit * stds / np.sqrt(n_seeds)

        color = METHOD_COLORS.get(display_name, "gray")
        ax.plot(steps, means, label=display_name, linewidth=1.0, color=color, linestyle="--")
        ax.fill_between(steps, means - cis, means + cis, alpha=0.1, color=color)


def style_axis(ax, title):
    """Apply common styling to an axis."""
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Steps (M)", fontsize=9)
    ax.set_ylabel("Episode Return", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=8)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.environ.setdefault(
        "WANDB_API_KEY",
        "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",
    )
    api = wandb.Api(timeout=120)

    figures_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures"
    )
    os.makedirs(figures_dir, exist_ok=True)

    # ── Fetch data from wandb ────────────────────────────────────────────
    print("Fetching MGMD baseline (HalfCheetah)...")
    baseline_data = fetch_runs(api, MGMD_BASELINE_IDS)

    print("\nFetching MGMD KL-Budget...")
    kl_budget_data = fetch_runs(api, MGMD_KL_BUDGET_IDS)

    # Add March 27 HalfCheetah to KL-Budget pool for all-envs plot
    print("\nFetching MGMD KL-Budget extra HalfCheetah (March 27)...")
    kl_budget_hc_extra = fetch_runs(api, {"HalfCheetah-v4": MGMD_KL_BUDGET_EXTRA_HC})

    print("\nFetching MGMD dist_shift_eta...")
    dist_shift_data = fetch_runs(api, MGMD_DIST_SHIFT_ETA_IDS)

    print("\nLoading LSAC baselines...")
    lsac_baselines = load_lsac_baselines()
    if lsac_baselines:
        print(f"  Loaded for: {list(lsac_baselines.keys())}")
    else:
        print("  No LSAC baseline data available.")

    # ── Plot 1: HalfCheetah — MGMD vs MGMD (KL-Budget) ──────────────────
    print("\n=== Generating HalfCheetah plot ===")
    fig, ax = plt.subplots(figsize=(10, 6))

    plot_method(
        ax, baseline_data.get("HalfCheetah-v4", []),
        "MGMD", METHOD_COLORS["MGMD"],
    )
    plot_method(
        ax, kl_budget_data.get("HalfCheetah-v4", []),
        "MGMD (KL-Budget)", METHOD_COLORS["MGMD (KL-Budget)"],
    )

    # Add LSAC baselines if available
    if "HalfCheetah-v4" in lsac_baselines:
        plot_lsac_baselines(ax, lsac_baselines["HalfCheetah-v4"])

    ax.set_xlabel("Steps (M)", fontsize=12)
    ax.set_ylabel("Episode Return", fontsize=12)
    ax.set_title("HalfCheetah-v4: MGMD Training Curves", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f"{x:.1f}M" if x < 1 else f"{x:.0f}M")
    )

    hc_path = os.path.join(figures_dir, "halfcheetah_mgmd_vs_kl_budget.png")
    plt.tight_layout()
    plt.savefig(hc_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {hc_path}")
    plt.close()

    # ── Plot 2: All environments — KL-Budget + dist_shift_eta ────────────
    print("\n=== Generating all-environments plot ===")
    n_envs = len(ALL_ENVS)
    n_cols = 3
    n_rows = (n_envs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()

    for idx, env in enumerate(ALL_ENVS):
        ax = axes[idx]

        # KL-Budget: combine March 25 + March 27 for HalfCheetah, just March 27 otherwise
        kl_curves = list(kl_budget_data.get(env, []))
        if env == "HalfCheetah-v4":
            kl_curves += kl_budget_hc_extra.get(env, [])
        plot_method(ax, kl_curves, "MGMD (KL-Budget)", METHOD_COLORS["MGMD (KL-Budget)"])

        # dist_shift_eta
        plot_method(
            ax, dist_shift_data.get(env, []),
            "MGMD (dist_shift_eta)", METHOD_COLORS["MGMD (dist_shift_eta)"],
        )

        # LSAC baselines
        env_key = LSAC_ENV_KEYS.get(env)
        if env_key and env in lsac_baselines:
            plot_lsac_baselines(ax, lsac_baselines[env])

        style_axis(ax, env.replace("-v4", ""))

    # Hide unused axes
    for idx in range(n_envs, len(axes)):
        axes[idx].set_visible(False)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 5),
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    all_path = os.path.join(figures_dir, "training_curves_all_envs_new.png")
    plt.savefig(all_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {all_path}")
    plt.close()


if __name__ == "__main__":
    main()
