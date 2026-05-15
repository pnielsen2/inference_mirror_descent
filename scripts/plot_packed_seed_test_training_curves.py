#!/usr/bin/env python3
"""Training curves for the packed-seed test runs, plotted against LSAC's
model-free baselines (SAC, TD3, DIPO, PPO, TRPO).

MGMD curve aggregation matches plot_sweep_4env_training_curves.py:
  - bin raw episode_return history every --bin-size steps
  - aggregate across runs per environment via mean + 50% t-CI
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from scipy import stats

SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_DIR = SCRIPT_DIR.parent
FIG_DIR = REPO_DIR / "figures"
LSAC_DATA_DIR = Path(os.path.expanduser("~/LSAC/data"))

WANDB_PROJECT = "pnielsen2-harvard/diffusion_online_rl"
DEFAULT_RUN_PATHS = [
    "pnielsen2-harvard/diffusion_online_rl/s9jvqf7s",
    "pnielsen2-harvard/diffusion_online_rl/xh3n8lg8",
    "pnielsen2-harvard/diffusion_online_rl/s99ltjhw",
    "pnielsen2-harvard/diffusion_online_rl/vrhhshhp",
    "pnielsen2-harvard/diffusion_online_rl/t4ygx90k",
    "pnielsen2-harvard/diffusion_online_rl/ojynfimf",
    "pnielsen2-harvard/diffusion_online_rl/m5eespyx",
    "pnielsen2-harvard/diffusion_online_rl/5dt4pbkw",
]
DEFAULT_ENV_ORDER = ["HalfCheetah-v3", "Ant-v3", "Walker2d-v3", "Humanoid-v3"]

BASELINE_ALGOS = ["SAC", "TD3", "DIPO", "PPO", "TRPO"]
METHOD_COLORS = {
    "MGMD (packed seeds)": "C0",
    "SAC": "C1",
    "TD3": "C2",
    "DIPO": "C3",
    "PPO": "C4",
    "TRPO": "C5",
}
LSAC_N_SEEDS = 10
LSAC_CI_LEVEL = 0.90
MGMD_CI_LEVEL = 0.50
PACKED_LABEL = "MGMD (packed seeds)"


def env_key_for_lsac(env: str) -> str:
    return env.split("-")[0].lower()


def load_lsac_baseline(env: str):
    key = env_key_for_lsac(env)
    pkl = LSAC_DATA_DIR / f"all_data_{key}.pkl"
    if not pkl.exists():
        return None
    import pickle
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    return pd.read_csv(StringIO(data))


def normalize_run_path(run_path: str) -> str:
    run_path = run_path.strip().strip("/")
    if run_path.count("/") == 0:
        return f"{WANDB_PROJECT}/{run_path}"
    return run_path


def fetch_history(run, env: str):
    candidate_keys = [f"episode_return/{env}", "sample/episode_return"]
    for key in candidate_keys:
        try:
            hist = run.history(keys=[key, "_step"], samples=20000)
        except Exception as e:
            print(f"    [warn] {run.id} wandb history failed for {key} ({e})", flush=True)
            continue
        if hist is None or key not in hist.columns:
            continue
        hist = hist.dropna(subset=[key])
        if len(hist) == 0:
            continue
        return hist[["_step", key]].rename(columns={key: "return"}).reset_index(drop=True)

    try:
        hist = run.history(samples=20000)
    except Exception as e:
        print(f"    [warn] {run.id} fallback history fetch failed ({e})", flush=True)
        return None
    if hist is None or "_step" not in hist.columns:
        return None
    for col in hist.columns:
        if col == "_step" or "episode_return" not in col:
            continue
        sub = hist.dropna(subset=[col])
        if len(sub) == 0:
            continue
        return sub[["_step", col]].rename(columns={col: "return"}).reset_index(drop=True)
    return None


def bin_curve(hist, bins, bin_size):
    steps = hist["_step"].values
    returns = hist["return"].values
    out = np.full(len(bins), np.nan)
    for i, t in enumerate(bins):
        mask = (steps > t - bin_size) & (steps <= t)
        if mask.any():
            out[i] = returns[mask].mean()
    return out


def aggregate(curves, ci_level=MGMD_CI_LEVEL):
    arr = np.stack(curves)
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
    t_crit = stats.t.ppf(1 - (1 - ci_level) / 2, df=n_seeds - 1)
    for algo in BASELINE_ALGOS:
        if algo not in df["algo"].values:
            continue
        ad = df[df["algo"] == algo].sort_values("steps")
        steps = ad["steps"].values / 1e6
        means = ad["rew_mean"].values
        stds = ad["rew_std"].values
        cis = t_crit * stds / np.sqrt(n_seeds)
        color = METHOD_COLORS[algo]
        ax.plot(steps, means, label=algo, linewidth=1.2, color=color)
        ax.fill_between(steps, means - cis, means + cis, alpha=0.15, color=color)


def pretty_env_name(env: str) -> str:
    for suffix in ("-v3", "-v4", "-v5"):
        if env.endswith(suffix):
            return env[: -len(suffix)]
    return env


def ordered_envs(seen_envs):
    seen_envs = list(seen_envs)
    base = [env for env in DEFAULT_ENV_ORDER if env in seen_envs]
    extras = sorted(env for env in seen_envs if env not in DEFAULT_ENV_ORDER)
    return base + extras


def gather_legend(fig):
    handles_by_label = {}
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            if label not in handles_by_label:
                handles_by_label[label] = handle
    return list(handles_by_label.values()), list(handles_by_label.keys())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-path", dest="run_paths", action="append", default=None,
                    help="W&B run path to include. Can be repeated. Default: the 8 packed-seed test runs.")
    ap.add_argument("--bin-size", type=int, default=5000)
    ap.add_argument("--max-steps", type=int, default=1_000_000)
    ap.add_argument("--out", type=Path, default=None,
                    help="Override output path. Default: figures/training_curves_packed_seed_test.png")
    args = ap.parse_args()

    run_paths = args.run_paths or DEFAULT_RUN_PATHS
    bins = np.arange(args.bin_size, args.max_steps + args.bin_size, args.bin_size)

    api = wandb.Api(timeout=120)
    print("Fetching run handles by id...", flush=True)
    match = []
    for run_path in run_paths:
        full_path = normalize_run_path(run_path)
        run = api.run(full_path)
        env = run.config.get("env")
        seed = run.config.get("seed")
        match.append((run, env, seed, full_path))
        print(f"  {run.id}: env={env}, seed={seed}", flush=True)

    envs = ordered_envs({env for (_, env, _, _) in match if env is not None})
    by_env = {env: [] for env in envs}
    for run, env, seed, full_path in match:
        if env is None:
            print(f"  [warn] {full_path} has no env in config; skipping", flush=True)
            continue
        by_env.setdefault(env, []).append((run, seed))

    print("Fetching episode-return histories...", flush=True)
    hist_by_env = {env: [] for env in by_env}

    def _one(run, env, seed):
        return env, seed, fetch_history(run, env)

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(match)))) as ex:
        futs = []
        for env in envs:
            for run, seed in by_env.get(env, []):
                futs.append(ex.submit(_one, run, env, seed))
        for fut in as_completed(futs):
            env, seed, hist = fut.result()
            if hist is not None and len(hist) > 0:
                hist_by_env[env].append((seed, hist))

    for env in envs:
        seeds = [seed for seed, _ in hist_by_env[env]]
        print(f"  {env:<16s}: {len(hist_by_env[env])} runs, {len(set(seeds))} distinct seeds", flush=True)

    mgmd_per_env = {}
    for env in envs:
        if not hist_by_env[env]:
            continue
        curves = [bin_curve(hist, bins, args.bin_size) for (_, hist) in hist_by_env[env]]
        mean, ci, n = aggregate(curves, ci_level=MGMD_CI_LEVEL)
        mgmd_per_env[env] = (mean, ci, n)

    n_envs = max(1, len(envs))
    n_cols = 2
    n_rows = int(np.ceil(n_envs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 3.5 * n_rows))
    axes = np.atleast_1d(axes).flatten()
    mgmd_color = METHOD_COLORS[PACKED_LABEL]

    for idx, env in enumerate(envs):
        ax = axes[idx]
        if env in mgmd_per_env:
            mean, ci, n = mgmd_per_env[env]
            ax.plot(bins / 1e6, mean, label=PACKED_LABEL, color=mgmd_color, linewidth=1.5)
            ax.fill_between(bins / 1e6, mean - ci, mean + ci, alpha=0.2, color=mgmd_color)
        bl = load_lsac_baseline(env)
        if bl is not None:
            plot_baselines(ax, bl)
        ax.set_title(pretty_env_name(env), fontsize=12)
        ax.set_xlabel("Steps (M)", fontsize=10)
        ax.set_ylabel("Episode Return", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=9)
        ax.set_xlim(0, args.max_steps / 1e6)

    for idx in range(len(envs), len(axes)):
        axes[idx].set_visible(False)

    handles, labels = gather_legend(fig)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles),
                   fontsize=10, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)

    FIG_DIR.mkdir(exist_ok=True)
    out = args.out if args.out is not None else FIG_DIR / "training_curves_packed_seed_test.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
