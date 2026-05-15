#!/usr/bin/env python3
"""Training curves for a single MGMD config from a launch.py sweep, plotted
against LSAC's model-free baselines (SAC, TD3, DIPO, PPO, TRPO).

Env set: HalfCheetah-v3, Ant-v3, Walker2d-v3, Humanoid-v3.

Required --sweep-id N picks which sweep's per-sweep TOPSIS outputs directory
(scripts/topsis_out/sweep_<N>/) to read for the run_id -> history lookup.

Which config: by default the top TOPSIS config from that sweep; override via
--config-tag to plot an arbitrary config by tag (must already have runs under
the given sweep_id). Output filename is labeled with sweep_id so re-runs for
different sweeps don't overwrite each other.

MGMD curve aggregation:
  - bin raw episode_return history every --bin-size steps (mean of raw
    returns whose _step falls in (t - bin_size, t] at bin boundary t)
  - aggregate across seeds of the chosen config via mean + 50% t-CI
"""

from __future__ import annotations

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
TOPSIS_OUT_ROOT = REPO_DIR / "scripts" / "topsis_out"

WANDB_PROJECT = "pnielsen2-harvard/diffusion_online_rl"
ENVS = ["HalfCheetah-v3", "Ant-v3", "Walker2d-v3", "Humanoid-v3"]

BASELINE_ALGOS = ["SAC", "TD3", "DIPO", "PPO", "TRPO"]
METHOD_COLORS = {
    "MGMD": "C0",
    "SAC":  "C1",
    "TD3":  "C2",
    "DIPO": "C3",
    "PPO":  "C4",
    "TRPO": "C5",
}
LSAC_N_SEEDS = 10
LSAC_CI_LEVEL = 0.90
MGMD_CI_LEVEL = 0.50


def env_key_for_lsac(env: str) -> str:
    """Map HalfCheetah-v3 -> halfcheetah (the LSAC pickle stem)."""
    return env.split("-")[0].lower()


def load_lsac_baseline(env: str):
    """Load LSAC baseline data for one env. Returns DataFrame or None."""
    key = env_key_for_lsac(env)
    pkl = LSAC_DATA_DIR / f"all_data_{key}.pkl"
    if not pkl.exists():
        return None
    import pickle
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    return pd.read_csv(StringIO(data))


def fetch_history(r, env):
    """Return DataFrame with columns _step, return, or None. Pure wandb."""
    key = f"episode_return/{env}"
    try:
        h = r.history(keys=[key, "_step"], samples=20000)
    except Exception as e:
        print(f"    [warn] {r.id} wandb history failed ({e})", flush=True)
        return None
    if h is None or key not in h.columns:
        return None
    h = h.dropna(subset=[key])
    if len(h) == 0:
        return None
    return h[["_step", key]].rename(columns={key: "return"}).reset_index(drop=True)


def bin_curve(hist, bins, bin_size):
    """Mean of raw returns whose _step falls in (t - bin_size, t] at each t."""
    steps = hist["_step"].values
    returns = hist["return"].values
    out = np.full(len(bins), np.nan)
    for i, t in enumerate(bins):
        mask = (steps > t - bin_size) & (steps <= t)
        if mask.any():
            out[i] = returns[mask].mean()
    return out


def aggregate(curves, ci_level=MGMD_CI_LEVEL):
    arr = np.stack(curves)  # (n, n_bins)
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


def top_topsis_config_tag(topsis_dir: Path):
    p = topsis_dir / "topsis_ranking.csv"
    if not p.exists():
        raise SystemExit(
            f"No TOPSIS ranking at {p}. Run "
            f"`python scripts/compute_topsis.py --sweep-id <N>` first."
        )
    df = pd.read_csv(p)
    if "config_tag" not in df.columns:
        raise SystemExit(
            f"{p} has no 'config_tag' column. Regenerate with the current "
            "compute_topsis.py."
        )
    df = df[~df["config_tag"].astype(str).str.startswith("baseline_")]
    return df.sort_values("topsis_score", ascending=False).iloc[0]["config_tag"]


def _maybe_int(v):
    return None if pd.isna(v) else int(v)


def _match_from_cache(api, metrics_csv: Path, cfg_tag: str):
    if not metrics_csv.exists():
        return None
    mdf = pd.read_csv(metrics_csv)
    if "config_tag" not in mdf.columns or "run_id" not in mdf.columns:
        return None
    cfg_df = mdf[mdf["config_tag"] == cfg_tag]
    if cfg_df.empty:
        return []
    print(f"  {len(cfg_df)} cached run rows match config_tag={cfg_tag}", flush=True)
    print("Fetching run handles by id...", flush=True)
    match = []
    for _, row in cfg_df.iterrows():
        r = api.run(f"{WANDB_PROJECT}/{row['run_id']}")
        match.append((
            r,
            row["env"],
            _maybe_int(row.get("seed")),
            _maybe_int(row.get("seed_index")),
        ))
    return match


def _match_from_wandb(api, sweep_id: int, cfg_tag: str):
    print(
        f"  local sweep cache did not provide run rows for config_tag={cfg_tag}; "
        "querying wandb directly...",
        flush=True,
    )
    stub_runs = list(api.runs(
        WANDB_PROJECT,
        filters={"config.sweep_id": int(sweep_id), "config.config_tag": cfg_tag},
        per_page=500,
    ))
    if not stub_runs:
        stub_runs = list(api.runs(
            WANDB_PROJECT,
            filters={"config.sweep_id": int(sweep_id)},
            per_page=500,
        ))
    if not stub_runs:
        raise SystemExit(f"No wandb runs found for sweep {sweep_id}")
    def _full_fetch(run_id: str):
        return api.run(f"{WANDB_PROJECT}/{run_id}")
    with ThreadPoolExecutor(max_workers=8) as ex:
        runs = list(ex.map(_full_fetch, [r.id for r in stub_runs]))
    match = []
    for r in runs:
        if r.config.get("config_tag") != cfg_tag:
            continue
        env = r.group if (r.group in ENVS) else r.config.get("env")
        if env not in ENVS:
            continue
        seed = r.config.get("seed")
        seed_index = r.config.get("seed_index")
        match.append((
            r,
            env,
            int(seed) if seed is not None else None,
            int(seed_index) if seed_index is not None else None,
        ))
    if not match:
        raise SystemExit(
            f"No wandb runs found for sweep {sweep_id} with config_tag={cfg_tag}. "
            f"If you expected them, rerun `python scripts/compute_topsis.py --sweep-id {sweep_id}` "
            "to refresh the local cache and verify the tag."
        )
    print(f"  {len(match)} wandb runs match config_tag={cfg_tag}", flush=True)
    return match


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep-id", "--sweep_id", type=int, default=None,
                    help="Which sweep to plot from. Required unless "
                        "--config-tag encodes it as 'sweep<N>_...' (the "
                         "build_config_tag convention in vmap_off_policy.py), "
                         "in which case <N> is parsed from the tag.")
    ap.add_argument("--config-tag", "--config_tag", "--config-key",
                    dest="config_tag", type=str, default=None,
                    help="Which config to plot. Default: TOPSIS top config "
                         "from scripts/topsis_out/sweep_<N>/topsis_ranking.csv. "
                         "Pass a full config_tag string to plot an arbitrary "
                         "config (it must exist in the sweep's all_runs_metrics.csv). "
                         "If the tag starts with 'sweep<N>_' and --sweep-id "
                         "isn't given, <N> is inferred from the tag.")
    ap.add_argument("--bin-size", type=int, default=5000)
    ap.add_argument("--max-steps", type=int, default=1_000_000)
    ap.add_argument("--out", type=Path, default=None,
                    help="Override output path. Default: "
                         "figures/training_curves_sweep<N>.png (or "
                         "sweep<N>_<short_tag>.png when --config-tag is set).")
    args = ap.parse_args()

    # Resolve sweep_id: explicit flag wins; else parse from config_tag's
    # 'sweep<N>_...' prefix. If both are present and disagree, error out so
    # users can't accidentally plot a sweep's tag against another sweep's data.
    import re
    tag_sweep_id = None
    if args.config_tag:
        m = re.match(r"sweep(\d+)_", args.config_tag)
        if m:
            tag_sweep_id = int(m.group(1))
    if args.sweep_id is None:
        if tag_sweep_id is None:
            ap.error("provide --sweep-id (or a --config-tag whose prefix is "
                     "'sweep<N>_') so we know which TOPSIS folder to read")
        sweep_id = tag_sweep_id
    else:
        sweep_id = args.sweep_id
        if tag_sweep_id is not None and tag_sweep_id != sweep_id:
            ap.error(f"--sweep-id={sweep_id} contradicts --config-tag prefix "
                     f"'sweep{tag_sweep_id}_...'")

    topsis_dir = TOPSIS_OUT_ROOT / f"sweep_{sweep_id}"
    cfg_tag = args.config_tag or top_topsis_config_tag(topsis_dir)
    print(f"Plotting MGMD config_tag from sweep {sweep_id}: {cfg_tag}")

    bins = np.arange(args.bin_size, args.max_steps + args.bin_size, args.bin_size)

    api = wandb.Api(timeout=120)
    metrics_csv = topsis_dir / "all_runs_metrics.csv"
    match = _match_from_cache(api, metrics_csv, cfg_tag)
    if match is None or not match:
        match = _match_from_wandb(api, sweep_id, cfg_tag)

    by_env: dict = {env: [] for env in ENVS}
    for (r, env, sv, si) in match:
        by_env[env].append((r, sv, si))

    print("Fetching episode-return histories...")
    hist_by_env: dict = {env: [] for env in ENVS}
    def _one(item, env):
        r, sv, si = item
        return (env, sv, fetch_history(r, env))
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = []
        for env in ENVS:
            for item in by_env[env]:
                futs.append(ex.submit(_one, item, env))
        for fut in as_completed(futs):
            env, sv, h = fut.result()
            if h is not None and len(h) > 0:
                hist_by_env[env].append((sv, h))

    for env in ENVS:
        n_seeds = len({sv for (sv, _) in hist_by_env[env]})
        print(f"  {env:<16s}: {len(hist_by_env[env])} runs, {n_seeds} distinct seeds")

    # Bin each run; aggregate per env.
    mgmd_per_env = {}
    for env in ENVS:
        if not hist_by_env[env]:
            continue
        curves = [bin_curve(h, bins, args.bin_size) for (_, h) in hist_by_env[env]]
        mean, ci, n = aggregate(curves, ci_level=MGMD_CI_LEVEL)
        mgmd_per_env[env] = (mean, ci, n)

    # Plot 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    mgmd_color = METHOD_COLORS["MGMD"]
    for idx, env in enumerate(ENVS):
        ax = axes[idx]
        if env in mgmd_per_env:
            mean, ci, n = mgmd_per_env[env]
            ax.plot(bins / 1e6, mean, label="MGMD", color=mgmd_color, linewidth=1.5)
            ax.fill_between(bins / 1e6, mean - ci, mean + ci, alpha=0.2, color=mgmd_color)
        bl = load_lsac_baseline(env)
        if bl is not None:
            plot_baselines(ax, bl)
        ax.set_title(env.replace("-v3", ""), fontsize=12)
        ax.set_xlabel("Steps (M)", fontsize=10)
        ax.set_ylabel("Episode Return", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=9)
        ax.set_xlim(0, args.max_steps / 1e6)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               fontsize=10, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.10)

    FIG_DIR.mkdir(exist_ok=True)
    if args.out is not None:
        out = args.out
    else:
        # Short suffix to disambiguate multi-config plots for the same sweep,
        # but only when the user passed a config_tag (otherwise it's the
        # TOPSIS-top and the sweep-id alone identifies it).
        if args.config_tag is not None:
            # Take a short hash of the full tag for filename friendliness.
            import hashlib
            short = hashlib.sha1(cfg_tag.encode()).hexdigest()[:8]
            out = FIG_DIR / f"training_curves_sweep{sweep_id}_{short}.png"
        else:
            out = FIG_DIR / f"training_curves_sweep{sweep_id}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
