import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from scipy import stats

REPO_DIR = Path(__file__).parent.parent.resolve()
FIG_DIR = REPO_DIR / "figures"
WANDB_PROJECT = "pnielsen2-harvard/diffusion_online_rl"
DEFAULT_ENV_ORDER = [
    "Ant-v3",
    "HalfCheetah-v3",
    "Hopper-v3",
    "Humanoid-v3",
    "Swimmer-v3",
    "Walker2d-v3",
]


def resolve_env(run):
    env = run.group if run.group else None
    if env:
        return env
    return run.config.get("env")


def fetch_history(run, env: str):
    key = f"episode_return/{env}"
    try:
        rows = list(run.scan_history(keys=["_step", key], page_size=10_000))
    except Exception as e:
        print(f"    [warn] {run.id} scan_history failed ({e})", flush=True)
        try:
            hist = run.history(keys=[key, "_step"], samples=100_000)
        except Exception as ee:
            print(f"    [warn] {run.id} history failed ({ee})", flush=True)
            return None
        if hist is None or key not in hist.columns:
            return None
        hist = hist.dropna(subset=[key])
        if len(hist) == 0:
            return None
        return hist[["_step", key]].rename(columns={key: "return"}).sort_values("_step").reset_index(drop=True)
    if not rows:
        return None
    hist = pd.DataFrame(rows)
    if key not in hist.columns:
        return None
    hist = hist.dropna(subset=[key])
    if len(hist) == 0:
        return None
    return hist[["_step", key]].rename(columns={key: "return"}).sort_values("_step").reset_index(drop=True)


def windowed_mean_curve(hist: pd.DataFrame, bins: np.ndarray, bin_size: int):
    steps = hist["_step"].to_numpy(dtype=int)
    returns = hist["return"].to_numpy(dtype=float)
    curve = np.full(len(bins), np.nan)
    for i, t in enumerate(bins):
        mask = (steps > t - bin_size) & (steps <= t)
        if mask.any():
            curve[i] = float(np.mean(returns[mask]))
    return curve


def aggregate_curves(curves, ci_level: float):
    arr = np.stack(curves)
    counts = np.sum(~np.isnan(arr), axis=0)
    mean = np.full(arr.shape[1], np.nan)
    ci = np.full(arr.shape[1], np.nan)
    valid = counts > 0
    if valid.any():
        mean[valid] = np.nanmean(arr[:, valid], axis=0)
    ci[counts == 1] = 0.0
    for n in sorted({int(v) for v in counts if v > 1}):
        mask = counts == n
        std = np.nanstd(arr[:, mask], axis=0, ddof=1)
        sem = std / np.sqrt(float(n))
        t_crit = stats.t.ppf(1 - (1 - ci_level) / 2, df=n - 1)
        ci[mask] = t_crit * sem
    return mean, ci, counts


def full_runs_for_sweep(api: wandb.Api, sweep_id: int):
    stub_runs = list(api.runs(WANDB_PROJECT, filters={"config.sweep_id": int(sweep_id)}, per_page=500))
    print(f"Found {len(stub_runs)} stub runs for sweep {sweep_id}", flush=True)
    if not stub_runs:
        return []

    def _full_fetch(run_id: str):
        return api.run(f"{WANDB_PROJECT}/{run_id}")

    with ThreadPoolExecutor(max_workers=8) as ex:
        return list(ex.map(_full_fetch, [run.id for run in stub_runs]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-id", "--sweep_id", type=int, required=True)
    ap.add_argument("--exclude-env", "--exclude_env", action="append", default=[])
    ap.add_argument("--bin-size", type=int, default=10_000)
    ap.add_argument("--ci-level", type=float, default=0.90)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    exclude_envs = set(args.exclude_env)
    api = wandb.Api(timeout=120)
    runs = full_runs_for_sweep(api, args.sweep_id)
    if not runs:
        raise SystemExit(f"No runs found for sweep {args.sweep_id}")

    filtered = []
    for run in runs:
        env = resolve_env(run)
        if env is None or env in exclude_envs:
            continue
        filtered.append((run, env))

    if not filtered:
        raise SystemExit("No runs remain after filtering excluded environments")

    env_counts = Counter(env for _, env in filtered)
    env_order = [env for env in DEFAULT_ENV_ORDER if env in env_counts and env not in exclude_envs]
    env_order += [env for env in sorted(env_counts) if env not in env_order]
    print("Runs per environment:", dict(env_counts), flush=True)

    histories_by_env = defaultdict(list)

    def _fetch(item):
        run, env = item
        return env, run.id, run.state, fetch_history(run, env)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(_fetch, item) for item in filtered]
        for fut in as_completed(futs):
            env, run_id, state, hist = fut.result()
            if hist is None or len(hist) == 0:
                print(f"  {env:<16s} {run_id} {state:<10s} NO_HISTORY", flush=True)
                continue
            histories_by_env[env].append((run_id, state, hist))

    if not histories_by_env:
        raise SystemExit("No usable histories found")

    derived_max_step = max(int(hist["_step"].max()) for entries in histories_by_env.values() for _, _, hist in entries)
    max_steps = args.max_steps if args.max_steps is not None else int(np.ceil(derived_max_step / args.bin_size) * args.bin_size)
    bins = np.arange(args.bin_size, max_steps + args.bin_size, args.bin_size, dtype=int)

    aggregates = {}
    for env in env_order:
        entries = histories_by_env.get(env, [])
        if not entries:
            continue
        curves = [windowed_mean_curve(hist, bins, args.bin_size) for _, _, hist in entries]
        mean, ci, counts = aggregate_curves(curves, args.ci_level)
        aggregates[env] = (mean, ci, counts, len(entries))
        print(f"  {env:<16s}: {len(entries)} runs with usable history", flush=True)

    if not aggregates:
        raise SystemExit("No aggregates could be computed")

    n_envs = len(env_order)
    fig, axes = plt.subplots(1, n_envs, figsize=(4.5 * n_envs, 4.2), squeeze=False, sharex=True)
    axes = axes.flatten()

    for idx, env in enumerate(env_order):
        ax = axes[idx]
        if env not in aggregates:
            ax.set_title(env.replace("-v3", ""))
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        mean, ci, counts, n_runs = aggregates[env]
        x = bins / 1e6
        valid = ~np.isnan(mean)
        ax.plot(x[valid], mean[valid], color="C0", linewidth=1.8, label="Mean return")
        band_mask = valid & ~np.isnan(ci)
        if band_mask.any():
            ax.fill_between(x[band_mask], mean[band_mask] - ci[band_mask], mean[band_mask] + ci[band_mask], color="C0", alpha=0.2, label="90% CI")
        ax.set_title(f"{env.replace('-v3', '')}\n(n={n_runs})", fontsize=11)
        ax.set_xlabel("Env steps (M)", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Episode return", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=9)
        ax.set_xlim(0, max_steps / 1e6)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(handles), fontsize=10, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.17)

    FIG_DIR.mkdir(exist_ok=True)
    out_path = args.out or (FIG_DIR / f"training_curves_sweep{args.sweep_id}_all_runs_excl_{'-'.join(env.replace('-v3', '').lower() for env in sorted(exclude_envs)) if exclude_envs else 'none'}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}", flush=True)
    plt.close()


if __name__ == "__main__":
    main()
