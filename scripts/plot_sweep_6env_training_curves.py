#!/usr/bin/env python3

from __future__ import annotations

import ast
import argparse
import hashlib
import json
import re
import shlex
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from matplotlib import rcParams
from scipy import stats

import plot_sweep_4env_training_curves as base

ENVS = [
    "Ant-v3",
    "HalfCheetah-v3",
    "Hopper-v3",
    "Humanoid-v3",
    "Swimmer-v3",
    "Walker2d-v3",
]
METHOD_COLORS = dict(base.METHOD_COLORS)
GRID_STEP = 5
SMOOTH_WINDOW_EPISODES = 55
MGMD_CI_LEVEL = 0.90

base.ENVS = ENVS


def configure_plot_style():
    rcParams["grid.alpha"] = 0.3
    rcParams["grid.linestyle"] = "--"


def fetch_history(run, env: str):
    key = f"episode_return/{env}"
    try:
        rows = list(run.scan_history(keys=["_step", key], page_size=10_000))
    except Exception as e:
        print(f"    [warn] {run.id} wandb scan_history failed ({e})", flush=True)
        try:
            hist = run.history(keys=[key, "_step"], samples=100_000)
        except Exception as ee:
            print(f"    [warn] {run.id} wandb history failed ({ee})", flush=True)
            return None
        if hist is None or key not in hist.columns:
            return None
        hist = hist.dropna(subset=[key])
        if len(hist) == 0:
            return None
        return hist[["_step", key]].rename(columns={key: "return"}).reset_index(drop=True)
    if not rows:
        return None
    hist = pd.DataFrame(rows)
    if key not in hist.columns:
        return None
    hist = hist.dropna(subset=[key])
    if len(hist) == 0:
        return None
    return hist[["_step", key]].rename(columns={key: "return"}).reset_index(drop=True)


def seed_key(seed, seed_index, run_id: str):
    if seed_index is not None:
        return f"seed_index:{seed_index}"
    if seed is not None:
        return f"seed:{seed}"
    return f"run:{run_id}"


def _maybe_int(v):
    return None if pd.isna(v) else int(v)


def _cached_metadata(metrics_csv: Path, cfg_tag: str):
    if not metrics_csv.exists():
        return {}
    mdf = pd.read_csv(metrics_csv)
    if "config_tag" not in mdf.columns or "run_id" not in mdf.columns:
        return {}
    cfg_df = mdf[mdf["config_tag"] == cfg_tag]
    if cfg_df.empty:
        return {}
    meta = {}
    for _, row in cfg_df.iterrows():
        meta[str(row["run_id"])] = (
            row.get("env"),
            _maybe_int(row.get("seed")),
            _maybe_int(row.get("seed_index")),
        )
    print(f"  {len(meta)} cached run rows match config_tag={cfg_tag}", flush=True)
    return meta


def _coerce_scalar(v):
    if isinstance(v, str):
        if v == "True":
            return True
        if v == "False":
            return False
        if v == "None":
            return None
        if re.fullmatch(r"[-+]?\d+", v):
            try:
                return int(v)
            except Exception:
                pass
        if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", v):
            try:
                return float(v)
            except Exception:
                pass
    return v


def _coerce_nested(v):
    if isinstance(v, dict):
        return {k: _coerce_nested(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_coerce_nested(val) for val in v]
    return _coerce_scalar(v)


def _format_tag_value(v) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        return f"{v:g}"
    return str(v)


def _parse_pack_inline(text: str | None):
    if not text:
        return None
    normalized = re.sub(r'([\{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', text)
    return _coerce_nested(ast.literal_eval(normalized))


def _parse_running_command(command: str):
    tokens = shlex.split(command)
    args = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:].replace("-", "_")
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            args[key] = _coerce_scalar(tokens[i + 1])
            i += 2
        else:
            args[key] = True
            i += 1
    return args


def _config_tag_from_args(args: dict, seed_index: int):
    hp_pack = None
    if args.get("hp_pack_inline"):
        hp_pack = _parse_pack_inline(args.get("hp_pack_inline"))
    elif args.get("hp_pack"):
        pack_path = Path(str(args["hp_pack"]))
        if pack_path.exists():
            with open(pack_path) as f:
                hp_pack = _coerce_nested(json.load(f))
    hparams = {k: v for k, v in args.items() if k not in {"hp_pack", "hp_pack_inline", "config_tag_keys"}}
    tag_keys = args.get("config_tag_keys")
    if tag_keys is not None and tag_keys is not True:
        tag_keys = [k.strip() for k in str(tag_keys).split(",") if k.strip()]
    else:
        tag_keys = None
    parts = []
    if tag_keys:
        for k in sorted(tag_keys):
            if hp_pack is not None and k in hp_pack:
                v = hp_pack[k][seed_index]
            elif k in hparams:
                v = hparams[k]
            else:
                continue
            parts.append(f"{k}={_format_tag_value(v)}")
    elif hp_pack is not None:
        for k in sorted(k for k in hp_pack if k != "seed"):
            parts.append(f"{k}={_format_tag_value(hp_pack[k][seed_index])}")
    body = "_".join(parts) or "single"
    sweep_id = args.get("sweep_id")
    return f"sweep{sweep_id}_{body}" if sweep_id is not None else body


def _parse_started_at(line: str):
    parts = line.strip().split()
    if len(parts) < 6:
        return None
    return datetime.strptime(f"{parts[1]} {parts[2]} {parts[3]} {parts[5]}", "%b %d %H:%M:%S %Y")


def _local_log_root(args: dict):
    if args.get("cluster"):
        return Path("/n/netscratch/nali_lab_seas/Lab/haitongma/sdac_logs") / "logs"
    return base.REPO_DIR / "logs"


def _locate_local_csv(args: dict, started_at):
    env = args.get("env")
    alg = str(args.get("alg", ""))
    seed = int(args.get("seed", 0))
    suffix = str(args.get("suffix", ""))
    env_dir = _local_log_root(args) / str(env)
    if not env_dir.exists():
        return None
    candidates = []
    pattern = re.compile(rf"^{re.escape(alg)}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}})_s{seed}_(.*)$")
    for child in env_dir.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if not m or m.group(2) != suffix:
            continue
        csv_path = child / "episode_returns.csv"
        if not csv_path.exists():
            continue
        try:
            dt = datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S")
        except Exception:
            continue
        dist = abs((dt - started_at).total_seconds()) if started_at is not None else float("inf")
        candidates.append((dist, -dt.timestamp(), csv_path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _read_local_slurm_headers(slurm_out: Path):
    running = None
    started_at = None
    with open(slurm_out) as f:
        for i, line in enumerate(f):
            if line.startswith("Running: "):
                running = line[len("Running: "):].strip()
            elif line.startswith("Started at: "):
                started_at = _parse_started_at(line[len("Started at: "):])
            if running is not None and started_at is not None:
                break
            if i >= 20:
                break
    return running, started_at


def discover_local_histories(sweep_id: int, cfg_tag: str):
    slurm_root = base.REPO_DIR / "logs" / "slurm"
    found = []
    if not slurm_root.exists():
        return found
    seen = set()
    out_files = sorted(slurm_root.glob("**/*.out"), key=lambda p: p.stat().st_mtime, reverse=True)
    for slurm_out in out_files:
        running, started_at = _read_local_slurm_headers(slurm_out)
        if running is None:
            continue
        args = _parse_running_command(running)
        if args.get("sweep_id") != sweep_id:
            continue
        env = args.get("env")
        if env not in ENVS:
            continue
        csv_path = _locate_local_csv(args, started_at)
        if csv_path is None:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        if "seed" not in df.columns or "step" not in df.columns:
            continue
        value_col = f"episode_return/{env}"
        if value_col not in df.columns:
            candidates = [c for c in df.columns if c.startswith("episode_return/")]
            if len(candidates) != 1:
                continue
            value_col = candidates[0]
        for seed_index_val in sorted(df["seed"].dropna().astype(int).unique().tolist()):
            if _config_tag_from_args(args, seed_index_val) != cfg_tag:
                continue
            key = (csv_path, int(seed_index_val))
            if key in seen:
                continue
            seen.add(key)
            hist = (
                df[df["seed"].astype(int) == int(seed_index_val)][["step", value_col]]
                .rename(columns={"step": "_step", value_col: "return"})
                .reset_index(drop=True)
            )
            if len(hist) == 0:
                continue
            found.append((env, None, int(seed_index_val), f"local:{csv_path}:{seed_index_val}", hist))
    return found


def _api_runs_with_retry(api, filters, attempts: int = 4):
    last_exc = None
    for attempt in range(1, attempts + 1):
        try:
            return list(api.runs(base.WANDB_PROJECT, filters=filters, per_page=500))
        except Exception as e:
            last_exc = e
            if attempt == attempts:
                break
            wait_s = min(30, 2 ** (attempt - 1))
            print(
                f"  wandb runs query failed on attempt {attempt}/{attempts} ({e}); retrying in {wait_s}s...",
                flush=True,
            )
            time.sleep(wait_s)
    raise last_exc


def resolve_match(api, metrics_csv: Path, sweep_id: int, cfg_tag: str):
    cached_meta = _cached_metadata(metrics_csv, cfg_tag)
    cached_envs = {env for env, _, _ in cached_meta.values() if env in ENVS}
    if cached_meta:
        print(
            f"  cached run rows cover {len(cached_envs)}/{len(ENVS)} target envs",
            flush=True,
        )
    print("Querying wandb for matching runs...", flush=True)
    runs = _api_runs_with_retry(
        api,
        {"config.sweep_id": int(sweep_id), "config.config_tag": cfg_tag},
    )
    if not runs:
        print(
            "  exact wandb filter returned no runs; falling back to sweep-wide query...",
            flush=True,
        )
        runs = _api_runs_with_retry(api, {"config.sweep_id": int(sweep_id)})
    if not runs:
        raise SystemExit(f"No wandb runs found for sweep {sweep_id}")
    match = []
    for run in runs:
        cached = cached_meta.get(run.id)
        if cached is not None:
            env, seed, seed_index = cached
        else:
            env = run.group if (run.group in ENVS) else run.config.get("env")
            seed = run.config.get("seed")
            seed_index = run.config.get("seed_index")
            seed = int(seed) if seed is not None else None
            seed_index = int(seed_index) if seed_index is not None else None
        if env not in ENVS:
            continue
        match.append((run, env, seed, seed_index))
    if not match:
        raise SystemExit(
            f"No wandb runs found for sweep {sweep_id} with config_tag={cfg_tag}. "
            f"If you expected them, rerun `python scripts/compute_topsis.py --sweep-id {sweep_id}` "
            "to refresh the local cache and verify the tag."
        )
    print(f"  {len(match)} wandb runs match config_tag={cfg_tag}", flush=True)
    return match


def smooth_dense(hist: pd.DataFrame, grid: np.ndarray, grid_step: int, window_episodes: int):
    ordered = hist.sort_values("_step").reset_index(drop=True)
    steps = ordered["_step"].to_numpy(dtype=int)
    returns = ordered["return"].to_numpy(dtype=float)
    dense = np.full(len(grid), np.nan)
    if len(steps) == 0:
        return dense
    rolling = pd.Series(returns).rolling(window=window_episodes, min_periods=1).mean().to_numpy()
    right = np.searchsorted(steps, grid, side="right")
    left = np.searchsorted(steps, np.maximum(grid - grid_step, 0), side="right")
    mask = (grid > 0) & (right > left)
    if not mask.any():
        return dense
    point_x = grid[mask]
    point_y = rolling[right[mask] - 1]
    if len(point_x) == 1:
        dense[grid == point_x[0]] = point_y[0]
        return dense
    interp_mask = (grid >= point_x[0]) & (grid <= point_x[-1])
    dense[interp_mask] = np.interp(grid[interp_mask], point_x, point_y)
    return dense


def merge_curves(curves):
    arr = np.stack(curves)
    counts = np.sum(~np.isnan(arr), axis=0)
    merged = np.full(arr.shape[1], np.nan)
    valid = counts > 0
    if valid.any():
        merged[valid] = np.nansum(arr[:, valid], axis=0) / counts[valid]
    return merged


def aggregate(curves):
    arr = np.stack(curves)
    counts = np.sum(~np.isnan(arr), axis=0)
    median = np.full(arr.shape[1], np.nan)
    valid = counts > 0
    if valid.any():
        median[valid] = np.nanmedian(arr[:, valid], axis=0)
    ci = np.full(arr.shape[1], np.nan)
    for n in sorted({int(v) for v in counts if v > 1}):
        mask = counts == n
        std = np.nanstd(arr[:, mask], axis=0, ddof=1)
        sem = std / np.sqrt(float(n))
        t_crit = stats.t.ppf(1 - (1 - MGMD_CI_LEVEL) / 2, df=n - 1)
        ci[mask] = t_crit * sem
    ci[counts == 1] = 0.0
    return median, ci


def plot_baselines(ax, df: pd.DataFrame):
    base.plot_baselines(ax, df)


def baseline_y_bounds(df: pd.DataFrame):
    y_parts = []
    t_crit = stats.t.ppf(1 - (1 - base.LSAC_CI_LEVEL) / 2, df=base.LSAC_N_SEEDS - 1)
    for algo in base.BASELINE_ALGOS:
        if algo not in df["algo"].values:
            continue
        algo_data = df[df["algo"] == algo].sort_values("steps")
        means = algo_data["rew_mean"].to_numpy(dtype=float)
        stds = algo_data["rew_std"].to_numpy(dtype=float)
        cis = t_crit * stds / np.sqrt(base.LSAC_N_SEEDS)
        y_parts.append(means - cis)
        y_parts.append(means + cis)
    return y_parts


def fit_y_axis(ax, y_parts):
    finite = []
    for part in y_parts:
        arr = np.asarray(part, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite.append(arr)
    if not finite:
        return
    vals = np.concatenate(finite)
    ymin = float(np.min(vals))
    ymax = float(np.max(vals))
    if ymin == ymax:
        pad = max(1.0, 0.05 * max(abs(ymin), 1.0))
    else:
        pad = 0.05 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)


def _add_histories(dense_by_env_seed, hist_entries, grid: np.ndarray, grid_step: int, window_episodes: int):
    for env, seed, seed_index, run_id, hist in hist_entries:
        if env not in ENVS or hist is None or len(hist) == 0:
            continue
        dense = smooth_dense(hist, grid, grid_step, window_episodes)
        if np.all(np.isnan(dense)):
            continue
        dense_by_env_seed[env][seed_key(seed, seed_index, run_id)].append(dense)


def _fetch_remote_histories(match, covered_keys):
    by_env = {env: [] for env in ENVS}
    for run, env, seed, seed_index in match:
        key = seed_key(seed, seed_index, run.id)
        if env in by_env and key not in covered_keys[env]:
            by_env[env].append((run, seed, seed_index))
    print("Fetching episode-return histories from wandb...")
    fetched = []

    def _one(item, env):
        run, seed, seed_index = item
        hist = fetch_history(run, env)
        return env, seed, seed_index, run.id, hist

    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = []
        for env in ENVS:
            for item in by_env[env]:
                futs.append(ex.submit(_one, item, env))
        for fut in as_completed(futs):
            fetched.append(fut.result())
    return fetched


def collect_mgmd_curves(local_histories, match, grid: np.ndarray, grid_step: int, window_episodes: int):
    dense_by_env_seed = {env: defaultdict(list) for env in ENVS}
    if local_histories:
        print("Loading local episode-return histories...")
        _add_histories(dense_by_env_seed, local_histories, grid, grid_step, window_episodes)
    local_envs = {env for env in ENVS if dense_by_env_seed[env]}
    if len(local_envs) < len(ENVS):
        covered_keys = {env: set(dense_by_env_seed[env].keys()) for env in ENVS}
        remote_histories = _fetch_remote_histories(match, covered_keys)
        _add_histories(dense_by_env_seed, remote_histories, grid, grid_step, window_episodes)
    mgmd_per_env = {}
    for env in ENVS:
        run_count = sum(len(curves) for curves in dense_by_env_seed[env].values())
        seed_count = len(dense_by_env_seed[env])
        print(f"  {env:<16s}: {run_count} runs, {seed_count} distinct seeds")
        if not dense_by_env_seed[env]:
            continue
        seed_curves = [merge_curves(curves) for curves in dense_by_env_seed[env].values()]
        median, ci = aggregate(seed_curves)
        mgmd_per_env[env] = (median, ci)
    return mgmd_per_env


def build_output_path(args, cfg_tag: str, sweep_id: int):
    if args.out is not None:
        return args.out
    if args.config_tag is not None:
        short = hashlib.sha1(cfg_tag.encode()).hexdigest()[:8]
        return base.FIG_DIR / f"training_curves_6env_sweep{sweep_id}_{short}.png"
    return base.FIG_DIR / f"training_curves_6env_sweep{sweep_id}.png"


def main():
    ap = argparse.ArgumentParser(description="Plot 6-env smoothed training curves for a sweep config.")
    ap.add_argument("--sweep-id", "--sweep_id", type=int, default=None)
    ap.add_argument(
        "--config-tag",
        "--config_tag",
        "--config-key",
        dest="config_tag",
        type=str,
        default=None,
    )
    ap.add_argument("--max-steps", type=int, default=1_000_000)
    ap.add_argument("--grid-step", type=int, default=GRID_STEP)
    ap.add_argument("--smooth-window-episodes", type=int, default=SMOOTH_WINDOW_EPISODES)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    tag_sweep_id = None
    if args.config_tag:
        tag_match = re.match(r"sweep(\d+)_", args.config_tag)
        if tag_match:
            tag_sweep_id = int(tag_match.group(1))
    if args.sweep_id is None:
        if tag_sweep_id is None:
            ap.error("provide --sweep-id (or a --config-tag whose prefix is 'sweep<N>_')")
        sweep_id = tag_sweep_id
    else:
        sweep_id = args.sweep_id
        if tag_sweep_id is not None and tag_sweep_id != sweep_id:
            ap.error(f"--sweep-id={sweep_id} contradicts --config-tag prefix 'sweep{tag_sweep_id}_...'")

    configure_plot_style()

    topsis_dir = base.TOPSIS_OUT_ROOT / f"sweep_{sweep_id}"
    cfg_tag = args.config_tag or base.top_topsis_config_tag(topsis_dir)
    print(f"Plotting MGMD config_tag from sweep {sweep_id}: {cfg_tag}")

    grid = np.arange(0, args.max_steps + args.grid_step, args.grid_step, dtype=int)

    metrics_csv = topsis_dir / "all_runs_metrics.csv"
    local_histories = discover_local_histories(sweep_id, cfg_tag)
    if local_histories:
        print(f"Found {len(local_histories)} local seed histories for {cfg_tag}")
    api = None
    match = []
    local_envs = {env for env, _, _, _, _ in local_histories}
    if len(local_envs) < len(ENVS):
        api = wandb.Api(timeout=120)
        match = resolve_match(api, metrics_csv, sweep_id, cfg_tag)

    mgmd_per_env = collect_mgmd_curves(local_histories, match, grid, args.grid_step, args.smooth_window_episodes)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    axes = axes.flatten()
    mgmd_color = METHOD_COLORS["MGMD"]
    for idx, env in enumerate(ENVS):
        ax = axes[idx]
        y_parts = []
        if env in mgmd_per_env:
            median, ci = mgmd_per_env[env]
            x = grid / 1e6
            valid = ~np.isnan(median)
            if valid.any():
                ax.plot(x[valid], median[valid], label="MGMD", color=mgmd_color, linewidth=1.5)
                y_parts.append(median[valid])
                band_mask = valid & ~np.isnan(ci)
                if band_mask.any():
                    ax.fill_between(
                        x[band_mask],
                        median[band_mask] - ci[band_mask],
                        median[band_mask] + ci[band_mask],
                        alpha=0.2,
                        color=mgmd_color,
                    )
                    y_parts.append(median[band_mask] - ci[band_mask])
                    y_parts.append(median[band_mask] + ci[band_mask])
        baseline_df = base.load_lsac_baseline(env)
        if baseline_df is not None:
            plot_baselines(ax, baseline_df)
            y_parts.extend(baseline_y_bounds(baseline_df))
        ax.set_xlim(0, args.max_steps / 1e6)
        ax.set_title(env.replace("-v3", ""), fontsize=12)
        ax.set_xlabel("Steps (M)", fontsize=10)
        ax.set_ylabel("Episode Return", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=9)
        fit_y_axis(ax, y_parts)

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
    out = build_output_path(args, cfg_tag, sweep_id)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
