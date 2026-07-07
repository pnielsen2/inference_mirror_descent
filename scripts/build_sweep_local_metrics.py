#!/usr/bin/env python3
"""Reconstruct per-(config_tag, env) tail-return metrics for a sweep entirely
from LOCAL data -- no wandb network calls.

Sources (all on disk):
  * Offline wandb run dirs written by launch.py --wandb_offline:
        $WANDB_OFFLINE_BASE/sweep_<N>/job_*/wandb/offline-run-*/files/config.yaml
    give per-slot env, config_tag, seed_index, denoising_predictor, and the
    ablated hyperparameters.
  * The run's display name (``<exp_dir.name>-s<slot>``) recoverable from the
    run-*.wandb header tells us exp_dir.name; wandb-metadata.json's ``program``
    path tells us the codebase snapshot dir, so the episode-return mirror lives
    at  <codebase>/logs/<env>/<exp_dir.name>/episode_returns.csv .
  * The episode-return mirror CSV (seed=slot,step,episode_return/<env>) holds
    the actual return curve per vmap slot -- written by SampleMetricsRecorder
    precisely so analysis survives wandb rate-limit drops.
  * LSAC baselines (~/LSAC/data/*.pkl) for the benchmark + log score.

Metric per (config_tag, env) matches compute_topsis: per slot, mean of episode
returns whose env step is within TAIL_WINDOW_STEPS of that slot's max step;
averaged across slots (seeds) sharing the config_tag.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from compute_topsis import (  # noqa: E402
    ENVS,
    MIN_EPISODES,
    TAIL_WINDOW_STEPS,
    load_benchmark_scores,
    log_score,
)

from relax.utils.fs import WANDB_OFFLINE_BASE as DEFAULT_WANDB_BASE  # noqa: E402
# A run counts as "finished" once its last logged env step reaches this fraction
# of its configured total_step; unfinished runs are excluded so their
# short-horizon tail metric never pollutes the heatmaps.
COMPLETION_FRAC = 0.95
RUN_NAME_RE = re.compile(rb"([A-Za-z0-9]+_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_s\d+_[^\x00]*?)-s(\d+)")
HP_KEYS = ["T", "eta", "guidance_gradient_space", "guidance_strength_multiplier"]
# Extra config fields carried through for downstream slicing (e.g. the
# advantage-normalization sweep) and completion checks.
EXTRA_KEYS = [
    "total_step",
    "advantage_normalization",
    "batch_advantage_normalization",
    "q_loss_normalization",
]


def _cfg_value(cfg: dict, key: str):
    node = cfg.get(key)
    if isinstance(node, dict):
        return node.get("value")
    return node


def _load_config_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    out = {"config_tag": _cfg_value(cfg, "config_tag"),
           "env": _cfg_value(cfg, "env"),
           "seed_index": _cfg_value(cfg, "seed_index"),
           "denoising_predictor": _cfg_value(cfg, "denoising_predictor")}
    for k in HP_KEYS + EXTRA_KEYS:
        out[k] = _cfg_value(cfg, k)
    return out


def _read_run_basename(wandb_file: Path) -> str | None:
    """Return exp_dir.name from a run-*.wandb header (run display name minus
    the trailing -s<slot>). Reads only the first chunk -- the name sits near
    the top of the transaction log."""
    try:
        with open(wandb_file, "rb") as f:
            head = f.read(262144)
    except OSError:
        return None
    m = RUN_NAME_RE.search(head)
    if not m:
        return None
    return m.group(1).decode("utf-8", "replace")


def _codebase_dir_from_metadata(meta_path: Path) -> Path | None:
    try:
        meta = json.load(open(meta_path))
    except (OSError, json.JSONDecodeError):
        return None
    program = meta.get("program")
    if not program:
        return None
    # <codebase>/scripts/train_mujoco.py -> <codebase>
    return Path(program).resolve().parents[1]


def _slot_tail_metric(steps: np.ndarray, returns: np.ndarray):
    """Return (tail_mean, max_step, n_episodes) or None if too few episodes."""
    if returns.size < MIN_EPISODES:
        return None
    max_step = float(steps.max())
    tail = returns[steps > max_step - TAIL_WINDOW_STEPS]
    if tail.size == 0:
        return None
    return float(np.mean(tail)), max_step, int(returns.size)


def _process_job(job_dir: Path) -> list[dict]:
    """Return per-(slot) metric records for one slurm job's offline runs."""
    run_dirs = sorted((job_dir / "wandb").glob("offline-run-*"))
    if not run_dirs:
        run_dirs = sorted(job_dir.glob("**/offline-run-*"))
    if not run_dirs:
        return []

    # Per-slot config from each offline run.
    slot_cfg: dict[int, dict] = {}
    base_name = None
    codebase = None
    env = None
    for rd in run_dirs:
        cfg_path = rd / "files" / "config.yaml"
        if not cfg_path.exists():
            continue
        info = _load_config_yaml(cfg_path)
        si = info.get("seed_index")
        if si is None:
            continue
        slot_cfg[int(si)] = info
        env = env or info.get("env")
        if base_name is None:
            wandb_files = list(rd.glob("run-*.wandb"))
            if wandb_files:
                base_name = _read_run_basename(wandb_files[0])
        if codebase is None:
            codebase = _codebase_dir_from_metadata(rd / "files" / "wandb-metadata.json")

    if not slot_cfg or base_name is None or codebase is None or env is None:
        return []

    csv_path = codebase / "logs" / env / base_name / "episode_returns.csv"
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    ep_col = f"episode_return/{env}"
    if ep_col not in df.columns or "seed" not in df.columns or "step" not in df.columns:
        return []
    df = df.dropna(subset=[ep_col])

    records = []
    for slot, sub in df.groupby("seed"):
        info = slot_cfg.get(int(slot))
        if info is None or not info.get("config_tag"):
            continue
        res = _slot_tail_metric(
            sub["step"].to_numpy(dtype=float), sub[ep_col].to_numpy(dtype=float)
        )
        if res is None:
            continue
        metric, max_step, n_eps = res
        rec = {
            "config_tag": info["config_tag"],
            "env": env,
            "seed_index": int(slot),
            "denoising_predictor": info.get("denoising_predictor"),
            "metric": metric,
            "max_step": max_step,
            "n_episodes": n_eps,
        }
        for k in HP_KEYS + EXTRA_KEYS:
            rec[k] = info.get(k)
        records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep-id", "--sweep_id", type=int, default=89)
    ap.add_argument("--wandb-base", type=Path, default=DEFAULT_WANDB_BASE)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    sweep_root = args.wandb_base / f"sweep_{args.sweep_id}"
    job_dirs = sorted(sweep_root.glob("job_*"))
    if not job_dirs:
        sys.exit(f"No job dirs under {sweep_root}")
    print(f"Found {len(job_dirs)} job dirs under {sweep_root}", flush=True)

    all_records: list[dict] = []
    for i, jd in enumerate(job_dirs, 1):
        recs = _process_job(jd)
        all_records.extend(recs)
        print(f"  [{i}/{len(job_dirs)}] {jd.name}: {len(recs)} slot metrics", flush=True)

    if not all_records:
        sys.exit("No slot metrics reconstructed; check paths.")

    runs = pd.DataFrame(all_records)
    runs["sweep_id"] = int(args.sweep_id)
    total = pd.to_numeric(runs["total_step"], errors="coerce")
    runs["finished"] = runs["max_step"] >= (COMPLETION_FRAC * total)
    n_unfinished = int((~runs["finished"]).sum())
    print(f"\nReconstructed {len(runs)} (config_tag, env, seed) slot metrics "
          f"across {runs['config_tag'].nunique()} config tags "
          f"({n_unfinished} unfinished slots excluded from aggregation).")

    benchmark_scores = load_benchmark_scores(ENVS)

    # Per-(config_tag, env): mean over FINISHED seeds, then log score vs benchmark.
    fin = runs[runs["finished"]]
    per_env = (
        fin.groupby(["config_tag", "env", "denoising_predictor"], dropna=False)["metric"]
        .agg(mean="mean", count="count")
        .reset_index()
    )
    per_env["benchmark_score"] = per_env["env"].map(benchmark_scores)
    per_env["log_score"] = [
        log_score(m, b) for m, b in zip(per_env["mean"], per_env["benchmark_score"])
    ]

    # Config index (hyperparameters + denoiser + adv-norm flags) for plotting
    # axes. Built from all runs so tags appear even if unfinished, but the
    # per-env / overall tables only carry finished data.
    config_index = (
        runs.drop_duplicates(subset=["config_tag"])[
            ["config_tag", "sweep_id", "denoising_predictor"] + HP_KEYS + EXTRA_KEYS
        ]
        .reset_index(drop=True)
    )

    # Overall per config_tag: mean log score across envs, requiring all ENVS.
    cov = per_env.groupby("config_tag")["env"].nunique()
    log_mean = per_env.groupby("config_tag")["log_score"].mean()
    overall = pd.DataFrame({"config_tag": cov.index, "n_envs": cov.values})
    overall = overall.merge(log_mean.rename("log_score").reset_index(), on="config_tag")
    overall = overall.merge(
        config_index[["config_tag", "denoising_predictor"]], on="config_tag", how="left"
    )
    overall["fraction_of_baseline"] = np.power(2.0, overall["log_score"])
    overall.loc[overall["n_envs"] < len(ENVS), "fraction_of_baseline"] = np.nan
    overall.loc[overall["n_envs"] < len(ENVS), "log_score"] = np.nan

    out_dir = args.out_dir or (SCRIPT_DIR / "topsis_out" / f"sweep_{args.sweep_id}" / "local")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out_dir / "per_slot_metrics.csv", index=False)
    per_env.to_csv(out_dir / "per_config_env_metrics.csv", index=False)
    config_index.to_csv(out_dir / "config_index.csv", index=False)
    overall.to_csv(out_dir / "overall_scores.csv", index=False)

    n_full = int((cov >= len(ENVS)).sum())
    print(f"\nWrote outputs to {out_dir}")
    print(f"  per_config_env rows: {len(per_env)}")
    print(f"  config tags: {len(config_index)} "
          f"(DDPM_mean={int((config_index['denoising_predictor']=='DDPM_mean').sum())}, "
          f"Identity={int((config_index['denoising_predictor']=='Identity').sum())})")
    print(f"  config tags with all {len(ENVS)} envs (overall plottable): {n_full}")


if __name__ == "__main__":
    main()
