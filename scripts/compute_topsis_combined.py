#!/usr/bin/env python3
"""Compute a single TOPSIS ranking over the FINISHED config tags pooled across
several sweeps, using the locally-reconstructed metrics (no wandb).

It reuses the exact scoring primitives from compute_topsis.py (benchmark/log
score, per-env quantile reward, the TOPSIS distance formula, H1 smoothing, and
the seed bootstrap). The only difference is the data source: the per-slot tail
metrics come from build_sweep_local_metrics.py's ``per_slot_metrics.csv`` for
each sweep, filtered to finished runs, rather than being fetched from wandb.

Because TOPSIS and the QL/QU quantile thresholds are computed over the pooled
population, combining sweeps genuinely re-normalizes the ranking -- this is the
combined ranking, not a concatenation of per-sweep rankings.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from compute_topsis import (  # noqa: E402
    ENVS,
    QL,
    QU,
    add_h1_smoothed_log_scores,
    build_per_env_metrics,
    load_benchmark_scores,
    run_bootstrap,
)

TOPSIS_OUT_ROOT = SCRIPT_DIR / "topsis_out"
HP_COLS = ["denoising_predictor", "guidance_gradient_space",
           "guidance_strength_multiplier", "T", "eta"]


def _load_cell_df(sweeps):
    frames = []
    for sw in sweeps:
        p = TOPSIS_OUT_ROOT / f"sweep_{sw}" / "local" / "per_slot_metrics.csv"
        if not p.exists():
            raise FileNotFoundError(f"missing {p}; run build_sweep_local_metrics.py --sweep-id {sw}")
        df = pd.read_csv(p)
        df["sweep_id"] = sw
        frames.append(df)
    runs = pd.concat(frames, ignore_index=True)
    runs = runs[runs["finished"] == True].copy()  # noqa: E712
    runs = runs.rename(columns={"seed_index": "seed"})
    return runs


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweeps", type=int, nargs="+", default=[89, 90, 91, 92])
    ap.add_argument("--out-dir", type=Path, default=TOPSIS_OUT_ROOT / "combined")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_cell_df(args.sweeps)
    print(f"Finished slot metrics pooled across sweeps {args.sweeps}: {len(runs)} rows, "
          f"{runs['config_tag'].nunique()} config tags.")

    benchmark_scores = load_benchmark_scores(ENVS)

    # Config table + hp columns for H1 smoothing / provenance.
    configs_df = runs.drop_duplicates(subset=["config_tag"])[
        ["config_tag", "sweep_id"] + HP_COLS
    ].reset_index(drop=True)

    # One row per (env, config, seed); prefer the longest history on collisions.
    cell_df = (
        runs.sort_values("n_episodes", ascending=False)
        .drop_duplicates(subset=["env", "config_tag", "seed"], keep="first")
        .copy()
    )

    per_env = build_per_env_metrics(cell_df, benchmark_scores)
    per_env = add_h1_smoothed_log_scores(per_env, configs_df, HP_COLS, ENVS)

    # Rankable = full coverage over every env (i.e. finished in all 6 envs).
    pivot = per_env.pivot(index="config_tag", columns="env", values="mean")
    for env in ENVS:
        if env not in pivot.columns:
            pivot[env] = np.nan
    pivot = pivot[ENVS].dropna()
    if pivot.empty:
        sys.exit("No config tags finished in all 6 envs; nothing to rank.")
    rankable_keys = set(pivot.index)
    print(f"Rankable config tags (finished in all {len(ENVS)} envs): {len(rankable_keys)}")

    # Per-env QL/QU thresholds across the pooled runs -> quantile reward.
    ql_percentile = cell_df.groupby("env")["metric"].quantile(QL)
    qu_percentile = cell_df.groupby("env")["metric"].quantile(QU)

    def quantile_reward(raw, env):
        low, high = ql_percentile[env], qu_percentile[env]
        if high == low:
            return 0.0
        return (raw - low) / (high - low)

    cell_df["quantile_reward"] = [
        quantile_reward(r, e) for r, e in zip(cell_df["metric"], cell_df["env"])
    ]
    q_per_cell = cell_df.pivot_table(
        index="config_tag", columns="env", values="quantile_reward", aggfunc="mean"
    )
    q_per_cell = q_per_cell.loc[[k for k in q_per_cell.index if k in rankable_keys], ENVS]

    joint_quantile = (
        cell_df[cell_df["config_tag"].isin(rankable_keys)]
        .groupby("config_tag")["quantile_reward"].mean().rename("quantile_score")
    )

    # TOPSIS on [0,1]-shifted per-(config,env) quantile values (uniform weights).
    D01 = (q_per_cell + 1.0) / 2.0
    w = np.ones(len(ENVS)) / len(ENVS)
    Dplus = np.sqrt((((1.0 - D01) ** 2) * w).sum(axis=1))
    Dminus = np.sqrt(((D01 ** 2) * w).sum(axis=1))
    topsis_score = (Dminus / (Dplus + Dminus)).rename("topsis_score")

    log_avg = (
        per_env[per_env["config_tag"].isin(rankable_keys)]
        .groupby("config_tag")["log_score"].mean().rename("log_score")
    )
    h1_avg = (
        per_env[per_env["config_tag"].isin(rankable_keys)]
        .groupby("config_tag")["h1_smoothed_log_score"].mean().rename("h1_smoothed_log_score")
    )

    ranking = (
        pivot.join(topsis_score).join(joint_quantile).join(log_avg).join(h1_avg)
        .sort_values("topsis_score", ascending=False)
        .reset_index()
    )
    ranking = ranking.merge(configs_df[["config_tag", "sweep_id"]], on="config_tag", how="left")

    print(f"\nRunning {args.n_boot} bootstrap iterations over seeds within each (config, env) ...")
    boot_df = run_bootstrap(cell_df, rankable_keys, ENVS, benchmark_scores, n_iter=args.n_boot)
    ranking = ranking.merge(boot_df, on="config_tag", how="left")

    env_round = {e: 1 for e in ENVS}
    score_round = {c: 4 for c in ranking.columns
                   if c not in ENVS + ["config_tag", "sweep_id"] and ranking[c].dtype.kind == "f"}
    ranking = ranking.round({**env_round, **score_round})

    out_path = out_dir / "topsis_ranking_combined.csv"
    ranking.to_csv(out_path, index=False)
    per_env.to_csv(out_dir / "per_config_env_metrics_combined.csv", index=False)
    configs_df.to_csv(out_dir / "combined_configs.csv", index=False)

    cols = ["config_tag", "sweep_id", "topsis_score", "quantile_score", "log_score",
            "topsis_score_ci_lower", "topsis_score_ci_upper", "topsis_p_best"]
    cols = [c for c in cols if c in ranking.columns]
    print(f"\nWrote {out_path}")
    print(f"\n=== Top 15 by TOPSIS (pooled sweeps {args.sweeps}) ===")
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(ranking[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
