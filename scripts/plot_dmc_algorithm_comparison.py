#!/usr/bin/env python3
"""Eval-return curves for MGMD, DPMD and SAC on the dm_control tasks.

One panel per env, one line per algorithm (bold = mean over seeds, thin = the
individual seeds). Seeds are drawn individually on purpose: on ``dog-run`` the
two seeds of both MGMD and SAC straddle a critic divergence, so a seed-mean
alone hides the bimodality.

THE THREE PROTOCOLS ARE NOT IDENTICAL -- read the legend before comparing:

  MGMD  best-of-32 under mean(Q1,Q2), 10 episodes every 25k steps
        <codebase>/logs/<env>/<run>/eval_episode_returns.csv
  SAC   best-of-32 under mean(Q1,Q2), 10 episodes every 10k steps
        <root>/runs/<env>_s<seed>_<stamp>/eval_episode_returns.csv
  DPMD  DETERMINISTIC (the repo's native protocol), 20 episodes every 50k steps
        <root>/logs/<env>/<run>/log.csv

MGMD and SAC are therefore directly comparable; DPMD's line is a deterministic
policy and carries no best-of-N argmax, which on these tasks is a handicap of
unknown size rather than a fixed offset.

Usage
-----
    python scripts/plot_dmc_algorithm_comparison.py
    python scripts/plot_dmc_algorithm_comparison.py --out figures/dmc/compare.png
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent

SWEEP_ROOT = Path("/n/holylabs/kdbrantley_lab/Lab/pnielsen/wandb")
DPMD_ROOT = Path("/n/netscratch/kdbrantley_lab/Lab/pnielsen/dpmd_v3_baseline/logs")
SAC_ROOT = Path("/n/netscratch/kdbrantley_lab/Lab/pnielsen/sac_dmc_baseline/runs")

ENVS = ["dmc/humanoid-walk-v0", "dmc/humanoid_CMU-run-v0", "dmc/dog-run-v0"]

STYLE = {
    "MGMD": dict(color="#1f77b4"),
    "DPMD": dict(color="#d62728"),
    "SAC": dict(color="#2ca02c"),
}
LABEL = {
    "MGMD": "MGMD (best-of-32, 10 ep / 25k)",
    "DPMD": "DPMD (deterministic, 20 ep / 50k)",
    "SAC": "SAC (best-of-32, 10 ep / 10k)",
}


def _mean_by_step(pairs: dict[int, list[float]]) -> tuple[np.ndarray, np.ndarray]:
    steps = np.array(sorted(pairs))
    vals = np.array([float(np.mean(pairs[s])) for s in steps])
    return steps, vals


def load_mgmd(sweep_id: int) -> dict[str, dict[str, tuple]]:
    """env -> {seed_label: (steps, returns)} from the sweep's codebase mirror.

    The codebase that produced a run is recovered from its wandb-metadata
    ``program`` field, per AGENTS.md; every slot of a job shares one CSV and is
    told apart by the ``seed_index`` column.
    """
    out: dict[str, dict[str, tuple]] = defaultdict(dict)
    sweep = SWEEP_ROOT / f"sweep_{sweep_id}"
    if not sweep.is_dir():
        return out
    seen: set[Path] = set()
    for run in sorted(sweep.glob("job_*/wandb/offline-run-*")):
        cfg_path = run / "files" / "config.yaml"
        md_path = run / "files" / "wandb-metadata.json"
        if not cfg_path.exists() or not md_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        env = (cfg.get("env") or {}).get("value")
        program = json.loads(md_path.read_text()).get("program")
        if not env or not program:
            continue
        codebase = Path(program).resolve().parent.parent
        for csv_path in sorted((codebase / "logs" / env).glob("*/eval_episode_returns.csv")):
            if csv_path in seen:
                continue
            seen.add(csv_path)
            per_seed: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
            with csv_path.open() as f:
                for row in csv.DictReader(f):
                    per_seed[row["seed_index"]][int(row["step"])].append(float(row["episode_return"]))
            for slot, pairs in per_seed.items():
                out[env][f"slot{slot}"] = _mean_by_step(pairs)
    return out


def load_dpmd() -> dict[str, dict[str, tuple]]:
    """env -> {seed_label: (steps, returns)} from each run's native log.csv."""
    out: dict[str, dict[str, tuple]] = defaultdict(dict)
    for log in sorted(DPMD_ROOT.glob("dmc/*/dpmd_*dmc*/log.csv")):
        env = f"dmc/{log.parent.parent.name}"
        seed = "s?"
        for part in log.parent.name.split("_"):
            if part.startswith("s") and part[1:].isdigit():
                seed = part
        steps, rets = [], []
        with log.open() as f:
            for row in csv.DictReader(f):
                steps.append(int(row["step"]))
                rets.append(float(row["avg_ret"]))
        if steps:
            out[env][seed] = (np.array(steps), np.array(rets))
    return out


def load_sac() -> dict[str, dict[str, tuple]]:
    """env -> {seed_label: (steps, returns)}; dir name encodes env and seed."""
    out: dict[str, dict[str, tuple]] = defaultdict(dict)
    for run in sorted(SAC_ROOT.glob("dmc_*_s*")):
        csv_path = run / "eval_episode_returns.csv"
        if not csv_path.exists():
            continue
        name = run.name
        # dmc_<task>_s<seed>_<stamp>  ->  task may itself contain '_' and '-'
        parts = name.split("_")
        seed_i = max(i for i, p in enumerate(parts) if p.startswith("s") and p[1:].isdigit())
        task = "_".join(parts[1:seed_i])
        env = f"dmc/{task}-v0"
        seed = parts[seed_i]
        pairs: dict[int, list[float]] = defaultdict(list)
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                pairs[int(row["env_step"])].append(float(row["episode_return"]))
        if pairs:
            out[env][seed] = _mean_by_step(pairs)
    return out


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling mean that shrinks at the edges, NaN-safe.

    Each eval point is only 10 (MGMD/SAC) or 20 (DPMD) episodes, so the raw
    curves are dominated by episode noise; the thin per-seed lines stay raw.
    """
    if window <= 1:
        return y
    out = np.full_like(y, np.nan, dtype=float)
    half = window // 2
    for i in range(len(y)):
        seg = y[max(0, i - half): i + half + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


def _seed_mean(series: dict[str, tuple]) -> tuple[np.ndarray, np.ndarray]:
    """Mean over seeds on the union step grid.

    Each seed is interpolated only inside its own extent (NaN outside), so a
    run that has not caught up yet cannot drag the mean down.
    """
    grid = np.unique(np.concatenate([s for s, _ in series.values()]))
    stack = [np.interp(grid, s, v, left=np.nan, right=np.nan) for s, v in series.values()]
    return grid, np.nanmean(np.vstack(stack), axis=0)


def _plot_algo(ax, series: dict[str, tuple], algo: str, window: int) -> int:
    """Thin per-seed lines plus a bold smoothed mean on the shared step grid."""
    if not series:
        return 0
    style = STYLE[algo]
    for steps, vals in series.values():
        ax.plot(steps, vals, lw=0.7, alpha=0.30, **style)
    grid, mean = _seed_mean(series)
    mean = _smooth(mean, window)
    ok = ~np.isnan(mean)
    ax.plot(grid[ok], mean[ok], lw=2.2,
            label=f"{LABEL[algo]}  [n={len(series)}]", **style)
    return len(series)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep-id", type=int, default=242)
    p.add_argument("--smooth", type=int, default=5,
                   help="Rolling-mean window (in eval points) for the seed-mean line.")
    p.add_argument("--out", type=Path,
                   default=REPO / "figures" / "dmc_baselines" / "eval_return_comparison.png")
    args = p.parse_args()

    data = {"MGMD": load_mgmd(args.sweep_id), "DPMD": load_dpmd(), "SAC": load_sac()}

    fig, axes = plt.subplots(1, len(ENVS), figsize=(16.5, 4.9))
    axes = np.atleast_1d(axes)
    for col, env in enumerate(ENVS):
        ax = axes[col]
        finals = []
        for algo in ("MGMD", "DPMD", "SAC"):
            series = data[algo].get(env, {})
            n = _plot_algo(ax, series, algo, args.smooth)
            if n:
                last = max(s.max() for s, _ in series.values())
                grid, mean = _seed_mean(series)
                tail = mean[grid >= 0.8 * grid.max()]
                finals.append(f"{algo}: {np.nanmean(tail):.1f}")
                print(f"  {env:<24} {algo:<5} n={n} up to step {last:,}  "
                      f"last-20% mean={np.nanmean(tail):.2f}")
        ax.set_title(env, fontsize=11)
        ax.annotate("final 20% mean -- " + "   ".join(finals),
                    xy=(0.02, 0.02), xycoords="axes fraction", fontsize=7.5,
                    va="bottom", color="#333333")
        ax.set_xlabel("env steps")
        ax.grid(alpha=0.25, lw=0.5)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    axes[0].set_ylabel("eval episode return")
    axes[0].legend(fontsize=7.5, loc="upper left", framealpha=0.9)

    fig.suptitle(
        "dm_control eval returns (task max = 1000). Bold = seed mean (rolling-"
        f"{args.smooth} smoothed), thin = individual seeds, raw. "
        "MGMD and SAC are best-of-32 under mean(Q1,Q2); DPMD is deterministic.",
        fontsize=9.5, y=1.005,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
