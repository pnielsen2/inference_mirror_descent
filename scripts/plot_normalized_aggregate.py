#!/usr/bin/env python3
"""One curve per algorithm: mean normalised score over all 60 runs.

Each environment gets a single scale factor so the six of them can be averaged
together. The factor is the best seed-averaged checkpoint anyone reached in that
environment:

    normaliser(env) = max over (algorithm, step) of
                      mean over seeds of (mean over episodes at that checkpoint)

and every run's checkpoint is divided by it. The map is linear through the
origin, so 0 stays 0, the best point of the best algorithm sits at 1, and early
checkpoints with negative return stay negative rather than being clipped.

The maximum is taken over algorithms as well as steps -- one shared scale per
environment. Normalising each algorithm by its own peak would send all of them
to 1 and destroy the comparison.

With the scale fixed, the 6 envs x 10 seeds = 60 runs of an algorithm are pooled
and averaged pointwise, giving one curve. No smoothing: each point is a single
checkpoint's 10-episode average.

The band is a 50% bootstrap CI for that mean, resampled over runs and stratified
by environment (10 seeds drawn with replacement within each env), which keeps
every env's contribution at 10 runs the way the design intends. A bootstrap is
used rather than the order-statistic interval of the per-env figures because the
statistic here is a mean, not a median.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from plot_eval_method_comparison import (  # noqa: E402
    BASELINE_DIR,
    ENVS,
    PRESETS,
    load_series,
)
# One definition of the seed x step matrix, shared with the per-env figures.
from plot_eval_method_comparison import checkpoint_matrix as per_checkpoint  # noqa: E402


def run_curves(df: pd.DataFrame, max_step: int, window: int) -> pd.DataFrame:
    """seed x step matrix, optionally smoothed along each run's own curve.

    ``window <= 1`` leaves the raw checkpoints alone. Otherwise a trailing mean
    over ``window`` consecutive checkpoints is taken per run, requiring a full
    window so every surviving point averages the same number of checkpoints.
    """
    m = per_checkpoint(df, max_step)
    if window > 1:
        m = m.T.rolling(window, min_periods=window).mean().T
    return m


def env_normaliser(mats: list[pd.DataFrame], mode: str) -> float:
    """Upper end of the scale for one environment, over every algorithm.

    ``seed_mean`` -- best checkpoint of the seed-averaged curve, so the scale is
    a level a whole algorithm reached on average.
    ``run_max``   -- best checkpoint of any individual run, so the scale is the
    single best thing observed. Strictly the larger of the two.
    """
    peaks = []
    for m in mats:
        a = m.to_numpy()
        # A trailing average leaves leading columns all-NaN; drop them so the
        # column mean is never taken over an empty slice.
        a = a[:, np.isfinite(a).any(axis=0)]
        if a.size == 0:
            continue
        peaks.append(np.nanmax(np.nanmean(a, axis=0) if mode == "seed_mean" else a))
    return float(np.nanmax(peaks))


def bootstrap_ci(runs: np.ndarray, env_ids: np.ndarray, level: float,
                 n_boot: int, rng: np.random.Generator,
                 chunk: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Percentile CI for the mean of ``runs`` (n_runs x n_steps).

    Resampling is stratified: within each env, its rows are drawn with
    replacement, so each bootstrap sample has the same env balance as the data.
    """
    groups = [np.flatnonzero(env_ids == e) for e in np.unique(env_ids)]
    lo_q, hi_q = 50.0 * (1.0 - level), 50.0 * (1.0 + level)

    means = np.empty((n_boot, runs.shape[1]), dtype=float)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        # Within each env draw its own rows with replacement, then glue the
        # groups back together so every resample has the original env balance.
        picks = np.concatenate(
            [g[rng.integers(0, g.size, size=(b, g.size))] for g in groups],
            axis=1,
        )
        means[done:done + b] = np.nanmean(runs[picks], axis=1)
        done += b
    return (np.nanpercentile(means, lo_q, axis=0),
            np.nanpercentile(means, hi_q, axis=0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    ap.add_argument("--preset", choices=tuple(PRESETS), default="simple")
    ap.add_argument("--max-step", type=int, default=1_000_000)
    ap.add_argument("--window", type=int, default=0,
                    help="checkpoints in the per-run trailing average "
                         "(0 or 1 = no smoothing, the default)")
    ap.add_argument("--normalizer", choices=("seed_mean", "run_max"),
                    default="seed_mean",
                    help="'seed_mean': scale by the best seed-averaged "
                         "checkpoint; 'run_max': by the best individual run")
    ap.add_argument("--ci-level", type=float, default=0.50)
    ap.add_argument("--n-boot", type=int, default=20_000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--title", action="store_true", help="add a figure title")
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to a name derived from the options")
    args = ap.parse_args()

    series = PRESETS[args.preset]
    if args.out is None:
        stem = "normalized_score_aggregate"
        if args.window > 1:
            stem += f"_w{args.window}"
        if args.normalizer != "seed_mean":
            stem += f"_{args.normalizer}"
        args.out = REPO / "figures" / f"{stem}.pdf"
    # style.use resets rcParams, so the font setting has to follow it. Type 42
    # keeps text as text in the PDF (Type 3, matplotlib's default, is rejected
    # by several venues and is not editable downstream).
    plt.style.use("default")
    matplotlib.rcParams["pdf.fonttype"] = 42

    # per_ckpt[label][env] = seed x step frame
    per_ckpt: dict[str, dict[str, pd.DataFrame]] = {}
    for algorithm, variant, method, label, *_ in series:
        loaded = load_series(algorithm, variant, method, args.baseline_dir)
        per_ckpt[label] = {env: run_curves(df, args.max_step, args.window)
                           for env, df in loaded.items()}

    normaliser: dict[str, float] = {}
    for env in ENVS:
        mats = [m[env] for m in per_ckpt.values() if env in m]
        if mats:
            normaliser[env] = env_normaliser(mats, args.normalizer)
    label_txt = ("max seed-averaged checkpoint" if args.normalizer == "seed_mean"
                 else "max over individual runs")
    smooth_txt = (f", {args.window}-checkpoint trailing average"
                  if args.window > 1 else ", no smoothing")
    print(f"Per-env normaliser ({label_txt}, any algorithm{smooth_txt}):")
    for env, v in normaliser.items():
        owner = max(
            ((lbl, env_normaliser([m[env]], args.normalizer))
             for lbl, m in per_ckpt.items() if env in m),
            key=lambda t: t[1])[0]
        print(f"  {env:16s} {v:10.2f}   ({owner})")

    steps = None
    rng = np.random.default_rng(args.boot_seed)
    rows = []
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for algorithm, variant, method, label, colour, _linestyle, z in series:
        stacked, env_ids = [], []
        for env in ENVS:
            if env not in per_ckpt[label] or env not in normaliser:
                continue
            m = per_ckpt[label][env]
            if steps is None:
                steps = m.columns.to_numpy()
            m = m.reindex(columns=steps)
            stacked.append(m.to_numpy() / normaliser[env])
            env_ids.extend([env] * m.shape[0])
        if not stacked:
            continue
        runs = np.vstack(stacked)
        env_ids = np.asarray(env_ids)
        # A trailing average leaves the first window-1 checkpoints undefined for
        # every run; drop those columns rather than feed all-NaN slices to the
        # bootstrap.
        keep = np.isfinite(runs).any(axis=0)
        runs, kept_steps = runs[:, keep], steps[keep]
        mean = np.nanmean(runs, axis=0)
        lo, hi = bootstrap_ci(runs, env_ids, args.ci_level, args.n_boot, rng)

        print(f"  {label:14s} runs={runs.shape[0]:3d}  final={mean[-1]:.3f} "
              f"[{lo[-1]:.3f}, {hi[-1]:.3f}]")

        x = kept_steps / 1e6
        ax.plot(x, mean, color=colour, label=label,
                linewidth=2.2 if colour == "black" else 1.6, zorder=z)
        ax.fill_between(x, lo, hi, color=colour, alpha=0.15, linewidth=0,
                        zorder=z - 1)
        rows.append(pd.DataFrame({
            "algorithm": algorithm, "variant": variant, "eval_method": method,
            "series": label, "step": kept_steps, "n_runs": runs.shape[0],
            "mean_normalized_score": mean, "ci_lo": lo, "ci_hi": hi,
            "ci_level": args.ci_level, "window": args.window,
            "normalizer_mode": args.normalizer,
        }))

    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Normalized score (mean over 60 runs)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    if args.title:
        ax.set_title(
            f"Mean normalized score over 6 envs x 10 seeds\n"
            f"band: {100 * args.ci_level:.0f}% bootstrap CI for the mean",
            fontsize=11)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}")

    if rows:
        out_csv = args.out.with_suffix(".csv")
        pd.concat(rows, ignore_index=True).to_csv(out_csv, index=False)
        pd.DataFrame(
            [{"env": e, "normalizer": v} for e, v in normaliser.items()]
        ).to_csv(args.out.with_name(args.out.stem + "_normalizers.csv"), index=False)
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
