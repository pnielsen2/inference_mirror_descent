#!/usr/bin/env python3
"""MGMD against the baselines, one curve per (algorithm, eval method), per env.

Reads the unified tables written by ``build_baseline_index.py`` and reduces each
run to a trailing average before comparing, in three steps:

1. Within a checkpoint, average the evaluation episodes -> one value per
   (seed, checkpoint).
2. Along a seed's own curve, take a trailing mean over ``--window`` consecutive
   checkpoints. The window is required to be full, so every plotted point is the
   average of exactly that many checkpoints and the curves all begin at the same
   env step rather than at a partially-filled average.
3. Across seeds, plot the median together with a distribution-free confidence
   interval for it, built from the order statistics of the seed values.

The interval comes from the binomial identity
``P(X_(i) <= median <= X_(j)) = sum_{k=i}^{j-1} C(n,k) / 2^n``, which holds for
any continuous distribution. With ten seeds the attainable symmetric coverages
are 24.6% (ranks 5-6) and 65.6% (ranks 4-7), so a nominal 50% cannot be hit
exactly; the narrowest interval that still *covers* the nominal level is used and
its achieved coverage is printed and recorded in the output CSV.

Only the 1M-step horizon common to every method is plotted (see --max-step);
DPMD's Humanoid runs continue to 2M and would otherwise be compared against
other methods' final performance at half the interaction budget.
"""
from __future__ import annotations

import argparse
from itertools import combinations
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
BASELINE_DIR = REPO / "baselines"
WANDB_ROOT = Path("/n/holylabs/kdbrantley_lab/Lab/pnielsen/wandb")

ENVS = ["Ant-v3", "HalfCheetah-v3", "Hopper-v3", "Humanoid-v3", "Swimmer-v3", "Walker2d-v3"]
ACTION_NOISE_ENVS = ["Ant-v3", "HalfCheetah-v3", "Humanoid-v3", "Walker2d-v3"]

POOLED_MGMD_VARIANTS = {
    "sweeps265_267_action_noise1_reweight": (
        (265, "sweep265_action_noise=1_buffer_size=250000_candidate_selection=reweight"),
        (267, "sweep267_action_noise=1_buffer_size=250000_candidate_selection=reweight"),
    ),
    "sweeps265_267_action_noise1_boltzmann": (
        (265, "sweep265_action_noise=1_buffer_size=250000_candidate_selection=boltzmann"),
        (267, "sweep267_action_noise=1_buffer_size=250000_candidate_selection=boltzmann"),
    ),
    "sweeps265_267_action_noise1_best": (
        (265, "sweep265_action_noise=1_buffer_size=250000_candidate_selection=best"),
        (267, "sweep267_action_noise=1_buffer_size=250000_candidate_selection=best"),
    ),
}

# (algorithm, variant, eval_method, legend label, colour, linestyle, zorder)
#
# Colour identifies the algorithm and linestyle the eval method, because the two
# DIPO curves land within ~17 return of each other once aggregated (they are
# genuinely different measurements -- per episode they differ by ~255 on average
# -- but the critic's pick and the deterministic action agree in aggregate). A
# shared colour with a dashed overlay keeps both visible where they coincide.
#
# MGMD is ours, drawn black and on top. DPMD's trainN1_no_noise variant is
# deliberately absent: it stays in baselines/ but 'original' is the canonical
# DPMD baseline.
SERIES_FULL = [
    ("MGMD", "sweep244_rollout_alpha1", "best_of_32", "MGMD (ours) — best-of-32", "black", "-", 5),
    ("DIPO", "default", "best_of_32", "DIPO — best-of-32", "C0", "-", 2),
    ("DIPO", "default", "mean_action", "DIPO — mean action", "C0", "--", 2),
    ("DPMD", "original", "best_of_32", "DPMD — best-of-32", "C1", "-", 2),
    ("SAC", "default", "best_of_32", "SAC — best-of-32", "C2", "-", 2),
    ("SAC", "default", "mean_action", "SAC — mean action", "C2", "--", 2),
]

# One curve per algorithm: each baseline is shown under the eval method it is
# normally reported with, so the eval method no longer needs to be in the label
# and every line can be solid.
SERIES_SIMPLE = [
    ("MGMD", "sweep244_rollout_alpha1", "best_of_32", "MGMD (ours)", "black", "-", 5),
    ("DIPO", "default", "mean_action", "DIPO", "C0", "-", 2),
    ("DPMD", "original", "best_of_32", "DPMD", "C1", "-", 2),
    ("SAC", "default", "mean_action", "SAC", "C2", "-", 2),
]

# Ablation: the annealing path shortened, everything else identical (sweeps 247
# and 249 are sweep 244's rollout_alpha=1 arm at --diffusion_steps 50 and 25).
# Labelled by the quantity being varied rather than by algorithm, since every
# curve is MGMD.
SERIES_DIFFUSION_STEPS = [
    ("MGMD", "sweep244_rollout_alpha1", "best_of_32", "100 diffusion steps", "black", "-", 5),
    ("MGMD", "sweep247_diffusion50", "best_of_32", "50 diffusion steps", "C3", "-", 4),
    ("MGMD", "sweep249_diffusion25", "best_of_32", "25 diffusion steps", "C0", "-", 3),
]

SERIES_ACTION_NOISE_SELECTION = [
    ("MGMD", "sweep247_diffusion50", "best_of_32", "50 diffusion steps", "black", "-", 5),
    ("MGMD", "sweeps265_267_action_noise1_reweight", "best_of_32", "Reweight", "C0", "-", 4),
    ("MGMD", "sweeps265_267_action_noise1_boltzmann", "best_of_32", "Boltzmann", "C1", "-", 3),
    ("MGMD", "sweeps265_267_action_noise1_best", "best_of_32", "Best-of-32", "C2", "-", 2),
]

SERIES_SWEEP261 = [
    ("MGMD", "sweep261_alpha0", "best_of_32", "Sweep 261 (α=0; n=2)", "black", "-", 5),
    ("MGMD", "sweep247_diffusion50", "best_of_32", "50-step reference (α=1; n=10)", "C3", "-", 4),
]

SERIES_SWEEP262_BETA128 = [
    ("MGMD", "sweep262_alpha0_beta128", "best_of_32", "Sweep 262 (α=0, β=128; n=2)", "black", "-", 5),
    ("MGMD", "sweep247_diffusion50", "best_of_32", "50-step reference (α=1, β=64; n=10)", "C3", "-", 4),
]

# SAC's critic depth: its published two-hidden-layer critic against the same
# configuration re-run at the three-layer critic the diffusion methods use. Both
# SAC curves are its own deterministic action, as in SERIES_SIMPLE, so the only
# thing that differs between them is the critic.
SERIES_SAC_CRITIC = [
    ("MGMD", "sweep244_rollout_alpha1", "best_of_32", "MGMD (ours)", "black", "-", 5),
    ("DIPO", "default", "mean_action", "DIPO", "C0", "-", 2),
    ("DPMD", "original", "best_of_32", "DPMD", "C1", "-", 2),
    ("SAC", "default", "mean_action", "SAC (2-layer critic)", "C2", "-", 2),
    ("SAC", "critic3", "mean_action", "SAC (3-layer critic)", "C4", "-", 3),
]

PRESETS = {
    "full": SERIES_FULL,
    "sac_critic": SERIES_SAC_CRITIC,
    "simple": SERIES_SIMPLE,
    "diffusion_steps": SERIES_DIFFUSION_STEPS,
    "action_noise_selection": SERIES_ACTION_NOISE_SELECTION,
    "sweep261": SERIES_SWEEP261,
    "sweep262_beta128": SERIES_SWEEP262_BETA128,
}

PRESET_ENVS = {
    "action_noise_selection": ACTION_NOISE_ENVS,
}

# Output stem per preset; anything not listed falls back to "algorithm_comparison".
PRESET_STEMS = {
    "full": "eval_method_comparison",
    "simple": "algorithm_comparison",
    "diffusion_steps": "diffusion_steps_ablation",
    "action_noise_selection": "action_noise1_candidate_selection_vs_diffusion50",
    "sac_critic": "sac_critic_depth",
    "sweep261": "sweep261_vs_diffusion50",
    "sweep262_beta128": "sweep262_beta128_vs_diffusion50",
}


def median_ci_ranks(n: int, level: float) -> tuple[int, int, float]:
    """Narrowest order-statistic interval for the median with coverage >= level.

    Returns 0-based indices into the sorted sample plus the exact coverage.
    Ties on width are broken toward the interval centred on the median.
    """
    total = 2.0 ** n
    best = None
    for i, j in combinations(range(1, n + 2), 2):  # 1-based ranks, i < j
        if j > n:
            continue
        coverage = sum(comb(n, k) for k in range(i, j)) / total
        if coverage < level:
            continue
        width = j - i
        asymmetry = abs((i - 1) - (n - j))
        key = (width, asymmetry, coverage)
        if best is None or key < best[0]:
            best = (key, i - 1, j - 1, coverage)
    if best is None:  # pragma: no cover - only if level is unattainable
        return 0, n - 1, 1.0
    return best[1], best[2], best[3]


def load_series(algorithm: str, variant: str, eval_method: str,
                baseline_dir: Path) -> dict[str, pd.DataFrame]:
    """env -> tidy frame of (seed, step, episode_return) for one series."""
    if algorithm == "MGMD" and variant in POOLED_MGMD_VARIANTS:
        return load_pooled_mgmd_series(variant, eval_method)

    out = {}
    for env in ENVS:
        path = baseline_dir / algorithm / f"{algorithm}__{env}__eval_episodes.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, usecols=["variant", "eval_method", "seed",
                                        "step", "episode_return"])
        df = df[(df["variant"] == variant) & (df["eval_method"] == eval_method)]
        if not df.empty:
            out[env] = df
    return out


def load_pooled_mgmd_series(variant: str, eval_method: str) -> dict[str, pd.DataFrame]:
    """Pool local eval mirrors while keeping same-numbered cross-sweep runs distinct."""
    from build_baseline_index import _load_mgmd_one, _mgmd_jobs

    per_env: dict[str, list[pd.DataFrame]] = {env: [] for env in ACTION_NOISE_ENVS}
    for sweep_id, config_tag in POOLED_MGMD_VARIANTS[variant]:
        jobs = _mgmd_jobs(WANDB_ROOT / f"sweep_{sweep_id}", config_tag)
        for env in ACTION_NOISE_ENVS:
            frame = _load_mgmd_one(env, f"sweep{sweep_id}", jobs)
            if not frame.empty:
                per_env[env].append(frame)

    out = {}
    for env, frames in per_env.items():
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df = df[df["eval_method"] == eval_method].copy()
        if df.empty:
            continue
        df["seed"] = df.groupby(["variant", "seed"], sort=True).ngroup()
        df["variant"] = variant
        out[env] = df
    return out


def checkpoint_matrix(df: pd.DataFrame, max_step: int) -> pd.DataFrame:
    """seed x step matrix of each checkpoint's episode-average return."""
    df = df[df["step"] <= max_step]
    return (df.groupby(["seed", "step"])["episode_return"]
            .mean().unstack("step").sort_index(axis=1))


def trailing_curves(df: pd.DataFrame, window: int, max_step: int) -> pd.DataFrame:
    """seed x step matrix of trailing-mean returns, clipped to max_step first.

    ``window <= 1`` returns the raw checkpoints untouched.
    """
    per_ckpt = checkpoint_matrix(df, max_step)
    if window <= 1:
        return per_ckpt
    # Trailing mean along each seed's own checkpoint sequence; min_periods=window
    # keeps partially-filled windows out, so every point averages the same count.
    # Rolled on the transpose (steps down the index) since DataFrame.rolling no
    # longer accepts axis=1.
    return per_ckpt.T.rolling(window, min_periods=window).mean().T


def trailing_mean(a: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean along the last axis, NaN until the window is full."""
    a = np.asarray(a, dtype=float)
    if window <= 1:
        return a.copy()
    out = np.full(a.shape, np.nan)
    if a.shape[-1] >= window:
        sw = np.lib.stride_tricks.sliding_window_view(a, window, axis=-1)
        out[..., window - 1:] = sw.mean(axis=-1)
    return out


def _smoothed_median_partial(raw: np.ndarray, steps: np.ndarray, window: int,
                             level: float, n_boot: int,
                             rng: np.random.Generator, min_seeds: int,
                             chunk: int) -> pd.DataFrame:
    groups: dict[tuple[tuple[int, ...], ...], list[int]] = {}
    for end in range(window - 1, raw.shape[1]):
        availability = tuple(
            tuple(np.flatnonzero(np.isfinite(raw[:, step])).tolist())
            for step in range(end - window + 1, end + 1)
        )
        if all(len(run_ids) >= min_seeds for run_ids in availability):
            groups.setdefault(availability, []).append(end)

    rows = []
    centre_curve = trailing_mean(np.nanmedian(raw, axis=0), window)
    lo_q, hi_q = 50.0 * (1.0 - level), 50.0 * (1.0 + level)
    for availability, ends in groups.items():
        reps = np.empty((n_boot, len(ends)), dtype=float)
        done = 0
        while done < n_boot:
            b = min(chunk, n_boot - done)
            sampled_runs = {
                run_ids: np.asarray(run_ids)[
                    rng.integers(0, len(run_ids), size=(b, len(run_ids)))
                ]
                for run_ids in dict.fromkeys(availability)
            }
            smoothed = np.zeros((b, len(ends)), dtype=float)
            for offset, run_ids in enumerate(availability):
                draws = sampled_runs[run_ids]
                for col, end in enumerate(ends):
                    step = end - window + 1 + offset
                    smoothed[:, col] += np.median(
                        raw[draws, step], axis=1) / window
            reps[done:done + b] = smoothed
            done += b
        lo, hi = np.percentile(reps, [lo_q, hi_q], axis=0)
        for end, ci_lo, ci_hi in zip(ends, lo, hi):
            rows.append({
                "step": int(steps[end]),
                "statistic": "median",
                "center": float(centre_curve[end]),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "n_seeds": len(availability[-1]),
                "ci_coverage": level,
            })
    if not rows:
        return pd.DataFrame(columns=[
            "step", "statistic", "center", "ci_lo", "ci_hi", "n_seeds",
            "ci_coverage",
        ])
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def smoothed_median(per_ckpt: pd.DataFrame, window: int, level: float,
                    n_boot: int, rng: np.random.Generator,
                    min_seeds: int = 1, chunk: int = 512) -> pd.DataFrame:
    """Trailing mean OF the across-seed median, with a bootstrap CI.

    The other order of operations (``trailing_curves`` then :func:`aggregate`)
    smooths each run and then takes the median of the smoothed runs. Here the
    median is taken first, per checkpoint, and that median curve is smoothed --
    a different statistic, since the median does not commute with a moving
    average.

    Its sampling distribution has no convenient order-statistic form, so it is
    bootstrapped: resample the runs with replacement, redo median-then-smooth on
    each resample, and read the interval off the middle ``level`` of the
    resulting smoothed curves at each step.

    Caveat, not a bug: the band can occasionally exclude ``centre``. The
    bootstrap distribution of a median over an even, small number of runs (10
    here, so the median is the mean of the 5th and 6th order statistics) is
    discrete and can sit off-centre relative to the original sample median, so
    the middle ``level`` of it need not contain that median. On this data it
    happens at ~1% of plotted points and by <=22% of the band width. The
    percentile interval is reported as specified rather than silently swapped for
    a bias-corrected variant; if that matters, BCa or reporting the bootstrap
    median as the centre would remove it.
    """
    raw = per_ckpt.to_numpy(dtype=float)
    n_runs, n_steps = raw.shape
    if not np.isfinite(raw).all():
        return _smoothed_median_partial(
            raw, per_ckpt.columns.to_numpy(), window, level, n_boot, rng,
            min_seeds, chunk)

    centre = trailing_mean(np.nanmedian(raw, axis=0), window)

    curves = np.empty((n_boot, n_steps), dtype=float)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, n_runs, size=(b, n_runs))
        # median over the resampled runs, then smooth each resampled curve
        curves[done:done + b] = trailing_mean(
            np.nanmedian(raw[idx], axis=1), window)
        done += b

    # The trailing average leaves the first window-1 checkpoints undefined for
    # every resample; drop them before taking percentiles so no all-NaN slice is
    # reduced.
    keep = np.isfinite(centre)
    lo_q, hi_q = 50.0 * (1.0 - level), 50.0 * (1.0 + level)
    lo = np.nanpercentile(curves[:, keep], lo_q, axis=0)
    hi = np.nanpercentile(curves[:, keep], hi_q, axis=0)

    return pd.DataFrame({
        "step": per_ckpt.columns.to_numpy()[keep],
        "statistic": "median",
        "center": centre[keep],
        "ci_lo": lo,
        "ci_hi": hi,
        "n_seeds": n_runs,
        "ci_coverage": level,
    })


def aggregate(curves: pd.DataFrame, level: float,
              statistic: str = "median", ci_method: str = "order_statistic",
              n_boot: int = 20_000,
              rng: np.random.Generator | None = None,
              min_seeds: int = 1) -> pd.DataFrame:
    """Across-seed centre and its CI, per step.

    ``curves`` is seed x step, already reduced along each seed's own curve, so
    every column is one number per seed and the CI is over seeds alone.

    Centre: ``median`` or ``mean`` across seeds.
    Interval:
      ``order_statistic`` -- distribution-free, median only; coverage can only
      take the values attainable from the sample size, so the achieved level is
      reported alongside.
      ``bootstrap``       -- percentile interval from resampling seeds.
      ``t``               -- Student-t, mean only; exact for normal seed effects.
    """
    rows = []
    for step in curves.columns:
        vals = curves[step].dropna().to_numpy()
        if vals.size == 0 or vals.size < min_seeds:
            continue
        if ci_method == "bootstrap":
            if rng is None:
                rng = np.random.default_rng(0)
            draws = vals[rng.integers(0, vals.size, size=(n_boot, vals.size))]
            reps = (draws.mean(axis=1) if statistic == "mean"
                    else np.median(draws, axis=1))
            centre = float(vals.mean() if statistic == "mean" else np.median(vals))
            lo, hi = np.percentile(reps, [50 * (1 - level), 50 * (1 + level)])
            lo, hi, coverage = float(lo), float(hi), level
        elif statistic == "mean":
            n = vals.size
            centre = float(vals.mean())
            if n > 1:
                half = float(sps.t.ppf(0.5 * (1.0 + level), n - 1)
                             * vals.std(ddof=1) / np.sqrt(n))
            else:
                half = 0.0
            lo, hi, coverage = centre - half, centre + half, level
        else:
            vals = np.sort(vals)
            lo_i, hi_i, coverage = median_ci_ranks(vals.size, level)
            centre = float(np.median(vals))
            lo, hi = float(vals[lo_i]), float(vals[hi_i])
        rows.append({
            "step": int(step),
            "statistic": statistic,
            "ci_method": ci_method,
            "center": centre,
            "ci_lo": lo,
            "ci_hi": hi,
            "n_seeds": int(vals.size),
            "ci_coverage": coverage,
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    ap.add_argument("--window", type=int, default=10,
                    help="checkpoints per trailing average (default 10)")
    ap.add_argument("--max-step", type=int, default=1_000_000,
                    help="common horizon; DPMD Humanoid runs past it (default 1e6)")
    ap.add_argument("--ci-level", type=float, default=0.50)
    ap.add_argument("--centre", "--center", dest="centre",
                    choices=("median", "mean"), default="median",
                    help="across-seed statistic: 'median' with an "
                         "order-statistic CI, or 'mean' with a Student-t CI")
    ap.add_argument("--ci-method",
                    choices=("order_statistic", "bootstrap", "t"), default=None,
                    help="interval for --aggregation median_of_trailing. "
                         "Default: order_statistic for the median, t for the mean.")
    ap.add_argument("--aggregation",
                    choices=("median_of_trailing", "trailing_of_median"),
                    default="median_of_trailing",
                    help="'median_of_trailing': smooth each run, then take the "
                         "across-seed median (order-statistic CI). "
                         "'trailing_of_median': take the across-seed median per "
                         "checkpoint, then smooth it (bootstrap CI).")
    ap.add_argument("--n-boot", type=int, default=20_000,
                    help="bootstrap resamples for --aggregation trailing_of_median")
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--min-seeds", type=int, default=1,
                    help="drop checkpoints backed by fewer than this many seeds. "
                         "Only meaningful while runs are still in flight, where "
                         "the seed count otherwise falls along the curve.")
    ap.add_argument("--preset", choices=tuple(PRESETS), default="full",
                    help="'full' splits every eval method out; 'simple' shows "
                         "one curve per algorithm (default full)")
    ap.add_argument("--layout", default="2x3", metavar="ROWSxCOLS",
                    help="subplot grid for the six envs, e.g. 3x2 "
                         "(default 2x3). Panel size is held fixed, so the "
                         "figure size follows the grid.")
    ap.add_argument("--no-title", dest="title", action="store_false",
                    help="omit the figure title")
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to a name derived from --preset")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="defaults to the figure path with a .csv suffix")
    args = ap.parse_args()

    if args.centre == "mean" and args.aggregation != "median_of_trailing":
        ap.error("--aggregation trailing_of_median is median-specific; "
                 "use --centre median with it")
    if args.ci_method is None:
        args.ci_method = "t" if args.centre == "mean" else "order_statistic"
    if args.ci_method == "t" and args.centre != "mean":
        ap.error("--ci-method t applies to --centre mean")
    if args.ci_method == "order_statistic" and args.centre != "median":
        ap.error("--ci-method order_statistic applies to --centre median")

    series = PRESETS[args.preset]
    plot_envs = PRESET_ENVS.get(args.preset, ENVS)
    try:
        nrows, ncols = (int(v) for v in args.layout.lower().split("x"))
    except ValueError:
        ap.error("--layout takes ROWSxCOLS, e.g. 3x2")
    if nrows < 1 or ncols < 1 or nrows * ncols < len(plot_envs):
        ap.error(f"--layout {args.layout} leaves fewer than {len(plot_envs)} panels")

    if args.out is None:
        stem = PRESET_STEMS.get(args.preset, "algorithm_comparison")
        if args.aggregation != "median_of_trailing":
            stem += f"_{args.aggregation}"
        if args.centre != "median":
            stem += f"_{args.centre}"
        if args.ci_method == "bootstrap":
            stem += "_bootci"
        if args.window <= 1:
            stem += "_raw"
        if abs(args.ci_level - 0.50) > 1e-9:
            stem += f"_ci{round(100 * args.ci_level)}"
        args.out = REPO / "figures" / f"{stem}.pdf"

    # style.use resets rcParams, so the font setting has to follow it. Type 42
    # keeps text as text in the PDF (Type 3, matplotlib's default, is rejected
    # by several venues and is not editable downstream).
    plt.style.use("default")
    matplotlib.rcParams["pdf.fonttype"] = 42

    loaded = {}
    for algorithm, variant, method, label, *_ in series:
        per_env = load_series(algorithm, variant, method, args.baseline_dir)
        if not per_env:
            print(f"  WARNING no data for {label}")
        loaded[label] = per_env

    agg_rows = []
    rng = np.random.default_rng(args.boot_seed)
    # Per-panel size of the original 2x3 figure, so a different grid keeps the
    # same axes shape instead of stretching them.
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(16 / 3 * ncols, 8.5 / 2 * nrows))
    axes = np.atleast_1d(axes)
    for ax in axes.ravel()[len(plot_envs):]:
        ax.set_visible(False)
    for ax, env in zip(axes.ravel(), plot_envs):
        for algorithm, variant, method, label, colour, linestyle, z in series:
            per_env = loaded[label]
            if env not in per_env:
                continue
            if args.aggregation == "median_of_trailing":
                stats = aggregate(
                    trailing_curves(per_env[env], args.window, args.max_step),
                    args.ci_level, args.centre, args.ci_method,
                    args.n_boot, rng, args.min_seeds)
            else:
                stats = smoothed_median(
                    checkpoint_matrix(per_env[env], args.max_step),
                    args.window, args.ci_level, args.n_boot, rng,
                    args.min_seeds)
            if stats.empty:
                continue
            x = stats["step"] / 1e6
            ax.plot(x, stats["center"], color=colour, label=label,
                    linestyle=linestyle,
                    linewidth=2.2 if colour == "black" else 1.6, zorder=z)
            ax.fill_between(x, stats["ci_lo"], stats["ci_hi"], color=colour,
                            alpha=0.15, linewidth=0, zorder=z - 1)
            stats.insert(0, "aggregation", args.aggregation)
            stats.insert(0, "env", env)
            stats.insert(0, "series", label)
            stats.insert(0, "eval_method", method)
            stats.insert(0, "variant", variant)
            stats.insert(0, "algorithm", algorithm)
            agg_rows.append(stats)
        ax.set_title(env)
        ax.set_xlabel("Environment steps (millions)")
        ax.set_ylabel("Evaluation return")
        ax.grid(alpha=0.3)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(series),
               frameon=False, bbox_to_anchor=(0.5, -0.01))

    coverage = sorted({r["ci_coverage"].iloc[0] for r in agg_rows}) if agg_rows else []
    coverage_txt = ", ".join(f"{100 * c:.1f}%" for c in coverage)
    if args.aggregation == "median_of_trailing":
        smooth_txt = (f"{args.window}-checkpoint trailing average per seed, "
                      if args.window > 1 else "no smoothing, ")
        centre_txt = f"{smooth_txt}{args.centre} over 10 seeds"
        if args.ci_method == "bootstrap":
            band_txt = (f"{100 * args.ci_level:.0f}% percentile bootstrap CI "
                        f"over seeds ({args.n_boot} resamples)")
        elif args.ci_method == "t":
            band_txt = f"Student-t {100 * args.ci_level:.0f}% CI for the mean"
        else:
            band_txt = ("distribution-free CI for the median "
                        f"(nominal {100 * args.ci_level:.0f}%, "
                        f"achieved {coverage_txt})")
    else:
        centre_txt = (f"median over 10 seeds per checkpoint, then a "
                      f"{args.window}-checkpoint trailing average")
        band_txt = (f"middle {100 * args.ci_level:.0f}% of {args.n_boot} "
                    "bootstrap smoothed-median curves")
    print(f"CI: {band_txt}")
    if args.title:
        fig.suptitle(f"Evaluation return: {centre_txt}\nbands: {band_txt}",
                     fontsize=12)
    fig.tight_layout(rect=(0, 0.045, 1, 0.94 if args.title else 1.0))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Wrote {args.out}")

    out_csv = args.out_csv or args.out.with_suffix(".csv")
    if agg_rows:
        pd.concat(agg_rows, ignore_index=True).to_csv(out_csv, index=False)
        print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
