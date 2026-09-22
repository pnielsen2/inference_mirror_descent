#!/usr/bin/env python3
"""How the base policy is sampled, at equal N: eta=0 MALA chain vs DDIM vs DIPO.

Read-outs of the *same* snapshot over the *same* 100 paired evaluation episodes,
differing only in how an action is produced:

* ``eta=0 MALA chain`` (always drawn) -- MGMD's MALA-corrected chain run untilted
  (eta = beta = 0), N candidates ranked by min-Q. The ``base_best_of_n``
  protocol of ``evaluate_snapshot_action_selection.py``.
* ``Unguided DDIM`` (``--ddim-dir``) -- a plain deterministic DDIM chain, same N,
  same min-Q ranking (``evaluate_snapshot_ddim_best_of_n.py``).
* ``DIPO-style eval`` (``--dipo-dir``) -- DIPO's evaluation sampler: the
  deterministic DDPM posterior-mean chain over clipped x0 predictions, *one*
  action per state and no critic involvement
  (``evaluate_snapshot_dipo_eval.py``). It has no N, so it is drawn as a
  horizontal reference line rather than a curve.

Both comparison samplers are optional, so the same script draws the pair that is
wanted (or just the best-of-N curve). One figure per (snapshot step, training
eta), laid out like the final-K figures: one row per env, one column per
diffusion-step count.
"""
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.plot_snapshot_final_k_ddpm import ENVIRONMENTS, load_protocol

VALUES = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
# The eta=0 sweep is split across two runs: the baseline dir covers N <= 256 and
# the extension dir the two points above it.
BASELINE_VALUES = VALUES[VALUES <= 256]
EXTENSION_VALUES = VALUES[VALUES > 256]
CURVES = (
    ("tilt", "Base best-of-N: min-Q ranking", "#111827", "X", "--"),
    ("ddim", "Base best-of-N: unguided DDIM", "#dc2626", "o", "-"),
)
DIPO_STYLE = ("DIPO-style eval: DDPM-mean, N=1", "#2563eb", ":")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--best-of-n-extension-dir", type=Path, required=True)
    parser.add_argument("--ddim-dir", type=Path)
    parser.add_argument("--dipo-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-eta", type=int, choices=(32, 64), required=True)
    parser.add_argument("--snapshot-step", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    # Preview mode: draw whatever settings already have a full episode set instead
    # of requiring every point, so a run in progress can still be inspected.
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.snapshot_step <= 0:
        parser.error("--snapshot-step must be positive")
    return args


def load_curves(args, slug, steps):
    """``{"tilt": {N: returns}, "ddim": ..., "dipo": {1: returns}}`` for a panel."""
    stem = f"{slug}_traineta{args.training_eta}_diff{steps}"
    step_suffix = "" if args.snapshot_step == 1000000 else f"_step{args.snapshot_step}"
    load = lambda path, protocol, values: load_protocol(
        path, protocol, values, args.episodes, args.training_eta, args.allow_partial)
    tilt = load(args.baseline_dir / f"{stem}_min{step_suffix}.csv",
                "base_best_of_n", BASELINE_VALUES)
    tilt.update(load(
        args.best_of_n_extension_dir / f"{stem}_bestmin_step{args.snapshot_step}.csv",
        "base_best_of_n", EXTENSION_VALUES))
    ddim = {}
    if args.ddim_dir is not None:
        ddim = load(args.ddim_dir / f"{stem}_ddimmin_step{args.snapshot_step}.csv",
                    "ddim_best_of_n", VALUES)
    dipo = {}
    if args.dipo_dir is not None:
        dipo = load(args.dipo_dir / f"{stem}_dipo_step{args.snapshot_step}.csv",
                    "dipo_ddpm_mean", [1])
    return {"tilt": tilt, "ddim": ddim, "dipo": dipo}


def mean_and_sem(sample):
    sample = np.asarray(sample, np.float64)
    return sample.mean(), sample.std(ddof=1) / math.sqrt(len(sample))


def main():
    args = parse_args()
    step_label = "1M" if args.snapshot_step == 1000000 else f"{args.snapshot_step // 1000}k"
    fig, axes = plt.subplots(len(ENVIRONMENTS), 2, figsize=(12, 21),
                             sharex=True, sharey="row")
    requested = [curve for curve in CURVES if curve[0] != "ddim" or args.ddim_dir]
    drawn, expected = 0, 0
    for row, (env, slug) in enumerate(ENVIRONMENTS):
        for col, steps in enumerate((40, 80)):
            ax = axes[row, col]
            curves = load_curves(args, slug, steps)
            expected += len(requested) * len(VALUES) + bool(args.dipo_dir)
            for key, label, color, marker, linestyle in requested:
                grouped = curves[key]
                present = [int(v) for v in VALUES if int(v) in grouped]
                if not present:
                    continue
                stats = [mean_and_sem(grouped[v]) for v in present]
                means = np.array([mean for mean, _ in stats])
                sems = np.array([sem for _, sem in stats])
                ax.plot(present, means, color=color, marker=marker, linestyle=linestyle,
                        linewidth=2.0, markersize=5.0, label=label)
                ax.fill_between(present, means - sems, means + sems, color=color,
                                alpha=0.15, linewidth=0)
                drawn += len(present)
            if curves["dipo"]:
                # N-independent: one action per state, no ranking. Spanning the
                # axis is what makes it readable as a reference level.
                label, color, linestyle = DIPO_STYLE
                mean, sem = mean_and_sem(curves["dipo"][1])
                span = [int(VALUES[0]), int(VALUES[-1])]
                ax.plot(span, [mean, mean], color=color, linestyle=linestyle,
                        linewidth=2.2, label=label)
                ax.fill_between(span, mean - sem, mean + sem, color=color,
                                alpha=0.15, linewidth=0)
                drawn += 1
            if not any(curves.values()):
                ax.text(0.5, 0.5, "no complete settings yet", transform=ax.transAxes,
                        ha="center", va="center", fontsize=11, color="#6b7280")
            ax.set_xscale("log", base=2)
            ax.set_xticks(VALUES)
            ax.set_xticklabels([str(value) for value in VALUES])
            ax.grid(True, which="major", alpha=0.25)
            ax.set_title(f"{env} | {steps} diffusion steps", fontsize=12, weight="bold")
            if row == len(ENVIRONMENTS) - 1:
                ax.set_xlabel("N (best-of-N candidates)")
            if col == 0:
                ax.set_ylabel("Episode return")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 0.955), columnspacing=1.8)
    coverage = 100.0 * drawn / expected
    title = (f"Sampling the base policy at {step_label} steps "
             f"(training eta={args.training_eta})")
    if args.allow_partial:
        title += f" [PARTIAL: {coverage:.0f}% of points]"
    fig.suptitle(title, fontsize=15, weight="bold", y=0.99)
    footnote = (
        f"Mean over {args.episodes} paired episodes; shaded bands are +/-1 standard error. "
        "Every read-out is of the same snapshot at the run's own diffusion-step count and "
        "differs only in the sampler. Best-of-N draws N untilted candidates"
        + (" (MALA chain at eta=0, or unguided DDIM)" if args.ddim_dir else
           " from the MALA chain at eta=0")
        + " and executes the highest min-Q one."
    )
    if args.dipo_dir:
        footnote += (
            " The DIPO-style line is N-independent: one deterministic DDPM-mean action per "
            "state, critic never consulted (DIPO itself evaluates over 10 episodes, not 100)."
        )
    if args.allow_partial:
        footnote += " PREVIEW of a run still in progress: incomplete points are omitted."
    fig.text(0.5, 0.008, footnote, ha="center", fontsize=9)
    fig.tight_layout(rect=(0.03, 0.02, 0.99, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"{args.output}  ({args.output.stat().st_size / 1024:.1f} KiB, "
          f"{coverage:.0f}% of points)")


if __name__ == "__main__":
    main()
