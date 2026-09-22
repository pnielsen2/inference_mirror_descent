#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VALUES = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
ENVIRONMENTS = (
    ("Ant-v3", "ant"),
    ("Humanoid-v3", "humanoid"),
    ("Walker2d-v3", "walker2d"),
    ("Hopper-v3", "hopper"),
    ("HalfCheetah-v3", "halfcheetah"),
    ("Swimmer-v3", "swimmer"),
)
STYLES = (
    ("mean", "tilted_eta", "Tilted: mean-Q guidance", "#2563eb", "o", "-"),
    ("min", "tilted_eta", "Tilted: min-Q guidance", "#059669", "^", "-"),
    ("mean", "base_best_of_n", "Base best-of-N: mean-Q ranking", "#dc2626", "s", "--"),
    ("min", "base_best_of_n", "Base best-of-N: min-Q ranking", "#d97706", "D", "--"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-eta", type=int, choices=(32, 64), required=True)
    parser.add_argument("--snapshot-step", type=int, default=1000000)
    parser.add_argument("--episodes", type=int, default=100)
    return parser.parse_args()


def load_samples(path, episodes, training_eta):
    grouped = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            if int(float(row["training_eta"])) != training_eta:
                raise ValueError(f"unexpected training eta in {path}: {row['training_eta']}")
            grouped[(row["protocol"], int(row["value"]))].append(
                float(row["episode_return"])
            )
    expected = {
        (protocol, int(value))
        for protocol in ("tilted_eta", "base_best_of_n")
        for value in VALUES
    }
    counts = {key: len(grouped[key]) for key in expected}
    bad = {key: count for key, count in counts.items() if count != episodes}
    unexpected = set(grouped) - expected
    if bad or unexpected:
        raise ValueError(f"invalid results in {path}: bad counts={bad}, unexpected={unexpected}")
    return grouped


def main():
    args = parse_args()
    if args.snapshot_step <= 0:
        raise ValueError("--snapshot-step must be positive")
    step_suffix = "" if args.snapshot_step == 1000000 else f"_step{args.snapshot_step}"
    step_label = "1M" if args.snapshot_step == 1000000 else f"{args.snapshot_step // 1000}k"
    fig, axes = plt.subplots(
        len(ENVIRONMENTS), 2, figsize=(14, 23), sharex=True, sharey="row"
    )
    for row, (env, slug) in enumerate(ENVIRONMENTS):
        for col, steps in enumerate((40, 80)):
            ax = axes[row, col]
            datasets = {
                agg: load_samples(
                    args.input_dir
                    / f"{slug}_traineta{args.training_eta}_diff{steps}_{agg}{step_suffix}.csv",
                    args.episodes,
                    args.training_eta,
                )
                for agg in ("mean", "min")
            }
            for agg, protocol, label, color, marker, linestyle in STYLES:
                samples = [
                    np.asarray(datasets[agg][(protocol, int(value))], np.float64)
                    for value in VALUES
                ]
                means = np.array([sample.mean() for sample in samples])
                sems = np.array([
                    sample.std(ddof=1) / math.sqrt(len(sample)) for sample in samples
                ])
                ax.plot(
                    VALUES, means, color=color, marker=marker, linestyle=linestyle,
                    linewidth=2.0, markersize=4.8, label=label,
                )
                ax.fill_between(
                    VALUES, means - sems, means + sems, color=color,
                    alpha=0.12, linewidth=0,
                )
            ax.set_xscale("log", base=2)
            ax.set_xticks(VALUES)
            ax.set_xticklabels([str(value) for value in VALUES])
            ax.grid(True, which="major", alpha=0.25)
            ax.set_title(f"{env} | {steps} diffusion steps", fontsize=12, weight="bold")
            if row == len(ENVIRONMENTS) - 1:
                ax.set_xlabel("eta for tilted policy / N for base-policy best-of-N")
            if col == 0:
                ax.set_ylabel("Episode return")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 0.965), columnspacing=2.2,
    )
    fig.suptitle(
        f"Action selection from {step_label}-step snapshots (training eta={args.training_eta})",
        fontsize=16, weight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.008,
        f"Lines show mean over {args.episodes} paired episodes; shaded bands are +/-1 standard error. "
        "Mean/min denotes online Q-ensemble aggregation for guidance or candidate ranking.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0.03, 0.025, 0.99, 0.945))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(args.output)
    print(f"{args.output.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
