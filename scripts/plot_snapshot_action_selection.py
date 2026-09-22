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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--file-suffix", default="")
    parser.add_argument("--q-agg-sample", choices=("min", "mean"), required=True)
    parser.add_argument("--episodes", type=int, default=10)
    return parser.parse_args()


def load_samples(path, episodes):
    grouped = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            grouped[(row["protocol"], int(row["value"]))].append(
                float(row["episode_return"]))
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
    suffix = args.file_suffix
    panels = {
        ("Ant-v3", 40): args.input_dir / f"ant_diff40{suffix}.csv",
        ("Ant-v3", 80): args.input_dir / f"ant_diff80{suffix}.csv",
        ("Humanoid-v3", 40): args.input_dir / f"humanoid_diff40{suffix}.csv",
        ("Humanoid-v3", 80): args.input_dir / f"humanoid_diff80{suffix}.csv",
    }
    protocols = {
        "tilted_eta": ("Tilted policy: N=1, x=eta", "#2563eb", "o"),
        "base_best_of_n": ("Base policy: beta=0, x=N", "#dc2626", "s"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True, sharey="row")
    for row, env in enumerate(("Ant-v3", "Humanoid-v3")):
        for col, steps in enumerate((40, 80)):
            ax = axes[row, col]
            grouped = load_samples(panels[(env, steps)], args.episodes)
            for protocol, (label, color, marker) in protocols.items():
                samples = [np.asarray(grouped[(protocol, int(value))]) for value in VALUES]
                means = np.array([sample.mean() for sample in samples])
                sems = np.array([
                    sample.std(ddof=1) / math.sqrt(len(sample)) for sample in samples
                ])
                ax.plot(
                    VALUES, means, color=color, marker=marker, linewidth=2.1,
                    markersize=5.5, label=label,
                )
                ax.fill_between(
                    VALUES, means - sems, means + sems, color=color,
                    alpha=0.18, linewidth=0,
                )
            ax.set_xscale("log", base=2)
            ax.set_xticks(VALUES)
            ax.set_xticklabels([str(value) for value in VALUES])
            ax.grid(True, which="major", alpha=0.25)
            ax.set_title(f"{env} | {steps} diffusion steps", fontsize=12, weight="bold")
            if row == 1:
                ax.set_xlabel("eta for tilted policy / N for base-policy best-of-N")
            if col == 0:
                ax.set_ylabel("Episode return")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 0.945),
    )
    fig.suptitle(
        "Action selection from 1M-step snapshots (training eta=64)",
        fontsize=15, weight="bold", y=0.99,
    )
    fig.text(
        0.5, 0.015,
        f"Lines show mean over {args.episodes} paired episodes; shaded bands are +/-1 standard error. "
        f"Online Q ensemble aggregation: {args.q_agg_sample}.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0.03, 0.05, 0.99, 0.91))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(args.output)
    print(f"{args.output.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
