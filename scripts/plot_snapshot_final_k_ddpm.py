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


VALUES = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
# Best-of-N is priced per candidate, so it is evaluated only through N=1024.
BEST_OF_N_VALUES = VALUES[VALUES <= 1024]
ENVIRONMENTS = (
    ("Ant-v3", "ant"),
    ("Humanoid-v3", "humanoid"),
    ("Walker2d-v3", "walker2d"),
    ("Hopper-v3", "hopper"),
    ("HalfCheetah-v3", "halfcheetah"),
    ("Swimmer-v3", "swimmer"),
)
STYLES = (
    (0, "Identity (K=0)", "#2563eb", "o"),
    (1, "Final K=1", "#059669", "^"),
    (2, "Final K=2", "#d97706", "D"),
    (4, "Final K=4", "#dc2626", "s"),
    (8, "Final K=8", "#7c3aed", "v"),
    (16, "Final K=16", "#0891b2", "P"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--final-k-dir", type=Path, required=True)
    parser.add_argument("--best-of-n-extension-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-eta", type=int, choices=(32, 64), required=True)
    parser.add_argument("--snapshot-step", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    # Preview mode: draw whatever settings already have a full episode set instead of
    # requiring every point, so a run in progress can still be inspected.
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def load_protocol(path, protocol, values, episodes, training_eta, allow_partial=False):
    # Keyed by episode index, not appended, because a setting can appear twice in
    # one CSV: two shards that started before either had written rows both saw it
    # as unfinished and both ran it. The evaluation is deterministic given
    # (snapshot, eval_seed, episodes), so such a repeat is a duplicate rather
    # than a second sample -- it is dropped here, and any repeat that does NOT
    # agree is a real problem and raises.
    per_episode = defaultdict(dict)
    if allow_partial and not path.exists():
        return {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["protocol"] != protocol:
                continue
            if int(float(row["training_eta"])) != training_eta:
                raise ValueError(f"unexpected training eta in {path}: {row['training_eta']}")
            value, episode = int(row["value"]), int(row["episode_index"])
            episode_return = float(row["episode_return"])
            previous = per_episode[value].setdefault(episode, episode_return)
            if previous != episode_return:
                raise ValueError(f"conflicting {protocol} rows in {path}: value={value} "
                                 f"episode={episode} has {previous} and {episode_return}")
    grouped = defaultdict(list, {
        value: [returns[episode] for episode in sorted(returns)]
        for value, returns in per_episode.items()
    })
    if allow_partial:
        return {
            int(value): grouped[int(value)]
            for value in values
            if len(grouped[int(value)]) >= episodes
        }
    bad = {int(value): len(grouped[int(value)]) for value in values if len(grouped[int(value)]) != episodes}
    if bad:
        raise ValueError(f"invalid {protocol} results in {path}: {bad}")
    return grouped


def main():
    args = parse_args()
    if args.snapshot_step <= 0:
        raise ValueError("--snapshot-step must be positive")
    baseline_step_suffix = "" if args.snapshot_step == 1000000 else f"_step{args.snapshot_step}"
    step_label = "1M" if args.snapshot_step == 1000000 else f"{args.snapshot_step // 1000}k"
    fig, axes = plt.subplots(
        len(ENVIRONMENTS), 2, figsize=(14, 23), sharex=True, sharey="row"
    )
    panel_coverage = []
    for row, (env, slug) in enumerate(ENVIRONMENTS):
        for col, steps in enumerate((40, 80)):
            ax = axes[row, col]
            samples_by_k = {
                0: load_protocol(
                    args.baseline_dir
                    / f"{slug}_traineta{args.training_eta}_diff{steps}_mean{baseline_step_suffix}.csv",
                    "tilted_eta",
                    VALUES,
                    args.episodes,
                    args.training_eta,
                    args.allow_partial,
                )
            }
            for k in (1, 2, 4, 8, 16):
                samples_by_k[k] = load_protocol(
                    args.final_k_dir
                    / f"{slug}_traineta{args.training_eta}_diff{steps}_k{k}_step{args.snapshot_step}.csv",
                    "tilted_eta",
                    VALUES,
                    args.episodes,
                    args.training_eta,
                    args.allow_partial,
                )
            best_of_n = load_protocol(
                args.baseline_dir
                / f"{slug}_traineta{args.training_eta}_diff{steps}_min{baseline_step_suffix}.csv",
                "base_best_of_n",
                BEST_OF_N_VALUES[BEST_OF_N_VALUES <= 256],
                args.episodes,
                args.training_eta,
                args.allow_partial,
            )
            extension = load_protocol(
                args.best_of_n_extension_dir
                / f"{slug}_traineta{args.training_eta}_diff{steps}_bestmin_step{args.snapshot_step}.csv",
                "base_best_of_n",
                BEST_OF_N_VALUES[BEST_OF_N_VALUES > 256],
                args.episodes,
                args.training_eta,
                args.allow_partial,
            )
            best_of_n.update(extension)

            def draw(grouped, allowed, color, marker, linestyle, label, width, size, alpha):
                present = [value for value in allowed if int(value) in grouped]
                if not present:
                    return 0
                samples = [np.asarray(grouped[int(value)], np.float64) for value in present]
                means = np.array([sample.mean() for sample in samples])
                sems = np.array([
                    sample.std(ddof=1) / math.sqrt(len(sample)) for sample in samples
                ])
                ax.plot(
                    present, means, color=color, marker=marker, linestyle=linestyle,
                    linewidth=width, markersize=size, label=label,
                )
                ax.fill_between(
                    present, means - sems, means + sems, color=color, alpha=alpha, linewidth=0,
                )
                return len(present)

            drawn = 0
            for k, label, color, marker in STYLES:
                drawn += draw(samples_by_k[k], VALUES, color, marker, "-", label, 1.8, 4.5, 0.09)
            drawn += draw(
                best_of_n, BEST_OF_N_VALUES, "#111827", "X", "--",
                "Base best-of-N: min-Q ranking", 2.0, 4.8, 0.08,
            )
            if not drawn:
                ax.text(0.5, 0.5, "no complete settings yet", transform=ax.transAxes,
                        ha="center", va="center", fontsize=11, color="#6b7280")
            panel_coverage.append(drawn / (len(STYLES) * len(VALUES) + len(BEST_OF_N_VALUES)))
            ax.set_xscale("log", base=2)
            ax.set_xticks(VALUES)
            ax.set_xticklabels([str(value) for value in VALUES])
            ax.grid(True, which="major", alpha=0.25)
            ax.set_title(f"{env} | {steps} diffusion steps", fontsize=12, weight="bold")
            if row == len(ENVIRONMENTS) - 1:
                ax.set_xlabel("Sampling eta / best-of-N N")
            if col == 0:
                ax.set_ylabel("Episode return")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, 0.965), columnspacing=1.5,
    )
    coverage = 100.0 * sum(panel_coverage) / len(panel_coverage)
    title = f"Final-K DDPM-mean predictors at {step_label} steps (training eta={args.training_eta})"
    if args.allow_partial:
        title += f" [PARTIAL: {coverage:.0f}% of points]"
    fig.suptitle(title, fontsize=16, weight="bold", y=0.995)
    footnote = (
        f"Mean over {args.episodes} paired episodes; shaded bands are +/-1 standard error. "
        "Tilted curves use mean-Q guidance; best-of-N uses min-Q ranking and is evaluated through N=1024. "
        "K final transitions use DDPM_mean."
    )
    if args.allow_partial:
        footnote += " PREVIEW of a run still in progress: incomplete points are omitted, so curves may be gapped."
    fig.text(0.5, 0.008, footnote, ha="center", fontsize=9)
    fig.tight_layout(rect=(0.03, 0.025, 0.99, 0.94))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(args.output)
    print(f"{args.output.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
