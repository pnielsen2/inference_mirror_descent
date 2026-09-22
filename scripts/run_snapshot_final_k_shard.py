#!/usr/bin/env python3
import argparse
import csv
import heapq
import json
import subprocess
import sys
from pathlib import Path


K_VALUES = (1, 2, 4, 8, 16)
ALL_VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
EXTENDED_VALUES = (512, 1024)
# Tilted sampling draws one candidate per state, so its cost is independent of the
# value and the full range stays affordable even where best-of-N does not.
TILTED_EXTENDED_VALUES = (512, 1024, 2048, 4096)
# Cost of one setting relative to the measured base_best_of_n=256 time, fit to the
# observed N=1 vs N=256 runtimes: a fixed per-step share plus a share linear in N.
FIXED_COST_SHARE = 0.115


def parse_args():
    parser = argparse.ArgumentParser()
    # The plan is built once and written to disk: shards start at different times, so
    # deriving it from on-disk progress inside each shard would give each of them a
    # different assignment and silently drop snapshots.
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int, default=64)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--final-k-dir", type=Path, required=True)
    parser.add_argument("--best-of-n-extension-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--candidate-chunk-size", type=int, default=256)
    args = parser.parse_args()
    if not args.write_plan and args.shard_index is None:
        parser.error("--shard-index is required unless --write-plan is set")
    return args


def result_name(base_name, step):
    return base_name if step == 1000000 else f"{base_name[:-4]}_step{step}.csv"


def snapshot_at_step(snapshot_1m, step):
    prefix = snapshot_1m.name.rsplit("_", 1)[0]
    return snapshot_1m.with_name(f"{prefix}_{step}")


def n256_elapsed(path):
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["protocol"] == "base_best_of_n" and int(row["value"]) == 256:
                return float(row["elapsed_seconds"])
    raise ValueError(f"missing base_best_of_n=256 in {path}")


def setting_cost(n256_seconds, value):
    return n256_seconds * (FIXED_COST_SHARE + (1.0 - FIXED_COST_SHARE) * value / 256.0)


def complete_values(path, protocol, episodes):
    """Values of ``protocol`` in ``path`` that already have ``episodes`` rows."""
    if not path.exists():
        return set()
    counts = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["protocol"] != protocol:
                continue
            value = int(row["value"])
            counts[value] = counts.get(value, 0) + 1
    return {value for value, count in counts.items() if count >= episodes}


def remaining_cost(paths, n256_seconds, episodes):
    """Estimated seconds still owed for one snapshot, skipping finished settings."""
    best_done = complete_values(paths["best"], "base_best_of_n", episodes)
    total = sum(
        setting_cost(n256_seconds, value)
        for value in EXTENDED_VALUES
        if value not in best_done
    )
    tilted_done = complete_values(paths["tilted"], "tilted_eta", episodes)
    total += sum(
        setting_cost(n256_seconds, 1)
        for value in TILTED_EXTENDED_VALUES
        if value not in tilted_done
    )
    for path in paths["final_k"]:
        done = complete_values(path, "tilted_eta", episodes)
        total += sum(
            setting_cost(n256_seconds, 1) for value in ALL_VALUES if value not in done
        )
    return total


def build_assignments(baseline_dir, final_k_dir, best_of_n_dir, num_shards, episodes):
    lines = [line.split("|") for line in (baseline_dir / "manifest.txt").read_text().splitlines()]
    configs = [(lines[i - 1], lines[i]) for i in range(1, 48, 2)]
    work = []
    for step in range(100000, 1000001, 100000):
        for config_index, (mean_config, min_config) in enumerate(configs):
            mean_output, min_output = mean_config[1], min_config[1]
            n256_seconds = n256_elapsed(baseline_dir / result_name(min_output, step))
            final_stem = mean_output.removesuffix("_mean.csv")
            paths = {
                "best": best_of_n_dir / f"{min_output.removesuffix('_min.csv')}_bestmin_step{step}.csv",
                "tilted": baseline_dir / result_name(mean_output, step),
                "final_k": [
                    final_k_dir / f"{final_stem}_k{k}_step{step}.csv" for k in K_VALUES
                ],
            }
            weight = remaining_cost(paths, n256_seconds, episodes)
            if weight <= 0.0:
                continue
            work.append((weight, step, config_index, mean_config, min_config))
    shards = [(0.0, index, []) for index in range(num_shards)]
    heapq.heapify(shards)
    for item in sorted(work, key=lambda value: value[0], reverse=True):
        load, shard_index, items = heapq.heappop(shards)
        items.append(item)
        heapq.heappush(shards, (load + item[0], shard_index, items))
    return {shard_index: (load, items) for load, shard_index, items in shards}


def run_evaluator(snapshot, output, episodes, eval_seed, values, protocol, q_agg_sample,
                  candidate_chunk_size, ddpm_mean_final_steps=None):
    command = [
        sys.executable,
        "scripts/evaluate_snapshot_action_selection.py",
        "--snapshot", str(snapshot),
        "--output", str(output),
        "--episodes", str(episodes),
        "--values", *(str(value) for value in values),
        "--protocol", protocol,
        "--eval-seed", str(eval_seed),
        "--q-agg-sample", q_agg_sample,
        "--candidate-chunk-size", str(candidate_chunk_size),
    ]
    if ddpm_mean_final_steps is not None:
        command.extend(("--ddpm-mean-final-steps", str(ddpm_mean_final_steps)))
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def write_plan(args):
    assignments = build_assignments(
        args.baseline_dir, args.final_k_dir, args.best_of_n_extension_dir,
        args.num_shards, args.episodes,
    )
    plan = {
        "num_shards": args.num_shards,
        "shards": {
            str(shard_index): {
                "estimated_hours": load / 3600.0,
                "items": [[step, config_index] for _, step, config_index, _, _ in items],
            }
            for shard_index, (load, items) in assignments.items()
        },
    }
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(json.dumps(plan, indent=1))
    total = sum(entry["estimated_hours"] for entry in plan["shards"].values())
    counts = [len(entry["items"]) for entry in plan["shards"].values()]
    hours = [entry["estimated_hours"] for entry in plan["shards"].values()]
    print(
        f"wrote {args.plan}: shards={args.num_shards} snapshots={sum(counts)} "
        f"total_hours={total:.2f} per_shard_hours={min(hours):.2f}-{max(hours):.2f} "
        f"snapshots_per_shard={min(counts)}-{max(counts)}"
    )


def main():
    args = parse_args()
    if args.write_plan:
        write_plan(args)
        return
    plan = json.loads(args.plan.read_text())
    if not 0 <= args.shard_index < plan["num_shards"]:
        raise ValueError(f"shard index {args.shard_index} outside [0, {plan['num_shards']})")
    entry = plan["shards"][str(args.shard_index)]
    lines = [line.split("|") for line in (args.baseline_dir / "manifest.txt").read_text().splitlines()]
    configs = [(lines[i - 1], lines[i]) for i in range(1, 48, 2)]
    print(
        f"shard={args.shard_index}/{plan['num_shards']} snapshots={len(entry['items'])} "
        f"estimated_remaining_hours={entry['estimated_hours']:.2f}",
        flush=True,
    )
    for step, config_index in sorted(entry["items"]):
        mean_config, min_config = configs[config_index]
        _, mean_output, snapshot_1m, _ = mean_config
        _, min_output, _, _ = min_config
        snapshot = snapshot_at_step(Path(snapshot_1m), step)
        stem = min_output.removesuffix("_min.csv")
        run_evaluator(
            snapshot,
            args.best_of_n_extension_dir / f"{stem}_bestmin_step{step}.csv",
            args.episodes,
            args.eval_seed,
            EXTENDED_VALUES,
            "base",
            "min",
            args.candidate_chunk_size,
        )
        run_evaluator(
            snapshot,
            args.baseline_dir / result_name(mean_output, step),
            args.episodes,
            args.eval_seed,
            TILTED_EXTENDED_VALUES,
            "tilted",
            "mean",
            args.candidate_chunk_size,
        )
        final_stem = mean_output.removesuffix("_mean.csv")
        for k in K_VALUES:
            run_evaluator(
                snapshot,
                args.final_k_dir / f"{final_stem}_k{k}_step{step}.csv",
                args.episodes,
                args.eval_seed,
                ALL_VALUES,
                "tilted",
                "mean",
                args.candidate_chunk_size,
                k,
            )
    print(f"shard={args.shard_index} complete", flush=True)


if __name__ == "__main__":
    main()
