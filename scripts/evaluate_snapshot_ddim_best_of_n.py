#!/usr/bin/env python3
"""Best-of-N over base-policy candidates drawn by *unguided DDIM*.

The companion measurement to ``evaluate_snapshot_action_selection.py``'s
``base_best_of_n`` protocol, which draws its N candidates from the same MALA
chain at eta = 0 (beta = 0). Both rank with ``--q_agg_sample min`` and share the
episode count, eval seed and per-episode env seeds, so at equal N the two
curves are paired episode-for-episode and differ only in *how* the base policy
was sampled.

One invocation covers one manifest config (env x diffusion_steps x training eta)
across *all* of its snapshot steps, because that is what makes the sweep cheap:
a run's state shapes never change, so the ``--values`` compiled samplers are
built once for the first step and reused for the other nine. Loading a new
checkpoint is then a pickle read and a pointer swap.

    python -m scripts.evaluate_snapshot_ddim_best_of_n \
        --manifest .../action_selection_eval_100ep_eta32_eta64_1m/manifest.txt \
        --config-index 0 --output-dir .../action_selection_eval_100ep_ddim_bestmin

Rows already present in an output CSV are skipped, so re-running after a
timeout resumes rather than duplicating.
"""
import argparse
import pickle
import time
from pathlib import Path

import jax
import numpy as np

from scripts.evaluate_snapshot_action_selection import (
    _repair_legacy_state,
    append_results,
    completed_settings,
    evaluate,
    load_algorithm,
    set_sampling_eta,
)

PROTOCOL = "ddim_best_of_n"
VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
STEPS = tuple(range(100000, 1000001, 100000))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config-index", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, nargs="+", default=list(STEPS))
    # Splitting a config's steps across tasks costs one reload + recompile per
    # task, and buys short jobs, which the backfill scheduler slots into the gaps
    # it holds for higher-priority reservations. Strided so every shard gets a
    # near-equal count.
    parser.add_argument("--step-shard", type=int, default=0)
    parser.add_argument("--num-step-shards", type=int, default=1)
    parser.add_argument("--values", type=int, nargs="+", default=list(VALUES))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=0)
    # DDIM candidates carry no MALA chain and no guidance gradient, so a whole
    # sweep's N fits in one batch; chunking only costs kernel launches.
    parser.add_argument("--candidate-chunk-size", type=int, default=1024)
    args = parser.parse_args()
    if args.episodes <= 0 or any(v <= 0 for v in args.values):
        parser.error("--episodes and every --values entry must be positive")
    if args.candidate_chunk_size <= 0:
        parser.error("--candidate-chunk-size must be positive")
    if not 0 <= args.step_shard < args.num_step_shards:
        parser.error("--step-shard must be in [0, --num-step-shards)")
    return args


def read_config(manifest: Path, config_index: int):
    """``(stem, snapshot_dir_of_the_1M_checkpoint)`` for one manifest config.

    The manifest holds a (mean-Q, min-Q) line pair per config; the min-Q line is
    the one whose CSV stem names the best-of-N files, so the DDIM files sit
    beside their eta=0 counterparts under the same stem.
    """
    lines = [line.split("|") for line in manifest.read_text().splitlines()]
    configs = [(lines[i - 1], lines[i]) for i in range(1, len(lines), 2)]
    if not 0 <= config_index < len(configs):
        raise ValueError(f"config index {config_index} outside [0, {len(configs)})")
    _, min_output, snapshot_1m, agg = configs[config_index][1]
    if agg != "min":
        raise ValueError(f"expected the min-Q line of config {config_index}, got {agg!r}")
    return min_output.removesuffix("_min.csv"), Path(snapshot_1m)


def snapshot_at_step(snapshot_1m: Path, step: int) -> Path:
    prefix = snapshot_1m.name.rsplit("_", 1)[0]
    return snapshot_1m.with_name(f"{prefix}_{step}")


def load_state(snapshot: Path):
    with (snapshot / "algorithm_state.pkl").open("rb") as f:
        return _repair_legacy_state(pickle.load(f))


def main():
    args = parse_args()
    stem, snapshot_1m = read_config(args.manifest, args.config_index)
    steps = sorted(args.steps)[args.step_shard::args.num_step_shards]
    outputs = {step: args.output_dir / f"{stem}_ddimmin_step{step}.csv" for step in steps}
    todo = {
        step: [v for v in args.values
               if (PROTOCOL, v) not in completed_settings(outputs[step], args.episodes)]
        for step in steps
    }
    pending = {step: values for step, values in todo.items() if values}
    print(f"config={args.config_index} stem={stem} "
          f"shard={args.step_shard}/{args.num_step_shards} steps={sorted(steps)} "
          f"pending_steps={len(pending)} settings={sum(map(len, pending.values()))}",
          flush=True)
    if not pending:
        return

    # Q ranking must be min-Q, matching the eta=0 best-of-N curve. The snapshot's
    # own --q_agg_sample (mean for these runs) is a training-time choice and does
    # not constrain how evaluation ranks candidates.
    algorithm, hparams = load_algorithm(snapshot_1m, q_agg_sample="min")
    latent_action = bool(hparams.get("latent_action", False))
    print(f"loaded {hparams['env']} diffusion_steps={hparams['diffusion_steps']} "
          f"training_eta={hparams['eta']} seed={hparams['seed']} "
          f"q_agg_sample={algorithm.cfg.q_agg_sample} sampler=ddim(unguided)", flush=True)

    for step, values in sorted(pending.items()):
        snapshot = snapshot_at_step(snapshot_1m, step)
        # eta = 0 leaves the base policy untilted. The DDIM chain ignores eta
        # outright; setting it keeps the state identical to the one the eta=0
        # MALA best-of-N ran on, so nothing but the sampler differs.
        set_sampling_eta(algorithm, load_state(snapshot), 0.0)
        for value in values:
            started = time.monotonic()
            returns, lengths = evaluate(
                algorithm, hparams["env"], args.episodes, value, args.eval_seed,
                latent_action, args.candidate_chunk_size, sampler_kind="ddim")
            elapsed = time.monotonic() - started
            append_results(outputs[step], snapshot, hparams, PROTOCOL, value,
                           returns, lengths, args.eval_seed, elapsed)
            print(f"step={step} {PROTOCOL}={value}: mean={returns.mean():.1f} "
                  f"std={returns.std():.1f} ({elapsed:.0f}s)", flush=True)
    print(f"config={args.config_index} complete", flush=True)


if __name__ == "__main__":
    main()
