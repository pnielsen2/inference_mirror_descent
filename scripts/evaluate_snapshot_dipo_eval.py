#!/usr/bin/env python3
"""DIPO's evaluation protocol, replayed on the action-selection snapshots.

DIPO (`/n/home09/pnielsen/DIPO`) evaluates with `agent.sample_action(state,
eval=True)`, which is `Diffusion.sample(..., eval=True)`: `noise_ratio = 0`, so
every `p_sample` drops its ancestral noise and returns the DDPM posterior mean
over an x0 prediction clamped to [-1, 1], for the full reverse chain. One action
per state, and the critic is not consulted -- there is no best-of-N. So unlike
`evaluate_snapshot_ddim_best_of_n.py` there is nothing to sweep here: one
evaluation per snapshot, which on this plot is a horizontal reference line.

`--episodes` defaults to 100 rather than DIPO's 10, because the env seeds are
`default_rng(eval_seed).integers(...)` sized by the episode count -- matching the
count is what pairs these episodes with the eta=0 and DDIM best-of-N runs -- and
one N=1 setting costs ~15s either way.

    python -m scripts.evaluate_snapshot_dipo_eval \
        --manifest .../action_selection_eval_100ep_eta32_eta64_1m/manifest.txt \
        --config-shard 0 --num-config-shards 6 --output-dir .../..._dipo_eval

Rows already present in an output CSV are skipped, so re-running after a timeout
resumes rather than duplicating.
"""
import argparse
import time
from pathlib import Path

# The manifest/snapshot/state helpers are shared with the best-of-N runner; it is
# the one place the manifest layout is decoded.
from scripts.evaluate_snapshot_ddim_best_of_n import (
    STEPS,
    load_state,
    read_config,
    snapshot_at_step,
)
from scripts.evaluate_snapshot_action_selection import (
    append_results,
    completed_settings,
    evaluate,
    load_algorithm,
    set_sampling_eta,
)

PROTOCOL = "dipo_ddpm_mean"
NUM_CONFIGS = 24


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-shard", type=int, default=0)
    parser.add_argument("--num-config-shards", type=int, default=1)
    parser.add_argument("--steps", type=int, nargs="+", default=list(STEPS))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=0)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if not 0 <= args.config_shard < args.num_config_shards:
        parser.error("--config-shard must be in [0, --num-config-shards)")
    return args


def main():
    args = parse_args()
    configs = range(NUM_CONFIGS)[args.config_shard::args.num_config_shards]
    print(f"shard={args.config_shard}/{args.num_config_shards} configs={list(configs)}", flush=True)
    for config_index in configs:
        stem, snapshot_1m = read_config(args.manifest, config_index)
        outputs = {step: args.output_dir / f"{stem}_dipo_step{step}.csv" for step in args.steps}
        pending = [step for step in sorted(args.steps)
                   if (PROTOCOL, 1) not in completed_settings(outputs[step], args.episodes)]
        if not pending:
            print(f"config={config_index} stem={stem}: already complete", flush=True)
            continue
        # The DDPM-mean chain ignores the Q ensemble entirely, so --q_agg_sample
        # is immaterial here; min keeps the loaded config identical to the
        # best-of-N runs it is plotted against.
        algorithm, hparams = load_algorithm(snapshot_1m, q_agg_sample="min")
        latent_action = bool(hparams.get("latent_action", False))
        print(f"config={config_index} stem={stem} steps={pending} "
              f"env={hparams['env']} diffusion_steps={hparams['diffusion_steps']} "
              f"training_eta={hparams['eta']} sampler=ddpm_mean(unguided, deterministic)",
              flush=True)
        for step in pending:
            snapshot = snapshot_at_step(snapshot_1m, step)
            # eta = 0 leaves the base policy untilted; the chain ignores eta
            # outright, so this only keeps the state identical to the one the
            # eta=0 best-of-N ran on.
            set_sampling_eta(algorithm, load_state(snapshot), 0.0)
            started = time.monotonic()
            returns, lengths = evaluate(
                algorithm, hparams["env"], args.episodes, 1, args.eval_seed,
                latent_action, 1, sampler_kind="ddpm_mean")
            elapsed = time.monotonic() - started
            append_results(outputs[step], snapshot, hparams, PROTOCOL, 1,
                           returns, lengths, args.eval_seed, elapsed)
            print(f"config={config_index} step={step} {PROTOCOL}: mean={returns.mean():.1f} "
                  f"std={returns.std():.1f} ({elapsed:.0f}s)", flush=True)
    print(f"shard={args.config_shard} complete", flush=True)


if __name__ == "__main__":
    main()
