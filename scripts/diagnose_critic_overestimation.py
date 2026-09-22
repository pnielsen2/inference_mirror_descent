#!/usr/bin/env python3
"""Compare a snapshot's critic against the value its own rollouts can justify.

For a discount gamma the value of a policy earning mean per-step reward r is
r / (1 - gamma). Each snapshot ships the replay minibatch that produced it, so
Q(s, a) on that batch is directly comparable to the return the same run was
recording at the same env step. A critic that sits far above the implied value is
overestimating, which is the failure mode target-policy smoothing exists to stop:
removing the sampler's residual action noise (raising --predictor_final_steps)
makes the TD next-action approach argmax_a Q and removes that smoothing.

Only the critic is evaluated, so this needs no diffusion sampling and runs on CPU
in seconds per snapshot.
"""
import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np

FIELDS = (
    "snapshot", "run", "env", "step", "diffusion_steps", "training_eta",
    "denoising_predictor", "predictor_final_steps", "seed",
    "q_mean_replay", "q_std_replay", "q_min_replay", "q_max_replay",
    "q_spread_ratio", "rollout_return", "rollout_length", "implied_value",
    "overestimation_ratio", "overestimation_abs",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, nargs="+", required=True)
    parser.add_argument("--codebase-logs", type=Path, nargs="*", default=[],
                        help="logs/ dirs holding episode_returns.csv mirrors.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    return parser.parse_args()


def rollout_at(logs_dirs, env, run_prefix, step, seed_slot):
    """Mean rollout return and episode length near ``step`` for this run/slot."""
    for logs in logs_dirs:
        env_dir = Path(logs) / env
        if not env_dir.is_dir():
            continue
        for exp in env_dir.iterdir():
            if not exp.name.startswith(run_prefix):
                continue
            path = exp / "episode_returns.csv"
            if not path.exists():
                continue
            with path.open() as f:
                reader = csv.DictReader(f)
                column = next(c for c in reader.fieldnames if c.startswith("episode_return"))
                rows = [
                    (int(r["seed"]), int(r["step"]), float(r[column]))
                    for r in reader
                ]
            if not rows:
                continue
            data = np.array(rows, dtype=np.float64)
            window = (np.abs(data[:, 1] - step) <= 0.05 * max(step, 1))
            if seed_slot is not None and (data[:, 0] == seed_slot).any():
                window &= data[:, 0] == seed_slot
            if not window.any():
                continue
            return float(data[window, 2].mean()), exp.name
    return float("nan"), ""


def main():
    args = parse_args()
    import jax
    import jax.numpy as jnp
    from relax.algorithm.mgmd import _aggregate_q
    from scripts.evaluate_snapshot_action_selection import load_algorithm

    rows = []
    for snapshot in args.snapshot:
        algorithm, hparams = load_algorithm(snapshot)
        state = jax.tree.map(lambda x: x[0], algorithm.state)
        with (snapshot / "replay_batches.pkl").open("rb") as f:
            replay = pickle.load(f)[0]
        obs = jnp.asarray(np.asarray(replay.obs))
        act = jnp.asarray(np.asarray(replay.action))
        q = np.asarray(
            _aggregate_q([algorithm.model.q(qp, obs, act) for qp in state.params.q],
                         algorithm.cfg.q_agg_sample),
            np.float64,
        )
        step = int(snapshot.name.rsplit("_", 1)[-1])
        run_prefix = snapshot.parent.name.split("__")[0]
        seed = int(hparams["seed"])
        ret, exp_name = rollout_at(args.codebase_logs, hparams["env"], run_prefix, step, None)
        # Episodes are capped at 1000 steps in these envs; per-step reward ~ ret/1000.
        length = 1000.0
        implied = (ret / length) / (1.0 - args.gamma)
        rows.append({
            "snapshot": str(snapshot), "run": exp_name, "env": hparams["env"],
            "step": step, "diffusion_steps": hparams["diffusion_steps"],
            "training_eta": hparams["eta"],
            "denoising_predictor": hparams.get("denoising_predictor", ""),
            "predictor_final_steps": hparams.get("predictor_final_steps", ""),
            "seed": seed,
            "q_mean_replay": float(q.mean()), "q_std_replay": float(q.std(ddof=1)),
            "q_min_replay": float(q.min()), "q_max_replay": float(q.max()),
            "q_spread_ratio": float(q.std(ddof=1) / max(abs(q.mean()), 1e-9)),
            "rollout_return": ret, "rollout_length": length,
            "implied_value": implied,
            "overestimation_ratio": float(q.mean() / implied) if implied else float("nan"),
            "overestimation_abs": float(q.mean() - implied),
        })
        print(json.dumps({k: rows[-1][k] for k in (
            "env", "step", "predictor_final_steps", "q_mean_replay", "rollout_return",
            "implied_value", "overestimation_ratio")}), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    exists = args.output.exists()
    with args.output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
