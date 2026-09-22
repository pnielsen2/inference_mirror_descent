#!/usr/bin/env python3
"""Does a sharper sampler walk into the critic ensemble's disagreement?

``--q_agg_sample mean`` makes the MALA chain climb the ENSEMBLE MEAN of Q, but the
TD backup is hardcoded to the clipped-double-Q ``min``. Any action where the two
critics disagree is therefore worth more to the sampler than it is to the backup.
A sampler that lands on the tilted mode more exactly (larger
--predictor_final_steps) searches harder, so it can find exactly those
disagreement points, and the value booked for them is the pessimistic ``min``.

This measures that gap directly on a snapshot:

    gap(K) = E[ mean_i Q_i(a_K) - min_i Q_i(a_K) ]

reported raw, relative to the within-state spread of the mean-Q, and as the drift
in the backup it would induce at the discounted horizon, gap/(1-gamma).
"""
import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from relax.algorithm.mala_sampler import build_mala_sampler
from relax.algorithm.mgmd import _aggregate_q
from scripts.evaluate_snapshot_action_selection import load_algorithm

FIELDS = ("snapshot", "env", "diffusion_steps", "training_eta", "step", "final_k",
          "q_agg_sample", "q_mean_agg", "gap_mean_minus_min", "gap_rel_sd_within",
          "gap_horizon", "q_sd_within", "ens_disagree_replay")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-k", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16])
    parser.add_argument("--n-states", type=int, default=256)
    parser.add_argument("--n-draws", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    import pickle
    algorithm, hparams = load_algorithm(args.snapshot)
    state = jax.tree.map(lambda x: x[0], algorithm.state)
    with (args.snapshot / "replay_batches.pkl").open("rb") as f:
        replay = pickle.load(f)[0]
    obs = jnp.asarray(np.asarray(replay.obs)[: args.n_states])
    replay_act = jnp.asarray(np.asarray(replay.action)[: args.n_states])

    def per_critic(o, a):
        return [np.asarray(algorithm.model.q(qp, o, a), np.float64) for qp in state.params.q]

    qs = per_critic(obs, replay_act)
    stacked = np.stack(qs, 0)
    disagree_replay = float(np.mean(stacked.mean(0) - stacked.min(0)))

    rows = []
    for k in args.final_k:
        predictor = "Identity" if k == 0 else "DDPM_mean"
        sampler = build_mala_sampler(
            **{**algorithm._sampler_kw, "denoising_predictor": predictor,
               "predictor_final_steps": k, "num_denoised_actions": args.n_draws}
        )
        agg = lambda q: _aggregate_q(q, algorithm.cfg.q_agg_sample)
        out = jax.jit(lambda key, s, o: sampler(key, s, o, agg))(
            jax.random.key(args.seed), state, obs)
        act = out.action                                    # [M, B, d]
        obs_m = jnp.broadcast_to(obs, (args.n_draws, *obs.shape))
        stack = np.stack(per_critic(obs_m, act), 0)         # [n_q, M, B]
        mean_q, min_q = stack.mean(0), stack.min(0)
        gap = float(np.mean(mean_q - min_q))
        sd_within = float(np.mean(mean_q.std(axis=0, ddof=1)))
        rows.append({
            "snapshot": str(args.snapshot), "env": hparams["env"],
            "diffusion_steps": hparams["diffusion_steps"],
            "training_eta": hparams["eta"],
            "step": args.snapshot.name.rsplit("_", 1)[-1], "final_k": k,
            "q_agg_sample": algorithm.cfg.q_agg_sample,
            "q_mean_agg": float(mean_q.mean()),
            "gap_mean_minus_min": gap,
            "gap_rel_sd_within": gap / max(sd_within, 1e-9),
            "gap_horizon": gap / (1.0 - args.gamma),
            "q_sd_within": sd_within,
            "ens_disagree_replay": disagree_replay,
        })
        print(f"{hparams['env']:15s} T={hparams['diffusion_steps']:<3} K={k:<3} "
              f"gap={gap:8.4f}  gap/sd={rows[-1]['gap_rel_sd_within']:7.3f}  "
              f"gap/(1-g)={rows[-1]['gap_horizon']:9.2f}  Q={rows[-1]['q_mean_agg']:9.2f}",
              flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    exists = args.output.exists()
    with args.output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
