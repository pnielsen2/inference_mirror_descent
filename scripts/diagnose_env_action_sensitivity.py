#!/usr/bin/env python3
"""Per-environment action-noise sensitivity of a saved MGMD snapshot.

Answers, without any retraining, why one guidance setting cannot serve every env:

* How wide is the tilted policy at a fixed state (``a_sd_within``), and how much of
  the action box does it use (``a_abs_mean``, ``a_sat_frac``)?
* How much value does a sigma-sized action perturbation cost
  (``dq_sigma*``), i.e. E_eps[Q(s, a + sigma eps)] - Q(s, a)? The identity
  predictor hands the env exactly such a perturbation, with sigma = the
  schedule's level-0 noise, so this is the value price of K=0 vs K=1.
* The curvature that sets that price to leading order, tr(H) of Q in the action
  (Hutchinson), since E[Q(a+sigma eps)] - Q(a) = 0.5 sigma^2 tr(H) + O(sigma^4).
* Whether the guidance normalizer is calibrated for the env: ``q_sd_within`` is
  the within-state spread the tilt actually acts on, ``q_sd_pooled`` the batch
  spread ``--ema_advantage_normalization`` divides by. Their ratio is the factor
  by which the realized tilt departs from the intended one.

All value quantities are also reported relative to ``q_sd_within``, which is the
scale that makes them comparable across envs with different reward magnitudes.
"""
import argparse
import csv
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from relax.algorithm.mala_sampler import build_mala_sampler
from relax.algorithm.mgmd import _aggregate_q
from scripts.evaluate_snapshot_action_selection import load_algorithm

FIELDS = (
    "snapshot", "env", "diffusion_steps", "training_eta", "training_seed", "step",
    "predictor", "final_k", "sample_eta", "act_dim", "n_states", "n_draws",
    "sigma_level0", "sigma_level1",
    "a_sd_within", "a_sd_across", "a_abs_mean", "a_sat_frac",
    "q_sd_within", "q_sd_pooled", "q_sd_ratio", "q_mean",
    "tr_hess", "tr_hess_rel",
    "dq_sigma_l0", "dq_sigma_l0_rel", "dq_sigma_l1", "dq_sigma_l1_rel",
    "dq_sigma_0p05", "dq_sigma_0p05_rel", "dq_sigma_0p10", "dq_sigma_0p10_rel",
    "sigma_for_1pct_qsd",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-eta", type=float, help="Override the tilt strength.")
    parser.add_argument("--predictor", default="Identity")
    parser.add_argument("--final-k", type=int, default=0)
    parser.add_argument("--n-states", type=int, default=256)
    parser.add_argument("--n-draws", type=int, default=16)
    parser.add_argument("--n-noise", type=int, default=32)
    parser.add_argument("--n-hutchinson", type=int, default=16)
    parser.add_argument("--slot", type=int, default=0,
                        help="vmap slot to analyse in a packed run.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def set_eta(algorithm, base_state, eta):
    algorithm.state = base_state._replace(
        beta=jnp.full_like(base_state.beta, eta),
        hp=base_state.hp._replace(
            alpha=jnp.ones_like(base_state.hp.alpha),
            T=jnp.zeros_like(base_state.hp.T),
            eta=jnp.full_like(base_state.hp.eta, eta),
        ),
    )


def main():
    args = parse_args()
    algorithm, hparams = load_algorithm(args.snapshot)
    cfg = dataclasses.replace(
        algorithm.cfg, denoising_predictor=args.predictor,
        predictor_final_steps=args.final_k,
    )
    algorithm.cfg = cfg
    base_state = algorithm.state
    # vmap-packed runs carry per-slot eta in hp_pack_inline and leave the scalar
    # hparams as NaN; taking the NaN would make every MALA proposal reject and the
    # sampler would silently return the N(0, I) it started from.
    eta = args.sample_eta
    if eta is None:
        eta = float(hparams["eta"])
        if not np.isfinite(eta):
            pack = hparams.get("hp_pack_inline")
            pack = json.loads(pack) if isinstance(pack, str) else pack
            if not (pack and "eta" in pack):
                raise ValueError(
                    f"{args.snapshot}: hparams eta is {hparams['eta']} and no hp_pack_inline "
                    "eta to fall back on; pass --sample-eta explicitly."
                )
            eta = float(pack["eta"][args.slot])
    if not np.isfinite(eta):
        raise ValueError(f"non-finite sampling eta {eta}")
    print(f"sampling eta = {eta} (slot {args.slot})", flush=True)
    set_eta(algorithm, base_state, eta)
    state = jax.tree.map(lambda x: x[args.slot], algorithm.state)

    with (args.snapshot / "replay_batches.pkl").open("rb") as f:
        import pickle
        replay = pickle.load(f)[0]
    obs = jnp.asarray(np.asarray(replay.obs)[: args.n_states])
    act_dim = int(np.asarray(replay.action).shape[-1])

    schedule = algorithm.model.schedule_for(state.hp, state.log_snr_levels)
    sigma = np.sqrt(np.asarray(schedule.one_minus_alphas_cumprod, np.float64))

    # M iid tilted draws per state, from the same chain the rollout uses.
    sampler = build_mala_sampler(
        **{**algorithm._sampler_kw,
           "denoising_predictor": args.predictor,
           "predictor_final_steps": args.final_k,
           "num_denoised_actions": args.n_draws}
    )
    agg = lambda qs: _aggregate_q(qs, cfg.q_agg_sample)
    result = jax.jit(lambda k, s, o: sampler(k, s, o, agg))(
        jax.random.key(args.seed), state, obs)
    actions = np.asarray(result.action, np.float64)          # [M, B, d]
    q_draws = np.asarray(result.q, np.float64)               # [M, B]

    def q_of(a):
        return _aggregate_q([algorithm.model.q(qp, obs, a) for qp in state.params.q],
                            cfg.q_agg_sample)

    q_fn = jax.jit(q_of)
    a0 = jnp.asarray(actions.mean(axis=0))                   # per-state posterior mean
    q0 = np.asarray(q_fn(a0), np.float64)

    # Value cost of a sigma-sized isotropic action perturbation.
    def dq_at(scale, key):
        eps = jax.random.normal(key, (args.n_noise, *a0.shape))
        qs = jax.vmap(lambda e: q_fn(jnp.clip(a0 + scale * e, -1.0, 1.0)))(eps)
        return float(np.mean(np.asarray(qs, np.float64) - q0))

    keys = jax.random.split(jax.random.key(args.seed + 1), 8)
    dq = {name: dq_at(scale, keys[i]) for i, (name, scale) in enumerate(
        (("l0", sigma[0]), ("l1", sigma[1]), ("0p05", 0.05), ("0p10", 0.10)))}

    # tr(H) by Hutchinson on the same states: E[v^T H v], v Rademacher.
    def hvp(v):
        return jax.jvp(lambda a: jax.grad(lambda aa: jnp.sum(q_fn(aa)))(a), (a0,), (v,))[1]

    hkeys = jax.random.split(jax.random.key(args.seed + 2), args.n_hutchinson)
    quads = []
    for hk in hkeys:
        v = jnp.sign(jax.random.normal(hk, a0.shape))
        quads.append(np.asarray(jnp.sum(v * hvp(v), axis=-1), np.float64))
    tr_hess = float(np.mean(quads))

    q_sd_within = float(np.mean(q_draws.std(axis=0, ddof=1)))
    q_sd_pooled = float(q_draws.std(ddof=1))
    a_sd_within = float(np.mean(actions.std(axis=0, ddof=1)))
    rel = lambda x: x / max(q_sd_within, 1e-9)
    # sigma at which the second-order cost reaches 1% of the within-state Q spread.
    sigma_1pct = float(np.sqrt(max(0.02 * q_sd_within / max(abs(tr_hess), 1e-12), 0.0)))

    row = {
        "snapshot": str(args.snapshot), "env": hparams["env"],
        "diffusion_steps": hparams["diffusion_steps"], "training_eta": hparams["eta"],
        "training_seed": hparams["seed"], "step": args.snapshot.name.rsplit("_", 1)[-1],
        "predictor": args.predictor, "final_k": args.final_k, "sample_eta": eta,
        "act_dim": act_dim, "n_states": int(obs.shape[0]), "n_draws": args.n_draws,
        "sigma_level0": float(sigma[0]), "sigma_level1": float(sigma[1]),
        "a_sd_within": a_sd_within,
        "a_sd_across": float(np.mean(actions.mean(axis=0).std(axis=0, ddof=1))),
        "a_abs_mean": float(np.mean(np.abs(actions))),
        "a_sat_frac": float(np.mean(np.abs(actions) > 0.99)),
        "q_sd_within": q_sd_within, "q_sd_pooled": q_sd_pooled,
        "q_sd_ratio": q_sd_pooled / max(q_sd_within, 1e-9),
        "q_mean": float(q_draws.mean()),
        "tr_hess": tr_hess, "tr_hess_rel": rel(tr_hess),
        "dq_sigma_l0": dq["l0"], "dq_sigma_l0_rel": rel(dq["l0"]),
        "dq_sigma_l1": dq["l1"], "dq_sigma_l1_rel": rel(dq["l1"]),
        "dq_sigma_0p05": dq["0p05"], "dq_sigma_0p05_rel": rel(dq["0p05"]),
        "dq_sigma_0p10": dq["0p10"], "dq_sigma_0p10_rel": rel(dq["0p10"]),
        "sigma_for_1pct_qsd": sigma_1pct,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    exists = args.output.exists()
    with args.output.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(json.dumps({k: row[k] for k in (
        "env", "diffusion_steps", "training_eta", "step", "final_k", "a_sd_within",
        "a_abs_mean", "a_sat_frac", "q_sd_within", "q_sd_ratio", "tr_hess_rel",
        "dq_sigma_l0_rel", "sigma_for_1pct_qsd")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
