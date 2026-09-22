#!/usr/bin/env python3
"""Sensitivity of the critic Q to small action perturbations, from snapshots.

For each diagnostic snapshot we

1. roll out ``--episodes`` episodes with the snapshot's own tilted policy,
2. sample ``--actions`` random visited ``(s, a)`` pairs from those episodes,
3. form the action-space Hessian ``H = d^2 Q / da^2`` at each pair,
4. replace its spectrum by absolute values, ``H~ = V |Lambda| V^T``, so that
   positive and negative curvature cannot cancel, and
5. report the expected inflation of the resulting quadratic model under an
   isotropic action perturbation of scale ``eps``.

Because ``delta ~ N(0, eps^2 I)`` has ``E[delta] = 0`` and
``E[delta delta^T] = eps^2 I``, the linear term drops out and the quantity has
the closed form

    S(s, a) = E_delta[ Qhat(a + delta) ] - Q(a)
            = 0.5 * eps^2 * tr(H~)
            = 0.5 * eps^2 * sum_i |lambda_i|

so no Monte-Carlo over ``delta`` is required.

``eps`` defaults to the standard deviation of the cleanest level of the run's
own noise schedule, ``sqrt(1 - abar_1)``.

Example
-------
    python -m scripts.snapshot_q_hessian_sensitivity \
        --steps 100000 500000 1000000 --q-agg mean --slot 0 \
        --episodes 10 --actions 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.special import erf

from relax.algorithm.mgmd import _aggregate_q
from relax.env import create_vector_env
from relax.utils.diffusion import BetaScheduleCoefficients
from scripts.evaluate_snapshot_action_selection import load_algorithm

SNAP_ROOT = Path("/n/holylabs/kdbrantley_lab/Lab/pnielsen/diagnostic_snapshots")
OUT_CSV = Path("analysis_cache/sweep225_q_hessian_sensitivity.csv")


# ----------------------------------------------------------------------------
# snapshot discovery
# ----------------------------------------------------------------------------
def discover(root: Path, prefix: str, steps, q_agg: str | None):
    """Return one record per (snapshot dir, step) matching the filters."""
    out = []
    for d in sorted(root.glob(f"{prefix}*")):
        for step in steps:
            hits = sorted(d.glob(f"*_step_{step}"))
            if not hits:
                continue
            snap = hits[0]
            hp_path = snap / "hparams.json"
            if not hp_path.exists():
                continue
            hp = json.loads(hp_path.read_text())
            if q_agg is not None and hp.get("q_agg_sample") != q_agg:
                continue
            out.append({
                "snapshot": snap,
                "env": hp["env"],
                "step": int(step),
                "seed": hp.get("seed"),
                "q_agg_sample": hp.get("q_agg_sample"),
                "job_dir": d.name,
            })
    return out


def slice_slot(state, i: int):
    """Keep vmap slot ``i`` while preserving the leading run axis (size 1)."""
    return jax.tree.map(lambda x: jnp.asarray(x)[i:i + 1], state)


def read_q_loss_norm(snapshot: Path, slot: int) -> float:
    """The sampler's Q-loss standardizer at this snapshot.

    Mirrors ``Critic/q_loss_norm`` in :mod:`relax.algorithm.mgmd`:
    ``sqrt(max(advantage_second_moment_ema, 1e-6))``, where that EMA slot holds
    an exponential average of the critic TD loss under
    ``--q_loss_normalization``. Being an RMS TD error it carries the units of
    ``Q``, so dividing a Q-difference by it gives a scale-free number.
    """
    import pickle
    with (snapshot / "algorithm_state.pkl").open("rb") as f:
        state = pickle.load(f)
    m2 = np.asarray(state.advantage_second_moment_ema, dtype=np.float64)
    return float(np.sqrt(np.maximum(m2[slot], 1e-6)))


def schedule_epsilon(hparams: dict) -> float:
    """sqrt(1 - abar_1): the forward-noise std at the cleanest level."""
    n = int(hparams["diffusion_steps"])
    kind = hparams.get("beta_schedule_type", "cosine")
    if kind == "cosine":
        betas = BetaScheduleCoefficients.cosine_beta_schedule(n)
    elif kind == "linear":
        betas = BetaScheduleCoefficients.linear_beta_schedule(n)
    else:
        raise SystemExit(f"epsilon not implemented for beta_schedule_type={kind}; "
                         f"pass --epsilon explicitly")
    abar = np.cumprod(1.0 - np.asarray(betas, np.float64))
    return float(np.sqrt(1.0 - abar[0]))


# ----------------------------------------------------------------------------
# rollouts
# ----------------------------------------------------------------------------
def rollout_pairs(alg, env_name: str, episodes: int, eval_seed: int,
                  latent_action: bool, max_steps: int | None = None):
    """Run ``episodes`` episodes in parallel; return visited (obs, action) pairs.

    Actions are the raw policy actions (pre-squash), i.e. exactly what the
    critic is evaluated on during training.
    """
    import os
    budget = min(episodes, len(os.sched_getaffinity(0)))
    workers = max(w for w in range(1, budget + 1) if episodes % w == 0)
    seeds = np.random.default_rng(eval_seed).integers(0, 2**32 - 1, episodes).tolist()
    env, _, _ = create_vector_env(env_name, episodes, eval_seed,
                                  num_workers=workers, seeds_override=seeds)
    cap = int(gymnasium.spec(env_name).max_episode_steps or 1000)
    if max_steps is not None:
        cap = min(cap, max_steps)

    active = np.ones(episodes, bool)
    returns = np.zeros(episodes, np.float64)
    obs_list, act_list = [], []
    key = jax.random.key(eval_seed)
    try:
        for step in range(cap):
            if not active.any():
                break
            obs = env.get_current_obs().reshape(1, episodes, -1)
            keys = jax.random.split(jax.random.fold_in(key, step), 1)
            action = np.asarray(alg.get_eval_action_vmap(keys, obs, 1))[0]
            obs_list.append(np.asarray(obs[0])[active])
            act_list.append(action[active])
            env_action = erf(action / np.sqrt(2.0)) if latent_action else action
            _, reward, terminated, truncated, _ = env.step(env_action)
            returns += reward * active
            active &= ~(terminated | truncated)
    finally:
        env.close()
    return np.concatenate(obs_list, 0), np.concatenate(act_list, 0), returns


# ----------------------------------------------------------------------------
# curvature
# ----------------------------------------------------------------------------
def build_hessian_fn(alg):
    """Return jitted fns for Q(s,a) and its action-space Hessian, slot 0."""
    qps = [jax.tree.map(lambda x: x[0], qp) for qp in alg.state.params.q]
    agg = alg.cfg.q_agg_sample

    def q_scalar(a, s):
        qs = [alg.model.q(qp, s[None, :], a[None, :])[0] for qp in qps]
        return _aggregate_q(qs, agg)

    hess = jax.jit(jax.hessian(q_scalar))
    qval = jax.jit(q_scalar)
    return qval, hess


def curvature_rows(alg, obs, act, eps: float):
    qval, hess = build_hessian_fn(alg)
    rows = []
    for i in range(obs.shape[0]):
        s = jnp.asarray(obs[i])
        a = jnp.asarray(act[i])
        H = np.asarray(hess(a, s), dtype=np.float64)
        H = 0.5 * (H + H.T)                      # enforce exact symmetry
        lam = np.linalg.eigvalsh(H)
        abs_lam = np.abs(lam)
        rows.append({
            "q": float(qval(a, s)),
            "sum_abs_eig": float(abs_lam.sum()),
            "max_abs_eig": float(abs_lam.max()),
            "trace_raw": float(lam.sum()),
            "n_negative_eig": int((lam < 0).sum()),
            "act_dim": int(lam.size),
            "sensitivity": float(0.5 * eps**2 * abs_lam.sum()),
        })
    return rows


# ----------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=SNAP_ROOT)
    p.add_argument("--prefix", default="mgmd_2026-09-02_21-",
                   help="snapshot dir prefix identifying the sweep")
    p.add_argument("--steps", type=int, nargs="+",
                   default=[100000, 500000, 1000000])
    p.add_argument("--q-agg", default="mean")
    p.add_argument("--slot", type=int, default=0, help="vmap seed slot to use")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--actions", type=int, default=30)
    p.add_argument("--eval-seed", type=int, default=0)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--epsilon", type=float, default=None,
                   help="override; default sqrt(1-abar_1) of the run's schedule")
    p.add_argument("--max-steps", type=int, default=None,
                   help="cap env steps per episode (smoke tests)")
    p.add_argument("--out", type=Path, default=OUT_CSV)
    args = p.parse_args()

    jobs = discover(args.root, args.prefix, args.steps, args.q_agg)
    if not jobs:
        raise SystemExit("no snapshots matched")
    print(f"{len(jobs)} snapshots matched "
          f"({len({j['env'] for j in jobs})} envs x {len(args.steps)} steps)")

    rng = np.random.default_rng(args.sample_seed)
    all_rows = []
    for n, job in enumerate(jobs, 1):
        t0 = time.time()
        alg, hp = load_algorithm(job["snapshot"])
        alg.state = slice_slot(alg.state, args.slot)
        eps = args.epsilon if args.epsilon is not None else schedule_epsilon(hp)
        latent = bool(hp.get("latent_action", False))

        obs, act, returns = rollout_pairs(
            alg, job["env"], args.episodes, args.eval_seed, latent, args.max_steps)
        if obs.shape[0] < args.actions:
            raise SystemExit(f"only {obs.shape[0]} visited pairs for {job['env']}")
        pick = rng.choice(obs.shape[0], size=args.actions, replace=False)

        q_loss_norm = read_q_loss_norm(job["snapshot"], args.slot)
        rows = curvature_rows(alg, obs[pick], act[pick], eps)
        for r in rows:
            r.update(env=job["env"], step=job["step"], seed=job["seed"],
                     q_agg_sample=job["q_agg_sample"], epsilon=eps,
                     q_loss_norm=q_loss_norm,
                     episode_return_mean=float(returns.mean()),
                     n_visited=int(obs.shape[0]))
        all_rows.extend(rows)

        sens = np.array([r["sensitivity"] for r in rows])
        print(f"[{n:2d}/{len(jobs)}] {job['env']:<15s} step={job['step']:>8,d} "
              f"eps={eps:.6f} pairs={obs.shape[0]:>5d} "
              f"ret={returns.mean():9.1f} "
              f"S median={np.median(sens):.4g} "
              f"S/sigma_Q={np.median(sens)/q_loss_norm:.4g} "
              f"({time.time()-t0:.0f}s)")

    df = pd.DataFrame(all_rows)
    args.out.parent.mkdir(exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
