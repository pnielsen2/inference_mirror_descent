#!/usr/bin/env python3
"""Evaluate saved MGMD states under tilted-N=1 and base-policy best-of-N protocols."""
import argparse
import csv
import json
import os
import pickle
import time
from pathlib import Path

import gymnasium
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import erf

from relax.algorithm.mgmd import MGMD
from relax.algorithm.mgmd_types import HParams, MGMDConfig
from relax.cli.train_args import build_parser
from relax.cli.train_setup import _mish
from relax.env import create_vector_env
from relax.network.actor_critic import ActorCritic


VALUES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
FIELDS = (
    "snapshot", "env", "diffusion_steps", "training_eta", "training_seed",
    "protocol", "value", "episode_index", "episode_return", "episode_length",
    "eval_seed", "elapsed_seconds",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--values", type=int, nargs="+", default=list(VALUES))
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--protocol", choices=("both", "tilted", "base"), default="both")
    parser.add_argument("--q-agg-sample", choices=("min", "mean"))
    parser.add_argument("--ddpm-mean-final-steps", type=int)
    parser.add_argument("--candidate-chunk-size", type=int, default=256)
    parser.add_argument("--dry-load", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0 or any(v <= 0 for v in args.values):
        parser.error("--episodes and every --values entry must be positive")
    if args.ddpm_mean_final_steps is not None and args.ddpm_mean_final_steps <= 0:
        parser.error("--ddpm-mean-final-steps must be positive")
    if args.candidate_chunk_size <= 0:
        parser.error("--candidate-chunk-size must be positive")
    return args


def _training_args(hparams):
    args = build_parser().parse_args([])
    for key, value in hparams.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def _repair_legacy_state(state):
    """Make a state pickled by an older revision loadable by today's samplers.

    Two rescues, both from HParams having grown since these snapshots were
    written. One older layout stored ``hp`` in the tuple slot ``distill_buffer``
    now occupies. And a NamedTuple unpickles positionally, so fields appended
    after the write are filled from the class defaults -- which are the historical
    values those runs used (h_max 0.5, acceptance target 0.574), but as scalars
    rather than the per-slot arrays every leaf needs to survive the samplers'
    ``vmap`` over seeds. Broadcast them onto the state's slot axis.
    """
    if isinstance(state.distill_buffer, HParams):
        state = state._replace(hp=state.distill_buffer, distill_buffer=None)
    state = jax.tree.map(jnp.asarray, state)
    slots = state.beta.shape[0]
    return jax.tree.map(
        lambda x: jnp.broadcast_to(x, (slots,)) if x.ndim == 0 else x, state)


def load_algorithm(snapshot: Path, q_agg_sample=None, ddpm_mean_final_steps=None):
    hparams = json.loads((snapshot / "hparams.json").read_text())
    rollout_obs = np.load(snapshot / "rollout_obs.npy")
    with (snapshot / "replay_batches.pkl").open("rb") as f:
        replay_batches = pickle.load(f)
    with (snapshot / "algorithm_state.pkl").open("rb") as f:
        state = _repair_legacy_state(pickle.load(f))

    args = _training_args(hparams)
    if q_agg_sample is not None:
        args.q_agg_sample = q_agg_sample
    if ddpm_mean_final_steps is not None:
        args.denoising_predictor = "Identity_then_DDPM_mean_final_k"
        args.ddpm_mean_final_steps = ddpm_mean_final_steps
    obs_dim = int(rollout_obs.shape[-1])
    act_dim = int(np.asarray(replay_batches[0].action).shape[-1])
    model = ActorCritic.create(
        obs_dim, act_dim, [args.hidden_dim] * args.hidden_num,
        [args.diffusion_hidden_dim] * args.hidden_num, _mish,
        num_timesteps=args.diffusion_steps,
        beta_schedule_type=args.beta_schedule_type,
        mala_steps=args.mala_steps,
        num_q_networks=args.num_q_networks,
        x_recon_clip_radius=(float("inf") if args.latent_action else 1.0),
        snr_max=args.snr_max,
        policy_parameterization=args.policy_parameterization,
        policy_final_layer=args.policy_final_layer,
        orthogonal_init=args.orthogonal_init,
        noise_cond_theta=args.noise_cond_theta,
        inference_spacing=args.inference_spacing,
        log_snr_min=args.log_snr_min,
        log_snr_max=args.log_snr_max,
        karras_rho=args.karras_rho,
    )
    params = jax.tree.map(lambda x: x[0], state.params)
    cfg = MGMDConfig.from_args(args)
    algorithm = MGMD(model, params, cfg, obs_dim=obs_dim, hidden_dim=args.hidden_dim)
    algorithm.state = state
    return algorithm, hparams


def set_sampling_eta(algorithm: MGMD, base_state, eta: float):
    eta_value = jnp.full_like(base_state.beta, eta)
    algorithm.state = base_state._replace(
        beta=eta_value,
        hp=base_state.hp._replace(
            alpha=jnp.ones_like(base_state.hp.alpha),
            T=jnp.zeros_like(base_state.hp.T),
            eta=jnp.full_like(base_state.hp.eta, eta),
        ),
    )


def select_action(algorithm: MGMD, keys, obs, candidates: int, chunk_size: int,
                  sampler_kind: str = "mala"):
    if candidates <= chunk_size:
        return algorithm.get_eval_action_vmap(keys, obs, candidates, sampler_kind=sampler_kind)
    best_action = None
    best_q = None
    remaining = candidates
    chunk_index = 0
    while remaining:
        current_size = min(chunk_size, remaining)
        chunk_keys = keys if chunk_index == 0 else jax.vmap(
            lambda key: jax.random.fold_in(key, chunk_index)
        )(keys)
        action, q = algorithm.get_eval_action_vmap(
            chunk_keys, obs, current_size, return_q=True, sampler_kind=sampler_kind
        )
        if best_q is None:
            best_action, best_q = action, q
        else:
            replace = q > best_q
            best_action = np.where(replace[..., None], action, best_action)
            best_q = np.where(replace, q, best_q)
        remaining -= current_size
        chunk_index += 1
    return best_action


def evaluate(algorithm: MGMD, env_name: str, episodes: int, candidates: int,
             eval_seed: int, latent_action: bool, candidate_chunk_size: int,
             sampler_kind: str = "mala"):
    budget = min(episodes, len(os.sched_getaffinity(0)))
    workers = max(w for w in range(1, budget + 1) if episodes % w == 0)
    seeds = np.random.default_rng(eval_seed).integers(0, 2**32 - 1, episodes).tolist()
    env, _, _ = create_vector_env(
        env_name, episodes, eval_seed, num_workers=workers, seeds_override=seeds)
    max_steps = int(gymnasium.spec(env_name).max_episode_steps or 1000)
    returns = np.zeros(episodes, np.float64)
    lengths = np.zeros(episodes, np.int64)
    active = np.ones(episodes, bool)
    key = jax.random.key(eval_seed)
    try:
        for step in range(max_steps):
            if not active.any():
                break
            obs = env.get_current_obs().reshape(1, episodes, -1)
            action = select_action(
                algorithm, jax.random.split(jax.random.fold_in(key, step), 1),
                obs, candidates, candidate_chunk_size, sampler_kind,
            )[0]
            if latent_action:
                action = erf(action / np.sqrt(2.0))
            _, reward, terminated, truncated, _ = env.step(action)
            returns += reward * active
            lengths += active
            active &= ~(terminated | truncated)
    finally:
        env.close()
    return returns, lengths


def completed_settings(path: Path, episodes: int):
    if not path.exists():
        return set()
    counts = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            key = (row["protocol"], int(row["value"]))
            counts[key] = counts.get(key, 0) + 1
    return {key for key, count in counts.items() if count >= episodes}


def append_results(path: Path, snapshot: Path, hparams: dict, protocol: str,
                   value: int, returns, lengths, eval_seed: int, elapsed: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        for episode, (ret, length) in enumerate(zip(returns, lengths)):
            writer.writerow({
                "snapshot": str(snapshot), "env": hparams["env"],
                "diffusion_steps": hparams["diffusion_steps"],
                "training_eta": hparams["eta"], "training_seed": hparams["seed"],
                "protocol": protocol, "value": value, "episode_index": episode,
                "episode_return": float(ret), "episode_length": int(length),
                "eval_seed": eval_seed, "elapsed_seconds": elapsed,
            })


def main():
    args = parse_args()
    algorithm, hparams = load_algorithm(
        args.snapshot, args.q_agg_sample, args.ddpm_mean_final_steps
    )
    print(
        f"loaded {hparams['env']} diffusion_steps={hparams['diffusion_steps']} "
        f"training_eta={hparams['eta']} seed={hparams['seed']} "
        f"q_agg_sample={algorithm.cfg.q_agg_sample} "
        f"denoising_predictor={algorithm.cfg.denoising_predictor} "
        f"ddpm_mean_final_steps={algorithm.cfg.ddpm_mean_final_steps}", flush=True)
    if args.dry_load:
        return

    base_state = algorithm.state
    done = completed_settings(args.output, args.episodes)
    settings = []
    if args.protocol in ("both", "tilted"):
        settings.extend(("tilted_eta", value, value, 1) for value in args.values)
    if args.protocol in ("both", "base"):
        settings.extend(("base_best_of_n", value, 0, value) for value in args.values)

    for protocol, value, eta, candidates in settings:
        if (protocol, value) in done:
            print(f"skip {protocol}={value}: already complete", flush=True)
            continue
        set_sampling_eta(algorithm, base_state, eta)
        started = time.monotonic()
        returns, lengths = evaluate(
            algorithm, hparams["env"], args.episodes, candidates,
            args.eval_seed, bool(hparams.get("latent_action", False)),
            args.candidate_chunk_size)
        elapsed = time.monotonic() - started
        append_results(
            args.output, args.snapshot, hparams, protocol, value,
            returns, lengths, args.eval_seed, elapsed)
        print(
            f"{protocol}={value}: mean={returns.mean():.1f} "
            f"std={returns.std():.1f} ({elapsed:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
