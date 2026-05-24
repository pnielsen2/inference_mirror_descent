"""Master-seed -> per-site RNG derivation for ``scripts/train_mujoco.py``.

A single integer master seed is expanded into the five independent RNG sites
needed downstream: env, env-action, buffer, network init, and training. When an
``hp_pack`` supplies per-vmap-entry master seeds, every per-seed site is
re-derived from its entry's master so packed multi-seed runs reproduce the
corresponding standalone ``--seed S_i`` runs exactly.
"""
from dataclasses import dataclass
from typing import List, Optional

import jax
import jax.numpy as jnp

from relax.utils.random_utils import seeding


def _derive_seeds(master: int):
    """Map a master int seed to the tuple (env_seed, env_action_seed,
    buffer_seed, init_network_seed, train_seed)."""
    rng, _ = seeding(int(master))
    return tuple(int(x) for x in rng.integers(0, 2**32 - 1, 5))


@dataclass
class SeedBundle:
    """All per-seed RNG state needed downstream.

    ``per_entry_env_seeds`` / ``per_entry_action_seeds`` are populated only
    when an hp_pack supplies per-vmap-entry master seeds; in that case
    create_vector_env reproduces the env/action RNGs that N standalone
    --seed S_i runs would have used. Otherwise they are None and the
    standalone env_seed / env_action_seed pair drives the VectorEnv.
    """
    env_seed: int
    env_action_seed: int
    per_entry_env_seeds: Optional[List[int]]
    per_entry_action_seeds: Optional[List[int]]
    buffer_seeds: List[int]      # length N_seeds
    init_keys: jax.Array         # [N_seeds] PRNG keys
    train_keys: jax.Array        # [N_seeds] PRNG keys
    per_entry_masters: Optional[List[int]]   # for logging


def derive_seed_bundle(master_seed: int, N_seeds: int,
                       hp_pack: Optional[dict]) -> SeedBundle:
    """Derive every RNG site (env / action / buffer / network init /
    training key) from ``master_seed``, plus optional per-vmap-entry
    masters from ``hp_pack["seed"]``. When per-entry masters are present,
    every seed site is derived from its entry's master so the pack matches
    standalone --seed S_i runs exactly.
    """
    env_seed, env_action_seed, buffer_seed, init_network_seed, train_seed = _derive_seeds(master_seed)

    per_entry_masters = None
    if hp_pack is not None and "seed" in hp_pack:
        per_entry_masters = [int(s) for s in hp_pack["seed"]]
        if len(per_entry_masters) != N_seeds:
            raise ValueError(
                f"--hp_pack 'seed' has length {len(per_entry_masters)}; "
                f"expected {N_seeds} (= --parallel_seeds)"
            )

    if per_entry_masters is not None:
        derived = [_derive_seeds(m) for m in per_entry_masters]
        per_entry_env_seeds = [t[0] for t in derived]
        per_entry_action_seeds = [t[1] for t in derived]
        buffer_seeds = [t[2] for t in derived]
        init_keys = jnp.stack([jax.random.key(t[3]) for t in derived])
        train_keys = jnp.stack([jax.random.key(t[4]) for t in derived])
    else:
        per_entry_env_seeds = None
        per_entry_action_seeds = None
        buffer_seeds = [buffer_seed + i for i in range(N_seeds)]
        init_keys = jax.random.split(jax.random.key(init_network_seed), N_seeds)
        train_keys = jax.random.split(jax.random.key(train_seed), N_seeds)

    return SeedBundle(
        env_seed=env_seed,
        env_action_seed=env_action_seed,
        per_entry_env_seeds=per_entry_env_seeds,
        per_entry_action_seeds=per_entry_action_seeds,
        buffer_seeds=buffer_seeds,
        init_keys=init_keys,
        train_keys=train_keys,
        per_entry_masters=per_entry_masters,
    )
