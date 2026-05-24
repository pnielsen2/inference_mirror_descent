"""Setup helpers for ``scripts/train_mujoco.py``.

Pulled out of the entry-point so a walkthrough reader lands on the main
script and sees the algorithmic flow (parse args → build env → build
model → run trainer) without paging past per-seed init loops or
KL-budget plumbing.
"""
from typing import List, Tuple

import jax, jax.numpy as jnp

from relax.buffer import TreeBuffer
from relax.network.actor_critic import ActorCritic, ActorCriticParams
from relax.utils.seeding import SeedBundle


def resolve_kl_budget(args, act_dim: int) -> None:
    """Mutate ``args`` in place to resolve the composite MD guidance parameters.

    Resolution order (later steps override earlier ones):

    * ``--kl_budget_per_dim`` is converted to a total ``args.kl_budget``
      (multiplied by ``act_dim``).
    * ``--T`` derives ``args.beta = (1 - args.alpha) / T``.
    * When a KL budget is set (either flag), ``beta`` is overridden as
      the initial KL-budget coefficient ``sqrt(2 * δ / M_0)`` with
      ``M_0 = args.initial_advantage_second_moment_ema``; the V-network /
      on-policy advantage EMA path is enabled downstream (MGMD reads
      ``cfg.kl_budget is not None``).
    * If ``beta`` is still ``None`` after the above (neither ``--beta``,
      ``--T``, nor ``--kl_budget`` was supplied), it defaults to ``0.0``.

    Mutual-exclusion checks between ``--kl_budget`` / ``--kl_budget_per_dim``
    and between ``--beta`` / ``--T`` are enforced upstream in
    :func:`relax.cli.train_args.validate_args`.
    """
    if args.kl_budget_per_dim is not None:
        args.kl_budget = args.kl_budget_per_dim * act_dim
    if args.T is not None:
        args.beta = (1.0 - args.alpha) / args.T
    if args.kl_budget is not None:
        args.beta = float((2.0 * args.kl_budget / max(args.initial_advantage_second_moment_ema, 1e-6)) ** 0.5)
    if args.beta is None:
        args.beta = 0.0


def _mish(x: jax.Array) -> jax.Array:
    return x * jnp.tanh(jax.nn.softplus(x))


def build_per_seed_state(
    args,
    seeds: SeedBundle,
    obs_dim: int,
    act_dim: int,
) -> Tuple[ActorCritic, List[ActorCriticParams], List[TreeBuffer]]:
    """Construct ``N_seeds`` independent (model, params, replay-buffer) tuples.

    All N copies of the model share the same architecture and the same
    static fields; only the haiku params and the buffer's RNG state differ
    across seeds. The first model instance is returned for downstream code
    that needs a single ``ActorCritic`` handle (the haiku transforms are
    pure functions of params, so it's fine to share the model and vmap
    over the per-seed param pytrees).
    """
    N_seeds = len(seeds.init_keys)
    hidden_sizes = [args.hidden_dim] * args.hidden_num
    diffusion_hidden_sizes = [args.diffusion_hidden_dim] * args.hidden_num

    buffers_list = [
        TreeBuffer.from_experience(
            obs_dim, act_dim, size=args.buffer_size,
            seed=seeds.buffer_seeds[i],
        )
        for i in range(N_seeds)
    ]

    model = ActorCritic.create(
        obs_dim,
        act_dim,
        hidden_sizes,
        diffusion_hidden_sizes,
        _mish,
        num_timesteps=args.diffusion_steps,
        beta_schedule_type=args.beta_schedule_type,
        mala_steps=args.mala_steps,
        num_q_networks=args.num_q_networks,
        x_recon_clip_radius=1.0,
        snr_max=args.snr_max,
        policy_parameterization=args.policy_parameterization,
        policy_final_layer=args.policy_final_layer,
    )
    params_list = [model.init_params(k) for k in seeds.init_keys]
    return model, params_list, buffers_list
