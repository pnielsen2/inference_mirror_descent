"""Setup helpers for ``scripts/train_mujoco.py``.

Pulled out of the entry-point so a walkthrough reader lands on the main
script and sees the algorithmic flow (parse args → build env → build
agent → run trainer) without paging past per-seed init loops or
KL-budget plumbing.
"""
from typing import List, Tuple

import jax, jax.numpy as jnp

from relax.buffer import TreeBuffer
from relax.network.diffv2 import Diffv2Net, Diffv2Params, create_diffv2_net
from relax.utils.seeding import SeedBundle


def resolve_kl_budget(args, act_dim: int) -> None:
    """Mutate ``args`` in place to apply the KL-budget → η / V-net promotion.

    * ``--kl_budget_per_dim`` is converted to a total ``args.kl_budget``
      (multiplied by ``act_dim``).
    * When a KL budget is set (either flag), ``critic_normalization`` is
      forced to ``'ema'`` (V-network is trained) and ``tfg_eta`` is
      derived as ``sqrt(2 * δ)``.

    The mutual-exclusion check between ``--kl_budget`` and
    ``--kl_budget_per_dim`` is enforced upstream in
    :func:`scripts._train_args.validate_args`.
    """
    if args.kl_budget_per_dim is not None:
        args.kl_budget = args.kl_budget_per_dim * act_dim
    if args.kl_budget is not None:
        args.critic_normalization = "ema"
        args.tfg_eta = float((2.0 * args.kl_budget) ** 0.5)


def _mish(x: jax.Array) -> jax.Array:
    return x * jnp.tanh(jax.nn.softplus(x))


def build_per_seed_state(
    args,
    seeds: SeedBundle,
    obs_dim: int,
    act_dim: int,
) -> Tuple[Diffv2Net, List[Diffv2Params], List[TreeBuffer]]:
    """Construct ``N_seeds`` independent (agent, params, replay-buffer) tuples.

    All N copies of the agent share the same architecture and the same
    static fields; only the haiku params and the buffer's RNG state differ
    across seeds. The first agent instance is returned for downstream code
    that needs a single ``Diffv2Net`` handle (the haiku transforms are
    pure functions of params, so it's fine to share the agent and vmap
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

    def _make_diffv2(net_key):
        return create_diffv2_net(
            net_key,
            obs_dim,
            act_dim,
            hidden_sizes,
            diffusion_hidden_sizes,
            _mish,
            num_timesteps=args.diffusion_steps,
            beta_schedule_scale=args.beta_schedule_scale,
            beta_schedule_type=args.beta_schedule_type,
            mala_steps=args.mala_steps,
            num_q_networks=args.num_q_networks,
            x_recon_clip_radius=1.0,
            snr_max=args.snr_max,
        )

    pairs = [_make_diffv2(k) for k in seeds.init_keys]
    agent = pairs[0][0]
    params_list = [p for (_a, p) in pairs]
    return agent, params_list, buffers_list
