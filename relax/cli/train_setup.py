"""Setup helpers for ``scripts/train_mujoco.py``.

Pulled out of the entry-point so a walkthrough reader lands on the main
script and sees the algorithmic flow (parse args → build env → build
model → run trainer) without paging past per-seed init loops or
KL-budget plumbing.
"""
import json
from typing import List, Tuple

import jax, jax.numpy as jnp

from relax.buffer import TreeBuffer
from relax.network.actor_critic import ActorCritic, ActorCriticParams
from relax.utils.seeding import SeedBundle


def resolve_cmd_params(alpha, beta, T, eta):
    """Resolve the four CMD parameters from exactly two specified values.

    Supports the well-defined ``T=0`` limit cases, e.g. ``(T, eta)`` gives
    ``alpha=1`` and ``beta=eta``.
    """
    provided = [name for name, value in (("alpha", alpha), ("beta", beta), ("T", T), ("eta", eta)) if value is not None]
    if len(provided) != 2:
        raise ValueError(f"Exactly two of alpha, beta, T, and eta must be specified, got {provided}.")

    if alpha is not None and beta is not None:
        if beta == 0.0:
            raise ValueError("Cannot derive T and eta from --alpha and --beta when beta is 0.")
        T = (1.0 - alpha) / beta
        eta = beta / alpha
    elif alpha is not None and T is not None:
        if T == 0.0:
            raise ValueError("Cannot derive beta and eta from --alpha and --T when T is 0.")
        beta = (1.0 - alpha) / T
        eta = beta / alpha
    elif alpha is not None and eta is not None:
        if eta == 0.0:
            raise ValueError("Cannot derive T from --alpha and --eta when eta is 0.")
        T = (1.0 / alpha - 1.0) / eta
        beta = eta * alpha
    elif beta is not None and T is not None:
        alpha = 1.0 - beta * T
        eta = beta / alpha
    elif beta is not None and eta is not None:
        if beta == 0.0 or eta == 0.0:
            raise ValueError("Cannot derive a unique CMD update from --beta and --eta when either is 0.")
        alpha = beta / eta
        T = (eta - beta) / (beta * eta)
    elif T is not None and eta is not None:
        alpha = 1.0 / (1.0 + eta * T)
        beta = eta / (1.0 + eta * T)

    if alpha <= 0.0:
        raise ValueError(f"Resolved alpha must be > 0, got {alpha}.")
    if beta < 0.0:
        raise ValueError(f"Resolved beta must be >= 0, got {beta}.")
    if T < 0.0:
        raise ValueError(f"Resolved T must be >= 0, got {T}.")
    if eta < 0.0:
        raise ValueError(f"Resolved eta must be >= 0, got {eta}.")

    return float(alpha), float(beta), float(T), float(eta)


def resolve_kl_budget(args, act_dim: int) -> None:
    """Mutate ``args`` in place to resolve composite-MD parameters.

    Exactly two of ``alpha``, ``beta``, ``T``, and ``eta`` are specified on the
    CLI. This fills in all four using

        alpha = 1 / (1 + eta * T)
        beta = eta / (1 + eta * T) = (1 - alpha) / T

    The algorithm consumes the resolved ``alpha`` and ``beta``. If a KL budget
    is enabled, it still overrides the initial ``beta`` after this deterministic
    CMD parameterization step.
    """
    if args.kl_budget_per_dim is not None:
        args.kl_budget = args.kl_budget_per_dim * act_dim

    alpha = args.alpha
    beta = args.beta
    T = args.T
    eta = args.eta

    specified = [value for value in (alpha, beta, T, eta) if value is not None]
    if len(specified) == 2:
        args.alpha, args.beta, args.T, args.eta = resolve_cmd_params(alpha, beta, T, eta)
    elif args.hp_pack_inline is not None and len(specified) < 2:
        hp_pack = json.loads(args.hp_pack_inline)
        cmd_pack_keys = {name for name in ("alpha", "beta", "T", "eta") if name in hp_pack}
        base_cmd_keys = [name for name, value in (("alpha", alpha), ("beta", beta), ("T", T), ("eta", eta)) if value is not None]
        if len(specified) + len(cmd_pack_keys) < 2:
            raise ValueError(
                "Base CLI plus --hp_pack_inline must specify at least two of alpha, beta, T, and eta. "
                f"Base provided {base_cmd_keys}, "
                f"hp_pack provides {sorted(cmd_pack_keys)}."
            )
        args.alpha = float("nan") if alpha is None else float(alpha)
        args.beta = float("nan") if beta is None else float(beta)
        args.T = float("nan") if T is None else float(T)
        args.eta = float("nan") if eta is None else float(eta)
    else:
        raise ValueError("Exactly two of alpha, beta, T, and eta must be specified.")

    if args.kl_budget is not None:
        args.beta = float((2.0 * args.kl_budget / max(args.initial_advantage_second_moment_ema, 1e-6)) ** 0.5)


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
