"""Per-seed hyperparameter pack overrides for the vmapped DPMD train state.

A "hp_pack" is the per-vmap-slot override dict produced by
``scripts/launch.py`` (from the ``--ablate`` "easy" axes plus ``--seeds``)
and forwarded into ``scripts/train_mujoco.py`` via ``--hp_pack_inline``.
Each key is an argparse attribute name; each value is a length-N list giving
that hp's value for vmap slot ``s in [0, N)``.

This module owns:

* ``CLI_TO_FIELD`` -- the rename map from argparse attribute names to
  ``Diffv2TrainState`` field names. Keys not in the map share their argparse
  name with the state field (``lr_q``, ``shape_ema_tau``, ``reward_scale``,
  ...).
* ``ALLOWED_KEYS`` -- the set of argparse attribute names that are legal in
  a hp_pack. Pure validation surface; mismatches raise loudly.
* ``apply(state, hp_pack, N_seeds)`` -- pure function that returns a new
  ``Diffv2TrainState`` with the per-slot overrides applied. When
  ``kl_budget`` is overridden, ``tfg_eta`` is automatically rederived as
  ``sqrt(2 * kl_budget)`` so the two stay consistent.
"""
import jax.numpy as jnp


# argparse attribute name -> Diffv2TrainState field name. Identity mapping
# is implicit (e.g. "lr_q" stays "lr_q").
CLI_TO_FIELD = {
    "tau": "polyak_tau",
    "advantage_ema_tau": "adv_ema_tau",
    "guidance_strength_multiplier": "guidance_mult",
    "kl_budget": "kl_budget_val",
    "initial_advantage_second_moment_ema": "advantage_second_moment_ema",
    "initial_dist_shift_shape_ema": "dist_shift_shape_ema",
    "tfg_eta": "tfg_eta",
}


ALLOWED_KEYS = {
    "lr_q", "lr_policy", "gamma", "tau", "advantage_ema_tau",
    "guidance_strength_multiplier", "shape_ema_tau", "tfg_eta", "kl_budget",
    "initial_advantage_second_moment_ema", "initial_dist_shift_shape_ema",
    "reward_scale", "x0_hat_clip_radius", "mala_adapt_rate",
    "q_td_huber_width",
    "seed",
}


def apply(state, hp_pack: dict, N_seeds: int):
    """Return ``state`` with per-seed hp_pack overrides applied.

    ``hp_pack`` keys are argparse attribute names; values are length-N lists.
    The ``"seed"`` key is consumed earlier (in ``derive_seed_bundle``) and
    skipped here. When ``kl_budget`` is overridden, ``tfg_eta`` is
    automatically rederived as ``sqrt(2 * kl_budget)`` so the two stay
    consistent.
    """
    # Fields on the top-level ``Diffv2TrainState``; all other override targets
    # live inside ``state.hp``.
    _TOPLEVEL_TARGETS = {"tfg_eta", "advantage_second_moment_ema", "dist_shift_shape_ema"}
    hp_overrides = {}
    top_overrides = {}
    for k, v in hp_pack.items():
        if k not in ALLOWED_KEYS:
            raise ValueError(
                f"--hp_pack key '{k}' is not a per-seed vmappable hp. "
                f"Allowed: {sorted(ALLOWED_KEYS)}"
            )
        if k == "seed":
            continue
        arr = jnp.asarray(v, dtype=jnp.float32)
        if arr.shape != (N_seeds,):
            raise ValueError(f"--hp_pack '{k}' has shape {arr.shape}; expected ({N_seeds},)")
        target = CLI_TO_FIELD.get(k, k)
        (top_overrides if target in _TOPLEVEL_TARGETS else hp_overrides)[target] = arr
    if not hp_overrides and not top_overrides:
        return state
    if hp_overrides:
        state = state._replace(hp=state.hp._replace(**hp_overrides))
    if top_overrides:
        state = state._replace(**top_overrides)
    if "kl_budget_val" in hp_overrides:
        kl_budget_v = jnp.asarray(state.hp.kl_budget_val, dtype=jnp.float32)
        state = state._replace(
            tfg_eta=jnp.sqrt(jnp.maximum(jnp.float32(0.0), jnp.float32(2.0) * kl_budget_v))
        )
    applied = list(hp_overrides.keys()) + list(top_overrides.keys())
    print(f"[hp_pack] applied per-seed overrides: {applied}")
    return state
