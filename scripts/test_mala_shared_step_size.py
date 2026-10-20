#!/usr/bin/env python3
"""Regression tests for ``--mala_step_size_mode``.

    JAX_PLATFORMS=cpu python scripts/test_mala_shared_step_size.py

Both modes parameterize the MALA step size the same way, ``h_t = c (1 - alpha_t)``
with ``1 - alpha_t = schedule.betas[t]``, and differ only in how many ``c``'s
there are and what drives each. The checks below pin that claim from both ends:

  1. ``per_level`` (the default) is untouched: with one MALA step per level each
     level's coefficient moves by ``rate * (accept_t - target)`` on its own
     acceptance, exactly as before the flag existed.
  2. ``shared`` carries ONE coefficient: every entry of the returned ``[T]``
     array is equal, and equals ``rate * (mean_t accept_t - target)`` from the
     zero init -- i.e. one Robbins-Monro step per pass off the level-average.
  3. The parameterization really is shared: at ``--mala_adapt_rate 0`` (nothing
     adapts) the two modes produce bit-identical actions, both at the default
     ``c = 1`` and at a state carrying ``c = 0.5``, so ``shared`` computes the
     same ``c (1 - alpha_t)`` the per-level rule does.
  4. ``shared`` collapses a state adapted per level onto its geometric mean
     rather than reading one arbitrary level.
  5. ``--mala_steps 0`` leaves the shared coefficient alone: there are no
     acceptances to average, so nothing may walk it down.
  6. The coefficient is adapted per vmap slot against that slot's own
     ``--mala_target_acceptance_rate``, which is what makes the target rate an
     "easy" (hp_pack-packed) ablation axis under ``shared``.
"""
import jax
import jax.numpy as jnp
import numpy as np

from relax.algorithm.mala_sampler import build_mala_sampler
from relax.algorithm.mgmd import MGMD
from relax.algorithm.mgmd_types import MGMDConfig
from relax.network.actor_critic import ActorCritic

OBS_DIM, ACT_DIM, BATCH, T = 5, 3, 16, 4
TARGET, RATE = 0.7, 0.2
KEY = jax.random.PRNGKey(3)
AGG = lambda qs: qs[0]


def build(mode, adapt_rate=RATE, mala_steps=1):
    model = ActorCritic.create(OBS_DIM, ACT_DIM, [32, 32], [32, 32],
                               num_timesteps=T, beta_schedule_type="cosine",
                               mala_steps=mala_steps)
    cfg = MGMDConfig(alpha=1.0, beta=64.0, T=0.0, eta=64.0, batch_size=BATCH,
                     mala_step_size_mode=mode, mala_adapt_rate=adapt_rate,
                     mala_target_acceptance_rate=TARGET,
                     ema_advantage_normalization=True,
                     guidance_snr_anneal="abar", x0_hat_clip_radius=3.0,
                     q_agg_sample="mean", delay_policy_update=1)
    algo = MGMD(model, model.init_params(jax.random.PRNGKey(0)), cfg,
                obs_dim=OBS_DIM, hidden_dim=32)
    return algo


def fake_obs(seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(BATCH, OBS_DIM)), jnp.float32)


def run(algo, state=None):
    sampler = build_mala_sampler(**algo._sampler_kw)
    return sampler(KEY, algo.state if state is None else state, fake_obs(), AGG)


def main():
    # ---- 1. per_level: one coefficient per level, on its own acceptance ----
    per_level = run(build("per_level"))
    expected = RATE * (np.asarray(per_level.per_level_acc) - TARGET)
    assert np.allclose(np.asarray(per_level.log_eta_scales), expected, atol=1e-6), \
        f"per_level scales {per_level.log_eta_scales} != {expected}"
    assert np.ptp(np.asarray(per_level.log_eta_scales)) > 0, \
        "per_level levels moved identically; the test is not discriminating"
    print("1. per_level adapts one coefficient per level                   OK")

    # ---- 2. shared: one coefficient, off the level-averaged acceptance ----
    shared = run(build("shared"))
    scales = np.asarray(shared.log_eta_scales)
    assert np.ptp(scales) == 0.0, f"shared scales are not all equal: {scales}"
    want = RATE * (float(np.mean(np.asarray(shared.per_level_acc))) - TARGET)
    assert abs(float(scales[0]) - want) < 1e-6, f"shared c {scales[0]} != {want}"
    print("2. shared adapts one coefficient off the level mean             OK")

    # ---- 3. same h_t = c (1 - alpha_t) in both modes --------------------
    for label, log_c in (("c = 1 (init)", 0.0), ("c = 0.5", float(np.log(0.5)))):
        frozen = {}
        for mode in ("per_level", "shared"):
            algo = build(mode, adapt_rate=0.0)
            state = algo.state._replace(
                log_eta_scales=jnp.full((T,), jnp.float32(log_c)))
            frozen[mode] = run(algo, state).action
        delta = float(jnp.max(jnp.abs(frozen["per_level"] - frozen["shared"])))
        assert delta == 0.0, f"{label}: modes disagree on the step size ({delta:.3e})"
        print(f"3. adapt_rate 0, {label:12s}: modes bit-identical            OK")

    # ---- 4. entering shared from a per-level state ----------------------
    algo = build("shared", adapt_rate=0.0)
    mixed = jnp.asarray(np.log([0.25, 0.5, 1.0, 2.0]), jnp.float32)
    out = run(algo, algo.state._replace(log_eta_scales=mixed))
    got = np.asarray(out.log_eta_scales)
    assert np.ptp(got) == 0.0 and abs(float(got[0]) - float(jnp.mean(mixed))) < 1e-6, \
        f"shared did not enter at the geometric mean: {got}"
    print("4. shared enters a per-level state at its geometric mean        OK")

    # ---- 5. no MALA steps => no acceptances => no update ----------------
    none = run(build("shared", mala_steps=0))
    assert float(jnp.max(jnp.abs(none.log_eta_scales))) == 0.0, \
        f"mala_steps=0 moved the shared coefficient: {none.log_eta_scales}"
    print("5. mala_steps 0 leaves the shared coefficient at its init       OK")

    # ---- 6. one coefficient per vmap slot, on that slot's own target ----
    algo = build("shared")
    targets = np.array([0.2, 0.9], dtype=np.float32)
    state = algo.make_vmapped_state([algo.state.params, algo.state.params])
    state = state._replace(hp=state.hp._replace(
        mala_target_acceptance_rate=jnp.asarray(targets)))
    sampler = build_mala_sampler(**algo._sampler_kw)
    out = jax.vmap(lambda k, s, o: sampler(k, s, o, AGG))(
        jax.random.split(KEY, 2), state,
        jnp.broadcast_to(fake_obs(), (2, BATCH, OBS_DIM)))
    scales, acc = np.asarray(out.log_eta_scales), np.asarray(out.per_level_acc)
    for i, target in enumerate(targets):
        assert np.ptp(scales[i]) == 0.0, f"slot {i} scales differ: {scales[i]}"
        want = RATE * (float(np.mean(acc[i])) - float(target))
        assert abs(float(scales[i, 0]) - want) < 1e-6, \
            f"slot {i} (target {target}): c {scales[i, 0]} != {want}"
    assert scales[0, 0] != scales[1, 0], \
        "both slots landed on the same c; the per-slot target did not bind"
    print("6. the shared coefficient is per-slot, on the slot's target     OK")

    print("\nall checks passed")


if __name__ == "__main__":
    main()
