#!/usr/bin/env python3
"""Regression tests for the DDIM_unguided denoising predictor.

    JAX_PLATFORMS=cpu python scripts/test_ddim_unguided.py

Checks:
  1. Every predictor still runs and returns finite actions (the e_grad
     plumbing did not break the existing three).
  2. DDIM_unguided is bit-identical across guidance_strength_multiplier at a
     fixed guidance_gradient_space -- the predictor uses no guidance at all.
     (It is *not* invariant to the grad space: the MALA chain itself uses it.)
  3. DDIM_unguided costs no more FLOPs than Identity: the eps it consumes is
     the gradient the last MALA step already computed. This is also the
     decisive check that it is unguided -- a guided predictor has to evaluate
     grad_a Q, which shows up as a large FLOP increase.
  4. mala_steps=0 is rejected at build time rather than silently using eps=0.
"""
import jax
import jax.numpy as jnp

from relax.algorithm.mala_sampler import build_mala_sampler
from relax.algorithm.mgmd_types import Diffv2TrainState, Diffv2OptStates, HParams
from relax.network.actor_critic import ActorCritic

OBS_DIM, ACT_DIM, BATCH, T = 5, 3, 64, 20
PREDICTORS = ["Identity", "DDPM_mean", "DDIM", "DDIM_unguided"]


def make_model(mala_steps=2):
    return ActorCritic.create(OBS_DIM, ACT_DIM, [32, 32], [32, 32],
                              num_timesteps=T, beta_schedule_type="cosine",
                              mala_steps=mala_steps)


def make_state(model, guidance_mult=1.0, beta=2.0):
    params = model.init_params(jax.random.PRNGKey(0))
    return Diffv2TrainState(
        params=params,
        opt_state=Diffv2OptStates(q=(), policy=()),
        step=0,
        log_eta_scales=jnp.zeros((T,), jnp.float32),
        beta=jnp.float32(beta),
        hp=HParams(guidance_mult=jnp.float32(guidance_mult), alpha=jnp.float32(1.0)),
    )


def sample(model, state, predictor, ggs, obs):
    fn = build_mala_sampler(
        model=model, value_head=None, timesteps=T,
        batch_independent_guidance=False, ema_normalization=False,
        denoising_predictor=predictor, guidance_gradient_space=ggs,
    )
    agg = lambda qs: jnp.minimum(*qs) if len(qs) == 2 else qs[0]
    return jax.jit(lambda k, s, o: fn(k, s, o, agg))(jax.random.PRNGKey(7), state, obs)


def flops(model, state, predictor, ggs, obs):
    fn = build_mala_sampler(
        model=model, value_head=None, timesteps=T,
        batch_independent_guidance=False, ema_normalization=False,
        denoising_predictor=predictor, guidance_gradient_space=ggs,
    )
    agg = lambda qs: jnp.minimum(*qs) if len(qs) == 2 else qs[0]
    c = jax.jit(lambda k, s, o: fn(k, s, o, agg)).lower(
        jax.random.PRNGKey(7), state, obs).compile()
    return c.cost_analysis()["flops"]


def main():
    model = make_model()
    obs = jax.random.normal(jax.random.PRNGKey(1), (BATCH, OBS_DIM))
    ok = True

    print("--- 1. all predictors run, finite actions ---")
    for p in PREDICTORS:
        for ggs in ["xt", "x0hatclipped"]:
            r = sample(model, make_state(model), p, ggs, obs)
            fin = bool(jnp.all(jnp.isfinite(r.action)) & jnp.all(jnp.isfinite(r.log_eta_scales)))
            ok &= fin
            print(f"  {p:<14} ggs={ggs:<13} finite={fin}  "
                  f"acc={float(jnp.mean(r.per_level_acc)):.3f}  "
                  f"at_clip={float(jnp.mean(jnp.abs(r.action) >= 1.0)):.2f}")

    # The final action is jnp.clip(x_0, -1, 1) and log_eta_scales pins at its
    # clamp before warmup, so both saturate with an untrained net. Compare the
    # per-level acceptance rates: unclipped floats that depend on the whole
    # chain, including every predictor output.
    print("--- 2. DDIM_unguided ignores guidance_strength_multiplier ---")
    key = lambda r: r.per_level_acc
    for ggs in ["xt", "x0hatclipped"]:
        ref = key(sample(model, make_state(model, guidance_mult=1.0), "DDIM_unguided", ggs, obs))
        for c in [0.05, 100.0]:
            d = float(jnp.max(jnp.abs(
                key(sample(model, make_state(model, guidance_mult=c), "DDIM_unguided", ggs, obs)) - ref)))
            ok &= d == 0.0
            print(f"  ggs={ggs:<13} c={c:<6} max|diff vs c=1.0| = {d:.3e}  identical={d == 0.0}")
        # Informational only: at an untrained init every sampler observable is
        # saturated (actions at the +-1 clip, log_eta_scale at its clamp,
        # acceptance ~1), so this control cannot discriminate before warmup.
        # Test 3 is the load-bearing check that no guidance is computed.
        gref = key(sample(model, make_state(model, guidance_mult=1.0), "DDIM", ggs, obs))
        gd = float(jnp.max(jnp.abs(
            key(sample(model, make_state(model, guidance_mult=0.05), "DDIM", ggs, obs)) - gref)))
        print(f"  ggs={ggs:<13} [info] guided DDIM c=0.05 vs 1.0 max|diff| = {gd:.3e}"
              f"{'' if gd > 0 else '  (saturated at untrained init; see test 3)'}")

    print("--- 3. FLOPs: DDIM_unguided must not exceed Identity ---")
    for ggs in ["xt", "x0hatclipped"]:
        f = {p: flops(model, make_state(model), p, ggs, obs) for p in PREDICTORS}
        base = f["Identity"]
        print(f"  ggs={ggs}")
        for p in PREDICTORS:
            print(f"    {p:<14} {f[p]:>14,.0f}  ({f[p] / base:.3f}x Identity)")
        free = f["DDIM_unguided"] <= base * 1.001
        ok &= free
        print(f"    DDIM_unguided free: {free}")

    print("--- 4. mala_steps=0 is rejected ---")
    try:
        build_mala_sampler(
            model=make_model(mala_steps=0), value_head=None, timesteps=T,
            batch_independent_guidance=False, ema_normalization=False,
            denoising_predictor="DDIM_unguided", guidance_gradient_space="xt",
        )(jax.random.PRNGKey(0), make_state(make_model(0)), obs, lambda qs: qs[0])
        print("  FAIL: no error raised")
        ok = False
    except ValueError as e:
        print(f"  raised ValueError: {e}")

    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
