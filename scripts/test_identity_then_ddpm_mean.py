#!/usr/bin/env python3
"""Regression tests for the Identity_then_DDPM_mean denoising predictor.

    JAX_PLATFORMS=cpu python scripts/test_identity_then_ddpm_mean.py

Checks:
  1. The chain itself is bit-identical to Identity's -- per-level acceptance and
     clip fractions, adapted step sizes, every level's sample (collect_levels)
     and every score-optimal cost (schedule_cost) -- while the returned action
     differs. Nothing moves between levels, which is what --beta_schedule_type
     adaptive assumes when it scores adjacent levels at a shared sample.
  2. The action is the DDPM posterior mean at t = 0 of that shared level-0
     sample. With beta = 0 and guidance_mult = 0 that mean is the plain Tweedie
     estimate, recomputed here from model.eps_pred; with guidance on, the action
     moves with the guidance multiplier, which Identity's cannot (the multiplier
     enters only the predictor, never the MH target).
  3. At T = 1 there are no between-level transitions left, so it must be
     bit-identical to DDPM_mean.
  4. It costs ONE extra guided predictor pass per sampler call, not T of them.
     Measured by counting ``dot_general`` in the jaxpr weighted by every
     enclosing scan's trip count -- the level-0 step is peeled out of the level
     loop, so it duplicates that level's *code*, and a compiled FLOP total
     (which counts a loop body once, however many times it runs) would read that
     as work.

Run with latent_action (no +-1 endpoint clip) and x_recon_clip_radius = inf, so
that at an untrained init the observables compared here are unsaturated floats.
"""
import jax
import jax.numpy as jnp
from jax.extend.core import ClosedJaxpr, Jaxpr

import relax.algorithm.mala_sampler as mala_sampler
from relax.algorithm.mala_sampler import build_mala_sampler, build_target_energy, retained_policy_exponent
from relax.algorithm.mgmd_types import Diffv2TrainState, Diffv2OptStates, HParams
from relax.cli.train_setup import resolve_cmd_params
from relax.network.actor_critic import ActorCritic
from relax.utils.diffusion import NoiseLevel, tweedie_x0

OBS_DIM, ACT_DIM, BATCH, T = 5, 3, 32, 8
NEW = "Identity_then_DDPM_mean"
NEW_K = "Identity_then_DDPM_mean_final_k"
PREDICTORS = ["Identity", NEW, NEW_K, "DDPM_mean", "DDIM", "DDIM_unguided"]


def make_model(timesteps=T, mala_steps=2):
    return ActorCritic.create(OBS_DIM, ACT_DIM, [32, 32], [32, 32],
                              num_timesteps=timesteps, beta_schedule_type="cosine",
                              mala_steps=mala_steps, x_recon_clip_radius=float("inf"))


def make_state(model, timesteps=T, guidance_mult=1.0, beta=2.0, alpha=1.0):
    return Diffv2TrainState(
        params=model.init_params(jax.random.PRNGKey(0)),
        opt_state=Diffv2OptStates(q=(), policy=()),
        step=0,
        log_eta_scales=jnp.zeros((timesteps,), jnp.float32),
        beta=jnp.float32(beta),
        hp=HParams(guidance_mult=jnp.float32(guidance_mult), alpha=jnp.float32(alpha),
                   x0_hat_clip_radius=jnp.float32(jnp.inf)),
    )


AGG = lambda qs: jnp.minimum(*qs) if len(qs) == 2 else qs[0]


def sampler(model, predictor, timesteps=T, **kw):
    return build_mala_sampler(
        model=model, value_head=None, timesteps=timesteps,
        batch_independent_guidance=False, ema_normalization=False,
        denoising_predictor=predictor, guidance_gradient_space="xt",
        latent_action=True, **kw)


def sample(model, state, predictor, obs, timesteps=T, **kw):
    fn = sampler(model, predictor, timesteps, **kw)
    return jax.jit(lambda k, s, o: fn(k, s, o, AGG))(jax.random.PRNGKey(7), state, obs)


def _sub_jaxprs(params):
    for v in params.values():
        for x in (v if isinstance(v, (tuple, list)) else (v,)):
            if isinstance(x, ClosedJaxpr):
                yield x.jaxpr
            elif isinstance(x, Jaxpr):
                yield x


def count_matmuls(jaxpr, weight=1):
    """``dot_general`` count, each weighted by how many times it actually runs."""
    total = 0
    for eqn in jaxpr.eqns:
        total += weight * (eqn.primitive.name == "dot_general")
        # fori_loop with static bounds lowers to scan, so its trip count is a
        # parameter we can read; a data-dependent while_loop's would not be.
        w = weight * eqn.params["length"] if eqn.primitive.name == "scan" else weight
        for sub in _sub_jaxprs(eqn.params):
            total += count_matmuls(sub, w)
    return total


def matmuls(model, state, predictor, obs):
    fn = sampler(model, predictor)
    return count_matmuls(jax.make_jaxpr(lambda k, s, o: fn(k, s, o, AGG))(
        jax.random.PRNGKey(7), state, obs).jaxpr)


def maxdiff(a, b):
    return float(jnp.max(jnp.abs(a - b)))


def main():
    model = make_model()
    obs = jax.random.normal(jax.random.PRNGKey(1), (BATCH, OBS_DIM))
    ok = True

    print("--- 0. all predictors run, finite actions ---")
    for p in PREDICTORS:
        r = sample(model, make_state(model), p, obs)
        fin = bool(jnp.all(jnp.isfinite(r.action)) & jnp.all(jnp.isfinite(r.log_eta_scales)))
        ok &= fin
        print(f"  {p:<24} finite={fin}  acc={float(jnp.mean(r.per_level_acc)):.3f}  "
              f"|a|max={float(jnp.max(jnp.abs(r.action))):.3f}")

    print("--- 1. chain identical to Identity, action different ---")
    kw = dict(collect_levels=True, schedule_cost=True)
    ident = sample(model, make_state(model), "Identity", obs, **kw)
    new = sample(model, make_state(model), NEW, obs, **kw)
    for name in ("per_level_acc", "per_level_clip", "log_eta_scales", "x_levels", "schedule_cost"):
        d = maxdiff(getattr(new, name), getattr(ident, name))
        ok &= d == 0.0
        print(f"  {name:<16} max|diff| = {d:.3e}  identical={d == 0.0}")
    moved = maxdiff(new.action, ident.action)
    ok &= moved > 0.0
    print(f"  action           max|diff| = {moved:.3e}  differs={moved > 0.0}")

    print("--- 2a. unguided (beta=0, mult=0): action == Tweedie x0 at level 0 ---")
    st = make_state(model, guidance_mult=0.0, beta=0.0)
    r = sample(model, st, NEW, obs, collect_levels=True)
    x0_level = r.x_levels[0]                      # [K, batch, act_dim], K = 1
    obs_k = jnp.broadcast_to(obs, (1, *obs.shape))
    lvl = NoiseLevel.at(model.schedule_for(st.hp, st.log_snr_levels), 0)
    eps = model.eps_pred(st.params.policy, lvl, obs_k, x0_level)
    expected = tweedie_x0(lvl, x0_level, eps)     # coef1[0] = 1, coef2[0] = 0 at t = 0
    d = maxdiff(r.action, expected)
    scale = float(jnp.max(jnp.abs(expected)))
    ok &= d <= 1e-5 * max(scale, 1.0)
    print(f"  max|action - x0_hat(level-0 sample)| = {d:.3e}   (|x0_hat|max = {scale:.3f})")
    # Control: Identity returns the noisy level-0 sample itself, not its x0_hat.
    di = maxdiff(sample(model, st, "Identity", obs, collect_levels=True).action, x0_level)
    print(f"  [control] Identity returns the level-0 sample: max|diff| = {di:.3e}")
    ok &= di == 0.0

    print("--- 2aa. retained-policy alpha endpoints and x0_hat stay separate ---")
    schedule = model.schedule_for(st.hp, st.log_snr_levels)
    levels = [NoiseLevel.at(schedule, i) for i in range(T)]
    for clean_alpha in (0.0, 0.3, 1.0):
        values = jnp.stack([
            retained_policy_exponent(jnp.float32(clean_alpha), lvl, schedule)
            for lvl in levels
        ])
        assert float(values[0]) == float(jnp.float32(clean_alpha))
        assert float(values[-1]) == 1.0
        assert bool(jnp.all(jnp.diff(values) >= 0.0))
    alpha0, beta64, temp, eta = resolve_cmd_params(0.0, 64.0, None, None)
    assert alpha0 == 0.0 and beta64 == 64.0 and temp == 1.0 / 64.0 and jnp.isinf(eta)

    fixed_x = jax.random.normal(jax.random.PRNGKey(13), (1, BATCH, ACT_DIM))
    fixed_obs = jnp.broadcast_to(obs, (1, *obs.shape))
    lvl = levels[T // 2]
    for clean_alpha in (0.0, 1.0):
        alpha_state = make_state(model, beta=2.0, alpha=clean_alpha)
        target, _, _ = build_target_energy(
            alpha_state, fixed_obs, AGG, model=model, ema_normalization=False,
            batch_advantage_normalization=False, ema_advantage_normalization=False)
        actual = target(lvl, fixed_x)[0]
        base_eps = model.eps_pred(alpha_state.params.policy, lvl, fixed_obs, fixed_x)
        base_x0_hat = tweedie_x0(lvl, fixed_x, base_eps)
        base_q = AGG([model.q(qp, fixed_obs, base_x0_hat)
                      for qp in alpha_state.params.q])
        base_energy = model.energy_fn(alpha_state.params.policy, lvl, fixed_obs, fixed_x)
        alpha_level = retained_policy_exponent(alpha_state.hp.alpha, lvl, schedule)
        expected = alpha_level * base_energy - jnp.float32(2.0) * base_q
        assert maxdiff(actual, expected) < 2e-6

    alpha_zero_state = make_state(model, guidance_mult=0.0, beta=0.0, alpha=0.0)
    alpha_zero_result = sample(model, alpha_zero_state, NEW, obs, collect_levels=True)
    alpha_zero_x = alpha_zero_result.x_levels[0]
    alpha_zero_level = NoiseLevel.at(
        model.schedule_for(alpha_zero_state.hp, alpha_zero_state.log_snr_levels), 0)
    alpha_zero_eps = model.eps_pred(
        alpha_zero_state.params.policy, alpha_zero_level, obs_k, alpha_zero_x)
    alpha_zero_expected = tweedie_x0(alpha_zero_level, alpha_zero_x, alpha_zero_eps)
    assert maxdiff(alpha_zero_result.action, alpha_zero_expected) \
        <= 1e-5 * max(float(jnp.max(jnp.abs(alpha_zero_expected))), 1.0)
    print("  noisiest=1, cleanest=configured alpha; base Tweedie x0 unchanged")

    print("--- 2b. the final step is guided (Identity's action is not) ---")
    for p in ("Identity", NEW):
        ref = sample(model, make_state(model, guidance_mult=1.0), p, obs).action
        d = maxdiff(sample(model, make_state(model, guidance_mult=50.0), p, obs).action, ref)
        expect_move = p == NEW
        ok &= (d > 0.0) == expect_move
        print(f"  {p:<24} mult 1 -> 50: max|diff| = {d:.3e}  "
              f"{'moves' if d > 0 else 'unchanged'} (expected "
              f"{'moves' if expect_move else 'unchanged'})")

    print("--- 3. T=1 is exactly DDPM_mean ---")
    m1 = make_model(timesteps=1)
    s1 = make_state(m1, timesteps=1)
    a = sample(m1, s1, NEW, obs, timesteps=1)
    b = sample(m1, s1, "DDPM_mean", obs, timesteps=1)
    for name in ("action", "per_level_acc", "log_eta_scales"):
        d = maxdiff(getattr(a, name), getattr(b, name))
        ok &= d == 0.0
        print(f"  {name:<16} max|diff| = {d:.3e}  identical={d == 0.0}")

    print(f"--- 4. one guided predictor pass per call, not T={T} ---")
    st = make_state(model)
    counted_predictors = ["Identity", NEW, "DDPM_mean", "DDIM", "DDIM_unguided"]
    n = {p: matmuls(model, st, p, obs) for p in counted_predictors}
    for p in counted_predictors:
        print(f"  {p:<24} {n[p]:>8,d} matmuls  ({n[p] / n['Identity']:.3f}x Identity)")
    # DDPM_mean pays the predictor at all T levels, so its excess over Identity is
    # exactly T times one pass -- and the new predictor's excess is exactly one.
    per_pass, extra = n["DDPM_mean"] - n["Identity"], n[NEW] - n["Identity"]
    exact = extra * T == per_pass
    ok &= exact
    print(f"  extra over Identity = {extra} = 1/{per_pass / max(extra, 1):.0f} of "
          f"DDPM_mean's {per_pass}; == one of T passes: {exact}")

    print("--- 5. final-k endpoints agree at k=1 and k=T ---")
    for k, reference in ((1, NEW), (T, "DDPM_mean")):
        result = sample(model, st, NEW_K, obs, ddpm_mean_final_steps=k)
        expected = sample(model, st, reference, obs)
        for name in ("action", "per_level_acc", "log_eta_scales"):
            d = maxdiff(getattr(result, name), getattr(expected, name))
            ok &= d == 0.0
            print(f"  k={k:<2} {name:<16} max|diff| = {d:.3e}  identical={d == 0.0}")

    print("--- 6. --predictor_final_steps restricts any predictor to the last k ---")
    # The generic flag must reproduce the DDPM_mean-specific predictor exactly, and at
    # k = 0 / k = T leave the unrestricted predictor's graph untouched.
    for k in (1, 2, T):
        a = sample(model, st, "DDPM_mean", obs, predictor_final_steps=k)
        b = sample(model, st, NEW_K, obs, ddpm_mean_final_steps=k)
        d = maxdiff(a.action, b.action)
        ok &= d == 0.0
        print(f"  DDPM_mean k={k:<2} vs {NEW_K} k={k:<2}: max|diff| = {d:.3e}  identical={d == 0.0}")
    for predictor in ("DDPM_mean", "DDIM", "DDIM_unguided"):
        full = sample(model, st, predictor, obs).action
        for k, expect_same in ((0, True), (T, True), (1, False)):
            d = maxdiff(sample(model, st, predictor, obs, predictor_final_steps=k).action, full)
            same = d == 0.0
            ok &= same == expect_same
            print(f"  {predictor:<14} k={k:<2} {'==' if same else '!='} unrestricted "
                  f"(expected {'==' if expect_same else '!='})")
        # Restricting to the last k must leave the MALA chain itself untouched.
        d = maxdiff(sample(model, st, predictor, obs, predictor_final_steps=2).per_level_acc,
                    sample(model, st, "Identity", obs).per_level_acc)
        print(f"  {predictor:<14} k=2 acceptance vs Identity: max|diff| = {d:.3e}")

    print("--- 7. alpha=1 is bit-identical to the constant-alpha target ---")
    compatibility_kw = dict(collect_levels=True, schedule_cost=True)
    scheduled = sample(model, make_state(model, alpha=1.0), NEW, obs, **compatibility_kw)
    original_retained_policy_exponent = mala_sampler.retained_policy_exponent
    try:
        mala_sampler.retained_policy_exponent = lambda alpha, level, schedule: alpha
        legacy = sample(model, make_state(model, alpha=1.0), NEW, obs, **compatibility_kw)
    finally:
        mala_sampler.retained_policy_exponent = original_retained_policy_exponent
    for name in ("action", "per_level_acc", "per_level_clip", "log_eta_scales",
                 "x_levels", "schedule_cost"):
        d = maxdiff(getattr(scheduled, name), getattr(legacy, name))
        ok &= d == 0.0
        print(f"  {name:<16} max|diff| = {d:.3e}  identical={d == 0.0}")

    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
