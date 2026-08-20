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

from relax.algorithm.mala_sampler import build_mala_sampler
from relax.algorithm.mgmd_types import Diffv2TrainState, Diffv2OptStates, HParams
from relax.network.actor_critic import ActorCritic
from relax.utils.diffusion import NoiseLevel, tweedie_x0

OBS_DIM, ACT_DIM, BATCH, T = 5, 3, 32, 8
NEW = "Identity_then_DDPM_mean"
PREDICTORS = ["Identity", NEW, "DDPM_mean", "DDIM", "DDIM_unguided"]


def make_model(timesteps=T, mala_steps=2):
    return ActorCritic.create(OBS_DIM, ACT_DIM, [32, 32], [32, 32],
                              num_timesteps=timesteps, beta_schedule_type="cosine",
                              mala_steps=mala_steps, x_recon_clip_radius=float("inf"))


def make_state(model, timesteps=T, guidance_mult=1.0, beta=2.0):
    return Diffv2TrainState(
        params=model.init_params(jax.random.PRNGKey(0)),
        opt_state=Diffv2OptStates(q=(), policy=()),
        step=0,
        log_eta_scales=jnp.zeros((timesteps,), jnp.float32),
        beta=jnp.float32(beta),
        hp=HParams(guidance_mult=jnp.float32(guidance_mult), alpha=jnp.float32(1.0),
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
    n = {p: matmuls(model, st, p, obs) for p in PREDICTORS}
    for p in PREDICTORS:
        print(f"  {p:<24} {n[p]:>8,d} matmuls  ({n[p] / n['Identity']:.3f}x Identity)")
    # DDPM_mean pays the predictor at all T levels, so its excess over Identity is
    # exactly T times one pass -- and the new predictor's excess is exactly one.
    per_pass, extra = n["DDPM_mean"] - n["Identity"], n[NEW] - n["Identity"]
    exact = extra * T == per_pass
    ok &= exact
    print(f"  extra over Identity = {extra} = 1/{per_pass / max(extra, 1):.0f} of "
          f"DDPM_mean's {per_pass}; == one of T passes: {exact}")

    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
