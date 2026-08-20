#!/usr/bin/env python3
"""Regression tests for --beta_schedule_type adaptive.

    JAX_PLATFORMS=cpu python scripts/test_adaptive_schedule.py

The schedule is the score-optimal one of Williams, Campbell, Doucet and Syed,
NeurIPS 2024: free log-SNR knots, pinned at both ends, moved each batch toward
the layout that spends equal cost per step. Checks:

  1. The knots start at cosine's shape cut to [log_snr_min, log_snr_max], with
     the ends exactly on the bounds, and take no s_hat anywhere.
  2. The derived coefficients are self-consistent for an arbitrary ladder (abar =
     sigmoid(lambda), betas reproduce abar under cumprod, t_cond = lambda) and
     stay finite at extreme lambda, where the continuous distillation draw
     reaches.
  3. NoiseLevel.at(grid) == NoiseLevel.from_log_snr(lambda) on-grid; the
     continuous schedule interpolates the knots and hits them on the u grid; and
     the fixed families still condition on the level index.
  4. The recorded per-interval cost is what it claims: the squared score gap
     between adjacent levels at the SHARED sample, weighted by that level's MALA
     step size squared (v = h, not the paper's sigma), recomputed from scratch
     against the levels the sampler visited, including the N(0,I) end whose score
     is -x. Asking for it does not perturb the trajectory.
  5. The update is Algorithm 1: endpoints stay pinned, the ladder stays
     decreasing, an already-equal-cost ladder is a fixed point, cost concentrated
     in one region pulls knots into it, and gamma scales the move linearly.
  6. Only the four requested scalars are logged, and a non-adaptive run logs none
     of them.
  7. Adaptive costs no policy/Q calls in the sampler that cosine does not: the
     sampler FLOPs at fixed T match between the two schedule types.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from relax.algorithm import noise_schedule
from relax.algorithm.mala_sampler import build_mala_sampler, build_target_energy
from relax.algorithm.mgmd import MGMD, _aggregate_q
from relax.algorithm.mgmd_types import MGMDConfig
from relax.cli.train_setup import _mish
from relax.network.actor_critic import ActorCritic
from relax.utils.diffusion import (
    NoiseLevel, build_beta_schedule, cosine_log_snr_knots, log_snr_at,
    schedule_from_log_snr, shift_schedule,
)
from relax.utils.experience import Experience

OBS, ACT, T, K, B, SEEDS = 4, 3, 6, 2, 8, 2
# Read the bounds off the config rather than restating them, so this tests the
# range the flags actually produce instead of a second copy of their defaults.
LAM_MAX = MGMDConfig.noise_schedule_log_snr_max
LAM_MIN = MGMDConfig.noise_schedule_log_snr_min
ENERGY_KW = dict(ema_normalization=False, batch_advantage_normalization=False,
                 ema_advantage_normalization=False)
AGG = lambda qs: _aggregate_q(qs, "mean")


def make(schedule_type, **cfg_kw):
    model = ActorCritic.create(
        OBS, ACT, [16, 16], [16, 16], _mish, num_timesteps=T,
        beta_schedule_type=schedule_type, mala_steps=1, num_q_networks=2,
        x_recon_clip_radius=float("inf"), snr_max=124.0,
        policy_parameterization="E", policy_final_layer="ff")
    params_list = [model.init_params(jax.random.PRNGKey(s)) for s in range(SEEDS)]
    cfg = MGMDConfig(
        alpha=1.0, beta=1.0, T=0.0, eta=1.0, num_denoised_actions=K, batch_size=B,
        denoising_predictor="Identity" if schedule_type == "adaptive" else "DDPM_mean",
        guidance_gradient_space="xt", batch_independent_guidance=True,
        x0_hat_clip_radius=3.0, q_agg_sample="mean", latent_action=True,
        ema_within_advantage_normalization=True, **cfg_kw)
    alg = MGMD(model, params_list[0], cfg, obs_dim=OBS, hidden_dim=16)
    alg.state = alg.make_vmapped_state(params_list)
    return alg


def main():
    import inspect

    print("--- 1. knots: cosine truncated to the bounds, ends pinned, no s_hat ---")
    # Read off the built state, so this tests the initialization the flags
    # actually produce rather than a second copy of their defaults.
    alg = make("adaptive")
    state = jax.tree.map(lambda x: x[0], alg.state)
    hp = state.hp
    lam = np.asarray(state.log_snr_levels, np.float64)
    assert lam.shape == (T,), lam.shape
    assert lam[0] == LAM_MAX and lam[-1] == LAM_MIN, (lam[0], lam[-1])
    assert np.all(np.diff(lam) < 0), "must be cleanest-first"
    # Interior knots must lie on cosine's own log-SNR curve, i.e. resampling that
    # curve is what placed them -- not a linear ramp between the bounds.
    s = 0.008
    t_of = lambda l: np.arccos(np.sqrt(np.cos(s / (1 + s) * np.pi / 2) ** 2
                                       / (1 + np.exp(-l)))) * 2 * (1 + s) / np.pi - s
    tt = t_of(lam)
    assert np.allclose(np.diff(tt), np.diff(tt)[0], rtol=1e-4), "not evenly resampled in cosine t"
    assert not np.allclose(lam, np.linspace(LAM_MAX, LAM_MIN, T), rtol=1e-2), "that is a linear ramp"
    for fn in (schedule_from_log_snr, cosine_log_snr_knots, log_snr_at):
        assert "s_hat" not in inspect.signature(fn).parameters, fn
    # ... and the schedule the model hands out ignores hp.s_hat entirely.
    a = alg.model.schedule_for(hp, state.log_snr_levels)
    b = alg.model.schedule_for(hp._replace(s_hat=jnp.float32(0.3)), state.log_snr_levels)
    assert np.allclose(np.asarray(a.alphas_cumprod), np.asarray(b.alphas_cumprod))
    print(f"  lambda = {np.round(lam, 3)}  on cosine's curve, s_hat-free")

    print("--- 2. coefficients self-consistent, finite at extreme lambda ---")
    sc = schedule_from_log_snr(jnp.asarray(lam, jnp.float32))
    abar = np.asarray(sc.alphas_cumprod, np.float64)
    assert np.allclose(abar, 1 / (1 + np.exp(-lam)), atol=1e-6)
    assert np.allclose(np.asarray(sc.one_minus_alphas_cumprod), 1 - abar, atol=1e-6)
    assert np.allclose(np.asarray(sc.t_cond), lam, atol=1e-5), "net conditions on lambda"
    assert np.allclose(np.cumprod(1 - np.asarray(sc.betas, np.float64)), abar, rtol=1e-4)
    # beta_k = 1 - abar_k/abar_{k-1} would saturate to exactly 1 in float32 once a
    # gap spans ~18 nats (alphas = 0 then, which only the DDPM/DDIM posteriors
    # would care about, and adaptive forbids those). The default bounds keep every
    # gap well short of that even at this toy T, so <= is the tolerance, not the
    # expectation -- widening --noise_schedule_log_snr_{max,min} is what would use it.
    assert np.all((np.asarray(sc.betas) > 0) & (np.asarray(sc.betas) <= 1))
    assert np.all(np.asarray(sc.alphas) > 0), "a gap wide enough to make alpha vanish"
    assert all(np.isfinite(np.asarray(v)).all() for v in sc), "a coefficient blew up"
    for l in (-40.0, 40.0):
        assert all(np.isfinite(np.asarray(v)) for v in NoiseLevel.from_log_snr(jnp.float32(l)))
    print(f"  betas in ({float(sc.betas.min()):.2e}, {float(sc.betas.max()):.6f}), "
          f"abar within {1 - abar.max():.1e} of 1 and {abar.min():.1e} of 0; all finite")

    print("--- 3. on-grid == off-grid, and the continuous schedule hits the knots ---")
    for f, x, y in zip(NoiseLevel._fields, NoiseLevel.at(sc, 3),
                       NoiseLevel.from_log_snr(sc.t_cond[3])):
        assert np.allclose(np.asarray(x), np.asarray(y), rtol=2e-5), (f, x, y)
    u = jnp.arange(T, dtype=jnp.float32) / (T - 1)
    assert np.allclose(np.asarray(log_snr_at(state.log_snr_levels, u)), lam, rtol=1e-6)
    mid = np.asarray(log_snr_at(state.log_snr_levels, jnp.float32(0.5 / (T - 1))))
    assert np.isclose(mid, (lam[0] + lam[1]) / 2, rtol=1e-5), "should interpolate between knots"
    cos = shift_schedule(build_beta_schedule(T, "cosine", 124.0), jnp.float32(1.0))
    assert np.allclose(np.asarray(cos.t_cond), np.arange(T)), "fixed family: index"
    print("  ok")

    print("--- 4. the recorded cost is h^2 * squared score gap at the shared x ---")
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.standard_normal((B, OBS)), jnp.float32)
    energy_total, _, _ = build_target_energy(
        state, jnp.broadcast_to(obs, (K, *obs.shape)), AGG, model=alg.model, **ENERGY_KW)
    # grad_x U at an arbitrary log-SNR: the score, up to the sign the cost squares away.
    gradU = jax.jit(jax.grad(lambda x, l: jnp.sum(energy_total(NoiseLevel.from_log_snr(l), x)[0])))

    def sample(model, **kw):
        fn = build_mala_sampler(
            model=model, value_head=None, timesteps=T, batch_independent_guidance=True,
            denoising_predictor="Identity", guidance_gradient_space="xt",
            num_denoised_actions=K, latent_action=True, **kw, **ENERGY_KW)
        return jax.jit(lambda k, s, o: fn(k, s, o, AGG))(jax.random.PRNGKey(7), state, obs)

    lam_j = jnp.asarray(lam, jnp.float32)
    # Run at more than one MALA step too: the cost must be taken at the state the
    # level INHERITED (the m == 0 evaluation, not a later one), and the drift
    # handed on must be the one accepted at the LAST step. Both are selections
    # inside the chain loop that a single-step chain cannot distinguish.
    for steps in (1, 3):
        m = alg.model if steps == 1 else dataclasses.replace(alg.model, mala_steps=steps)
        r = sample(m, schedule_cost=True, collect_levels=True)
        cost, xl = np.asarray(r.schedule_cost, np.float64), r.x_levels
        if steps == 1:
            cost_1step = cost.copy()          # the un-annealed reference for 4b
        assert cost.shape == (T,), cost.shape
        x_T = jax.random.normal(jax.random.split(jax.random.PRNGKey(7), 2)[0], (K, B, ACT))
        assert np.allclose(np.asarray(xl[T]), np.asarray(x_T)), "level T != the N(0,I) start"
        assert np.allclose(np.asarray(xl[0]), np.asarray(r.action)), "level 0 != the action"
        for j in range(T):
            x = xl[j + 1]                   # the state level j inherits from level j+1
            # The score it arrives with: level j+1's, or the prior's (-x) at the top.
            prev = x if j + 1 == T else gradU(x, lam_j[j + 1])
            gap = gradU(x, lam_j[j]) - prev
            # v(t') = this level's MALA step size (log_eta_scale = 0 at init),
            # not the paper's sigma(t') -- see noise_schedule.
            h = float(np.clip(max(float(sc.betas[j]), 1e-8), 1e-8, 0.5))
            ref = h * h * float(jnp.mean(jnp.sum(gap * gap, -1)))
            assert np.isclose(cost[j], ref, rtol=2e-4), (steps, j, cost[j], ref)
        assert np.allclose(np.asarray(sample(m, schedule_cost=False).action),
                           np.asarray(r.action), rtol=2e-4, atol=2e-5), "scoring moved the chain"
        print(f"  mala_steps={steps}: all {T} intervals match an independent recomputation, "
              f"and the action is unchanged by scoring")
    print(f"  cost = {np.array2string(cost, precision=3)}")

    print("--- 4b. --guidance_snr_anneal damps the low-SNR cost blow-up ---")
    # The guided term's gradient carries 1/sqrt(abar) from the Tweedie map, which
    # an inexact score does not cancel; undamped it makes the noisy end
    # astronomically expensive and collapses the ladder onto lambda_min.
    ranges = {}
    for mode in ("none", "sqrt_abar", "abar"):
        fn = build_mala_sampler(
            model=alg.model, value_head=None, timesteps=T, batch_independent_guidance=True,
            denoising_predictor="Identity", guidance_gradient_space="xt",
            num_denoised_actions=K, latent_action=True, schedule_cost=True,
            guidance_snr_anneal=mode, **ENERGY_KW)
        out = jax.jit(lambda k, s, o: fn(k, s, o, AGG))(jax.random.PRNGKey(7), state, obs)
        c = np.asarray(out.schedule_cost, np.float64)[:-1]
        ranges[mode] = c.max() / max(c.min(), 1e-300)
        if mode == "none":
            assert np.allclose(c, cost_1step[:-1]), \
                "'none' must reproduce the un-annealed cost exactly"
    assert ranges["abar"] < ranges["sqrt_abar"] < ranges["none"], ranges
    print("  cost dynamic range  " + "  ".join(f"{m}={ranges[m]:.2e}" for m in ranges))
    # beta_eff must be untouched at the clean end and killed at the noisy end.
    from relax.algorithm.mala_sampler import guidance_snr_anneal_factor
    for mode, at_min in (("abar", 3.06e-7), ("sqrt_abar", 5.53e-4)):
        hi = float(guidance_snr_anneal_factor(NoiseLevel.from_log_snr(jnp.float32(LAM_MAX)), mode))
        lo = float(guidance_snr_anneal_factor(NoiseLevel.from_log_snr(jnp.float32(LAM_MIN)), mode))
        assert abs(hi - 1.0) < 1e-5 and np.isclose(lo, at_min, rtol=1e-2), (mode, hi, lo)
        print(f"  {mode:10s}: beta_eff/beta = {hi:.6f} at lambda={LAM_MAX:+.0f}, {lo:.2e} at {LAM_MIN:+.0f}")

    print("--- 5. Algorithm 1: pinned ends, monotone, equal-cost fixed point ---")
    update = jax.jit(noise_schedule.build_updater(timesteps=T))
    at_gamma = lambda g, c: np.asarray(update(
        state._replace(hp=state.hp._replace(noise_schedule_gamma=jnp.float32(g))),
        jnp.asarray(c, jnp.float32))[0].log_snr_levels, np.float64)

    moved = at_gamma(1.0, cost)
    assert moved[0] == lam[0] and moved[-1] == lam[-1], "endpoints must not move"
    assert np.all(np.diff(moved) < 0), "ladder must stay decreasing"
    # Equal per-interval cost is exactly the condition Theorem 3.1 asks for, so
    # the fully-adopted update must return the ladder unchanged.
    assert np.allclose(at_gamma(1.0, np.ones(T)), lam, atol=1e-4), "equal cost is not a fixed point"
    # Cost piled into the clean third must pull knots toward it (their mean rises).
    lop = np.full(T, 1e-6); lop[:T // 3] = 1.0
    assert at_gamma(1.0, lop).mean() > lam.mean() + 1.0, at_gamma(1.0, lop)
    # ... and gamma is a plain convex combination of old and new.
    for g in (0.1, 0.5):
        assert np.allclose(at_gamma(g, lop), (1 - g) * lam + g * at_gamma(1.0, lop), atol=1e-4), g
    print(f"  ends pinned at ({moved[0]:.1f}, {moved[-1]:.1f}); equal cost is a fixed point; "
          f"lopsided cost moves the mean {lam.mean():.2f} -> {at_gamma(1.0, lop).mean():.2f}")

    print("--- 6. exactly the expected logged scalars, none for a fixed schedule ---")
    ex = Experience.create_example(OBS, ACT, B)
    data = jax.tree.map(
        lambda x: jnp.asarray(rng.standard_normal((SEEDS,) + np.shape(x)), jnp.float32)
        if np.asarray(x).dtype == np.float32 else
        jnp.broadcast_to(jnp.asarray(x), (SEEDS,) + np.shape(x)), ex)
    keys = jax.random.split(jax.random.PRNGKey(0), SEEDS)
    scalar_info, array_info, _ = alg.update_vmap(keys, data, env_step=0.0)
    assert {k for k in scalar_info if k.startswith("Schedule")} == {
        "Schedule/path_length", "Schedule/total_cost",
        "Schedule/noise_schedule_mean", "Schedule/noise_schedule_std",
        # Health check on the Tweedie map: with a consistent denoiser x0_hat
        # shrinks toward the prior mean as noise rises, so the clip fraction must
        # FALL toward the noisy end. Rising means d x0hat/dx still carries
        # 1/sqrt(abar) and the cost is not yet worth descending.
        "Schedule/x0hat_clip_cleanest", "Schedule/x0hat_clip_noisiest"}, sorted(scalar_info)
    assert not any(k.startswith("Schedule") for k in array_info), sorted(array_info)
    for k, v in scalar_info.items():
        if k.startswith("Schedule"):
            assert np.isfinite(np.asarray(v)).all(), k
            # The mean is a log-SNR and may sit either side of 0; the rest cannot.
            assert k.endswith("mean") or np.all(np.asarray(v) >= 0), (k, v)
    print("  " + "  ".join(f"{k.split('/')[1]}={np.round(np.asarray(v), 3)}"
                           for k, v in sorted(scalar_info.items()) if k.startswith("Schedule")))
    cos_alg = make("cosine")
    cos_scalar, cos_array, _ = cos_alg.update_vmap(keys, data, env_step=0.0)
    assert not any(k.startswith("Schedule") for k in {**cos_scalar, **cos_array})
    assert cos_alg.state.log_snr_levels is None, "fixed families carry no knots"
    print("  cosine run logs no Schedule/* keys and carries no knots")

    print("--- 7. adaptive adds no per-sample work to the sampler ---")
    def sampler_flops(alg_, batch):
        fn = build_mala_sampler(
            model=alg_.model, value_head=None, timesteps=T,
            batch_independent_guidance=True, denoising_predictor="Identity",
            guidance_gradient_space="xt", num_denoised_actions=K, **ENERGY_KW)
        s = jax.tree.map(lambda x: x[0], alg_.state)
        o = jnp.zeros((batch, OBS), jnp.float32)
        return jax.jit(lambda k, st, oo: fn(k, st, oo, AGG)).lower(
            jax.random.PRNGKey(7), s, o).compile().cost_analysis()["flops"]
    # Every network call is per-sample, so any extra one would make the gap grow
    # with the batch. A gap that is CONSTANT in batch size is the O(T) schedule
    # construction (log_sigmoid vs the s_hat shift) and nothing else.
    gaps = [sampler_flops(alg, b) - sampler_flops(cos_alg, b) for b in (B, 4 * B)]
    assert gaps[0] == gaps[1], gaps
    assert abs(gaps[0]) / sampler_flops(cos_alg, B) < 1e-3, gaps
    print(f"  flop gap vs cosine is {gaps[0]:+.0f} at batch {B} and batch {4 * B}: "
          "batch-independent, so no extra network calls")

    print("\nALL PASS")


if __name__ == "__main__":
    main()
