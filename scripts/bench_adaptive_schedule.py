#!/usr/bin/env python3
"""What the adaptive schedule step costs, on whatever device JAX picks.

    python scripts/bench_adaptive_schedule.py                  # GPU if visible
    JAX_PLATFORMS=cpu python scripts/bench_adaptive_schedule.py

Three variants of one tilted-action pass, jitted end to end and timed with the
same state, obs and PRNG key:

  1. baseline   -- adaptive sampling, no schedule work at all.
  2. post-hoc   -- sampling, then a score-optimal cost that RE-EVALUATES both
                   levels' drifts at every stored sample (2N-1 extra guided-energy
                   gradients).
  3. fused      -- sampling that differences the drifts MH already computed, then
                   Algorithm 1 over the resulting per-interval costs (algebra
                   only, no network calls at all).

Variant 2 is reproduced here rather than kept in ``relax/`` -- it is the obvious
implementation the fused path avoids, retained only so the two can be timed
against each other.
"""
import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from relax.algorithm import noise_schedule
from relax.algorithm.mala_sampler import build_mala_sampler, build_target_energy
from relax.algorithm.mgmd import MGMD, _aggregate_q
from relax.algorithm.mgmd_types import MGMDConfig
from relax.cli.train_setup import _mish
from relax.network.actor_critic import ActorCritic
from relax.utils.diffusion import NoiseLevel

AGG = lambda qs: _aggregate_q(qs, "mean")


def build(args):
    model = ActorCritic.create(
        args.obs_dim, args.act_dim, [args.hidden] * 3, [args.hidden] * 3, _mish,
        num_timesteps=args.timesteps, beta_schedule_type="adaptive",
        mala_steps=args.mala_steps, num_q_networks=2, x_recon_clip_radius=float("inf"),
        policy_parameterization="E", policy_final_layer="ff")
    cfg = MGMDConfig(alpha=1.0, beta=1.0, T=0.0, eta=1.0, num_denoised_actions=args.k,
                     denoising_predictor="Identity", guidance_gradient_space="xt",
                     batch_independent_guidance=True, x0_hat_clip_radius=3.0,
                     q_agg_sample="mean", latent_action=True,
                     noise_schedule_gamma=args.gamma,
                     ema_within_advantage_normalization=True)
    params = model.init_params(jax.random.PRNGKey(0))
    alg = MGMD(model, params, cfg, obs_dim=args.obs_dim, hidden_dim=args.hidden)
    alg.state = jax.tree.map(lambda x: x[0], alg.make_vmapped_state([params]))
    return alg


def posthoc_updater(alg, args, energy_kw, obs, fused):
    """The obvious implementation: re-take both drifts at every stored sample."""
    T, K = args.timesteps, args.k

    def update(state, x_levels):
        en, _, _ = build_target_energy(
            state, jnp.broadcast_to(obs, (K, *obs.shape)), AGG,
            model=alg.model, **energy_kw)
        gradU = jax.grad(lambda x, l: jnp.sum(en(NoiseLevel.from_log_snr(l), x)[0]))
        lam = state.log_snr_levels
        omac = jax.nn.sigmoid(-lam)

        def interval(i, cost):
            x = x_levels[i + 1]
            prev = jnp.where(i + 1 == T, x, gradU(x, lam[jnp.minimum(i + 1, T - 1)]))
            gap = gradU(x, lam[i]) - prev
            return cost.at[i].set(omac[i] * jnp.mean(jnp.sum(gap * gap, -1)))

        cost = jax.lax.fori_loop(0, T, interval, jnp.zeros((T,), jnp.float32))
        return fused(state, cost)

    return update


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=80)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--obs_dim", type=int, default=17)
    p.add_argument("--act_dim", type=int, default=6)
    p.add_argument("--mala_steps", type=int, default=1)
    p.add_argument("--gamma", type=float, default=0.01)
    p.add_argument("--reps", type=int, default=7)
    args = p.parse_args()

    print(f"device: {jax.devices()[0].device_kind} ({jax.default_backend()})")
    print(f"T={args.timesteps} batch={args.batch} K={args.k} hidden={args.hidden} "
          f"mala_steps={args.mala_steps}")

    alg = build(args)
    state = alg.state
    obs = jnp.asarray(np.random.default_rng(0).standard_normal(
        (args.batch, args.obs_dim)), jnp.float32)
    key = jax.random.PRNGKey(7)
    energy_kw = dict(model=alg.model, ema_normalization=False,
                     batch_advantage_normalization=False, ema_advantage_normalization=False)
    sampler_kw = dict(value_head=None, timesteps=args.timesteps,
                      batch_independent_guidance=True, denoising_predictor="Identity",
                      guidance_gradient_space="xt", num_denoised_actions=args.k,
                      latent_action=True, compute_final_q=False, **energy_kw)

    baseline = build_mala_sampler(**sampler_kw)
    with_levels = build_mala_sampler(**sampler_kw, collect_levels=True)
    with_cost = build_mala_sampler(**sampler_kw, schedule_cost=True)
    fused = noise_schedule.build_updater(timesteps=args.timesteps)
    posthoc = posthoc_updater(alg, args,
                              {k: v for k, v in energy_kw.items() if k != "model"},
                              obs, fused)

    variants = {
        "1. baseline (no schedule)": lambda k, s, o: baseline(k, s, o, AGG).action,
        "2. post-hoc recompute": lambda k, s, o: posthoc(s, with_levels(k, s, o, AGG).x_levels),
        "3. fused from MH drifts": lambda k, s, o: fused(s, with_cost(k, s, o, AGG).schedule_cost),
    }

    results = {}
    for name, fn in variants.items():
        f = jax.jit(fn)
        ts = []
        for i in range(args.reps):
            t0 = time.perf_counter()
            jax.block_until_ready(f(key, state, obs))
            ts.append(time.perf_counter() - t0)
        results[name] = float(np.median(ts[2:]))   # drop compile + first call
    base = results["1. baseline (no schedule)"]
    print()
    for name, t in results.items():
        print(f"  {name:28s} {t * 1e3:9.2f} ms   {t / base:6.3f}x baseline"
              f"   (+{(t / base - 1) * 100:5.1f}%)")


if __name__ == "__main__":
    main()
