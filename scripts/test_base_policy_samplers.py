#!/usr/bin/env python3
"""Regression tests for the unguided base-policy samplers (DDIM, DDPM-mean).

    JAX_PLATFORMS=cpu python scripts/test_base_policy_samplers.py

Checks, for each transition:
  1. ``build_base_policy_sampler`` reproduces the generic MALA build at the
     matching ``denoising_predictor``, ``mala_steps=0``, ``beta=0``, ``alpha=1``
     -- i.e. it is "the base policy, sampled that way" as the generic code path
     already defines it, not a re-derivation of it. Run with the final clip off
     as well as on: at an untrained init the clipped action saturates at +-1, so
     only the unclipped comparison is sharp.

     Agreement is to ``TOL`` relative, not bit-exact. The two share the *one*
     implementation of each transition (``ddim_from_eps`` /
     ``ddpm_mean_from_eps``), so there is no formula to disagree on; what differs
     is the graph each sits in -- the generic build carries the zero-weighted
     guidance arithmetic alongside it -- and XLA is free to contract
     ``a*b + c*d`` into an fma in one and not the other. That is already visible
     at ``timesteps=1`` for DDPM-mean, at 0.3 ulp, and reaches ~2 ulp over a
     20-level chain. DDIM happens to come out exact.
  2. It is materially cheaper than that build, which still evaluates the
     guidance gradient (a backward pass through the Q net *and* the policy net)
     at every level before multiplying it by a traced zero. That gap is the
     whole reason the dedicated path exists.
  3. It ignores beta and alpha entirely, as an unguided chain must, while the
     generic build does not -- the control that check 1 pinned the right beta.
  4. It is reachable through ``MGMD.get_eval_action_vmap(sampler_kind=...)`` and
     the best-of-N read-out there is the same argmax-over-K the MALA path uses,
     so a comparison of samplers changes only the sampler.
"""
import jax
import jax.numpy as jnp

from relax.algorithm.base_policy_sampler import build_base_policy_sampler
from relax.algorithm.mala_sampler import build_mala_sampler
from relax.algorithm.mgmd_types import Diffv2TrainState, Diffv2OptStates, HParams, MGMDConfig
from relax.cli.train_args import build_parser
from relax.network.actor_critic import ActorCritic

OBS_DIM, ACT_DIM, BATCH, T, K = 5, 3, 64, 20, 4
AGG = lambda qs: jnp.minimum(*qs) if len(qs) == 2 else qs[0]
# transition name -> the --denoising_predictor it has to reproduce
TRANSITIONS = {"ddim": "DDIM", "ddpm_mean": "DDPM_mean"}
# ~8 float32 ulp: fusion-level agreement, well under any difference a wrong
# transition would produce (check 3's control moves the action by O(1)).
TOL = 1e-6


def make_model(mala_steps):
    return ActorCritic.create(OBS_DIM, ACT_DIM, [32, 32], [32, 32],
                              num_timesteps=T, beta_schedule_type="cosine",
                              mala_steps=mala_steps)


def make_state(model, beta=0.0, alpha=1.0, x0_hat_clip_radius=1.0):
    return Diffv2TrainState(
        params=model.init_params(jax.random.PRNGKey(0)),
        opt_state=Diffv2OptStates(q=(), policy=()),
        step=0,
        log_eta_scales=jnp.zeros((T,), jnp.float32),
        beta=jnp.float32(beta),
        hp=HParams(alpha=jnp.float32(alpha), eta=jnp.float32(beta),
                   x0_hat_clip_radius=jnp.float32(x0_hat_clip_radius)),
    )


def generic_fn(transition, model, k_actions, latent_action=False):
    """The generic build the dedicated sampler has to reproduce."""
    return build_mala_sampler(
        model=model, value_head=None, timesteps=T,
        batch_independent_guidance=False, ema_normalization=False,
        denoising_predictor=TRANSITIONS[transition], guidance_gradient_space="xt",
        num_denoised_actions=k_actions, latent_action=latent_action,
    )


def base_fn(transition, model, k_actions, latent_action=False):
    return build_base_policy_sampler(
        model=model, timesteps=T, transition=transition,
        num_denoised_actions=k_actions, latent_action=latent_action)


def run(fn, state, obs, key=jax.random.PRNGKey(7)):
    return jax.jit(lambda k, s, o: fn(k, s, o, AGG))(key, state, obs)


def flops(fn, state, obs):
    compiled = jax.jit(lambda k, s, o: fn(k, s, o, AGG)).lower(
        jax.random.PRNGKey(7), state, obs).compile()
    return compiled.cost_analysis()["flops"]


def eval_args():
    args = build_parser().parse_args([])
    args.diffusion_steps, args.beta_schedule_type = T, "cosine"
    args.mala_steps, args.denoising_predictor = 1, "Identity"
    args.alpha, args.beta, args.T, args.eta = 1.0, 0.0, 0.0, 0.0
    args.q_agg_sample = "min"
    return args


def main():
    model_no_mala, model_mala = make_model(0), make_model(1)
    obs = jax.random.normal(jax.random.PRNGKey(1), (BATCH, OBS_DIM))
    state = make_state(model_no_mala)
    ok = True
    # For scale in check 2: what the eta=0 tilted chain these are compared
    # against costs.
    tilted_flops = flops(build_mala_sampler(
        model=model_mala, value_head=None, timesteps=T,
        batch_independent_guidance=False, ema_normalization=False,
        denoising_predictor="Identity", guidance_gradient_space="xt",
    ), make_state(model_mala), obs)

    for transition, predictor in TRANSITIONS.items():
        print(f"\n=========== transition {transition} (--denoising_predictor {predictor}) ===========")
        generic = lambda k, latent=False: generic_fn(transition, model_no_mala, k, latent)
        ours = lambda k, latent=False: base_fn(transition, model_no_mala, k, latent)

        print(f"--- 1. reproduces the generic {predictor} build at mala_steps=0, beta=0 ---")
        for latent_action in (True, False):
            for k_actions in (1, K):
                reference = run(generic(k_actions, latent_action), state, obs)
                mine = run(ours(k_actions, latent_action), state, obs)
                scale = float(jnp.max(jnp.abs(reference.action)))
                d_action = float(jnp.max(jnp.abs(mine.action - reference.action))) / max(scale, 1.0)
                d_q = float(jnp.max(jnp.abs(mine.q - reference.q)))
                same = d_action <= TOL and d_q <= TOL
                ok &= same
                print(f"  clip={not latent_action!s:<5} K={k_actions:<3} "
                      f"rel|d action|={d_action:.3e}  max|d q|={d_q:.3e}  agrees={same}")

        print("--- 2. cheaper than that build (no zero-weighted guidance gradient) ---")
        generic_flops = flops(generic(1), state, obs)
        our_flops = flops(ours(1), state, obs)
        cheaper = our_flops < generic_flops
        ok &= cheaper
        print(f"  generic {predictor:<10} build {generic_flops:>14,.0f}")
        print(f"  build_base_policy_sampler  {our_flops:>14,.0f}  "
              f"({generic_flops / our_flops:.2f}x cheaper)  cheaper={cheaper}")
        print(f"  [info] eta=0 Identity chain at mala_steps=1: {tilted_flops:,.0f} "
              f"({tilted_flops / our_flops:.2f}x this sampler)")

        print("--- 3. unguided: beta and alpha do not reach it ---")
        # Unclipped, so the comparison is not made trivial by saturation at +-1.
        reference = run(ours(1, True), state, obs).action
        for beta, alpha in ((32.0, 1.0), (0.0, 0.5)):
            tilted_state = make_state(model_no_mala, beta=beta, alpha=alpha)
            d = float(jnp.max(jnp.abs(run(ours(1, True), tilted_state, obs).action - reference)))
            ok &= d == 0.0
            print(f"  beta={beta:<6} alpha={alpha:<5} max|d| = {d:.3e}  invariant={d == 0.0}")
        # Control: the generic build does move with beta, so check 1 was not
        # comparing two samplers that both happened to ignore the state. The
        # x0_hat clip has to be opened up for it: an untrained net's Tweedie
        # estimate lands far outside the default radius 1, where jnp.clip zeroes
        # the guidance gradient exactly and even a guided predictor stops
        # depending on beta.
        guided = lambda beta: run(
            generic(1, True),
            make_state(model_no_mala, beta=beta, x0_hat_clip_radius=1e6), obs).action
        moved = float(jnp.max(jnp.abs(guided(32.0) - guided(0.0))))
        ok &= moved > 0.0
        print(f"  [control] generic build at beta=32 vs 0: max|d| = {moved:.3e}  moved={moved > 0.0}")

        print(f"--- 4. reached through MGMD.get_eval_action_vmap(sampler_kind='{transition}') ---")
        from relax.algorithm.mgmd import MGMD
        algorithm = MGMD(model_mala, model_mala.init_params(jax.random.PRNGKey(0)),
                         MGMDConfig.from_args(eval_args()), obs_dim=OBS_DIM, hidden_dim=32)
        # The eval read-out is vmapped over seeds, so the state carries a leading
        # [num_runs] axis -- one slot here, as an offline snapshot study has.
        algorithm.state = jax.tree.map(lambda x: jnp.asarray(x)[None], algorithm.state)
        keys = jax.random.split(jax.random.PRNGKey(7), 1)
        action, q = algorithm.get_eval_action_vmap(keys, obs[None], K, return_q=True,
                                                  sampler_kind=transition)
        direct = run(base_fn(transition, model_mala, K),
                     jax.tree.map(lambda x: x[0], algorithm.state), obs, key=keys[0])
        best = jnp.argmax(direct.q, axis=0)
        expected = jnp.take_along_axis(direct.action, best[None, :, None], axis=0)[0]
        d = float(jnp.max(jnp.abs(jnp.asarray(action[0]) - expected)))
        ok &= d == 0.0
        print(f"  max|d vs direct best-of-{K}| = {d:.3e}  identical={d == 0.0}  "
              f"q range=({float(jnp.min(q)):.3f}, {float(jnp.max(q)):.3f})")

    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
