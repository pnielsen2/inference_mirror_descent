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
from pathlib import Path
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax

from relax.algorithm import distillation, hp_pack
from relax.algorithm.base_policy_sampler import build_base_policy_sampler
from relax.algorithm.mala_sampler import build_mala_sampler, ddim_from_eps
from relax.algorithm.mgmd import MGMD, _boltzmann_weights
from relax.algorithm.mgmd_types import Diffv2TrainState, Diffv2OptStates, HParams, MGMDConfig
from relax.utils.experience import Experience
from relax.utils.diffusion import NoiseLevel, build_beta_schedule
from relax.cli.train_args import build_parser
from relax.network.actor_critic import ActorCritic
from relax.trainer.evaluation import Evaluator

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


def check_clipped_transitions():
    schedule = build_beta_schedule(50, "cosine", 124.)
    initial = jax.random.normal(jax.random.PRNGKey(0), (64, ACT_DIM))
    mish = lambda x: x * jnp.tanh(jax.nn.softplus(x))

    def chain(radius):
        def step(x, t):
            # A finite, high-gain Mish denoiser: the old chain leaves its input
            # range, overflows, and produces NaNs before the final action clip.
            eps = 1000 * (mish(x) - mish(-x))
            x = ddim_from_eps(schedule, t, x, eps, radius)
            return x, x
        return jax.lax.scan(step, initial, jnp.arange(49, -1, -1))[1]

    raw, clipped = chain(None), chain(1.)
    assert not np.isfinite(np.asarray(raw)).all()
    assert np.isfinite(np.asarray(clipped)).all()
    bound = jnp.abs(initial) / schedule.sqrt_one_minus_alphas_cumprod[-1] + 1
    assert bool(jnp.all(jnp.abs(clipped) <= bound[None] + 1e-5))
    assert float(jnp.max(jnp.abs(clipped[-1]))) <= 1
    # Independent DDIM equation using epsilon re-derived from the clipped x0.
    t, eps = 49, jnp.full_like(initial, 10.)
    alpha, sigma = schedule.sqrt_alphas_cumprod[t], schedule.sqrt_one_minus_alphas_cumprod[t]
    x0 = jnp.clip((initial - sigma * eps) / alpha, -1, 1)
    corrected_eps = (initial - alpha * x0) / sigma
    expected = (jnp.sqrt(schedule.alphas_cumprod_prev[t]) * x0
                + jnp.sqrt(schedule.one_minus_alphas_cumprod_prev[t]) * corrected_eps)
    np.testing.assert_allclose(ddim_from_eps(schedule, t, initial, eps, 1.), expected, atol=1e-6)
    print("clipped DDIM: reproduces corrected-epsilon equation; overflowing chain stays bounded")


def main():
    check_clipped_transitions()
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

    print("\n--- 5. direct eps policy produces vector predictions and DDIM samples ---")
    small_T, small_B, small_K = 4, 8, 3
    direct_model = ActorCritic.create(
        OBS_DIM, ACT_DIM, [16, 16], [16, 16], num_timesteps=small_T,
        beta_schedule_type="cosine", mala_steps=1, policy_parameterization="eps")
    direct_params = direct_model.init_params(jax.random.PRNGKey(0))
    direct_state = Diffv2TrainState(
        params=direct_params, opt_state=Diffv2OptStates(q=(), policy=()), step=0,
        log_eta_scales=jnp.zeros((small_T,), jnp.float32), beta=jnp.float32(64),
        hp=HParams(eta=jnp.float32(2)))
    small_obs = jax.random.normal(jax.random.PRNGKey(1), (small_B, OBS_DIM))
    level = direct_model.schedule_for(direct_state.hp).t_cond[0]
    eps = direct_model._policy_scalar.apply(direct_params.policy, small_obs,
                                            jnp.zeros((small_B, ACT_DIM)), level)
    direct_out = build_base_policy_sampler(
        model=direct_model, timesteps=small_T, transition="ddim",
        num_denoised_actions=small_K)(jax.random.PRNGKey(7), direct_state, small_obs, AGG)
    assert direct_model.energy_fn is None and eps.shape == (small_B, ACT_DIM)
    assert direct_out.action.shape == (small_K, small_B, ACT_DIM)
    assert bool(jnp.all(jnp.isfinite(direct_out.action)))
    print("  vector eps output and K-candidate DDIM sample                    OK")

    print("--- 6. Boltzmann weights use eta, the configured normalizer, and candidate axis ---")
    weight_cfg = MGMDConfig(eta=2.0, ema_advantage_normalization=True)
    weight_state = direct_state._replace(
        q_running_mean=jnp.float32(3), q_running_std=jnp.float32(4),
        hp=direct_state.hp._replace(eta=jnp.float32(2)))
    q = jnp.asarray([[0., 3.], [2., 1.], [4., -1.]], jnp.float32)
    weights = _boltzmann_weights(q, weight_state, weight_cfg)
    expected = jax.nn.softmax(jnp.float32(0.5) * (q - 3), axis=0)
    assert float(jnp.max(jnp.abs(weights - expected))) < 1e-7
    assert float(jnp.max(jnp.abs(jnp.sum(weights, axis=0) - 1))) < 1e-7
    print("  exact normalized softmax weights                                 OK")

    print("--- 7. selection and reweighting compose under a full vmapped update ---")
    rng = np.random.default_rng(0)
    rand = lambda *shape: jnp.asarray(rng.normal(size=shape), jnp.float32)
    data = Experience(
        obs=rand(1, small_B, OBS_DIM),
        action=jnp.clip(rand(1, small_B, ACT_DIM), -1, 1),
        reward=rand(1, small_B), next_obs=rand(1, small_B, OBS_DIM),
        done=jnp.zeros((1, small_B), jnp.float32))
    run_keys = jax.random.split(jax.random.PRNGKey(9), 1)

    def candidate_run(selection, reweight, noise=(0.,), count=small_K, sampler="ddim"):
        model = ActorCritic.create(
            OBS_DIM, ACT_DIM, [16, 16], [16, 16], num_timesteps=small_T,
            beta_schedule_type="cosine", mala_steps=1, policy_parameterization="eps")
        params = [model.init_params(jax.random.PRNGKey(0))] * len(noise)
        cfg = MGMDConfig(
            alpha=1., beta=64., T=0., eta=2., batch_size=small_B,
            candidate_sampler=sampler, candidate_selection=selection,
            boltzmann_reweight=reweight, num_denoised_actions=count,
            delay_policy_update=1, ema_advantage_normalization=True,
            q_agg_sample="mean")
        algo = MGMD(model, params[0], cfg, obs_dim=OBS_DIM, hidden_dim=16)
        algo.state = algo.make_vmapped_state(params)
        algo.state = hp_pack.apply(algo.state, {"action_noise": list(noise)}, len(noise))
        info, _, actions = algo.update_vmap(
            jnp.repeat(run_keys, len(noise), axis=0),
            jax.tree.map(lambda x: jnp.repeat(x, len(noise), axis=0), data), env_step=0.)
        return algo, info, actions

    uniform, uniform_info, _ = candidate_run("first", False)
    selected, selected_info, selected_action = candidate_run("boltzmann", False)
    weighted, weighted_info, weighted_action = candidate_run("first", True)
    combined, combined_info, combined_action = candidate_run("boltzmann", True)
    assert selected_action.shape == weighted_action.shape == combined_action.shape == (1, small_B, ACT_DIM)
    assert selected.state.distill_buffer.shape == (1, small_B, OBS_DIM + ACT_DIM)
    assert weighted.state.distill_buffer.shape == (1, small_K * small_B, OBS_DIM + ACT_DIM + 1)
    stored_weights = weighted.state.distill_buffer[0, :, -1].reshape(small_K, small_B)
    assert float(jnp.max(jnp.abs(jnp.sum(stored_weights, axis=0) - 1))) < 1e-6
    assert int(np.asarray(uniform.state.opt_state.policy.count)[0]) == small_K
    assert int(np.asarray(selected.state.opt_state.policy.count)[0]) == 1
    assert int(np.asarray(weighted.state.opt_state.policy.count)[0]) == 1
    assert int(np.asarray(combined.state.opt_state.policy.count)[0]) == 1
    assert "Boltzmann/ess_mean" not in uniform_info
    assert 1 <= float(selected_info["Boltzmann/ess_mean"][0]) <= small_K
    assert 1 <= float(weighted_info["Boltzmann/ess_mean"][0]) <= small_K
    policy_diffs = jax.tree.leaves(jax.tree.map(
        lambda x, y: jnp.max(jnp.abs(x - y)),
        weighted.state.params.policy, combined.state.params.policy))
    assert max(float(x) for x in policy_diffs) == 0.0
    assert float(jnp.max(jnp.abs(weighted_info["losses/Q_loss"] - combined_info["losses/Q_loss"]))) == 0.0
    assert float(jnp.max(jnp.abs(uniform_info["losses/Q_loss"] - weighted_info["losses/Q_loss"]))) > 0.0
    assert all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree.leaves(combined.state.params))
    rollout_obs = np.asarray(data.next_obs[:, :2])
    rollout_action, rollout_q, _ = combined.get_action_vmap(run_keys, rollout_obs)
    eval_action = combined.get_eval_action_vmap(run_keys, rollout_obs, 2)
    base_eval_action = combined.get_eval_action_vmap(
        run_keys, rollout_obs, 2, sampler_kind="ddim")
    assert rollout_action.shape == (1, 2, ACT_DIM) and rollout_q.shape == (1, 2)
    assert eval_action.shape == base_eval_action.shape == (1, 2, ACT_DIM)
    print("  first/Boltzmann rollout and one-batch weighted distillation       OK")

    print("--- 8. all six N=32 variants: targets, loss scale, TD and packed noise ---")
    candidate_key, time_key, noise_key = jax.random.split(run_keys[0], 3)
    for sampler in ("ddim", "ddim_clipped", "ddpm"):
        proposals = build_base_policy_sampler(
            model=direct_model, timesteps=small_T, transition=sampler, num_denoised_actions=32)(
                candidate_key, direct_state, data.next_obs[0], lambda qs: sum(qs) / len(qs))
        probabilities = jax.nn.softmax(2 * proposals.q, axis=0)
        perturbation = 0.5 * jax.random.normal(jax.random.fold_in(candidate_key, 2), proposals.action.shape)
        for mode in ("reweight", "boltzmann", "best"):
            algo, info, actions = candidate_run(mode, False, (0., 1.), count=32, sampler=sampler)
            index = (jnp.argmax(proposals.q, axis=0) if mode == "best" else
                     jax.random.categorical(jax.random.fold_in(candidate_key, 1), 2 * proposals.q, axis=0))
            for slot in range(2):
                targets = proposals.action + slot * perturbation
                chosen = targets[index, jnp.arange(small_B)]
                np.testing.assert_allclose(actions[slot], chosen, atol=2e-6)
                targets = targets if mode == "reweight" else chosen[None]
                rows = algo.state.distill_buffer[slot]
                np.testing.assert_allclose(rows[:, OBS_DIM:OBS_DIM + ACT_DIM],
                                           targets.reshape(-1, ACT_DIM), atol=2e-6)
                if mode == "reweight":
                    np.testing.assert_allclose(rows[:, -1].reshape(32, small_B), probabilities, atol=2e-6)
                # Independent loss oracle: one sum over probabilities per STATE,
                # then a batch mean, with the same epsilon-MSE scale in every arm.
                n = rows.shape[0]
                level = NoiseLevel.at(direct_model.schedule_for(direct_state.hp),
                                     jax.random.randint(time_key, (n,), 0, small_T))
                eps = jax.random.normal(noise_key, (n, ACT_DIM))
                pred = direct_model.eps_pred(direct_params.policy, level, rows[:, :OBS_DIM],
                                            direct_model.q_sample(level, targets.reshape(n, ACT_DIM), eps))
                losses = jnp.mean((pred - eps)**2, axis=-1).reshape(-1, small_B)
                expected_loss = (jnp.sum(probabilities * losses, axis=0).mean()
                                 if mode == "reweight" else losses.mean())
                np.testing.assert_allclose(info["losses/Policy_epsilon_MSE"][slot], expected_loss, rtol=2e-6)
                target_obs = jnp.broadcast_to(data.next_obs[0], (*targets.shape[:-1], OBS_DIM))
                target_q = jnp.minimum(*[direct_model.q(p, target_obs, targets) for p in direct_params.target_q])
                target_q = (jnp.sum(probabilities * target_q, axis=0) if mode == "reweight" else target_q[0])
                backup = 0.2 * data.reward[0] + 0.99 * target_q
                td_loss = jnp.mean(jnp.stack([
                    (direct_model.q(p, data.obs[0], data.action[0]) - backup)**2 for p in direct_params.q]))
                np.testing.assert_allclose(info["losses/Q_loss"][slot], td_loss, rtol=2e-6)
            np.testing.assert_array_equal(info["Policy/action_noise_std"], [0., 0.5])
            np.testing.assert_array_equal(algo.state.opt_state.policy.count, [1, 1])
            assert np.max(np.abs(np.asarray(actions[1]))) > 1  # No clip after injection.
            # Identical params/keys, varying only noise: eval stays clean.
            algo.state = jax.tree.map(lambda x: jnp.repeat(x[:1], 2, axis=0), algo.state)
            algo.state = hp_pack.apply(algo.state, {"action_noise": [0., 1.]}, 2)
            keys = jnp.repeat(run_keys, 2, axis=0)
            observations = np.repeat(rollout_obs, 2, axis=0)
            evaluation = algo.get_eval_action_vmap(keys, observations, 32, sampler_kind=sampler)
            np.testing.assert_array_equal(evaluation[0], evaluation[1])
            assert np.isfinite(evaluation).all() and np.max(np.abs(evaluation)) <= 1
            rollout, q, _ = algo.get_action_vmap(keys, observations)
            assert not np.array_equal(rollout[0], rollout[1])
            actual_q, _ = algo.eval_q_v_vmap(observations, rollout)
            np.testing.assert_allclose(q, actual_q, atol=2e-6)
        print(f"  {sampler}: all six variants, TD/loss oracle, rollout and evaluation OK")

    # DDPM must add no posterior noise on its final transition.
    one_model = replace(direct_model, num_timesteps=1, schedule=build_beta_schedule(1, "cosine", 124.))
    one = lambda kind: build_base_policy_sampler(model=one_model, timesteps=1, transition=kind)(
        candidate_key, direct_state, data.next_obs[0], AGG).action
    np.testing.assert_array_equal(one("ddpm"), one("ddpm_mean"))

    print("--- 9. tabulated noise matches DPMD's original entropy-controller loss ---")
    optim = optax.adam(0.007)
    initial = jnp.log(jnp.float32(5))
    def controller(carry, _):
        log_alpha, opt_state = carry
        def loss(x):
            entropy = 0.5 * ACT_DIM * jnp.log(2 * jnp.pi * jnp.e * (0.1 * jnp.exp(x))**2)
            return -x * (-jax.lax.stop_gradient(entropy) - 0.9 * ACT_DIM)
        delta, opt_state = optim.update(jax.grad(loss)(log_alpha), opt_state)
        log_alpha = optax.apply_updates(log_alpha, delta)
        return (log_alpha, opt_state), 0.1 * jnp.exp(log_alpha)
    table = distillation.dpmd_noise_schedule(ACT_DIM, 1000000)
    _, reference = jax.lax.scan(controller, (initial, optim.init(initial)), None, length=len(table) - 1)
    np.testing.assert_allclose(table[1:], reference, rtol=2e-6)
    assert float(table[0]) == 0.5 and abs(float(table[-1]) - 0.0984) < 0.001
    single = jax.tree.map(lambda x: x[1], algo.state)
    for step, index in ((0, 0), (1, 1), (1250, 1), (1251, 2)):
        np.testing.assert_allclose(algo._action_noise_std(single._replace(step=jnp.int32(step))), table[index])
    print("  original Adam schedule and environment-step boundaries           OK")

    print("--- 10. evaluator forwards explicit base-policy sampler choice ---")
    class FakeAlgorithm:
        call = None

        def get_eval_action_vmap(self, key, obs, num_actions, return_q=False,
                                 sampler_kind=None):
            self.call = (num_actions, sampler_kind, obs.shape)
            return np.zeros((*obs.shape[:-1], ACT_DIM), np.float32)

    class FakeEnv:
        def get_current_obs(self):
            return np.zeros((2, OBS_DIM), np.float32)

        def step(self, action):
            return self.get_current_obs(), np.zeros(2), np.ones(2, bool), np.zeros(2, bool), {}

        def close(self):
            pass

    fake_algorithm = FakeAlgorithm()
    evaluator = Evaluator(
        algorithm=fake_algorithm, logger=None, log_path=Path("/tmp"), env_name="fake",
        num_runs=1, to_env_action=lambda x: x, every=1, n_episodes=2,
        best_of_n=32, sampler_kind="ddim", seed=0)
    evaluator._make_env = lambda: (FakeEnv(), 1)
    evaluator._log = lambda *args: None
    evaluator._run(1)
    assert fake_algorithm.call == (32, "ddim", (1, 2, OBS_DIM))
    assert build_parser().parse_args(["--eval_sampler", "ddim"]).eval_sampler == "ddim"
    print("  best-of-32 directly uses 32 base DDIM candidates                 OK")

    # Reject invalid actions before notifying any native simulator worker.
    from relax.env.vector.process3 import ProcessVectorEnv
    env = ProcessVectorEnv.__new__(ProcessVectorEnv)
    env.workers = None
    env.action = np.zeros((2, ACT_DIM), np.float32)
    for value in (np.nan, np.inf, -np.inf, 1e40):
        invalid = np.zeros((2, ACT_DIM), np.float64)
        invalid[1, 0] = value
        try:
            with np.errstate(over="ignore"):
                env.step(invalid)
        except FloatingPointError as exc:
            assert "env indices [1]" in str(exc)
            assert not env.action.any()
        else:
            raise AssertionError("non-finite action reached simulator")
    print("  NaN/Inf actions fail before simulator dispatch                   OK")

    print("\nALL PASS" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
