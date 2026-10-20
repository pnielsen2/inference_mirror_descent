#!/usr/bin/env python3
"""Regression tests for ``--td_next_action``.

    JAX_PLATFORMS=cpu python scripts/test_td_next_action.py

The flag redirects only the critic's backup action. The checks below pin both
halves of that claim:

  1. ``tilted`` (the default) is a no-op: the update produces bit-identical
     parameters, losses and RNG consumption to a build that predates the flag,
     which is emulated here by asserting no backup sampler is constructed and
     that a full update matches a reference run of the same config.
  2. ``ddim`` / ``ddpm_mean`` actually move the backup: the critic loss differs
     from the tilted run on the very first update.
  3. Only the critic is redirected. The distillation targets, the executed
     rollout action and the guidance normalizer must still come from the tilted
     chain, so the policy parameters after one update are UNCHANGED relative to
     the tilted run (the policy sees the same targets), while the Q parameters
     differ.
  4. The backup actions the ablation uses are the unguided base-policy samples,
     matching ``build_base_policy_sampler`` called directly.
"""
import jax
import jax.numpy as jnp
import numpy as np

from relax.algorithm.base_policy_sampler import build_base_policy_sampler
from relax.algorithm.mgmd import MGMD
from relax.algorithm.mgmd_types import MGMDConfig
from relax.network.actor_critic import ActorCritic
from relax.utils.experience import Experience

OBS_DIM, ACT_DIM, BATCH, T = 5, 3, 16, 4


def build(td_next_action):
    model = ActorCritic.create(OBS_DIM, ACT_DIM, [32, 32], [32, 32],
                               num_timesteps=T, beta_schedule_type="cosine",
                               mala_steps=1)
    params_list = [model.init_params(jax.random.PRNGKey(0))]
    cfg = MGMDConfig(alpha=1.0, beta=64.0, T=0.0, eta=64.0,
                     batch_size=BATCH, td_next_action=td_next_action,
                     ema_advantage_normalization=True,
                     guidance_snr_anneal="abar", x0_hat_clip_radius=3.0,
                     q_agg_sample="mean", delay_policy_update=1)
    algo = MGMD(model, params_list[0], cfg, obs_dim=OBS_DIM, hidden_dim=32)
    # update_vmap maps over a leading seed axis, so the state must be stacked.
    algo.state = algo.make_vmapped_state(params_list)
    return algo


def fake_batch(seed=0):
    rng = np.random.default_rng(seed)
    f = lambda *s: jnp.asarray(rng.normal(size=s), jnp.float32)
    return Experience(
        obs=f(1, BATCH, OBS_DIM), action=jnp.clip(f(1, BATCH, ACT_DIM), -1, 1),
        reward=f(1, BATCH), next_obs=f(1, BATCH, OBS_DIM),
        done=jnp.zeros((1, BATCH), jnp.float32),
    )


RUN_KEYS = jax.random.split(jax.random.PRNGKey(7), 1)   # [num_runs, 2]


def run_one_update(algo, data):
    info, _, actions = algo.update_vmap(RUN_KEYS, data, env_step=0.0)
    return info, actions, algo.state


def tree_max_diff(a, b):
    leaves = jax.tree.leaves(jax.tree.map(lambda x, y: jnp.max(jnp.abs(x - y)), a, b))
    return float(max(float(v) for v in leaves)) if leaves else 0.0


def main():
    data = fake_batch()

    tilted = build("tilted")
    assert tilted._backup_sampler is None, "tilted must not build a backup sampler"
    info_t, act_t, state_t = run_one_update(tilted, data)

    # 1. determinism / no-op: rebuilding and rerunning reproduces it exactly
    again = build("tilted")
    info_t2, act_t2, state_t2 = run_one_update(again, data)
    assert tree_max_diff(state_t.params, state_t2.params) == 0.0
    print("1. tilted is deterministic and builds no backup sampler        OK")

    results = {}
    for kind in ("ddim", "ddpm_mean"):
        algo = build(kind)
        assert algo._backup_sampler is not None
        info, act, state = run_one_update(algo, data)
        results[kind] = (info, act, state)

        # 2. the critic target moved
        dq = abs(float(info["losses/Q_loss"][0]) - float(info_t["losses/Q_loss"][0]))
        assert dq > 0, f"{kind}: Q loss identical to tilted -- backup did not change"

        # 3. only the critic moved: identical policy params, different Q params
        d_policy = tree_max_diff(state.params.policy, state_t.params.policy)
        d_q = tree_max_diff(state.params.q, state_t.params.q)
        assert d_policy == 0.0, f"{kind}: policy params changed ({d_policy:.3e})"
        assert d_q > 0.0, f"{kind}: Q params unchanged"
        # the executed/returned rollout action must still be the tilted one
        assert float(jnp.max(jnp.abs(act - act_t))) == 0.0, \
            f"{kind}: returned rollout action is not the tilted sample"
        print(f"2/3. {kind:9s}: Q loss delta {dq:.4e}, policy identical, "
              f"rollout action identical                     OK")

    # 4. the backup actions are the plain base-policy samples
    algo = build("ddim")
    state = algo.state
    slot = jax.tree.map(lambda x: x[0], state)
    direct = build_base_policy_sampler(
        model=algo.model, timesteps=algo._timesteps, transition="ddim",
        num_denoised_actions=1, compute_final_q=False)
    # Reconstruct the key the update hands the backup sampler: the per-run key
    # (post-vmap), split 3 ways, then folded once.
    key = jax.random.fold_in(jax.random.split(RUN_KEYS[0], 3)[0], 1)
    a_direct = direct(key, slot, data.next_obs[0], lambda qs: qs[0]).action
    a_via = algo._backup_sampler(key, slot, data.next_obs[0], lambda qs: qs[0]).action
    assert float(jnp.max(jnp.abs(a_direct - a_via))) == 0.0
    print("4. backup sampler is the unguided base-policy chain            OK")

    print("\nall checks passed")


if __name__ == "__main__":
    main()
