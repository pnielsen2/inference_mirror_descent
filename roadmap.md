# Implementation Roadmap: Soft Policy Iteration with Diagnostic Logging

This document is the concrete, step-by-step implementation plan derived from `assumptions.md`. It is organized into **phases** that can be executed sequentially. Each phase lists the code changes required, the new flags to add, the metrics to wire up, and acceptance criteria.

See `diagnostic_metrics_tree.md` for the full hierarchical breakdown of logged quantities and their analytical relationships.

---

## Table of Contents

1. [Phase 0: Logging Overhaul — Strip and Re-foundation](#phase-0)
2. [Phase 1: Soft Policy Iteration Training Loop](#phase-1)
3. [Phase 2: Q-Learning Phase Diagnostics](#phase-2)
4. [Phase 3: Guidance-Accuracy Diagnostics](#phase-3)
5. [Phase 4: Q-Shift Diagnostics](#phase-4)
6. [Phase 5: Distillation Diagnostics](#phase-5)
7. [Phase 6: Expensive / Debug-Mode Diagnostics](#phase-6)
8. [Phase 7: Hyperparameter Sensitivity & Automation](#phase-7)
9. [Phase 8: Push for SOTA Episode Return](#phase-8)
10. [Phase 9: Re-introduce Sample Efficiency](#phase-9)
11. [Flag Reference](#flag-reference)
12. [File-Level Change Map](#file-level-change-map)

---

<a id="phase-0"></a>
## Phase 0: Simplify Parameter Structure & Logging

**Goal**: Strip the current `DPMD` implementation down to the essentials needed for soft policy iteration. Remove sample-efficiency tricks (target networks, replay buffer, reweighting, DSAC-T, critic normalization) and clean up logging to match the diagnostic tree. The result should be a minimal, correct implementation we can build on.

### 0.0 Terminology

Throughout this document:

- **Iteration**: A single step of all `b` parallel environments + the associated gradient updates (possibly multiple via `--update_per_iteration`). This is the inner loop that already exists in the training code.
- **Policy improvement step**: The event where the base policy π_k is replaced by the distilled policy π_{k+1}, and Q is updated accordingly. This is the outer loop we are adding.

### 0.1 Reference config (current best)

The best-performing config before this work is:

```bash
scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4 \
    --dpmd_constant_weight --tfg_lambda 16.0 --num_particles 1 \
    --mala_steps 2 --q_critic_agg mean --beta_schedule_type cosine \
    --beta_schedule_scale 1 --dpmd_no_entropy_tuning --buffer_size 200000 \
    --x0_hat_clip_radius 3.0 --mala_adapt_rate .2 --mala_per_level_eta \
    --q_td_huber_width 30.0 --update_per_iteration 4 --lr_q 0.00015 \
    --mala_guided_predictor
```

Key flags already in use that we keep: `--dpmd_constant_weight`, `--dpmd_no_entropy_tuning`, `--q_critic_agg mean`, `--mala_steps 2`, `--mala_per_level_eta`, `--mala_adapt_rate .2`, `--mala_guided_predictor`, `--q_td_huber_width 30.0`, `--update_per_iteration 4`, `--lr_q 0.00015`.

### 0.2 Current `Diffv2Params` structure (what exists today)

```python
class Diffv2Params(NamedTuple):       # relax/network/diffv2.py:14-21
    q1: hk.Params                      # Live Q1 network
    q2: hk.Params                      # Live Q2 network
    target_q1: hk.Params               # EMA shadow of Q1 (for stable TD targets)
    target_q2: hk.Params               # EMA shadow of Q2
    policy: hk.Params                  # Live policy network
    target_poicy: hk.Params            # EMA shadow of policy (typo in field name)
    log_alpha: jax.Array               # Entropy temperature (log scale)
```

**What each target param does in the current code**:
- `target_q1`/`target_q2`: Used for computing TD backup targets in `stateless_update`. Updated via Polyak averaging (`tau=0.005`) at line 1383–1387.
- `target_poicy`: Updated via Polyak averaging at line 1388. **Only used in `get_policy_params_to_save()`** — it is saved to checkpoints but never used for action sampling or TD targets. Functionally dead weight during training.

**Simplification for soft-PI mode**: Target networks are a sample-efficiency trick for stabilizing off-policy TD learning. In our setting (on-policy data from guided actions, no replay buffer), they are unnecessary. We defer them to Phase 9.

### 0.3 What to remove / simplify

**Remove from `Diffv2Params`** (in soft-PI mode):
- `target_q1`, `target_q2` → use live `q1`/`q2` directly for TD backups
- `target_poicy` → unused; remove
- `log_alpha` → keep for now (alpha is fixed via `--dpmd_no_entropy_tuning` but the field still needs to exist)

**Implementation approach**: Rather than refactoring `Diffv2Params` (which would break legacy mode), set `target_q1 = q1`, `target_q2 = q2`, `target_poicy = policy` at init and **skip the Polyak update** in `stateless_update` when in soft-PI mode. This is a one-line change (gate `delay_target_update` calls behind a flag).

**Remove from `stateless_update`**:
- Q-reweighting logic (`q_weights_*`, `scale_q_*`, `running_q_*`, `running_mean`, `running_std`) — already disabled by `--dpmd_constant_weight`
- Entropy tuning (`log_alpha_loss_fn`, `delay_alpha_param_update`) — already disabled by `--dpmd_no_entropy_tuning`
- DSAC-T refinements (`dsac_omega_*`, `dsac_b_*`, `dsac_adaptive_clip_*`) — sample-efficiency, defer
- Critic normalization (`value_params`, `advantage_second_moment_ema`, `norm_adv/*`) — sample-efficiency, defer
- Off-policy TD mode (`off_policy_td`, `next_action_buffer`) — we are on-policy
- Reward critic mode (`use_reward_critic`) — not used in reference config

These are not deleted from the code (legacy mode still works), but in soft-PI mode they are skipped. The simplest approach: add a `self.soft_pi_mode: bool` attribute to `DPMD` and gate the removed code behind `not self.soft_pi_mode`.

### 0.4 Logging cleanup

**Current `info` dict → new keys** (in soft-PI mode):

| Current key | Action | New key |
|-------------|--------|---------|
| `q1_loss` | Rename | `q_learning/td_loss_q1` |
| `q2_loss` | Rename | `q_learning/td_loss_q2` |
| `policy_loss` | Rename | `distillation/policy_loss` |
| `MALA/acceptance_rate` | Keep | `MALA/acceptance_rate` |
| `MALA/eta_scale` (or `hist/...`) | Keep | `MALA/eta_scale` |
| `act/mean_abs_action` | Keep | `act/mean_abs_action` |
| `act/std_action` | Keep | `act/std_action` |
| `act/clip_frac` | Keep | `act/clip_frac` |
| `td/q1_ce_loss` .. `td/td_error2_std` | Replace | `q_learning/bellman_error_mean`, `q_learning/bellman_error_std` |
| `q1_mean`, `q1_max`, `q1_min` | Remove | — |
| `alpha` | Keep | `training/alpha` |
| `q_weights_*`, `scale_q_*`, `running_q_*` | Remove | — |
| `entropy_approx` | Remove | — |
| `dsac_t/*` | Remove | — |
| `norm_adv/*` | Remove | — |

**Episode-level logging** (from `SampleLog` / `VectorSampleLog`): keep `sample/episode_return`, `sample/episode_length`. Remove everything else.

**New wandb prefixes**:

```
sample/          — episode-level metrics (return, length)
pi_step/         — policy improvement step tracking (id, step count)
q_learning/      — Q-function training diagnostics
guidance/        — MGMD guidance accuracy
q_shift/         — frozen vs live Q divergence
distillation/    — policy distillation diagnostics
act/             — action distribution health
MALA/            — MALA sampler diagnostics
debug/           — expensive metrics (only in diagnostic mode)
timing/          — wall-clock timing (unchanged)
training/        — alpha, etc.
```

### 0.5 Implementation steps

1. Add `self.soft_pi_mode: bool` to `DPMD.__init__()` (passed from CLI).
2. In `stateless_update`, when `self.soft_pi_mode`:
   - Skip Polyak target updates (set target params = live params).
   - Skip alpha update.
   - Skip Q-reweighting / running mean/std tracking.
   - Skip DSAC-T and critic normalization logic.
   - Return `info` dict with new keys.
3. Gate the removed code with `if not self.soft_pi_mode:` so legacy mode is untouched.
4. Add `--soft_pi_mode` flag to `scripts/train_mujoco.py`.

### Acceptance criteria
- [ ] Legacy config (no `--soft_pi_mode`) produces identical behavior and logging.
- [ ] With `--soft_pi_mode`, Polyak updates are skipped, simplified `info` dict is returned.
- [ ] New metric keys appear in wandb under the correct prefixes.

---

<a id="phase-1"></a>
## Phase 1: Soft Policy Iteration Training Loop

**Goal**: Add the outer loop that performs policy improvement steps. The inner loop (iterations) already exists — each iteration steps `b` parallel environments once and performs `update_per_iteration` gradient updates. We add an outer loop that periodically freezes Q for guidance, runs many iterations of guided rollout + simultaneous Q/policy updates, then transitions to a new base policy.

### 1.1 Loop structure

```
for policy_improvement_step k = 0, 1, ..., num_pi_steps - 1:

    # --- Start of policy improvement step k ---
    Freeze Q_k ← copy of current live Q params (used for MGMD guidance only)

    for iteration i = 1, ..., iterations_per_pi_step:

        # Step all b parallel environments once
        For each of the b envs:
            1. Observe s
            2. Sample guided action ã ~ MGMD(π_k, Q_k)(·|s)
               [π_k = current base policy provides the prior]
               [Q_k = frozen Q provides the guidance signal]
            3. Step env: (s, ã) → (r, s', done)

        # Gradient updates (update_per_iteration times, using the fresh batch)
        For each gradient step:
            4. Update live Q using TD(0) on the batch (s, ã, r, s', done)
               [TD target uses live Q directly — no target network]
            5. Train π_{k+1} by distilling: score-match on ã given s
               [policy learns to imitate the guided distribution]

        # Log metrics for this iteration
        6. Log q_learning/*, guidance/*, q_shift/*, distillation/* metrics

    # --- Policy improvement step transition ---
    Set π_{k+1} ← current distilled policy params
    Set Q_{k+1} ← current live Q params
    Log pi_step/id, pi_step/total_iterations
```

### 1.2 Key design decisions

#### 1.2.1 Batching via parallel environments (no replay buffer)

Each iteration steps `b` environments in parallel, producing a batch of `b` transitions `(s, ã, r, s', done)`. This batch is used directly for gradient updates — **no replay buffer is needed**.

- The `b` parallel envs provide decorrelated transitions (different states, different stochastic dynamics), which is sufficient for stable gradient estimates.
- `update_per_iteration` controls how many gradient steps are taken per env step (currently 4 in the reference config). Each gradient step uses the same batch of `b` transitions from that iteration.
- No transitions are stored across iterations. Each batch is used and discarded.

**TD(λ)**: If we later want TD(λ) > 0, we need a small trajectory buffer to compute λ-returns. The buffer stores the last `L` transitions per environment, where `L` is the truncation length for the λ-return. For TD(0) (the default), no buffer is needed.

| Parameter | Source | Meaning |
|-----------|--------|---------|
| `b` | `--num_envs` (existing flag) | Number of parallel environments |
| `update_per_iteration` | `--update_per_iteration` (existing, default 4) | Gradient steps per env step |
| `L` | Derived from `--td_lambda` | Trajectory buffer length (0 for TD(0)) |

#### 1.2.2 Frozen Q for guidance vs live Q for TD

Within a policy improvement step:
- **Frozen Q_k**: Snapshot of Q params taken at the start of the policy improvement step. Used **only** for MGMD guidance (computing ∇Q for the Langevin/guidance correction). Never updated.
- **Live Q**: Continuously updated via TD learning on the guided-policy transitions. Used for TD backups (no target network). At the end of the policy improvement step, this becomes Q_{k+1}.

The frozen Q_k provides a **stable guidance signal** throughout the policy improvement step — the actions sampled by MGMD are always guided by the same Q. Meanwhile the live Q adapts to the state distribution induced by those guided actions.

**Implementation**: Add `q_frozen_params` as a field in `Diffv2TrainState` (or as a separate attribute on the `DPMD` class). At the start of each policy improvement step, copy `q1_params → q_frozen_q1`, `q2_params → q_frozen_q2`. In the MGMD sampling functions, use `q_frozen_*` instead of the live Q params for guidance.

#### 1.2.3 No target networks

In the reference config, target networks (EMA) are used for computing TD backup targets:
```python
# Current code (line 824-825):
q1_target_mean, _ = self.agent.q(target_q1_params, next_obs, next_action)
q2_target_mean, _ = self.agent.q(target_q2_params, next_obs, next_action)
```

In soft-PI mode, we use the **live Q params directly** for TD backups:
```python
# Soft-PI mode:
q1_target_mean, _ = self.agent.q(q1_params, next_obs, next_action)
q2_target_mean, _ = self.agent.q(q2_params, next_obs, next_action)
```

This is simpler and avoids the lag introduced by Polyak averaging. Since we're on-policy (no stale replay data), the live Q should be a reasonable bootstrap target. If instability arises, we can re-introduce target networks in Phase 9.

#### 1.2.4 Policy initialization

At k=0, the base policy π_0 should:
1. Have a Gaussian prior energy function (use `--gaussian_prior_baseline` which already exists).
2. Use rescaled NormalCDF output so actions are uniform on [-1,1] (use `--latent_action_space` which already exists).

The combination `--gaussian_prior_baseline --latent_action_space` should produce π_0 that samples from Uniform[-1,1]^d.

#### 1.2.5 Policy improvement step transitions

At each transition (end of policy improvement step k):
1. `policy_params` already contains the distilled π_{k+1} (it was being trained throughout).
2. `q1_params`, `q2_params` already contain the live Q trained on guided-policy data.
3. Freeze new Q: `q_frozen_q1 ← q1_params`, `q_frozen_q2 ← q2_params`.
4. Reset the policy improvement step counter.
5. If using TD(λ), clear the trajectory buffer.

There is **no need to copy params** — the live policy IS the new base policy, and the live Q IS the new Q. The only action is freezing a new guidance Q snapshot.

### 1.3 New flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--soft_pi_mode` | bool | False | Enable soft policy iteration mode |
| `--iterations_per_pi_step` | int | 100000 | Number of iterations (env steps) per policy improvement step |
| `--num_pi_steps` | int | 10 | Number of policy improvement steps |
| `--td_lambda` | float | 0.0 | TD(λ) parameter (0 = TD(0), no trajectory buffer needed) |
| `--diagnostic_mode` | bool | False | Enable expensive diagnostic metrics |
| `--diagnostic_every` | int | 10000 | Interval for expensive diagnostics (iterations) |

Existing flags used as-is: `--num_envs` (batch size `b`), `--update_per_iteration` (gradient steps per env step).

### 1.4 File changes

| File | Change |
|------|--------|
| `scripts/train_mujoco.py` | Add new flags; pass `soft_pi_mode` config to trainer and algorithm |
| `relax/trainer/off_policy.py` | Add outer policy improvement step loop around the existing iteration loop. At each PI step boundary: freeze Q, log `pi_step/*` markers. |
| `relax/algorithm/dpmd.py` | Add `self.soft_pi_mode` flag. Add `q_frozen_q1`/`q_frozen_q2` to state or as class attributes. In `stateless_update`: skip Polyak updates, use live Q for TD targets, use frozen Q for guidance. Add `freeze_guidance_q()` method. |
| `relax/network/diffv2.py` | No structural changes needed (keep `Diffv2Params` as-is for backward compat; just set target = live at init). |

### Acceptance criteria
- [ ] With `--soft_pi_mode`, the training loop has an outer policy improvement step loop.
- [ ] At each PI step boundary, Q is frozen for guidance.
- [ ] Within a PI step, guided actions are sampled using frozen Q, live Q is updated via TD, policy is distilled from guided actions — all simultaneously in each iteration.
- [ ] No replay buffer; batches come from `b` parallel environments.
- [ ] No target networks; live Q is used for TD backups.
- [ ] `pi_step/id` and `pi_step/total_iterations` are logged at each transition.
- [ ] The reference config still works in legacy mode (no `--soft_pi_mode`).

---

<a id="phase-2"></a>
## Phase 2: Q-Learning Diagnostics

**Goal**: Throughout each policy improvement step, log metrics (per iteration) that tell us whether the live Q is converging on the guided policy's value function.

### 2.1 Cheap metrics (log every `update_log_n_step`)

These are computed from quantities already available in the Q update:

| Metric | Source | Implementation |
|--------|--------|----------------|
| `q_learning/bellman_error_mean` | `mean(Q̃(s,ã) - (r + γ Q̃(s',ã')))` | Already have `td_error1` in update; just expose the mean |
| `q_learning/bellman_error_std` | std of the above | Already have `td_error1_std`; rename |
| `q_learning/bellman_error_max` | max |td error| | Add `jnp.max(jnp.abs(td_error))` |
| `q_learning/td_loss_q1` | Already `q1_loss` | Rename |
| `q_learning/td_loss_q2` | Already `q2_loss` | Rename |
| `pi_step/bellman_error_start` | Bellman error at iteration 0 of current PI step | Store at PI step transition |
| `pi_step/bellman_error_current` | Current bellman error | Same as `bellman_error_mean` |

### 2.2 Implementation

Modify `stateless_update` to compute and return these quantities. Since Q and policy are updated simultaneously in each step, no special `q_only` mode is needed — the existing update function already does both.

The key change is renaming metrics and adding `bellman_error_max`.

### Acceptance criteria
- [ ] `q_learning/bellman_error_mean` is logged and visibly decreases over each policy improvement step.
- [ ] `pi_step/bellman_error_start` captures the error at the beginning of each policy improvement step.

---

<a id="phase-3"></a>
## Phase 3: Guidance-Accuracy Diagnostics

**Goal**: Throughout each policy improvement step, log metrics (per iteration) that measure how well MGMD approximates the ideal mirror-descent update.

### 3.1 Cheap metrics

These can be computed from quantities available during each update step:

| Metric | Computation | Cost |
|--------|-------------|------|
| `guidance/E_Q_base` | Sample N actions from π_k, evaluate Q_k, take mean | cheap (N=16 actions from base policy during update) |
| `guidance/E_Q_guided` | Q_k evaluated on the MGMD-guided action that was actually taken | cheap (already have this) |
| `guidance/delta_Q_guided` | `E_Q_guided − E_Q_base` | cheap |
| `guidance/grad_Q_norm_per_level` | During MGMD sampling, log `||∇_x Q||` at each diffusion level | cheap (gradient is already computed for guidance) |
| `guidance/correction_norm_per_level` | `||λ_t σ_t ∇_x Q||` | cheap |

### 3.2 Implementation

#### 3.2.1 E_Q_base

During each update step, after computing the MGMD-guided action, also sample a batch of actions from the base policy π_k (unguided DDIM) and evaluate the frozen Q_k on them:

```python
# In the update step:
base_actions = sample_base_policy(key, obs_batch)  # unguided DDIM
q_base = evaluate_Q(q_params, obs_batch, base_actions)
E_Q_base = jnp.mean(q_base)
```

This requires one extra forward pass through the diffusion model + Q network per update step. For a batch of 256, this is ~2x the sampling cost, which is acceptable for soft-PI mode.

#### 3.2.2 Per-level diagnostics

The MGMD sampling loop (in `stateless_get_action_mala_full` or `stateless_get_action_tfg_recur`) already computes `grad_q` at each level. To log per-level norms:

- Add accumulators inside the `body_fn` / `level_body` scan that collect `||grad_q||` and `||correction||` at each level.
- Return these as arrays of shape `[num_timesteps]`.
- Log as wandb histograms or as individual scalars for key levels (first, middle, last).

### Acceptance criteria
- [ ] `guidance/E_Q_base` and `guidance/E_Q_guided` are logged each iteration.
- [ ] `guidance/delta_Q_guided` is positive (guidance improves Q under frozen Q_k).
- [ ] Per-level gradient norms are visible in wandb.

---

<a id="phase-4"></a>
## Phase 4: Q-Shift Diagnostics

**Goal**: Track how much the frozen guidance Q_k diverges from the live Q as the policy improvement step progresses and the guided policy's state distribution shifts.

### 4.1 Cheap metrics

| Metric | Computation |
|--------|-------------|
| `q_shift/E_Q_frozen_under_guided` | Evaluate frozen Q_k on current (s, ã) transitions |
| `q_shift/E_Q_live_under_guided` | Evaluate live Q̃_k (being trained) on same transitions |
| `q_shift/delta` | `E_Q_live − E_Q_frozen` (starts at ~0, may drift as state distribution changes) |
| `q_shift/bellman_error_frozen_Q` | Bellman error of frozen Q_k on current transitions |

### 4.2 Implementation

The frozen Q_k snapshot already exists from Phase 1 (used for guidance). During each update step:

```python
# Evaluate frozen Q_k on current transitions
q_frozen = evaluate_Q(q_frozen_params, obs, guided_action)
# Evaluate live Q̃_k on current transitions
q_live = evaluate_Q(q_live_params, obs, guided_action)
q_shift_delta = jnp.mean(q_live) - jnp.mean(q_frozen)
```

The `q_frozen_params` are already stored as part of the policy improvement step setup (needed for MGMD guidance), so no additional storage is required.

### 4.3 File changes

| File | Change |
|------|--------|
| `relax/algorithm/dpmd.py` | Add frozen-Q evaluation in the update step; return q_shift metrics in info dict |
| `relax/trainer/off_policy.py` | Pass `q_frozen_params` to update if not already accessible |

### Acceptance criteria
- [ ] `q_shift/delta` starts at ~0 at the beginning of each policy improvement step and drifts over iterations.
- [ ] `q_shift/bellman_error_frozen_Q` increases over the policy improvement step (frozen Q becomes stale).
- [ ] The magnitude of `q_shift/delta` gives actionable signal for tuning η and `iterations_per_pi_step`.

---

<a id="phase-5"></a>
## Phase 5: Distillation Diagnostics

**Goal**: Track how well the new base policy π_{k+1} matches the MGMD-guided policy π̃_k.

### 5.1 Cheap metrics

| Metric | Computation |
|--------|-------------|
| `distillation/policy_loss` | Score-matching loss of π_{k+1} on MGMD-sampled actions (already `policy_loss`) |
| `distillation/E_Q_distilled` | Sample actions from π_{k+1}, evaluate Q̃_k |
| `distillation/E_Q_mgmd` | Q̃_k evaluated on MGMD actions (same as guided action Q) |
| `distillation/Q_gap` | `E_Q_distilled − E_Q_mgmd` (should be ≤ 0) |
| `distillation/action_mse` | `mean(||a_{π_{k+1}}(s) − ã||²)` for the same states |

### 5.2 Implementation

During each update step (Q and policy are updated simultaneously):

```python
# Sample from the new (being-distilled) policy
distilled_actions = sample_base_policy(key, obs_batch, policy_params=new_policy_params)
q_distilled = evaluate_Q(q_current_params, obs_batch, distilled_actions)
q_mgmd = evaluate_Q(q_current_params, obs_batch, guided_actions)
distillation_Q_gap = jnp.mean(q_distilled) - jnp.mean(q_mgmd)
action_mse = jnp.mean((distilled_actions - guided_actions) ** 2)
```

### Acceptance criteria
- [ ] `distillation/policy_loss` decreases over each policy improvement step.
- [ ] `distillation/Q_gap` is ≤ 0 and its magnitude decreases as distillation progresses.
- [ ] `distillation/action_mse` decreases over each policy improvement step.

---

<a id="phase-6"></a>
## Phase 6: Expensive / Debug-Mode Diagnostics

**Goal**: Implement expensive metrics that are only computed in `--diagnostic_mode` at `--diagnostic_every` intervals. These sacrifice speed for deep insight.

### 6.1 Q-Accuracy (expensive)

| Metric | Method |
|--------|--------|
| `debug/q_pred_vs_mc_return_corr` | Roll out π_k for full episodes from states in the current batch. Compute MC returns. Correlate with Q_k predictions. |
| `debug/q_pred_vs_mc_return_mse` | MSE of the above. |

**Implementation**: Add a `diagnostic_q_accuracy()` method that performs short MC rollouts (e.g., 100 steps) from a batch of states and compares to Q predictions.

### 6.2 Guidance Fidelity (expensive)

| Metric | Method |
|--------|--------|
| `debug/E_Q_ideal_md` | Importance-weighted estimate: sample many actions from π_k, reweight by `exp(η Q_k)` / Z, compute weighted mean Q |
| `debug/delta_Q_ideal` | `E_Q_ideal_md − E_Q_base` |
| `debug/guidance_efficiency` | `delta_Q_guided / delta_Q_ideal` |
| `debug/entropy_guided` | Sample M actions from MGMD for each of K states. Estimate entropy via k-NN or kernel methods. |
| `debug/entropy_base` | Same for base policy. |
| `debug/entropy_ideal_md` | `H(π_k) + η * E_{π*}[Q] - log Z` (Z estimated from importance weights) |
| `debug/entropy_gap` | `entropy_guided − entropy_ideal_md` |
| `debug/kl_guided_vs_ideal` | Approximate via: `E_{π̃}[log(π̃/π*)]` using kernel density estimates |

**Implementation for E_Q_ideal_md** (importance-weighted):

```python
# For each state s in a batch:
# 1. Sample N actions from π_k: a_1, ..., a_N
# 2. Compute Q_k(s, a_i) for each
# 3. Compute importance weights: w_i = exp(η * Q_k(s, a_i))
# 4. Normalize: w̃_i = w_i / sum(w_j)
# 5. E_Q_ideal_md(s) = sum(w̃_i * Q_k(s, a_i))
```

This requires N forward passes through the base policy + Q for each state. With N=64 and batch=32, that's 2048 forward passes — expensive but feasible for periodic diagnostics.

**Implementation for entropy estimation**:

Use the Kozachenko-Leonenko k-NN entropy estimator:
```python
# For each state s:
# 1. Sample M actions from the distribution (MGMD or base): a_1, ..., a_M
# 2. For each a_i, find distance to k-th nearest neighbor: r_k(a_i)
# 3. H ≈ d * mean(log(r_k)) + log(M-1) - digamma(k) + d * log(2)
```

With M=128 per state and batch=32, this is 4096 samples — again expensive but periodic.

### 6.3 Implementation plan

1. Add `diagnostic_step()` method to `OffPolicyTrainer` that is called every `diagnostic_every` steps.
2. This method calls algorithm-level diagnostic methods that return the expensive metrics.
3. Gate all expensive computation behind `self.diagnostic_mode`.

### Acceptance criteria
- [ ] `--diagnostic_mode` enables expensive metrics without affecting normal training speed.
- [ ] `debug/guidance_efficiency` ∈ [0, 1] (or at least close to 1 when guidance is working).
- [ ] `debug/entropy_gap` reveals whether guidance is collapsing.

---

<a id="phase-7"></a>
## Phase 7: Hyperparameter Sensitivity & Automation

**Goal**: Use the diagnostic tree to automatically suggest or tune key hyperparameters.

### 7.1 η (tfg_lambda) tuning

The optimal η balances:
- **Benefit**: `delta_Q_ideal` (increases with η)
- **Cost**: `q_shift_delta` (becomes more negative with η)

**Strategy**: Run a sweep over η with `--diagnostic_mode`, plotting:
- `guidance/delta_Q_guided` vs η
- `q_shift/delta` vs η
- Their sum (net benefit) vs η

The η that maximizes the sum is the optimal step size for the current policy improvement step.

### 7.2 Policy improvement step length tuning

Observe:
- `q_learning/bellman_error_mean` over iterations within a PI step → if it plateaus, Q has converged.
- `distillation/policy_loss` over iterations within a PI step → if it plateaus, distillation is done.
- `q_shift/delta` over iterations within a PI step → when it starts diverging sharply, the frozen guidance Q is too stale.

The PI step should end when **both** Q and distillation have converged, **or** when Q-shift becomes too large (whichever comes first).

**Strategy**: Add optional `--adaptive_pi_step_length` flag that ends a PI step when a convergence criterion is met (e.g., both bellman error and policy loss rates of decrease drop below threshold, or q_shift magnitude exceeds a threshold).

### 7.3 Later: automated tree-based prioritization

Once all metrics are logged, build a simple dashboard that:
1. Computes "sensitivity" of episode return to each factor.
2. Highlights the factor with the largest negative contribution.
3. Suggests which phase/hyperparameter to focus on.

This is aspirational and belongs in Phase 9+.

### Acceptance criteria
- [ ] η sweep produces a clear optimal η visible in wandb.
- [ ] `iterations_per_pi_step` can be tuned based on convergence plots.

---

<a id="phase-8"></a>
## Phase 8: Push for SOTA Episode Return

**Goal**: Using the diagnostic framework, iterate on the algorithm to maximize final episode return on all environments of interest, without worrying about sample efficiency.

### 8.1 Environment targets

Standard MuJoCo continuous control:
- HalfCheetah-v4
- Hopper-v4
- Walker2d-v4
- Ant-v4
- Humanoid-v4

### 8.2 Optimization loop

For each environment:

1. Run soft-PI with conservative settings (small η, many iterations per PI step).
2. Use diagnostic tree to identify the weakest link.
3. Fix that link (tune η, add more MALA steps, increase network capacity, etc.).
4. Repeat until episode return exceeds published SOTA.

### 8.3 Candidate improvements to try (ordered by diagnostic tree priority)

| If bottleneck is... | Try... |
|---------------------|--------|
| Q not converging (2.1) | TD-lambda, larger Q network, lower LR, more iterations per PI step |
| Guidance inaccurate (2.2) | More MALA steps, per-level η adaptation, more diffusion steps, recurrence |
| Entropy collapse (2.2.2) | Reduce η, add entropy bonus, use energy_multiplier < 1 |
| Q-shift too large (2.3) | Reduce η, fewer iterations per PI step, use more conservative step size |
| Distillation gap (2.4) | More iterations per PI step (more distillation steps), larger policy network, lower policy LR |

### Acceptance criteria
- [ ] Episode return matches or exceeds published SOTA on all 5 environments.

---

<a id="phase-9"></a>
## Phase 9: Re-introduce Sample Efficiency

**Goal**: Once SOTA return is achieved, bring back sample-efficiency tricks one at a time, measuring their impact on both sample efficiency and final return.

### 9.1 Tricks to re-introduce (in order)

1. **Replay buffer**: Allow reuse of transitions across PI steps.
2. **Off-policy reuse**: Update Q on old transitions between PI steps.
3. **Target networks with Polyak averaging**: Already implemented; tune τ.
4. **DSAC-T refinements**: Distributional critic, adaptive clipping, omega scaling.
5. **Critic normalization**: EMA or distributional V(s) for advantage normalization.
6. **Hypergradient tuning**: Automatic LR adaptation.

### 9.2 Methodology

For each trick:
1. Enable it.
2. Run on all environments.
3. Compare: (a) sample efficiency (return vs env steps), (b) final return, (c) diagnostic tree metrics.
4. Keep it only if it helps sample efficiency without hurting final return.

### Acceptance criteria
- [ ] Final return matches Phase 8 with fewer environment steps.

---

<a id="flag-reference"></a>
## Flag Reference

### New flags for soft-PI mode

| Flag | Type | Default | Phase | Description |
|------|------|---------|-------|-------------|
| `--soft_pi_mode` | bool | False | 0 | Enable soft policy iteration training loop |
| `--iterations_per_pi_step` | int | 100000 | 1 | Iterations (env steps) per policy improvement step |
| `--num_pi_steps` | int | 10 | 1 | Number of policy improvement steps |
| `--td_lambda` | float | 0.0 | 1 | TD(λ) parameter (0 = TD(0), no trajectory buffer) |
| `--diagnostic_mode` | bool | False | 6 | Enable expensive diagnostic metrics |
| `--diagnostic_every` | int | 10000 | 6 | Interval for expensive diagnostics (iterations) |
| `--adaptive_pi_step_length` | bool | False | 7 | End PI steps early based on convergence |
| `--pi_step_convergence_threshold` | float | 0.01 | 7 | Threshold for adaptive PI step termination |

### Existing flags to use in soft-PI mode

These are already used in the reference config and should be kept:

| Flag | Recommended setting | Why |
|------|-------------------|-----|
| `--dpmd_constant_weight` | True | No Q-reweighting in policy loss (already in ref config) |
| `--dpmd_no_entropy_tuning` | True | No exploration noise (already in ref config) |
| `--q_critic_agg` | mean | Q aggregation strategy (already in ref config) |
| `--mala_steps` | 2 | MGMD corrector steps (already in ref config) |
| `--mala_per_level_eta` | True | Per-level step size adaptation (already in ref config) |
| `--mala_adapt_rate` | 0.2 | MALA step size adaptation rate (already in ref config) |
| `--mala_guided_predictor` | True | Use guided predictor in MALA (already in ref config) |
| `--tfg_lambda` | 16.0 | Mirror descent step size η (already in ref config) |
| `--q_td_huber_width` | 30.0 | Huber loss width for TD (already in ref config) |
| `--update_per_iteration` | 4 | Gradient steps per env step (already in ref config) |
| `--lr_q` | 0.00015 | Q learning rate (already in ref config) |
| `--x0_hat_clip_radius` | 3.0 | Tweedie estimate clipping (already in ref config) |
| `--num_envs` | TBD | Parallel envs for decorrelated batches (**new usage**: replaces replay buffer) |
| `--gaussian_prior_baseline` | True | π_0 samples from Gaussian prior |
| `--latent_action_space` | True | Actions are uniform on [-1,1] at init |

---

<a id="file-level-change-map"></a>
## File-Level Change Map

This section maps each implementation phase to the specific files that need to be modified and the nature of the changes.

### `scripts/train_mujoco.py`
- **Phase 0**: Add `--soft_pi_mode` flag.
- **Phase 1**: Add `--iterations_per_pi_step`, `--num_pi_steps`, `--td_lambda` flags. Pass them to trainer/algorithm.
- **Phase 7**: Add `--adaptive_pi_step_length`, `--pi_step_convergence_threshold`.

### `relax/trainer/off_policy.py`
- **Phase 0**: Add `self.soft_pi_mode` attribute.
- **Phase 1**: Add outer policy improvement step loop around existing iteration loop. At PI step boundaries: call `algorithm.freeze_guidance_q()`, log `pi_step/*` markers. Remove replay buffer usage in soft-PI mode (batches come directly from parallel envs).
- **Phase 2**: Log PI step markers.
- **Phase 3**: Call algorithm's guidance diagnostic methods during each update step.
- **Phase 4**: Frozen Q already exists from Phase 1; add q_shift metric computation.
- **Phase 5**: Call distillation diagnostic methods.
- **Phase 6**: Add `diagnostic_step()` method, called conditionally.
- **Phase 7**: Add adaptive PI step termination logic.

### `relax/algorithm/dpmd.py`
- **Phase 0**: Add `self.soft_pi_mode` flag. Gate Polyak updates, reweighting, DSAC-T, critic normalization behind `not self.soft_pi_mode`. Return simplified `info` dict with new metric keys.
- **Phase 1**: Add `q_frozen_q1`/`q_frozen_q2` storage. Add `freeze_guidance_q()` method (copies live Q → frozen Q). In MGMD sampling: use frozen Q for guidance. In TD backups: use live Q directly (no target networks).
- **Phase 2**: Compute and return `bellman_error_mean`, `bellman_error_std`, `bellman_error_max`.
- **Phase 3**: Add `compute_guidance_diagnostics()` — samples from base and guided policies, computes `E_Q_base`, `E_Q_guided`, `delta_Q_guided`. Add per-level gradient norm tracking in sampling functions.
- **Phase 4**: Add frozen-Q evaluation in update step; return q_shift metrics in info dict.
- **Phase 5**: Compute distillation diagnostics during each update step.
- **Phase 6**: Add `compute_expensive_diagnostics()` — MC rollouts, importance-weighted ideal MD estimates, entropy estimation.

### `relax/algorithm/base.py`
- **Phase 1**: Add `freeze_guidance_q()` interface method.

### `relax/network/diffv2.py`
- **Phase 0/1**: No structural changes needed. `Diffv2Params` kept as-is for backward compat (target fields set = live fields in soft-PI mode).

### `relax/trainer/accumulator.py`
- **Phase 0**: No changes needed.

### `relax/utils/` (new file: `diagnostics.py`)
- **Phase 6**: Implement k-NN entropy estimator, importance-weighted expectation utilities, MC rollout helper.

---

## Summary of Execution Order

```
Phase 0  →  Simplify params & logging, add soft_pi_mode flag
Phase 1  →  Add outer policy improvement step loop + frozen Q guidance (the core change)
Phase 2  →  Wire up Q-learning diagnostics
Phase 3  →  Wire up guidance-accuracy diagnostics
Phase 4  →  Wire up Q-shift diagnostics
Phase 5  →  Wire up distillation diagnostics
   ↓
   At this point, run experiments to verify the loop works and
   diagnostics are informative. Iterate on η, iterations_per_pi_step, etc.
   ↓
Phase 6  →  Add expensive debug-mode diagnostics
Phase 7  →  Hyperparameter sensitivity analysis
Phase 8  →  Push for SOTA on all environments
Phase 9  →  Re-introduce sample efficiency tricks
```

Phases 0–5 are the critical path. Each can be implemented and tested in ~1–2 days. Phases 6–9 are iterative and may take longer depending on experimental results.
