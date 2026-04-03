# Diagnostic Metrics Tree: Episode Return Decomposition

This document defines a hierarchical tree of logged quantities rooted at **episode return**. The goal is that every parent node can be analytically (or approximately) reconstructed from its children, so that sensitivity to each factor is transparent and prioritization of engineering effort is straightforward.

Each metric is annotated with:
- **Cost**: `cheap` (negligible overhead, log every update) or `expensive` (requires extra forward passes / sampling; log only in debug mode).
- **Timeline**: `soon` (directly described in `assumptions.md` and needed for the first implementation) or `later` (natural extension of a branch not yet prioritized).

---

## 0. Notation

| Symbol | Meaning |
|--------|---------|
| π_k | Base (distilled) policy at iteration k |
| π̃_k | MGMD-guided policy (mirror-descent update of π_k using Q_k) |
| Q_k | Frozen Q-function snapshot at start of iteration k (used for MGMD guidance) |
| Q̃_k | Live Q-function being trained on π̃_k transitions during iteration k (initialized from Q_k) |
| η | Mirror-descent step size (`tfg_lambda`) |
| π* | Ideal mirror-descent update: π*_k(a|s) ∝ π_k(a|s) exp(η Q_k(s,a)) |
| J(π) | Expected episode return of policy π |

---

## 1. Top-Level Decomposition

```
J(π̃_k)  =  J(π_k)  +  ΔJ_policy_improvement
         =  J(π_k)  +  [ΔJ_ideal_md  -  ΔJ_guidance_gap  -  ΔJ_Q_shift  -  ΔJ_distillation_gap]
```

where:

| Term | Definition |
|------|-----------|
| **J(π_k)** | Episode return of the current base policy |
| **ΔJ_ideal_md** | Improvement predicted by the ideal mirror-descent update: V^{π*_k}(s_0) − V^{π_k}(s_0) |
| **ΔJ_guidance_gap** | Loss due to MGMD sampling not perfectly recovering π* |
| **ΔJ_Q_shift** | Loss due to state-distribution shift: difference between old-Q-predicted improvement and actual improvement under the new state distribution |
| **ΔJ_distillation_gap** | Loss due to imperfect distillation of π̃_k into the new base policy π_{k+1} |

This is the master equation. Every branch below expands one of these terms.

---

## 2. The Metrics Tree

```
episode_return  [cheap, soon]
├── base_policy_return  [cheap, soon]
│   ├── mean_episode_reward_base  [cheap, soon]
│   └── episode_length_base  [cheap, soon]
│
├── ΔJ_policy_improvement  [cheap (approx), soon]
│   │
│   ├── 2.1  Q-Accuracy Branch  (Is Q_k correct for π_k?)
│   │   ├── bellman_error_mean  [cheap, soon]
│   │   ├── bellman_error_std  [cheap, soon]
│   │   ├── bellman_error_max  [cheap, soon]
│   │   ├── td_loss_q1  [cheap, soon]
│   │   ├── td_loss_q2  [cheap, soon]
│   │   ├── q_pred_vs_mc_return_corr  [expensive, soon]
│   │   │   (correlation between Q_k(s,a) and actual MC return from s,a under π_k)
│   │   ├── q_pred_vs_mc_return_mse  [expensive, soon]
│   │   └── q_overestimation_bias  [expensive, later]
│   │       (E[Q_k(s,a) - Q^{π_k}(s,a)] estimated via MC rollouts)
│   │
│   ├── 2.2  Guidance-Accuracy Branch  (Does MGMD sample from π*?)
│   │   │
│   │   ├── 2.2.1  Q-Improvement Metrics
│   │   │   ├── E_Q_base  [cheap, soon]
│   │   │   │   E_{a ~ π_k}[Q_k(s,a)] averaged over states in batch
│   │   │   ├── E_Q_guided  [cheap, soon]
│   │   │   │   E_{a ~ π̃_k}[Q_k(s,a)] averaged over states in batch
│   │   │   ├── E_Q_ideal_md  [expensive, soon]
│   │   │   │   E_{a ~ π*}[Q_k(s,a)] via importance-weighted samples from π_k
│   │   │   ├── delta_Q_guided  [cheap, soon]
│   │   │   │   = E_Q_guided − E_Q_base  (actual improvement from MGMD guidance)
│   │   │   ├── delta_Q_ideal  [expensive, soon]
│   │   │   │   = E_Q_ideal_md − E_Q_base  (improvement mirror descent predicts)
│   │   │   └── guidance_efficiency  [expensive, soon]
│   │   │       = delta_Q_guided / delta_Q_ideal  (fraction of ideal improvement achieved)
│   │   │
│   │   ├── 2.2.2  Distributional Fidelity Metrics
│   │   │   ├── kl_guided_vs_ideal  [expensive, soon]
│   │   │   │   KL(π̃_k || π*)  — the gold standard; requires density of π̃_k
│   │   │   │   Approximation: use reverse-KL importance sampling estimator
│   │   │   │     KL ≈ E_{a~π̃}[log π̃(a|s) - log π*(a|s)]
│   │   │   │   We lack π̃ density, so we estimate via:
│   │   │   │     (a) Kernel density estimation on MGMD samples
│   │   │   │     (b) Sliced-Wasserstein distance as a proxy
│   │   │   │     (c) Classifier two-sample test (train a small net to distinguish π̃ vs π* samples)
│   │   │   │
│   │   │   ├── kl_tilt_base  [expensive, soon]  (already implemented: mirror_descent/kl_tilt_base)
│   │   │   │   KL(π̃_k || π_k) — how far the guided policy has moved from base
│   │   │   │
│   │   │   ├── entropy_guided  [expensive, soon]
│   │   │   │   H(π̃_k) estimated from multi-sample batches
│   │   │   │   Detects "collapse" where guidance reduces entropy more than π* would
│   │   │   │
│   │   │   ├── entropy_ideal_md  [expensive, soon]
│   │   │   │   H(π*_k) = H(π_k) + η E_{π*}[Q] − log Z  (analytically from definition)
│   │   │   │
│   │   │   ├── entropy_base  [expensive, soon]
│   │   │   │   H(π_k) estimated from base policy samples
│   │   │   │
│   │   │   └── entropy_gap  [expensive, soon]
│   │   │       = entropy_guided − entropy_ideal_md
│   │   │       Negative → guidance is collapsing; positive → guidance is too diffuse
│   │   │
│   │   ├── 2.2.3  Per-Diffusion-Level Guidance Diagnostics
│   │   │   ├── grad_Q_norm_per_level  [cheap, soon]
│   │   │   │   ||∇_x Q(x̂_0(x_t, t))||  at each noise level t
│   │   │   ├── guided_eps_correction_norm_per_level  [cheap, soon]
│   │   │   │   ||λ_t σ_t ∇_x Q||  — the actual correction applied to noise pred
│   │   │   ├── base_eps_norm_per_level  [cheap, later]
│   │   │   │   ||ε_θ(x_t, t)||  — magnitude of base model prediction
│   │   │   ├── guidance_to_base_ratio_per_level  [cheap, later]
│   │   │   │   ratio of correction to base, per level; if >> 1, guidance dominates
│   │   │   └── mala_acceptance_rate_per_level  [cheap, soon]
│   │   │       (already partially implemented)
│   │   │
│   │   └── 2.2.4  MALA-Specific Diagnostics (when using energy-mode MGMD)
│   │       ├── mala_acceptance_rate  [cheap, soon]  (already implemented)
│   │       ├── mala_eta_scale  [cheap, soon]  (already implemented)
│   │       ├── mala_proposal_energy_delta  [cheap, later]
│   │       │   E[E(x') − E(x)] for proposals — indicates quality of gradient
│   │       └── mala_step_size_vs_beta  [cheap, later]
│   │           η_k / β_t  — effective step size relative to noise schedule
│   │
│   ├── 2.3  Q-Shift Branch  (Does frozen guidance Q diverge from live Q?)
│   │   │   Within each iteration, Q_k is frozen (used for guidance) while Q̃_k
│   │   │   is trained on guided-policy transitions. Both Q and policy update
│   │   │   simultaneously; this branch tracks how the frozen guidance signal
│   │   │   drifts from the live Q as the state distribution evolves.
│   │   ├── E_Q_frozen_under_guided  [cheap, soon]
│   │   │   E_{(s,a) ~ π̃_k}[Q_k(s,a)]  — frozen guidance Q on current transitions
│   │   ├── E_Q_live_under_guided  [cheap, soon]
│   │   │   E_{(s,a) ~ π̃_k}[Q̃_k(s,a)]  — live Q (being trained) on same transitions
│   │   ├── q_shift_delta  [cheap, soon]
│   │   │   = E_Q_live − E_Q_frozen
│   │   │   Starts at ~0 (Q̃ initialized from Q_k) and drifts as the
│   │   │   guided policy's state distribution diverges from π_k's
│   │   ├── q_shift_bellman_error_frozen  [cheap, soon]
│   │   │   Bellman error of frozen Q_k on current guided-policy transitions
│   │   │   Increases over the iteration → frozen Q becoming stale
│   │   ├── state_distribution_divergence  [expensive, later]
│   │   │   Some measure of d(ρ^{π̃_k}, ρ^{π_k}) — e.g., MMD on state batches
│   │   └── eta_sensitivity  [expensive, later]
│   │       Sweep η at fixed Q and measure q_shift_delta as function of η
│   │       Helps determine optimal step size
│   │
│   └── 2.4  Distillation Branch  (Is the new base policy close to π̃_k?)
│       ├── distillation_policy_loss  [cheap, soon]
│       │   Score-matching / ε-MSE loss when training π_{k+1} on actions from π̃_k
│       ├── E_Q_distilled  [cheap, soon]
│       │   E_{a ~ π_{k+1}}[Q̃_k(s,a)]  — how well the distilled policy performs under Q̃
│       ├── E_Q_mgmd  [cheap, soon]
│       │   E_{a ~ π̃_k}[Q̃_k(s,a)]  — how well the MGMD guided policy performs under Q̃
│       ├── distillation_Q_gap  [cheap, soon]
│       │   = E_Q_distilled − E_Q_mgmd  (should be ≤ 0; magnitude = distillation cost)
│       ├── kl_distilled_vs_guided  [expensive, later]
│       │   KL(π_{k+1} || π̃_k) — the true distillation KL
│       └── action_mse_distilled_vs_guided  [cheap, soon]
│           Mean ||a_{π_{k+1}} − a_{π̃_k}||² over matched states
│           (Cheap proxy for distributional closeness)
│
└── 3.  Auxiliary / Sanity-Check Metrics
    ├── 3.1  Action Distribution Health
    │   ├── mean_abs_action  [cheap, soon]
    │   ├── std_action  [cheap, soon]
    │   ├── clip_frac  [cheap, soon]
    │   └── action_entropy_estimate  [expensive, later]
    │
    ├── 3.2  Training Stability
    │   ├── q_grad_norm  [cheap, later]
    │   ├── policy_grad_norm  [cheap, later]
    │   ├── q_param_norm  [cheap, later]
    │   ├── policy_param_norm  [cheap, later]
    │   └── q_pred_variance_across_ensemble  [cheap, later]
    │       (|Q1 − Q2| as proxy for epistemic uncertainty)
    │
    └── 3.3  Policy Improvement Step Markers
        ├── pi_step_id  [cheap, soon]
        │   Integer identifying current policy improvement step (k = 0, 1, 2, ...)
        ├── pi_step_iteration  [cheap, soon]
        │   Iterations elapsed within current PI step
        ├── pi_step_bellman_error_start  [cheap, soon]
        │   Bellman error at iteration 0 of current PI step (for convergence tracking)
        └── pi_step_bellman_error_current  [cheap, soon]
            Bellman error now (watch it decrease over the PI step)
```

---

## 3. Analytical Relationships Between Nodes

### 3.1 Episode Return ↔ Policy Improvement

```
J(π̃_k) = J(π_k) + ΔJ_policy_improvement
```

Where `ΔJ_policy_improvement` decomposes (approximately, via performance-difference lemma) as:

```
ΔJ_policy_improvement ≈ (1/(1−γ)) * E_{s ~ ρ^{π̃_k}} [ E_{a ~ π̃_k}[Q_k(s,a)] − E_{a ~ π_k}[Q_k(s,a)] ]
                        + correction_due_to_Q_shift
                        + correction_due_to_distillation
```

### 3.2 Ideal Mirror-Descent Improvement

The ideal MD update π* ∝ π_k exp(η Q_k) gives:

```
ΔJ_ideal_md = (1/(1−γ)) * E_{s ~ ρ^{π_k}} [ E_{a ~ π*}[Q_k(s,a)] − E_{a ~ π_k}[Q_k(s,a)] ]
```

This is upper-bounded by `η * Var_{a ~ π_k}[Q_k(s,a)]` (for small η) and provides the ceiling for what a single iteration can achieve.

### 3.3 Guidance Gap

```
ΔJ_guidance_gap = ΔJ_ideal_md − actual_Q_improvement_from_guidance
                = (1/(1−γ)) * E_s [ E_{π*}[Q_k] − E_{π̃_k}[Q_k] ]
```

This is driven by `kl_guided_vs_ideal` (the KL between the MGMD samples and the ideal MD update).

### 3.4 Q-Shift

```
ΔJ_Q_shift = (1/(1−γ)) * E_{s ~ ρ^{π̃_k}} [ Q̃_k(s, π̃_k(s)) − Q_k(s, π̃_k(s)) ]
           = (1/(1−γ)) * q_shift_delta
```

### 3.5 Sensitivity of Episode Return to η

```
∂J/∂η ≈ (1/(1−γ)) * E_s [ Var_{a~π_k}[Q_k(s,a)] ] − (∂/∂η) ΔJ_Q_shift(η)
```

The first term is always positive (more guidance = more improvement under old Q). The second term captures the cost: larger η → larger state distribution shift → more Q error. The optimal η balances these.

---

## 4. Cheap vs Expensive Summary Table

### Cheap Metrics (log every N update steps during normal training)

| Metric | Tree Node | Timeline |
|--------|-----------|----------|
| `pi_step/id` | 3.3 | soon |
| `pi_step/iteration` | 3.3 | soon |
| `pi_step/bellman_error_start` | 3.3 | soon |
| `pi_step/bellman_error_current` | 3.3 | soon |
| `q_learning/bellman_error_mean` | 2.1 | soon |
| `q_learning/bellman_error_std` | 2.1 | soon |
| `q_learning/td_loss_q1` | 2.1 | soon |
| `q_learning/td_loss_q2` | 2.1 | soon |
| `guidance/E_Q_base` | 2.2.1 | soon |
| `guidance/E_Q_guided` | 2.2.1 | soon |
| `guidance/delta_Q_guided` | 2.2.1 | soon |
| `q_shift/E_Q_frozen_under_guided` | 2.3 | soon |
| `q_shift/E_Q_live_under_guided` | 2.3 | soon |
| `q_shift/delta` | 2.3 | soon |
| `q_shift/bellman_error_frozen_Q` | 2.3 | soon |
| `distillation/policy_loss` | 2.4 | soon |
| `distillation/E_Q_distilled` | 2.4 | soon |
| `distillation/E_Q_mgmd` | 2.4 | soon |
| `distillation/Q_gap` | 2.4 | soon |
| `distillation/action_mse` | 2.4 | soon |
| `guidance/grad_Q_norm_per_level` | 2.2.3 | soon |
| `guidance/correction_norm_per_level` | 2.2.3 | soon |
| `MALA/acceptance_rate` | 2.2.4 | soon |
| `MALA/eta_scale` | 2.2.4 | soon |
| `act/mean_abs_action` | 3.1 | soon |
| `act/std_action` | 3.1 | soon |
| `act/clip_frac` | 3.1 | soon |
| `sample/episode_return` | root | soon |
| `sample/episode_length` | root | soon |

### Expensive Metrics (log only in debug/diagnostic mode)

| Metric | Tree Node | Timeline |
|--------|-----------|----------|
| `debug/q_pred_vs_mc_return_corr` | 2.1 | soon |
| `debug/q_pred_vs_mc_return_mse` | 2.1 | soon |
| `debug/E_Q_ideal_md` | 2.2.1 | soon |
| `debug/delta_Q_ideal` | 2.2.1 | soon |
| `debug/guidance_efficiency` | 2.2.1 | soon |
| `debug/kl_guided_vs_ideal` | 2.2.2 | soon |
| `debug/kl_tilt_base` | 2.2.2 | soon |
| `debug/entropy_guided` | 2.2.2 | soon |
| `debug/entropy_ideal_md` | 2.2.2 | soon |
| `debug/entropy_base` | 2.2.2 | soon |
| `debug/entropy_gap` | 2.2.2 | soon |
| `debug/q_overestimation_bias` | 2.1 | later |
| `debug/state_distribution_divergence` | 2.3 | later |
| `debug/eta_sensitivity` | 2.3 | later |
| `debug/kl_distilled_vs_guided` | 2.4 | later |
| `debug/action_entropy_estimate` | 3.1 | later |

---

## 5. How to Use This Tree for Prioritization

1. **Start at the root**: observe `episode_return`. If it is not improving across policy improvement steps, drill down.

2. **Check Q-accuracy (2.1)**: If `bellman_error_mean` is not decreasing over iterations within a PI step, the live Q is not converging → fix Q learning (learning rate, TD-lambda, network capacity, more iterations per PI step, etc.).

3. **Check guidance accuracy (2.2)**: If Q is good but `delta_Q_guided` is much less than `delta_Q_ideal`, guidance is the bottleneck → improve MGMD (more MALA steps, better step sizes, more diffusion steps, etc.).

4. **Check entropy gap (2.2.2)**: If `entropy_gap < 0`, guidance is collapsing the distribution → reduce η or improve the sampling procedure.

5. **Check Q-shift (2.3)**: If `q_shift/bellman_error_frozen_Q` grows large or `q_shift/delta` drifts significantly, the frozen guidance Q_k is stale relative to the live Q → reduce η, use fewer iterations per PI step, or use more conservative step sizes.

6. **Check distillation (2.4)**: If `distillation_Q_gap` is large in magnitude, distillation is losing quality → train longer, increase network capacity, or use a better distillation loss.

7. **Across PI steps**: the `pi_step/*` markers let you see convergence speed within each policy improvement step and decide when to transition. Plot bellman error over iterations to see if `iterations_per_pi_step` is sufficient.
