# Why one MGMD config does not work on all six MuJoCo envs

Investigation of 2026-09-01. Everything below is measured from artefacts already on
disk (diagnostic snapshots, rollout mirrors, and the 100-episode action-selection
evaluations). No new training runs were needed to reach the conclusion.

---

## 0. Terminology

Terms are defined once here and used consistently afterwards.

| Term | Definition |
|---|---|
| **Noise level / ladder** | The diffusion schedule's `diffusion_steps` = `T` levels. Level `t=T-1` is noisiest, `t=0` cleanest. The sampler walks `t: T-1 -> 0`. |
| **`abar_t`** | `alphas_cumprod[t]`. The forward process is `x_t = sqrt(abar_t)*a + sqrt(1-abar_t)*eps`, `a` the clean action, `eps ~ N(0,I)`. |
| **`sigma_t`** | `sqrt(1-abar_t)`, the per-coordinate noise std at level `t`, in normalized action units where the action box is `[-1,1]`. |
| **MALA** | The Metropolis-adjusted Langevin chain run *at* each level, targeting that level's tilted density. This is what actually moves the sample when the predictor is `Identity`. |
| **Denoising predictor** | The transition *between* adjacent levels, applied after the MALA chain at that level. `Identity` leaves the sample where MALA put it; `DDPM_mean` replaces it with the guided DDPM posterior mean. |
| **`K`** = `--predictor_final_steps` | Apply the chosen predictor only to the last `K` transitions (`t < K`, the cleanest ones) and `Identity` to every noisier one. `K=0` disables the restriction (predictor everywhere). |
| **Tilt** | The target the chain samples: `pi_new ∝ pi_old^alpha * exp(beta*Q)`. With `--T 0 --eta X` the resolver gives `alpha=1, beta=X` (verified in `relax/cli/train_setup.py:resolve_cmd_params`). |
| **`sd_within`** | Std of `Q` over 16 iid sampler draws **at a fixed state**, then averaged over states. This is the value variation the tilt actually discriminates between. |
| **`sd_pooled`** | Std of `Q` over all (state, draw) pairs jointly. Dominated by how much states differ in value. This is what `--ema_advantage_normalization` divides the guidance `Q` by. |
| **`q_ratio`** | `sd_pooled / sd_within`. The factor by which the realized tilt departs from the intended one. |
| **`a_sd_within`** ("dispersion") | Std of the *action* over 16 iid sampler draws at a fixed state, averaged over coordinates and states. The policy's per-state stochasticity. |
| **Blindness** | `(true return change) / (return change the critic predicts)` for a given intervention. |

Scripts written for this investigation:

- `scripts/diagnose_env_action_sensitivity.py` — action/`Q` geometry per snapshot.
- `scripts/diagnose_critic_overestimation.py` — critic vs the value its own rollouts justify.
- `scripts/diagnose_ensemble_pessimism_gap.py` — `mean(Q_i) - min(Q_i)` at sampled actions.

---

## 1. Two premises confirmed, one refuted

### 1.1 `Identity_then_DDPM_mean` **is** `DDPM_mean` with `K=1` — confirmed

`relax/algorithm/mala_sampler.py` peels the final level out of the loop and applies
`ddpm_mean_step` at `t=0` only. The generalized path with `K=1` selects
`ddpm_mean_step` under `t_idx < 1`, i.e. also only `t=0`. `scripts/test_identity_then_ddpm_mean.py`
now asserts these produce **bit-identical** actions, acceptance rates and step sizes,
and that `K=T` is bit-identical to unrestricted `DDPM_mean`. Both pass.

### 1.2 `Identity` returns a *noisy* action — confirmed, but the magnitude is small

The `Identity` read-out returns the level-0 sample, i.e.

```
executed action = sqrt(abar_0)*a + sigma_0*eps  =  0.99876*a + 0.0499*eps   (T=40)
                                               =  0.99957*a + 0.0294*eps   (T=80)
```

So `Identity` does inject action noise, and `--fused_denoising` makes that same
sample the rollout action, the TD next-action and the distillation target at once.

**But this noise is negligible next to the policy's own spread.** Measured
`a_sd_within` is 0.25–0.41, so removing `sigma_0 = 0.05` in quadrature changes total
dispersion by **under 2%**:

| env | `a_sd_within` | `sigma_0` (T=40) | dispersion change if `K=1` |
|---|---:|---:|---:|
| Ant | 0.271 | 0.050 | −1.7% |
| HalfCheetah | 0.254 | 0.050 | −1.9% |
| Walker2d | 0.336 | 0.050 | −1.1% |
| Humanoid | 0.408 | 0.050 | −0.8% |
| Hopper | 0.405 | 0.050 | −0.8% |
| Swimmer | 0.376 | 0.050 | −0.9% |

**`K` is therefore not an exploration-noise knob.** It is a *greediness* knob: at a
fixed policy, `K=2` lands on actions whose `Q` is higher by 0.05–0.45 `sd_within`.

### 1.3 Ant does **not** have a larger action penalty — refuted

In the installed `gym/envs/mujoco/*_v3.py`, **every** env uses
`ctrl_cost_weight = 0.1`. (The 0.5 figure circulating in the literature is Ant-v4/v5.)
Control cost at `|a|=1` is `0.1 * act_dim`:

| env | act_dim | ctrl cost at \|a\|=1 | healthy reward | cost / healthy |
|---|---:|---:|---:|---:|
| Humanoid | 17 | 1.70 | 5.0 | 0.34 |
| **Ant** | 8 | **0.80** | **1.0** | **0.80** |
| HalfCheetah | 6 | 0.60 | — | — |
| Walker2d | 6 | 0.60 | 1.0 | 0.60 |
| Hopper | 3 | 0.30 | 1.0 | 0.30 |
| Swimmer | 2 | 0.20 | — | — |

Ant's coefficient is ordinary; what is distinctive is that its control cost is 80% of
its survival bonus — the worst ratio of the six. So the *intuition* that Ant punishes
action magnitude hardest is defensible, but not via a bigger coefficient. Consistently,
Ant's learned policy is the narrowest (`a_sd_within` 0.271, `|a|` 0.371).

---

## 2. Three candidate mechanisms measured and refuted

These are recorded because each was plausible and each is now excluded by data.

### 2.1 "Action jitter costs real value" — refuted

Converting `E_eps[Q(a + sigma_0*eps)] - Q(a)` into return units
(`dReturn = dQ*(1-gamma)*L`, `gamma=0.99`, `L=1000`):

| env | `dQ(sigma_0)` | dReturn | % of return |
|---|---:|---:|---:|
| HalfCheetah | −0.074 | −0.7 | 0.01% |
| Ant | −0.021 | −0.2 | 0.01% |
| Walker2d | −0.007 | −0.1 | 0.00% |

The one-step value cost is negligible **everywhere**, and it does not rank the envs
correctly (Hopper has the largest deployment gain from denoising but essentially zero
measured curvature cost). An earlier version of this analysis reported a
`sigma_tol` threshold — the noise level at which the second-order cost
`0.5*sigma^2*|tr H|` reaches 1% of `sd_within` — and claimed the envs "split cleanly".
**That threshold was built on an arbitrary 1% constant and is withdrawn.** It is
invariant to `Q` rescaling (numerator and denominator scale together, so the advantage
normalizer does not affect it), but invariance is not relevance: a single-step `Q`
perturbation cannot see a policy-level change that compounds over 1000 steps.

### 2.2 "Sharper sampling removes target smoothing and the critic overestimates" — refuted

Comparing `Q` on the snapshot's own replay batch against the value implied by that
run's own episodes, `V = (R/L)*(1-gamma^L)/(1-gamma)`, with `L` estimated from the
spacing of episode completions (an earlier version hardcoded `L=1000`, which is
invalid precisely for the collapsed, early-terminating runs):

| trained with | env | R | L | V_implied | Q | Q/V |
|---|---|---:|---:|---:|---:|---:|
| Identity | Ant | 2038 | 888 | 248 | 193 | 0.78 |
| Identity | HalfCheetah | 10226 | 1000 | 1023 | 1000 | 0.98 |
| Identity | Walker2d | 4223 | 1000 | 422 | 419 | 0.99 |
| `K=2` | Ant | 5575 | 1000 | 557 | 590 | 1.06 |
| `K=2` | HalfCheetah | 842 | 1000 | 84 | **−102** | −1.21 |
| `K=2` | Hopper | 1184 | 327 | 350 | **90** | 0.26 |

The collapsed `K=2` critics **under**estimate, not overestimate. The overestimation /
target-smoothing story is dead.

### 2.3 "A sharper sampler walks into critic-ensemble disagreement" — refuted

`--q_agg_sample mean` climbs the ensemble *mean* while the backup is hardcoded to
`min`, so disagreement is worth more to the sampler than to the backup. Measuring
`gap = mean_i Q_i - min_i Q_i` at the sampled actions as a function of `K`:

| env | gap @K=0 | @K=2 | @K=16 |
|---|---:|---:|---:|
| Ant | 1.073 | 0.941 | 0.887 |
| HalfCheetah | 0.644 | 0.612 | 0.588 |
| Humanoid | 0.830 | 0.822 | 0.806 |

The gap **shrinks** with `K`. Refuted.

Worth keeping anyway: the gap is *large in absolute terms and roughly `K`-independent*.
`gap/(1-gamma)` is 33–107 in `Q` units against `Q` values of 164–1034, i.e. the
`mean`-sample/`min`-backup inconsistency is a standing bias of up to 65% of `Q`. That
is a separate issue from env transfer, but it is worth a look.

---

## 3. The finding that reframes the problem

### 3.1 The critic is 16–675x blind to the value of sampler precision

Same policy, same states; only the read-out sharpness changes (`K=0 -> K=16`).
`Q` is from the snapshot; return is from 100 paired evaluation episodes.

| env | Q(K=0) | Q(K=16) | dQ | predicted dReturn | **true dReturn** | blindness |
|---|---:|---:|---:|---:|---:|---:|
| Hopper | 269.8 | 270.1 | 0.26 | 2.6 | **1766** | **675x** |
| Ant | 163.8 | 164.7 | 0.83 | 8.3 | **1415** | **170x** |
| Humanoid | 483.1 | 483.6 | 0.48 | 4.8 | 598 | 125x |
| HalfCheetah | 1033.5 | 1035.2 | 1.75 | 17.5 | 1665 | 95x |
| Walker2d | 432.7 | 433.0 | 0.33 | 3.3 | 53 | 16x |

The critic says sharpening the sampler is worth ~0.5%; the environment says it is worth
up to 146%. The reason is structural: `Q(s,a)` is the value of taking `a` and then
following the *behaviour* policy, which is still the noisy one. The benefit of precision
accrues because *every subsequent step* is also sharper — a policy-level change the
critic never evaluated.

**Consequence: no quantity derived from `Q` at a single state can tell the algorithm how
sharp its sampler should be.** This is why every one-step diagnostic in §2 came back
negligible while the rollout differences are enormous, and why tuning `eta` (which
scales the tilt, not the precision) cannot fix it.

### 3.2 Deployment sharpness helps every env; training sharpness helps only Ant

At deployment on Identity-trained 1M snapshots (100 paired episodes, at the trained `eta`):

| env | K=0 | K=1 | K=2 | K=16 |
|---|---:|---:|---:|---:|
| Ant (T=80, eta=32) | 1602 | 1707 | 2259 | **3949** |
| Hopper (T=40, eta=32) | 1469 | 1503 | 2659 | **3141** |
| HalfCheetah (T=40, eta=32) | 9179 | 9654 | 10113 | **10854** |
| Humanoid (T=40, eta=32) | 5090 | 5210 | 5162 | **5402** |
| Walker2d (T=40, eta=32) | 3651 | 3726 | 3872 | 3820 |
| Swimmer (T=40, eta=32) | 113 | 113 | 116 | **126** |

Monotone improvement, never a regression. But **training** with `K=2` (sweep of
2026-08-31, 1M steps, all else equal) does the opposite:

| env | Identity-trained | `K=2`-trained | ratio |
|---|---:|---:|---:|
| **Ant** | 2038 | **5575** | **2.74** |
| Humanoid | 3958 | 3062 | 0.77 |
| Hopper | 1590 | 1184 | 0.74 |
| Walker2d | 4223 | 1807 | 0.43 |
| Swimmer | 124 | 19 | 0.15 |
| HalfCheetah | 10226 | 842 | 0.08 |

`K=5` collapses all six, Ant included.

### 3.3 What training with `K=2` actually does: it collapses policy dispersion

Measured on the `K=2`-trained weights (per-slot `eta` recovered from `hp_pack_inline`):

| env | `a_sd_within` Identity-trained | `a_sd_within` `K=2`-trained | retained | return retained |
|---|---:|---:|---:|---:|
| Swimmer | 0.376 | **0.057** | 0.15 | 0.15 |
| HalfCheetah | 0.254 | **0.069** | 0.27 | 0.08 |
| **Ant** | 0.271 | **0.117** | 0.43 | **2.74** |
| Hopper | 0.405 | 0.277 | 0.68 | 0.74 |
| Humanoid | 0.408 | 0.294 | 0.72 | 0.77 |
| Walker2d | 0.336 | 0.244 | 0.73 | 0.43 |

Replay-buffer action diversity moves the same way (0.19x–0.79x of baseline; Swimmer
worst at 0.19x). Swimmer's policy is effectively a point mass — read out with its own
`K=2` sampler its dispersion is **0.000**.

Note this is a *training feedback* effect, not a sampling effect: at a fixed
Identity-trained policy, changing the read-out from `K=0` to `K=2` moves dispersion only
0.271 -> 0.258. The collapse comes from the loop *greedier sampler -> narrower
distillation targets -> narrower policy -> narrower targets*.

`Spearman(dispersion retained, return retained) = 0.43, p = 0.40` — so dispersion
collapse alone does **not** predict the outcome. What predicts it is dispersion collapse
*relative to the env's own optimum*: the two largest collapses (Swimmer 0.15,
HalfCheetah 0.27) are the two worst outcomes, while Ant — whose baseline was far below
its achievable score — is the one env for which narrowing was the correct move.

---

### 3.4 Ant is bimodal across seeds; its "failure" is largely a *reliability* problem

Scanning all 80 launch mirrors on disk (1154 `episode_returns.csv` files) for the best
per-env tail return, two launches appeared to clear a "good" bar on all six envs. That
was an artefact of taking the best slot per env: within a launch those are **six
different hyperparameter cells**. Resolving sweep 125
(`logs/slurm/20260726_023415`: `Identity`, `T=80`, `eta ∈ {16..256}`, `s_hat ∈ {0.2..0.8}`)
down to individual `(eta, s_hat)` cells, **the best single cell clears only 3/6**
(`eta=256, s_hat=0.8`). No single configuration on disk has ever worked on all six.

More important, the per-seed spread *inside one cell* is enormous on Ant and negligible
elsewhere:

| env | seed spread within a cell |
|---|---|
| **Ant** | up to **8.1x** (e.g. `eta=16, s_hat=0.2`: 4534, 2769, 812, 560) |
| HalfCheetah | 1.0–1.4x |

Ant's outcomes are **bimodal** — a "walking" mode near 4500–6300 and a collapsed mode
near 550–1200 — which is the well-known Ant local optimum. So Ant's mean return is a
mixture, and the right statistic is the **success rate**:

| config | Ant tail >= 4000 | median | max |
|---|---:|---:|---:|
| `DDPM_mean` `K=2` | **7/8 = 88%** | 6230 | 6772 |
| `Identity` (diagnostic snapshots) | 0/4 = 0% | 1986 | 2115 |
| `Identity` sweep125, best 3 cells | 5/12 = 42% | 1902 | 5047 |

Fisher exact, `K=2` vs pooled `Identity`: **p = 0.027**, odds ratio 15.4.

This reframes §3.2: `K=2` does not raise Ant's *ceiling* — `Identity` already reaches
~5000 in its good seeds — it raises the *probability of landing in the good mode* from
~30% to ~88%. Narrower dispersion makes the walking mode more reliable.

**Methodological consequence:** with 1–2 seeds and an 8x within-cell spread, an apparent
"Ant improvement" cannot be distinguished from seed luck. Several of the single-seed
comparisons in the sweep history are therefore uninformative about Ant, and future Ant
claims need success rates over >= 6–8 seeds, not means over 2.

---

## 4. Diagnosis

**The algorithm has no explicit control over policy dispersion, and the optimal
dispersion is strongly env-dependent.**

Dispersion is an emergent by-product of `eta`, `K`, `T`, the schedule and the
distillation loop. Ant's optimum is far narrower (≈0.12 gave 5575, vs 0.271 giving 2038)
than Humanoid/Hopper/Walker2d's (≈0.24–0.41). Any single configuration therefore lands
at the right dispersion for at most a subset of envs. Because of §3.1 the critic cannot
report this, so the algorithm cannot self-correct.

This is the user's third hypothesis — "some environments use the inaccuracy as necessary
exploration noise, whereas in Ant it is hurtful" — and it is the one the data supports.
It is also exactly the effect ED2 (arXiv:2111.15382, §4.3) reports for MuJoCo: removing
additive exploration noise **substantially improves Ant**, does nothing for
Hopper/Walker, and **hurts Humanoid**. SAC solves this class of problem with automatic
entropy tuning (arXiv:1812.05905); MGMD currently has no analogue.

### A second, independent scale bug

`--ema_advantage_normalization` divides the guidance `Q` by `sd_pooled`, but the KL cost
of the tilt depends on `sd_within`. Their ratio is strongly env-dependent, and it lines
up almost perfectly with how much more `eta` each env wants (the `eta` maximizing
deployment return over the trained `eta`, geometric mean over T and eta):

| env | `q_ratio` = `sd_pooled/sd_within` | wanted `eta` factor |
|---|---:|---:|
| HalfCheetah | 43.6 | 1.19 |
| Ant | 49.5 | 0.35 |
| Walker2d | 101.5 | 1.68 |
| Humanoid | 110.8 | 4.76 |
| Hopper | 125.9 | 6.73 |
| Swimmer | 183.7 | 22.63 |

`Spearman rho = 0.943, p = 0.005`. Switching to `--ema_within_advantage_normalization`
multiplies the effective tilt by `q_ratio`, which moves every env in the right
direction and shrinks the required per-env `eta` spread from **64x to 17x**.

Two caveats. First, the switch raises the effective tilt by a factor of ~91 on average,
so `eta` must be divided by roughly that; the one existing `ema_within` sweep
(2026-08-04) used `eta` 16–128, i.e. ~100x too strong, and its results fit that reading
(Swimmer reached 152, the best value seen anywhere and the env most under-tilted before;
Humanoid collapsed to 819, the most over-tilted after). That sweep was also confounded
with `--latent_action` and `--estimate_s_hat`. Second, a 17x residual spread means this
is a real improvement but not by itself a universal config.

Corroborating the same scale bug: with the pooled normalizer the tilt is invisible at
the clean end. `dlogp = eta*|dQ(sigma_0)|/sd_pooled` is 0.0005–0.07 for all six envs,
i.e. `<< 1`, so the final level is close to unguided.

---

## 5. Recommendations, in priority order

### P0 — Measure Ant as a success rate over >= 6 seeds before drawing any Ant conclusion

Free, and it gates everything else. Ant's within-cell seed spread is 8x and its outcome
distribution is bimodal (§3.4), so a 2-seed mean carries almost no information. Report
`fraction of seeds with tail >= 4000` for Ant, and keep means only for the five stable
envs. Concretely: any candidate config should be run with Ant packed at 8 seeds
(`--ablate env Ant-v3 --seeds-per-config 8`, 2 jobs at vmap x4, ~6–9 h) rather than
spending the same compute on more hyperparameter cells at 2 seeds.

Without this, the two arms in P1/P3 below cannot be told apart on the one env that
matters most.

### P1 — Add an explicit, automatically-tuned dispersion target (the real fix)

This directly addresses §4. The analogue of SAC's automatic entropy tuning: measure the
policy's within-state action dispersion online and adapt a scalar to hit a target,
instead of letting dispersion fall out of `eta`/`K`/`T`.

The measurement already exists. `--estimate_s_hat` "EMAs the log of the per-state action
sd over the K denoised actions" — that *is* `a_sd_within`. It requires
`--num_denoised_actions >= 2`, which is already supported. What is missing is using it
as a **control target** rather than only as a schedule shift.

Cheapest useful version, no new theory: a dual update on `eta`, i.e.
`log eta <- log eta + lr * (a_sd_estimate - a_sd_target)`, with `a_sd_target` a single
number shared by all envs.

- Effort: ~half a day to implement plus a test.
- Compute: one 6-env x 2-seed arm ~9 h wall (3 jobs at vmap x4, T=40).
- The decisive question it answers: **does a single dispersion target transfer across
  envs even though a single `eta` does not?** The measured optima (Ant ≈0.12,
  others ≈0.24–0.41) suggest not perfectly, so run 2–3 target values in the same sweep.

### P2 — Free win: sharpen the *deployment* read-out only

§3.2 shows sharper read-out never hurts at deployment and helps a lot on Ant/Hopper.
This needs **no retraining**: it changes only how the policy is evaluated. Report
evaluation episodes with `K` large (e.g. 8–16) while training with `Identity`.

- Effort: none for evaluation; the decoupling flags
  `--rollout_denoising_predictor` / `--rollout_predictor_final_steps` are implemented
  and tested if the *rollout* is also wanted sharper.
- Caveat: sharpening the rollout also narrows the data, which is the §3.3 failure mode,
  so prefer sharpening evaluation only until P1 controls dispersion.

### P3 — Re-run `ema_within` with `eta` rescaled by ~90

Tests the `rho = 0.943` finding at a sane operating point, unconfounded this time
(no `latent_action`, no `estimate_s_hat`). Ablate `eta` over roughly `{0.25, 1, 4}`.

- Effort: none (flag exists). Compute: ~9 h for 6 envs x 3 eta x 1 seed.
- Expected: cross-env optimal `eta` spread shrinks from 64x toward ~17x.

### Explicitly **not** recommended

**Do not pursue `--predictor_final_steps` as the route to a universal config.** It
narrows dispersion in all six envs (0.15x–0.73x) and only Ant benefits; `K=5` collapses
everything. Its Ant success is real but is a symptom of Ant's dispersion being wrong at
baseline — which P1 addresses directly and controllably.

---

### Revised reading of the `K` evidence

§3.4 changes what `K=2` is doing on Ant: it buys **reliability**, not a higher ceiling.
That is still a real effect (p = 0.027) and still comes with dispersion collapse in the
other five envs, so the conclusion of §5 is unchanged — but the mechanism to target is
"probability of the good mode", which is what a dispersion controller can regulate and
what `eta` alone cannot.

---

## 6. Open items and caveats

1. **`Q`-scale of `mean`-sample vs `min`-backup.** The `mean(Q_i) - min(Q_i)` gap is up
   to 65% of `Q` at the horizon and is roughly `K`-independent. Not the transfer story,
   but a large standing bias worth a dedicated look.
2. **Only 1–2 seeds per cell** for the training comparisons in §3.2/§3.3, and the
   Identity baselines come from a slightly older codebase snapshot than the `K=2` runs
   (same core config, `eta` 32/64, `T` 40/80). The effects are large but the seed count
   is thin, especially Walker2d (0.43) whose per-slot spread was wide (42–4726).
3. **Diagnostic bug found and fixed mid-investigation.** vmap-packed runs store per-slot
   `eta` in `hp_pack_inline` and leave the scalar `hparams` as `NaN`. Taking the `NaN`
   made every MALA proposal reject, so the sampler silently returned the clipped
   `N(0,I)` it started from — which looked exactly like "the energy network has
   degenerated" for all six envs. `diagnose_env_action_sensitivity.py` now falls back to
   `hp_pack_inline[eta][slot]` and raises on non-finite `eta`, and gained a `--slot`
   flag. **A conclusion drawn from the buggy run ("training with `K=2` degenerates the
   energy net") was withdrawn.** Any other script reading `hparams["eta"]` from a packed
   run needs the same guard.
4. **`L` from episode spacing** is a median over completions in a +/-5% step window; for
   runs with very few completed episodes it is noisy.
5. The deployment `eta` optimum is used in §4 as a proxy for the *training* `eta`
   optimum. They need not coincide.
6. **A background inventory subagent reported that `figures/`, `scripts/topsis_out/` and
   all `episode_returns.csv` mirrors were absent.** Direct checks contradict every one of
   those claims: `figures/` exists, `topsis_out/` exists, and there are **1154**
   `episode_returns.csv` files under 80 `logs/slurm/*/codebase/logs` trees. That report
   was discarded; nothing in this document depends on it. The sweep-ID range it listed
   (roughly 50–212 under the wandb offline base, 146 `sweep_*` dirs) is plausible but
   unverified.
7. The sweep-125 slot -> `(eta, s_hat)` mapping in §3.4 comes from each job's
   `--hp_pack_inline` matched to the exp dir by its `_s<seed>_` suffix, with the
   `episode_returns.csv` `seed` column read as the vmap slot index (per `AGENTS.md`).
   Ant showed 4 samples per cell against HalfCheetah's 2, so some Ant cells may be
   double-counted across jobs; the bimodality and the spread ratio are unaffected, but
   the exact success-rate denominators for sweep 125 should be re-derived from
   `config.yaml` if they are load-bearing.
