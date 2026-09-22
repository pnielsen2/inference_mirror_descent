# Critical response: why Ant and the other environments prefer opposite sampler behavior

Date: 2026-09-01

This note re-checks the claims in `notes/env_transfer_investigation.md`, the earlier analysis in this conversation, the actual launch snapshots for sweeps 158/159/161 and 210/211/212, the 100-episode paired snapshot evaluations, and additional snapshot-only diagnostics.

## Executive answer

The shortest defensible summary is:

- **With Identity training, Ant is under-sharpened.** Its policy keeps more conditional action dispersion and pays an unusually large control-cost burden. Ant uses `ctrl_cost_weight=0.5` over eight actuators, and an Identity-trained replay batch paid about **0.84 control cost per step**, versus 0.25 for HalfCheetah and 0.31 for Humanoid. Small action errors can also destabilize a gait or cause termination. Sharpening therefore helps Ant more than most tasks.
- **With final-2 training, several other environments are over-sharpened.** Because the same sharpened sample becomes the rollout action, TD next-action, and distillation target, a small one-step contraction is fed back recursively. HalfCheetah and Swimmer can become nearly deterministic at each state and lock onto a bad action mode. Hopper and some Walker seeds show a second failure involving critic/occupancy instability rather than complete entropy collapse.
- **Final-2 does not merely remove a fixed amount of exploration noise.** Early in training it is only mildly sharper and learns faster than final-5/full DDPM. As the learned policy narrows, the Tweedie posterior-mean map becomes increasingly contractive; the contraction can become nonlinear and abrupt. That produces rise-then-collapse rather than a simple early plateau.
- **No existing frozen-policy evaluation setting solves the stated problem.** Across all 312 common `(training eta, diffusion steps, final-k, sampling eta)` settings in the saved 100-episode grid, none reached both Ant > 6000 and Humanoid > 5000. Maximum Ant was 3949. Deployment sharpening is useful, but it is not the missing complete solution.

The most promising universal mechanism is therefore not a fixed `K`, a fixed absolute action-SD target, or an eta-only controller. It is:

1. keep final-2-like sharpening;
2. impose a **minimum conditional-diversity floor** using a direct actuator such as Identity/DDPM blending, explicit output noise, or positive/adaptive `T`;
3. adapt `eta` separately to a KL/update-size target;
4. separate rollout, TD-target, and distillation predictors.

A floor is preferable to forcing every environment to the same dispersion: it would be inactive for successful Ant (`a_sd_within ~= 0.11`) but would activate before HalfCheetah/Swimmer approach zero.

---

## 1. Is `Identity_then_DDPM_mean` really final-1 DDPM?

**Yes, at the implementation level.**

In `relax/algorithm/mala_sampler.py`, `Identity_then_DDPM_mean` uses Identity between all noise levels and applies `ddpm_mean_step` only for the final level-0-to-clean readout. Generic `DDPM_mean` with `predictor_final_steps=1` selects the same operation at `t_idx < 1`. At `t=0`, the DDPM coefficients give

\[
\mu_0 = \hat x_0,
\qquad
\hat x_0 = \frac{x_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon_g(x_t)}{\sqrt{\bar\alpha_t}}.
\]

The chain state, MALA acceptances, and adapted MALA scales are identical to Identity up to that final readout. The live regression test also checks bit identity between the two final-1 spellings.

### Correction to my earlier performance claim

I previously called sweeps 158/159/161 a “matched one-seed” comparison. That wording was wrong.

The code snapshots are operationally well controlled:

- sweeps 158 and 159 have identical source;
- between sweep 159 and 161, the executable algorithm change is the addition of `Identity_then_DDPM_mean`; the `noise_schedule.py` change is documentation only for these cosine runs;
- all substantive CLI hyperparameters match.

But the seeds do **not** match:

| Run family | Ant seeds | Humanoid seeds | Execution |
|---|---|---|---|
| Identity sweep 159 | 0, 1, 2 for eta 32, 64, 128 | 3, 4, 5 | standalone |
| Final-1 sweep 161 | 6, 7, 8 for eta 32, 64, 128 | 9, 10, 11 | vmap-packed |

Thus the large training-return differences cannot isolate the endpoint predictor from random-seed variation. This matters because current final-2 runs are visibly bifurcating: at 80 steps/eta 64, HalfCheetah ended at -248 and 9034 across two seeds, while Walker ended at 84 and 4567.

### The genuinely paired frozen-policy test

Holding policy parameters, critic parameters, reset seeds, and sampling eta fixed, and changing only Identity to final-1 gave:

| Environment | eta | Identity | Final-1 | Paired delta | 95% paired bootstrap CI |
|---|---:|---:|---:|---:|---:|
| Ant, 80 steps | 32 | 1602 | 1707 | +106 | [-225, 426] |
| Ant, 80 steps | 64 | 2620 | 2766 | +147 | [-146, 439] |
| Humanoid, 80 steps | 32 | 1766 | 1595 | -172 | [-408, 61] |
| Humanoid, 80 steps | 64 | 5122 | 4983 | -139 | [-440, 164] |

Every interval includes zero. Therefore:

- the endpoint actions definitely differ;
- final-1 can alter trajectories;
- but the available data do **not** establish that final-1 itself caused the 1651-to-4517 Ant training difference;
- random seed could explain much or all of that one-seed comparison;
- a compounding training feedback effect remains plausible, but needs same-seed training arms.

The residual level-0 noise is also small relative to a healthy policy’s total conditional spread. On Identity snapshots, final-1 reduced conditional action SD by only about 1–3%. That small difference can compound during training, but it does not justify attributing a one-seed 2000-point difference to a single deployment readout.

---

## 2. Why final-2 can destabilize instead of merely plateauing

A fixed exploration-noise reduction would normally suggest faster convergence to a possibly suboptimal plateau. That is not the system implemented here.

Under `--fused_denoising`, the same sampled action is reused as:

1. the action executed in the environment;
2. the TD next-action;
3. the supervised distillation target.

Consequently the sampling distribution changes the future policy, replay distribution, and critic target. Final-2 is part of a recursive operator, not a fixed observation-noise filter.

### Posterior-mean contraction becomes stronger as the policy narrows

For a local Gaussian policy with variance `v`, forward noise

\[
Y=\sqrt{\bar\alpha}X+\sqrt{1-\bar\alpha}\epsilon
\]

and an exact posterior-mean teacher, the teacher variance is

\[
f(v)=\operatorname{Var}(\mathbb E[X\mid Y])
=\frac{\bar\alpha v^2}{\bar\alpha v+1-\bar\alpha}.
\]

Hence

\[
\frac{f(v)}{v}=\frac{\bar\alpha v}{\bar\alpha v+1-\bar\alpha}<1.
\]

Near zero, `f(v)` is proportional to `v^2`, so contraction accelerates. The real sampler also includes Q guidance, clipping, and MALA, so this is an explanatory local model rather than a proof. The observed phase transition nevertheless matches it closely.

### HalfCheetah is an actual rise-then-collapse case

For the 80-step/eta-32 final-2 run, slot 0:

| Step | Trailing return | Identity-readout SD | Final-2-readout SD | Policy epsilon-MSE | Clean MALA scale |
|---:|---:|---:|---:|---:|---:|
| 400k | 5595 | 0.132 | 0.123 | 0.078 | 11.65 |
| 500k | 6539 | 0.042 | **0.0049** | **0.0017** | **0.92** |
| 600k | -303 | 0.039 | **0.00044** | **0.00014** | 0.86 |

This directly answers the plateau question:

- at 500k the policy is still scoring 6539;
- its final-2 conditional dispersion has already fallen by about 25x;
- by 600k return has collapsed below zero;
- both eta-32 seeds exhibit the return collapse.

The Identity view shows that the learned level-0/MALA distribution itself narrows from 0.132 to roughly the residual noise floor of 0.04. Final-2 then amplifies that narrowing by another factor of 8 at 500k and nearly 90 at 600k.

MALA acceptance stays almost exactly 0.574 throughout, so acceptance is not a health diagnostic. Robbins-Monro adaptation maintains acceptance by shrinking the clean-level step size 12.7x. This removes corrector movement precisely when the distribution is becoming narrow. Expected squared jump distance or accepted-move covariance should be monitored instead.

### Eta cannot reopen the collapsed policy

On the final collapsed HalfCheetah checkpoint, holding everything else fixed:

| Sampling eta | Final-2 conditional action SD |
|---:|---:|
| 0 | 0.000535 |
| 1 | 0.000535 |
| 32 | about 0.0005 |
| 128 | 0.000655 |

Changing eta over this entire range does not restore diversity. In contrast, changing only the readout to Identity restores SD to about 0.035.

Thus an eta-only dispersion controller cannot recover after collapse. The direct actuator must be predictor interpolation, explicit noise, or entropy temperature `T`.

### Full DDPM and final-5 wording

Sweeps 210/211/212 use the same per-cell seeds. Sweep 211 and 212 have identical source, so the poor early behavior of final-5 relative to final-2 is a controlled result. However:

- sweep 211 was terminated at approximately 16–32% progress;
- most final-5 cells never learned a strong policy;
- some rose modestly and declined, but it is not accurate to say final-5 was proven to “collapse all six at 1M.”

The defensible statement is: **final-5 and unrestricted DDPM mean are much worse early than final-2 and often under-learn or decline at the observed partial horizons.** This is consistent with excessive mean-only transport contraction.

---

## 3. The frozen-policy result is not sufficient for the stated goals

The critique is correct: Ant 3734 and Humanoid 4371 are too low.

I searched all 312 common frozen-policy settings in the saved 100-episode grid. Results:

- configurations reaching Ant > 6000 and Humanoid > 5000: **0**;
- configurations reaching all six stated/LSAC goals: **0**;
- maximum Ant: **3949**, obtained with Identity-trained, 80 steps, training eta 32, final-16, sampling eta 32;
- maximum Humanoid: 5479, but the corresponding Ant score is 2853.

The best Ant/Humanoid maximin point was only about 64% of both goals.

So deployment-only sharpening should still be used when reporting/evaluating a trained policy because it is usually a free gain, but it cannot be presented as the solution to cross-environment training.

A paired 100-episode test at 80 steps/training eta 32 confirms the one-time effect:

| Environment | Final-2 minus Identity | 95% paired bootstrap CI |
|---|---:|---:|
| Ant | +658 | [325, 995] |
| HalfCheetah | +588 | [390, 859] |
| Hopper | +1151 | [930, 1359] |
| Humanoid | -97 | [-352, 154] |
| Swimmer | +1.3 | [0.3, 2.2] |
| Walker | +72 | [27, 149] |

This supports only the narrower claim: **one-time final-2 evaluation is beneficial in five of six frozen cases.** It does not get Ant close to 6000 and says nothing by itself about recursive training stability.

---

## 4. Concise mechanism: Ant versus the others

### Why Ant fails when the others succeed under Identity

1. **Ant genuinely has the largest action penalty coefficient.** The exact `gymnasium` v3 source used by training has:

   | Environment | `ctrl_cost_weight` |
   |---|---:|
   | Ant | **0.5** |
   | Humanoid | 0.1 |
   | HalfCheetah | 0.1 |
   | Hopper | 0.001 |
   | Walker | 0.001 |
   | Swimmer | 0.0001 |

   `notes/env_transfer_investigation.md` incorrectly says every environment uses 0.1. It appears to have inspected a different `gym` source rather than the `gymnasium` package imported by `relax/env/__init__.py`.

2. On matched Identity-trained replay batches, mean per-step control cost was approximately:

   | Ant | HalfCheetah | Humanoid | Hopper | Walker | Swimmer |
   |---:|---:|---:|---:|---:|---:|
   | **0.84** | 0.25 | 0.31 | 0.0010 | 0.0019 | 0.0001 |

3. Removing only the fixed level-0 Gaussian component would save only a few return points directly. The large effect must come from a **policy-level change in torque magnitude, coordination, gait stability, and survival**, not the instantaneous quadratic price of `sigma_0` alone.

4. Ant’s Q landscape is unusually heterogeneous: on replay actions it had the largest raw across-state Hessian-trace SD, although HalfCheetah had larger mean curvature and Hessian norm. Rare sensitive states and termination boundaries can amplify small action errors.

5. Final-2 training is consistent with learning a lower-torque Ant policy: replay control cost fell from about 0.84 to 0.56 per step and mean absolute action fell from 0.376 to 0.298. These comparisons use different training seeds and therefore support, but do not prove, the mechanism.

In short, Identity leaves Ant too diffuse/imprecise to discover and retain its best low-cost stable gait.

### Why the others fail when Ant succeeds under final-2

1. Final-2 recursively sharpens all policies, not just Ant.
2. Ant settles at a still-usable conditional SD around 0.11 and both 80-step/eta-64 seeds exceed 6000.
3. HalfCheetah and Swimmer can pass through a nonlinear contraction threshold and become nearly deterministic at each state. They then cannot explore alternatives or correct a wrong mode.
4. HalfCheetah does not merely become low-action: its final-2 replay control cost rises from about 0.25 to 0.40 per step and mean absolute action rises to 0.79. It has locked onto a bad, high-magnitude mode. This cross-run comparison uses different seeds, so it is descriptive rather than causal.
5. Hopper retains substantial conditional dispersion but shows an abrupt Q-value/critic-disagreement transition, so its failure is not explained by entropy collapse alone.
6. Walker and HalfCheetah are seed-bimodal, indicating an unstable feedback system rather than one deterministic optimum.

Thus “Ant wants less exploration, the others want more” is directionally useful but incomplete. The sharper formulation is:

> Ant benefits from recursive precision/low-torque bias; other environments need enough conditional support to avoid irreversible mode lock, and some also need critic-update trust controls.

---

## 5. Critical assessment of `env_transfer_investigation.md`

### Findings I agree with

- Identity and final-1 have identical MALA chains and different endpoint actions.
- One-time deployment sharpening is usually beneficial.
- Training with final-2 can recursively reduce conditional action dispersion.
- Pooled Q normalization measures substantial across-state variation irrelevant to the per-state tilt; within-state Q scale is an important missing statistic.
- Only one or two training seeds per cell are insufficient.
- Explicit control of conditional stochasticity is likely necessary.

### Factual errors or overclaims

1. **Control costs are wrong.** Ant is 0.5, Hopper/Walker are 0.001, and Swimmer is 0.0001 in the actual `gymnasium` v3 source. They are not all 0.1.
2. **The final-1 training table is not seed-matched.** It cannot establish the predictor as the cause of the large differences.
3. **“K=5 collapses all six” is stronger than the records support.** Sweep 211 was stopped early; the runs mostly under-learned at the observed horizon.
4. **The critic hypotheses are not fully refuted.** Average `Q/V` calibration cannot rule out local action-gradient error, correlated error shared by both critics, or a critic acting as a downstream amplifier. A shrinking `mean-min` gap with K only refutes the narrow hypothesis that larger K seeks greater two-head disagreement.
5. **The Q-blindness ratio overstates its conclusion.** Frozen-snapshot Q is `Q` for the old policy, whereas the measured return uses a different predictor at every future step. A pointwise old-policy Q difference is not expected to equal the total return of a full policy switch. This demonstrates a policy-evaluation mismatch, not that no Q-derived policy statistic can ever guide precision.
6. **The eta controller lacks an actuator test.** New frozen-policy tests show:

   | Snapshot | eta 8 SD | eta 32 SD | eta 128 SD |
   |---|---:|---:|---:|
   | Ant Identity-trained | 0.245 | 0.238 | 0.238 |
   | HalfCheetah Identity-trained | 0.210 | 0.202 | 0.189 |

   Eta is a weak direct dispersion actuator, especially on Ant, and cannot reopen a collapsed final-2 policy at all. This does not rule out eta helping prevent collapse cumulatively over many updates, but that feedback would be indirect and should not be assumed stable or monotonic.
7. **The `q_ratio` correlation is suggestive, not decisive.** It has only six environment-level points and uses deployment-optimal eta as a proxy for training-optimal eta. The proposed low-eta `ema_within` sweep is worth doing, but the reported `rho=0.943` should not be treated as validation of a universal controller.
8. **Diagnostic code and note are out of sync.** The current `diagnose_critic_overestimation.py` still hardcodes rollout length 1000, despite the note saying it was replaced by an episode-spacing estimate. Also, `diagnose_env_action_sensitivity.py --slot` selects the requested state slot but still loads `replay_batches.pkl[0]` rather than `replay_batches.pkl[slot]`. Slot-0 results are valid; nonzero-slot results currently mix a slot-specific model with slot-0 observations.
9. **“K is not an exploration-noise knob” is too categorical.** Its immediate effect on a broad policy is small, but recursive training magnifies that small contraction. K is not a clean or linear entropy knob, but it does affect training-time exploration/support.

### Assessment of its recommendations

- **P1, explicit dispersion control:** correct direction, but `eta` is the wrong primary actuator and a single equality target is unlikely to be universal.
- **P2, deployment-only sharpening:** valid free improvement, but exhaustive search proves it cannot reach the Ant/Humanoid goals from the current Identity snapshots.
- **P3, low-eta `ema_within`:** a good cheap test of the Q-scale issue, but orthogonal to the final-readout entropy collapse and not yet a demonstrated fix.

---

## 6. Revised recommendation

### Priority 0: fix diagnostics before interpreting more packed slots

- In `diagnose_env_action_sensitivity.py`, load `pickle.load(f)[args.slot]`.
- Make `training_eta` in output reflect the recovered per-slot eta.
- Make the corrected episode-length logic in the note reproducible in `diagnose_critic_overestimation.py`.
- Log conditional action SD, within-state Q SD, and MALA accepted squared jump distance during training.

### Priority 1: use a direct minimum-dispersion actuator

Keep final-2, but prevent collapse with one of:

1. `a = x0_hat + rho * (x_level0 - x0_hat)`, adapting `rho` upward only below a dispersion floor;
2. additive action/teacher noise chosen so conditional SD stays above a floor;
3. adaptive positive `T`, which flattens the retained-policy factor.

Test floors such as 0.03, 0.06, and 0.10. A floor, rather than a common equality target, lets Ant remain near its successful 0.11 while preventing HalfCheetah/Swimmer from approaching zero.

Use the direct measured signal—not pooled action variance—because a deterministic state-feedback policy can still have large variance across states.

### Priority 2: use eta for policy-update size, not entropy

Adapt eta to a per-dimension KL or normalized target-displacement budget. Combine this with low-eta `ema_within` tests such as `{0.25, 1, 4}`, with an upper bound on effective guidance. This addresses the cross-environment Q-scale problem without asking eta to recover missing stochastic support.

### Priority 3: decouple the three predictor roles

The live tree now has rollout-only predictor flags. The most informative next arm is:

```text
training TD/distillation predictor: Identity
rollout/evaluation predictor: DDPM_mean, final-k 8 or 16
fused_denoising: off
```

This tests whether Ant can benefit from precise executed actions and better data while the policy teacher and TD target retain smoothing. It will not necessarily reach 6000—frozen evaluation did not—but it cleanly separates behavior precision from recursive distillation contraction.

A second arm should use final-2 distillation plus the direct dispersion floor. That is the more likely route to preserving Ant’s >6000 behavior while preventing the other collapses.

### Priority 4: same-seed causal run, not another broad sweep

Before a six-environment Cartesian sweep, run the same master seeds under:

1. all Identity;
2. all final-2;
3. final-2 plus dispersion floor;
4. Identity TD/distillation plus sharp rollout.

Use Ant, HalfCheetah, Hopper, and Swimmer first. HalfCheetah must run through at least its observed 400k–600k transition; stopping at 200k would miss the failure. Abort arms automatically when conditional SD, policy epsilon-MSE, or MALA jump distance crosses a collapse threshold.

---

## Final confidence statement

High confidence:

- implementation equivalence of the two final-1 spellings;
- one-time final-2 usually improves frozen-policy execution;
- deployment-only sharpening cannot reach the requested Ant/Humanoid goals;
- final-2 creates a delayed conditional-collapse transition in HalfCheetah rather than a simple plateau;
- eta is too weak to be the sole dispersion actuator and cannot recover a collapsed final-2 policy;
- Ant’s v3 control penalty is genuinely much larger than most other environments;
- pooled Q scale is not the relevant per-state tilt scale.

Moderate confidence:

- Ant succeeds because recursive sharpening lands near a lower-torque, more stable gait while other tasks cross into bad mode lock;
- MALA step-size contraction participates in the feedback loop;
- a dispersion floor plus KL-controlled eta can produce a common configuration.

Not established:

- that final-1 training itself caused the one-seed sweep-161 improvements;
- that entropy collapse explains Hopper or every Walker failure;
- that a single absolute target dispersion transfers across all six;
- that low-eta `ema_within` alone will solve the problem.
