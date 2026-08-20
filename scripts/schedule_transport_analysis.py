#!/usr/bin/env python3
"""How noisy is the cleanest noise level, and which predictor transports best?

Two questions, both answerable in closed form for a Gaussian policy:

1. SCHEDULE GEOMETRY. ``build_beta_schedule`` pins alpha_bar_0 via --snr_max
   *independently of --diffusion_steps*, so adding steps only subdivides the
   noisy end and never makes the cleanest level cleaner. Compare against what a
   standard Nichol-Dhariwal cosine schedule puts at its cleanest level.

2. TRANSPORT FIDELITY. For clean actions ~ N(0, s^2 I) the exact level-t
   marginal variance is v_t = abar_t s^2 + (1 - abar_t), and the exact reverse
   kernel q(x_{t-1}|x_t) is known in closed form. Every implemented predictor is
   a specific (mean, noise) pair, so each one's realized Var(x_{t-1}) can be
   compared to the exact v_{t-1}. Retention = realized / exact; 1.0 is perfect.

Key identity: the clean-end retention of any Tweedie-mean-style step is
    1 / (1 + 1/(SNR_0 * s^2))
so the *effective* clean-end SNR is ``snr_max * s^2``. Raising --snr_max and
standardizing the action scale are therefore interchangeable knobs.
"""
from __future__ import annotations

import sys

import numpy as np

from relax.utils.diffusion import BetaScheduleCoefficients, build_beta_schedule


def legacy_schedule(T, kind, snr_max):
    """The pre-fix builder: rescale every beta to pin SNR_0 to ``snr_max``.

    ``build_beta_schedule`` no longer does this (cosine/linear are now native),
    but reproducing it here is what lets us quantify what the rescale cost.
    """
    if kind == "constant_kl":
        return build_beta_schedule(T, kind, snr_max)
    raw = (BetaScheduleCoefficients.cosine_beta_schedule(T) if kind == "cosine"
           else BetaScheduleCoefficients.linear_beta_schedule(T))
    scale = (1.0 - snr_max / (1.0 + snr_max)) / raw[0]
    return BetaScheduleCoefficients.from_beta(np.clip(scale * raw, 0, 0.999))


def nd_cosine_abar(T, s=0.008):
    """Nichol & Dhariwal cosine alpha_bar for a T-step grid, cleanest level first.

    abar_k = f(k/T)/f(0), k = 1..T. k=0 is the trivial abar=1 (no noise), so the
    cleanest level the model is ever trained on is k=1 -- and its SNR is what
    ``--snr_max`` is trying to reproduce.
    """
    k = np.arange(T + 1, dtype=np.float64)
    f = np.cos(((k / T + s) / (1 + s)) * np.pi / 2) ** 2
    return (f / f[0])[1:]          # index 0 = cleanest trained level


def retentions(sc, s):
    """Per-level variance retention of each predictor, for clean std ``s``."""
    ab = np.asarray(sc.alphas_cumprod, dtype=np.float64)
    abp = np.asarray(sc.alphas_cumprod_prev, dtype=np.float64)
    al = np.asarray(sc.alphas, dtype=np.float64)
    pv = np.asarray(sc.posterior_variance, dtype=np.float64)
    c1 = np.asarray(sc.posterior_mean_coef1, dtype=np.float64)
    c2 = np.asarray(sc.posterior_mean_coef2, dtype=np.float64)
    be = np.asarray(sc.betas, dtype=np.float64)

    v = ab * s * s + (1 - ab)            # Var(x_t)
    vp = abp * s * s + (1 - abp)         # exact Var(x_{t-1})
    out = {}

    # Tweedie x0_hat = E[x0|x_t] = (sqrt(abar) s^2 / v) x_t  (linear-Gaussian)
    g0 = np.sqrt(ab) * s * s / v
    # eps_hat = E[eps|x_t] = (sqrt(1-abar)/v) x_t
    ge = np.sqrt(1 - ab) / v

    out["Identity"] = v / vp
    # DDPM_mean: x_{t-1} = c1 * x0_hat + c2 * x_t, no noise
    out["DDPM_mean"] = ((c1 * g0 + c2) ** 2 * v) / vp
    # DDPM ancestral with the standard posterior variance beta_tilde
    out["DDPM_ancestral"] = ((c1 * g0 + c2) ** 2 * v + pv) / vp
    # DDPM ancestral with the larger beta_t variance (Ho et al.'s other choice)
    out["DDPM_anc_beta_t"] = ((c1 * g0 + c2) ** 2 * v + be) / vp
    # DDIM eta=0
    k = np.sqrt(abp / ab)
    cd = k + (np.sqrt(1 - abp) - k * np.sqrt(1 - ab)) * ge
    out["DDIM"] = (cd * cd * v) / vp
    # Exact reverse kernel: mean sqrt(alpha) vp/v * x_t, var vp - alpha vp^2/v
    mean_c = np.sqrt(al) * vp / v
    out["Exact"] = (mean_c ** 2 * v + (vp - al * vp * vp / v)) / vp
    return out


def main():
    print("=" * 100)
    print("1. SCHEDULE GEOMETRY: how clean is the cleanest level?")
    print("=" * 100)
    print(f"{'T':>5} {'snr_max':>9} {'abar_0':>10} {'sigma_0':>9} {'SNR_0':>10} "
          f"{'abar_max':>10} {'SNR_min':>9}")
    for T in (20, 40, 80):
        for snr in (124.0, 1000.0, 6265.0):
            sc = legacy_schedule(T, "cosine", snr)
            ab = np.asarray(sc.alphas_cumprod)
            print(f"{T:>5} {snr:>9.0f} {ab[0]:>10.6f} {np.sqrt(1-ab[0]):>9.4f} "
                  f"{ab[0]/(1-ab[0]):>10.1f} {ab[-1]:>10.2e} {ab[-1]/(1-ab[-1]):>9.2e}")
    print("\n  NOTE: abar_0 depends ONLY on --snr_max, never on T. Adding diffusion")
    print("  steps subdivides the NOISY end and leaves the clean end untouched.")

    print("\n  NATIVE (unrescaled) Nichol-Dhariwal cosine, cleanest trained level:")
    print(f"    {'T':>6} {'abar_0':>11} {'SNR_0':>10} {'abar_noisiest':>15} {'SNR_noisiest':>13}")
    for T in (20, 40, 80, 200, 1000):
        ab = nd_cosine_abar(T)
        print(f"    {T:>6} {ab[0]:>11.7f} {ab[0]/(1-ab[0]):>10.1f} "
              f"{ab[-1]:>15.3e} {ab[-1]/(1-ab[-1]):>13.3e}")
    print("  -> snr_max=124 is EXACTLY native cosine at T=20. So the default is not a")
    print("     mistake for T=20 -- but it is pinned there for every T, which means at")
    print("     T=40/80 the code inflates all betas to force a DIRTIER clean end than")
    print("     native cosine would give (native T=40 ~ 500, T=80 ~ 2000).")
    print("  Ho et al. DDPM linear, T=1000: abar_0 = 1-1e-4 = 0.9999 -> SNR_0 = 9999")

    print("\n" + "=" * 100)
    print("1b. THE TENSION: raising snr_max at fixed T destroys the NOISY end")
    print("=" * 100)
    print("  The builder rescales every beta by (1-abar_0_target)/raw_betas[0], so a")
    print("  cleaner clean end shrinks ALL betas and the noisiest level stops being")
    print("  pure noise -- but the sampler still initializes x_T ~ N(0, I).")
    print(f"\n  {'sched':>12} {'T':>5} {'snr_max':>9} {'SNR_0':>10} {'abar_noisiest':>15}"
          f" {'SNR_noisiest':>13} {'x_T ok?':>9}")
    for kind in ("cosine", "constant_kl"):
        for T in (20, 40, 80, 200):
            for snr in (124.0, 1000.0, 6265.0, 31000.0):
                sc = legacy_schedule(T, kind, snr)
                ab = np.asarray(sc.alphas_cumprod, dtype=np.float64)
                snr_hi = ab[-1] / max(1 - ab[-1], 1e-300)
                ok = "yes" if ab[-1] < 1e-3 else ("marginal" if ab[-1] < 1e-2 else "NO")
                print(f"  {kind:>12} {T:>5} {snr:>9.0f} {ab[0]/(1-ab[0]):>10.1f}"
                      f" {ab[-1]:>15.3e} {snr_hi:>13.3e} {ok:>9}")
            print()

    print("\n" + "=" * 100)
    print("2. EFFECTIVE clean-end SNR = snr_max * s^2  (action scale and snr_max trade off)")
    print("=" * 100)
    print(f"  {'s (per-dim action std)':<26}{'s^2':>8}{'snr_max=124':>14}{'=1000':>10}{'=6265':>10}{'=31000':>10}")
    for s in (1.0, 0.7, 0.45, 0.3):
        row = "".join(f"{124*s*s:>14.1f}" if c == 0 else f"{c*s*s:>10.1f}"
                      for c in (0, 1000.0, 6265.0, 31000.0))
        print(f"  s={s:<24.2f}{s*s:>8.3f}{row}")
    print("\n  Standard image diffusion operates at effective clean-end SNR ~ 6000-10000.")
    print("  This repo at snr_max=124 with the logged act_var~0.2 sits at ~25: 250-400x noisier.")

    print("\n" + "=" * 100)
    print("3. TRANSPORT FIDELITY: composed per-pass variance retention (1.0 = exact)")
    print("=" * 100)
    for snr in (124.0, 1000.0, 6265.0, 31000.0):
        print(f"\n--- snr_max = {snr:.0f} (cosine, T=20) ---")
        sc = legacy_schedule(20, "cosine", snr)
        print(f"  {'s':<7}" + "".join(f"{k:>17}" for k in
              ["Identity", "DDPM_mean", "DDPM_ancestral", "DDIM", "Exact"]))
        for s in (1.0, 0.7, 0.45, 0.3):
            r = retentions(sc, s)
            print(f"  {s:<7.2f}" + "".join(f"{r[k].prod():>17.4f}" for k in
                  ["Identity", "DDPM_mean", "DDPM_ancestral", "DDIM", "Exact"]))

    print("\n" + "=" * 100)
    print("4. Per-level retention at the clean end (T=20, s=0.45, snr_max=124)")
    print("=" * 100)
    sc = legacy_schedule(20, "cosine", 124.0)
    r = retentions(sc, 0.45)
    print(f"  {'t':>4}" + "".join(f"{k:>17}" for k in
          ["Identity", "DDPM_mean", "DDPM_ancestral", "DDIM", "Exact"]))
    for t in (19, 10, 5, 3, 2, 1, 0):
        print(f"  {t:>4}" + "".join(f"{r[k][t]:>17.4f}" for k in
              ["Identity", "DDPM_mean", "DDPM_ancestral", "DDIM", "Exact"]))

    print("\n" + "=" * 100)
    print("4b. WHAT IF snr_max TRACKS T, the way native cosine already does?")
    print("=" * 100)
    print("  Native cosine SNR_0 grows ~T^2. Pinning snr_max=124 throws that away, so")
    print("  --diffusion_steps 40/80 bought nothing at the clean end. Letting snr_max")
    print("  follow T keeps the noisy end pure AND cleans up the clean end:")
    native = {T: float(nd_cosine_abar(T)[0] / (1 - nd_cosine_abar(T)[0]))
              for T in (20, 40, 80, 200)}
    for s in (0.45, 1.0):
        print(f"\n  --- s = {s} ---")
        print(f"  {'T':>5} {'snr_max':>9} {'abar_noisiest':>15}" +
              "".join(f"{k:>16}" for k in ["Identity", "DDPM_mean", "DDPM_ancestral", "DDIM"]))
        for T, snr in native.items():
            sc = build_beta_schedule(T, "cosine", snr)
            ab = np.asarray(sc.alphas_cumprod, dtype=np.float64)
            r = retentions(sc, s)
            print(f"  {T:>5} {snr:>9.0f} {ab[-1]:>15.2e}" +
                  "".join(f"{r[k].prod():>16.4f}" for k in
                          ["Identity", "DDPM_mean", "DDPM_ancestral", "DDIM"]))

    print("\n" + "=" * 100)
    print("5. Identity's irreducible noise floor (it never removes level-0 noise)")
    print("=" * 100)
    print(f"  {'snr_max':>9} {'sigma_0':>9} {'floor std':>11} {'floor var':>11} "
          f"{'% of act_var=0.2':>18}")
    for snr in (124.0, 1000.0, 6265.0, 31000.0):
        sc = legacy_schedule(20, "cosine", snr)
        ab0 = float(np.asarray(sc.alphas_cumprod)[0])
        fl = np.sqrt(1 - ab0) / np.sqrt(ab0)
        print(f"  {snr:>9.0f} {np.sqrt(1-ab0):>9.4f} {fl:>11.4f} {fl*fl:>11.5f} "
              f"{100*fl*fl/0.2:>17.1f}%")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else None
    if out:
        with open(out, "w") as fh:
            sys.stdout = fh
            main()
    else:
        main()
