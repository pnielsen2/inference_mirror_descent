#!/usr/bin/env python3
"""Figures for notes/snr_schedule_shift/snr_schedule_shift.tex.

Run from the repository root:  python notes/snr_schedule_shift/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.equal_burden_schedule import sched_from_abar  # noqa: E402
from scripts.schedule_transport_analysis import retentions  # noqa: E402
from relax.utils.diffusion import build_beta_schedule  # noqa: E402

OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "figure.dpi": 140,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.6,
})

C = {"cos": "#1f77b4", "unif": "#d62728", "lin": "#7f7f7f",
     "shift": "#2ca02c", "accent": "#9467bd", "orange": "#ff7f0e"}

SNR_MAX = 124.0


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────
def cosine_abar(T):
    return np.asarray(build_beta_schedule(T, "cosine", SNR_MAX).alphas_cumprod,
                      dtype=np.float64)


def lam_of(ab):
    return np.log(ab / (1.0 - ab))


def abar_of(lam):
    return 1.0 / (1.0 + np.exp(-lam))


def shifted_abar(T, s_hat):
    """THE RECIPE: translate the cosine log-SNR trajectory down by 2 log s_hat."""
    return abar_of(lam_of(cosine_abar(T)) - 2.0 * np.log(s_hat))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# standard-normal quadrature nodes for the two-point-prior MMSE
_N = np.linspace(-11.0, 11.0, 3001)
_W = np.exp(-0.5 * _N ** 2)
_W = _W / _W.sum()


def twopoint_rate(lam_eff):
    """dI/dlam_eff for the prior x = +-s, as a function of lam_eff = log(gamma s^2)."""
    lam_eff = np.atleast_1d(np.asarray(lam_eff, dtype=np.float64))
    kap = np.exp(0.5 * lam_eff)
    out = np.empty_like(kap)
    for j in range(0, kap.size, 512):
        k = kap[j:j + 512, None]
        z = k * k + k * _N[None, :]
        out[j:j + 512] = 0.5 * k[:, 0] ** 2 * (1.0 - (np.tanh(z) ** 2 * _W).sum(1))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: the exact information clock (I-MMSE).
# ─────────────────────────────────────────────────────────────────────────────
def fig_information():
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.85))
    le = np.linspace(-10, 12, 3000)

    ax[0].plot(le, 0.5 * sigmoid(le), color=C["cos"],
               label=r"Gaussian: $\frac{1}{2}\sigma(\lambda^{\rm eff})$")
    ax[0].plot(le, twopoint_rate(le), color=C["orange"], ls="-",
               label=r"two-point prior $x=\pm s$")
    ax[0].plot(le, 0.5 * np.exp(le), color="k", ls=":", lw=1.0,
               label=r"universal low-SNR: $\frac{1}{2}e^{\lambda^{\rm eff}}$")
    ax[0].axhline(0.5, color=C["cos"], ls="--", lw=0.8)
    ax[0].annotate(r"$\frac{1}{2}$ nat per unit $\lambda^{\rm eff}$", xy=(3.4, 0.36),
                   fontsize=7.5, color=C["cos"])
    ax[0].set_xlabel(r"$\lambda^{\rm eff}=\lambda+2\log s$")
    ax[0].set_ylabel(r"$dI/d\lambda^{\rm eff}$  (nats)")
    ax[0].set_title(r"(a) exact information rate")
    ax[0].set_ylim(0, 0.58)
    ax[0].legend(loc="upper left", frameon=False)

    le2 = np.linspace(-9, 9, 2000)
    ax[1].semilogy(le2, sigmoid(-le2), color=C["unif"],
                   label=r"${\rm mmse}/s^2$: how much is left")
    ax[1].semilogy(le2, np.exp(le2), color=C["shift"],
                   label=r"$\gamma s^2$: channel quality")
    ax[1].semilogy(le2, 0.5 * sigmoid(le2), color=C["cos"], lw=2.2,
                   label=r"product$/2 = dI/d\lambda^{\rm eff}$")
    ax[1].axhline(0.5, color="k", ls=":", lw=0.8)
    ax[1].set_xlabel(r"$\lambda^{\rm eff}$")
    ax[1].set_ylabel("value")
    ax[1].set_title(r"(b) the two competing factors")
    ax[1].set_ylim(1e-4, 1e4)
    ax[1].legend(loc="upper left", frameon=False)

    le3 = np.linspace(-30, 14, 40000)
    Ig = 0.5 * np.log1p(np.exp(le3))
    rt = twopoint_rate(le3)
    Itp = np.concatenate([[0.0], np.cumsum(0.5 * (rt[1:] + rt[:-1]) * np.diff(le3))])
    ax[2].plot(le3, Ig, color=C["cos"], label="Gaussian prior")
    ax[2].plot(le3, Itp, color=C["orange"], label=r"two-point prior")
    ax[2].axhline(np.log(2), color=C["orange"], ls="--", lw=0.8)
    ax[2].annotate(r"$\log 2$ (all of it)", xy=(-9, 0.78), fontsize=7.5, color=C["orange"])
    ax[2].annotate(r"slope $\frac{1}{2}$:" "\n" "diverges", xy=(6.0, 2.0), fontsize=7.5,
                   color=C["cos"])
    ax[2].set_xlabel(r"$\lambda^{\rm eff}$")
    ax[2].set_ylabel(r"$I(x_0;x)$  (nats)")
    ax[2].set_title(r"(c) information delivered so far")
    ax[2].set_xlim(-10, 12)
    ax[2].set_ylim(0, 6.4)
    ax[2].legend(loc="upper left", frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "fig_information.pdf")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: log-SNR is unbounded; the angle is compact.
# ─────────────────────────────────────────────────────────────────────────────
def fig_coordinates():
    fig, ax = plt.subplots(1, 3, figsize=(9.4, 2.75))

    ab = np.linspace(1e-4, 1 - 1e-4, 4000)
    ax[0].plot(ab, lam_of(ab), color=C["cos"])
    for y in (-10, 10):
        ax[0].axhline(y, color="k", ls=":", lw=0.7)
    ax[0].set_xlabel(r"$\bar\alpha$")
    ax[0].set_ylabel(r"$\lambda=\log\mathrm{SNR}$")
    ax[0].set_title(r"(a) $\lambda$ is unbounded")
    ax[0].set_ylim(-12, 12)
    ax[0].annotate(r"$\lambda\to+\infty$", xy=(0.985, 9.5), ha="right", fontsize=7.5)
    ax[0].annotate(r"$\lambda\to-\infty$", xy=(0.015, -10.5), ha="left", fontsize=7.5)

    ax[1].plot(ab, np.arccos(np.sqrt(ab)), color=C["shift"])
    ax[1].axhline(np.pi / 2, color="k", ls=":", lw=0.7)
    ax[1].set_xlabel(r"$\bar\alpha$")
    ax[1].set_ylabel(r"$\theta=\arctan\tilde\sigma$")
    ax[1].set_title(r"(b) $\theta$ is compact: $[0,\pi/2]$")
    ax[1].set_yticks([0, np.pi / 4, np.pi / 2])
    ax[1].set_yticklabels(["0", r"$\pi/4$", r"$\pi/2$"])

    th = np.linspace(0, np.pi / 2, 400)
    ax[2].plot(np.cos(th), np.sin(th), color="k", lw=1.2)
    thk = np.linspace(0, np.pi / 2, 11)
    ax[2].plot(np.cos(thk), np.sin(thk), "o", color=C["cos"], ms=4.5,
               label=r"uniform $\theta$ (cosine)")
    ax[2].plot([0, np.cos(np.pi / 5)], [0, np.sin(np.pi / 5)], color=C["accent"], lw=1.0)
    ax[2].annotate(r"$\theta$", xy=(0.36, 0.13), color=C["accent"], fontsize=10)
    ax[2].set_xlabel(r"signal amplitude $\sqrt{\bar\alpha}$")
    ax[2].set_ylabel(r"noise amplitude $\sqrt{1-\bar\alpha}$")
    ax[2].set_title(r"(c) $s=1$: the locus is the unit circle")
    ax[2].set_aspect("equal")
    ax[2].set_xlim(-0.05, 1.12)
    ax[2].set_ylim(-0.05, 1.12)
    ax[2].legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "fig_coordinates.pdf")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: cosine is NOT truncated uniform-log-SNR.
# ─────────────────────────────────────────────────────────────────────────────
def fig_trajectories():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))

    T = 100
    th = (np.arange(T) + 1) / T * (np.pi / 2)
    lam_cos = lam_of(np.cos(th) ** 2)
    lam_lin = lam_of(np.asarray(
        build_beta_schedule(T, "linear", 1e9).alphas_cumprod, np.float64))
    lam_unif = np.linspace(12.0, -12.0, T)

    k = np.arange(1, T + 1)
    ax[0].plot(k, lam_cos[::-1], color=C["cos"], label=r"cosine ($\theta$ uniform)")
    ax[0].plot(k, lam_unif[::-1], color=C["unif"], ls="--",
               label=r"uniform $\lambda$, truncated $|\lambda|\leq 12$")
    ax[0].plot(k, lam_lin[::-1], color=C["lin"], ls="-.", label=r"linear $\beta$")
    ax[0].set_xlabel(r"step index $k$ (noisiest $\to$ cleanest)")
    ax[0].set_ylabel(r"$\lambda_k$")
    ax[0].set_title(r"(a) log-SNR trajectories, $T=100$")
    ax[0].legend(loc="upper left", frameon=False)
    ax[0].set_ylim(-14, 14)

    th2 = np.linspace(0.02, np.pi / 2 - 0.02, 2000)
    ax[1].plot(th2, 4.0 / np.sin(2 * th2), color=C["cos"])
    ax[1].axhline(4.0, color="k", ls=":", lw=0.8)
    ax[1].axvline(np.pi / 4, color=C["accent"], ls="--", lw=0.9)
    ax[1].annotate(r"min $=4$ at $\theta=\pi/4$" "\n" r"($\mathrm{SNR}=1$)",
                   xy=(np.pi / 4 + 0.06, 6.2), fontsize=7.5, color=C["accent"])
    ax[1].set_xlabel(r"$\theta$")
    ax[1].set_ylabel(r"$|d\lambda/d\theta| = 4/\sin 2\theta$")
    ax[1].set_title(r"(b) cosine's log-SNR speed")
    ax[1].set_ylim(0, 22)
    ax[1].set_xticks([0, np.pi / 4, np.pi / 2])
    ax[1].set_xticklabels(["0", r"$\pi/4$", r"$\pi/2$"])

    fig.tight_layout()
    fig.savefig(OUT / "fig_trajectories.pdf")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: the shift.
# ─────────────────────────────────────────────────────────────────────────────
def fig_shift():
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.85))

    T = 60
    th = (np.arange(T) + 1) / T * (np.pi / 2)
    lam_cos = lam_of(np.cos(th) ** 2)
    k = np.arange(1, T + 1)

    for s, col in zip([1.0, 0.6, 0.3, 0.15],
                      [C["cos"], C["orange"], C["shift"], C["unif"]]):
        ax[0].plot(k, (lam_cos + 2 * np.log(s))[::-1], color=col, label=rf"$s_i={s}$")
    ax[0].axhspan(-4, 4, color="k", alpha=0.07, lw=0)
    ax[0].axhline(0, color="k", ls=":", lw=0.8)
    ax[0].set_xlabel("step index $k$")
    ax[0].set_ylabel(r"$\lambda^{\rm eff}_i = \lambda_k + 2\log s_i$")
    ax[0].set_title(r"(a) one schedule, $d$ effective clocks")
    ax[0].legend(loc="lower right", frameon=False)
    ax[0].set_ylim(-16, 14)

    s = 0.3
    ax[1].plot(k, lam_cos[::-1], color=C["cos"], label=r"cosine $\lambda^{\rm target}_k$")
    ax[1].plot(k, (lam_cos + 2 * np.log(s))[::-1], color=C["unif"], ls="--",
               label=rf"its $\lambda^{{\rm eff}}$ at $s={s}$ (too noisy)")
    ax[1].plot(k, (lam_cos - 2 * np.log(s))[::-1], color=C["shift"],
               label=r"the recipe: $\lambda_k=\lambda^{\rm target}_k-2\log s$")
    ax[1].annotate("", xy=(42, lam_cos[::-1][41] - 2 * np.log(s)),
                   xytext=(42, lam_cos[::-1][41]),
                   arrowprops=dict(arrowstyle="->", color=C["shift"], lw=1.2))
    ax[1].annotate(r"$2\log\frac{1}{s}$", xy=(43.5, lam_cos[::-1][41] - np.log(s)),
                   fontsize=8.5, color=C["shift"])
    ax[1].axhline(0, color="k", ls=":", lw=0.8)
    ax[1].set_xlabel("step index $k$")
    ax[1].set_ylabel(r"$\lambda$")
    ax[1].set_title(r"(b) the correction is a rigid translation")
    ax[1].legend(loc="upper left", frameon=False)
    ax[1].set_ylim(-16, 16)

    psi_cos = np.arctan(np.tan(np.clip(th, 0, np.pi / 2 - 1e-9)) / s)
    ab_sh = abar_of(lam_cos - 2 * np.log(s))
    psi_sh = np.arctan(np.sqrt((1 - ab_sh) / ab_sh) / s)
    ax[2].plot(k, psi_cos[::-1], "o-", color=C["cos"], ms=3.0,
               label=r"cosine, unshifted")
    ax[2].plot(k, psi_sh[::-1], "s-", color=C["shift"], ms=3.0, mfc="none",
               label=r"shifted (exactly uniform)")
    ax[2].set_xlabel("step index $k$")
    ax[2].set_ylabel(r"$\psi_k=\arctan(\tilde\sigma_k/s)$")
    ax[2].set_title(rf"(c) cost-metric arc length, $s={s}$")
    ax[2].set_yticks([0, np.pi / 4, np.pi / 2])
    ax[2].set_yticklabels(["0", r"$\pi/4$", r"$\pi/2$"])
    ax[2].legend(loc="lower left", frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "fig_shift.pdf")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: picture of the derived optimum in the signal-noise plane.
# ─────────────────────────────────────────────────────────────────────────────
def fig_geometry():
    s = 0.5
    n = 9
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.1))
    tt = np.linspace(0, np.pi / 2, 400)

    th_u = np.linspace(0, np.pi / 2, n)
    pts_cos = (s * np.cos(th_u), np.sin(th_u))
    psi_u = np.linspace(0, np.pi / 2, n)
    ab_p = 1.0 / (1.0 + s * s * np.tan(np.clip(psi_u, 0, np.pi / 2 - 1e-9)) ** 2)
    pts_sh = (s * np.sqrt(ab_p), np.sqrt(1 - ab_p))

    R = 0.80
    for a, (px, py), lab, col in [
        (ax[0], pts_cos, r"cosine, unshifted (uniform $\theta$)", C["cos"]),
        (ax[1], pts_sh, r"shifted (uniform $\psi$)", C["shift"]),
    ]:
        a.plot(s * np.cos(tt), np.sin(tt), color="k", lw=1.3)
        a.plot(R * np.cos(tt), R * np.sin(tt), color="0.6", lw=0.7)
        ang = np.arctan2(py, px)
        for aa in ang:
            a.plot([0, R * np.cos(aa)], [0, R * np.sin(aa)], color=col, lw=0.8, alpha=0.6)
        a.plot(R * np.cos(ang), R * np.sin(ang), "o", color=col, ms=5.0)
        a.plot(px, py, "o", color="k", ms=3.0)
        a.set_aspect("equal")
        a.set_xlim(-0.04, 0.92)
        a.set_ylim(-0.04, 1.08)
        a.set_xlabel(r"signal amplitude $s\sqrt{\bar\alpha}$")
        a.set_title(lab)
        a.grid(False)
    ax[0].set_ylabel(r"noise amplitude $\sqrt{1-\bar\alpha}$")
    ax[0].annotate("angles crowd\nat the noisy end", xy=(0.40, 0.80), fontsize=7.5,
                   color=C["cos"])
    ax[1].annotate("angles evenly\nspaced", xy=(0.44, 0.74), fontsize=7.5,
                   color=C["shift"])
    fig.suptitle(rf"ellipse of semi-axes $(s,1)$, $s={s}$ (black); dots on the grey arc"
                 r" mark each level's polar angle $\psi$", fontsize=8.5, y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "fig_geometry.pdf")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: where the steps land.
# ─────────────────────────────────────────────────────────────────────────────
def fig_allocation():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.85))
    T, s = 20, 0.3
    ab_c = cosine_abar(T)
    ab_p = shifted_abar(T, s)

    for sig, lab, col in [(np.sqrt((1 - ab_c) / ab_c), "cosine, unshifted", C["cos"]),
                          (np.sqrt((1 - ab_p) / ab_p), rf"shifted ($\hat s={s}$)", C["shift"])]:
        ax[0].semilogy(np.arange(1, T + 1), sig[::-1], "o-", color=col, ms=3.5, label=lab)
    ax[0].axhline(s, color=C["unif"], ls="--", lw=1.0,
                  label=r"$\tilde\sigma = s$ (signal $=$ noise)")
    ax[0].set_xlabel("step index $k$")
    ax[0].set_ylabel(r"$\tilde\sigma_k$")
    ax[0].set_title(r"(a) noise level per step, $T=20$")
    ax[0].legend(loc="upper right", frameon=False)

    svals = (0.15, 0.3, 0.6, 1.0)
    frac_c, frac_p = [], []
    for ss in svals:
        sig_c = np.sqrt((1 - ab_c) / ab_c)
        abp = shifted_abar(T, ss)
        sig_p = np.sqrt((1 - abp) / abp)
        frac_c.append(float((sig_c < ss).mean()))
        frac_p.append(float((sig_p < ss).mean()))
    x = np.arange(4)
    ax[1].bar(x - 0.19, frac_c, 0.36, color=C["cos"], label="cosine, unshifted")
    ax[1].bar(x + 0.19, frac_p, 0.36, color=C["shift"], label="shifted")
    ax[1].axhline(0.5, color="k", ls=":", lw=0.9)
    ax[1].annotate("optimum $=1/2$", xy=(2.35, 0.53), fontsize=7.5)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([str(v) for v in svals])
    ax[1].set_xlabel(r"clean std $s$")
    ax[1].set_ylabel(r"fraction of steps with $\tilde\sigma_k<s$")
    ax[1].set_title(r"(b) budget spent where signal competes")
    ax[1].legend(loc="upper left", frameon=False)
    ax[1].set_ylim(0, 0.72)

    fig.tight_layout()
    fig.savefig(OUT / "fig_allocation.pdf")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: the cosh penalty.
# ─────────────────────────────────────────────────────────────────────────────
def fig_penalty():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.85))

    rr = np.logspace(-1.2, 1.2, 500)
    ax[0].semilogx(rr, np.cosh(np.log(rr)), color=C["shift"])
    T = 80
    meas_rho, meas_val = [], []
    s_true = 0.3
    for rho in (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        d_hat = 1 - retentions(sched_from_abar(shifted_abar(T, s_true / rho)),
                               s_true)["DDIM"].prod()
        d_opt = 1 - retentions(sched_from_abar(shifted_abar(T, s_true)),
                               s_true)["DDIM"].prod()
        meas_rho.append(rho)
        meas_val.append(d_hat / d_opt)
    ax[0].plot(meas_rho, meas_val, "o", color=C["unif"], ms=5, label=rf"measured, $T={T}$")
    ax[0].axhline(1.0, color="k", ls=":", lw=0.8)
    ax[0].axvspan(0.5, 2.0, color=C["shift"], alpha=0.10, lw=0)
    ax[0].annotate(r"within $2\times$: cost $\leq1.25$", xy=(0.52, 1.62), fontsize=7.5)
    ax[0].set_xlabel(r"$\rho=s/\hat s$ (mis-estimation factor)")
    ax[0].set_ylabel(r"$D(\hat s)/D_{\min}=\cosh(\log\rho)$")
    ax[0].set_title(r"(a) the penalty is flat near the optimum")
    ax[0].legend(loc="upper center", frameon=False)
    ax[0].set_ylim(0.7, 5.6)

    ss = np.logspace(-1.3, 0.6, 500)
    ax[1].loglog(ss, np.cosh(np.log(ss)), color=C["cos"])
    ax[1].axhline(1.0, color="k", ls=":", lw=0.8)
    ax[1].axvline(1.0, color="k", ls=":", lw=0.8)
    for sm in (0.45, 0.2, 0.1):
        ax[1].plot([sm], [np.cosh(np.log(sm))], "o", color=C["unif"], ms=5)
        ax[1].annotate(rf"$s={sm}$: ${np.cosh(np.log(sm)):.2f}\times$",
                       xy=(sm * 1.1, np.cosh(np.log(sm)) * 1.06), fontsize=7.5)
    ax[1].set_xlabel(r"clean std $s$")
    ax[1].set_ylabel(r"$D_{\cos}/D_{\min}=\cosh(\log s)$")
    ax[1].set_title(r"(b) unshifted cosine is the case $\hat s=1$")

    fig.tight_layout()
    fig.savefig(OUT / "fig_penalty.pdf")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: heterogeneous / state-dependent scale.
# ─────────────────────────────────────────────────────────────────────────────
def fig_heterogeneity():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.85))

    ratio = np.logspace(-1.1, 1.1, 500)
    for tau, col in zip([0.0, 0.5, 1.0, 1.5],
                        [C["cos"], C["shift"], C["orange"], C["unif"]]):
        floor = np.exp(tau ** 2 / 2)
        ax[0].loglog(ratio, floor * np.cosh(np.log(ratio)), color=col,
                     label=rf"$\tau={tau}$")
        ax[0].axhline(floor, color=col, ls=":", lw=0.8)
    ax[0].axvline(1.0, color="k", ls=":", lw=0.8)
    ax[0].set_ylim(0.78, 40)
    ax[0].annotate(r"dotted: the floors $e^{\tau^2/2}$, irreducible", xy=(0.09, 0.83),
                   fontsize=7.5)
    ax[0].set_xlabel(r"$\hat s/e^{m}$")
    ax[0].set_ylabel(r"$\mathcal{L}(\hat s)/D_{\min}$")
    ax[0].set_title(r"(a) heterogeneous $s$: cost of a global $\hat s$")
    ax[0].legend(loc="upper center", frameon=False, ncol=2)

    taus = np.linspace(0, 1.8, 300)
    ax[1].plot(taus, np.exp(taus ** 2 / 2), color=C["shift"],
               label=r"best global $\hat s=e^{m}$: $e^{\tau^2/2}$")
    ax[1].plot(taus, np.exp(taus ** 2 / 2) * np.cosh(taus ** 2 / 2), color=C["unif"],
               ls="--", label=r"using $\hat s=\mathbb{E}[s]$ instead")
    rng = np.random.default_rng(0)
    tt = np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    emp = []
    for t in tt:
        smp = 0.4 * np.exp(rng.normal(0, t, 400000))
        emp.append(np.sqrt(smp.mean() * (1 / smp).mean()))
    ax[1].plot(tt, emp, "o", color="k", ms=4,
               label=r"$\sqrt{M_+M_-}$, sampled")
    ax[1].axhline(1.0, color="k", ls=":", lw=0.8)
    ax[1].axhspan(1.0, 1.15, color=C["shift"], alpha=0.10, lw=0)
    ax[1].annotate("global shift fine", xy=(0.06, 1.045), fontsize=7.5)
    ax[1].annotate("normalize\nper state", xy=(1.42, 2.35), fontsize=7.5, color=C["unif"])
    ax[1].set_xlabel(r"$\tau=\mathrm{sd}(\log s)$")
    ax[1].set_ylabel(r"irreducible factor over $D_{\min}$")
    ax[1].set_title(r"(b) when one scalar stops being enough")
    ax[1].legend(loc="upper left", frameon=False)
    ax[1].set_ylim(0.9, 4.2)

    fig.tight_layout()
    fig.savefig(OUT / "fig_heterogeneity.pdf")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9: numerical verification of the closed forms.
# ─────────────────────────────────────────────────────────────────────────────
def fig_verification():
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.85))
    Ts = np.array([20, 30, 50, 80, 130, 200, 320, 500])
    slist = [0.2, 0.45, 0.7, 1.0]
    cols = [C["unif"], C["orange"], C["shift"], C["cos"]]

    for s, col in zip(slist, cols):
        d = [1 - retentions(build_beta_schedule(int(T), "cosine", SNR_MAX), s)["DDIM"].prod()
             for T in Ts]
        ax[0].semilogx(Ts, Ts * np.array(d), "o-", color=col, ms=3.5, label=rf"$s={s}$")
        ax[0].axhline(np.pi ** 2 * (1 + s * s) / (8 * s), color=col, ls=":", lw=1.0)
    ax[0].set_xlabel("$T$")
    ax[0].set_ylabel(r"$T\cdot D$")
    ax[0].set_title(r"(a) unshifted $\to\ \pi^2(1+s^2)/(8s)$ (dotted)")
    ax[0].legend(loc="upper right", frameon=False, ncol=2)

    for s, col, mk in zip(slist, cols, ["o", "s", "^", "v"]):
        d = [1 - retentions(sched_from_abar(shifted_abar(int(T), s)), s)["DDIM"].prod()
             for T in Ts]
        ax[1].semilogx(Ts, Ts * np.array(d), mk + "-", color=col, ms=5.0, mfc="none",
                       lw=1.0, label=rf"$s={s}$")
    ax[1].axhline(np.pi ** 2 / 4, color="k", ls=":", lw=1.1)
    ax[1].annotate(r"$\pi^2/4$", xy=(230, np.pi ** 2 / 4 + 0.02), fontsize=8)
    ax[1].annotate("all four coincide", xy=(24, 2.30), fontsize=7.5)
    ax[1].set_xlabel("$T$")
    ax[1].set_ylabel(r"$T\cdot D$")
    ax[1].set_title(r"(b) shifted $\to\ \pi^2/4$ for every $s$")
    ax[1].legend(loc="lower right", frameon=False, ncol=2)
    ax[1].set_ylim(2.25, 2.60)

    fig.tight_layout()
    fig.savefig(OUT / "fig_verification.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_information()
    fig_coordinates()
    fig_trajectories()
    fig_shift()
    fig_geometry()
    fig_allocation()
    fig_penalty()
    fig_heterogeneity()
    fig_verification()
    print("wrote:", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))
