#!/usr/bin/env python3
"""Plot critic action-curvature sensitivity across training snapshots.

Consumes the CSV from ``scripts.snapshot_q_hessian_sensitivity`` and renders

* ``*_sensitivity.png``  -- absolute S = 0.5 eps^2 sum|lambda_i| and the
  scale-free ratio S / |Q|, median with IQR band, one colour per environment;
* ``*_sensitivity_by_env.png`` -- per-env facets with every sampled action
  drawn individually, so the spread across the 30 actions is visible.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ENV_ORDER = [
    "Humanoid-v3", "Ant-v3", "Walker2d-v3",
    "Hopper-v3", "HalfCheetah-v3", "Swimmer-v3",
]
DEFAULT_CSV = Path("analysis_cache/sweep225_q_hessian_sensitivity.csv")
FIG_DIR = Path("figures")


def order_envs(envs):
    return [e for e in ENV_ORDER if e in envs] + [e for e in envs if e not in ENV_ORDER]


def agg(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(["env", "step"])[col]
    return pd.DataFrame({
        "med": g.median(),
        "lo": g.quantile(0.25),
        "hi": g.quantile(0.75),
    }).reset_index()


def band(ax, df, col, colors, envs, marker="o"):
    a = agg(df, col)
    for env in envs:
        s = a[a["env"] == env].sort_values("step")
        ax.plot(s["step"], s["med"], marker=marker, ms=5, lw=1.8,
                color=colors[env], label=env)
        ax.fill_between(s["step"], s["lo"], s["hi"], color=colors[env], alpha=0.18,
                        lw=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out-prefix", default="sweep225_qagg-mean")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    df["rel"] = df["sensitivity"] / df["q"].abs()
    if "q_loss_norm" not in df:
        raise SystemExit("csv lacks q_loss_norm; re-run snapshot_q_hessian_sensitivity")
    df["norm"] = df["sensitivity"] / df["q_loss_norm"]
    envs = order_envs(sorted(df["env"].unique()))
    cmap = plt.get_cmap("tab10")
    colors = {e: cmap(i % 10) for i, e in enumerate(envs)}
    eps = float(df["epsilon"].iloc[0])
    n_act = int(df.groupby(["env", "step"]).size().median())
    FIG_DIR.mkdir(exist_ok=True)

    sup = (r"Critic sensitivity to action perturbations   "
           r"$S=\mathbb{E}_{\delta\sim N(0,\epsilon^2 I)}[\hat Q(a{+}\delta)]-Q(a)"
           r"=\frac{1}{2}\epsilon^2\sum_i|\lambda_i|$"
           f"\nsweep 225, q_agg_sample=mean, {n_act} sampled actions/snapshot, "
           f"$\\epsilon$={eps:.5f} (= $\\sqrt{{1-\\bar\\alpha_1}}$, cosine N=100); "
           "median with IQR")

    # ---- figure 0: sigma_Q-normalised, every env on one axes ----------------
    fig, ax = plt.subplots(figsize=(10, 6))
    steps0 = np.array(sorted(df["step"].unique()), dtype=float)
    jw = 0.012 * (steps0.max() - steps0.min())
    rng0 = np.random.default_rng(0)
    for env in envs:
        sub = df[df["env"] == env]
        x = sub["step"].to_numpy(float) + rng0.uniform(-jw, jw, len(sub))
        ax.scatter(x, sub["norm"], s=7, alpha=0.28, color=colors[env],
                   edgecolors="none", zorder=1)
    band(ax, df, "norm", colors, envs)
    ax.set_yscale("log")
    ax.set_xlabel("training env step (snapshot)")
    ax.set_ylabel(r"$S\,/\,\sigma_Q$")
    ax.set_xticks(steps0)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.3, which="both", lw=0.5)
    ax.legend(fontsize=9, ncol=2, framealpha=0.9)
    ax.set_title(
        "Action-perturbation sensitivity of $Q$, as a fraction of the Q-loss "
        r"standardizer $\sigma_Q$" "\n"
        r"$S=\frac{1}{2}\epsilon^2\sum_i|\lambda_i|$,  "
        r"$\sigma_Q=\sqrt{\mathrm{EMA}(\mathcal{L}_Q)}$ (= Critic/q_loss_norm)" "\n"
        f"sweep 225, q_agg_sample=mean, $\\epsilon$={eps:.5f}; "
        f"dots = individual actions, line = median, band = IQR",
        fontsize=10.5)
    fig.tight_layout()
    out = FIG_DIR / f"{args.out_prefix}_q_sensitivity_normalized.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")

    # ---- figure 1: absolute + relative -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    band(axes[0], df, "sensitivity", colors, envs)
    axes[0].set_ylabel(r"$S=\frac{1}{2}\epsilon^2\sum_i|\lambda_i|$")
    axes[0].set_title("absolute", fontsize=11)

    band(axes[1], df, "rel", colors, envs)
    axes[1].set_ylabel(r"$S\,/\,|Q(s,a)|$")
    axes[1].set_title("relative to $|Q|$ (scale-free)", fontsize=11)

    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel("training env step (snapshot)")
        ax.grid(alpha=0.3, which="both", lw=0.5)
        ax.set_xticks(sorted(df["step"].unique()))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].legend(fontsize=8.5, ncol=2, framealpha=0.9)
    fig.suptitle(sup, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = FIG_DIR / f"{args.out_prefix}_q_sensitivity.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")

    # ---- figure 2: per-env facets, individual actions ----------------------
    n = len(envs)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.2 * nrow),
                             squeeze=False)
    steps = np.array(sorted(df["step"].unique()), dtype=float)
    jitter_w = 0.035 * (steps.max() - steps.min())
    rng = np.random.default_rng(0)
    for ax, env in zip(axes.ravel(), envs):
        sub = df[df["env"] == env]
        x = sub["step"].to_numpy(float) + rng.uniform(-jitter_w, jitter_w, len(sub))
        ax.scatter(x, sub["sensitivity"], s=9, alpha=0.45, color=colors[env],
                   edgecolors="none")
        a = agg(sub, "sensitivity").sort_values("step")
        ax.plot(a["step"], a["med"], color="k", lw=1.6, marker="o", ms=4,
                label="median")
        ax.set_yscale("log")
        ax.set_title(env, fontsize=11)
        ax.set_xticks(steps)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.grid(alpha=0.3, which="both", lw=0.5)
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("training env step", fontsize=9)
    for row in axes:
        row[0].set_ylabel(r"$S$", fontsize=10)
    fig.suptitle("Per-action sensitivity (each dot = one sampled action)\n" + sup,
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = FIG_DIR / f"{args.out_prefix}_q_sensitivity_by_env.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")

    # ---- console summary ---------------------------------------------------
    summ = (df.groupby(["env", "step"])
              .agg(S_med=("sensitivity", "median"),
                   sigma_Q=("q_loss_norm", "first"),
                   S_over_sigmaQ=("norm", "median"),
                   rel_med=("rel", "median"),
                   q_med=("q", "median"),
                   neg_eig_frac=("n_negative_eig", "mean"),
                   act_dim=("act_dim", "first"),
                   ret=("episode_return_mean", "first"))
              .reset_index())
    summ["neg_eig_frac"] = summ["neg_eig_frac"] / summ["act_dim"]
    with pd.option_context("display.float_format", "{:.4g}".format):
        print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
