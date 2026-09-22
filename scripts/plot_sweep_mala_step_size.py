#!/usr/bin/env python3
"""Show whether the MALA step-size cap was binding, per environment.

Plots ``MALA/step_size_max`` and ``MALA/step_size_mean`` (mean over the runs of
each env) against the cap that was in force, plus the acceptance rate against
its target. A run whose ``step_size_max`` sits exactly on the cap line is one
where the clip, not the Robbins-Monro rule, is setting the step size.

Example
-------
    python scripts/plot_sweep_mala_step_size.py --sweep 231 --cap 0.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scripts.offline_wandb import index_runs, read_history

DEFAULT_BASE = Path("/n/holylabs/kdbrantley_lab/Lab/pnielsen/wandb")
CACHE_DIR = Path("analysis_cache")
FIG_DIR = Path("figures")

ENV_ORDER = ["Humanoid-v3", "Ant-v3", "Walker2d-v3",
             "Hopper-v3", "HalfCheetah-v3", "Swimmer-v3"]
ACT_DIM = {"Swimmer-v3": 2, "Hopper-v3": 3, "Walker2d-v3": 6,
           "HalfCheetah-v3": 6, "Ant-v3": 8, "Humanoid-v3": 17}

KEYS = ["MALA/step_size_max", "MALA/step_size_mean", "MALA/step_size_noisiest",
        "MALA/acceptance_rate_mean"]


def load(sweep: int, base: Path, refresh: bool) -> pd.DataFrame:
    idx = index_runs([sweep], base=base)
    if idx.empty:
        raise SystemExit(f"no offline runs for sweep {sweep} under {base}")
    CACHE_DIR.mkdir(exist_ok=True)
    frames = []
    for _, r in idx.iterrows():
        cache = CACHE_DIR / f"sweep{sweep}_{r.run_id}_mala.csv"
        if cache.exists() and not refresh:
            h = pd.read_csv(cache)
        else:
            h = read_history(r.run_dir, keys=set(KEYS))
            h.to_csv(cache, index=False)
        if h.empty:
            continue
        w = h.pivot_table(index="step", columns="key", values="value").reset_index()
        frames.append(w.assign(env=r.env, run_id=r.run_id,
                               target=r.get("mala_target_acceptance_rate")))
    if not frames:
        raise SystemExit(f"sweep {sweep} logs no MALA/* metrics")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", type=int, required=True)
    p.add_argument("--cap", type=float, default=0.5,
                   help="step-size cap in force during the sweep")
    p.add_argument("--cap-coef", type=float, default=2.7,
                   help="also draw coef*act_dim^(-1/3), the dimension-aware cap")
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()

    df = load(args.sweep, args.base, args.refresh)
    envs = [e for e in ENV_ORDER if e in set(df["env"])]
    print(f"sweep {args.sweep}: {df['run_id'].nunique()} runs, {len(envs)} envs")

    ncol = 3
    nblock = (len(envs) + ncol - 1) // ncol
    fig, axes = plt.subplots(2 * nblock, ncol, figsize=(4.6 * ncol, 3.1 * 2 * nblock),
                             squeeze=False)
    for i, env in enumerate(envs):
        row, j = 2 * (i // ncol), i % ncol
        sub = df[df["env"] == env]
        g = sub.groupby("step")
        ax = axes[row][j]
        ax.plot(g["MALA/step_size_max"].mean().index,
                g["MALA/step_size_max"].mean().values,
                color="tab:red", lw=1.8, label="max over levels")
        ax.plot(g["MALA/step_size_mean"].mean().index,
                g["MALA/step_size_mean"].mean().values,
                color="tab:blue", lw=1.5, label="mean over levels")
        if "MALA/step_size_noisiest" in sub:
            ax.plot(g["MALA/step_size_noisiest"].mean().index,
                    g["MALA/step_size_noisiest"].mean().values,
                    color="tab:green", lw=1.2, ls=":", label="noisiest level")
        ax.axhline(args.cap, color="k", ls="--", lw=1.4,
                   label=f"cap in force = {args.cap}")
        new_cap = args.cap_coef * ACT_DIM[env] ** (-1 / 3)
        ax.axhline(new_cap, color="tab:purple", ls="-.", lw=1.2,
                   label=f"dim-aware cap = {new_cap:.2f}")
        ax.set_ylim(0, max(new_cap, args.cap) * 1.15)
        ax.set_title(f"{env}   (act_dim {ACT_DIM[env]})", fontsize=11)
        ax.set_ylabel("MALA step size h", fontsize=9)
        ax.grid(alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1e6:g}M")
        ax.legend(fontsize=7, loc="center right", framealpha=0.9)

        ax = axes[row + 1][j]
        for tgt, s in sorted(sub.groupby("target"), key=lambda x: float(x[0])):
            gg = s.groupby("step")["MALA/acceptance_rate_mean"].mean()
            line, = ax.plot(gg.index, gg.values, lw=1.3,
                            label=f"realized (target {float(tgt):.4g})")
            ax.axhline(float(tgt), color=line.get_color(), ls=":", lw=1.0)
        ax.set_ylim(0.55, 0.85)
        ax.set_ylabel("acceptance rate", fontsize=9)
        ax.grid(alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1e6:g}M")
        ax.set_xlabel("environment step", fontsize=9)
        ax.legend(fontsize=7, loc="upper right", framealpha=0.9, ncol=1)
    for ax in axes.ravel()[2 * len(envs):]:
        ax.axis("off")
    fig.suptitle(
        f"sweep {args.sweep}: the step-size clip at {args.cap}, not the "
        f"Robbins-Monro rule, is setting h\n"
        f"(curves averaged over the {df['run_id'].nunique() // len(envs)} runs per env; "
        f"acceptance stays above its target because the rule is asking for a larger step)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / f"sweep{args.sweep}_mala_step_size_cap.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")

    tail = (df.sort_values("step").groupby(["env", "run_id"]).tail(20)
            .groupby("env")[KEYS].mean().round(3).loc[envs])
    tail["cap"] = args.cap
    tail["frac_of_cap_mean"] = (tail["MALA/step_size_mean"] / args.cap).round(2)
    print("\ntail means (last 20 logged points per run):")
    print(tail.to_string())


if __name__ == "__main__":
    main()
