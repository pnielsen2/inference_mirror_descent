#!/usr/bin/env python3
"""Plot the critic loss (``losses/Q_loss``) training curve for every run in a
sweep, colour-coded by environment.

Reads offline wandb run dirs directly (no network) via
``scripts.offline_wandb``. Per-run histories are cached as CSV under
``analysis_cache/`` so re-plotting is cheap.

Example
-------
    python scripts/plot_sweep_q_loss.py --sweep 225 --q-agg mean
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

ENV_ORDER = [
    "Humanoid-v3",
    "Ant-v3",
    "Walker2d-v3",
    "Hopper-v3",
    "HalfCheetah-v3",
    "Swimmer-v3",
]


def load_curves(sweep: int, key: str, base: Path, q_agg: str | None,
                refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = index_runs([sweep], base=base)
    if idx.empty:
        raise SystemExit(f"no offline runs found for sweep {sweep} under {base}")
    if q_agg is not None:
        idx = idx[idx["q_agg_sample"] == q_agg]
        if idx.empty:
            raise SystemExit(f"sweep {sweep} has no runs with q_agg_sample={q_agg}")

    CACHE_DIR.mkdir(exist_ok=True)
    safe_key = key.replace("/", "_")
    frames = []
    for _, r in idx.iterrows():
        cache = CACHE_DIR / f"sweep{sweep}_{r.run_id}_{safe_key}.csv"
        if cache.exists() and not refresh:
            hist = pd.read_csv(cache)
        else:
            hist = read_history(r.run_dir, keys={key})
            hist = hist[hist["key"] == key][["step", "value"]].sort_values("step")
            hist.to_csv(cache, index=False)
        if hist.empty:
            print(f"  [warn] {r.run_id} ({r.env}) has no {key} history; skipping")
            continue
        frames.append(hist.assign(env=r.env, seed=r.seed, run_id=r.run_id))
        print(f"  {r.env:<16s} seed={r.seed:<3} {r.run_id}  "
              f"{len(hist):5d} pts  final step {int(hist['step'].iloc[-1]):>9,d}")
    if not frames:
        raise SystemExit(f"no {key} history found in any run")
    return pd.concat(frames, ignore_index=True), idx


def smooth(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 1:
        return df
    out = df.copy()
    out["value"] = (
        out.groupby("run_id")["value"]
        .transform(lambda s: s.rolling(window, min_periods=1, center=True).median())
    )
    return out


def env_colors(envs) -> dict:
    ordered = [e for e in ENV_ORDER if e in envs] + [e for e in envs if e not in ENV_ORDER]
    cmap = plt.get_cmap("tab10")
    return {e: cmap(i % 10) for i, e in enumerate(ordered)}, ordered


def plot_combined(df: pd.DataFrame, colors: dict, ordered: list, key: str,
                  title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for env in ordered:
        sub = df[df["env"] == env]
        for i, (_, g) in enumerate(sub.groupby("run_id")):
            ax.plot(g["step"], g["value"], color=colors[env], lw=1.2, alpha=0.85,
                    label=env if i == 0 else None)
    ax.set_yscale("log")
    ax.set_xlabel("environment step")
    ax.set_ylabel(f"{key}  (log scale)")
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both", lw=0.5)
    ax.legend(title="environment", fontsize=9, title_fontsize=9,
              loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def plot_facets(df: pd.DataFrame, colors: dict, ordered: list, key: str,
                title: str, out: Path) -> None:
    n = len(ordered)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow),
                             squeeze=False)
    for ax, env in zip(axes.ravel(), ordered):
        sub = df[df["env"] == env]
        for _, g in sub.groupby("run_id"):
            ax.plot(g["step"], g["value"], color=colors[env], lw=1.2, alpha=0.85)
        ax.set_title(env, fontsize=11)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both", lw=0.5)
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("environment step", fontsize=9)
    for row in axes:
        row[0].set_ylabel(key, fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", type=int, required=True)
    p.add_argument("--key", default="losses/Q_loss")
    p.add_argument("--q-agg", default=None,
                   help="filter on config q_agg_sample (e.g. mean, min)")
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--smooth", type=int, default=1,
                   help="centred rolling-median window in logged points")
    p.add_argument("--refresh", action="store_true",
                   help="re-scan .wandb logs instead of using the CSV cache")
    p.add_argument("--out-prefix", default=None)
    args = p.parse_args()

    print(f"indexing sweep {args.sweep} under {args.base} ...")
    df, idx = load_curves(args.sweep, args.key, args.base, args.q_agg, args.refresh)

    colors, ordered = env_colors(sorted(df["env"].unique()))
    df_s = smooth(df, args.smooth)

    tag = f"sweep{args.sweep}"
    if args.q_agg:
        tag += f"_qagg-{args.q_agg}"
    prefix = args.out_prefix or tag
    FIG_DIR.mkdir(exist_ok=True)

    n_runs = df["run_id"].nunique()
    last = int(df["step"].max())
    sub = f"sweep {args.sweep}"
    if args.q_agg:
        sub += f", q_agg_sample={args.q_agg}"
    sub += f" | {n_runs} runs | up to {last:,} env steps"
    if args.smooth > 1:
        sub += f" | rolling median w={args.smooth}"

    plot_combined(df_s, colors, ordered, args.key,
                  f"Critic loss ({args.key})\n{sub}",
                  FIG_DIR / f"{prefix}_q_loss.png")
    plot_facets(df_s, colors, ordered, args.key,
                f"Critic loss ({args.key}) — {sub}",
                FIG_DIR / f"{prefix}_q_loss_by_env.png")


if __name__ == "__main__":
    main()
