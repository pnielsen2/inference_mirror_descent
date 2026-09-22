#!/usr/bin/env python3
"""Per-environment eval curves for one ablation axis of a sweep, averaged over
every other axis (seeds included).

Reads offline wandb run dirs directly (no network) via ``scripts.offline_wandb``
and caches per-run histories as CSV under ``analysis_cache/``.

With no ``--sweep`` the newest sweep in which ``--group-key`` actually varies is
selected, so ``--group-key q_agg_sample`` alone answers "how did min vs mean
compare last time we ran both?".

Example
-------
    python scripts/plot_sweep_ablation_eval.py --group-key q_agg_sample
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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


def sweep_ids(base: Path) -> list[int]:
    ids = []
    for d in base.glob("sweep_*"):
        try:
            ids.append(int(d.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(ids, reverse=True)


def latest_sweep_with_axis(group_key: str, base: Path) -> tuple[int, pd.DataFrame]:
    """Newest sweep whose runs take more than one value of ``group_key``."""
    for sw in sweep_ids(base):
        idx = index_runs([sw], base=base)
        if idx.empty or group_key not in idx:
            continue
        n = idx[group_key].nunique(dropna=True)
        print(f"  sweep {sw:>4}: {len(idx):3d} runs, {n} distinct {group_key}")
        if n > 1:
            return sw, idx
    raise SystemExit(f"no sweep under {base} varies {group_key}")


def load_curves(idx: pd.DataFrame, sweep: int, key: str, refresh: bool) -> pd.DataFrame:
    CACHE_DIR.mkdir(exist_ok=True)
    safe_key = key.replace("/", "_")
    frames = []
    for _, r in idx.iterrows():
        cache = CACHE_DIR / f"sweep{r.get('sweep', sweep)}_{r.run_id}_{safe_key}.csv"
        if cache.exists() and not refresh:
            hist = pd.read_csv(cache)
        else:
            hist = read_history(r.run_dir, keys={key})
            hist = hist[hist["key"] == key][["step", "value"]].sort_values("step")
            hist.to_csv(cache, index=False)
        if hist.empty:
            print(f"  [warn] {r.run_id} ({r.env}) has no {key} history; skipping")
            continue
        frames.append(hist.assign(env=r.env, seed=r.seed, run_id=r.run_id, group=r.group))
    if not frames:
        raise SystemExit(f"no {key} history found in any run")
    return pd.concat(frames, ignore_index=True)


def aggregate(df: pd.DataFrame, common_horizon: bool) -> pd.DataFrame:
    """Mean over runs at each logged step, with the across-run sem and count.

    ``common_horizon`` truncates each (env, group) to the last step every one of
    its runs reached, so the mean is never a composition of different run sets.
    """
    out = []
    for (env, grp), sub in df.groupby(["env", "group"]):
        if common_horizon:
            horizon = sub.groupby("run_id")["step"].max().min()
            sub = sub[sub["step"] <= horizon]
        g = sub.groupby("step")["value"]
        agg = pd.DataFrame({"mean": g.mean(), "sd": g.std(ddof=1), "n": g.size()}).reset_index()
        agg["sem"] = agg["sd"] / np.sqrt(agg["n"].clip(lower=1))
        out.append(agg.assign(env=env, group=grp))
    return pd.concat(out, ignore_index=True)


def smooth(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 1:
        return df
    out = df.copy()
    for col in ("mean", "sem"):
        out[col] = out.groupby(["env", "group"])[col].transform(
            lambda s: s.rolling(window, min_periods=1, center=True).mean())
    return out


def plot_facets(agg: pd.DataFrame, key: str, group_key: str, title: str, out: Path,
                legend_loc: str = "lower right") -> None:
    envs = [e for e in ENV_ORDER if e in set(agg["env"])] + \
           [e for e in sorted(set(agg["env"])) if e not in ENV_ORDER]
    groups = sorted(agg["group"].unique(), key=str)
    cmap = plt.get_cmap("tab10")
    colors = {g: cmap(i % 10) for i, g in enumerate(groups)}

    ncol = 3
    nrow = (len(envs) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.2 * nrow), squeeze=False)
    for ax, env in zip(axes.ravel(), envs):
        sub = agg[agg["env"] == env]
        for g in groups:
            s = sub[sub["group"] == g].sort_values("step")
            if s.empty:
                continue
            ax.plot(s["step"], s["mean"], color=colors[g], lw=1.6,
                    label=f"{group_key}={g}  (n={int(s['n'].max())})")
            ax.fill_between(s["step"], s["mean"] - s["sem"], s["mean"] + s["sem"],
                            color=colors[g], alpha=0.2, lw=0)
        ax.set_title(env, fontsize=11)
        ax.grid(alpha=0.3, lw=0.5)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8, loc=legend_loc, framealpha=0.9)
    for ax in axes.ravel()[len(envs):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("environment step", fontsize=9)
    for row in axes:
        row[0].set_ylabel(key, fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", type=int, nargs="+", default=None,
                   help="sweep id(s), pooled into one comparison; "
                        "default = newest sweep where --group-key varies")
    p.add_argument("--group-key", default="q_agg_sample",
                   help="config field whose values become the compared curves")
    p.add_argument("--key", default="eval/episode_return_mean")
    p.add_argument("--base", type=Path, default=DEFAULT_BASE)
    p.add_argument("--smooth", type=int, default=1,
                   help="centred rolling-mean window in logged points")
    p.add_argument("--no-common-horizon", action="store_true",
                   help="keep steps beyond the shortest run of a group")
    p.add_argument("--refresh", action="store_true",
                   help="re-scan .wandb logs instead of using the CSV cache")
    p.add_argument("--legend-loc", default="lower right",
                   help="matplotlib legend location for each facet")
    p.add_argument("--out-prefix", default=None)
    args = p.parse_args()

    if args.sweep is None:
        print(f"searching for the newest sweep varying {args.group_key} ...")
        sweep, idx = latest_sweep_with_axis(args.group_key, args.base)
    else:
        idx = index_runs(args.sweep, base=args.base)
        sweep = "-".join(str(s) for s in args.sweep)
        if idx.empty:
            raise SystemExit(f"no offline runs for sweep {sweep} under {args.base}")
        if args.group_key not in idx:
            raise SystemExit(f"{args.group_key} is not indexed by offline_wandb.CONFIG_FIELDS")
    idx = idx.rename(columns={args.group_key: "group"})
    idx = idx[idx["group"].notna()]

    print(f"sweep {sweep}: {len(idx)} runs, {args.group_key} in "
          f"{sorted(idx['group'].unique(), key=str)}, {idx['env'].nunique()} envs")
    other = [c for c in idx.columns
             if c not in {"group", "env", "seed", "seed_index", "run_id", "run_dir",
                          "job", "sweep", "config_tag"} and idx[c].nunique(dropna=False) > 1]
    print(f"averaging over: seed + {other if other else '(no other varying axis)'}")

    df = load_curves(idx, sweep, args.key, args.refresh)
    agg = smooth(aggregate(df, not args.no_common_horizon), args.smooth)

    prefix = args.out_prefix or f"sweep{sweep}_{args.group_key}"
    safe_key = args.key.replace("/", "_")
    sub = (f"sweep {sweep} | {df['run_id'].nunique()} runs | mean over seeds"
           f"{' + ' + ', '.join(other) if other else ''}, band = sem across runs")
    if args.smooth > 1:
        sub += f" | rolling mean w={args.smooth}"
    plot_facets(agg, args.key, args.group_key, f"{args.key} by {args.group_key}\n{sub}",
                FIG_DIR / f"{prefix}_{safe_key}_by_env.png", legend_loc=args.legend_loc)

    final = (agg.sort_values("step").groupby(["env", "group"]).tail(1)
             .pivot(index="env", columns="group", values="mean").round(1))
    print("\nfinal-point mean (at each group's common horizon):")
    print(final.to_string())
    csv = CACHE_DIR / f"{prefix}_{safe_key}_agg.csv"
    agg.to_csv(csv, index=False)
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
