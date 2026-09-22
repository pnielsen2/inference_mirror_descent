#!/usr/bin/env python3
"""Per-env response along the axes of the adaptive-oracle table's top rows.

The table says the two best per-env-adaptive knobs on the same base config are
``buffer_size`` (oracle 0.994, gain +0.025) and ``eta`` (oracle 0.990, gain
+0.021). A row only reports the per-env ARGMAX and the resulting oracle, which
cannot say whether a finer grid would help: an axis whose envs all agree and
whose loser is far behind is finished, while one whose envs split between two
ADJACENT levels is a grid that is too coarse, and the value each env wants is
somewhere between the two it was offered.

So plot the whole 1-D slice: for every level of the axis, every env's fraction
of its best baseline, with a seed bootstrap interval. What justifies a finer
sweep is the shape -- picks split across neighbouring levels, with the gaps
between those levels comparable to the bootstrap noise and the never-picked
levels clearly worse.

Reads ``per_seed_values.csv`` / ``baselines.csv`` / ``adaptive_oracle.csv`` from
a ``compute_sweep_normalized_scores`` output dir; touches no run mirrors.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from compute_sweep_normalized_scores import _fmt, normalized_score  # noqa: E402
from compute_topsis import ENVS  # noqa: E402

SCORE_DIR = SCRIPT_DIR / "topsis_out" / "sweep_237+239" / "normalized_eval"
# The two rows to draw, as (axis, base_config) keys into adaptive_oracle.csv.
ROWS = (("buffer_size", "eta=64 gamma=0.99 rollout_alpha=1"),
        ("eta", "buffer_size=400000 gamma=0.99 rollout_alpha=1"))
ENV_COLORS = dict(zip(ENVS, plt.get_cmap("tab10").colors))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score-dir", type=Path, default=SCORE_DIR)
    ap.add_argument("--row", nargs=2, action="append", metavar=("AXIS", "BASE_CONFIG"),
                    help="repeatable; defaults to the table's top two rows")
    ap.add_argument("--envs", nargs="+", default=list(ENVS))
    ap.add_argument("--n-boot", type=int, default=20000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=SCRIPT_DIR.parent / "figures" /
                            "sweep237_adaptive_oracle_axis_response.png")
    return ap.parse_args(argv)


def tag_numeric(tag, key):
    """The numeric value ``key`` takes in a ``config_tag``, else None.

    Anchored on ``_<key>=`` so ``eta`` does not match the ``eta=`` inside
    ``beta=``/``theta=``, and restricted to a numeric literal so it stops at the
    next axis without needing to know the axis list.
    """
    m = re.search(rf"(?:^|_){re.escape(key)}=([0-9]+\.?[0-9]*(?:e[+-]?[0-9]+)?)", tag)
    return float(m.group(1)) if m else None


def members(tags, axis, base):
    """``{level: tag}`` for the tags matching ``base`` and differing in ``axis``.

    Levels are keyed the way the table formats them (``1e+06`` -> ``1000000``)
    so a level label here is the same string the table's ``pick_*`` column uses.
    """
    want = {}
    for item in base.split():
        k, v = item.split("=")
        want[k] = float(v)
    out = {}
    for t in tags:
        lv = tag_numeric(t, axis)
        if lv is None or axis in want:
            continue
        if any(tag_numeric(t, k) != v for k, v in want.items()):
            continue
        out[_fmt(lv)] = t
    return out


def response(seed_vals, denom, envs, level_tags, n_boot, rng):
    """Per-(level, env) fraction with a seed-bootstrap interval.

    Same resampling as the table's: resample each cell's seeds with replacement,
    which is the only noise source the 3 seeds let us estimate.
    """
    levels = sorted(level_tags, key=float)
    fr = np.empty((len(levels), len(envs)))
    lo = np.empty_like(fr)
    hi = np.empty_like(fr)
    boot = np.empty((len(levels), len(envs), n_boot))
    for li, lv in enumerate(levels):
        for ei, env in enumerate(envs):
            vals = np.asarray(seed_vals[(level_tags[lv], env)], float)
            fr[li, ei] = vals.mean() / denom[env]
            idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
            b = vals[idx].mean(axis=1) / denom[env]
            boot[li, ei] = b
            lo[li, ei], hi[li, ei] = np.percentile(b, [5, 95])
    # Score of each single level (no adaptation) and of the per-env max.
    sc = normalized_score(fr, axis=1)
    bs_fixed = normalized_score(boot.transpose(2, 0, 1).reshape(-1, len(envs)),
                                axis=1).reshape(n_boot, len(levels))
    bs_oracle = normalized_score(boot.max(axis=0).T, axis=1)
    return dict(levels=levels, frac=fr, lo=lo, hi=hi, score=sc,
                oracle=float(normalized_score(fr.max(axis=0)[None, :], axis=1)[0]),
                boot_gain=bs_oracle - bs_fixed.max(axis=1))


def plot(rows, table, args):
    fig, axs = plt.subplots(1, len(rows), figsize=(8.0 * len(rows), 6.8),
                            constrained_layout=True)
    axs = np.atleast_1d(axs)
    for ax, (axis, base, r) in zip(axs, rows):
        x = np.arange(len(r["levels"]), dtype=float)
        for ei, env in enumerate(args.envs):
            c = ENV_COLORS[env]
            best = int(np.argmax(r["frac"][:, ei]))
            ax.errorbar(x, r["frac"][:, ei],
                        yerr=[r["frac"][:, ei] - r["lo"][:, ei],
                              r["hi"][:, ei] - r["frac"][:, ei]],
                        color=c, marker="o", markersize=5, linewidth=1.8,
                        capsize=3, elinewidth=1.1, zorder=3,
                        label=f"{env.replace('-v3', '')} (best {r['levels'][best]})")
            ax.plot([x[best]], [r["frac"][best, ei]], marker="o", markersize=12,
                    markerfacecolor="none", markeredgecolor=c,
                    markeredgewidth=2.0, zorder=4)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1.3, zorder=1)
        ax.annotate("parity with best baseline", (0.985, 1.0),
                    xycoords=("axes fraction", "data"), ha="right", va="bottom",
                    fontsize=8, color="black")
        # Score of the best single level vs the per-env oracle: the row's claim.
        bi = int(np.argmax(r["score"]))
        g = r["boot_gain"]
        n_wanted = len({int(np.argmax(r["frac"][:, ei]))
                        for ei in range(len(args.envs))})
        ax.set_title(
            f"--{axis}        base: {base}\n"
            f"best single level {r['levels'][bi]}: {r['score'][bi]:.4f}"
            f"     per-env oracle: {r['oracle']:.4f}\n"
            f"gain +{r['oracle'] - r['score'][bi]:.4f} "
            f"[{np.percentile(g, 5):+.4f}, {np.percentile(g, 95):+.4f}], "
            f"{100 * np.mean(g > 0):.0f}% > 0"
            f"     envs split over {n_wanted} of {len(r['levels'])} levels",
            fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels(r["levels"])
        ax.set_xlim(-0.25, len(x) - 0.75)
        ax.set_xlabel(f"--{axis}")
        ax.set_ylabel("fraction of the env's best baseline (eval, 800k-1M)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2, loc="best", framealpha=0.9)
    fig.suptitle(
        "Per-env response along the adaptive-oracle table's top two axes "
        "(sweep 233+237 pooled seeds)\n"
        "ring = that env's argmax, the level the table's pick_* column reports;  "
        "error bars = 5-95% seed bootstrap\n"
        "a finer grid is worth running where the envs split across ADJACENT "
        "levels and the levels never picked are clearly worse",
        fontsize=12)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.out}")


def main(argv=None):
    args = parse_args(argv)
    psv = pd.read_csv(args.score_dir / "per_seed_values.csv")
    denom = pd.read_csv(args.score_dir / "baselines.csv").groupby("env")["value"].max()
    table = pd.read_csv(args.score_dir / "adaptive_oracle.csv")
    seed_vals = {k: g["value"].to_numpy()
                 for k, g in psv.groupby(["config_tag", "env"])}
    tags = sorted(psv["config_tag"].unique())
    rng = np.random.default_rng(args.boot_seed)

    rows = []
    for axis, base in (args.row or ROWS):
        lt = members(tags, axis, base)
        ref = table[(table.axis == axis) & (table.base_config == base)]
        if ref.empty:
            raise SystemExit(f"no adaptive_oracle.csv row for {axis!r} / {base!r}")
        want = ref.iloc[0]["levels"].split(",")
        if set(lt) != set(want):
            raise SystemExit(f"{axis}: found levels {sorted(lt)}, table says {sorted(want)}")
        r = response(seed_vals, denom, args.envs, lt, args.n_boot, rng)
        # The table row is the ground truth for this slice; reproduce it.
        assert abs(r["oracle"] - float(ref.iloc[0]["oracle"])) < 1e-9, (
            f"{axis}: oracle {r['oracle']} != table {ref.iloc[0]['oracle']}")
        for ei, env in enumerate(args.envs):
            got = _fmt(float(r["levels"][int(np.argmax(r["frac"][:, ei]))]))
            assert got == _fmt(ref.iloc[0][f"pick_{env}"]), (
                f"{axis}/{env}: pick {got} != table {ref.iloc[0][f'pick_{env}']}")
        print(f"--{axis} (base {base}): levels {r['levels']}, "
              f"oracle {r['oracle']:.4f} matches table")
        for ei, env in enumerate(args.envs):
            print(f"    {env:16s} " + "  ".join(
                f"{lv}:{r['frac'][li, ei]:.3f}" for li, lv in enumerate(r["levels"])))
        rows.append((axis, base, r))
    plot(rows, table, args)


if __name__ == "__main__":
    main()
