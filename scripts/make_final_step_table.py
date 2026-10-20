#!/usr/bin/env python3
"""LaTeX table of single-checkpoint evaluation return, with a two-level bootstrap.

Statistic, per environment and algorithm:

1. at one checkpoint (1M env steps by default), average each seed's evaluation
   episodes -> one value per seed;
2. take the median over seeds.

Interval: a nested (hierarchical) percentile bootstrap that resamples both levels
of the design. Each replicate draws 10 seeds with replacement and then, for every
drawn seed, redraws its 10 episodes with replacement before averaging them. Both
sources of noise -- which seeds you happened to run, and which episodes those
seeds happened to play -- therefore enter the interval, unlike a seed-only
bootstrap which conditions on the observed episode means.

Episodes are redrawn independently each time a seed is selected, so a seed drawn
twice contributes two independent episode averages; that is what makes the two
stages independent rather than a duplicated row.

This reads the episode-level tables in ``baselines/`` directly, since the curve
CSVs written by plot_eval_method_comparison.py have already averaged episodes
away. Requires ``\\usepackage{booktabs}``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from plot_eval_method_comparison import BASELINE_DIR, ENVS, PRESETS  # noqa: E402

COLUMN_ORDER = ["MGMD (ours)", "DIPO", "DPMD", "SAC"]


def episode_matrix(path: Path, variant: str, eval_method: str,
                   step: int) -> np.ndarray | None:
    """seed x episode array of returns at one checkpoint."""
    df = pd.read_csv(path, usecols=["variant", "eval_method", "seed", "step",
                                    "episode_index", "episode_return"])
    df = df[(df["variant"] == variant) & (df["eval_method"] == eval_method)
            & (df["step"] == step)]
    if df.empty:
        return None
    m = (df.pivot_table(index="seed", columns="episode_index",
                        values="episode_return")
         .sort_index().sort_index(axis=1))
    return m.to_numpy(dtype=float)


def nested_bootstrap(ep: np.ndarray, level: float, n_boot: int,
                     rng: np.random.Generator) -> tuple[float, float, float]:
    """Point estimate and percentile CI under resampling of seeds and episodes."""
    n_seeds, n_eps = ep.shape
    centre = float(np.median(ep.mean(axis=1)))
    seed_idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    ep_idx = rng.integers(0, n_eps, size=(n_boot, n_seeds, n_eps))
    # ep[seed, episode] gathered for every (replicate, drawn seed, drawn episode)
    drawn = ep[seed_idx[:, :, None], ep_idx]
    reps = np.median(drawn.mean(axis=2), axis=1)
    lo, hi = np.percentile(reps, [50 * (1 - level), 50 * (1 + level)])
    return centre, float(lo), float(hi)


def fmt(x: float, decimals: int) -> str:
    return f"{x:.{decimals}f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    ap.add_argument("--step", type=int, default=1_000_000)
    ap.add_argument("--ci-level", type=float, default=0.50)
    ap.add_argument("--n-boot", type=int, default=20_000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--label", default="tab:algorithm_comparison")
    ap.add_argument("--out", type=Path,
                    default=REPO / "figures" / "final_step_table.tex")
    ap.add_argument("--csv", type=Path,
                    default=REPO / "figures" / "final_step_table.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.boot_seed)
    recs = []
    for algorithm, variant, method, label, *_ in PRESETS["simple"]:
        for env in ENVS:
            path = args.baseline_dir / algorithm / f"{algorithm}__{env}__eval_episodes.csv"
            if not path.exists():
                continue
            ep = episode_matrix(path, variant, method, args.step)
            if ep is None:
                print(f"  WARNING no rows: {label} {env} @ {args.step}")
                continue
            centre, lo, hi = nested_bootstrap(ep, args.ci_level, args.n_boot, rng)
            recs.append(dict(series=label, algorithm=algorithm, variant=variant,
                             eval_method=method, env=env, step=args.step,
                             n_seeds=ep.shape[0], n_episodes=ep.shape[1],
                             center=centre, ci_lo=lo, ci_hi=hi,
                             ci_level=args.ci_level))
    df = pd.DataFrame(recs)
    df.to_csv(args.csv, index=False)

    cols = [c for c in COLUMN_ORDER if c in set(df["series"])]
    lines = [
        r"\begin{table}[t]",
        rf"\caption{{Evaluation return at {args.step / 1e6:.0f}M environment steps. "
        r"For each run we average its evaluation episodes at that checkpoint and "
        r"report the median over seeds, with a "
        rf"{100 * args.ci_level:.0f}\% percentile bootstrap confidence interval "
        r"obtained by resampling both seeds and episodes within each seed. "
        r"Bold marks algorithms whose estimate lies within the confidence "
        r"interval of the best-performing method for that environment.}",
        rf"\label{{{args.label}}}",
        "",
        r"\begin{center}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        "",
        r"\begin{tabular}{l|" + "c" * len(cols) + "}",
        r"\toprule",
        r"\textbf{Environment}",
    ] + [rf"& \textbf{{{c}}}" for c in cols] + [r"\\", r"\midrule", ""]

    for env in ENVS:
        g = df[df["env"] == env].set_index("series")
        if g.empty:
            continue
        decimals = 1 if g["center"].abs().max() < 1000 else 0
        threshold = g.loc[g["center"].idxmax(), "ci_lo"]
        lines.append(env)
        for c in cols:
            r = g.loc[c]
            val = fmt(r["center"], decimals)
            if r["center"] >= threshold:
                val = rf"\textbf{{{val}}}"
            cell = (rf"& {val} {{\scriptsize [{fmt(r['ci_lo'], decimals)}, "
                    rf"{fmt(r['ci_hi'], decimals)}]}}")
            lines.append(cell + (r" \\" if c == cols[-1] else ""))
        lines.append("")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{center}", r"\end{table}"]
    tex = "\n".join(lines)
    args.out.write_text(tex + "\n")
    print(tex)
    print(f"\nWrote {args.out}\nWrote {args.csv}", flush=True)


if __name__ == "__main__":
    main()
