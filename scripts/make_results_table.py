#!/usr/bin/env python3
"""LaTeX results table from a curve CSV written by plot_eval_method_comparison.py.

Rows are environments, columns algorithms, cells the value at one checkpoint
(the last by default) with its confidence interval underneath.

A cell is bolded when it is statistically indistinguishable from the best in its
row: take the row's top-scoring algorithm and bold everyone whose value falls
inside that algorithm's confidence interval. This marks the set of methods the
data cannot separate from the winner, rather than just the single argmax, which
would overstate how decisive a small gap is.

Requires ``\\usepackage{booktabs}`` (already present in paper/MGMD.tex).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent

# MGMD first, as the method under test; the baselines follow alphabetically.
COLUMN_ORDER = ["MGMD (ours)", "DIPO", "DPMD", "SAC"]


def fmt(x: float, decimals: int) -> str:
    return f"{x:,.{decimals}f}".replace(",", r"\,")


def build_table(df: pd.DataFrame, step: int, caption: str, label: str) -> str:
    sub = df[df["step"] == step]
    if sub.empty:
        raise SystemExit(f"no rows at step {step}")

    cols = [c for c in COLUMN_ORDER if c in set(sub["series"])]
    cols += [c for c in sorted(set(sub["series"])) if c not in cols]
    envs = sorted(sub["env"].unique())

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{l" + "c" * len(cols) + "}",
        r"\toprule",
        "Environment & " + " & ".join(cols) + r" \\",
        r"\midrule",
    ]

    for env in envs:
        g = sub[sub["env"] == env].set_index("series")
        # Scale the precision to the environment: Swimmer lives near 150, the
        # others in the thousands, so a single global rounding would either
        # drown Swimmer or pad the rest with meaningless digits.
        decimals = 1 if g["center"].abs().max() < 1000 else 0
        best = g["center"].idxmax()
        threshold = g.loc[best, "ci_lo"]

        cells = []
        for c in cols:
            if c not in g.index:
                cells.append("--")
                continue
            r = g.loc[c]
            val = fmt(r["center"], decimals)
            if r["center"] >= threshold:
                val = rf"\textbf{{{val}}}"
            ci = rf"{{\scriptsize [{fmt(r['ci_lo'], decimals)}, {fmt(r['ci_hi'], decimals)}]}}"
            cells.append(rf"\shortstack{{{val} \\ {ci}}}")
        lines.append(f"{env} & " + " & ".join(cells) + r" \\")
        # Two-line cells butt against each other without this.
        lines.append(r"\addlinespace[2pt]")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path,
                    default=REPO / "figures"
                    / "algorithm_comparison_trailing_of_median.csv")
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint to report (default: the last)")
    ap.add_argument("--caption", default=None)
    ap.add_argument("--label", default="tab:algorithm_comparison")
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to the CSV path with a _table.tex suffix")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    step = args.step if args.step is not None else int(df["step"].max())

    stat = df["statistic"].iloc[0]
    level = float(df["ci_coverage"].iloc[0])
    n_seeds = int(df["n_seeds"].iloc[0])
    caption = args.caption or (
        f"Evaluation return at {step / 1e6:.0f}M environment steps "
        f"({stat} over {n_seeds} seeds, {100 * level:.0f}\\% confidence interval). "
        r"\textbf{Bold} marks every method whose value lies within the interval "
        "of the best method for that environment."
    )

    tex = build_table(df, step, caption, args.label)
    out = args.out or args.csv.with_name(args.csv.stem + "_table.tex")
    out.write_text(tex + "\n")
    print(tex)
    print(f"\nWrote {out}", flush=True)


if __name__ == "__main__":
    main()
