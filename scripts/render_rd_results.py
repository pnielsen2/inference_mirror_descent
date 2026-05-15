#!/usr/bin/env python3
"""Render a beamer presentation summarizing an R_d sweep batch.

For each ablation (and each sub-ablation, treated as its own slide), produce a
matplotlib plot of per-level geometric means of the mean of the last K episode
returns for runs in that level, with bootstrap confidence intervals. Then
compile the whole thing with pdflatex.

Usage:
    python scripts/render_rd_results.py 3
    python scripts/render_rd_results.py 3 --last-k 11 --bootstrap 2000
"""

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR / "sweeps" / "rd_sweep"))
import rd_sweep  # noqa: E402


def load_manifest(batches_dir: Path, batch_id: int):
    path = batches_dir / str(batch_id) / "manifest.yaml"
    if not path.exists():
        sys.exit(f"render_rd_results.py: no manifest at {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def env_from_base(base_tokens):
    for i, t in enumerate(base_tokens):
        if t == "--env" and i + 1 < len(base_tokens):
            return base_tokens[i + 1]
    return None


def fetch_run_last_k_mean(api, project, suffix, env_name, last_k):
    """Return mean of last last_k episode_return values for the run with this suffix.

    None if the run isn't found or has no data.
    """
    runs = list(api.runs(project, filters={"config.suffix": suffix}))
    if not runs:
        return None
    run = max(runs, key=lambda r: r.created_at)
    keys = [f"episode_return/{env_name}", "sample/episode_return", "_step"]
    try:
        hist = run.history(keys=keys, samples=5000)
    except Exception:
        return None
    if hist is None or len(hist) == 0:
        return None
    series = None
    for k in keys[:2]:
        if k in hist.columns and hist[k].notna().any():
            series = hist[k].dropna().values
            break
    if series is None or len(series) == 0:
        return None
    tail = series[-last_k:]
    return float(np.mean(tail))


def geomean(xs):
    xs = [x for x in xs if x is not None and x > 0 and math.isfinite(x)]
    if not xs:
        return None
    return float(np.exp(np.mean(np.log(xs))))


def bootstrap_geomean_ci(xs, n_boot=2000, alpha=0.05, rng=None):
    """Percentile bootstrap CI for the geometric mean. Returns (lo, hi) or (None, None)."""
    xs = np.array([x for x in xs if x is not None and x > 0 and math.isfinite(x)], dtype=float)
    if len(xs) < 2:
        return None, None
    if rng is None:
        rng = np.random.default_rng(0)
    logs = np.log(xs)
    idx = rng.integers(0, len(logs), size=(n_boot, len(logs)))
    sampled = logs[idx].mean(axis=1)
    lo, hi = np.quantile(sampled, [alpha / 2, 1 - alpha / 2])
    return float(np.exp(lo)), float(np.exp(hi))


# ---------- Ablation traversal ----------

def iter_slides(ablations):
    """Yield (title, ablation, filter_path) for every ablation in the tree.

    filter_path is the chain of (ablation_name, level_name) selections that
    a run's own path must contain for it to count toward this slide. For
    top-level ablations filter_path is empty; for nested ablations it's the
    ancestor chain of level selections.
    """
    for ab, parent_path in rd_sweep.walk_ablations(ablations):
        if parent_path:
            title = " / ".join([f"{a}={l}" for a, l in parent_path] + [ab["name"]])
        else:
            title = ab["name"]
        yield title, ab, parent_path


def run_matches_filter(run_path, filter_path):
    """run_path is a list of {'ablation', 'level'}; filter_path is [(ab,lvl), ...]."""
    if not filter_path:
        return True
    path_set = {(p["ablation"], p["level"]) for p in run_path}
    return all((a, l) in path_set for a, l in filter_path)


def run_level_for_ablation(run_path, ab_name):
    for p in run_path:
        if p["ablation"] == ab_name:
            return p["level"]
    return None


# ---------- Plot / LaTeX ----------

def make_plot(ab, level_stats, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [lvl["name"] for lvl in ab["levels"]]
    xs = np.arange(len(names))
    ys = [level_stats[n]["gmean"] for n in names]
    los = [level_stats[n]["ci_lo"] for n in names]
    his = [level_stats[n]["ci_hi"] for n in names]
    ns = [level_stats[n]["n"] for n in names]

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for i, (y, lo, hi) in enumerate(zip(ys, los, his)):
        if y is None:
            continue
        err_lo = 0 if lo is None else max(0.0, y - lo)
        err_hi = 0 if hi is None else max(0.0, hi - y)
        ax.errorbar([xs[i]], [y], yerr=[[err_lo], [err_hi]], fmt="o",
                    capsize=4, color="C0", markersize=6)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{n}\n(n={ns[i]})" for i, n in enumerate(names)])
    ax.set_ylabel("geomean of mean(last-K episode_return)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def latex_escape(s):
    return (str(s)
            .replace("\\", r"\textbackslash{}")
            .replace("&", r"\&").replace("%", r"\%").replace("$", r"\$")
            .replace("#", r"\#").replace("_", r"\_").replace("{", r"\{")
            .replace("}", r"\}").replace("~", r"\textasciitilde{}")
            .replace("^", r"\textasciicircum{}")
            .replace("<", r"\textless{}").replace(">", r"\textgreater{}"))


BEAMER_PREAMBLE = r"""\documentclass{beamer}
\usepackage{graphicx}
\usepackage{booktabs}
\setbeamertemplate{navigation symbols}{}
\setbeamerfont{frametitle}{size=\small}
"""


def render_latex(out_dir, manifest, slides):
    tex_path = out_dir / "report.tex"
    lines = [BEAMER_PREAMBLE]
    title = f"R\\_d sweep batch {manifest['batch_id']}"
    lines.append(rf"\title{{{title}}}")
    lines.append(r"\date{" + latex_escape(manifest["created_at"]) + "}")
    lines.append(r"\begin{document}")
    lines.append(r"\frame{\titlepage}")

    lines.append(r"\begin{frame}{Summary}")
    lines.append(r"\begin{itemize}")
    lines.append(rf"\item batch {manifest['batch_id']}, n = {manifest['n']} runs")
    lines.append(rf"\item rd\_seed = {manifest['rd_seed']}")
    lines.append(rf"\item sweep file: \texttt{{{latex_escape(manifest['sweep_file'])}}}")
    lines.append(r"\end{itemize}")
    lines.append(r"\end{frame}")

    for s in slides:
        lines.append(r"\begin{frame}{" + latex_escape(s["title"]) + "}")
        lines.append(r"\centering")
        lines.append(rf"\includegraphics[width=0.95\textwidth]{{{s['plot']}}}")
        if s["note"]:
            lines.append(r"\vspace{0.2em}\footnotesize " + latex_escape(s["note"]))
        lines.append(r"\end{frame}")

    lines.append(r"\end{document}")
    tex_path.write_text("\n".join(lines))
    return tex_path


def compile_pdf(tex_path: Path):
    if shutil.which("pdflatex") is None:
        print("pdflatex not found; skipping compile. .tex is at", tex_path)
        return None
    for _ in range(2):
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent, capture_output=True, text=True,
        )
        if r.returncode != 0:
            print("pdflatex failed:")
            print(r.stdout[-2000:])
            return None
    return tex_path.with_suffix(".pdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("batch_id", type=int)
    ap.add_argument("--batches-dir", type=Path,
                    default=PROJECT_DIR / "sweeps" / "rd_sweep" / "batches")
    ap.add_argument("--last-k", type=int, default=11)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--wandb-project", type=str, default="diffusion_online_rl")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output dir for report (default: <batches-dir>/<id>/report)")
    ap.add_argument("--no-compile", action="store_true",
                    help="Only emit .tex, skip pdflatex")
    args = ap.parse_args()

    manifest = load_manifest(args.batches_dir, args.batch_id)
    sweep = manifest["sweep"]
    base_tokens = sweep["base"]
    env_name = env_from_base(base_tokens)
    if env_name is None:
        sys.exit("render_rd_results.py: couldn't find --env in base command")

    out_dir = args.out_dir or (args.batches_dir / str(args.batch_id) / "report")
    (out_dir / "figs").mkdir(parents=True, exist_ok=True)

    # Pull run metrics.
    import wandb
    api = wandb.Api(timeout=120)
    print(f"Fetching {len(manifest['runs'])} runs from wandb...")
    run_values = {}  # suffix -> mean of last-K episode returns
    for r in manifest["runs"]:
        v = fetch_run_last_k_mean(api, args.wandb_project, r["suffix"],
                                  env_name, args.last_k)
        run_values[r["suffix"]] = v
        status = "OK" if v is not None else "MISSING"
        print(f"  {r['suffix']}  {status}  {'' if v is None else f'{v:.2f}'}")

    # Build per-slide stats.
    rng = np.random.default_rng(0)
    slides = []
    for title, ab, filter_path in iter_slides(sweep["ablations"]):
        level_stats = {}
        for lvl in ab["levels"]:
            xs = []
            for r in manifest["runs"]:
                if not run_matches_filter(r["path"], filter_path):
                    continue
                if run_level_for_ablation(r["path"], ab["name"]) != lvl["name"]:
                    continue
                v = run_values.get(r["suffix"])
                if v is not None:
                    xs.append(v)
            gm = geomean(xs)
            lo, hi = bootstrap_geomean_ci(xs, n_boot=args.bootstrap, rng=rng)
            level_stats[lvl["name"]] = {
                "gmean": gm, "ci_lo": lo, "ci_hi": hi, "n": len(xs),
            }

        slug = title.replace(" ", "_").replace(">", "gt").replace("/", "_")
        plot_rel = f"figs/{slug}.pdf"
        plot_abs = out_dir / plot_rel
        make_plot(ab, level_stats, plot_abs, title)

        total = sum(s["n"] for s in level_stats.values())
        note = f"runs included: {total}"
        slides.append({"title": title, "plot": plot_rel, "note": note})

    tex_path = render_latex(out_dir, manifest, slides)
    print(f"Wrote {tex_path}")
    if not args.no_compile:
        pdf = compile_pdf(tex_path)
        if pdf:
            print(f"Compiled {pdf}")


if __name__ == "__main__":
    main()
