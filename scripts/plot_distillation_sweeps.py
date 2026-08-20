#!/usr/bin/env python3
"""Training curves + final-episode-return grids for the policy-distillation
sweeps (166, 174, and 175+177 pooled).

Reads only LOCAL files (same sources as build_sweep_local_metrics.py):
  * $WANDB_OFFLINE_BASE/sweep_<N>/job_*/wandb/offline-run-*/files/config.yaml
    for each vmap slot's env + hyperparameters,
  * <codebase>/logs/<env>/<exp_dir.name>/episode_returns.csv for the curves.

Two figure families per sweep, matching the visual language of the 8-3-26 deck:
  * training curves -- one subplot per env, one line per value of the sweep's
    "line" hyperparameter, mean over seeds with a t-based CI band
    (plot_sweep_training_curves_by_hp.py style);
  * final-return grids -- one figure per env, RdYlGn cells annotated with the
    tail-return mean (mean episode return in the last TAIL_WINDOW_STEPS = 50k
    env steps, the same metric compute_topsis / the deck's
    "<env>_tail_return_heatmap.png" use), plus an "Overall" figure of
    fraction-of-baseline vs the LSAC benchmark (plot_sweep131_132_grids.py
    style).

The tail metric is anchored on each slot's own last logged episode, so
in-progress sweeps (166 / 175 are finished, 174 / 177 are not) are plotted at
their current progress; each unfinished sweep's completion fraction is printed
and stamped into the figure title by name.

Sweeps that share a design and sweep disjoint cells of it are pooled into one
figure -- 175 (eta 8/16/32) and 177 (eta 64/128/256) are otherwise flag-for-flag
identical, so they form a single 3 x 6 (buffer x eta) grid.

Examples:
    python scripts/plot_distillation_sweeps.py
    python scripts/plot_distillation_sweeps.py --sweeps 175+177 --spread ci
    python scripts/plot_distillation_sweeps.py --sweeps 175   # 175 on its own
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from build_sweep_local_metrics import (  # noqa: E402
    _cfg_value,
    _codebase_dir_from_metadata,
    _read_run_basename,
)
from compute_topsis import (  # noqa: E402
    MIN_EPISODES,
    TAIL_WINDOW_STEPS,
    load_benchmark_scores,
    log_score,
)
from plot_sweep131_132_grids import (  # noqa: E402
    _color_bounds,
    _draw_panel,
    _fmt_num,
    _select,
    _sort_scalar,
)
from plot_sweep_training_curves_by_hp import (  # noqa: E402
    aggregate_curves,
    windowed_mean_curve,
)

from relax.utils.fs import WANDB_OFFLINE_BASE as DEFAULT_WANDB_BASE  # noqa: E402

REPO_DIR = SCRIPT_DIR.parent
FIG_ROOT = REPO_DIR / "figures" / "distillation_sweeps"
ENV_ORDER = [
    "Ant-v3",
    "HalfCheetah-v3",
    "Hopper-v3",
    "Humanoid-v3",
    "Swimmer-v3",
    "Walker2d-v3",
]
# Every hyperparameter any of these sweeps varies, plus what is needed to
# resolve a slot and judge completion.
CFG_KEYS = [
    "env",
    "seed_index",
    "config_tag",
    "total_step",
    "T",
    "eta",
    "diffusion_steps",
    "distillation_steps",
    "distillation_buffer_size",
    "lr_policy",
]
LABELS = {
    "eta": r"$\eta$",
    "T": "T",
    "diffusion_steps": "diffusion steps",
    "distillation_steps": "distill steps",
    "distillation_buffer_size": "distill buffer",
    "lr_policy": r"lr$_\pi$",
}
COMPLETION_FRAC = 0.95


# ------------------------------------------------------------------ specs ----
@dataclass
class SweepSpec:
    """How to slice one sweep (or a group of sweeps sharing a design) into
    figures.

    ``line_cols``  -> one curve per unique value combination (colour),
    ``facet_cols`` -> one curve-subplot column per combination (envs are rows),
    ``fig_cols``   -> a separate curve figure per combination,
    ``grid_*``     -> rows / columns / panels of the final-return grids.
    """

    title: str
    line_cols: tuple[str, ...]
    grid_rows: tuple[str, ...]
    grid_cols: tuple[str, ...]
    facet_cols: tuple[str, ...] = ()
    fig_cols: tuple[str, ...] = ()
    grid_facet_cols: tuple[str, ...] = ()
    held: dict[str, str] = field(default_factory=dict)


# Keyed by the sweep ids that share one design. Sweeps in the same key are
# pooled into one figure, which is only sound because they sweep *disjoint*
# cells of the same grid with every other flag identical (175: eta 8/16/32,
# 177: eta 64/128/256) -- so no cell mixes runs from two sweeps.
SWEEP_SPECS: dict[tuple[int, ...], SweepSpec] = {
    # eta x (distillation buffer, distillation steps) x diffusion steps on the
    # two hard envs, 1 seed per config.
    (166,): SweepSpec(
        title="distillation grid, Ant + Humanoid, 1 seed/config",
        line_cols=("eta",),
        facet_cols=("distillation_buffer_size", "distillation_steps"),
        fig_cols=("diffusion_steps",),
        grid_rows=("distillation_buffer_size", "distillation_steps"),
        grid_cols=("eta",),
        grid_facet_cols=("diffusion_steps",),
        held={"T": "0", "lr_policy": "3e-4"},
    ),
    # distillation_steps x lr_policy at eta=8 over all 6 envs.
    (174,): SweepSpec(
        title="distillation steps x lr$_\\pi$, $\\eta$=8",
        line_cols=("distillation_steps",),
        fig_cols=("lr_policy",),
        grid_rows=("lr_policy",),
        grid_cols=("distillation_steps",),
        held={"T": "0", "eta": "8", "diffusion_steps": "40", "distillation_buffer_size": "1"},
    ),
    # distillation_buffer_size x eta over all 6 envs; 177 extends eta upward.
    (175, 177): SweepSpec(
        title="distillation buffer size x $\\eta$",
        line_cols=("eta",),
        fig_cols=("distillation_buffer_size",),
        grid_rows=("distillation_buffer_size",),
        grid_cols=("eta",),
        held={"T": "0", "diffusion_steps": "40", "distillation_steps": "1", "lr_policy": "3e-4"},
    ),
}


# ---------------------------------------------------------------- loading ----
def _load_slot_config(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return {k: _cfg_value(cfg, k) for k in CFG_KEYS}


def collect_slots(sweep_root: Path, sweep_id: int) -> tuple[list[dict], Counter]:
    """Return (records, skipped). One record per vmap slot: its config values
    plus ``hist`` (a step-sorted DataFrame with ``_step`` / ``return``)."""
    job_dirs = sorted(sweep_root.glob("job_*"))
    if not job_dirs:
        raise SystemExit(f"No job dirs under {sweep_root}")
    print(f"Found {len(job_dirs)} job dirs under {sweep_root}", flush=True)

    records: list[dict] = []
    skipped: Counter = Counter()
    for i, jd in enumerate(job_dirs, 1):
        run_dirs = sorted((jd / "wandb").glob("offline-run-*"))
        if not run_dirs:
            skipped["no_offline_runs"] += 1
            continue

        slot_cfg, base_name, codebase, env = {}, None, None, None
        for rd in run_dirs:
            cfg_path = rd / "files" / "config.yaml"
            if not cfg_path.exists():
                continue
            info = _load_slot_config(cfg_path)
            if info.get("seed_index") is None:
                continue
            slot_cfg[int(info["seed_index"])] = info
            env = env or info.get("env")
            if base_name is None:
                wf = list(rd.glob("run-*.wandb"))
                if wf:
                    base_name = _read_run_basename(wf[0])
            if codebase is None:
                codebase = _codebase_dir_from_metadata(rd / "files" / "wandb-metadata.json")

        if not slot_cfg or base_name is None or codebase is None or env is None:
            skipped["unresolvable_job"] += 1
            continue

        csv_path = codebase / "logs" / env / base_name / "episode_returns.csv"
        if not csv_path.exists():
            skipped["missing_csv"] += 1
            continue

        ep_col = f"episode_return/{env}"
        df = pd.read_csv(csv_path)
        if not {"seed", "step", ep_col} <= set(df.columns):
            skipped["bad_csv_columns"] += 1
            continue
        df = df.dropna(subset=[ep_col])

        n_slots = 0
        for slot, sub in df.groupby("seed"):
            info = slot_cfg.get(int(slot))
            if info is None:
                continue
            hist = (
                sub[["step", ep_col]]
                .rename(columns={"step": "_step", ep_col: "return"})
                .sort_values("_step")
                .reset_index(drop=True)
            )
            if hist.empty:
                continue
            records.append({**info, "sweep_id": sweep_id, "hist": hist})
            n_slots += 1
        print(f"  [{i}/{len(job_dirs)}] {jd.name} {env:<15} {n_slots} slots", flush=True)

    if not records:
        raise SystemExit(f"No slot curves reconstructed under {sweep_root}")
    return records, skipped


def tail_metric(hist: pd.DataFrame):
    """(tail_mean, max_step) over the final TAIL_WINDOW_STEPS, or None."""
    steps = hist["_step"].to_numpy(dtype=float)
    returns = hist["return"].to_numpy(dtype=float)
    if returns.size < MIN_EPISODES:
        return None
    max_step = float(steps.max())
    tail = returns[steps > max_step - TAIL_WINDOW_STEPS]
    if tail.size == 0:
        return None
    return float(np.mean(tail)), max_step


def slot_table(records: list[dict]) -> pd.DataFrame:
    """One row per slot with the tail metric; drops slots too short to score.

    ``record_idx`` points back into ``records`` so a row's curve is unambiguous
    (seed_index is only unique *within* a job, so config keys alone would alias
    two seeds of the same config that landed in the same vmap slot)."""
    rows = []
    for i, r in enumerate(records):
        res = tail_metric(r["hist"])
        if res is None:
            continue
        metric, max_step = res
        row = {k: r.get(k) for k in CFG_KEYS}
        row["sweep_id"] = r.get("sweep_id")
        row["metric"] = metric
        row["max_step"] = max_step
        row["record_idx"] = i
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- labels ----
def _fmt_value(key: str, value) -> str:
    if key == "lr_policy":
        return f"{float(value):.1e}".replace("e-0", "e-")
    return _fmt_num(value)


def _combo_label(cols: tuple[str, ...], key: tuple) -> str:
    return ", ".join(f"{LABELS.get(c, c)}={_fmt_value(c, v)}" for c, v in zip(cols, key))


def _combo_ticks(cols: tuple[str, ...], key: tuple) -> str:
    """Bare values for grid tick labels; the axis label carries the names."""
    return " / ".join(_fmt_value(c, v) for c, v in zip(cols, key))


def _combo_slug(cols: tuple[str, ...], key: tuple) -> str:
    parts = [f"{c}{_fmt_value(c, v)}" for c, v in zip(cols, key)]
    return "_".join(parts).replace(".", "p").replace("+", "").replace("-", "m")


def _combos(df: pd.DataFrame, cols: tuple[str, ...]) -> list[tuple]:
    if not cols:
        return [()]
    tuples = {tuple(r) for r in df[list(cols)].dropna().to_numpy()}
    return sorted(tuples, key=lambda t: tuple(_sort_scalar(v) for v in t))


def _held_note(spec: SweepSpec) -> str:
    if not spec.held:
        return ""
    return "held: " + ", ".join(
        f"{LABELS.get(k, k)}={v}" for k, v in spec.held.items()
    )


def _two_line_title(headline: str, spec: SweepSpec, progress_note: str, *extra: str) -> str:
    """Headline first, caveats on their own line -- a single long caption would
    stretch the figure box and shrink everything else on the slide."""
    sub = [s for s in (*extra, _held_note(spec), progress_note) if s]
    return headline + ("\n" + " | ".join(sub) if sub else "")


def _progress(df: pd.DataFrame) -> float:
    """Median max_step / total_step over the sweep's slots."""
    total = pd.to_numeric(df["total_step"], errors="coerce").to_numpy(dtype=float)
    frac = df["max_step"].to_numpy(dtype=float) / np.where(total > 0, total, np.nan)
    frac = frac[np.isfinite(frac)]
    return float(np.median(frac)) if frac.size else float("nan")


def _sweep_label(ids: tuple[int, ...]) -> str:
    return ("Sweeps " if len(ids) > 1 else "Sweep ") + "+".join(str(i) for i in ids)


def _sweep_slug(ids: tuple[int, ...]) -> str:
    return "_".join(str(i) for i in ids)


def _progress_note(table: pd.DataFrame, ids: tuple[int, ...]) -> str:
    """Name the unfinished sweeps individually -- when two sweeps are pooled,
    one can be done and the other days from it, and a single pooled median would
    hide that."""
    parts = []
    for sid in ids:
        sub = table[table["sweep_id"] == sid]
        if sub.empty:
            continue
        frac = _progress(sub)
        if np.isfinite(frac) and frac < COMPLETION_FRAC:
            parts.append(f"sweep {sid} at median {frac:.0%}")
    return "IN PROGRESS: " + ", ".join(parts) + " of total_step" if parts else ""


# ---------------------------------------------------------- curve figures ----
def plot_curves(records, table, spec, ids, out_dir, args, progress_note):
    """One figure per ``fig_cols`` combination; envs as subplots, one coloured
    line per ``line_cols`` combination (mean over the remaining runs)."""
    df = table
    hists = [r["hist"] for r in records]
    outputs = []
    for fig_key in _combos(df, spec.fig_cols):
        fsub = _select(df, list(spec.fig_cols), fig_key) if spec.fig_cols else df
        if fsub.empty:
            continue
        facet_keys = _combos(fsub, spec.facet_cols)
        line_keys = _combos(fsub, spec.line_cols)
        envs = [e for e in ENV_ORDER if e in set(fsub["env"])]
        envs += [e for e in sorted(set(fsub["env"])) if e not in envs]

        # Bucket the per-slot curves by (env, facet, line).
        curves: dict[tuple, list[pd.DataFrame]] = defaultdict(list)
        for facet_key in facet_keys:
            fac = _select(fsub, list(spec.facet_cols), facet_key) if spec.facet_cols else fsub
            for line_key in line_keys:
                cell = _select(fac, list(spec.line_cols), line_key)
                for env, idx in zip(cell["env"], cell["record_idx"]):
                    curves[(env, facet_key, line_key)].append(hists[int(idx)])
        if not curves:
            continue

        derived_max = max(int(h["_step"].max()) for hs in curves.values() for h in hs)
        max_steps = args.max_steps or int(
            np.ceil(derived_max / args.bin_size) * args.bin_size
        )
        bins = np.arange(args.bin_size, max_steps + args.bin_size, args.bin_size, dtype=int)
        aggregates = {}
        for key, hs in curves.items():
            binned = [windowed_mean_curve(h, bins, args.bin_size) for h in hs]
            mean, ci, _ = aggregate_curves(binned, args.ci_level)
            aggregates[key] = (mean, ci, len(hs), binned)

        if spec.facet_cols:
            nrows, ncols = len(envs), len(facet_keys)
        else:
            ncols = 3 if len(envs) > 2 else len(envs)
            nrows = int(np.ceil(len(envs) / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(4.6 * ncols, 3.5 * nrows), squeeze=False
        )
        cmap = plt.get_cmap("viridis")
        colors = {
            lk: cmap(i / max(1, len(line_keys) - 1)) for i, lk in enumerate(line_keys)
        }
        x = bins / 1e6

        def panel_axes(env_idx, facet_idx):
            if spec.facet_cols:
                return axes[env_idx][facet_idx]
            flat = axes.ravel()
            return flat[env_idx]

        used = set()
        for e_i, env in enumerate(envs):
            for f_i, facet_key in enumerate(facet_keys):
                ax = panel_axes(e_i, f_i)
                used.add(ax)
                for line_key in line_keys:
                    entry = aggregates.get((env, facet_key, line_key))
                    if entry is None:
                        continue
                    mean, ci, n_runs, binned = entry
                    valid = ~np.isnan(mean)
                    if not valid.any():
                        continue
                    ax.plot(
                        x[valid], mean[valid], color=colors[line_key], linewidth=1.6,
                        label=f"{_combo_label(spec.line_cols, line_key)} (n={n_runs})",
                    )
                    if args.spread == "ci":
                        band = valid & ~np.isnan(ci)
                        if band.any():
                            ax.fill_between(
                                x[band], mean[band] - ci[band], mean[band] + ci[band],
                                color=colors[line_key], alpha=0.15, linewidth=0,
                            )
                    elif args.spread == "runs":
                        for run_curve in binned:
                            rv = ~np.isnan(run_curve)
                            if rv.any():
                                ax.plot(x[rv], run_curve[rv], color=colors[line_key],
                                        linewidth=0.7, alpha=0.35)
                title = env.replace("-v3", "")
                if spec.facet_cols:
                    title += f"\n{_combo_label(spec.facet_cols, facet_key)}"
                ax.set_title(title, fontsize=10)
                ax.set_xlabel("Env steps (M)", fontsize=9)
                if f_i == 0 or not spec.facet_cols:
                    ax.set_ylabel("Episode return", fontsize=9)
                ax.grid(True, alpha=0.3, linestyle="--")
                ax.tick_params(labelsize=8)
                ax.set_xlim(0, max_steps / 1e6)
                ax.legend(fontsize=7, loc="upper left", framealpha=0.85)

        for ax in axes.ravel():
            if ax not in used:
                ax.set_axis_off()

        spread_note = {
            "ci": f"bands = {args.ci_level:.0%} CI",
            "runs": "faint lines = individual runs",
            "none": "mean only",
        }[args.spread]
        headline = f"{_sweep_label(ids)} ({spec.title})"
        if spec.fig_cols:
            headline += f" - {_combo_label(spec.fig_cols, fig_key)}"
        fig.suptitle(_two_line_title(headline, spec, progress_note, spread_note),
                     fontsize=11)
        # Reserve the top strip for the (two-line) caption rather than letting
        # tight_layout run the panel titles into it.
        fig.tight_layout(rect=(0, 0, 1, 0.90 if nrows > 1 else 0.80))

        slug = _combo_slug(spec.fig_cols, fig_key) if spec.fig_cols else "all"
        out_path = out_dir / f"sweep{_sweep_slug(ids)}_training_curves_{slug}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)
        print(f"  wrote {out_path}", flush=True)
    return outputs


# ----------------------------------------------------------- grid figures ----
def _matrix(df, spec_rows, spec_cols, row_keys, col_keys, value_col="value"):
    m = np.full((len(row_keys), len(col_keys)), np.nan, dtype=float)
    for i, rk in enumerate(row_keys):
        rsub = _select(df, list(spec_rows), rk)
        for j, ck in enumerate(col_keys):
            cell = _select(rsub, list(spec_cols), ck)[value_col].to_numpy(dtype=float)
            cell = cell[np.isfinite(cell)]
            if cell.size:
                m[i, j] = cell.mean()
    return m


def _grid_figure(df, spec, suptitle, cbar_label, value_fmt, out_path, axes_keys):
    # Axes come from the whole sweep, not from ``df``, so every env's figure has
    # the same rows / columns and a cell missing in one env reads as grey rather
    # than silently shifting the axis.
    row_keys, col_keys, facet_keys = axes_keys
    mats = []
    for fk in facet_keys:
        sub = _select(df, list(spec.grid_facet_cols), fk) if spec.grid_facet_cols else df
        mats.append(_matrix(sub, spec.grid_rows, spec.grid_cols, row_keys, col_keys))
    vmin, vmax = _color_bounds(mats)

    ncol = len(facet_keys)
    # Floor the height so a 2-row grid still leaves the (rotated) colorbar label
    # room next to the caption instead of being clipped by it.
    figsize = ((0.75 * len(col_keys) + 1.9) * ncol,
               max(0.75 * len(row_keys) + 2.0, 3.4))
    fig, axes = plt.subplots(1, ncol, figsize=figsize, constrained_layout=True, squeeze=False)
    flat = list(axes.ravel())
    last_im = None
    for idx, (fk, m) in enumerate(zip(facet_keys, mats)):
        last_im = _draw_panel(
            flat[idx], m, vmin, vmax,
            _combo_label(spec.grid_facet_cols, fk) if spec.grid_facet_cols else "",
            [_combo_ticks(spec.grid_rows, rk) for rk in row_keys],
            [_combo_ticks(spec.grid_cols, ck) for ck in col_keys],
            " / ".join(LABELS.get(c, c) for c in spec.grid_cols),
            " / ".join(LABELS.get(c, c) for c in spec.grid_rows),
            value_fmt=value_fmt,
            fonts={"cell": 8.0, "tick": 8.0, "axis": 9.0, "title": 9.5},
        )
    # A narrow grid (few columns) is easily narrower than its caption, and an
    # over-wide suptitle overlaps the colorbar label -- so wrap it.
    fig.suptitle("\n".join(textwrap.fill(ln, 72) for ln in suptitle.split("\n")),
                 fontsize=11)
    cbar = fig.colorbar(last_im, ax=flat, shrink=0.85)
    cbar.set_label(cbar_label, fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)
    return out_path


def plot_grids(table, spec, ids, out_dir, progress_note):
    """Per-env tail-return grids + an overall fraction-of-baseline grid."""
    key_cols = list(dict.fromkeys(spec.grid_rows + spec.grid_cols + spec.grid_facet_cols))
    per_env = (
        table.groupby(["env"] + key_cols, dropna=False)["metric"]
        .agg(value="mean", n_seeds="count")
        .reset_index()
    )
    envs = [e for e in ENV_ORDER if e in set(per_env["env"])]
    envs += [e for e in sorted(set(per_env["env"])) if e not in envs]
    axes_keys = (_combos(per_env, spec.grid_rows),
                 _combos(per_env, spec.grid_cols),
                 _combos(per_env, spec.grid_facet_cols))

    outputs = []
    tail_note = f"tail return = mean episode return in the final {TAIL_WINDOW_STEPS // 1000}k env steps"
    for env in envs:
        sub = per_env[per_env["env"] == env]
        outputs.append(_grid_figure(
            sub, spec,
            _two_line_title(f"{_sweep_label(ids)} ({spec.title})\n"
                            f"{env}: final episode return", spec, progress_note),
            f"Episode return, final {TAIL_WINDOW_STEPS // 1000}k env steps",
            ".0f",
            out_dir / f"sweep{_sweep_slug(ids)}_{env.replace('-', '_')}_final_return_grid.png",
            axes_keys,
        ))

    # Overall: fraction of the LSAC baseline, averaged over the envs present.
    benchmarks = load_benchmark_scores(envs)
    per_env = per_env.copy()
    per_env["log_score"] = [
        log_score(v, benchmarks[e]) for v, e in zip(per_env["value"], per_env["env"])
    ]
    overall = (
        per_env.groupby(key_cols, dropna=False)
        .agg(log_score=("log_score", "mean"), n_envs=("env", "nunique"))
        .reset_index()
    )
    overall["value"] = np.power(2.0, overall["log_score"])
    # Only cells covering every env are comparable, exactly as in
    # build_sweep_local_metrics; a cell short of that is left blank.
    overall.loc[overall["n_envs"] < len(envs), "value"] = np.nan
    n_full = int(np.isfinite(overall["value"]).sum())
    # An overall grid whose cells are mostly blank (a sweep whose envs have not
    # all reported yet) says less than the per-env grids it is made from.
    if n_full >= 2 and 2 * n_full >= len(overall):
        outputs.append(_grid_figure(
            overall, spec,
            _two_line_title(
                f"{_sweep_label(ids)} ({spec.title})\n"
                f"Overall fraction of baseline (mean over {len(envs)} envs)",
                spec, progress_note),
            "Fraction of LSAC baseline (capped at 1.0)", ".3f",
            out_dir / f"sweep{_sweep_slug(ids)}_overall_fraction_of_baseline_grid.png",
            axes_keys,
        ))
    else:
        print(f"  only {n_full} config(s) cover all {len(envs)} envs; "
              f"skipped the overall fraction-of-baseline grid", flush=True)
    print(f"  {tail_note}", flush=True)
    return outputs


# ------------------------------------------------------------------- main ----
def _resolve_groups(tokens) -> list[tuple[tuple[int, ...], SweepSpec]]:
    """Turn CLI tokens into (sweep ids, spec) pairs.

    ``175+177`` pools those sweeps; a bare ``175`` reuses the same spec but plots
    that sweep on its own, which is how you get the pre-177 figures back."""
    if not tokens:
        return list(SWEEP_SPECS.items())
    groups = []
    for tok in tokens:
        ids = tuple(int(p) for p in str(tok).split("+"))
        spec = SWEEP_SPECS.get(ids)
        if spec is None:
            owners = [k for k in SWEEP_SPECS if set(ids) <= set(k)]
            if len(owners) != 1:
                raise SystemExit(
                    f"No SweepSpec covers {ids}; add one to SWEEP_SPECS "
                    f"(known groups: {sorted(SWEEP_SPECS)})"
                )
            spec = SWEEP_SPECS[owners[0]]
        groups.append((ids, spec))
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweeps", nargs="+", default=None,
                    help="Sweep groups to plot, e.g. '166' or '175+177' (pooled "
                         "into one figure). A single id from a pooled group plots "
                         "that sweep alone. Default: every group in SWEEP_SPECS.")
    ap.add_argument("--wandb-base", type=Path, default=DEFAULT_WANDB_BASE)
    ap.add_argument("--out-dir", type=Path, default=FIG_ROOT)
    ap.add_argument("--bin-size", type=int, default=10_000)
    ap.add_argument("--ci-level", type=float, default=0.90)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--spread", default="runs", choices=["ci", "runs", "none"],
                    help="Within-group spread. These sweeps carry 1-2 seeds per cell, "
                         "where a t-based CI is so wide it swamps the means "
                         "(n=2 -> t_crit=6.31 at 90%%), so the default draws one faint "
                         "line per run instead.")
    ap.add_argument("--skip-curves", action="store_true")
    ap.add_argument("--skip-grids", action="store_true")
    args = ap.parse_args()

    groups = _resolve_groups(args.sweeps)
    outputs = []
    for ids, spec in groups:
        print(f"\n=== {_sweep_label(ids).lower()} ===", flush=True)
        records, skipped = [], Counter()
        for sid in ids:
            recs, skip = collect_slots(args.wandb_base / f"sweep_{sid}", sid)
            records += recs
            skipped += skip
        table = slot_table(records)
        if table.empty:
            print(f"  no slot reached {MIN_EPISODES} episodes; skipping", flush=True)
            continue
        if skipped:
            print(f"  skipped jobs: {dict(skipped)}", flush=True)
        for sid in ids:
            sub = table[table["sweep_id"] == sid]
            print(f"  sweep {sid}: {len(sub)} scoreable slots, "
                  f"{sub['config_tag'].nunique()} configs, "
                  f"median completion {_progress(sub):.2f}", flush=True)
        progress_note = _progress_note(table, ids)
        out_dir = args.out_dir / f"sweep_{_sweep_slug(ids)}"
        if not args.skip_curves:
            outputs += plot_curves(records, table, spec, ids, out_dir, args, progress_note)
        if not args.skip_grids:
            outputs += plot_grids(table, spec, ids, out_dir, progress_note)

    print("\nFigures:")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
