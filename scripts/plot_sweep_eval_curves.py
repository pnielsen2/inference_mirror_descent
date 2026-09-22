#!/usr/bin/env python3
"""Evaluation curves for a finished sweep, either marginalized per ablation axis
or for one pooled config, always over the same three baselines.

Two modes:

``--mode axes`` (default)
    One PNG per (sweep, ablation axis): 6 panels, one per env, with one curve
    per value of that axis. Each curve is the mean over EVERY run in the sweep
    holding that axis value -- i.e. marginalized over the other axes -- so it
    reads as that axis's main effect rather than one arbitrary slice. With
    sweep 237's 4 axes that is 24 configs x 3 seeds = 24 curves behind a
    3-level axis's line and 36 behind a 2-level one.

``--mode pooled``
    One PNG for a single config, pooling ``--pool-tag`` seed sets from several
    sweeps into one line. Used for the config-identical sweep233/sweep237 pair,
    which gives 6 seeds instead of 3.

Both modes draw the DPMD (Haitong) diffusion-100 best-of-N, DIPO and SAC eval
curves from the loaders behind ``figures/sweep188_eval_only_curves.png``, and
assert the DIPO/SAC means match those loaders so these figures cannot silently
drift from the rest of the figure set.

Every line is "mean over seeds of that seed's eval-point mean, band = +/-1 SEM
over seeds", read from the ``eval_episode_returns.csv`` mirrors (10 best-of-32
episodes per slot every 50k steps), never the wandb API.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "scratch"))
from build_sweep_local_metrics import (  # noqa: E402
    _codebase_dir_from_metadata,
    _read_run_basename,
)
from compute_sweep_normalized_scores import (  # noqa: E402
    _fmt,
    _full_config,
    _resolve_run_dir,
    tag_axis_keys,
)
from compute_topsis import ENVS  # noqa: E402
from plot_best_config_curves import (  # noqa: E402
    DIPO_EVAL_STYLE,
    DIPO_RUNS_ROOT,
    SAC_V3_EVAL_STYLE,
    SAC_V3_GPU_ROOT,
    gradient_colors,
    load_cleanrl_sac_gpu_curves,
    load_dipo_runs,
)
from plot_sweep233_target_acc_eval_curves import (  # noqa: E402
    _aggregate,
    check_matches_reference,
    load_baseline_best_of_n,
    load_dipo_eval_per_seed,
    load_sac_eval_per_seed,
)

WANDB_BASE = Path("/n/holylabs/kdbrantley_lab/Lab/pnielsen/wandb")
BASELINE_CSV_DIR = SCRIPT_DIR.parent / "dpmd_baseline_diffusion100_plot_csv"
BASELINE_N = 32
MAX_STEP = 1_000_000
BASELINE_STYLE = dict(color="tab:purple", linestyle="--", linewidth=1.6,
                      marker="s", markersize=3.0)
OURS_STYLE = dict(color="black", linestyle="-", linewidth=2.1, marker="o",
                  markersize=3.5)
# The pooled config of the adaptive-oracle table's top rows, and the
# config-identical sweep233 run set the scoring script folds into its seed pool.
POOL_TAGS = (
    "sweep237_T=0_buffer_size=400000_eta=64_gamma=0.99"
    "_mcmc_proposal_type=euler_maruyama_rollout_alpha=1",
    "sweep233_mala_target_acceptance_rate=0.7",
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("axes", "pooled"), default="axes")
    ap.add_argument("--sweep-id", type=int, nargs="+", default=[237, 239])
    ap.add_argument("--pool-tag", nargs="+", default=list(POOL_TAGS),
                    help="--mode pooled: config_tags whose seeds form one line")
    ap.add_argument("--wandb-base", type=Path, default=WANDB_BASE)
    ap.add_argument("--baseline-csv-dir", type=Path, default=BASELINE_CSV_DIR)
    ap.add_argument("--baseline-n", type=int, default=BASELINE_N)
    ap.add_argument("--dipo-root", type=Path, default=DIPO_RUNS_ROOT)
    ap.add_argument("--sac-root", type=Path, default=SAC_V3_GPU_ROOT)
    ap.add_argument("--envs", nargs="+", default=list(ENVS))
    ap.add_argument("--max-step", type=int, default=MAX_STEP)
    ap.add_argument("--out-dir", type=Path, default=SCRIPT_DIR.parent / "figures")
    ap.add_argument("--out", type=Path, default=None,
                    help="--mode pooled: output path")
    ap.add_argument("--score-dir", type=Path,
                    default=SCRIPT_DIR / "topsis_out" / "sweep_237+239" / "normalized_eval",
                    help="compute_sweep_normalized_scores output to check against")
    ap.add_argument("--score-window", type=int, nargs=2, default=(800_000, 1_000_000))
    return ap.parse_args(argv)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_eval_per_seed(sweep_ids, wandb_base, max_step, envs):
    """``(per_seed, tag_cfg, tag_sweep)`` for every slot of every sweep.

    ``per_seed[tag][env][run_key]`` is a Series of step -> mean eval return.

    One pass over the job dirs: a job is one env and its 12 slots share a single
    ``eval_episode_returns.csv``, so the CSV is read ONCE and split by
    ``seed_index``, with each slot filed under its own ``config_tag``. Doing it
    per-tag instead (as ``plot_sweep233_target_acc_eval_curves.load_sweep_eval``
    does) re-reads every CSV once per tag, which for sweep 237's 24 tags is 24x
    the IO.
    """
    per_seed: dict = defaultdict(lambda: defaultdict(dict))
    tag_cfg, tag_sweep = {}, {}
    env_set = set(envs)
    for sid in sweep_ids:
        n_slots = 0
        for job_dir in sorted((Path(wandb_base) / f"sweep_{sid}").glob("job_*")):
            slot_cfg = {}
            base_name = codebase = env = None
            for rd in sorted((job_dir / "wandb").glob("offline-run-*")):
                cfg_path = rd / "files" / "config.yaml"
                if not cfg_path.exists():
                    continue
                info = _full_config(cfg_path)
                si = info.get("seed_index")
                if si is None:
                    continue
                slot_cfg[int(si)] = info
                env = env or info.get("env")
                if base_name is None:
                    wf = list(rd.glob("run-*.wandb"))
                    if wf:
                        base_name = _read_run_basename(wf[0])
                if codebase is None:
                    codebase = _codebase_dir_from_metadata(
                        rd / "files" / "wandb-metadata.json")
            if not slot_cfg or env is None or env not in env_set:
                continue
            if base_name is None or codebase is None:
                print(f"  WARNING: {job_dir.name}: no run name / codebase")
                continue
            run_dir = _resolve_run_dir(codebase, env, base_name)
            if run_dir is None:
                print(f"  WARNING: no log dir for {env} {base_name} under {codebase}")
                continue
            ev_path = Path(run_dir) / "eval_episode_returns.csv"
            if not ev_path.exists():
                print(f"  WARNING: no eval mirror at {ev_path}")
                continue
            ev = pd.read_csv(ev_path, on_bad_lines="skip")
            if not {"seed_index", "step", "episode_return"} <= set(ev.columns):
                continue
            ev = ev[ev["step"] <= max_step].dropna(subset=["episode_return"])
            for slot, info in slot_cfg.items():
                tag = info.get("config_tag")
                if tag is None:
                    continue
                tag_cfg.setdefault(tag, info)
                tag_sweep.setdefault(tag, sid)
                sub = ev[ev["seed_index"] == slot]
                if sub.empty:
                    continue
                per_seed[tag][env][f"{sid}/{job_dir.name}/slot{slot}"] = (
                    sub.groupby("step")["episode_return"].mean())
                n_slots += 1
        print(f"sweep {sid}: {n_slots} slots with eval curves")
    return per_seed, tag_cfg, tag_sweep


def sweep_axes(per_seed, tag_cfg, tag_sweep, sid):
    """The axes sweep ``sid`` actually varies, and its tags."""
    tags = [t for t in per_seed if tag_sweep.get(t) == sid]
    keys = sorted({k for t in tags for k in tag_axis_keys(t, tag_cfg[t])})
    axes = [k for k in keys if len({_fmt(tag_cfg[t].get(k)) for t in tags}) > 1]
    return tags, axes


def marginal_curves(per_seed, tag_cfg, tags, axis, envs):
    """``{level: {env: (steps, mean, sem, n_seeds)}}`` marginalizing other axes.

    Every seed of every config holding ``axis == level`` contributes one curve,
    so the level's line is that value's average over the rest of the sweep.
    """
    by_level = defaultdict(lambda: defaultdict(dict))
    for t in tags:
        level = _fmt(tag_cfg[t].get(axis))
        for env in envs:
            for run_key, series in per_seed[t].get(env, {}).items():
                by_level[level][env][f"{t}|{run_key}"] = series
    return {lv: {e: _aggregate(seeds) for e, seeds in envmap.items()}
            for lv, envmap in by_level.items()}


def pooled_curve(per_seed, tags, envs):
    """One line from the seeds of several config-identical tags."""
    out = defaultdict(dict)
    for t in tags:
        for env in envs:
            for run_key, series in per_seed.get(t, {}).get(env, {}).items():
                out[env][f"{t}|{run_key}"] = series
    return {e: _aggregate(seeds) for e, seeds in out.items()}


# --------------------------------------------------------------------------
# plotting
# --------------------------------------------------------------------------
def _draw_baselines(ax, env, baseline, dipo, sac, n, alpha=0.10):
    for curves, label, style in (
        (baseline, f"DPMD baseline diff=100 (best-of-{n})", BASELINE_STYLE),
        (dipo, "DIPO", DIPO_EVAL_STYLE),
        (sac, "SAC (best-of-32)", SAC_V3_EVAL_STYLE),
    ):
        c = (curves or {}).get(env)
        if c is None:
            continue
        steps, mean, sem, ns = c
        ax.plot(steps / 1e6, mean, zorder=3, label=f"{label}, n={ns}", **style)
        ax.fill_between(steps / 1e6, mean - sem, mean + sem,
                        color=style["color"], alpha=alpha, linewidth=0)


def _finish(fig, axes, args, suptitle, out):
    handles = labels = None
    for ax, env in zip(axes.ravel(), args.envs):
        ax.set_xlim(0, args.max_step / 1e6)
        ax.set_title(env, fontsize=11)
        ax.set_xlabel("env step (millions)")
        ax.set_ylabel("episode return")
        ax.grid(True, alpha=0.3)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(suptitle, fontsize=13)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_axis(levels, axis, sid, other_axes, baseline, dipo, sac, args, out):
    # Reds ramp: the baselines own purple (DPMD), blue (DIPO) and green (SAC).
    keys = sorted(levels, key=lambda s: (float(s) if _isnum(s) else float("inf"), s))
    colors = gradient_colors(len(keys), cmap_name="Reds", lo=0.62, hi=1.0)
    # A marker per level as well as a shade: three reds plus three baselines in
    # one panel is not separable by hue alone once the bands overlap.
    markers = ("o", "^", "s", "D", "v")
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
    for ax, env in zip(axs.ravel(), args.envs):
        for lv, color, mk in zip(keys, colors, markers):
            c = levels[lv].get(env)
            if c is None:
                continue
            steps, mean, sem, n = c
            ax.plot(steps / 1e6, mean, color=color, linestyle="-", linewidth=2.0,
                    marker=mk, markersize=3.2, markevery=2, zorder=4,
                    label=f"{axis}={lv} (n={n})")
            ax.fill_between(steps / 1e6, mean - sem, mean + sem, color=color,
                            alpha=0.12, linewidth=0)
        _draw_baselines(ax, env, baseline, dipo, sac, args.baseline_n)
    marg = ", ".join(other_axes) if other_axes else "nothing"
    _finish(fig, axs, args,
            f"Sweep {sid} evaluation curves by --{axis}, first "
            f"{args.max_step // 1000}k env steps\n"
            f"each line: every run of the sweep at that {axis}, marginalized over "
            f"{marg};  mean over seeds of that seed's eval-point mean, "
            "band = +/-1 SEM over seeds\n"
            f"vs the DPMD diffusion-100 baseline at best-of-{args.baseline_n} and "
            "the DIPO / SAC eval curves of figures/sweep188_eval_only_curves.png",
            out)


def plot_pooled(ours, tags, label, n_by_tag, baseline, dipo, sac, args, out):
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.6), constrained_layout=True)
    for ax, env in zip(axs.ravel(), args.envs):
        c = ours.get(env)
        if c is not None:
            steps, mean, sem, n = c
            ax.plot(steps / 1e6, mean, zorder=5,
                    label=f"ours: {_abbrev(label)} (n={n})", **OURS_STYLE)
            ax.fill_between(steps / 1e6, mean - sem, mean + sem,
                            color=OURS_STYLE["color"], alpha=0.15, linewidth=0)
        _draw_baselines(ax, env, baseline, dipo, sac, args.baseline_n, alpha=0.12)
    _finish(fig, axs, args,
            f"Evaluation curves, {label}, first {args.max_step // 1000}k env steps\n"
            + _seed_line(tags, n_by_tag) + "\n"
            f"vs the DPMD diffusion-100 baseline at best-of-{args.baseline_n} and "
            "the DIPO / SAC eval curves of figures/sweep188_eval_only_curves.png;  "
            "band = +/-1 SEM over seeds",
            out)


def _seed_line(tags, n_by_tag):
    """How the line's seeds were obtained, spelled out per contributing sweep."""
    parts = [f"{t.split('_', 1)[0]} ({n_by_tag[t]} seeds)" for t in tags]
    total = sum(n_by_tag[t] for t in tags)
    if len(tags) == 1:
        return f"{parts[0].replace(' (', ', ').rstrip(')')} per env"
    return (f"seeds pooled over {len(tags)} config-identical run sets: "
            + " + ".join(parts)
            + f" = {total} seeds per env")


def config_label(tag, tag_cfg, tag_sweep, per_seed):
    """``k=v`` for the axes the tag's OWN sweep varied.

    Not the whole tag: sweep 237 pins ``T=0`` and
    ``mcmc_proposal_type=euler_maruyama`` in every tag, so printing the tag
    verbatim buries the four values that actually identify the config.
    """
    _, axes = sweep_axes(per_seed, tag_cfg, tag_sweep, tag_sweep[tag])
    return " ".join(f"{k}={_fmt(tag_cfg[tag].get(k))}" for k in axes)


# The 6-seed pooled figure is already referenced by name, so it keeps that name.
REFERENCE_POOLED_OUT = "sweep233+237_buffer400k_eta64_pooled_eval_curves.png"


def pooled_out_path(tags, label, out_dir):
    """A name keyed by BOTH the pooled sweeps and the config.

    A fixed default would let a second ``--pool-tag`` silently overwrite the
    first config's figure, which is exactly the mistake that makes two
    same-looking PNGs impossible to tell apart later.
    """
    if tuple(tags) == POOL_TAGS:
        return out_dir / REFERENCE_POOLED_OUT
    ids = "+".join(sorted({t.split("_", 1)[0].removeprefix("sweep") for t in tags}))
    slug = _abbrev(label).replace("=", "").replace(" ", "_")
    return out_dir / f"sweep{ids}_{slug}_eval_curves.png"


_ABBREV = {"buffer_size": "buffer", "rollout_alpha": "r_alpha"}


def _abbrev(label):
    """Shorten a ``k=v`` label enough to sit in a 6-panel legend."""
    out = []
    for item in label.split():
        k, _, v = item.partition("=")
        if k == "buffer_size" and _isnum(v):
            f = float(v)
            v = f"{f / 1e6:g}M" if f >= 1e6 else f"{f / 1e3:g}k"
        out.append(f"{_ABBREV.get(k, k)}={v}")
    return " ".join(out)


def verify_against_table(ours, args):
    """The pooled curve's scoring-window mean must be the table's fraction.

    The figure and ``fractional_scores.csv`` are computed by different code off
    the same mirrors -- this one averages the eval points it plots, the scoring
    script averages per seed then over seeds -- so agreement to float noise is a
    real check that the figure pools exactly the runs the table scored.
    """
    fs_path = args.score_dir / "fractional_scores.csv"
    bl_path = args.score_dir / "baselines.csv"
    if not (fs_path.exists() and bl_path.exists()):
        print(f"  (no scored table at {args.score_dir}, skipping check)")
        return
    denom = pd.read_csv(bl_path).groupby("env")["value"].max()
    fs = pd.read_csv(fs_path).set_index("config_tag")
    lo, hi = args.score_window
    for tag in args.pool_tag:
        if tag in fs.index:
            row = fs.loc[tag]
            break
    else:
        print(f"  (none of {args.pool_tag} in {fs_path.name}, skipping check)")
        return
    worst = 0.0
    for env in args.envs:
        c = ours.get(env)
        if c is None or env not in denom.index or f"frac_{env}" not in row:
            continue
        steps, mean = c[0], c[1]
        w = (steps >= lo) & (steps <= hi)
        got = float(mean[w].mean()) / float(denom[env])
        worst = max(worst, abs(got - float(row[f"frac_{env}"])))
    assert worst < 1e-6, f"pooled curve disagrees with {fs_path.name} by {worst:.2e}"
    print(f"  window [{lo}, {hi}] fractions match {fs_path.name} "
          f"(max |diff| {worst:.1e}, n_seeds_min={row.get('n_seeds_min')})")


def _isnum(s):
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


def main(argv=None):
    args = parse_args(argv)
    baseline = load_baseline_best_of_n(args.baseline_csv_dir, args.baseline_n,
                                       args.max_step, args.envs)
    dipo = load_dipo_eval_per_seed(args.dipo_root, args.max_step, args.envs)
    sac = load_sac_eval_per_seed(args.sac_root, args.max_step, args.envs)
    # These must be the very lines the sweep188 figure drew.
    _, dipo_ref = load_dipo_runs(root=args.dipo_root)
    _, sac_ref = load_cleanrl_sac_gpu_curves(root=args.sac_root)
    check_matches_reference(dipo, dipo_ref, "DIPO", args.max_step)
    check_matches_reference(sac, sac_ref, "SAC", args.max_step)

    sweeps = args.sweep_id
    if args.mode == "pooled":
        sweeps = sorted({int(t.split("_", 1)[0].removeprefix("sweep"))
                         for t in args.pool_tag})
    per_seed, tag_cfg, tag_sweep = load_eval_per_seed(
        sweeps, args.wandb_base, args.max_step, args.envs)

    if args.mode == "pooled":
        missing = [t for t in args.pool_tag if t not in per_seed]
        if missing:
            raise SystemExit(f"no eval curves for {missing}")
        ours = pooled_curve(per_seed, args.pool_tag, args.envs)
        for env in args.envs:
            c = ours.get(env)
            print(f"  {env:16s} n_seeds={0 if c is None else c[3]}"
                  f" pts={0 if c is None else len(c[0])}")
        verify_against_table(ours, args)
        label = config_label(args.pool_tag[0], tag_cfg, tag_sweep, per_seed)
        n_by_tag = {t: min(len(per_seed[t].get(e, {})) for e in args.envs)
                    for t in args.pool_tag}
        print(f"  config: {label}")
        out = args.out or pooled_out_path(args.pool_tag, label, args.out_dir)
        plot_pooled(ours, args.pool_tag, label, n_by_tag, baseline, dipo, sac,
                    args, out)
        return

    for sid in sweeps:
        tags, axes = sweep_axes(per_seed, tag_cfg, tag_sweep, sid)
        print(f"sweep {sid}: {len(tags)} tags, axes {axes}")
        for axis in axes:
            levels = marginal_curves(per_seed, tag_cfg, tags, axis, args.envs)
            others = [a for a in axes if a != axis]
            n_by_env = {lv: {e: (c[3] if c else 0) for e, c in m.items()}
                        for lv, m in levels.items()}
            print(f"  --{axis}: levels " + ", ".join(
                f"{lv} (n={min(n_by_env[lv].values())}-{max(n_by_env[lv].values())})"
                for lv in sorted(levels)))
            out = args.out_dir / f"sweep{sid}_axis_{axis}_eval_curves.png"
            plot_axis(levels, axis, sid, others, baseline, dipo, sac, args, out)


if __name__ == "__main__":
    main()
