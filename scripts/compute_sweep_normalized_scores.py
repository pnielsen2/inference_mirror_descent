#!/usr/bin/env python3
"""Baseline-normalized evaluation score per sweep config, with a seed bootstrap.

The metric, for one config and one env:

  1. ``ours``    = mean over the eval points in [--window-start, --window-end]
                   of that point's mean episode return, averaged over the
                   config's seeds. At --eval-every 50k and a 800k-1M window that
                   is 5 eval points x 10 episodes per seed.
  2. ``denom``   = max over the three baselines (DPMD/Haitong diffusion-100 at
                   best-of-N, DIPO, SAC) of the SAME quantity, each averaged over
                   all of its seeds and all of its eval points in the window.
                   Each baseline keeps its own native eval cadence (DPMD and DIPO
                   are logged every 10k, SAC and ours every 50k), so this is
                   "average of whatever that source evaluated in the window".
  3. ``frac``    = ours / denom -- the fractional score for that (config, env).

and then, across the six envs, the single normalized score is

  * ``min(frac)``                         if every env's frac > 1, so a config
                                          that beats every baseline everywhere is
                                          ranked by its weakest margin; and
  * ``geomean(min(frac, 1))``             otherwise, which caps credit for
                                          beating a baseline and makes the score
                                          multiplicative in the shortfalls.

Unlike ``compute_topsis.log_score`` this only caps in the second branch, so a
config that clears every baseline is still ordered by how much it clears them by.

Uncertainty comes from resampling the seeds *within* each (config, env) cell with
replacement (--n-boot replicates, independently per cell, since no seed is shared
between configs). The baselines are held fixed -- they are the reference, not a
competitor being ranked -- so the reported ``boot_best_frac`` is the fraction of
replicates in which that config had the highest normalized score.

Only configs whose runs have ALL finished are scored: every slot that appears in
the sweep's offline config.yamls must have the full set of eval points in the
window. ``--watch`` recomputes and rewrites once no sweep job is left in squeue.

Locating a run's mirrors: ``_read_run_basename`` returns the wandb display name,
which for these sweeps carries a launch prefix (e.g. ``v3BTmgmd_2026-...``) that
the on-disk ``logs/<env>/`` directory does not have, so the path is resolved by
suffix match -- the same fix ``plot_sweep233_target_acc_eval_curves.py`` needed
and that ``build_sweep_local_metrics.py`` lacks.
"""
from __future__ import annotations

import argparse
import getpass
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

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
from compute_topsis import ENVS  # noqa: E402

# The DIPO / SAC roots come from the module that drew the reference figure
# (figures/sweep233_target_acc0.7_eval_curves.png), so the baselines here are the
# same files that figure's lines came from. It lives in scratch/, which is not
# importable in every checkout, hence the literal fallback.
sys.path.insert(0, str(SCRIPT_DIR.parent / "scratch"))
try:
    from plot_best_config_curves import DIPO_RUNS_ROOT, SAC_V3_GPU_ROOT  # noqa: E402
except ImportError:  # pragma: no cover
    DIPO_RUNS_ROOT = "/n/home09/pnielsen/DIPO/record"
    SAC_V3_GPU_ROOT = ("/n/netscratch/kdbrantley_lab/Lab/pnielsen/"
                       "sac_v3_gpu_baselines/20260819_170015")

SWEEP_ID = 237
WANDB_BASE = Path("/n/holylabs/kdbrantley_lab/Lab/pnielsen/wandb")
BASELINE_CSV_DIR = SCRIPT_DIR.parent / "dpmd_baseline_diffusion100_plot_csv"
BASELINE_N = 32
WINDOW = (800_000, 1_000_000)
N_BOOT = 20_000
POLL_INTERVAL = 600

# Extra runs to fold into a config's seed pool, as
# ``<source sweep>:<source config_tag>:<target config_tag>``.
#
# sweep233's mala_target_acceptance_rate=0.7 config is the same point in
# hyperparameter space as sweep237's buffer_size=400000 eta=64 gamma=0.99
# rollout_alpha=1: all 38 algorithmic keys agree, including the RESOLVED
# alpha=1.0 / beta=64.0 / T=0 / eta=64, buffer_size=400000, gamma=0.99 and
# mala_target_acceptance_rate=0.7. The two flags sweep233's codebase lacked sit
# at their no-op values in sweep237 (--rollout_alpha 1.0, whose
# ``rollout_tilt_rows`` is then 0 so every batch slice is the full batch and
# get_action skips the tilt; and --mcmc_proposal_type euler_maruyama, whose
# proposal variance 2h and MH density are algebraically what the old code had).
# Its 3 seeds per env are disjoint from sweep237's, so pooling gives 6.
# The remaining difference is --parallel_runs 12 vs 9, i.e. the vmap width, not
# an algorithmic knob.
DEFAULT_POOLS = (
    "233:sweep233_mala_target_acceptance_rate=0.7:"
    "sweep237_T=0_buffer_size=400000_eta=64_gamma=0.99"
    "_mcmc_proposal_type=euler_maruyama_rollout_alpha=1",
)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-id", type=int, nargs="+", default=[SWEEP_ID],
                    help="One or more sweeps to score together. config_tags are "
                         "sweep-prefixed so they cannot collide; each config is "
                         "labelled by the axes ITS OWN sweep varied.")
    ap.add_argument("--wandb-base", type=Path, default=WANDB_BASE)
    ap.add_argument("--window-start", type=int, default=WINDOW[0])
    ap.add_argument("--window-end", type=int, default=WINDOW[1])
    ap.add_argument("--baseline-csv-dir", type=Path, default=BASELINE_CSV_DIR)
    ap.add_argument("--baseline-n", type=int, default=BASELINE_N,
                    help="best-of-N column of the DPMD baseline CSVs")
    ap.add_argument("--dipo-root", type=Path, default=Path(DIPO_RUNS_ROOT))
    ap.add_argument("--sac-root", type=Path, default=Path(SAC_V3_GPU_ROOT))
    ap.add_argument("--envs", nargs="+", default=list(ENVS))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--pool", nargs="*", default=list(DEFAULT_POOLS),
                    metavar="SWEEP:SRC_TAG:DST_TAG",
                    help="Fold a config-identical run set from another sweep into "
                         "DST_TAG's seed pool. Refused if any seed collides with "
                         "DST_TAG's own within an env. Pass --pool with no value "
                         "to disable.")
    ap.add_argument("--include-incomplete", action="store_true",
                    help="Also score configs with unfinished runs (marked as such). "
                         "Off by default: partial seeds bias the cell means.")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--watch", action="store_true",
                    help="Compute now, then poll squeue and recompute once every "
                         "job of this sweep has left the queue.")
    ap.add_argument("--poll-interval", type=int, default=POLL_INTERVAL)
    ap.add_argument("--max-wait-hours", type=float, default=72.0)
    ap.add_argument("--no-adaptive", action="store_true",
                    help="Skip the per-env oracle (adaptive-parameter) table.")
    args = ap.parse_args(argv)
    args.sweep_id = list(dict.fromkeys(args.sweep_id))
    if args.out_dir is None:
        args.out_dir = (SCRIPT_DIR / "topsis_out"
                        / f"sweep_{'+'.join(str(s) for s in args.sweep_id)}"
                        / "normalized_eval")
    return args


# --------------------------------------------------------------------------
# our sweep
# --------------------------------------------------------------------------
def _full_config(path: Path) -> dict:
    """Every key of a wandb ``config.yaml``, unwrapped from its ``{value: ...}``.

    ``build_sweep_local_metrics._load_config_yaml`` returns only its ``HP_KEYS +
    EXTRA_KEYS`` whitelist, so a knob that is not on that list (``gamma``,
    ``buffer_size``, ``eval_every``) would silently read as ``None`` here. This
    script labels configs by whichever axes the sweep actually varied, so it
    needs the whole config rather than that fixed subset.
    """
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError):
        return {}
    return normalize_config(
        {k: _cfg_value(cfg, k) for k in cfg if not str(k).startswith("_")})


# The two flags sweep 233/237's codebase had were replaced by a single
# --use_target_networks in the codebase 239/240/241 ran.
_TARGET_NET_OLD = ("use_target_policy_training", "use_target_q_sampling_training")


def normalize_config(cfg):
    """Rewrite a pre-rename config into the current key, when it is safe to.

    Sweeps snapshot their codebase and record its commit, and between 237's
    (e8252c0) and 241's (19b06ad) the two flags in ``_TARGET_NET_OLD`` became one
    ``--use_target_networks``. With the old flags BOTH OFF the two codebases are
    bit-identical on that path -- every new call site goes through
    ``_tilt_params``, which returns its argument unchanged when the flag is off,
    and the one reordered statement (the target-Q Polyak update moving above the
    guidance normalizer) is not read by the flag-off branch, which normalizes on
    the online ensemble. So ``off/off`` and ``use_target_networks=False`` are the
    same algorithm and their runs pool.

    Either old flag being ON is left alone: the new flag also routes rollout and
    evaluation sampling through the target pair, which neither old flag did, so
    ON is not the same knob and must not be merged into it.
    """
    if not any(k in cfg for k in _TARGET_NET_OLD):
        return cfg
    if any(bool(cfg.get(k)) for k in _TARGET_NET_OLD):
        return cfg
    out = {k: v for k, v in cfg.items() if k not in _TARGET_NET_OLD}
    out["use_target_networks"] = False
    return out


def _resolve_run_dir(codebase: Path, env: str, base_name: str):
    """``codebase/logs/<env>/<exp_dir>`` for a wandb display name."""
    env_dir = Path(codebase) / "logs" / env
    direct = env_dir / base_name
    if direct.exists():
        return direct
    if not env_dir.is_dir():
        return None
    matches = [d for d in env_dir.iterdir() if d.is_dir() and base_name.endswith(d.name)]
    # Longest match: two runs of one sweep in an env dir differ only by their
    # timestamp/seed, so the longer suffix is the unambiguous one.
    return max(matches, key=lambda d: len(d.name)) if matches else None


def _window_points(df, lo, hi):
    """{step: mean episode return} for the eval points inside the window."""
    sub = df[(df["step"] >= lo) & (df["step"] <= hi)]
    if sub.empty:
        return {}
    return sub.groupby("step")["episode_return"].mean().to_dict()


def load_sweep(sweep_id, wandb_base, lo, hi, envs):
    """Per-seed window means for a sweep.

    Returns ``(values, launched, tag_cfg, tag_env_cfg, expected_pts, notes)``
    where ``values[(tag, env)][run_key]`` is ``(window mean, n eval points,
    seed)``, ``launched[(tag, env)]`` is every run key the sweep actually
    launched (from the offline config.yamls, independent of how far it got),
    ``tag_cfg[tag]`` is one representative flattened config for that tag,
    ``tag_env_cfg[(tag, env)]`` is the config as resolved IN THAT ENV -- some
    values are derived from the env (``mala_step_size_max_effective`` is
    ``mala_step_size_max_coef`` times an action-space quantity), so two configs
    are only comparable env by env -- and ``expected_pts`` is the eval-point
    count a finished slot must have.

    ``run_key`` is ``<sweep>/<job dir>/slot<n>``, unique across sweeps, so seed
    pools merged from two sweeps cannot silently overwrite each other on a
    shared slot index.
    """
    sweep_dir = Path(wandb_base) / f"sweep_{sweep_id}"
    if not sweep_dir.is_dir():
        raise SystemExit(f"no sweep dir at {sweep_dir}")
    values = defaultdict(dict)
    launched = defaultdict(set)
    tag_cfg, tag_env_cfg = {}, {}
    eval_everys, notes = set(), []
    env_set = set(envs)

    for job_dir in sorted(sweep_dir.glob("job_*")):
        run_dirs = sorted(job_dir.glob("wandb/offline-run-*"))
        if not run_dirs:
            continue  # empty requeue shell
        slot_cfg, base_name, codebase, env = {}, None, None, None
        for rd in run_dirs:
            cfg_path = rd / "files" / "config.yaml"
            if not cfg_path.exists():
                continue
            info = _full_config(cfg_path)
            si = info.get("seed_index")
            if si is None:
                continue
            slot_cfg[int(si)] = info
            env = env or info.get("env")
            if info.get("eval_every"):
                eval_everys.add(int(info["eval_every"]))
            if base_name is None:
                wf = list(rd.glob("run-*.wandb"))
                if wf:
                    base_name = _read_run_basename(wf[0])
            if codebase is None:
                codebase = _codebase_dir_from_metadata(rd / "files" / "wandb-metadata.json")
        if not slot_cfg or env is None or env not in env_set:
            continue
        def key_of(slot):
            return f"{sweep_id}/{job_dir.name}/slot{slot}"

        for slot, info in slot_cfg.items():
            launched[(info.get("config_tag"), env)].add(key_of(slot))
            tag_cfg.setdefault(info.get("config_tag"), info)
            tag_env_cfg.setdefault((info.get("config_tag"), env), info)

        run_dir = _resolve_run_dir(codebase, env, base_name) if (codebase and base_name) else None
        if run_dir is None:
            notes.append(f"no log dir for {env} {base_name} under {codebase}")
            continue
        ev_path = run_dir / "eval_episode_returns.csv"
        if not ev_path.exists():
            notes.append(f"no eval mirror at {ev_path}")
            continue
        ev = pd.read_csv(ev_path, on_bad_lines="skip")
        if not {"seed_index", "step", "episode_return"} <= set(ev.columns):
            notes.append(f"unexpected columns in {ev_path}")
            continue
        ev = ev.dropna(subset=["episode_return"])
        for slot, info in slot_cfg.items():
            pts = _window_points(ev[ev["seed_index"] == slot], lo, hi)
            if pts:
                values[(info.get("config_tag"), env)][key_of(slot)] = (
                    float(np.mean(list(pts.values()))), len(pts), info.get("seed"))

    if len(eval_everys) != 1:
        notes.append(f"mixed/absent --eval_every in sweep configs: {sorted(eval_everys)}")
    ee = min(eval_everys) if eval_everys else 50_000
    expected_pts = int(round((hi - lo) / ee)) + 1
    return values, launched, tag_cfg, tag_env_cfg, expected_pts, notes


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------
def baseline_dpmd(csv_dir, n, lo, hi, envs):
    """env -> (value, n_seeds). Pre-aggregated over seeds already."""
    out = {}
    for env in envs:
        path = Path(csv_dir) / f"{env.replace('-', '_')}_diffusion100.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        sel = df[(df["N"] == n) & (df["step"] >= lo) & (df["step"] <= hi)]
        if sel.empty:
            continue
        out[env] = (float(sel["mean_return"].mean()), int(sel["n_seeds"].max()),
                    int(len(sel)))
    return out


def _per_seed_mean(per_seed_points):
    if not per_seed_points:
        return None
    vals = [np.mean(list(p.values())) for p in per_seed_points.values() if p]
    if not vals:
        return None
    npts = int(np.min([len(p) for p in per_seed_points.values() if p]))
    return float(np.mean(vals)), len(vals), npts


def baseline_dipo(root, lo, hi, envs):
    out = {}
    for env in envs:
        per_seed = {}
        for seed_dir in sorted(Path(root).glob(f"{env}/policy_type=*/ratio=*/seed=*")):
            ev = seed_dir / "eval.csv"
            if not ev.exists():
                continue
            df = pd.read_csv(ev, on_bad_lines="skip")
            if not {"env_step", "mean_return"} <= set(df.columns):
                continue
            df = df[["env_step", "mean_return"]].apply(pd.to_numeric, errors="coerce").dropna()
            df = df[(df["env_step"] >= lo) & (df["env_step"] <= hi)]
            if not df.empty:
                per_seed[str(seed_dir)] = df.groupby("env_step")["mean_return"].mean().to_dict()
        got = _per_seed_mean(per_seed)
        if got:
            out[env] = got
    return out


def baseline_sac(root, lo, hi, envs):
    out = {}
    for env in envs:
        per_seed = {}
        for seed_dir in sorted(Path(root).glob(f"{env}/seed_*")):
            ev = seed_dir / "eval_episode_returns.csv"
            if not ev.exists():
                continue
            df = pd.read_csv(ev, on_bad_lines="skip")
            if not {"env_step", "episode_return"} <= set(df.columns):
                continue
            df = df[["env_step", "episode_return"]].apply(pd.to_numeric, errors="coerce").dropna()
            df = df[(df["env_step"] >= lo) & (df["env_step"] <= hi)]
            if not df.empty:
                per_seed[seed_dir.name] = df.groupby("env_step")["episode_return"].mean().to_dict()
        got = _per_seed_mean(per_seed)
        if got:
            out[env] = got
    return out


# Keys that identify a RUN rather than an algorithm, so two configs agreeing on
# everything else are the same point in hyperparameter space despite these.
# ``parallel_seeds``/``parallel_runs`` are the vmap width, ``eval_every`` is the
# observation cadence, the rest are bookkeeping.
_SIG_IGNORE = frozenset({
    "seed", "seed_index", "config_tag", "config_tag_keys", "sweep_id",
    "parallel_seeds", "parallel_runs", "env", "eval_every", "exp_name",
    "run_name", "job_name", "group", "notes", "tags", "id", "name",
    "output_dir", "log_dir", "save_dir", "wandb_id", "wandb_run_id",
    "wandb_project", "wandb_entity", "host", "hostname", "device",
})


def config_signature(cfg):
    """Hashable identity of a config, ignoring run bookkeeping."""
    return tuple(sorted((k, _fmt(v)) for k, v in cfg.items()
                        if k not in _SIG_IGNORE))


def find_duplicate_configs(tags, tag_env_cfg, envs):
    """Group tags that are the SAME algorithmic config into pooling sets.

    Two tags match only if they agree in EVERY env: comparing one
    representative config each would compare whichever env happened to be
    loaded first, and several values are env-derived. A key present in one and
    absent in the other counts as a difference, so a knob introduced between
    two sweeps blocks the merge rather than being silently assumed to be a
    no-op.

    Returns ``[[canonical_tag, other, ...], ...]`` for the groups of size > 1.
    """
    by_sig = defaultdict(list)
    for t in tags:
        per_env = [tag_env_cfg.get((t, e)) for e in envs]
        if any(c is None for c in per_env):
            continue
        by_sig[tuple(config_signature(c) for c in per_env)].append(t)
    return [sorted(g) for g in by_sig.values() if len(g) > 1]


def parse_pools(specs):
    """``['233:src:dst']`` -> ``[(233, 'src', 'dst')]``. Tags contain '=' and
    '_' but never ':', so a plain 3-way split is unambiguous."""
    out = []
    for spec in specs or ():
        parts = spec.split(":")
        if len(parts) != 3 or not parts[0].strip().isdigit():
            raise SystemExit(f"--pool expects SWEEP:SRC_TAG:DST_TAG, got {spec!r}")
        out.append((int(parts[0]), parts[1], parts[2]))
    return out


def build_denominators(args):
    """env -> (denom, winning baseline name); plus a tidy per-baseline frame."""
    lo, hi, envs = args.window_start, args.window_end, args.envs
    sources = {
        "DPMD": baseline_dpmd(args.baseline_csv_dir, args.baseline_n, lo, hi, envs),
        "DIPO": baseline_dipo(args.dipo_root, lo, hi, envs),
        "SAC": baseline_sac(args.sac_root, lo, hi, envs),
    }
    rows, denom = [], {}
    for env in envs:
        best_name, best_val = None, -np.inf
        for name, per_env in sources.items():
            got = per_env.get(env)
            if got is None:
                rows.append(dict(env=env, baseline=name, value=np.nan, n_seeds=0, n_points=0))
                continue
            val, nseeds, npts = got
            rows.append(dict(env=env, baseline=name, value=val, n_seeds=nseeds, n_points=npts))
            if val > best_val:
                best_name, best_val = name, val
        if best_name is None:
            raise SystemExit(f"no baseline data for {env}")
        if best_val <= 0:
            raise SystemExit(f"non-positive baseline denominator for {env}: {best_val}")
        denom[env] = (float(best_val), best_name)
    return denom, pd.DataFrame(rows)


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def normalized_score(fracs: np.ndarray, axis=-1) -> np.ndarray:
    """min(frac) where every env beats its baseline, else geomean(min(frac,1))."""
    capped = np.clip(np.minimum(fracs, 1.0), 0.0, None)
    geo = np.exp(np.mean(np.log(np.where(capped > 0, capped, 1e-12)), axis=axis))
    geo = np.where(np.any(capped <= 0, axis=axis), 0.0, geo)
    return np.where(np.all(fracs > 1.0, axis=axis), np.min(fracs, axis=axis), geo)


def bootstrap_best(seed_vals, denom, envs, n_boot, rng):
    """(scores [n_cfg, n_boot], best_frac [n_cfg]) resampling seeds per cell.

    ``seed_vals[cfg][env]`` is the 1-D array of that cell's per-seed values.
    Cells are resampled independently: no seed is shared across configs, so a
    common random draw would impose a correlation that does not exist.
    """
    cfgs = list(seed_vals)
    fracs = np.empty((len(cfgs), len(envs), n_boot), float)
    for ci, cfg in enumerate(cfgs):
        for ei, env in enumerate(envs):
            vals = np.asarray(seed_vals[cfg][env], float)
            idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
            fracs[ci, ei] = vals[idx].mean(axis=1) / denom[env][0]
    scores = normalized_score(fracs, axis=1)          # [n_cfg, n_boot]
    winners = np.argmax(scores, axis=0)
    best_frac = np.bincount(winners, minlength=len(cfgs)) / float(n_boot)
    return cfgs, scores, best_frac


def _fmt(v):
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def tag_axis_keys(tag, cfg):
    """The config keys a ``config_tag`` encodes.

    The tag is ``sweep<N>_<k>=<v>_<k>=<v>...`` where BOTH keys and values may
    contain underscores (``rollout_alpha=1``, ``mcmc_proposal_type=euler_maruyama``),
    so the tag cannot be tokenized unambiguously. Instead each real config key is
    tested against the tag, anchored at a ``_`` boundary so that e.g. ``eta`` does
    not match the ``eta=`` inside ``beta=``.

    That anchor alone is not enough when one key ends with another: ``_alpha=``
    is a substring of ``_rollout_alpha=``. Both candidates end at the same
    offset, so keep only the longest key per match offset.
    """
    by_end = {}
    for k in cfg:
        needle = f"_{k}="
        pos = tag.find(needle)
        if pos < 0:
            if not tag.startswith(f"{k}="):
                continue
            pos, needle = 0, f"{k}="
        end = pos + len(needle)
        if len(k) > len(by_end.get(end, "")):
            by_end[end] = k
    return sorted(by_end.values())


def axis_candidates(tag, cfg):
    """The axes a config's sweep varied.

    ``config_tag_keys`` records them explicitly, which beats re-deriving them
    from the tag string; ``tag_axis_keys`` stays as the fallback for configs
    written before that key existed.
    """
    declared = cfg.get("config_tag_keys")
    if isinstance(declared, str) and declared.strip():
        keys = [k.strip() for k in declared.split(",") if k.strip()]
        if all(k in cfg for k in keys):
            return keys
    return tag_axis_keys(tag, cfg)


def short_label(cfg, varying):
    """``k=v`` for just the axes that differ between the scored configs."""
    if not varying:
        return "(single config)"
    return " ".join(f"{k}={_fmt(cfg.get(k))}" for k in varying)


# --------------------------------------------------------------------------
# "which knob is worth making adaptive": per-env oracle over one axis
# --------------------------------------------------------------------------
def _groups_over(tags, tag_cfg, axis, others):
    """Partition ``tags`` into sets that differ ONLY in ``axis``.

    The key is the value of every other varying axis, so each group is a
    complete 1-D slice along ``axis`` and taking a per-env max inside it is a
    choice of ``axis`` alone -- not a choice of some other knob wearing this
    axis's name.
    """
    out = defaultdict(list)
    for t in tags:
        out[tuple(_fmt(tag_cfg[t].get(k)) for k in others)].append(t)
    return out


def _fold_means(vals, n_folds):
    """``(selection mean, held-out mean)`` per fold of a cell's seed values.

    Contiguous folds of the sorted-by-seed values: fold ``j`` is held out and
    the remaining seeds form the selection estimate. This is what makes the
    honest oracle honest -- the seeds that pick the axis value are not the
    seeds that score it, so a value that merely got lucky does not get to
    bank that luck.
    """
    vals = np.asarray(vals, float)
    idx = np.array_split(np.arange(vals.size), n_folds)
    out = []
    for hold in idx:
        mask = np.ones(vals.size, bool)
        mask[hold] = False
        if not mask.any() or hold.size == 0:
            return None
        out.append((vals[mask].mean(), vals[hold].mean()))
    return out


def adaptive_oracle(seed_vals, denom, envs, tags, tag_cfg, varying, n_boot, rng,
                    n_folds=3, extra_group_keys=(), tag_sweeps=None,
                    min_levels=2):
    """Score attainable if ONE axis may be chosen per environment.

    For each axis, every group of configs differing only in that axis yields an
    oracle whose per-env frac is the best of the axis's values. Three numbers
    per axis:

    * ``oracle``  -- in-sample per-env max. An upper bound only: the max of
      noisy cell means is biased upward (winner's curse), the more so the more
      values the axis has.
    * ``honest``  -- the axis value is chosen on ``n_folds - 1`` seeds' worth of
      each cell and scored on the held-out seeds, averaged over folds. Unbiased
      for the gain a real per-env choice would realize.
    * ``fixed``   -- best single config in the group, i.e. no adaptation.

    One row per (axis, group): the group IS a base config from the sweep with
    ``axis`` freed, so a row answers "take this config, make this one knob
    per-env, what do I get" directly. ``fixed`` needs no correction: nothing is
    selected, so its held-out and in-sample values agree.
    """
    rows = []
    for axis in varying:
        # Grouping on the other swept axes AND on the keys that differ between
        # sweeps outside any swept axis: 241 ran without target networks and at
        # other polyak taus, so a group free to mix it with 239/240 would
        # attribute a target-network effect to this axis.
        others = [k for k in varying if k != axis] + [
            k for k in extra_group_keys if k != axis]
        levels = sorted({_fmt(tag_cfg[t].get(axis)) for t in tags})
        if len(levels) < 2:
            continue
        groups = _groups_over(tags, tag_cfg, axis, others)
        # One candidate per VALUE of the axis. Two sweeps can each contribute a
        # config at the same value (237 and 241 both ran buffer_size=200000 and
        # were left unpooled because they share seeds), and a per-env max over
        # both would be picking the luckier of two runs of ONE config, which is
        # not an adaptive choice and inflates the oracle. Keep the better-seeded
        # one, earliest sweep breaking the tie.
        def rank(t):
            return (min(seed_vals[t][e].size for e in envs),
                    -min(tag_sweeps.get(t, {0})))
        deduped = {}
        for key, members in groups.items():
            best = {}
            for t in members:
                lv = _fmt(tag_cfg[t].get(axis))
                if lv not in best or rank(t) > rank(best[lv]):
                    best[lv] = t
            deduped[key] = [best[lv] for lv in sorted(best)]
        # Two levels are enough to choose between, so a group is kept when it
        # holds at least that many; ``n_levels_in_group`` records how many, since
        # a max over more candidates is biased up by more.
        groups = {k: v for k, v in deduped.items() if len(v) >= min_levels}
        if not groups:
            continue
        for key, members in groups.items():
            fr = np.array([[seed_vals[t][e].mean() / denom[e][0] for e in envs]
                           for t in members])                      # [level, env]
            oracle_fr = fr.max(axis=0)
            fixed_sc = normalized_score(fr, axis=1).max()
            oracle_sc = float(normalized_score(oracle_fr[None, :], axis=1)[0])

            folds = {(t, e): _fold_means(seed_vals[t][e], n_folds)
                     for t in members for e in envs}
            honest_sc = np.nan
            if all(f is not None for f in folds.values()):
                held = np.empty((n_folds, len(envs)))
                for j in range(n_folds):
                    for ei, e in enumerate(envs):
                        sel = [folds[(t, e)][j][0] for t in members]
                        held[j, ei] = folds[(members[int(np.argmax(sel))], e)][j][1]
                honest_fr = held.mean(axis=0) / np.array([denom[e][0] for e in envs])
                honest_sc = float(normalized_score(honest_fr[None, :], axis=1)[0])

            # Bootstrap: resample each cell's seeds, redo the per-env max.
            bs_or = np.empty(n_boot)
            bs_fx = np.empty(n_boot)
            bfr = np.empty((len(members), len(envs), n_boot))
            for mi, t in enumerate(members):
                for ei, e in enumerate(envs):
                    vals = np.asarray(seed_vals[t][e], float)
                    idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
                    bfr[mi, ei] = vals[idx].mean(axis=1) / denom[e][0]
            bs_or = normalized_score(bfr.max(axis=0).T, axis=1)
            bs_fx = normalized_score(bfr.transpose(2, 0, 1).reshape(-1, len(envs)),
                                     axis=1).reshape(n_boot, len(members)).max(axis=1)
            boot_gain = bs_or - bs_fx
            picks = {e: _fmt(tag_cfg[members[int(i)]].get(axis))
                     for e, i in zip(envs, fr.argmax(axis=0))}

            group_levels = [_fmt(tag_cfg[t].get(axis)) for t in members]
            sweeps = sorted({s for t in members
                             for s in (tag_sweeps or {}).get(t, ())})
            varying_others = [k for k in varying if k != axis]
            rows.append(dict(
                axis=axis, n_levels=len(levels), levels=",".join(levels),
                n_levels_in_group=len(members),
                group_levels=",".join(sorted(group_levels)),
                sweeps="+".join(str(s) for s in sweeps),
                base_config=" ".join(
                    f"{k}={v}" for k, v in zip(others, key)
                    if k in varying_others) or "(all)",
                regime=" ".join(f"{k}={v}" for k, v in zip(others, key)
                                if k not in varying_others) or "(one regime)",
                **{f"frac_{e}": float(v) for e, v in zip(envs, oracle_fr)},
                **{f"pick_{e}": picks[e] for e in envs},
                min_frac=float(oracle_fr.min()),
                fixed=float(fixed_sc), oracle=oracle_sc,
                gain=oracle_sc - float(fixed_sc),
                oracle_p05=float(np.percentile(bs_or, 5)),
                oracle_p95=float(np.percentile(bs_or, 95)),
                gain_p05=float(np.percentile(boot_gain, 5)),
                gain_p95=float(np.percentile(boot_gain, 95)),
                gain_frac_pos=float(np.mean(boot_gain > 0)),
                honest=honest_sc, honest_gain=honest_sc - float(fixed_sc),
                picks=", ".join(f"{e.replace('-v3', '')}:{v}" for e, v in picks.items()),
                n_distinct_picks=len(set(picks.values())),
            ))
    return pd.DataFrame(rows).sort_values("oracle", ascending=False)


def all_axes_oracle(seed_vals, denom, envs, tags, n_boot, rng):
    """Per-env max over EVERY config in the sweep: the ceiling for the table."""
    fr = np.array([[seed_vals[t][e].mean() / denom[e][0] for e in envs] for t in tags])
    sc = float(normalized_score(fr.max(axis=0)[None, :], axis=1)[0])
    bfr = np.empty((len(tags), len(envs), n_boot))
    for ti, t in enumerate(tags):
        for ei, e in enumerate(envs):
            vals = np.asarray(seed_vals[t][e], float)
            idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
            bfr[ti, ei] = vals[idx].mean(axis=1) / denom[e][0]
    bs = normalized_score(bfr.max(axis=0).T, axis=1)
    return sc, float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def render_adaptive(ad, ceilings, args):
    """One combined table: every (axis, base config), ranked by oracle.

    The base config is a real config of the sweep with ``axis`` freed, so a row
    reads as "take this config, make this one knob per-env" -- which is the
    decision being made, hence the ranking by the resulting score rather than
    by the gain.
    """
    lines = [
        f"# Making one parameter per-env adaptive "
        f"(sweeps {', '.join(str(s) for s in args.sweep_id)})",
        "",
        "Every row is an existing config with ONE knob freed to take a "
        "different value in each environment. Per env, that knob takes "
        "whichever of its values has the highest mean fractional score; "
        "`oracle` scores the resulting per-env vector.",
        "",
        "- `base config` = the other axes, held fixed. Every combination of "
        "them that ran at two or more values of the knob appears.",
        "- `fixed` = the best single config in that same slice, i.e. the same "
        "base config with the knob NOT adaptive. `gain` = `oracle - fixed`.",
        "- `lv` = how many values of the knob that slice actually ran. A max "
        "over more of them is biased up by more, so rows are only strictly "
        "comparable at equal `lv`.",
        "- `sw` = the sweeps the slice draws on. Slices never mix the target-"
        "network regimes: sweeps 239/240 share one, 241 has another, and the "
        "differing keys are part of the grouping.",
        "- Bootstrap CIs resample each cell's seeds and redo the per-env "
        "argmax, so the selection is carried through.",
        "- **In-sample**: the same seed means pick the per-env value and score "
        "it, which biases the max up. `adaptive_oracle.csv` also carries a "
        "`honest` column that re-picks on 2 of the 3 seeds and scores on the "
        "third -- a *lower* bound, a 2-seed selector being much noisier than "
        "these means. The truth is between the two.",
        "",
    ]
    if ad is None or ad.empty:
        return "\n".join(lines + ["No axis has two levels within one slice."])
    allrows = ad.sort_values("oracle", ascending=False)
    envs = [c[len("frac_"):] for c in allrows.columns if c.startswith("frac_")]
    lines += [
        "Each env cell is that env's oracle fraction of baseline, followed by "
        "the value of the knob that achieved it. `min` is the smallest of them, "
        "which is what caps the score.",
        "",
        "| # | sw | lv | adaptive knob | base config | "
        + " | ".join(e.replace("-v3", "") for e in envs)
        + " | min | fixed | oracle | oracle 5-95% | gain | gain 5-95% | P(gain>0) |",
        "|---|---|---|---|---|" + "|".join(["---"] * len(envs))
        + "|---|---|---|---|---|---|---|",
    ]
    for i, r in enumerate(allrows.to_dict("records"), 1):
        cells = [f"{r[f'frac_{e}']:.3f} @{r[f'pick_{e}']}" for e in envs]
        lines.append(
            f"| {i} | {r['sweeps']} | {r['n_levels_in_group']} | "
            f"**{r['axis']}** | {r['base_config']} | "
            + " | ".join(cells) + " | "
            f"{r['min_frac']:.3f} | "
            f"{r['fixed']:.4f} | **{r['oracle']:.4f}** | "
            f"{r['oracle_p05']:.4f}-{r['oracle_p95']:.4f} | "
            f"{r['gain']:+.4f} | "
            f"{r['gain_p05']:+.4f}..{r['gain_p95']:+.4f} | "
            f"{100 * r['gain_frac_pos']:.0f}% |")
    lines.append("")
    lines.append("Ceilings -- per-env pick over EVERY config of a regime, the most "
                 "any amount of per-env tuning within that regime could reach:")
    lines.append("")
    for name, (sc, p05, p95) in ceilings.items():
        lines.append(f"- {name}: **{sc:.4f}** (boot 5-95% {p05:.4f}-{p95:.4f})")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
def compute(args):
    lo, hi, envs = args.window_start, args.window_end, args.envs
    seed_lists = {}
    # Each sweep is loaded on its own -- expected_pts is derived from that
    # sweep's --eval_every -- and then merged. config_tags carry a sweep<N>_
    # prefix, so the merged dicts cannot collide even where two sweeps share an
    # axis (both 237 and 239 vary rollout_alpha).
    values, launched, tag_cfg, tag_sweep = {}, {}, {}, {}
    tag_env_cfg, expected_by_sweep = {}, {}
    for sid in args.sweep_id:
        v, l, tc, tec, exp, notes = load_sweep(sid, args.wandb_base, lo, hi, envs)
        for n in notes:
            print(f"  WARNING (sweep {sid}): {n}")
        print(f"sweep {sid}: eval points required per finished slot "
              f"in [{lo}, {hi}]: {exp}")
        expected_by_sweep[sid] = exp
        for key, rec in v.items():
            values.setdefault(key, {}).update(rec)
        for key, runs in l.items():
            launched.setdefault(key, set()).update(runs)
        for t, cfg in tc.items():
            tag_cfg.setdefault(t, cfg)
            tag_sweep.setdefault(t, sid)
        tag_env_cfg.update(tec)
    expected_pts = max(expected_by_sweep.values()) if expected_by_sweep else 0

    def exp_of(run_key):
        """Eval points a finished run needs, from ITS OWN sweep's cadence.

        A pooled config can hold runs from two sweeps, and 241 evaluates every
        25k against 237/239/240's 50k, so a single global count would mark the
        coarser sweep's finished runs as short.
        """
        head = str(run_key).split("/", 1)[0]
        return expected_by_sweep.get(int(head), expected_pts) if head.isdigit() \
            else expected_pts

    def is_complete(tag):
        """Every run the tag launched, in every env, has the full window."""
        for env in envs:
            want = launched.get((tag, env), set())
            have = values.get((tag, env), {})
            if not want:
                return False
            if any(k not in have or have[k][1] < exp_of(k) for k in want):
                return False
        return True

    # Sweeps overlap: 239's lr_policy=0.0003 lr_q=0.0003 rollout_alpha=1 is the
    # same algorithm as 240's lr_policy=0.0003 lr_q=0.0003
    # policy_noise_samples_per_target=1, each sweep's varied axis sitting at the
    # other's default. Merge such tags so the config appears once, on the union
    # of its seeds, instead of twice on half the seeds each.
    dup_groups = find_duplicate_configs(sorted({t for t, _ in launched}),
                                        tag_env_cfg, envs)
    tag_sweeps = {t: {sid} for t, sid in tag_sweep.items()}
    merged_from, blocked_merges = {}, []

    def seeds_of(tag, env):
        return {rec[2] for rec in values.get((tag, env), {}).values()}

    for group in dup_groups:
        canon = min(group, key=lambda t: (tag_sweep.get(t, 0), t))
        for other in group:
            if other == canon:
                continue
            # Same config is not enough: the seeds have to be DIFFERENT runs.
            # Two sweeps number seeds by grid position, so they can re-use the
            # same seed for the same config, and a seed fixes the network init
            # and the env reset stream -- pooling those would average one draw
            # with a rounding-perturbed copy of itself and call it two.
            clash = {e: sorted(seeds_of(canon, e) & seeds_of(other, e))
                     for e in envs if seeds_of(canon, e) & seeds_of(other, e)}
            if clash:
                blocked_merges.append((canon, other, clash))
                continue
            # Both sides must be finished. Folding a half-finished run set into a
            # complete one would make the union incomplete and drop a config that
            # was ready, and averaging the partial half would bias the cell.
            unfinished = [t for t in (canon, other) if not is_complete(t)]
            if unfinished:
                blocked_merges.append(
                    (canon, other, {"unfinished": unfinished}))
                continue
            for env in envs:
                if (other, env) in values:
                    values.setdefault((canon, env), {}).update(values.pop((other, env)))
                if (other, env) in launched:
                    launched.setdefault((canon, env), set()).update(
                        launched.pop((other, env)))
            tag_sweeps[canon] |= tag_sweeps.pop(other, set())
            merged_from.setdefault(canon, []).append(other)
            tag_cfg.pop(other, None)
        if merged_from.get(canon):
            print(f"identical config, pooling seeds:\n   {canon}"
                  + "".join(f"\n   + {o}" for o in merged_from[canon]))
    for canon, other, why in blocked_merges:
        reason = ("one side is UNFINISHED" if "unfinished" in why
                  else "SHARED seeds")
        print(f"identical config but {reason}, left separate:\n   {canon}\n"
              f"   vs {other}\n"
              + "".join(f"      {e}: {v}\n" for e, v in why.items()))

    # Explicit pools, for config-identical run sets that find_duplicate_configs
    # cannot see: sweep233's codebase predates two flags, so its config lacks
    # keys 237's has and the signatures differ even though the algorithm agrees.
    # A spec whose target is not in this table is simply not applicable.
    all_tags = {t for t, _ in launched}
    pools = [p for p in parse_pools(args.pool) if p[2] in all_tags]
    pooled_into = defaultdict(list)
    for src_sweep, src_tag, dst_tag in pools:
        if src_sweep in args.sweep_id:
            sv, sl, scfg = values, launched, tag_cfg
            sexp = expected_by_sweep.get(src_sweep, expected_pts)
        else:
            sv, sl, scfg, stec, sexp, snotes = load_sweep(
                src_sweep, args.wandb_base, lo, hi, envs)
            for n in snotes:
                print(f"  WARNING (pool sweep {src_sweep}): {n}")
        if not any(t == src_tag for t, _ in sl):
            raise SystemExit(f"--pool source {src_tag!r} not found in sweep {src_sweep}")
        for env in envs:
            want = sl.get((src_tag, env), set())
            have = sv.get((src_tag, env), {})
            full = {k: rec for k, rec in have.items() if rec[1] >= sexp}
            if not want or set(full) < want:
                raise SystemExit(
                    f"--pool source {src_tag} is not finished in {env} "
                    f"({len(full)}/{len(want)} runs have the full window); "
                    "pooling a partial run set would bias the cell mean")
            for k, rec in full.items():
                pooled_into[(dst_tag, env)].append((k, rec))
        print(f"pooling sweep {src_sweep} {src_tag}\n   -> {dst_tag}")

    tags = sorted({t for t, _ in launched})

    # A config is finished when every launched run, in every env, has the full
    # window. Anything short of that is a partial mean over seeds/steps.
    status, seed_vals, per_seed_rows, pool_note = {}, {}, [], {}
    for tag in tags:
        ok = True
        cell, cell_seeds = {}, {}
        for env in envs:
            want = launched.get((tag, env), set())
            have = dict(values.get((tag, env), {}))
            full = {k: rec for k, rec in have.items() if rec[1] >= exp_of(k)}
            if not want or set(full) < want:
                ok = False
            extra = pooled_into.get((tag, env), [])
            if extra and ok:
                # Seeds must be disjoint or the "6 seeds" would double-count a
                # run: identical config + identical seed = the same trajectory.
                own = {rec[2] for rec in full.values()}
                clash = own & {rec[2] for _, rec in extra}
                if clash:
                    raise SystemExit(
                        f"refusing to pool into {tag} / {env}: seeds {sorted(clash)} "
                        "appear in both run sets, so they are the same runs")
                full = {**full, **dict(extra)}
            for k, rec in sorted(have.items()):
                per_seed_rows.append(dict(config_tag=tag, env=env, run=k, seed=rec[2],
                                          value=rec[0], n_eval_points=rec[1],
                                          complete=bool(rec[1] >= exp_of(k)),
                                          pooled=False))
            for k, rec in extra:
                per_seed_rows.append(dict(config_tag=tag, env=env, run=k, seed=rec[2],
                                          value=rec[0], n_eval_points=rec[1],
                                          complete=True, pooled=True))
            cell[env] = np.array([full[k][0] for k in sorted(full)], float)
            cell_seeds[env] = sorted(str(full[k][2]) for k in full)
        status[tag] = ok
        if (pooled_into.get((tag, envs[0])) or tag in merged_from) and ok:
            pool_note[tag] = True
            for env in envs:
                seen = cell_seeds[env]
                if len(set(seen)) != len(seen):
                    raise SystemExit(
                        f"refusing to pool {tag} / {env}: seed(s) appear twice "
                        f"({seen}), so the same run would be counted twice")
        if (ok or args.include_incomplete) and all(cell[e].size for e in envs):
            seed_vals[tag] = cell
            for env in envs:
                seed_lists.setdefault(tag, {})[env] = cell_seeds[env]

    n_done = sum(status.values())
    print(f"configs: {len(tags)} total, {n_done} finished all runs, "
          f"{len(seed_vals)} scored")
    if not seed_vals:
        raise SystemExit("no config has a complete set of runs yet")
    for tag in sorted(seed_vals):
        ns = {e: seed_vals[tag][e].size for e in envs}
        if len(set(ns.values())) > 1 or pool_note.get(tag):
            print(f"  seeds per env for {tag}: {ns}"
                  + ("  (pooled)" if pool_note.get(tag) else ""))

    denom, base_df = build_denominators(args)
    print("\nbaseline window means (max = denominator):")
    for env in envs:
        vals = {r.baseline: r.value for r in base_df[base_df.env == env].itertuples()}
        pretty = "  ".join(f"{k}={v:8.1f}" for k, v in vals.items() if pd.notna(v))
        print(f"  {env:16s} {pretty}   -> denom {denom[env][0]:8.1f} ({denom[env][1]})")

    # Label every config by the axes varying across the WHOLE table, not by its
    # own sweep's axes: with several sweeps in one ranking, a label that omits
    # an axis another sweep moved would make two different configs print
    # identically. An axis one sweep pins and another varies still appears,
    # showing the pinned value on the rows that held it fixed.
    varying_by_sweep = {}
    for sid in args.sweep_id:
        mine = [t for t in seed_vals if sid in tag_sweeps.get(t, set())]
        axis_keys = sorted({k for t in mine for k in axis_candidates(t, tag_cfg[t])})
        varying_by_sweep[sid] = [k for k in axis_keys
                                 if len({_fmt(tag_cfg[t].get(k)) for t in mine}) > 1]
        print(f"sweep {sid}: axes in tags {axis_keys}")
        print(f"sweep {sid}: axes varying among its scored configs "
              f"{varying_by_sweep[sid]}")
    candidates = sorted({k for t in seed_vals for k in axis_candidates(t, tag_cfg[t])})
    varying = [k for k in candidates
               if len({_fmt(tag_cfg[t].get(k)) for t in seed_vals}) > 1]
    print(f"axes varying across the whole table: {varying}")
    print(f"axes in some tag but constant across the table (dropped from labels): "
          f"{[k for k in candidates if k not in varying]}")

    rng = np.random.default_rng(args.boot_seed)
    cfgs, boot_scores, best_frac = bootstrap_best(
        seed_vals, denom, envs, args.n_boot, rng)

    rows = []
    for ci, tag in enumerate(cfgs):
        fr = np.array([seed_vals[tag][e].mean() / denom[e][0] for e in envs])
        row = dict(config_tag=tag,
                   sweep="+".join(str(s) for s in sorted(tag_sweeps.get(tag, ()))),
                   label=short_label(tag_cfg[tag], varying),
                   complete=status[tag])
        for env, f in zip(envs, fr):
            row[f"frac_{env}"] = f
        row.update(
            pooled=bool(pool_note.get(tag, False)),
            min_frac=float(fr.min()),
            all_above_1=bool(np.all(fr > 1.0)),
            normalized_score=float(normalized_score(fr[None, :], axis=1)[0]),
            boot_best_frac=float(best_frac[ci]),
            boot_score_p05=float(np.percentile(boot_scores[ci], 5)),
            boot_score_p50=float(np.percentile(boot_scores[ci], 50)),
            boot_score_p95=float(np.percentile(boot_scores[ci], 95)),
            n_seeds_min=int(min(seed_vals[tag][e].size for e in envs)),
        )
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("normalized_score", ascending=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_dir / "fractional_scores.csv", index=False)
    base_df.to_csv(args.out_dir / "baselines.csv", index=False)
    pd.DataFrame(per_seed_rows).to_csv(args.out_dir / "per_seed_values.csv", index=False)
    pd.DataFrame([dict(config_tag=t, complete=status[t]) for t in tags]).to_csv(
        args.out_dir / "config_status.csv", index=False)
    pd.DataFrame([dict(config_tag=t, env=e, n_seeds=len(s), seeds=",".join(s))
                  for t, per_env in sorted(seed_lists.items())
                  for e, s in per_env.items()]).to_csv(
        args.out_dir / "seed_manifest.csv", index=False)

    cross = cross_sweep_diffs(tag_cfg, tag_sweeps, args.sweep_id, set(varying),
                              tag_env_cfg, envs[0])
    for k, vals in cross:
        print(f"  cross-sweep difference outside any swept axis: {k} = "
              + ", ".join(f"sweep {s}:{v}" for s, v in vals.items()))
    missing = [("+".join(str(s) for s in sorted(tag_sweeps.get(t, ()))),
                short_label(tag_cfg[t], varying))
               for t in tags if not status[t] and t in tag_cfg]
    scored_levels = {k: {_fmt(tag_cfg[t].get(k)) for t in seed_vals} for k in varying}
    missing_levels = []
    for k in varying:
        gone = sorted({_fmt(tag_cfg[t].get(k)) for t in tags
                       if not status[t] and t in tag_cfg} - scored_levels[k])
        if gone:
            missing_levels.append((k, gone))
    cadence = {s: (hi - lo) // max(exp - 1, 1)
               for s, exp in sorted(expected_by_sweep.items())}
    table = render(df, envs, args, n_done, len(tags), denom, cross,
                   sorted(missing), missing_levels, cadence, blocked_merges)
    (args.out_dir / "table.md").write_text(table)
    print("\n" + table)

    # Second table, over the configs that ARE finished. An unfinished config is
    # absent from seed_vals, so a slice missing a value of the knob simply has
    # one fewer level to choose from rather than mixing a complete level with a
    # partial one.
    if not args.no_adaptive:
        confounds = [k for k, _ in cross]
        ad = adaptive_oracle(seed_vals, denom, envs, sorted(seed_vals), tag_cfg,
                             varying, args.n_boot,
                             np.random.default_rng(args.boot_seed + 1),
                             extra_group_keys=confounds, tag_sweeps=tag_sweeps)
        # One ceiling per regime: a per-env pick spanning the target-network
        # change would not be a config anyone could run.
        regimes = defaultdict(list)
        for t in sorted(seed_vals):
            regimes[tuple(_fmt(tag_cfg[t].get(k)) for k in confounds)].append(t)
        ceilings = {}
        for i, (key, members) in enumerate(sorted(regimes.items())):
            # Same rule as the per-axis groups: one candidate per distinct
            # config, so an unpooled twin cannot give the per-env max a second
            # draw at the same setting.
            best = {}
            for t in members:
                sig = config_signature(tag_cfg[t])
                if sig not in best or (
                        min(seed_vals[t][e].size for e in envs),
                        -min(tag_sweeps.get(t, {0}))) > (
                        min(seed_vals[best[sig]][e].size for e in envs),
                        -min(tag_sweeps.get(best[sig], {0}))):
                    best[sig] = t
            members = sorted(best.values())
            sw = sorted({s for t in members for s in tag_sweeps.get(t, ())})
            name = ("sweep" + ("s " if len(sw) > 1 else " ")
                    + "+".join(str(s) for s in sw)
                    + f", {len(members)} configs")
            ceilings[name] = all_axes_oracle(
                seed_vals, denom, envs, members, args.n_boot,
                np.random.default_rng(args.boot_seed + 2 + i))
        ad.to_csv(args.out_dir / "adaptive_oracle.csv", index=False)
        atable = render_adaptive(ad, ceilings, args)
        (args.out_dir / "adaptive_oracle.md").write_text(atable)
        print("\n" + atable)
    print(f"wrote {args.out_dir}/{{fractional_scores,baselines,per_seed_values,"
          f"config_status,seed_manifest}}.csv and table.md")
    return df


# ``parallel`` covers parallel_runs/parallel_seeds: how many runs are vmapped
# together is a packing decision, not a knob any result depends on, and sweeps
# routinely differ on it (239:12, 240:18, 241:9).
# ``eval_every`` is how often the window is SAMPLED, not how the run trains: a
# 25k-cadence seed and a 50k-cadence one both estimate the mean return over the
# same window, the finer one from more points. Grouping on it would split sweeps
# that share an algorithm, so it is reported separately instead.
_CROSS_IGNORE = ("seed", "config_tag", "sweep_id", "job", "dir", "path", "name",
                 "time", "host", "device", "gpu", "wandb", "note", "tag", "id",
                 "parallel", "eval_every")


def cross_sweep_diffs(tag_cfg, tag_sweeps, sweeps, varying, tag_env_cfg, ref_env):
    """Keys on which two combined sweeps disagree but NEITHER sweep varied.

    Merging sweeps puts configs in one ranking, which is only a like-for-like
    comparison on the axes the sweeps swept. Anything else that changed between
    them (241 ran without --use_target_networks, at a different q_polyak_tau and
    target-Q delay) is silently confounded with the sweep, so it is listed.

    Every representative is read in the SAME env: several values are derived
    from the env, so representatives from different envs would differ on keys no
    sweep chose.
    """
    reps = {}
    for sid in sweeps:
        mine = [t for t in tag_cfg if sid in tag_sweeps.get(t, set())]
        # A pooled config belongs to several sweeps and carries the config of
        # whichever one won canonicalization, so it cannot represent the others:
        # prefer a tag that is this sweep's alone.
        exclusive = [t for t in mine if tag_sweeps.get(t) == {sid}]
        pick = sorted(exclusive or mine)
        if pick:
            reps[sid] = tag_env_cfg.get((pick[0], ref_env)) or tag_cfg[pick[0]]
    if len(reps) < 2:
        return []
    # Union, not intersection: a knob introduced between the two sweeps (237
    # predates --use_target_networks) is absent from the older config entirely,
    # and that is the biggest confound of all, not one to be skipped.
    out = []
    for k in sorted(set().union(*(set(c) for c in reps.values()))):
        if k in varying or any(w in k.lower() for w in _CROSS_IGNORE):
            continue
        vals = {sid: (_fmt(c[k]) if k in c else "(absent)") for sid, c in reps.items()}
        if len(set(vals.values())) > 1:
            out.append((k, vals))
    return out


def render(df, envs, args, n_done, n_tags, denom, cross=(), missing=(),
           missing_levels=(), cadence=None, blocked=()):
    short = {e: e.replace("-v3", "") for e in envs}
    head = (["sweep", "config"] + [short[e] for e in envs]
            + ["min", "score", "boot 5-95%", "best%", "seeds"])
    lines = [
        f"# Sweep {'+'.join(str(s) for s in args.sweep_id)} "
        "baseline-normalized eval scores",
        "",
        f"window {args.window_start//1000}k-{args.window_end//1000}k env steps; "
        f"{n_done}/{n_tags} configs finished all runs; "
        f"{args.n_boot} seed bootstrap replicates.",
        "",
        "denominator per env = max(DPMD, DIPO, SAC): "
        + ", ".join(f"{short[e]} {denom[e][0]:.0f} ({denom[e][1]})" for e in envs),
        "",
        "| " + " | ".join(head) + " |",
        "|" + "|".join(["---"] * len(head)) + "|",
    ]
    # dict records, not itertuples: env column names contain '-', which
    # itertuples renames to positional _N attributes.
    for r in df.to_dict("records"):
        cells = [str(r.get("sweep", "")),
                 r["label"] + ("" if r["complete"] else " (PARTIAL)")]
        for e in envs:
            cells.append(f"{r[f'frac_{e}']:.3f}")
        cells += [
            f"{r['min_frac']:.3f}",
            f"**{r['normalized_score']:.4f}**" + ("*" if r["all_above_1"] else ""),
            f"{r['boot_score_p05']:.3f}-{r['boot_score_p95']:.3f}",
            f"{100 * r['boot_best_frac']:.1f}",
            f"{r['n_seeds_min']}" + (" (pooled)" if r.get("pooled") else ""),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    if cadence and len(set(cadence.values())) > 1:
        lines += [
            "Eval cadence differs between these sweeps ("
            + ", ".join(f"sweep {s}: every {ee // 1000}k" for s, ee in cadence.items())
            + "), so a seed's window mean averages "
            + " or ".join(str(1 + (args.window_end - args.window_start) // ee)
                          for ee in sorted(set(cadence.values()), reverse=True))
            + " eval points depending on its sweep. Both estimate the same "
            "window; the finer one is less noisy. Pooled cells can mix the two.",
            "",
        ]
    if blocked:
        lines += [
            f"**{len(blocked)} pair(s) of configs are algorithmically identical "
            "but were NOT pooled**, so the same config appears on two rows, one "
            "per sweep:",
            "",
        ]
        for canon, other, why in blocked:
            if "unfinished" in why:
                reason = "one side is unfinished, and pooling a partial run set " \
                         "would both bias the cell and un-finish the complete side"
            else:
                reason = ("they share seeds ("
                          + "; ".join(f"{e}: {v}" for e, v in why.items())
                          + "), and a seed fixes the init and env stream, so "
                          "these are the same draw twice, not two draws")
            lines.append(f"- `{canon}` vs `{other}`: {reason}.")
        lines.append("")
    if missing:
        lines += [
            f"**{len(missing)} config(s) are not in this table**: their runs are "
            "unfinished in the window, and a mean over part of a seed set is "
            "not comparable to a mean over all of it.",
            "",
        ]
        if missing_levels:
            lines.append("Levels that therefore appear NOWHERE above, so no "
                         "comparison below can see them:")
            lines.append("")
            for k, vals in missing_levels:
                lines.append(f"- `{k}`: {', '.join(vals)}")
            lines.append("")
        lines.append("<details><summary>the unfinished configs</summary>")
        lines.append("")
        for sw, lab in missing:
            lines.append(f"- {sw}: {lab}")
        lines += ["", "</details>", ""]
    if cross:
        lines += [
            "**The `sweep` column is itself a confound.** These sweeps also "
            "differ on knobs neither of them swept, so a row's rank mixes its "
            "own axes with these:",
            "",
        ]
        for k, vals in cross:
            lines.append("- `" + k + "`: "
                         + ", ".join(f"sweep {s} = {v}" for s, v in vals.items()))
        lines.append("")
    lines.append("`score` = min(frac) when every env's frac > 1 (marked `*`), else "
                 "geomean(min(frac, 1)).")
    lines.append("`best%` = share of bootstrap replicates in which that config had the "
                 "highest score.")
    lines.append("`seeds` marked `(pooled)` merge a config-identical run set from another "
                 "sweep; see `seed_manifest.csv` for the exact seeds behind every cell.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
def sweep_job_ids(sweep_id, wandb_base):
    d = Path(wandb_base) / f"sweep_{sweep_id}"
    ids = set()
    for job in d.glob("job_*"):
        tail = job.name.split("_", 1)[1]
        if tail.isdigit():
            ids.add(tail)
    return ids


def queued_jobs(ids):
    """Subset of ``ids`` still in squeue. Filters by user to dodge the
    'Invalid job id' error squeue raises for ids it has already purged."""
    try:
        out = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-o", "%i"],
                             capture_output=True, text=True, timeout=120)
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"  WARNING: squeue failed ({exc}); assuming nothing queued")
        return set()
    live = set()
    for line in out.stdout.split():
        live.add(line.split("_")[0].strip())
    return {i for i in ids if i in live}


def main(argv=None):
    args = parse_args(argv)
    compute(args)
    if not args.watch:
        return
    ids = set()
    for s in args.sweep_id:
        ids |= sweep_job_ids(s, args.wandb_base)
    print(f"\n--watch: tracking {len(ids)} job ids of sweep(s) "
          f"{', '.join(str(s) for s in args.sweep_id)}")
    deadline = time.time() + args.max_wait_hours * 3600
    while time.time() < deadline:
        still = queued_jobs(ids)
        if not still:
            print(f"[{time.strftime('%H:%M:%S')}] all jobs left the queue; "
                  "recomputing final scores")
            compute(args)
            return
        print(f"[{time.strftime('%H:%M:%S')}] {len(still)} job(s) still queued/running; "
              f"sleeping {args.poll_interval}s", flush=True)
        time.sleep(args.poll_interval)
    print("--watch: hit --max-wait-hours without the sweep finishing")


if __name__ == "__main__":
    main()
