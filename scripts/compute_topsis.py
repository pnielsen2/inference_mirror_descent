#!/usr/bin/env python3
"""TOPSIS + joint quantile score across a sweep, using the LSAC Table 1 metric.

Pulls runs from wandb by ``config.sweep_id == N``. Per-slot hyperparameters and
``config_tag`` are read directly from each run's wandb config. No local file
dependencies (pack files, launch dirs). Every config that produced data for
the ranked envs is ranked -- no admissibility filtering.

Metric per run = mean of all episode returns whose env-step falls within the
last ``TAIL_WINDOW_STEPS`` (50k) of that run. The tail is anchored on the
run's latest logged episode, so in-progress runs automatically use their
current tail without any branch on run state.

Outputs under scripts/topsis_out/:
  topsis_ranking.csv       one row per config with per-env means, topsis_score,
                           quantile_score, and config_tag for pasting into
                           wandb's filter.
  per_cell_metrics.csv     one row per (env, config, seed) cell.
  per_config_env_metrics.csv
  quantile_reward_per_config_env.csv
  per_env_stats_top5_by_{topsis,quantile}.csv
  hp_level_summary.csv     per-(hp, value) avg_log_score + win fractions.
"""

from __future__ import annotations

import argparse
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from scipy import stats

ENTITY = "pnielsen2-harvard"
PROJECT = "diffusion_online_rl"
ENVS = ["HalfCheetah-v3", "Ant-v3", "Walker2d-v3", "Humanoid-v3"]
# Metric per run: mean of episode returns logged in the last
# TAIL_WINDOW_STEPS env steps. Anchor is the max _step in the run's
# episode-return stream (wandb's _step for "episode_return/{env}" is env_step,
# set explicitly by VmapOffPolicyTrainer.add_scalar_per_seed). Runs that are
# still in progress automatically use their current tail -- no branching on
# run state required, just "last 50k relative to whatever has been logged so
# far." 50k on a 1M-step run ~= final 5%.
TAIL_WINDOW_STEPS = 50_000
MIN_EPISODES = 20  # need at least this many total returns to compute the metric
OUT_ROOT = Path("/n/home09/pnielsen/inference_mirror_descent/scripts/topsis_out")
LSAC_DATA_DIR = Path(os.path.expanduser("~/LSAC/data"))
BASELINE_ALGOS = ["SAC", "TD3", "DIPO", "PPO", "TRPO"]
LSAC_N_SEEDS = 10
LSAC_CI_LEVEL = 0.90

# Quantile-normalization thresholds (match DoubleMirrorDescent/compute_top_run.py).
QL, QU = 0.10, 0.90

# Config fields that are NOT ablation axes -- bookkeeping the trainer or
# launch.py always writes regardless of what was swept. Anything else in a
# run's config that varies across the sweep is treated as an hp axis.
_NON_HP_CONFIG_KEYS = frozenset({
    "seed", "seed_index", "sweep_id", "config_tag", "config_tag_keys",
    "parallel_seeds", "hp_pack", "hp_pack_inline", "env",
})


def env_key_for_lsac(env: str) -> str:
    return env.split("-")[0].lower()


def load_lsac_baseline(env: str):
    pkl = LSAC_DATA_DIR / f"all_data_{env_key_for_lsac(env)}.pkl"
    if not pkl.exists():
        return None
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    return pd.read_csv(StringIO(data))


def baseline_benchmark_score(df: pd.DataFrame, tail_window_steps: int = TAIL_WINDOW_STEPS,
                             ci_level: float = LSAC_CI_LEVEL, n_seeds: int = LSAC_N_SEEDS):
    if df is None or df.empty:
        return None
    t_crit = stats.t.ppf(1 - (1 - ci_level) / 2, df=n_seeds - 1)
    best = None
    for algo in BASELINE_ALGOS:
        ad = df[df["algo"] == algo].sort_values("steps")
        if ad.empty:
            continue
        steps = ad["steps"].to_numpy(dtype=float)
        means = ad["rew_mean"].to_numpy(dtype=float)
        stds = ad["rew_std"].to_numpy(dtype=float)
        upper = means + t_crit * stds / np.sqrt(n_seeds)
        max_step = float(steps.max())
        tail = upper[steps > max_step - tail_window_steps]
        if tail.size == 0:
            tail = upper[-1:]
        score = float(np.mean(tail))
        if best is None or score > best:
            best = score
    return best


def load_benchmark_scores(envs):
    benchmark_scores = {}
    missing = []
    for env in envs:
        df = load_lsac_baseline(env)
        score = baseline_benchmark_score(df)
        if score is None or not np.isfinite(score):
            missing.append(env)
            continue
        benchmark_scores[env] = float(score)
    if missing:
        raise RuntimeError(f"missing LSAC benchmark data for envs: {missing}")
    return benchmark_scores


def log_score(mean_score: float, benchmark_score: float):
    if benchmark_score is None or benchmark_score <= 0:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = mean_score / benchmark_score
    if not np.isfinite(ratio):
        ratio = 0.0
    return float(np.log2(np.clip(ratio, 0.01, 1.0)))


def log_score_array(mean_scores, benchmark_score: float):
    if benchmark_score is None or benchmark_score <= 0:
        return None
    arr = np.asarray(mean_scores, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = arr / benchmark_score
    ratio = np.where(np.isfinite(ratio), ratio, 0.0)
    return np.log2(np.clip(ratio, 0.01, 1.0))


def build_per_env_metrics(cell_df, benchmark_scores):
    per_env = cell_df.groupby(["config_tag", "env"])["metric"].agg(["mean", "std", "count"]).reset_index()
    per_env["benchmark_score"] = per_env["env"].map(benchmark_scores)
    per_env["log_score"] = [
        log_score(mean_score, benchmark_score)
        for mean_score, benchmark_score in zip(per_env["mean"], per_env["benchmark_score"])
    ]
    return per_env


def _values_differ(a, b):
    """Treat two NaN/missing values as equal; otherwise compare directly."""
    a_missing = a is None or (isinstance(a, float) and np.isnan(a))
    b_missing = b is None or (isinstance(b, float) and np.isnan(b))
    if a_missing and b_missing:
        return False
    if a_missing or b_missing:
        return True
    return a != b


def build_h1_neighbor_map(config_tags, configs_df, hp_cols):
    """Map each config_tag to the list of config_tags that differ in exactly
    one swept hyperparameter (Hamming distance 1) in hp-value space."""
    sub = configs_df.drop_duplicates("config_tag").set_index("config_tag")
    hp_values = {}
    for c in config_tags:
        if c in sub.index:
            hp_values[c] = tuple(sub.loc[c, hp_cols].tolist())
        else:
            hp_values[c] = None

    neighbors = {}
    for c in config_tags:
        vc = hp_values[c]
        nb = []
        if vc is not None:
            for c2 in config_tags:
                if c2 == c:
                    continue
                v2 = hp_values[c2]
                if v2 is None:
                    continue
                dist = sum(1 for a, b in zip(vc, v2) if _values_differ(a, b))
                if dist == 1:
                    nb.append(c2)
        neighbors[c] = nb
    return neighbors


def add_h1_smoothed_log_scores(per_env, configs_df, hp_cols, envs):
    """Add ``h1_smoothed_log_score``: for each (config, env), the mean of the
    config's own log-score pooled with the log-scores of all configs at Hamming
    distance 1 (exactly one differing swept hyperparameter) in the same env.
    """
    per_env = per_env.copy()
    if not hp_cols:
        per_env["h1_smoothed_log_score"] = per_env["log_score"]
        return per_env

    config_tags = sorted(per_env["config_tag"].unique())
    neighbors = build_h1_neighbor_map(config_tags, configs_df, hp_cols)

    log_by_cell = {
        (c, e): v
        for c, e, v in zip(per_env["config_tag"], per_env["env"], per_env["log_score"])
    }

    smoothed_col = []
    for c, e, own in zip(per_env["config_tag"], per_env["env"], per_env["log_score"]):
        vals = []
        if own is not None and np.isfinite(own):
            vals.append(float(own))
        for nb in neighbors.get(c, []):
            v = log_by_cell.get((nb, e))
            if v is not None and np.isfinite(v):
                vals.append(float(v))
        smoothed_col.append(float(np.mean(vals)) if vals else own)

    per_env["h1_smoothed_log_score"] = smoothed_col
    return per_env


def run_bootstrap(cell_df, rankable_keys, envs, benchmark_scores, n_iter=1000, seed=0):
    """Bootstrap seeds within each (config, env) cell with replacement.

    Each iteration is treated as an independent alternate sweep: seeds are
    resampled (size = the cell's observed seed count) independently per
    (config, env), then the per-env QL/QU thresholds are recomputed from
    that iteration's pooled samples before deriving quantile_reward. The
    returned DataFrame carries 2.5/97.5 percentile CI bounds for each
    score plus the fraction of iterations each config is best on each
    score (and their mean).
    """
    rng = np.random.default_rng(seed)
    configs = sorted(rankable_keys)
    n_configs = len(configs)
    n_envs = len(envs)
    config_idx = {c: i for i, c in enumerate(configs)}
    env_idx = {e: i for i, e in enumerate(envs)}

    cell_metrics = {}  # (config_tag, env) -> np.array of metric values across seeds
    df = cell_df[cell_df["config_tag"].isin(rankable_keys)]
    for (c, e), grp in df.groupby(["config_tag", "env"]):
        cell_metrics[(c, e)] = grp["metric"].to_numpy()

    # Draw all bootstrap samples up front so we can pool per env to get
    # per-iteration QL/QU thresholds before computing quantile_reward.
    samples = {}  # (c, e) -> (n_iter, n_seeds_ce)
    for (c, e), metrics in cell_metrics.items():
        n = len(metrics)
        if n == 0:
            continue
        samples[(c, e)] = rng.choice(metrics, size=(n_iter, n), replace=True)

    ql_iter = {}  # env -> (n_iter,)
    qu_iter = {}
    for e in envs:
        pooled = [samples[(c, e)] for c in configs if (c, e) in samples]
        if not pooled:
            continue
        arr = np.concatenate(pooled, axis=1)  # (n_iter, total_seeds_in_env)
        ql_iter[e] = np.quantile(arr, QL, axis=1)
        qu_iter[e] = np.quantile(arr, QU, axis=1)

    q_mat = np.zeros((n_iter, n_configs, n_envs))  # mean quantile_reward per (iter, config, env)
    q_sum = np.zeros((n_iter, n_configs))
    q_cnt = np.zeros(n_configs)
    l_mat = np.full((n_iter, n_configs, n_envs), np.nan)

    for (c, e), sampled in samples.items():
        ci = config_idx[c]
        ei = env_idx[e]
        n = sampled.shape[1]

        low = ql_iter[e][:, None]   # (n_iter, 1)
        high = qu_iter[e][:, None]  # (n_iter, 1)
        denom = high - low
        denom_safe = np.where(denom == 0, 1.0, denom)
        qr = np.where(sampled < low, -1.0,
                np.where(sampled >= high, 1.0,
                         (sampled - low) / denom_safe))
        qr = np.where(denom == 0, 0.0, qr)
        q_mat[:, ci, ei] = qr.mean(axis=1)
        q_sum[:, ci] += qr.sum(axis=1)
        q_cnt[ci] += n

        benchmark = benchmark_scores.get(e)
        if benchmark is not None:
            l_mat[:, ci, ei] = log_score_array(sampled.mean(axis=1), benchmark)

    w = np.ones(n_envs) / n_envs
    D01 = (q_mat + 1.0) / 2.0
    Dplus = np.sqrt((((1.0 - D01) ** 2) * w).sum(axis=2))
    Dminus = np.sqrt(((D01 ** 2) * w).sum(axis=2))
    topsis_b = Dminus / (Dplus + Dminus)
    q_cnt_safe = np.where(q_cnt > 0, q_cnt, 1)
    l_cnt = np.isfinite(l_mat).sum(axis=2)
    l_cnt_safe = np.where(l_cnt > 0, l_cnt, 1)
    quantile_b = q_sum / q_cnt_safe
    log_b = np.where(l_cnt > 0, np.nansum(l_mat, axis=2) / l_cnt_safe, np.nan)

    def ci_and_pbest(arr):
        lo = np.quantile(arr, 0.025, axis=0)
        hi = np.quantile(arr, 0.975, axis=0)
        best = np.argmax(arr, axis=1)
        p = np.bincount(best, minlength=n_configs) / n_iter
        return lo, hi, p

    ts_lo, ts_hi, ts_p = ci_and_pbest(topsis_b)
    q_lo, q_hi, q_p = ci_and_pbest(quantile_b)
    l_lo, l_hi, l_p = ci_and_pbest(log_b)
    avg_p = (ts_p + q_p + l_p) / 3.0

    return pd.DataFrame({
        "config_tag": configs,
        "topsis_score_ci_lower": ts_lo,
        "topsis_score_ci_upper": ts_hi,
        "quantile_score_ci_lower": q_lo,
        "quantile_score_ci_upper": q_hi,
        "log_score_ci_lower": l_lo,
        "log_score_ci_upper": l_hi,
        "topsis_p_best": ts_p,
        "quantile_p_best": q_p,
        "log_score_p_best": l_p,
        "avg_p_best": avg_p,
    })


def build_hp_level_summary(ranking, configs_df, hp_cols, metrics):
    """Per-(hp, value) table: avg_log_score plus win-fraction of each metric
    across same-other-hp slices.

    A "slice" is a set of configs that share every hp except the target one.
    For each slice we rank by each metric and record which value of the
    target hp is top. win_{metric} = (# slices where this value is top) /
    (# non-degenerate slices). Slices with only one candidate contribute a
    trivial win to their single value; we count them so the fractions sum
    to 1 across the hp's values.
    """
    if not hp_cols:
        return None
    # configs_df must carry the hp columns AND be joinable on config_tag.
    configs_small = configs_df.drop_duplicates("config_tag").set_index("config_tag")
    usable_hps = [c for c in hp_cols if c in configs_small.columns
                  and configs_small[c].nunique(dropna=True) >= 2]
    if not usable_hps:
        return None
    joined = (
        ranking.set_index("config_tag")
        .join(configs_small[usable_hps], how="inner")
        .reset_index()
    )

    rows = []
    for hp in usable_hps:
        other = [h for h in usable_hps if h != hp]
        win_counts = {m: {} for m in metrics}
        total_slices = 0
        if other:
            slice_groups = joined.groupby(other, dropna=False)
        else:
            slice_groups = [(None, joined)]
        for _, grp in slice_groups:
            if len(grp) == 0:
                continue
            total_slices += 1
            for m in metrics:
                if m not in grp.columns or grp[m].isna().all():
                    continue
                winner = grp.loc[grp[m].idxmax(), hp]
                win_counts[m][winner] = win_counts[m].get(winner, 0) + 1

        for val in sorted(joined[hp].dropna().unique()):
            row = {"hp": hp, "value": val}
            sub = joined[joined[hp] == val]
            avg = float(sub["log_score"].mean()) if "log_score" in sub.columns else np.nan
            row["avg_log_score"] = round(avg, 3) if not np.isnan(avg) else np.nan
            win_fracs = []
            for m in metrics:
                w = win_counts[m].get(val, 0)
                frac = w / total_slices if total_slices else np.nan
                row[f"win_{m}"] = round(frac, 3) if not np.isnan(frac) else np.nan
                win_fracs.append(frac)
            row["avg_win"] = round(float(np.nanmean(win_fracs)), 3) if win_fracs else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def discover_hp_names(runs):
    """Return the list of hp-axis names that were varied in this sweep.

    Wandb-only, zero hardcoded hp names:
      1. Prefer ``config_tag_keys`` from the first run that carries it -- this
         is the authoritative list launch.py wrote, matching exactly what the
         trainer baked into ``config_tag``.
      2. Otherwise (legacy sweeps that predate the field) discover axes by
         finding config keys whose values differ across runs. Values that
         match exactly for every run can't be ablation axes.

    Either way, ``_NON_HP_CONFIG_KEYS`` bookkeeping fields are excluded.
    """
    for r in runs:
        raw = r.config.get("config_tag_keys")
        if raw:
            if isinstance(raw, str):
                return [k.strip() for k in raw.split(",")
                        if k.strip() and k.strip() not in _NON_HP_CONFIG_KEYS]
            if isinstance(raw, list):
                return [str(k) for k in raw if str(k) not in _NON_HP_CONFIG_KEYS]
    # Fallback: values-vary detection.
    value_sets: dict = {}
    for r in runs:
        for k, v in r.config.items():
            if k in _NON_HP_CONFIG_KEYS:
                continue
            try:
                hash(v)
            except TypeError:
                continue  # unhashable (list/dict) -- skip
            value_sets.setdefault(k, set()).add(v)
    return sorted(k for k, vs in value_sets.items() if len(vs) > 1)


def _per_slot_from_run(r, hp_names):
    """Return (per_slot_dict, config_tag, seed_val, seed_idx) or None.

    per_slot is keyed by the ablation-axis names discovered for this sweep,
    plus ``seed``. config_tag is taken verbatim from the run's wandb config --
    the trainer built it from the same hp values and (when available) the
    same tag_keys list, so there's nothing to reconstruct.
    """
    seed_idx = r.config.get("seed_index")
    if "config_tag" not in r.config:
        return None
    per_slot = {k: r.config[k] for k in hp_names if k in r.config}
    per_slot["seed"] = r.config.get("seed")
    return per_slot, r.config["config_tag"], r.config.get("seed"), seed_idx


def fetch_run_history(r, hp_names, benchmark_scores, config_tag_override=None):
    """Return a dict summarizing one wandb run, or None if unusable.

    Pulls episode-return history, computes the LSAC metric, and attaches
    per-slot identifying hyperparameters + config_tag.
    """
    try:
        env = r.group if (r.group in ENVS) else r.config.get("env")
        if env not in ENVS:
            return None
        if config_tag_override is None:
            res = _per_slot_from_run(r, hp_names)
        else:
            per_slot = {k: r.config[k] for k in hp_names if k in r.config}
            per_slot["seed"] = r.config.get("seed")
            res = (per_slot, config_tag_override, r.config.get("seed"), r.config.get("seed_index"))
        if res is None:
            return None
        per_slot, config_tag, seed_val, seed_idx = res
        ep_key = f"episode_return/{env}"
        hist = r.history(keys=[ep_key, "_step"], samples=10000, pandas=True)
        if hist is None or hist.empty or ep_key not in hist.columns:
            return None
        # Keep only rows that actually carry an episode return; _step is the
        # env step at which the episode ended (see VmapOffPolicyTrainer).
        eps = hist[["_step", ep_key]].dropna(subset=[ep_key])
        if len(eps) < MIN_EPISODES:
            return None
        max_step = int(eps["_step"].max())
        tail = eps[eps["_step"] > max_step - TAIL_WINDOW_STEPS]
        if tail.empty:
            return None
        metric = float(tail[ep_key].mean())
        benchmark_score = benchmark_scores.get(env)
        n = len(eps)
        out = {
            "run_id": r.id,
            "run_name": r.name,
            "env": env,
            "config_tag": config_tag,
            "seed": int(seed_val) if seed_val is not None else None,
            "seed_index": int(seed_idx) if seed_idx is not None else None,
            "state": r.state,
            "n_episodes": n,
            "mean_score": metric,
            "metric": metric,
            "benchmark_score": benchmark_score,
        }
        for k, v in per_slot.items():
            if k != "seed":
                out[k] = v
        return out
    except Exception as e:
        return {"error": str(e), "run_id": getattr(r, "id", "?"), "run_name": getattr(r, "name", "?")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep-id", "--sweep_id", type=int, required=True,
                    help="Pull runs where config.sweep_id == N.")
    ap.add_argument("--out-dir", "--out_dir", type=Path, default=None,
                    help="Override output directory. Default: "
                         f"{OUT_ROOT}/sweep_<N> so each sweep's outputs "
                         "land in their own folder and don't overwrite earlier ones.")
    args = ap.parse_args()

    OUT_DIR = args.out_dir if args.out_dir is not None \
              else OUT_ROOT / f"sweep_{args.sweep_id}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to {OUT_DIR}")
    api = wandb.Api(timeout=60)
    benchmark_scores = load_benchmark_scores(ENVS)

    print(f"Querying wandb for config.sweep_id == {args.sweep_id} ...")
    stub_runs = list(api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"config.sweep_id": int(args.sweep_id)},
        per_page=500,
    ))
    print(f"  {len(stub_runs)} runs in sweep {args.sweep_id}")

    # ``api.runs(filters=...)`` returns paginated stub ``Run`` objects whose
    # ``.config`` is empty regardless of what the backend stores. To read the
    # real per-run config we must re-fetch each by id via ``api.run(path)``.
    # That's one HTTP call per run; 8-way parallel keeps it manageable.
    def _full_fetch(rid):
        return api.run(f"{ENTITY}/{PROJECT}/{rid}")
    print(f"Fetching full configs for {len(stub_runs)} runs (8-way parallel)...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        runs = list(ex.map(_full_fetch, [r.id for r in stub_runs]))
    print(f"  full configs loaded.")

    hp_names = discover_hp_names(runs)
    print(f"Discovered hp axes for this sweep: {hp_names}")

    # Fetch histories in parallel.
    records, errors = [], []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_run_history, r, hp_names, benchmark_scores): r for r in runs}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            if rec is None:
                continue
            if "error" in rec:
                errors.append(rec)
                continue
            records.append(rec)
            if i % 20 == 0:
                print(f"  fetched {i}/{len(futs)} run histories")
    print(f"Usable run-history records: {len(records)}; errors: {len(errors)}")
    if errors:
        print("First 3 errors:", errors[:3])

    runs_df = pd.DataFrame(records)
    if runs_df.empty:
        print("No usable run data; aborting.")
        return
    runs_df.to_csv(OUT_DIR / "all_runs_metrics.csv", index=False)
    print(f"\nRun-level metric table written to {OUT_DIR/'all_runs_metrics.csv'}")

    # Discover hp columns from whatever made it through fetch_run_history.
    BOOKKEEPING = {"run_id", "run_name", "env", "config_tag", "seed",
                   "seed_index", "state", "n_episodes", "mean_score", "metric",
                   "benchmark_score"}
    hp_cols = [c for c in runs_df.columns if c not in BOOKKEEPING]
    configs_df = runs_df.drop_duplicates(subset=["config_tag"])[["config_tag"] + hp_cols]
    if hp_cols:
        configs_df = configs_df.sort_values(hp_cols)
    else:
        configs_df = configs_df.sort_values("config_tag")
    configs_df = configs_df.reset_index(drop=True)
    configs_df.insert(0, "config_id", range(len(configs_df)))
    configs_df.to_csv(OUT_DIR / "launched_configs.csv", index=False)
    print(f"\n{len(configs_df)} unique configs observed in sweep "
          f"(hp columns: {hp_cols})")

    # Dedup (env, config, seed) collisions, preferring the longest history.
    runs_df = runs_df.sort_values("n_episodes", ascending=False)
    cell_df = runs_df.drop_duplicates(subset=["env", "config_tag", "seed"], keep="first").copy()
    collisions = len(runs_df) - len(cell_df)
    print(f"De-duplicated {collisions} collision entries; "
          f"{len(cell_df)} unique (env, config, seed) cells.")

    cell_df.to_csv(OUT_DIR / "per_cell_metrics.csv", index=False)
    per_env = build_per_env_metrics(cell_df, benchmark_scores)
    per_env = add_h1_smoothed_log_scores(per_env, configs_df, hp_cols, ENVS)
    per_env.to_csv(OUT_DIR / "per_config_env_metrics.csv", index=False)
    print(f"Per-(config,env) seed-averaged metrics written (rows: {len(per_env)})")

    # Ranking: every config that has at least one data point in every env
    # we want to rank over. No admissibility thresholding -- any number of
    # seeds per cell is allowed; the mean metric is whatever it is.
    pivot = per_env.pivot(index="config_tag", columns="env", values="mean")
    for env in ENVS:
        if env not in pivot.columns:
            pivot[env] = np.nan
    pivot = pivot[ENVS].dropna()
    print(f"\nRanking matrix (configs with data for every env): "
          f"{pivot.shape[0]} x {pivot.shape[1]}")
    if pivot.empty:
        print("No configs with full per-env coverage; aborting ranking.")
        return
    pivot.to_csv(OUT_DIR / "topsis_input_raw_rewards.csv")

    rankable_keys = set(pivot.index)

    # Per-env quantile thresholds across runs (not configs).
    ql_percentile = cell_df.groupby("env")["metric"].quantile(QL)
    qu_percentile = cell_df.groupby("env")["metric"].quantile(QU)

    def quantile_reward(raw, env):
        low = ql_percentile[env]
        high = qu_percentile[env]
        if high == low:
            return 0.0
        if raw < low:
            return -1.0
        if raw >= high:
            return 1.0
        return (raw - low) / (high - low)

    cell_df = cell_df.copy()
    cell_df["quantile_reward"] = [
        quantile_reward(r, e) for r, e in zip(cell_df["metric"], cell_df["env"])
    ]
    q_per_cell = cell_df.pivot_table(
        index="config_tag", columns="env", values="quantile_reward", aggfunc="mean"
    )
    q_per_cell = q_per_cell.loc[[k for k in q_per_cell.index if k in rankable_keys], ENVS]
    q_per_cell.to_csv(OUT_DIR / "quantile_reward_per_config_env.csv")

    joint_quantile = (
        cell_df[cell_df["config_tag"].isin(rankable_keys)]
        .groupby("config_tag")["quantile_reward"].mean()
        .rename("quantile_score")
    )

    # TOPSIS on [0,1]-shifted per-(config,env) quantile values.
    D01 = (q_per_cell + 1.0) / 2.0
    w = np.ones(len(ENVS)) / len(ENVS)
    Dplus = np.sqrt((((1.0 - D01) ** 2) * w).sum(axis=1))
    Dminus = np.sqrt(((D01 ** 2) * w).sum(axis=1))
    topsis_score = (Dminus / (Dplus + Dminus)).rename("topsis_score")

    log_avg = (
        per_env[per_env["config_tag"].isin(rankable_keys)]
        .groupby("config_tag")["log_score"].mean()
        .rename("log_score")
    )

    h1_smoothed_avg = (
        per_env[per_env["config_tag"].isin(rankable_keys)]
        .groupby("config_tag")["h1_smoothed_log_score"].mean()
        .rename("h1_smoothed_log_score")
    )

    ranking = (
        pivot.join(topsis_score).join(joint_quantile).join(log_avg).join(h1_smoothed_avg)
        .sort_values("topsis_score", ascending=False)
        .reset_index()
    )

    print("\nRunning 1000 bootstrap iterations over seeds within each (config, env) ...")
    boot_df = run_bootstrap(cell_df, rankable_keys, ENVS, benchmark_scores)
    ranking = ranking.merge(boot_df, on="config_tag", how="left")

    score_cols = [c for c in ("topsis_score", "quantile_score", "log_score", "h1_smoothed_log_score")
                  if c in ranking.columns]
    p_best_map = {"topsis_score": "topsis_p_best",
                  "quantile_score": "quantile_p_best",
                  "log_score": "log_score_p_best"}
    score_block = []
    for c in score_cols:
        score_block.append(c)
        for suffix in ("_ci_lower", "_ci_upper"):
            col = f"{c}{suffix}"
            if col in ranking.columns:
                score_block.append(col)
        pb = p_best_map.get(c)
        if pb in ranking.columns:
            score_block.append(pb)
    if "avg_p_best" in ranking.columns:
        score_block.append("avg_p_best")
    env_cols = [e for e in ENVS if e in ranking.columns]
    front = score_block + env_cols + ["config_tag"]
    cols = front + [c for c in ranking.columns if c not in front]
    ranking = ranking[cols]
    env_round = {e: 1 for e in env_cols}
    score_round = {c: 3 for c in score_block if c not in env_cols and c != "config_tag"}
    ranking = ranking.round({**env_round, **score_round})
    ranking.to_csv(OUT_DIR / "topsis_ranking.csv", index=False)
    print("\n=== Config ranking (sorted by TOPSIS; quantile_score shown too) ===")
    print(ranking.round(3).to_string())

    hp_summary = build_hp_level_summary(
        ranking=ranking,
        configs_df=configs_df,
        hp_cols=hp_cols,
        metrics=("topsis_score", "quantile_score", "log_score"),
    )
    if hp_summary is not None and not hp_summary.empty:
        hp_summary.to_csv(OUT_DIR / "hp_level_summary.csv", index=False)
        print("\n=== Per-hp level summary "
              "(avg_log_score + win-fraction for each score; slices fix all other hps) ===")
        print(hp_summary.round(3).to_string(index=False))

    print("\n=== Top configs by TOPSIS ===")
    top_topsis = topsis_score.sort_values(ascending=False).head(20)
    print(top_topsis.round(4).to_string())

    print("\n=== Top configs by joint quantile score ===")
    top_q = joint_quantile.sort_values(ascending=False).head(20)
    print(top_q.round(4).to_string())

    def per_env_stats(cfg_list, label):
        mask = cell_df["config_tag"].isin(cfg_list)
        s = (
            cell_df[mask]
            .groupby(["config_tag", "env"])["metric"]
            .agg(["mean", "std"])
            .reindex(pd.MultiIndex.from_product([cfg_list, ENVS], names=["config_tag", "env"]))
        )
        s.to_csv(OUT_DIR / f"per_env_stats_top5_by_{label}.csv")
        return s

    per_env_stats(top_topsis.head(5).index.tolist(), "topsis")
    per_env_stats(top_q.head(5).index.tolist(), "quantile")

    per_env_std = per_env[per_env["config_tag"].isin(rankable_keys)].pivot(
        index="config_tag", columns="env", values="std")
    per_env_std.to_csv(OUT_DIR / "topsis_input_seed_std.csv")

    coll = (
        runs_df.groupby(["env", "config_tag", "seed"]).agg(
            n_wandb_runs=("run_id", "count"),
            max_episodes=("n_episodes", "max"),
            min_episodes=("n_episodes", "min"),
        ).reset_index()
    )
    coll = coll[coll["n_wandb_runs"] > 1]
    coll.to_csv(OUT_DIR / "collisions.csv", index=False)
    print(f"\nCollision cells (>1 wandb run sharing a (env,config,seed)): {len(coll)}")
    if len(coll):
        print(coll.to_string(index=False))

    print(f"\nAll outputs written under {OUT_DIR}")


if __name__ == "__main__":
    main()
