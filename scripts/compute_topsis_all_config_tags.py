#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore

from compute_topsis import (
    OUT_ROOT,
    QL,
    QU,
    MIN_EPISODES,
    TAIL_WINDOW_STEPS,
    build_per_env_metrics,
    load_benchmark_scores,
    run_bootstrap,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
WANDB_ROOT = Path("/n/netscratch/kdbrantley_lab/Lab/pnielsen/wandb")
V5_ENVS = [
    "Ant-v5",
    "HalfCheetah-v5",
    "Hopper-v5",
    "Humanoid-v5",
    "Swimmer-v5",
    "Walker2d-v5",
]


def _parse_started_at(text: str):
    parts = text.strip().split()
    if len(parts) < 6:
        return None
    return datetime.strptime(
        f"{parts[1]} {parts[2]} {parts[3]} {parts[5]}",
        "%b %d %H:%M:%S %Y",
    )


def _coerce_scalar(v):
    if not isinstance(v, str):
        return v
    if v in {"True", "true"}:
        return True
    if v in {"False", "false"}:
        return False
    if v in {"None", "null"}:
        return None
    if re.fullmatch(r"[-+]?\d+", v):
        try:
            return int(v)
        except Exception:
            pass
    if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?\.inf", v):
        try:
            return float(v)
        except Exception:
            pass
    return v


def _parse_running_command(command: str):
    tokens = shlex.split(command)
    args = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:].replace("-", "_")
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            args[key] = _coerce_scalar(tokens[i + 1])
            i += 2
        else:
            args[key] = True
            i += 1
    return args


def _locate_local_csv(args: dict, started_at):
    env = args.get("env")
    alg = str(args.get("alg", ""))
    seed = int(args.get("seed", 0))
    suffix = str(args.get("suffix", ""))
    env_dir = LOCAL_LOG_ROOT / str(env)
    if not env_dir.exists():
        return None
    candidates = []
    pattern = re.compile(
        rf"^{re.escape(alg)}_(\d{{4}}-\d{{2}}-\d{{2}}_\d{{2}}-\d{{2}}-\d{{2}})_s{seed}_(.*)$"
    )
    for child in env_dir.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if not m or m.group(2) != suffix:
            continue
        csv_path = child / "episode_returns.csv"
        if not csv_path.exists():
            continue
        try:
            dt = datetime.strptime(m.group(1), "%Y-%m-%d_%H-%M-%S")
        except Exception:
            continue
        dist = abs((dt - started_at).total_seconds()) if started_at is not None else float("inf")
        candidates.append((dist, -dt.timestamp(), csv_path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


def _read_local_slurm_headers(slurm_out: Path):
    running = None
    started_at = None
    wandb_dir = None
    with open(slurm_out) as f:
        for i, line in enumerate(f):
            if line.startswith("Running: "):
                running = line[len("Running: "):].strip()
            elif line.startswith("Started at: "):
                started_at = _parse_started_at(line[len("Started at: "):])
            elif line.startswith("WANDB_DIR="):
                wandb_dir = Path(line[len("WANDB_DIR="):].strip())
            if i >= 30:
                break
    return running, started_at, wandb_dir


def _coerce_yaml_scalar(raw: str):
    raw = raw.strip()
    if raw == "":
        return ""
    if (raw.startswith("\"") and raw.endswith("\"")) or (raw.startswith("'") and raw.endswith("'")):
        raw = raw[1:-1]
    if raw.lower() in {"null", "none"}:
        return None
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    try:
        return int(raw)
    except Exception:
        pass
    try:
        return float(raw)
    except Exception:
        pass
    return raw


def _read_wandb_config_values(config_path: Path):
    values = {}
    current = None
    with open(config_path) as f:
        for line in f:
            if line and not line.startswith(" ") and line.rstrip().endswith(":"):
                current = line.split(":", 1)[0]
                continue
            if current is not None and line.startswith("  value:"):
                values[current] = _coerce_yaml_scalar(line.split(":", 1)[1])
                current = None
    return values


def _offline_run_id(config_path: Path) -> str:
    return config_path.parents[1].name.rsplit("-", 1)[-1]


def _history_key(item) -> str:
    if len(item.nested_key) > 0:
        return "/".join(item.nested_key)
    return item.key


def _read_history_from_run_file(run_file: Path, env: str):
    keys = [f"episode_return/{env}", "sample/episode_return"]
    ds = datastore.DataStore()
    rows = []
    try:
        ds.open_for_scan(str(run_file))
        while True:
            data = ds.scan_data()
            if data is None:
                break
            rec = wandb_internal_pb2.Record()
            rec.ParseFromString(data)
            if rec.WhichOneof("record_type") != "history":
                continue
            step_val = None
            return_val = None
            for item in rec.history.item:
                key = _history_key(item)
                if key == "_step":
                    step_val = json.loads(item.value_json)
                elif key in keys and return_val is None:
                    return_val = json.loads(item.value_json)
            if step_val is None or return_val is None:
                continue
            rows.append({"_step": step_val, "return": return_val})
    finally:
        try:
            ds.close()
        except Exception:
            pass
    if not rows:
        return None
    hist = pd.DataFrame(rows)
    hist["_step"] = pd.to_numeric(hist["_step"], errors="coerce")
    hist["return"] = pd.to_numeric(hist["return"], errors="coerce")
    hist = hist.dropna().reset_index(drop=True)
    if hist.empty:
        return None
    return hist


def collect_local_records(benchmark_scores):
    records = []
    config_cache = {}
    history_cache = {}
    config_paths = sorted(WANDB_ROOT.glob("sweep_*/job_*/wandb/offline-run-*/files/config.yaml"))
    matched_jobs = set()
    for config_path in config_paths:
        if config_path not in config_cache:
            config_cache[config_path] = _read_wandb_config_values(config_path)
        cfg = config_cache[config_path]
        env = cfg.get("env")
        if env not in V5_ENVS:
            continue
        raw_config_tag = cfg.get("config_tag")
        if not isinstance(raw_config_tag, str) or not raw_config_tag.strip():
            continue
        seed_index = cfg.get("seed_index")
        if seed_index is None:
            seed_index = cfg.get("seed")
        try:
            seed_index = int(seed_index)
        except Exception:
            continue
        run_seed = cfg.get("seed")
        try:
            run_seed = None if run_seed is None else int(run_seed)
        except Exception:
            run_seed = None
        run_id = _offline_run_id(config_path)
        run_file = config_path.parents[1] / f"run-{run_id}.wandb"
        if not run_file.exists():
            continue
        if run_file not in history_cache:
            try:
                history_cache[run_file] = _read_history_from_run_file(run_file, env)
            except Exception:
                history_cache[run_file] = None
        eps = history_cache[run_file]
        if eps is None:
            continue
        matched_jobs.add(config_path.parents[3].name)
        if len(eps) < MIN_EPISODES:
            continue
        max_step = int(eps["_step"].max())
        tail = eps[eps["_step"] > max_step - TAIL_WINDOW_STEPS]
        if tail.empty:
            continue
        metric = float(tail["return"].mean())
        records.append({
            "run_id": run_id,
            "run_name": raw_config_tag.strip(),
            "env": env,
            "config_tag": raw_config_tag.strip(),
            "seed": run_seed,
            "seed_index": seed_index,
            "state": "local",
            "n_episodes": int(len(eps)),
            "mean_score": metric,
            "metric": metric,
            "benchmark_score": benchmark_scores.get(env),
            "group_kind": "config_tag",
            "source_path": str(config_path),
            "local_wandb_run_path": str(run_file),
            "job_id": config_path.parents[3].name,
        })
    print(f"Matched {len(matched_jobs)} v5 jobs with local offline W&B runs.")
    print(f"Usable local run-history records: {len(records)}")
    return records


def build_ranking(cell_df, envs, benchmark_scores):
    if cell_df.empty:
        return None

    per_env = build_per_env_metrics(cell_df, benchmark_scores)
    pivot = per_env.pivot(index="config_tag", columns="env", values="mean")
    for env in envs:
        if env not in pivot.columns:
            pivot[env] = np.nan
    pivot = pivot[envs].dropna()
    if pivot.empty:
        return None

    rankable_keys = set(pivot.index)
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
    for env in envs:
        if env not in q_per_cell.columns:
            q_per_cell[env] = np.nan
    q_per_cell = q_per_cell.loc[[k for k in q_per_cell.index if k in rankable_keys], envs]

    joint_quantile = (
        cell_df[cell_df["config_tag"].isin(rankable_keys)]
        .groupby("config_tag")["quantile_reward"].mean()
        .rename("quantile_score")
    )

    D01 = (q_per_cell + 1.0) / 2.0
    w = np.ones(len(envs)) / len(envs)
    Dplus = np.sqrt((((1.0 - D01) ** 2) * w).sum(axis=1))
    Dminus = np.sqrt(((D01 ** 2) * w).sum(axis=1))
    topsis_score = (Dminus / (Dplus + Dminus)).rename("topsis_score")

    log_avg = (
        per_env[per_env["config_tag"].isin(rankable_keys)]
        .groupby("config_tag")["log_score"].mean()
        .rename("log_score")
    )

    ranking = (
        pivot.join(topsis_score).join(joint_quantile).join(log_avg)
        .sort_values("topsis_score", ascending=False)
        .reset_index()
    )

    boot_df = run_bootstrap(cell_df, rankable_keys, envs, benchmark_scores)
    return ranking.merge(boot_df, on="config_tag", how="left")


def format_ranking(ranking, envs, extra_cols=()):
    score_cols = [c for c in ("topsis_score", "quantile_score", "log_score") if c in ranking.columns]
    p_best_map = {
        "topsis_score": "topsis_p_best",
        "quantile_score": "quantile_p_best",
        "log_score": "log_score_p_best",
    }
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
    env_cols = [e for e in envs if e in ranking.columns]
    extra_block = [c for c in extra_cols if c in ranking.columns]
    front = score_block + env_cols + ["config_tag"] + extra_block
    cols = front + [c for c in ranking.columns if c not in front]
    ranking = ranking[cols]
    env_round = {e: 1 for e in env_cols}
    score_round = {c: 3 for c in score_block if c not in env_cols and c != "config_tag"}
    return ranking.round({**env_round, **score_round})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-dir",
        "--out_dir",
        type=Path,
        default=None,
        help="Override output directory. Default: scripts/topsis_out/all_config_tags_v5.",
    )
    args = ap.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else OUT_ROOT / "all_config_tags_v5"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing outputs to {out_dir}")

    benchmark_scores = load_benchmark_scores(V5_ENVS)

    print("Collecting local v5 run histories from SLURM + netscratch logs ...")
    records = collect_local_records(benchmark_scores)
    print(f"Usable run-history records: {len(records)}")

    runs_df = pd.DataFrame(records)
    if runs_df.empty:
        print("No usable run data; aborting.")
        return

    runs_df.to_csv(out_dir / "all_runs_metrics.csv", index=False)
    print(f"\nRun-level metric table written to {out_dir/'all_runs_metrics.csv'}")

    runs_df = runs_df.sort_values("n_episodes", ascending=False)
    cell_df = runs_df.drop_duplicates(subset=["env", "config_tag", "seed"], keep="first").copy()
    collisions = len(runs_df) - len(cell_df)
    print(f"De-duplicated {collisions} collision entries; {len(cell_df)} unique (env, config, seed) cells.")

    coverage = cell_df.groupby("config_tag")["env"].nunique()
    full_coverage_tags = set(coverage[coverage == len(V5_ENVS)].index)
    print(f"{len(full_coverage_tags)} config tags have runs on all {len(V5_ENVS)} v5 environments.")
    if not full_coverage_tags:
        print("No config tags with full v5 coverage; aborting.")
        return

    cell_df = cell_df[cell_df["config_tag"].isin(full_coverage_tags)].copy()
    cell_df.to_csv(out_dir / "per_cell_metrics.csv", index=False)
    per_env = build_per_env_metrics(cell_df, benchmark_scores)
    per_env.to_csv(out_dir / "per_config_env_metrics.csv", index=False)

    meta_df = (
        cell_df.groupby("config_tag", as_index=False)
        .agg(
            group_kind=("group_kind", "first"),
            source_path=("source_path", lambda s: next((x for x in s if isinstance(x, str) and x), "")),
            n_runs=("run_id", "count"),
        )
    )

    coll = (
        runs_df.groupby(["env", "config_tag", "seed"]).agg(
            n_local_runs=("run_id", "count"),
            max_episodes=("n_episodes", "max"),
            min_episodes=("n_episodes", "min"),
        ).reset_index()
    )
    coll = coll[coll["n_local_runs"] > 1]
    coll.to_csv(out_dir / "collisions.csv", index=False)

    for env in V5_ENVS:
        env_df = cell_df[cell_df["env"] == env]
        ranking = build_ranking(env_df, [env], benchmark_scores)
        if ranking is not None:
            env_meta_df = (
                env_df.groupby("config_tag", as_index=False)
                .agg(
                    group_kind=("group_kind", "first"),
                    source_path=("source_path", lambda s: next((x for x in s if isinstance(x, str) and x), "")),
                    n_runs=("run_id", "count"),
                )
            )
            ranking = ranking.merge(env_meta_df, on="config_tag", how="left")
            ranking = ranking.sort_values(env, ascending=False).reset_index(drop=True)
            ranking = format_ranking(ranking, [env], extra_cols=["group_kind", "source_path", "n_runs"])
            ranking.to_csv(out_dir / f"topsis_ranking_{env}.csv", index=False)
            print(ranking.round(3).to_string())
            print(f"\nWrote {out_dir / f'topsis_ranking_{env}.csv'}")

    ranking = build_ranking(cell_df, V5_ENVS, benchmark_scores)
    if ranking is not None:
        ranking = ranking.merge(meta_df, on="config_tag", how="left")
        ranking = format_ranking(ranking, V5_ENVS, extra_cols=["group_kind", "source_path", "n_runs"])
        ranking.to_csv(out_dir / "topsis_ranking.csv", index=False)
        print(ranking.round(3).to_string())
        print(f"\nWrote {out_dir / 'topsis_ranking.csv'}")


if __name__ == "__main__":
    main()
