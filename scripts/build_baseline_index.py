#!/usr/bin/env python3
"""Normalise every evaluation-episode source into one schema under ``baselines/``.

The four method families were produced by different launchers at different
times, so each arrived with its own column names and its own directory layout:

* DIPO   -- one file per env, a ``protocol`` column carrying both eval methods.
* DPMD   -- one file per (training variant, env), best-of-32 only.
* SAC    -- one directory per (env, seed) on netscratch, one CSV per eval method.
* MGMD   -- offline wandb runs, ten vmap slots per job, mirrored eval CSVs.

Every one of them is ultimately the same measurement: the return of one
evaluation episode, at one checkpoint, for one seed, under one action-selection
rule. This writes that single long-format table, one file per (method, env):

    algorithm variant env seed seed_index step eval_method eval_detail
    best_of_n episode_index episode_return episode_length source_path

``eval_method`` is the canonical, cross-algorithm name -- the thing a legend
entry should say -- while ``eval_detail`` keeps the mechanism that produced it,
which differs between algorithms even when the canonical name agrees (SAC scores
best-of-32 candidates with min(Q1,Q2), MGMD with the mean of its ensemble). Both
are kept because the first is what makes curves comparable and the second is
what makes them reproducible.

Run with --verify to re-read every written file and check it against its source
row-for-row before trusting it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

BASELINE_DIR = REPO / "baselines"
ENVS = ["Ant-v3", "HalfCheetah-v3", "Hopper-v3", "Humanoid-v3", "Swimmer-v3", "Walker2d-v3"]

# Canonical eval-method names. Two families only: consult the critic over N
# sampled candidates, or take the policy's own deterministic action.
BEST_OF_32 = "best_of_32"
MEAN_ACTION = "mean_action"

SCHEMA = [
    "algorithm", "variant", "env", "seed", "seed_index", "step",
    "eval_method", "eval_detail", "best_of_n",
    "episode_index", "episode_return", "episode_length", "source_path",
]

# ---- sources -------------------------------------------------------------
# The upstream exports, kept next to the normalised table they feed so the
# provenance travels with the data.
DIPO_SOURCE = BASELINE_DIR / "DIPO" / "source"
DPMD_SOURCE = BASELINE_DIR / "DPMD" / "source"
DPMD_VARIANTS = ["original", "trainN1_no_noise"]
# (launcher root, variant). "default" is SAC at its published two-hidden-layer
# critic; "critic3" is the same configuration re-run at the three-layer critic the
# diffusion baselines use, so the two are comparable up to that one change.
SAC_RUNS = [
    (Path("/n/netscratch/kdbrantley_lab/Lab/pnielsen/"
          "sac_v3_gpu_best32_mean_baselines/20260922_212513/sac"), "default"),
    (Path("/n/netscratch/kdbrantley_lab/Lab/pnielsen/"
          "sac_v3_gpu_critic3_baselines/20260924_151930/sac"), "critic3"),
]
WANDB_ROOT = Path("/n/holylabs/kdbrantley_lab/Lab/pnielsen/wandb")
# (sweep dir, config_tag, variant). One entry per MGMD training config that
# should be selectable downstream; the ``variant`` column is what picks between
# them, exactly as it does for DPMD's two variants.
MGMD_RUNS = [
    (WANDB_ROOT / "sweep_244", "sweep244_buffer_size=250000_rollout_alpha=1",
     "sweep244_rollout_alpha1"),
    (WANDB_ROOT / "sweep_247", "sweep247_buffer_size=250000",
     "sweep247_diffusion50"),
    (WANDB_ROOT / "sweep_249", "sweep249_buffer_size=250000",
     "sweep249_diffusion25"),
    (WANDB_ROOT / "sweep_261", "sweep261_buffer_size=250000",
     "sweep261_alpha0"),
    (WANDB_ROOT / "sweep_262", "sweep262_alpha=0_beta=128_buffer_size=250000",
     "sweep262_alpha0_beta128"),
]


def _rel(path) -> str:
    """Provenance as a repo-relative path where possible, else absolute."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO))
    except ValueError:
        return str(p)


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reindex(columns=SCHEMA)
    df["seed"] = df["seed"].astype(int)
    df["seed_index"] = df["seed_index"].astype(int)
    df["step"] = df["step"].astype(int)
    df["best_of_n"] = df["best_of_n"].astype(int)
    df["episode_index"] = df["episode_index"].astype(int)
    df["episode_return"] = df["episode_return"].astype(float)
    return df.sort_values(
        ["eval_method", "seed_index", "step", "episode_index"]
    ).reset_index(drop=True)


# ---- DIPO ----------------------------------------------------------------
def load_dipo(env: str) -> pd.DataFrame:
    src = DIPO_SOURCE / f"DIPO__{env}__dual_eval_raw.csv"
    if not src.exists():
        return pd.DataFrame(columns=SCHEMA)
    raw = pd.read_csv(src)
    # 'native' is Diffusion.sample(eval=True): noise_ratio=0, so every reverse
    # step returns its DDPM posterior mean and the critic is never consulted.
    method = raw["protocol"].map({"best_of_32": BEST_OF_32, "native": MEAN_ACTION})
    detail = raw["protocol"].map(
        {"best_of_32": "critic_best_of_32", "native": "ddpm_posterior_mean"}
    )
    out = pd.DataFrame({
        "algorithm": "DIPO",
        "variant": "default",
        "env": raw["env"],
        "seed": raw["seed"],
        "seed_index": raw["seed"],
        "step": raw["step"],
        "eval_method": method,
        "eval_detail": detail,
        "best_of_n": raw["best_of_n"],
        "episode_index": raw["episode_index"],
        "episode_return": raw["episode_return"],
        # DIPO's export never carried episode length.
        "episode_length": np.nan,
        "source_path": _rel(src),
    })
    return _finalise(out)


# ---- DPMD ----------------------------------------------------------------
def load_dpmd(env: str) -> pd.DataFrame:
    frames = []
    for variant in DPMD_VARIANTS:
        src = DPMD_SOURCE / variant / f"{variant}__{env}__evalN32_raw_10seeds.csv"
        if not src.exists():
            continue
        raw = pd.read_csv(src)
        frames.append(pd.DataFrame({
            "algorithm": "DPMD",
            "variant": variant,
            "env": raw["env"],
            "seed": raw["seed"],
            "seed_index": raw["seed"],
            "step": raw["step"],
            "eval_method": BEST_OF_32,
            "eval_detail": "critic_best_of_32",
            "best_of_n": raw["eval_best_of_n_actions"],
            "episode_index": raw["episode_index"],
            "episode_return": raw["episode_return"],
            "episode_length": raw["episode_length"],
            "source_path": _rel(src),
        }))
    if not frames:
        return pd.DataFrame(columns=SCHEMA)
    return _finalise(pd.concat(frames, ignore_index=True))


# ---- SAC -----------------------------------------------------------------
def _sac_seed_dirs(root: Path) -> dict[str, list[tuple[int, Path]]]:
    """env -> [(seed, run_dir)], read off the launcher's own task table."""
    tasks = pd.read_csv(root / "tasks.tsv", sep="\t")
    out: dict[str, list[tuple[int, Path]]] = {}
    for row in tasks.itertuples():
        for seed in (row.seed0, row.seed1):
            run_dir = root / "runs" / row.task_name / f"s{seed}"
            if (run_dir / "eval_episode_returns.csv").exists():
                out.setdefault(row.env_name, []).append((int(seed), run_dir))
    return {env: sorted(v) for env, v in out.items()}


def load_sac(env: str, runs: list[tuple[str, dict[str, list[tuple[int, Path]]]]]) -> pd.DataFrame:
    frames = [_load_sac_one(env, variant, dirs) for variant, dirs in runs]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SCHEMA)


def _load_sac_one(env: str, variant: str,
                  seed_dirs: dict[str, list[tuple[int, Path]]]) -> pd.DataFrame:
    frames = []
    for seed, run_dir in seed_dirs.get(env, []):
        best = run_dir / "eval_episode_returns.csv"
        raw = pd.read_csv(best)
        frames.append(pd.DataFrame({
            "algorithm": "SAC", "variant": variant, "env": env,
            "seed": seed, "seed_index": seed,
            "step": raw["env_step"],
            "eval_method": BEST_OF_32,
            # SAC scores its candidates with the pessimistic min, the same value
            # its own actor loss maximises.
            "eval_detail": "critic_best_of_32_minQ",
            "best_of_n": raw["eval_best_of_n_actions"],
            "episode_index": raw["episode_index"],
            "episode_return": raw["episode_return"],
            "episode_length": raw["episode_length"],
            "source_path": _rel(best),
        }))
        mean = run_dir / "eval_mean_episode_returns.csv"
        if mean.exists():
            raw = pd.read_csv(mean)
            frames.append(pd.DataFrame({
                "algorithm": "SAC", "variant": variant, "env": env,
                "seed": seed, "seed_index": seed,
                "step": raw["env_step"],
                "eval_method": MEAN_ACTION,
                "eval_detail": "tanh_gaussian_mean",
                "best_of_n": 1,
                "episode_index": raw["episode_index"],
                "episode_return": raw["episode_return"],
                "episode_length": raw["episode_length"],
                "source_path": _rel(mean),
            }))
    if not frames:
        return pd.DataFrame(columns=SCHEMA)
    return _finalise(pd.concat(frames, ignore_index=True))


# ---- MGMD ----------------------------------------------------------------
def _cfg_value(cfg: dict, key: str):
    node = cfg.get(key)
    return node.get("value") if isinstance(node, dict) else node


def _mgmd_jobs(sweep: Path, config_tag: str) -> dict[str, tuple[Path, dict[int, int]]]:
    """env -> (eval CSV, {slot: seed}) for the requested config tag.

    The mirrored eval CSV is keyed by vmap slot, so the slot->seed map is what
    turns it into ten identifiable runs.
    """
    from build_sweep_local_metrics import (
        _codebase_dir_from_metadata,
        _read_run_basename,
    )

    out: dict[str, tuple[Path, dict[int, int]]] = {}
    for job_dir in sorted(sweep.glob("job_*")):
        run_dirs = sorted((job_dir / "wandb").glob("offline-run-*"))
        slot_seed: dict[int, int] = {}
        env = base_name = codebase = None
        for rd in run_dirs:
            cfg_path = rd / "files" / "config.yaml"
            if not cfg_path.exists():
                continue
            cfg = yaml.safe_load(open(cfg_path)) or {}
            if _cfg_value(cfg, "config_tag") != config_tag:
                continue
            slot = _cfg_value(cfg, "seed_index")
            if slot is None:
                continue
            slot_seed[int(slot)] = int(_cfg_value(cfg, "seed"))
            env = env or _cfg_value(cfg, "env")
            if base_name is None:
                wf = list(rd.glob("run-*.wandb"))
                if wf:
                    base_name = _read_run_basename(wf[0])
            if codebase is None:
                codebase = _codebase_dir_from_metadata(
                    rd / "files" / "wandb-metadata.json")
        if not slot_seed or not (env and base_name and codebase):
            continue
        csv_path = codebase / "logs" / env / base_name / "eval_episode_returns.csv"
        if csv_path.exists():
            out[env] = (csv_path, slot_seed)
    return out


def load_mgmd(env: str, runs: list[tuple[str, dict[str, tuple[Path, dict[int, int]]]]]) -> pd.DataFrame:
    frames = [_load_mgmd_one(env, variant, jobs) for variant, jobs in runs]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=SCHEMA)


def _load_mgmd_one(env: str, variant: str,
                   jobs: dict[str, tuple[Path, dict[int, int]]]) -> pd.DataFrame:
    if env not in jobs:
        return pd.DataFrame(columns=SCHEMA)
    csv_path, slot_seed = jobs[env]
    raw = pd.read_csv(csv_path)
    raw = raw[raw["seed_index"].isin(slot_seed)]
    return _finalise(pd.DataFrame({
        "algorithm": "MGMD", "variant": variant, "env": env,
        "seed": raw["seed_index"].map(slot_seed),
        "seed_index": raw["seed_index"],
        "step": raw["step"],
        "eval_method": BEST_OF_32,
        # The tilted MALA sampler, whose guidance uses the ensemble mean.
        "eval_detail": "tilted_mala_best_of_32_meanQ",
        "best_of_n": raw["eval_best_of_n_actions"],
        "episode_index": raw["episode_index"],
        "episode_return": raw["episode_return"],
        "episode_length": raw["episode_length"],
        "source_path": _rel(csv_path),
    }))


# ---- driver --------------------------------------------------------------
def summarise(df: pd.DataFrame) -> list[dict]:
    rows = []
    for (alg, var, env, method), g in df.groupby(
            ["algorithm", "variant", "env", "eval_method"], dropna=False):
        per_ckpt = g.groupby(["seed_index", "step"]).size()
        rows.append({
            "algorithm": alg, "variant": var, "env": env, "eval_method": method,
            "eval_detail": g["eval_detail"].iloc[0],
            "n_seeds": g["seed_index"].nunique(),
            "n_checkpoints": g["step"].nunique(),
            "episodes_per_checkpoint": int(per_ckpt.min()),
            "first_step": int(g["step"].min()),
            "last_step": int(g["step"].max()),
            "n_rows": len(g),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=BASELINE_DIR)
    ap.add_argument("--verify", action="store_true",
                    help="re-read each written file and check totals against source")
    args = ap.parse_args()

    sac_runs = [(variant, _sac_seed_dirs(root)) for root, variant in SAC_RUNS]
    mgmd_runs = [(variant, _mgmd_jobs(sweep, tag)) for sweep, tag, variant in MGMD_RUNS]
    for variant, dirs in sac_runs:
        print(f"SAC {variant}: {sum(len(v) for v in dirs.values())} run dirs "
              f"across {len(dirs)} envs")
    for (_sweep, tag, variant), (_v, jobs) in zip(MGMD_RUNS, mgmd_runs):
        print(f"MGMD {variant}: {len(jobs)} envs at tag {tag!r}")

    loaders = {
        "DIPO": lambda e: load_dipo(e),
        "DPMD": lambda e: load_dpmd(e),
        "SAC": lambda e: load_sac(e, sac_runs),
        "MGMD": lambda e: load_mgmd(e, mgmd_runs),
    }

    index_rows: list[dict] = []
    for algorithm, loader in loaders.items():
        dest = args.out_dir / algorithm
        dest.mkdir(parents=True, exist_ok=True)
        for env in ENVS:
            df = loader(env)
            if df.empty:
                print(f"  {algorithm:5s} {env:15s} -- no data")
                continue
            path = dest / f"{algorithm}__{env}__eval_episodes.csv"
            df.to_csv(path, index=False)
            index_rows.extend(summarise(df))
            methods = "+".join(sorted(df["eval_method"].unique()))
            print(f"  {algorithm:5s} {env:15s} {len(df):6d} rows  [{methods}]")

    index = pd.DataFrame(index_rows).sort_values(
        ["algorithm", "variant", "eval_method", "env"]).reset_index(drop=True)
    index.to_csv(args.out_dir / "index.csv", index=False)
    print(f"\nWrote {args.out_dir / 'index.csv'} ({len(index)} groups)")

    bad = index[(index.n_seeds != 10) | (index.n_checkpoints != 100)
                | (index.episodes_per_checkpoint != 10)]
    if len(bad):
        print("\nGroups that are NOT 10 seeds x 100 checkpoints x 10 episodes:")
        print(bad.to_string(index=False))
    else:
        print("All groups are 10 seeds x 100 checkpoints x 10 episodes.")

    if args.verify:
        verify(args.out_dir)


def verify(out_dir: Path) -> None:
    """Check each unified file against the source it was built from."""
    print("\n--- verification ---")
    ok = True

    for env in ENVS:
        src = DIPO_SOURCE / f"DIPO__{env}__dual_eval_raw.csv"
        dst = out_dir / "DIPO" / f"DIPO__{env}__eval_episodes.csv"
        if src.exists() and dst.exists():
            a, b = pd.read_csv(src), pd.read_csv(dst)
            ok &= _cmp(f"DIPO {env}", len(a), len(b),
                       a.episode_return.sum(), b.episode_return.sum())

    for env in ENVS:
        dst = out_dir / "DPMD" / f"DPMD__{env}__eval_episodes.csv"
        if not dst.exists():
            continue
        n = ret = 0.0
        rows = 0
        for variant in DPMD_VARIANTS:
            src = DPMD_SOURCE / variant / f"{variant}__{env}__evalN32_raw_10seeds.csv"
            if src.exists():
                a = pd.read_csv(src)
                rows += len(a)
                ret += a.episode_return.sum()
        b = pd.read_csv(dst)
        ok &= _cmp(f"DPMD {env}", rows, len(b), ret, b.episode_return.sum())

    for root, variant in SAC_RUNS:
        sac_dirs = _sac_seed_dirs(root)
        for env in ENVS:
            dst = out_dir / "SAC" / f"SAC__{env}__eval_episodes.csv"
            if not dst.exists():
                continue
            rows = 0
            ret = 0.0
            for _seed, run_dir in sac_dirs.get(env, []):
                for name in ("eval_episode_returns.csv", "eval_mean_episode_returns.csv"):
                    p = run_dir / name
                    if p.exists():
                        a = pd.read_csv(p)
                        rows += len(a)
                        ret += a.episode_return.sum()
            b = pd.read_csv(dst)
            b = b[b["variant"] == variant]
            ok &= _cmp(f"SAC {variant} {env}", rows, len(b), ret, b.episode_return.sum())

    for sweep, tag, variant in MGMD_RUNS:
        jobs = _mgmd_jobs(sweep, tag)
        for env in ENVS:
            dst = out_dir / "MGMD" / f"MGMD__{env}__eval_episodes.csv"
            if env not in jobs or not dst.exists():
                continue
            csv_path, slot_seed = jobs[env]
            a = pd.read_csv(csv_path)
            a = a[a["seed_index"].isin(slot_seed)]
            b = pd.read_csv(dst)
            b = b[b["variant"] == variant]
            ok &= _cmp(f"MGMD {variant} {env}", len(a), len(b),
                       a.episode_return.sum(), b.episode_return.sum())

    print("verification PASSED" if ok else "verification FAILED")
    if not ok:
        sys.exit(1)


def _cmp(label: str, n_src: int, n_dst: int, sum_src: float, sum_dst: float) -> bool:
    same = n_src == n_dst and np.isclose(sum_src, sum_dst, rtol=0, atol=1e-6)
    print(f"  {'ok ' if same else 'BAD'} {label}: rows {n_src}=={n_dst}, "
          f"return sum {sum_src:.6f} vs {sum_dst:.6f}")
    return same


if __name__ == "__main__":
    main()
