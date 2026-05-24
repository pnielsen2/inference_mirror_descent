#!/usr/bin/env python3
"""SLURM launcher for R_d quasirandom hyperparameter sweeps.

Reads a sweep YAML (see sweeps/rd_sweep/sweep_configs/example.yaml), generates n points of
an R_d sequence (one dimension per top-level ablation), maps each point through
the level bins (recursively for sub-ablations), and submits one SLURM job per
point. Batches are persisted under ``sweeps/rd_sweep/batches/{batch_id}/`` with
a manifest.yaml; ``batch_id`` is auto-assigned to the smallest unused integer
unless ``--batch-id`` is provided.

Each run is tagged with ``--suffix rdsweep_b{B}_r{i}``; this suffix is stored
in the wandb run config (via train_mujoco's args_dict), so the rendering
script can find all runs belonging to a batch.

Usage:
    python scripts/launch_rd.py sweeps/rd_sweep/sweep_configs/example.yaml --n 40
    python scripts/launch_rd.py sweep.yaml --n 40 --dry-run
    python scripts/launch_rd.py sweep.yaml --n 40 --seas -t 0-12:00
"""

import argparse
import json
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

# Reuse the R_d + sweep parsing already built in sweeps/rd_sweep/rd_sweep.py.
PROJECT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_DIR / "sweeps" / "rd_sweep"))
import rd_sweep  # noqa: E402


VENV_BY_ENV_VERSION = {
    3: Path.home() / ".venvs" / "v3",
    4: Path.home() / ".venvs" / "general",
    5: Path.home() / ".venvs" / "general",
}


# Pack keys match the argparse attribute names of the underlying CLI flag, so
# the per-slot values logged to wandb share the same name users see on the CLI.
# train_mujoco.py translates any pack keys that differ from the internal
# Diffv2TrainState field name via its own _CLI_TO_FIELD map.
FLAG_TO_HP_KEY = {f"--{k}": k for k in (
    "lr_q",
    "lr_policy",
    "gamma",
    "polyak_tau",
    "advantage_ema_tau",
    "shape_ema_tau",
    "guidance_strength_multiplier",
    "kl_budget",
    "reward_scale",
    "x0_hat_clip_radius",
    "mala_adapt_rate",
    "q_td_huber_width",
    "q_critic_agg_idx",
)}


def _level_to_hp(ab, lvl):
    """Validate that an easy-ablation level emits a single mappable flag and
    return (hp_key, float_value)."""
    tokens = shlex.split(lvl.get("flags", "") or "")
    if len(tokens) != 2 or not tokens[0].startswith("--"):
        raise ValueError(
            f"easy ablation {ab['name']!r} level {lvl['name']!r}: flags must be "
            f"exactly '--<flag> <value>' (two tokens), got {tokens!r}"
        )
    flag, val = tokens
    if flag not in FLAG_TO_HP_KEY:
        raise ValueError(
            f"easy ablation {ab['name']!r}: flag {flag!r} has no hp_pack mapping. "
            f"Known easy flags: {sorted(FLAG_TO_HP_KEY)}"
        )
    try:
        return FLAG_TO_HP_KEY[flag], float(val)
    except ValueError:
        raise ValueError(
            f"easy ablation {ab['name']!r} level {lvl['name']!r}: value {val!r} "
            "is not a float"
        )


def split_run_flags(run, ablations):
    """Walk run['path'] following the actual tree structure and return
    (hard_flag_tokens, {hp_key: float_value}).

    Hard-ablation flags are accumulated verbatim (in tree-DFS order). Easy
    ablations' levels must emit exactly ``--<flag> <value>`` with flag in
    FLAG_TO_HP_KEY; the value is parsed as float and keyed by hp_pack key.

    We walk the tree rather than searching by name so that duplicate ablation
    names in different subtrees are disambiguated by path position.
    """
    hard = []
    easy = {}
    path = [(a, l) for a, l in run["path"]]  # shallow copy we can pop-front

    def _walk(siblings):
        while path:
            ab_name, lvl_name = path[0]
            ab = next((a for a in siblings if a["name"] == ab_name), None)
            if ab is None:
                return  # this path entry belongs to an outer/parent scope
            path.pop(0)
            lvl = next((l for l in ab["levels"] if l["name"] == lvl_name), None)
            if lvl is None:
                raise RuntimeError(
                    f"level {lvl_name!r} not found under ablation {ab_name!r}"
                )
            if ab.get("easy"):
                hp_key, hp_val = _level_to_hp(ab, lvl)
                easy[hp_key] = hp_val
            else:
                hard.extend(shlex.split(lvl.get("flags", "") or ""))
            _walk(lvl["ablations"])

    _walk(ablations)
    if path:
        raise RuntimeError(f"path entries unconsumed after tree walk: {path}")
    return tuple(hard), easy


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH -p {partition}
#SBATCH --account={account}
#SBATCH --gres={gres}
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH -t {time}
#SBATCH -o {log_dir}/%j.out
#SBATCH -e {log_dir}/%j.err
#SBATCH --job-name={job_name}
{requeue_directives}
export PATH="{venv_path}/bin:$PATH"
export VIRTUAL_ENV="{venv_path}"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/lib64:${{LD_LIBRARY_PATH:-}}"
export CPATH="$HOME/.local/glew/glew-2.1.0/include:${{CPATH:-}}"
{jax_env}
{mps_block}
cd {project_dir}

echo "Job ID: $SLURM_JOB_ID"
echo "Venv: $VIRTUAL_ENV"
echo "Batch: {batch_id}  Runs: {run_idx}"
echo "Started at: $(date)"

{cmd_block}

wait
echo "Finished at: $(date)"
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    p.add_argument("sweep", type=Path, help="Sweep YAML file")
    p.add_argument("--n", type=int, required=True, help="Number of runs (R_d points)")
    p.add_argument("--rd-seed", type=float, default=0.5,
                   help="R_d sequence offset in [0,1). Default 0.5.")
    p.add_argument("--batch-id", type=int, default=None,
                   help="Batch id (default: smallest unused under sweeps/rd_sweep/batches/)")
    p.add_argument("--seed-base", type=int, default=100,
                   help="Run seed is seed_base + run_idx (unless --seed already in base). Default 100.")
    p.add_argument("--batches-dir", type=Path,
                   default=PROJECT_DIR / "sweeps" / "rd_sweep" / "batches",
                   help="Where to store batch manifests and slurm scripts.")

    p.add_argument("--seas", action="store_true", help="Use SEAS cluster")
    p.add_argument("--partition", "-p", type=str, default=None)
    p.add_argument("--gpu-type", type=str, default=None)
    p.add_argument("--account", type=str, default=None)
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--cpus", "-c", type=int, default=2)
    p.add_argument("--mem", type=str, default="32G")
    p.add_argument("--time", "-t", type=str, default="0-16:00")
    p.add_argument("--requeue", action="store_true",
                   help="Use kempner_requeue partition with requeue directives")
    p.add_argument("--runs-per-gpu", type=int, default=1,
                   help="Pack K runs into one SLURM job sharing one GPU via MPS / "
                        "multi-process. Mutually exclusive with --vmap-per-gpu. "
                        "Default 1 (one run per job).")
    p.add_argument("--vmap-per-gpu", type=int, default=1,
                   help="Pack up to N runs that differ only in 'easy' ablations "
                        "(ones marked 'easy: true' in the sweep YAML, whose per-seed "
                        "hp override is vmap-packable) into a single SLURM job using "
                        "--parallel_runs N --hp_pack <json>. Runs that differ in any "
                        "non-easy ('hard') flag are placed in separate jobs. Default 1.")
    p.add_argument("--mps", dest="mps", action="store_true", default=True,
                   help="Start an NVIDIA MPS daemon in the job so the K chunk-mates share "
                        "SMs spatially instead of time-slicing. Default: on when K>1.")
    p.add_argument("--no-mps", dest="mps", action="store_false",
                   help="Disable the MPS daemon (use plain time-slicing).")

    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--job-name", type=str, default=None)

    args = p.parse_args()
    if args.seas:
        args.partition = args.partition or "seas_gpu"
        args.gpu_type = args.gpu_type or "nvidia_h200"
        args.account = args.account or "kdbrantley_lab"
    elif args.requeue:
        args.partition = args.partition or "kempner_requeue"
        args.gpu_type = args.gpu_type or "nvidia_h100_80gb_hbm3"
        args.account = args.account or "kempner_kdbrantley_lab"
    else:
        args.partition = args.partition or "kempner_h100"
        args.gpu_type = args.gpu_type or "nvidia_h100_80gb_hbm3"
        args.account = args.account or "kempner_kdbrantley_lab"
    return args


def next_batch_id(batches_dir: Path) -> int:
    """Smallest positive integer not used as a subdirectory of batches_dir."""
    if not batches_dir.exists():
        return 1
    used = set()
    for child in batches_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            used.add(int(child.name))
    i = 1
    while i in used:
        i += 1
    return i


def venv_for_command(cmd_tokens):
    cmd = " ".join(cmd_tokens)
    m = re.search(r'--env\s+(\S+)', cmd)
    if not m:
        sys.exit(f"launch_rd.py: command has no --env flag:\n  {cmd}")
    env_name = m.group(1)
    vm = re.search(r'-v(\d+)$', env_name)
    if not vm:
        sys.exit(f"launch_rd.py: env '{env_name}' has no -v<N> suffix")
    version = int(vm.group(1))
    if version not in VENV_BY_ENV_VERSION:
        sys.exit(f"launch_rd.py: no venv for env version v{version}")
    vp = VENV_BY_ENV_VERSION[version].resolve()
    if not (vp / "bin" / "python").exists():
        sys.exit(f"launch_rd.py: venv python missing: {vp / 'bin' / 'python'}")
    return vp


def infer_job_name(cmd_tokens):
    cmd = " ".join(cmd_tokens)
    alg = re.search(r'--alg\s+(\S+)', cmd)
    env = re.search(r'--env\s+(\S+)', cmd)
    parts = []
    if alg:
        parts.append(alg.group(1))
    if env:
        parts.append(''.join(c for c in env.group(1) if c.isupper() or c.isdigit()))
    return "_".join(parts) if parts else "rd"


def has_flag(tokens, flag):
    return any(t == flag for t in tokens)


def main():
    args = parse_args()

    if not args.sweep.exists():
        fallback = PROJECT_DIR / "sweeps" / "rd_sweep" / "sweep_configs" / args.sweep.name
        if fallback.exists():
            args.sweep = fallback
    sweep = rd_sweep.load_sweep(args.sweep)
    d = sweep["dim"]
    if d == 0:
        sys.exit("launch_rd.py: sweep has no ablations")

    batches_dir = args.batches_dir
    batches_dir.mkdir(parents=True, exist_ok=True)

    if args.batch_id is None:
        batch_id = next_batch_id(batches_dir)
    else:
        batch_id = args.batch_id
        if (batches_dir / str(batch_id)).exists():
            sys.exit(f"launch_rd.py: batch {batch_id} already exists at {batches_dir / str(batch_id)}")

    batch_dir = batches_dir / str(batch_id)
    slurm_dir = batch_dir / "slurm"
    scripts_dir = batch_dir / "scripts"
    if not args.dry_run:
        batch_dir.mkdir(parents=True, exist_ok=True)
        slurm_dir.mkdir(parents=True, exist_ok=True)
        scripts_dir.mkdir(parents=True, exist_ok=True)

    generated = rd_sweep.generate_runs(sweep, args.n, seed=args.rd_seed)

    with open(args.sweep) as f:
        sweep_raw = f.read()

    base_in_seed = has_flag(sweep["base"], "--seed")

    M = max(1, args.runs_per_gpu)  # MPS (multi-process) packing
    V = max(1, args.vmap_per_gpu)  # vmap (in-process) packing
    if M > 1 and V > 1:
        sys.exit("launch_rd.py: --runs-per-gpu and --vmap-per-gpu are mutually "
                 "exclusive (pick one packing strategy).")

    # Build the canonical per-run record. When V==1 we keep the legacy
    # suffix/tokens scheme (one wandb run per sweep point). When V>1 we also
    # decompose each run's flags into (hard, easy) for later packing.
    runs = []
    for i, gr in enumerate(generated):
        tokens = list(sweep["base"]) + list(gr["flags"])
        suffix = f"rdsweep_b{batch_id}_r{i}"
        tokens += ["--suffix", suffix]
        if not base_in_seed:
            tokens += ["--seed", str(args.seed_base + i)]
        rec = {
            "run_idx": i,
            "suffix": suffix,
            "point": [float(u) for u in gr["point"]],
            "path": [{"ablation": a, "level": l} for a, l in gr["path"]],
            "cmd": " ".join(shlex.quote(t) for t in tokens),
            "tokens": tokens,
        }
        if V > 1:
            hard, easy = split_run_flags(gr, sweep["ablations"])
            rec["hard_flags"] = list(hard)
            rec["easy_hp"] = easy
        runs.append(rec)

    # When vmap-packing, group runs by their hard-flag signature and chunk
    # each group into packs of at most V runs.
    packs = []  # each: {"pack_idx", "suffix", "run_ids", "hard_flags", "hp_pack", "hp_pack_path"}
    if V > 1:
        groups = defaultdict(list)
        for r in runs:
            groups[tuple(r["hard_flags"])].append(r["run_idx"])
        # Stable ordering: groups by the smallest run_idx they contain; within
        # a group, ascending run_idx.
        ordered = sorted(groups.items(), key=lambda kv: min(kv[1]))
        hp_packs_dir = batch_dir / "hp_packs"
        for hard_sig, run_ids in ordered:
            run_ids = sorted(run_ids)
            for start in range(0, len(run_ids), V):
                chunk = run_ids[start:start + V]
                # Collect easy hp assignments for each run in this pack. All
                # runs in a hard-signature group share the same set of easy
                # keys (because easy ablations appear at the same positions in
                # the tree conditional on hard choices); we verify that below.
                easy_dicts = [runs[rid]["easy_hp"] for rid in chunk]
                keys_union = set().union(*(ed.keys() for ed in easy_dicts))
                hp_pack = {}
                for k in sorted(keys_union):
                    col = []
                    for rid, ed in zip(chunk, easy_dicts):
                        if k not in ed:
                            raise RuntimeError(
                                f"internal: pack with hard signature {hard_sig!r} "
                                f"has run {rid} missing easy key {k!r}; "
                                f"easy-ablation layout is inconsistent within a "
                                f"hard-signature group"
                            )
                        col.append(ed[k])
                    hp_pack[k] = col
                p_idx = len(packs)
                hp_pack_path = hp_packs_dir / f"pack_{p_idx}.json" if hp_pack else None
                packs.append({
                    "pack_idx": p_idx,
                    "suffix": f"rdsweep_b{batch_id}_p{p_idx}",
                    "run_ids": chunk,
                    "hard_flags": list(hard_sig),
                    "hp_pack": hp_pack,
                    "hp_pack_path": str(hp_pack_path) if hp_pack_path else None,
                })

    manifest = {
        "batch_id": batch_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "n": args.n,
        "rd_seed": args.rd_seed,
        "seed_base": args.seed_base,
        "sweep_file": str(args.sweep.resolve()),
        "sweep": sweep,
        "sweep_raw": sweep_raw,
        "runs": [{k: v for k, v in r.items() if k != "tokens"} for r in runs],
        "runs_per_gpu": M,
        "vmap_per_gpu": V,
    }
    if V > 1:
        manifest["packs"] = packs
    manifest_path = batch_dir / "manifest.yaml"
    if not args.dry_run:
        with open(manifest_path, "w") as f:
            yaml.safe_dump(manifest, f, sort_keys=False)
        if V > 1 and any(p["hp_pack"] for p in packs):
            hp_packs_dir = batch_dir / "hp_packs"
            hp_packs_dir.mkdir(parents=True, exist_ok=True)
            for pk in packs:
                if pk["hp_pack"]:
                    with open(pk["hp_pack_path"], "w") as f:
                        json.dump(pk["hp_pack"], f, indent=2)

    job_name_base = args.job_name or infer_job_name(sweep["base"])

    print(f"Batch {batch_id}  n={args.n}  d={d}  dir={batch_dir}")
    print(f"Manifest: {manifest_path}")
    if V > 1:
        print(f"vmap packing: n={args.n} runs -> {len(packs)} packs "
              f"(vmap-per-gpu={V}, {len(set(tuple(r['hard_flags']) for r in runs))} "
              f"hard-flag groups)")
    print()

    if V > 1:
        submitted = _submit_vmap_packs(args, sweep, packs, runs, batch_id,
                                       scripts_dir, slurm_dir,
                                       job_name_base, base_in_seed)
    else:
        submitted = _submit_mps_chunks(args, runs, batch_id, scripts_dir,
                                       slurm_dir, job_name_base, M)

    if not args.dry_run:
        print()
        print(f"Submitted {submitted}/{args.n} runs")
        print(f"Slurm logs: {slurm_dir}")


def _submit_mps_chunks(args, runs, batch_id, scripts_dir, slurm_dir,
                       job_name_base, K):
    """Legacy path: one sbatch per K-sized chunk of runs (K=1 => one run per
    job; K>1 => MPS multi-process packing)."""
    chunks = [runs[i:i + K] for i in range(0, len(runs), K)]
    submitted = 0
    for chunk_idx, chunk in enumerate(chunks):
        venvs = {venv_for_command(r["tokens"]) for r in chunk}
        if len(venvs) > 1:
            sys.exit(f"launch_rd.py: chunk {chunk_idx} mixes venvs {venvs}; "
                     "use --runs-per-gpu 1 or homogenize environments.")
        venv_path = next(iter(venvs))

        run_ids = [r["run_idx"] for r in chunk]
        job_name = f"{job_name_base}_b{batch_id}_c{chunk_idx}"

        if args.requeue:
            requeue_directives = "#SBATCH --requeue\n#SBATCH --signal=B:SIGTERM@120"
        else:
            requeue_directives = ""

        gres = f"gpu:{args.gpu_type}:{args.num_gpus}"
        if K > 1:
            jax_env = (
                "export XLA_PYTHON_CLIENT_PREALLOCATE=false\n"
                f"export XLA_PYTHON_CLIENT_MEM_FRACTION={1.0 / K:.4f}"
            )
            if args.mps:
                mps_block = (
                    'export CUDA_MPS_PIPE_DIRECTORY="${SLURM_TMPDIR:-/tmp/$USER-$SLURM_JOB_ID}/nvidia-mps"\n'
                    'export CUDA_MPS_LOG_DIRECTORY="${SLURM_TMPDIR:-/tmp/$USER-$SLURM_JOB_ID}/nvidia-mps-log"\n'
                    'mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"\n'
                    'nvidia-cuda-mps-control -d && echo "[mps] daemon started" || echo "[mps] daemon FAILED to start"\n'
                    'trap "echo quit | nvidia-cuda-mps-control 2>/dev/null || true" EXIT'
                )
            else:
                mps_block = 'echo "[mps] disabled (--no-mps)"'
        else:
            jax_env = ""
            mps_block = ""

        cmd_lines = []
        for r in chunk:
            cmd = " ".join(shlex.quote(t) for t in r["tokens"])
            cmd_lines.append(f'echo "[r{r["run_idx"]}] {cmd}"')
            cmd_lines.append(
                f'srun --exact --overlap -n 1 -c $SLURM_CPUS_PER_TASK '
                f'--output="{slurm_dir}/%j.r{r["run_idx"]}.out" '
                f'--error="{slurm_dir}/%j.r{r["run_idx"]}.err" '
                f'bash -c {shlex.quote(cmd)} &'
            )
        cmd_block = "\n".join(cmd_lines)

        script = SBATCH_TEMPLATE.format(
            partition=args.partition,
            account=args.account,
            gres=gres,
            ntasks=len(chunk),
            cpus_per_task=args.cpus,
            mem=args.mem,
            time=args.time,
            log_dir=slurm_dir,
            job_name=job_name,
            project_dir=PROJECT_DIR,
            cmd_block=cmd_block,
            requeue_directives=requeue_directives,
            jax_env=jax_env,
            venv_path=venv_path,
            batch_id=batch_id,
            run_idx=",".join(str(i) for i in run_ids),
            mps_block=mps_block,
        )

        tags = [f"r{r['run_idx']}:" + "/".join(f"{p['ablation']}={p['level']}" for p in r["path"])
                for r in chunk]
        if args.dry_run:
            print(f"[chunk {chunk_idx}] runs={run_ids}  [venv={venv_path.name}]")
            for t in tags:
                print(f"  {t}")
            continue

        script_path = scripts_dir / f"{job_name}.sh"
        script_path.write_text(script)
        result = subprocess.run(["sbatch", str(script_path)],
                                capture_output=True, text=True)
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"[chunk {chunk_idx}] {job_id}  runs={run_ids}")
            submitted += len(chunk)
        else:
            print(f"[chunk {chunk_idx}] FAILED  runs={run_ids}")
            print(f"  stderr: {result.stderr.strip()}")
    return submitted


def _submit_vmap_packs(args, sweep, packs, runs, batch_id, scripts_dir,
                       slurm_dir, job_name_base, base_in_seed):
    """One sbatch per vmap pack: a single python process runs
    --parallel_runs K --hp_pack <json> with the pack's hard flags."""
    submitted = 0
    runs_by_idx = {r["run_idx"]: r for r in runs}
    for pk in packs:
        k_seeds = len(pk["run_ids"])
        # Build tokens: base + hard + parallel_runs + hp_pack + suffix + seed
        tokens = list(sweep["base"]) + list(pk["hard_flags"])
        tokens += ["--parallel_runs", str(k_seeds)]
        if pk["hp_pack_path"] is not None:
            tokens += ["--hp_pack", pk["hp_pack_path"]]
        tokens += ["--suffix", pk["suffix"]]
        if not base_in_seed:
            # Use first run's idx so seeding stays deterministic and distinct
            # across packs.
            tokens += ["--seed", str(args.seed_base + pk["run_ids"][0])]
        cmd = " ".join(shlex.quote(t) for t in tokens)

        venv_path = venv_for_command(tokens)
        job_name = f"{job_name_base}_b{batch_id}_p{pk['pack_idx']}"

        if args.requeue:
            requeue_directives = "#SBATCH --requeue\n#SBATCH --signal=B:SIGTERM@120"
        else:
            requeue_directives = ""

        gres = f"gpu:{args.gpu_type}:{args.num_gpus}"
        jax_env = ""  # vmap is single-process; default JAX mem policy is fine
        mps_block = ""

        cmd_lines = [
            f'echo "[p{pk["pack_idx"]}] runs={pk["run_ids"]} k={k_seeds}"',
            f'echo "[p{pk["pack_idx"]}] {cmd}"',
            f'srun --exact --overlap -n 1 -c $SLURM_CPUS_PER_TASK '
            f'--output="{slurm_dir}/%j.p{pk["pack_idx"]}.out" '
            f'--error="{slurm_dir}/%j.p{pk["pack_idx"]}.err" '
            f'bash -c {shlex.quote(cmd)} &',
        ]
        cmd_block = "\n".join(cmd_lines)

        script = SBATCH_TEMPLATE.format(
            partition=args.partition,
            account=args.account,
            gres=gres,
            ntasks=1,
            cpus_per_task=args.cpus,
            mem=args.mem,
            time=args.time,
            log_dir=slurm_dir,
            job_name=job_name,
            project_dir=PROJECT_DIR,
            cmd_block=cmd_block,
            requeue_directives=requeue_directives,
            jax_env=jax_env,
            venv_path=venv_path,
            batch_id=batch_id,
            run_idx=",".join(str(i) for i in pk["run_ids"]),
            mps_block=mps_block,
        )

        if args.dry_run:
            print(f"[pack {pk['pack_idx']}] runs={pk['run_ids']}  "
                  f"k_seeds={k_seeds}  [venv={venv_path.name}]")
            for rid in pk["run_ids"]:
                r = runs_by_idx[rid]
                tag = "/".join(f"{p['ablation']}={p['level']}" for p in r["path"])
                ehp = runs_by_idx[rid].get("easy_hp", {})
                print(f"  r{rid}: {tag}  easy={ehp}")
            continue

        script_path = scripts_dir / f"{job_name}.sh"
        script_path.write_text(script)
        result = subprocess.run(["sbatch", str(script_path)],
                                capture_output=True, text=True)
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"[pack {pk['pack_idx']}] {job_id}  runs={pk['run_ids']}  k={k_seeds}")
            submitted += k_seeds
        else:
            print(f"[pack {pk['pack_idx']}] FAILED  runs={pk['run_ids']}")
            print(f"  stderr: {result.stderr.strip()}")
    return submitted


if __name__ == "__main__":
    main()
