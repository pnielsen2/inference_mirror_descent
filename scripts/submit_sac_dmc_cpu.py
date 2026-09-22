#!/usr/bin/env python3
"""Launch CleanRL SAC on the DeepMind Control envs, on CPU, one job per run.

SAC baselines for the dm_control tasks, matching the evaluation protocol of the
MGMD sweeps so the curves are directly comparable: ``--eval-episodes 10`` every
``--eval-every`` env steps, executing the best of ``--eval-best-of-n`` candidate
actions scored by the twin critics.

Why a separate script from ``submit_mujoco_v5_standard_baselines.py``: that one
hardcodes ``--gres=<h100>`` on every sbatch it writes and its env list to the six
``-v5`` MuJoCo tasks. These runs are CPU-only and on ``dmc/`` ids, so rather than
thread a CPU/DMC mode through it (and risk disturbing the existing baselines),
this submits them directly.

Jobs are submitted in SEED-MAJOR order: every env at seed 0, then every env at
seed 1. SLURM dispatches roughly in submission order, so the first seed of each
env starts before the second seed of any env.

Examples
--------
    # Inspect what would be submitted
    python scripts/submit_sac_dmc_cpu.py --dry-run

    # Submit 6 runs (3 envs x seeds 0-1)
    python scripts/submit_sac_dmc_cpu.py
"""
import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ENVS = [
    "dmc/humanoid-walk-v0",
    "dmc/humanoid_CMU-run-v0",
    "dmc/dog-run-v0",
]

# Home is full, so runs land on netscratch like the other baseline launchers.
DEFAULT_ROOT = Path("/n/netscratch/kdbrantley_lab/Lab/pnielsen/sac_dmc_baseline")
VENV = Path.home() / ".venvs" / "general"

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH -p {partition}
#SBATCH -c {cpus}
#SBATCH --mem={mem}
#SBATCH -t {time}
#SBATCH -o {slurm_dir}/%j.out
#SBATCH -e {slurm_dir}/%j.err
#SBATCH --job-name={job_name}
set -euo pipefail

export PATH="{venv}/bin:$PATH"
export VIRTUAL_ENV="{venv}"

# Torch on CPU otherwise grabs every core on the node and thrashes against the
# other jobs sharing it; hold it to our own allocation.
export OMP_NUM_THREADS={cpus}
export MKL_NUM_THREADS={cpus}

# dm_control defaults to a glfw context and warns about a missing X11 DISPLAY on
# every env construction. Nothing here renders, so turn the context off.
export MUJOCO_GL=disable

# Run from the log dir so CleanRL's runs/ tensorboard tree lands on netscratch
# rather than in the repo; the script resolves its own imports off __file__.
mkdir -p "{run_dir}"
cd "{run_dir}"

echo "Job ID: $SLURM_JOB_ID"
echo "Env: {env}   Seed: {seed}"
echo "Codebase: {codebase}"
echo "Run dir: {run_dir}"
echo "Started at: $(date)"

python {codebase}/scripts/cleanrl/sac_continuous_action.py \\
    --env-id {env} \\
    --seed {seed} \\
    --no-cuda \\
    --no-track \\
    --total-timesteps {total_timesteps} \\
    --eval-every {eval_every} \\
    --eval-episodes {eval_episodes} \\
    --eval-best-of-n {eval_best_of_n} \\
    --eval-q-agg {eval_q_agg} \\
    --eval-csv-output "{run_dir}/eval_episode_returns.csv"

echo "Finished at: $(date)"
"""


def short_env_name(env: str) -> str:
    """Short, path-safe label for an env id.

    Interpolated into the sbatch script path and the run dir, so it must not
    contain a slash -- which ``dmc/`` ids do. Every dmc task also ends in
    ``-v0``, so the domain alone would collapse tasks together.
    """
    if env.startswith("dmc/"):
        task = env[len("dmc/"):]
        if task.endswith("-v0"):
            task = task[: -len("-v0")]
        return f"dmc_{task}"
    return env.split("-")[0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--envs", nargs="+", default=ENVS)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    p.add_argument("--total-timesteps", dest="total_timesteps", type=int,
                   default=int(1e6), help="Env steps per run.")

    p.add_argument("--eval-every", dest="eval_every", type=int, default=10000)
    p.add_argument("--eval-episodes", dest="eval_episodes", type=int, default=10)
    p.add_argument("--eval-best-of-n", dest="eval_best_of_n", type=int, default=32)
    p.add_argument("--eval-q-agg", dest="eval_q_agg", type=str, default="mean",
                   choices=["mean", "min"],
                   help="Twin-critic aggregation when scoring eval candidates. "
                        "'mean' matches the MGMD sweeps' --q_agg_sample mean, so "
                        "the best-of-N curves are measured the same way; 'min' is "
                        "SAC's own pessimistic value, as used in its actor loss.")

    p.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                   help="Netscratch root holding runs/ and launches/.")
    p.add_argument("--partition", type=str, default="shared",
                   help="CPU partition. 'shared' and 'sapphire' are non-preemptible; "
                        "'serial_requeue' is cheaper but can be requeued, and these "
                        "runs do not checkpoint.")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--mem", type=str, default="16G")
    p.add_argument("--time", type=str, default="1-00:00")

    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    p.add_argument("--no-snapshot", dest="no_snapshot", action="store_true",
                   help="Run from the live repo instead of an rsync snapshot.")
    return p.parse_args()


def snapshot_codebase(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "rsync", "-a", "--delete",
            "--exclude", ".git/",
            "--exclude", "logs/",
            "--exclude", "__pycache__/",
            "--exclude", "*.egg-info/",
            "--exclude", "wandb/",
            "--exclude", "runs/",
            f"{REPO}/", f"{dest}/",
        ],
        check=True,
    )
    return dest


def main():
    args = parse_args()

    if shutil.which("sbatch") is None and not args.dry_run:
        sys.exit("sbatch not found; run on a submit host or use --dry-run.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    launch_dir = args.root / "launches" / stamp
    slurm_dir = launch_dir / "slurm"

    if not args.dry_run:
        slurm_dir.mkdir(parents=True, exist_ok=True)

    if args.no_snapshot:
        codebase = REPO
    else:
        codebase = launch_dir / "codebase"
        if args.dry_run:
            print(f"[dry-run] would snapshot {REPO} -> {codebase}")
        else:
            snapshot_codebase(codebase)

    # Seed-major: every env at seed s before any env at seed s+1.
    jobs = [(seed, env) for seed in args.seeds for env in args.envs]

    print(f"SAC (CPU)  envs={len(args.envs)}  seeds={len(args.seeds)}  "
          f"-> {len(jobs)} runs of {args.total_timesteps} env steps")
    print(f"Launch dir: {launch_dir}")
    print(f"Eval:       {args.eval_episodes} episodes every {args.eval_every} steps, "
          f"best-of-{args.eval_best_of_n} scored by {args.eval_q_agg}(Q1,Q2)")
    print(f"Resources:  CPU-only  -p {args.partition}  -c {args.cpus}  "
          f"--mem {args.mem}  -t {args.time}")
    print("Submission order (seed-major):")

    submitted = []
    for i, (seed, env) in enumerate(jobs, 1):
        short = short_env_name(env)
        job_name = f"sac_s{seed}_{short}"
        run_dir = args.root / "runs" / f"{short}_s{seed}_{stamp}"
        script = SBATCH_TEMPLATE.format(
            partition=args.partition,
            cpus=args.cpus,
            mem=args.mem,
            time=args.time,
            slurm_dir=slurm_dir,
            job_name=job_name,
            venv=VENV,
            codebase=codebase,
            run_dir=run_dir,
            env=env,
            seed=seed,
            total_timesteps=args.total_timesteps,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            eval_best_of_n=args.eval_best_of_n,
            eval_q_agg=args.eval_q_agg,
        )

        if args.dry_run:
            print(f"  [{i:>2}/{len(jobs)}] seed={seed:<2} {env:<26} -> {job_name}")
            continue

        script_path = slurm_dir / f"{job_name}.sh"
        script_path.write_text(script)
        script_path.chmod(0o755)
        out = subprocess.run(["sbatch", str(script_path)],
                             capture_output=True, text=True)
        if out.returncode != 0:
            print(f"  [{i:>2}/{len(jobs)}] FAILED seed={seed} {env}: {out.stderr.strip()}")
            continue
        job_id = out.stdout.strip().split()[-1]
        submitted.append((job_id, seed, env))
        print(f"  [{i:>2}/{len(jobs)}] seed={seed:<2} {env:<26} -> job {job_id}")

    if args.dry_run:
        print("\n[dry-run] nothing submitted.")
        return

    (launch_dir / "submitted.txt").write_text(
        "\n".join(f"{j}\t{s}\t{e}" for j, s, e in submitted) + "\n"
    )
    print(f"\nSubmitted {len(submitted)}/{len(jobs)} jobs.")
    print(f"Record: {launch_dir / 'submitted.txt'}")


if __name__ == "__main__":
    main()
