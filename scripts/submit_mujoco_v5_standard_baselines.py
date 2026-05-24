from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

CLEANRL_WORKTREE = Path("/n/home09/pnielsen/inference_mirror_descent")
MGMD_WORKTREE = Path("/n/home09/pnielsen/diffusion_policy_online_rl_baseline_a50cf41")
PYTHON = Path("/n/home09/pnielsen/.venvs/general/bin/python")
SCRATCH_PARENT = Path("/n/netscratch/kdbrantley_lab/Lab/pnielsen/mujoco_v5_standard_baselines")
ACCOUNT = "kempner_kdbrantley_lab"
PARTITION = "kempner_h100"
GPU_RESOURCE = "gpu:nvidia_h100_80gb_hbm3:1"
ENVS = (
    ("Humanoid-v5", "hum"),
    ("Ant-v5", "ant"),
    ("Walker2d-v5", "walker"),
    ("Hopper-v5", "hopper"),
    ("HalfCheetah-v5", "hc"),
    ("Swimmer-v5", "swim"),
)
SEEDS = tuple(range(10))


@dataclass(frozen=True)
class AlgoSpec:
    key: str
    worktree: Path
    script_relpath: str
    env_flag: str
    cpus: int
    memory: str
    walltime: str


@dataclass(frozen=True)
class TaskSpec:
    task_name: str
    algorithm: str
    env_name: str
    env_slug: str
    seed0: int
    seed1: int


ALGO_SPECS = {
    "mgmd": AlgoSpec(
        key="mgmd",
        worktree=MGMD_WORKTREE,
        script_relpath="scripts/train_mujoco.py",
        env_flag="--env",
        cpus=24,
        memory="64G",
        walltime="12:00:00",
    ),
    "sac": AlgoSpec(
        key="sac",
        worktree=CLEANRL_WORKTREE,
        script_relpath="scripts/cleanrl/sac_continuous_action.py",
        env_flag="--env-id",
        cpus=8,
        memory="32G",
        walltime="12:00:00",
    ),
    "td3": AlgoSpec(
        key="td3",
        worktree=CLEANRL_WORKTREE,
        script_relpath="scripts/cleanrl/td3_continuous_action.py",
        env_flag="--env-id",
        cpus=8,
        memory="32G",
        walltime="12:00:00",
    ),
    "ppo": AlgoSpec(
        key="ppo",
        worktree=CLEANRL_WORKTREE,
        script_relpath="scripts/cleanrl/ppo_continuous_action.py",
        env_flag="--env-id",
        cpus=8,
        memory="32G",
        walltime="12:00:00",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--algos", nargs="+", choices=tuple(ALGO_SPECS), default=list(ALGO_SPECS))
    parser.add_argument("--submission-mode", choices=("array", "individual"), default="array")
    parser.add_argument("--max-concurrent", type=int, default=16)
    parser.add_argument("--total-timesteps", type=int, default=int(1e6))
    parser.add_argument("--mgmd-start-step", type=int, default=int(3e4))
    parser.add_argument("--mgmd-num-vec-envs", type=int, default=5)
    parser.add_argument("--mgmd-mem-fraction", default="0.04")
    return parser.parse_args()


def default_root() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SCRATCH_PARENT / timestamp


def shell_quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def build_tasks(algorithm: str) -> list[TaskSpec]:
    tasks: list[TaskSpec] = []
    for env_name, env_slug in ENVS:
        for index in range(0, len(SEEDS), 2):
            seed0 = SEEDS[index]
            seed1 = SEEDS[index + 1]
            tasks.append(
                TaskSpec(
                    task_name=f"{algorithm}_full_{env_slug}_s{seed0:02d}s{seed1:02d}",
                    algorithm=algorithm,
                    env_name=env_name,
                    env_slug=env_slug,
                    seed0=seed0,
                    seed1=seed1,
                )
            )
    return tasks


def tasks_tsv(tasks: list[TaskSpec]) -> str:
    lines = ["task_name\talgorithm\tenv_name\tenv_slug\tseed0\tseed1"]
    for task in tasks:
        lines.append(
            f"{task.task_name}\t{task.algorithm}\t{task.env_name}\t{task.env_slug}\t{task.seed0}\t{task.seed1}"
        )
    return "\n".join(lines) + "\n"


def cleanrl_sbatch_text(spec: AlgoSpec, tasks: list[TaskSpec], root: Path, args: argparse.Namespace) -> str:
    slurm_dir = root / "slurm"
    script_path = spec.worktree / spec.script_relpath
    lines = [
        "#!/bin/bash",
        f"#SBATCH -A {ACCOUNT}",
        f"#SBATCH -p {PARTITION}",
        f"#SBATCH -J {spec.key}_v5_pack2",
        f"#SBATCH -t {spec.walltime}",
        "#SBATCH -N 1",
        f"#SBATCH -c {spec.cpus}",
        f"#SBATCH --mem={spec.memory}",
        f"#SBATCH --gres={GPU_RESOURCE}",
        f"#SBATCH --array=0-{len(tasks) - 1}%{args.max_concurrent}",
        f"#SBATCH -o {slurm_dir / (spec.key + '_v5_pack2_%A_%a.out')}",
        f"#SBATCH -e {slurm_dir / (spec.key + '_v5_pack2_%A_%a.err')}",
        "set -euo pipefail",
        f"PYTHON={shell_quote(PYTHON)}",
        f"ROOT={shell_quote(root)}",
        f"TASK_FILE={shell_quote(root / 'tasks.tsv')}",
        f"SCRIPT_PATH={shell_quote(script_path)}",
        f"TOTAL_TIMESTEPS={args.total_timesteps}",
        f"SLURM_DIR={shell_quote(slurm_dir)}",
        'task_line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$TASK_FILE")',
        'IFS=$'"'"'\t'"'"' read -r TASK_NAME ALGORITHM ENV_NAME ENV_SLUG SEED0 SEED1 <<< "$task_line"',
        'TASK_ROOT="$ROOT/runs/$TASK_NAME"',
        'mkdir -p "$TASK_ROOT" "$SLURM_DIR"',
        'export OMP_NUM_THREADS=1',
        'export MKL_NUM_THREADS=1',
        'launch_seed() {',
        '  local seed="$1"',
        '  local run_root="$TASK_ROOT/s$seed"',
        '  mkdir -p "$run_root"',
        '  (',
        '    cd "$run_root"',
        '    "$PYTHON" "$SCRIPT_PATH" \\',
        f'      {spec.env_flag} "$ENV_NAME" \\',
        '      --seed "$seed" \\',
        '      --num-envs 1 \\',
        '      --total-timesteps "$TOTAL_TIMESTEPS" \\',
        '      --no-track \\',
        '      --csv-output "$run_root/episode_returns.csv"',
        '  ) &',
        '  LAST_PID=$!',
        '}',
        'LAST_PID=""',
        'launch_seed "$SEED0"',
        'pid0="$LAST_PID"',
        'launch_seed "$SEED1"',
        'pid1="$LAST_PID"',
        'wait "$pid0"',
        'seed0_rc=$?',
        'wait "$pid1"',
        'seed1_rc=$?',
        'printf "seed0=%s seed0_rc=%s\n" "$SEED0" "$seed0_rc"',
        'printf "seed1=%s seed1_rc=%s\n" "$SEED1" "$seed1_rc"',
        'if [ "$seed0_rc" -ne 0 ] || [ "$seed1_rc" -ne 0 ]; then',
        '  exit 1',
        'fi',
        'exit 0',
    ]
    return "\n".join(lines) + "\n"


def mgmd_sbatch_text(spec: AlgoSpec, tasks: list[TaskSpec], root: Path, args: argparse.Namespace) -> str:
    slurm_dir = root / "slurm"
    lines = [
        "#!/bin/bash",
        f"#SBATCH -A {ACCOUNT}",
        f"#SBATCH -p {PARTITION}",
        f"#SBATCH -J {spec.key}_v5_pack2",
        f"#SBATCH -t {spec.walltime}",
        "#SBATCH -N 1",
        f"#SBATCH -c {spec.cpus}",
        f"#SBATCH --mem={spec.memory}",
        f"#SBATCH --gres={GPU_RESOURCE}",
        f"#SBATCH --array=0-{len(tasks) - 1}%{args.max_concurrent}",
        f"#SBATCH -o {slurm_dir / (spec.key + '_v5_pack2_%A_%a.out')}",
        f"#SBATCH -e {slurm_dir / (spec.key + '_v5_pack2_%A_%a.err')}",
        "set -euo pipefail",
        f"WORKTREE={shell_quote(spec.worktree)}",
        f"PYTHON={shell_quote(PYTHON)}",
        f"ROOT={shell_quote(root)}",
        f"TASK_FILE={shell_quote(root / 'tasks.tsv')}",
        f"NUM_VEC_ENVS={args.mgmd_num_vec_envs}",
        f"START_STEP={args.mgmd_start_step}",
        f"TOTAL_TIMESTEPS={args.total_timesteps}",
        f"MEM_FRACTION={shell_quote(args.mgmd_mem_fraction)}",
        f"SLURM_DIR={shell_quote(slurm_dir)}",
        'task_line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$TASK_FILE")',
        'IFS=$'"'"'\t'"'"' read -r TASK_NAME ALGORITHM ENV_NAME ENV_SLUG SEED0 SEED1 <<< "$task_line"',
        'TASK_ROOT="$ROOT/runs/$TASK_NAME"',
        'WAND_ROOT="$ROOT/wandb/$TASK_NAME"',
        'mkdir -p "$TASK_ROOT" "$WAND_ROOT" "$SLURM_DIR"',
        'export OMP_NUM_THREADS=1',
        'export XLA_FLAGS="--xla_gpu_deterministic_ops=true"',
        'unset XLA_PYTHON_CLIENT_PREALLOCATE',
        'export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION"',
        'launch_seed() {',
        '  local seed="$1"',
        '  local wandb_dir="$2"',
        '  mkdir -p "$wandb_dir"',
        '  (',
        '    cd "$WORKTREE"',
        '    WANDB_MODE=offline WANDB_DIR="$wandb_dir" "$PYTHON" scripts/train_mujoco.py \\',
        '      --alg mgmd \\',
        '      --env "$ENV_NAME" \\',
        '      --seed "$seed" \\',
        '      --num_vec_envs "$NUM_VEC_ENVS" \\',
        '      --start_step "$START_STEP" \\',
        '      --total_step "$TOTAL_TIMESTEPS" \\',
        '      --suffix "${TASK_NAME}_s${seed}" \\',
        '      --log_root "$TASK_ROOT"',
        '  ) &',
        '  LAST_PID=$!',
        '}',
        'LAST_PID=""',
        'launch_seed "$SEED0" "$WAND_ROOT/s$SEED0"',
        'pid0="$LAST_PID"',
        'launch_seed "$SEED1" "$WAND_ROOT/s$SEED1"',
        'pid1="$LAST_PID"',
        'wait "$pid0"',
        'seed0_rc=$?',
        'wait "$pid1"',
        'seed1_rc=$?',
        'printf "seed0=%s seed0_rc=%s\n" "$SEED0" "$seed0_rc"',
        'printf "seed1=%s seed1_rc=%s\n" "$SEED1" "$seed1_rc"',
        'if [ "$seed0_rc" -ne 0 ] || [ "$seed1_rc" -ne 0 ]; then',
        '  exit 1',
        'fi',
        'exit 0',
    ]
    return "\n".join(lines) + "\n"


def cleanrl_task_sbatch_text(spec: AlgoSpec, task: TaskSpec, root: Path, args: argparse.Namespace) -> str:
    slurm_dir = root / "slurm"
    script_path = spec.worktree / spec.script_relpath
    task_root = root / "runs" / task.task_name
    lines = [
        "#!/bin/bash",
        f"#SBATCH -A {ACCOUNT}",
        f"#SBATCH -p {PARTITION}",
        f"#SBATCH -J {task.task_name}",
        f"#SBATCH -t {spec.walltime}",
        "#SBATCH -N 1",
        f"#SBATCH -c {spec.cpus}",
        f"#SBATCH --mem={spec.memory}",
        f"#SBATCH --gres={GPU_RESOURCE}",
        f"#SBATCH -o {slurm_dir / (task.task_name + '_%j.out')}",
        f"#SBATCH -e {slurm_dir / (task.task_name + '_%j.err')}",
        "set -euo pipefail",
        f"PYTHON={shell_quote(PYTHON)}",
        f"SCRIPT_PATH={shell_quote(script_path)}",
        f"ENV_NAME={shell_quote(task.env_name)}",
        f"TASK_NAME={shell_quote(task.task_name)}",
        f"TASK_ROOT={shell_quote(task_root)}",
        f"SEED0={task.seed0}",
        f"SEED1={task.seed1}",
        f"TOTAL_TIMESTEPS={args.total_timesteps}",
        f"SLURM_DIR={shell_quote(slurm_dir)}",
        'mkdir -p "$TASK_ROOT" "$SLURM_DIR"',
        'export OMP_NUM_THREADS=1',
        'export MKL_NUM_THREADS=1',
        'launch_seed() {',
        '  local seed="$1"',
        '  local run_root="$TASK_ROOT/s$seed"',
        '  mkdir -p "$run_root"',
        '  (',
        '    cd "$run_root"',
        '    "$PYTHON" "$SCRIPT_PATH" \\',
        f'      {spec.env_flag} "$ENV_NAME" \\',
        '      --seed "$seed" \\',
        '      --num-envs 1 \\',
        '      --total-timesteps "$TOTAL_TIMESTEPS" \\',
        '      --no-track \\',
        '      --csv-output "$run_root/episode_returns.csv"',
        '  ) &',
        '  LAST_PID=$!',
        '}',
        'LAST_PID=""',
        'launch_seed "$SEED0"',
        'pid0="$LAST_PID"',
        'launch_seed "$SEED1"',
        'pid1="$LAST_PID"',
        'wait "$pid0"',
        'seed0_rc=$?',
        'wait "$pid1"',
        'seed1_rc=$?',
        'printf "seed0=%s seed0_rc=%s\n" "$SEED0" "$seed0_rc"',
        'printf "seed1=%s seed1_rc=%s\n" "$SEED1" "$seed1_rc"',
        'if [ "$seed0_rc" -ne 0 ] || [ "$seed1_rc" -ne 0 ]; then',
        '  exit 1',
        'fi',
        'exit 0',
    ]
    return "\n".join(lines) + "\n"


def mgmd_task_sbatch_text(spec: AlgoSpec, task: TaskSpec, root: Path, args: argparse.Namespace) -> str:
    slurm_dir = root / "slurm"
    task_root = root / "runs" / task.task_name
    wandb_root = root / "wandb" / task.task_name
    lines = [
        "#!/bin/bash",
        f"#SBATCH -A {ACCOUNT}",
        f"#SBATCH -p {PARTITION}",
        f"#SBATCH -J {task.task_name}",
        f"#SBATCH -t {spec.walltime}",
        "#SBATCH -N 1",
        f"#SBATCH -c {spec.cpus}",
        f"#SBATCH --mem={spec.memory}",
        f"#SBATCH --gres={GPU_RESOURCE}",
        f"#SBATCH -o {slurm_dir / (task.task_name + '_%j.out')}",
        f"#SBATCH -e {slurm_dir / (task.task_name + '_%j.err')}",
        "set -euo pipefail",
        f"WORKTREE={shell_quote(spec.worktree)}",
        f"PYTHON={shell_quote(PYTHON)}",
        f"ENV_NAME={shell_quote(task.env_name)}",
        f"TASK_NAME={shell_quote(task.task_name)}",
        f"TASK_ROOT={shell_quote(task_root)}",
        f"WAND_ROOT={shell_quote(wandb_root)}",
        f"SEED0={task.seed0}",
        f"SEED1={task.seed1}",
        f"NUM_VEC_ENVS={args.mgmd_num_vec_envs}",
        f"START_STEP={args.mgmd_start_step}",
        f"TOTAL_TIMESTEPS={args.total_timesteps}",
        f"MEM_FRACTION={shell_quote(args.mgmd_mem_fraction)}",
        f"SLURM_DIR={shell_quote(slurm_dir)}",
        'mkdir -p "$TASK_ROOT" "$WAND_ROOT" "$SLURM_DIR"',
        'export OMP_NUM_THREADS=1',
        'export XLA_FLAGS="--xla_gpu_deterministic_ops=true"',
        'unset XLA_PYTHON_CLIENT_PREALLOCATE',
        'export XLA_PYTHON_CLIENT_MEM_FRACTION="$MEM_FRACTION"',
        'launch_seed() {',
        '  local seed="$1"',
        '  local wandb_dir="$2"',
        '  mkdir -p "$wandb_dir"',
        '  (',
        '    cd "$WORKTREE"',
        '    WANDB_MODE=offline WANDB_DIR="$wandb_dir" "$PYTHON" scripts/train_mujoco.py \\',
        '      --alg mgmd \\',
        '      --env "$ENV_NAME" \\',
        '      --seed "$seed" \\',
        '      --num_vec_envs "$NUM_VEC_ENVS" \\',
        '      --start_step "$START_STEP" \\',
        '      --total_step "$TOTAL_TIMESTEPS" \\',
        '      --suffix "${TASK_NAME}_s${seed}" \\',
        '      --log_root "$TASK_ROOT"',
        '  ) &',
        '  LAST_PID=$!',
        '}',
        'LAST_PID=""',
        'launch_seed "$SEED0" "$WAND_ROOT/s$SEED0"',
        'pid0="$LAST_PID"',
        'launch_seed "$SEED1" "$WAND_ROOT/s$SEED1"',
        'pid1="$LAST_PID"',
        'wait "$pid0"',
        'seed0_rc=$?',
        'wait "$pid1"',
        'seed1_rc=$?',
        'printf "seed0=%s seed0_rc=%s\n" "$SEED0" "$seed0_rc"',
        'printf "seed1=%s seed1_rc=%s\n" "$SEED1" "$seed1_rc"',
        'if [ "$seed0_rc" -ne 0 ] || [ "$seed1_rc" -ne 0 ]; then',
        '  exit 1',
        'fi',
        'exit 0',
    ]
    return "\n".join(lines) + "\n"


def write_artifacts(root: Path, algorithms: list[str], args: argparse.Namespace) -> dict[str, dict[str, object]]:
    if root.exists():
        if any(root.iterdir()):
            raise FileExistsError(f"Refusing to use non-empty root: {root}")
    else:
        root.mkdir(parents=True)

    payload: dict[str, dict[str, object]] = {}
    for algorithm in algorithms:
        spec = ALGO_SPECS[algorithm]
        algo_root = root / algorithm
        for subdir in ("sbatch", "slurm", "runs", "wandb"):
            (algo_root / subdir).mkdir(parents=True, exist_ok=True)
        tasks = build_tasks(algorithm)
        (algo_root / "tasks.tsv").write_text(tasks_tsv(tasks))
        payload_entry: dict[str, object] = {
            "spec": {
                "key": spec.key,
                "worktree": str(spec.worktree),
                "script_relpath": spec.script_relpath,
                "env_flag": spec.env_flag,
                "cpus": spec.cpus,
                "memory": spec.memory,
                "walltime": spec.walltime,
            },
            "root": str(algo_root),
            "tasks": [asdict(task) for task in tasks],
        }
        if args.submission_mode == "array":
            sbatch_path = algo_root / "sbatch" / f"{algorithm}_v5_pack2.sbatch"
            if algorithm == "mgmd":
                sbatch_path.write_text(mgmd_sbatch_text(spec, tasks, algo_root, args))
            else:
                sbatch_path.write_text(cleanrl_sbatch_text(spec, tasks, algo_root, args))
            payload_entry["sbatch"] = str(sbatch_path)
        else:
            sbatch_paths: list[str] = []
            for task in tasks:
                sbatch_path = algo_root / "sbatch" / f"{task.task_name}.sbatch"
                if algorithm == "mgmd":
                    sbatch_path.write_text(mgmd_task_sbatch_text(spec, task, algo_root, args))
                else:
                    sbatch_path.write_text(cleanrl_task_sbatch_text(spec, task, algo_root, args))
                sbatch_paths.append(str(sbatch_path))
            payload_entry["sbatch_paths"] = sbatch_paths
        payload[algorithm] = payload_entry

    manifest = {
        "root": str(root),
        "python": str(PYTHON),
        "account": ACCOUNT,
        "partition": PARTITION,
        "gpu_resource": GPU_RESOURCE,
        "envs": [{"env_name": env_name, "env_slug": env_slug} for env_name, env_slug in ENVS],
        "seeds": list(SEEDS),
        "submission_mode": args.submission_mode,
        "max_concurrent": args.max_concurrent,
        "total_timesteps": args.total_timesteps,
        "mgmd_start_step": args.mgmd_start_step,
        "mgmd_num_vec_envs": args.mgmd_num_vec_envs,
        "mgmd_mem_fraction": args.mgmd_mem_fraction,
        "algorithms": payload,
    }
    with open(root / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return payload


def submit_job(path: Path) -> str:
    result = subprocess.run(["sbatch", str(path)], check=True, text=True, capture_output=True)
    return result.stdout.strip()


def main() -> None:
    args = parse_args()
    algorithms = list(args.algos)
    root = args.root if args.root is not None else default_root()
    payload = write_artifacts(root, algorithms, args)
    print(f"Prepared Phase 4 launch artifacts under {root}")
    submissions = []
    for algorithm in algorithms:
        if args.submission_mode == "array":
            sbatch_path = Path(payload[algorithm]["sbatch"])
            print(f"{algorithm}: {sbatch_path}")
            if args.submit:
                submission = submit_job(sbatch_path)
                print(f"{algorithm}: {submission}")
                submissions.append(
                    {
                        "algorithm": algorithm,
                        "sbatch": str(sbatch_path),
                        "submission": submission,
                        "task_count": len(payload[algorithm]["tasks"]),
                    }
                )
        else:
            sbatch_paths = [Path(path) for path in payload[algorithm]["sbatch_paths"]]
            print(f"{algorithm}: prepared {len(sbatch_paths)} task scripts under {Path(payload[algorithm]['root']) / 'sbatch'}")
            if args.submit:
                for sbatch_path in sbatch_paths:
                    submission = submit_job(sbatch_path)
                    print(f"{algorithm}:{sbatch_path.stem}: {submission}")
                    submissions.append(
                        {
                            "algorithm": algorithm,
                            "task_name": sbatch_path.stem,
                            "sbatch": str(sbatch_path),
                            "submission": submission,
                        }
                    )
    if args.submit:
        with open(root / "submitted_jobs.json", "w") as f:
            json.dump(submissions, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
