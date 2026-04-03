#!/n/home09/pnielsen/.venvs/general/bin/python
"""
SLURM job launcher for experiment sweeps.

Usage examples:
    # Single job
    python scripts/launch.py --cmd "python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4"

    # Multiple seeds
    python scripts/launch.py --cmd "python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4" --seeds 0 1 2 3 4

    # Ablation sweep (changes from base command)
    python scripts/launch.py --cmd "python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4 --tfg_eta 16" \\
        --ablate tfg_eta 8 16 32

    # Multiple ablations with seeds
    python scripts/launch.py --cmd "python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4" \\
        --seeds 100 101 102 \\
        --ablate tfg_eta 8 16 32 \\
        --ablate num_particles 1 64 128

    # Dry run (print commands without submitting)
    python scripts/launch.py --cmd "..." --dry-run
"""

import argparse
import subprocess
import os
import re
import itertools
from pathlib import Path
from datetime import datetime


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH -p {partition}
#SBATCH --gres=gpu:{gpu_type}:{num_gpus}
#SBATCH -c {cpus}
#SBATCH --mem={mem}
#SBATCH -t {time}
#SBATCH -o {log_dir}/%j.out
#SBATCH -e {log_dir}/%j.err
#SBATCH --job-name={job_name}

export PATH="$HOME/.venvs/general/bin:$PATH"
export VIRTUAL_ENV="$HOME/.venvs/general"
cd {project_dir}

echo "Job ID: $SLURM_JOB_ID"
echo "Running: {cmd}"
echo "Started at: $(date)"

{cmd}

echo "Finished at: $(date)"
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch SLURM jobs for experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required
    parser.add_argument("--cmd", type=str, required=True,
                        help="Base python command to run")
    
    # Sweep options
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="List of seeds to run (adds --seed X to command)")
    parser.add_argument("--ablate", action="append", nargs="+", metavar=("FLAG", "VALUES"),
                        help="Ablation: --ablate flag_name val1 val2 val3. Can be used multiple times.")
    
    # SLURM options
    parser.add_argument("--partition", "-p", type=str, default="kempner_h100",
                        help="SLURM partition (default: kempner_h100)")
    parser.add_argument("--gpu-type", type=str, default="nvidia_h100",
                        help="GPU type (default: nvidia_h100)")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs (default: 1)")
    parser.add_argument("--cpus", "-c", type=int, default=2,
                        help="Number of CPUs (default: 2)")
    parser.add_argument("--mem", type=str, default="32G",
                        help="Memory (default: 32G)")
    parser.add_argument("--time", "-t", type=str, default="0-8:00",
                        help="Time limit (default: 0-8:00)")
    
    # Utility options
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without submitting")
    parser.add_argument("--job-name", type=str, default=None,
                        help="Job name prefix (default: inferred from command)")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for logs (default: logs/slurm/<timestamp>)")
    
    return parser.parse_args()


def infer_job_name(cmd: str) -> str:
    """Extract a short job name from the command."""
    # Try to extract --alg and --env
    alg_match = re.search(r'--alg\s+(\S+)', cmd)
    env_match = re.search(r'--env\s+(\S+)', cmd)
    
    parts = []
    if alg_match:
        parts.append(alg_match.group(1))
    if env_match:
        # Shorten env name (e.g., HalfCheetah-v4 -> HC)
        env = env_match.group(1)
        short_env = ''.join(c for c in env if c.isupper() or c.isdigit())
        parts.append(short_env)
    
    return "_".join(parts) if parts else "job"


def modify_cmd_for_flag(cmd: str, flag: str, value: str) -> str:
    """Replace or add a flag value in the command."""
    # Pattern to match --flag followed by its value
    pattern = rf'(--{re.escape(flag)})\s+\S+'
    
    if re.search(pattern, cmd):
        # Replace existing flag
        return re.sub(pattern, rf'\1 {value}', cmd)
    else:
        # Add new flag at end
        return f"{cmd} --{flag} {value}"


def generate_commands(base_cmd: str, seeds: list, ablations: list) -> list:
    """Generate all command combinations from seeds and ablations."""
    commands = []
    
    # Parse ablations into {flag: [values]} dict
    ablation_dict = {}
    if ablations:
        for ablation in ablations:
            flag = ablation[0]
            values = ablation[1:]
            ablation_dict[flag] = values
    
    # Generate all combinations of ablation values
    if ablation_dict:
        flags = list(ablation_dict.keys())
        value_lists = [ablation_dict[f] for f in flags]
        combinations = list(itertools.product(*value_lists))
    else:
        flags = []
        combinations = [()]  # Single empty combination
    
    # Generate commands for each combination
    for combo in combinations:
        cmd = base_cmd
        ablation_suffix = []
        
        # Apply ablation flags
        for flag, value in zip(flags, combo):
            cmd = modify_cmd_for_flag(cmd, flag, value)
            ablation_suffix.append(f"{flag}={value}")
        
        if seeds:
            # Generate one command per seed
            for seed in seeds:
                seed_cmd = modify_cmd_for_flag(cmd, "seed", str(seed))
                desc = f"seed={seed}"
                if ablation_suffix:
                    desc = ",".join(ablation_suffix) + f",seed={seed}"
                commands.append((seed_cmd, desc))
        else:
            # No seeds specified
            desc = ",".join(ablation_suffix) if ablation_suffix else "base"
            commands.append((cmd, desc))
    
    return commands


def main():
    args = parse_args()
    
    # Setup paths
    project_dir = Path(__file__).parent.parent.resolve()
    
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = project_dir / "logs" / "slurm" / timestamp
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Infer job name if not provided
    job_name_base = args.job_name or infer_job_name(args.cmd)
    
    # Generate all commands
    commands = generate_commands(args.cmd, args.seeds, args.ablate)
    
    print(f"Generated {len(commands)} job(s)")
    print(f"Log directory: {log_dir}")
    print()
    
    submitted = 0
    for i, (cmd, desc) in enumerate(commands):
        job_name = f"{job_name_base}_{i}" if len(commands) > 1 else job_name_base
        
        script = SBATCH_TEMPLATE.format(
            partition=args.partition,
            gpu_type=args.gpu_type,
            num_gpus=args.num_gpus,
            cpus=args.cpus,
            mem=args.mem,
            time=args.time,
            log_dir=log_dir,
            job_name=job_name,
            project_dir=project_dir,
            cmd=cmd,
        )
        
        if args.dry_run:
            print(f"[{i+1}/{len(commands)}] {desc}")
            print(f"  Command: {cmd}")
            print()
        else:
            # Write script to temp file and submit
            script_path = log_dir / f"{job_name}_{i}.sh"
            script_path.write_text(script)
            
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                print(f"[{i+1}/{len(commands)}] Submitted job {job_id}: {desc}")
                submitted += 1
            else:
                print(f"[{i+1}/{len(commands)}] FAILED: {desc}")
                print(f"  Error: {result.stderr}")
    
    if not args.dry_run:
        print()
        print(f"Submitted {submitted}/{len(commands)} jobs")
        print(f"Monitor with: squeue -u $USER")
        print(f"Logs in: {log_dir}")


if __name__ == "__main__":
    main()
