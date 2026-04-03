#!/n/home09/pnielsen/.venvs/general/bin/python
"""
Sequential Ablation Sweep Controller.

Instead of running a full grid search, this sweeps one ablation variable at a time,
picks the best value using matched-seed comparisons, updates the base config, and
moves to the next variable. Loops back to the first variable with new seeds until
a previously-seen configuration is encountered.

Usage:
    ./scripts/sweep.py --cmd "python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4 ..." \
        --num-seeds 5 \
        --ablate dpmd_no_entropy_tuning flag \
        --ablate num_particles 32 64 128 256 \
        --ablate particle_selection_lambda 8 16 32 64 128 \
        --ablate dpmd_long_lr_schedule flag \
        --sweep-name my_ablation_study

    # Resume a previous sweep
    python scripts/sweep.py --resume sweeps/my_ablation_study/state.json

    # Check status of running sweep
    python scripts/sweep.py --status sweeps/my_ablation_study/state.json
"""

import argparse
import subprocess
import os
import sys
import re
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from scipy import stats

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Will use log file parsing instead.")


# ============================================================================
# Configuration and State
# ============================================================================

@dataclass
class AblationVar:
    """An ablation variable to sweep over."""
    name: str
    values: List[str]  # For flags, ["on", "off"]; for params, ["8", "16", "32"]
    is_flag: bool = False  # True for boolean flags like --dpmd_no_entropy_tuning


@dataclass 
class SweepState:
    """Persistent state for a sweep."""
    # Configuration
    base_cmd: str
    ablation_vars: List[Dict]  # Serialized AblationVar list
    num_seeds: int
    sweep_name: str
    sweep_dir: str
    
    # SLURM settings
    partition: str = "kempner_h100"
    gpu_type: str = "nvidia_h100"
    num_gpus: int = 1
    cpus: int = 2
    mem: str = "32G"
    time_limit: str = "0-8:00"
    
    # Current state
    current_cmd: str = ""  # Current best command
    current_var_idx: int = 0  # Which ablation variable we're testing
    round_num: int = 0  # How many times we've looped through all variables
    seed_offset: int = 0  # Offset for generating new seeds each round
    round_start_cmd: str = ""  # Command at the start of the current round (for convergence check)
    
    # History
    history: List[Dict] = field(default_factory=list)  # Full history of decisions
    
    # Job tracking
    pending_jobs: List[Dict] = field(default_factory=list)  # Jobs submitted but not analyzed
    
    # W&B settings
    wandb_project: str = "diffusion_online_rl"
    wandb_group: str = ""
    use_wandb: bool = True  # Whether to use W&B for result extraction
    
    # Sweep mode options
    skip_winner: bool = False  # Skip rerunning the value that matches current command
    fresh_seeds_per_var: bool = False  # Use new seeds for each variable (not just each round)
    last_winner_value: str = ""  # Track the winner from previous variable for skip_winner mode
    
    def get_ablation_vars(self) -> List[AblationVar]:
        return [AblationVar(**v) for v in self.ablation_vars]


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH -p {partition}
#SBATCH --gres=gpu:{gpu_type}:{num_gpus}
#SBATCH -c {cpus}
#SBATCH --mem={mem}
#SBATCH -t {time_limit}
#SBATCH -o {log_file}.out
#SBATCH -e {log_file}.err
#SBATCH --job-name={job_name}

source ~/.venvs/general/bin/activate
cd {project_dir}

echo "Job ID: $SLURM_JOB_ID"
echo "Sweep: {sweep_name}"
echo "Round: {round_num}, Variable: {var_name}, Value: {var_value}, Seed: {seed}"
echo "Command: {cmd}"
echo "Started at: $(date)"

{cmd}

echo "Finished at: $(date)"
"""


# ============================================================================
# Command Manipulation
# ============================================================================

def modify_cmd_for_param(cmd: str, param: str, value: str) -> str:
    """Replace or add a parameter value in the command."""
    pattern = rf'(--{re.escape(param)})\s+\S+'
    if re.search(pattern, cmd):
        return re.sub(pattern, rf'\1 {value}', cmd)
    else:
        return f"{cmd} --{param} {value}"


def modify_cmd_for_flag(cmd: str, flag: str, enable: bool) -> str:
    """Add or remove a boolean flag from the command."""
    pattern = rf'\s*--{re.escape(flag)}(?=\s|$)'
    cmd_without = re.sub(pattern, '', cmd).strip()
    cmd_without = re.sub(r'\s+', ' ', cmd_without)  # Clean up extra spaces
    
    if enable:
        return f"{cmd_without} --{flag}"
    else:
        return cmd_without


def apply_ablation(cmd: str, var: AblationVar, value: str) -> str:
    """Apply an ablation value to a command."""
    if var.is_flag:
        return modify_cmd_for_flag(cmd, var.name, value == "on")
    else:
        return modify_cmd_for_param(cmd, var.name, value)


def get_config_hash(cmd: str) -> str:
    """Get a hash of the command for detecting repeated configs."""
    # Normalize: remove seed, sort flags
    normalized = re.sub(r'--seed\s+\d+', '', cmd)
    normalized = ' '.join(sorted(normalized.split()))
    return hashlib.md5(normalized.encode()).hexdigest()[:12]


def add_wandb_tags(cmd: str, sweep_name: str, round_num: int, var_name: str, var_value: str, seed: int) -> str:
    """Add W&B configuration to command for organization."""
    # Add suffix to distinguish runs - MUST include seed for unique identification
    suffix = f"sweep_{sweep_name}_r{round_num}_{var_name}_{var_value}_s{seed}"
    cmd = modify_cmd_for_param(cmd, "suffix", suffix)
    return cmd


# ============================================================================
# Job Submission and Monitoring
# ============================================================================

def get_current_value_for_var(cmd: str, var: AblationVar) -> str:
    """Determine what value the current command has for this ablation variable."""
    if var.is_flag:
        # Check if flag is present in command
        if f"--{var.name}" in cmd:
            return "on"
        else:
            return "off"
    else:
        # Extract parameter value from command
        match = re.search(rf'--{re.escape(var.name)}\s+(\S+)', cmd)
        if match:
            return match.group(1)
        return None


def submit_jobs(state: SweepState, var: AblationVar, project_dir: Path) -> List[Dict]:
    """Submit jobs for all values of an ablation variable."""
    jobs = []
    seeds = list(range(state.seed_offset, state.seed_offset + state.num_seeds))
    
    # Determine which value corresponds to the current command (the "winner" from previous)
    current_value = get_current_value_for_var(state.current_cmd, var)
    
    # Determine which values to test
    values_to_test = var.values
    skipped_value = None
    if state.skip_winner and current_value in var.values:
        # Skip the value that matches current command
        values_to_test = [v for v in var.values if v != current_value]
        skipped_value = current_value
        print(f"  (skip_winner: skipping {var.name}={current_value}, already tested)")
    
    for value in values_to_test:
        for seed in seeds:
            # Build command
            cmd = apply_ablation(state.current_cmd, var, value)
            cmd = modify_cmd_for_param(cmd, "seed", str(seed))
            cmd = add_wandb_tags(cmd, state.sweep_name, state.round_num, var.name, value, seed)
            
            # Create job script
            job_name = f"{state.sweep_name}_r{state.round_num}_{var.name}_{value}_s{seed}"
            log_file = Path(state.sweep_dir) / "logs" / job_name
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            script = SBATCH_TEMPLATE.format(
                partition=state.partition,
                gpu_type=state.gpu_type,
                num_gpus=state.num_gpus,
                cpus=state.cpus,
                mem=state.mem,
                time_limit=state.time_limit,
                log_file=log_file,
                job_name=job_name,
                project_dir=project_dir,
                sweep_name=state.sweep_name,
                round_num=state.round_num,
                var_name=var.name,
                var_value=value,
                seed=seed,
                cmd=cmd,
            )
            
            script_path = Path(state.sweep_dir) / "scripts" / f"{job_name}.sh"
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(script)
            
            # Submit
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                jobs.append({
                    "job_id": job_id,
                    "var_name": var.name,
                    "var_value": value,
                    "seed": seed,
                    "cmd": cmd,
                    "log_file": str(log_file),
                    "status": "pending",
                })
                print(f"  Submitted job {job_id}: {var.name}={value}, seed={seed}")
            else:
                print(f"  FAILED to submit: {var.name}={value}, seed={seed}")
                print(f"    Error: {result.stderr}")
    
    return jobs


def check_job_status(job_id: str) -> str:
    """Check if a SLURM job is still running."""
    result = subprocess.run(
        ["squeue", "-j", job_id, "-h", "-o", "%t"],
        capture_output=True,
        text=True
    )
    status = result.stdout.strip()
    if not status:
        return "completed"
    elif status in ["PD", "CF"]:
        return "pending"
    elif status in ["R", "CG"]:
        return "running"
    else:
        return "unknown"


def cancel_jobs(jobs: List[Dict]) -> int:
    """Cancel all pending/running jobs. Returns count of cancelled jobs."""
    cancelled = 0
    for job in jobs:
        job_id = job.get("job_id")
        if not job_id:
            continue
        status = check_job_status(job_id)
        if status in ["pending", "running"]:
            result = subprocess.run(
                ["scancel", job_id],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                cancelled += 1
                print(f"  Cancelled job {job_id}")
    return cancelled


def cancel_sweep_jobs(state: SweepState) -> int:
    """Cancel all jobs associated with a sweep by job name pattern."""
    # Cancel by job name prefix
    pattern = f"{state.sweep_name}_*"
    result = subprocess.run(
        ["scancel", f"--name={pattern}"],
        capture_output=True,
        text=True
    )
    
    # Also cancel any tracked pending jobs
    cancelled = 0
    if state.pending_jobs:
        cancelled = cancel_jobs(state.pending_jobs)
    
    return cancelled


def wait_for_jobs(jobs: List[Dict], check_interval: int = 60) -> None:
    """Wait for all jobs to complete."""
    print(f"\nWaiting for {len(jobs)} jobs to complete...")
    
    while True:
        pending = 0
        running = 0
        
        for job in jobs:
            if job["status"] not in ["completed", "failed"]:
                status = check_job_status(job["job_id"])
                job["status"] = status
                
                if status == "pending":
                    pending += 1
                elif status == "running":
                    running += 1
        
        if pending == 0 and running == 0:
            print("All jobs completed!")
            break
        
        print(f"  Status: {pending} pending, {running} running, "
              f"{len(jobs) - pending - running} completed")
        time.sleep(check_interval)


def extract_final_return_from_log(log_file: str) -> Optional[float]:
    """Extract the final episode return from a job's log file.
    
    Parses the SLURM .out file to find the training log directory,
    then reads tensorboard events for episode returns.
    """
    try:
        log_path = Path(f"{log_file}.out")
        if not log_path.exists():
            return None
        
        content = log_path.read_text()
        
        # Find the training log directory from the Git modification line
        # Format: "Git modification details logged to /path/to/.../dacer.diff"
        log_dir = None
        
        match = re.search(r'Git modification details logged to ([^\s]+)/dacer\.diff', content)
        if match:
            log_dir = Path(match.group(1))
        
        if log_dir is None or not log_dir.exists():
            return None
        
        # Find tensorboard events file
        events_files = list(log_dir.glob("events.out.tfevents.*"))
        if not events_files:
            return None
        
        # Read tensorboard events to get sample/episode_return
        try:
            from tensorboard.backend.event_processing import event_accumulator
            ea = event_accumulator.EventAccumulator(str(log_dir))
            ea.Reload()
            
            # Look for sample/episode_return tag
            if 'sample/episode_return' in ea.Tags().get('scalars', []):
                events = ea.Scalars('sample/episode_return')
                if events:
                    # Get last 10 values and average for stability
                    values = [e.value for e in events[-10:]]
                    return np.mean(values)
        except ImportError:
            # tensorboard not available, try parsing events file directly
            pass
        except Exception:
            pass
        
        return None
    except Exception as e:
        print(f"  Warning: Could not parse {log_file}: {e}")
        return None


def extract_final_return_from_wandb(suffix_pattern: str, project: str = "diffusion_online_rl") -> Optional[float]:
    """Extract final episode return from W&B by searching for runs with matching suffix."""
    if not WANDB_AVAILABLE:
        return None
    
    try:
        api = wandb.Api()
        # Get recent runs and filter by name containing the suffix
        # W&B run names are like: dpmd_2025-12-19_02-46-42_s0_sweep_boltzman_ablation_r0_...
        runs = api.runs(project, order="-created_at")
        
        for run in runs:
            if suffix_pattern in run.name:
                # Get the summary metrics (final values)
                if "sample/episode_return" in run.summary:
                    return run.summary["sample/episode_return"]
                
                # Alternatively, get history and compute final average
                try:
                    history = run.history(keys=["sample/episode_return"], samples=100)
                    if len(history) > 0:
                        returns = history["sample/episode_return"].dropna().values
                        if len(returns) > 0:
                            return np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns)
                except Exception:
                    pass
        
        return None
    except Exception as e:
        print(f"  Warning: Could not fetch from W&B for {suffix_pattern}: {e}")
        return None


def extract_final_return(job: Dict, use_wandb: bool = True) -> Optional[float]:
    """Extract final return, trying log file parsing first (faster), then W&B."""
    result = None
    
    # Try log file parsing first (reads the training log directory)
    log_file = job.get("log_file", "")
    if log_file:
        result = extract_final_return_from_log(log_file)
    
    # Fall back to W&B if log parsing failed
    if result is None and use_wandb and WANDB_AVAILABLE:
        # Use the suffix we added to the command to search W&B
        cmd = job.get("cmd", "")
        suffix_match = re.search(r'--suffix\s+(\S+)', cmd)
        if suffix_match:
            suffix = suffix_match.group(1)
            result = extract_final_return_from_wandb(suffix)
    
    return result


# ============================================================================
# Statistical Analysis
# ============================================================================

def analyze_results(jobs: List[Dict], var: AblationVar, num_seeds: int, use_wandb: bool = True,
                    skipped_value: str = None, skipped_stats: Dict = None) -> Dict:
    """Analyze results and determine best ablation value.
    
    Args:
        jobs: List of job dicts with results
        var: The ablation variable being tested
        num_seeds: Number of seeds per value
        use_wandb: Whether to use W&B for result extraction
        skipped_value: Value that was skipped (if skip_winner mode)
        skipped_stats: Previous stats for skipped value to carry forward
    """
    
    # Group results by value
    results_by_value = {v: [] for v in var.values}
    seeds_by_value = {v: [] for v in var.values}
    
    print(f"\nCollecting results for {var.name}...")
    for job in jobs:
        if job["var_name"] != var.name:
            continue
        
        ret = extract_final_return(job, use_wandb=use_wandb)
        if ret is not None:
            results_by_value[job["var_value"]].append(ret)
            seeds_by_value[job["var_value"]].append(job["seed"])
            print(f"  {var.name}={job['var_value']}, seed={job['seed']}: {ret:.2f}")
        else:
            print(f"  {var.name}={job['var_value']}, seed={job['seed']}: MISSING")
    
    # If we skipped a value, carry forward its stats
    if skipped_value and skipped_stats:
        print(f"  {var.name}={skipped_value}: (carried forward from previous, mean={skipped_stats['mean']:.2f})")
    
    # Compute statistics for each value
    stats_by_value = {}
    for value in var.values:
        # If this value was skipped, use carried forward stats
        if value == skipped_value and skipped_stats:
            stats_by_value[value] = skipped_stats.copy()
            stats_by_value[value]["carried_forward"] = True
            continue
        
        returns = results_by_value[value]
        if len(returns) > 0:
            stats_by_value[value] = {
                "mean": np.mean(returns),
                "std": np.std(returns, ddof=1) if len(returns) > 1 else 0,
                "se": np.std(returns, ddof=1) / np.sqrt(len(returns)) if len(returns) > 1 else 0,
                "n": len(returns),
                "returns": returns,
                "seeds": seeds_by_value[value],
            }
        else:
            stats_by_value[value] = {
                "mean": float('-inf'),
                "std": 0,
                "se": 0,
                "n": 0,
                "returns": [],
                "seeds": [],
            }
    
    # Find best value by mean
    best_value = max(var.values, key=lambda v: stats_by_value[v]["mean"])
    best_stats = stats_by_value[best_value]
    
    # Pairwise comparisons with best using paired t-test (matched seeds)
    comparisons = {}
    for value in var.values:
        if value == best_value:
            continue
        
        other_stats = stats_by_value[value]
        
        # Match by seed for paired comparison
        best_seeds = set(best_stats["seeds"])
        other_seeds = set(other_stats["seeds"])
        common_seeds = best_seeds & other_seeds
        
        if len(common_seeds) >= 2:
            # Get matched pairs
            best_by_seed = dict(zip(best_stats["seeds"], best_stats["returns"]))
            other_by_seed = dict(zip(other_stats["seeds"], other_stats["returns"]))
            
            paired_best = [best_by_seed[s] for s in sorted(common_seeds)]
            paired_other = [other_by_seed[s] for s in sorted(common_seeds)]
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(paired_best, paired_other)
            
            comparisons[value] = {
                "t_stat": t_stat,
                "p_value": p_value,
                "n_pairs": len(common_seeds),
                "mean_diff": np.mean(paired_best) - np.mean(paired_other),
            }
        else:
            # Fall back to unpaired test
            if other_stats["n"] >= 2 and best_stats["n"] >= 2:
                t_stat, p_value = stats.ttest_ind(
                    best_stats["returns"], 
                    other_stats["returns"]
                )
                comparisons[value] = {
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "n_pairs": 0,
                    "mean_diff": best_stats["mean"] - other_stats["mean"],
                    "note": "unpaired test (insufficient seed overlap)",
                }
    
    return {
        "variable": var.name,
        "best_value": best_value,
        "stats_by_value": stats_by_value,
        "comparisons": comparisons,
    }


def print_analysis_report(analysis: Dict) -> None:
    """Print a detailed analysis report."""
    print("\n" + "=" * 70)
    print(f"ANALYSIS: {analysis['variable']}")
    print("=" * 70)
    
    # Results table
    print(f"\n{'Value':<20} {'Mean':>12} {'SE':>10} {'N':>5}")
    print("-" * 50)
    
    sorted_values = sorted(
        analysis['stats_by_value'].keys(),
        key=lambda v: analysis['stats_by_value'][v]['mean'],
        reverse=True
    )
    
    for value in sorted_values:
        s = analysis['stats_by_value'][value]
        marker = " ***" if value == analysis['best_value'] else ""
        print(f"{value:<20} {s['mean']:>12.2f} {s['se']:>10.2f} {s['n']:>5}{marker}")
    
    # Statistical comparisons
    if analysis['comparisons']:
        print(f"\nPairwise comparisons vs best ({analysis['best_value']}):")
        print("-" * 50)
        
        for value, comp in analysis['comparisons'].items():
            sig = ""
            if comp['p_value'] < 0.01:
                sig = "**"
            elif comp['p_value'] < 0.05:
                sig = "*"
            elif comp['p_value'] < 0.1:
                sig = "."
            
            test_type = f"(n={comp['n_pairs']} paired)" if comp['n_pairs'] > 0 else "(unpaired)"
            print(f"  vs {value:<15}: diff={comp['mean_diff']:>8.2f}, "
                  f"p={comp['p_value']:.4f}{sig} {test_type}")
    
    print(f"\nBest value: {analysis['best_value']}")
    print("=" * 70)


# ============================================================================
# Main Sweep Controller
# ============================================================================

def save_state(state: SweepState) -> None:
    """Save sweep state to disk."""
    state_path = Path(state.sweep_dir) / "state.json"
    state_path.write_text(json.dumps(asdict(state), indent=2))


def load_state(state_path: str) -> SweepState:
    """Load sweep state from disk."""
    data = json.loads(Path(state_path).read_text())
    return SweepState(**data)


def run_final_validation(state: SweepState, project_dir: Path) -> Dict:
    """
    Run final validation: test the converged command on fresh seeds.
    Returns statistics for the final command.
    """
    print(f"\n{'#' * 70}")
    print("FINAL VALIDATION")
    print(f"Running converged command on {state.num_seeds} fresh seeds...")
    print(f"Command: {state.current_cmd}")
    print(f"{'#' * 70}")
    
    # Create a dummy ablation var that just tests the current command
    # We use a fake variable with a single value to reuse the job submission logic
    validation_var = AblationVar(name="_validation", values=["final"], is_flag=False)
    
    # Submit validation jobs
    jobs = []
    sweep_dir = Path(state.sweep_dir)
    logs_dir = sweep_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    for seed_idx in range(state.num_seeds):
        seed = state.seed_offset + seed_idx
        
        # Build command with seed
        cmd = state.current_cmd
        if "--seed" in cmd:
            cmd = re.sub(r'--seed\s+\d+', f'--seed {seed}', cmd)
        else:
            cmd = f"{cmd} --seed {seed}"
        
        # Add sweep suffix for W&B organization
        suffix = f"sweep_{state.sweep_name}_final_seed{seed}"
        if "--suffix" in cmd:
            cmd = re.sub(r'--suffix\s+\S+', f'--suffix {suffix}', cmd)
        else:
            cmd = f"{cmd} --suffix {suffix}"
        
        log_file = logs_dir / f"final_seed{seed}"
        job_name = f"{state.sweep_name}_final_s{seed}"
        
        job_id = submit_slurm_job(
            cmd=cmd,
            job_name=job_name,
            log_file=str(log_file),
            partition=state.partition,
            gpu_type=state.gpu_type,
            num_gpus=state.num_gpus,
            cpus=state.cpus,
            mem=state.mem,
            time_limit=state.time_limit,
            project_dir=project_dir,
        )
        
        jobs.append({
            "job_id": job_id,
            "seed": seed,
            "var_name": "_validation",
            "var_value": "final",
            "cmd": cmd,
            "log_file": str(log_file),
            "status": "pending",
        })
        print(f"  Submitted job {job_id}: seed={seed}")
    
    # Wait and collect results
    wait_for_jobs(jobs)
    
    returns = []
    print(f"\nCollecting final validation results...")
    for job in jobs:
        ret = extract_final_return(job, use_wandb=state.use_wandb)
        if ret is not None:
            returns.append(ret)
            print(f"  seed={job['seed']}: {ret:.2f}")
        else:
            print(f"  seed={job['seed']}: MISSING")
    
    if returns:
        return {
            "mean": np.mean(returns),
            "std": np.std(returns, ddof=1) if len(returns) > 1 else 0,
            "se": np.std(returns, ddof=1) / np.sqrt(len(returns)) if len(returns) > 1 else 0,
            "n": len(returns),
            "returns": returns,
        }
    else:
        return {"mean": float('-inf'), "std": 0, "se": 0, "n": 0, "returns": []}


def run_sweep_step(state: SweepState, project_dir: Path) -> Tuple[SweepState, bool, bool]:
    """
    Run one step of the sweep (test one ablation variable).
    Returns (updated_state, should_continue, converged).
    """
    vars = state.get_ablation_vars()
    current_var = vars[state.current_var_idx]
    
    # At the start of a new round, record the command for convergence checking
    if state.current_var_idx == 0:
        state.round_start_cmd = state.current_cmd
        save_state(state)
    
    # Determine if we should skip the current winner value
    current_value = get_current_value_for_var(state.current_cmd, current_var)
    skipped_value = None
    skipped_stats = None
    
    if state.skip_winner and current_value in current_var.values:
        skipped_value = current_value
        # Find previous stats for this value from history
        for h in reversed(state.history):
            if h["variable"] == current_var.name and current_value in h.get("analysis", {}):
                skipped_stats = h["analysis"][current_value]
                break
    
    # Calculate number of jobs to submit
    num_values = len(current_var.values) - (1 if skipped_value else 0)
    num_jobs = num_values * state.num_seeds
    
    print(f"\n{'#' * 70}")
    print(f"ROUND {state.round_num}, VARIABLE {state.current_var_idx + 1}/{len(vars)}: {current_var.name}")
    print(f"Current base command: {state.current_cmd}")
    print(f"Testing values: {current_var.values}")
    if state.skip_winner:
        print(f"Mode: skip_winner (skipping {current_value} if previously tested)")
    if state.fresh_seeds_per_var:
        print(f"Mode: fresh_seeds_per_var (seeds {state.seed_offset}-{state.seed_offset + state.num_seeds - 1})")
    print(f"{'#' * 70}")
    
    # Submit jobs
    print(f"\nSubmitting {num_jobs} jobs...")
    jobs = submit_jobs(state, current_var, project_dir)
    state.pending_jobs = jobs
    save_state(state)
    
    # Wait for completion
    wait_for_jobs(jobs)
    
    # Analyze results
    analysis = analyze_results(jobs, current_var, state.num_seeds, use_wandb=state.use_wandb,
                               skipped_value=skipped_value, skipped_stats=skipped_stats)
    print_analysis_report(analysis)
    
    # Update state with best value
    best_value = analysis['best_value']
    state.current_cmd = apply_ablation(state.current_cmd, current_var, best_value)
    
    # Record history
    state.history.append({
        "round": state.round_num,
        "var_idx": state.current_var_idx,
        "variable": current_var.name,
        "best_value": best_value,
        "analysis": {
            k: v for k, v in analysis['stats_by_value'].items()
            if k != 'returns'  # Don't store raw returns in history
        },
        "updated_cmd": state.current_cmd,
        "timestamp": datetime.now().isoformat(),
    })
    
    # Move to next variable
    state.current_var_idx = (state.current_var_idx + 1) % len(vars)
    state.pending_jobs = []
    
    # If fresh_seeds_per_var is enabled, increment seeds after each variable
    if state.fresh_seeds_per_var:
        state.seed_offset += state.num_seeds
    
    # Check for end of round
    converged = False
    if state.current_var_idx == 0:
        # Completed all variables in this round
        print(f"\n*** Completed round {state.round_num}. ***")
        
        # Check convergence: did the command change during this round?
        if state.current_cmd == state.round_start_cmd:
            print("*** CONVERGED: No changes made during this round. ***")
            converged = True
        else:
            print(f"Command changed during round {state.round_num}.")
            print(f"  Before: {state.round_start_cmd}")
            print(f"  After:  {state.current_cmd}")
            state.round_num += 1
            # Only increment seeds here if NOT using fresh_seeds_per_var (already incremented above)
            if not state.fresh_seeds_per_var:
                state.seed_offset += state.num_seeds
            print(f"*** Starting round {state.round_num} with new seeds (offset={state.seed_offset}). ***")
    
    save_state(state)
    
    return state, True, converged


def parse_ablation_arg(ablation: List[str]) -> AblationVar:
    """Parse an --ablate argument into an AblationVar."""
    name = ablation[0]
    
    if len(ablation) == 2 and ablation[1].lower() == "flag":
        # Boolean flag
        return AblationVar(name=name, values=["off", "on"], is_flag=True)
    else:
        # Parameter with values
        return AblationVar(name=name, values=ablation[1:], is_flag=False)


def main():
    parser = argparse.ArgumentParser(
        description="Sequential ablation sweep controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Main options
    parser.add_argument("--cmd", type=str,
                        help="Base python command to run")
    parser.add_argument("--num-seeds", type=int, default=5,
                        help="Number of seeds per ablation value (default: 5)")
    parser.add_argument("--ablate", action="append", nargs="+",
                        metavar=("FLAG", "VALUES"),
                        help="Ablation variable. Use 'flag' for boolean flags, or list values.")
    parser.add_argument("--sweep-name", type=str, default=None,
                        help="Name for this sweep (default: auto-generated)")
    
    # Resume/status/cancel
    parser.add_argument("--resume", type=str, metavar="STATE_FILE",
                        help="Resume a previous sweep from state file")
    parser.add_argument("--status", type=str, metavar="STATE_FILE",
                        help="Check status of a sweep")
    parser.add_argument("--cancel", type=str, metavar="STATE_FILE",
                        help="Cancel all pending jobs for a sweep")
    
    # Background execution (survives disconnection)
    parser.add_argument("--background", action="store_true",
                        help="Run the sweep controller as a SLURM job (survives disconnection)")
    
    # SLURM options
    parser.add_argument("--partition", "-p", type=str, default="kempner_h100")
    parser.add_argument("--gpu-type", type=str, default="nvidia_h100")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--cpus", "-c", type=int, default=2)
    parser.add_argument("--mem", type=str, default="32G")
    parser.add_argument("--time", "-t", type=str, default="0-6:00")
    
    # Result extraction options
    parser.add_argument("--no-wandb", action="store_true",
                        help="Don't use W&B for result extraction; use log file parsing instead")
    
    # Sweep mode options
    parser.add_argument("--skip-winner", action="store_true",
                        help="Skip rerunning the value that matches current command (saves compute)")
    parser.add_argument("--fresh-seeds-per-var", action="store_true",
                        help="Use fresh seeds for each variable (not just each round) to prevent seed overfitting")
    
    args = parser.parse_args()
    
    project_dir = Path(__file__).parent.parent.resolve()
    
    # Handle cancel
    if args.cancel:
        state = load_state(args.cancel)
        print(f"Cancelling jobs for sweep: {state.sweep_name}")
        cancelled = cancel_sweep_jobs(state)
        print(f"Cancelled {cancelled} jobs")
        return
    
    # Handle status check
    if args.status:
        state = load_state(args.status)
        print(f"Sweep: {state.sweep_name}")
        print(f"Round: {state.round_num}, Variable index: {state.current_var_idx}")
        print(f"Current command: {state.current_cmd}")
        print(f"Tested configs: {len(state.tested_configs)}")
        print(f"Pending jobs: {len(state.pending_jobs)}")
        
        if state.pending_jobs:
            print("\nPending job status:")
            for job in state.pending_jobs:
                status = check_job_status(job["job_id"])
                print(f"  {job['job_id']}: {status}")
        
        if state.history:
            print("\nHistory:")
            for h in state.history:
                print(f"  Round {h['round']}: {h['variable']} -> {h['best_value']}")
        return
    
    # Handle background execution - submit controller as a SLURM job
    if args.background and not os.environ.get("SWEEP_RUNNING_IN_SLURM"):
        # Re-run this script as a SLURM job (without --background)
        sweep_name = args.sweep_name or datetime.now().strftime("sweep_%Y%m%d_%H%M%S")
        sweep_dir = project_dir / "sweeps" / sweep_name
        sweep_dir.mkdir(parents=True, exist_ok=True)
        
        # Build the command to run (same args but without --background)
        script_path = Path(__file__).resolve()
        cmd_args = sys.argv[1:]
        cmd_args = [a for a in cmd_args if a != "--background"]
        controller_cmd = f"{script_path} {' '.join(cmd_args)}"
        
        # Controller job script - runs on CPU partition with minimal resources
        controller_script = f"""#!/bin/bash
#SBATCH -p serial_requeue
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 7-0:00
#SBATCH -o {sweep_dir}/controller.out
#SBATCH -e {sweep_dir}/controller.err
#SBATCH --job-name={sweep_name}_controller

export SWEEP_RUNNING_IN_SLURM=1
source ~/.venvs/general/bin/activate
cd {project_dir}

echo "Sweep controller started at: $(date)"
echo "Command: {controller_cmd}"

{controller_cmd}

echo "Sweep controller finished at: $(date)"
"""
        controller_script_path = sweep_dir / "controller.sh"
        controller_script_path.write_text(controller_script)
        
        result = subprocess.run(
            ["sbatch", str(controller_script_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            job_id = result.stdout.strip().split()[-1]
            print(f"Submitted sweep controller as SLURM job {job_id}")
            print(f"Sweep directory: {sweep_dir}")
            print(f"Monitor with: tail -f {sweep_dir}/controller.out")
            print(f"Check status: ./scripts/sweep.py --status {sweep_dir}/state.json")
            print(f"Cancel sweep: ./scripts/sweep.py --cancel {sweep_dir}/state.json")
        else:
            print(f"Failed to submit controller job: {result.stderr}")
        return
    
    # Handle resume
    if args.resume:
        state = load_state(args.resume)
        print(f"Resuming sweep: {state.sweep_name}")
        
        # If there are pending jobs, wait for them first
        if state.pending_jobs:
            print("Found pending jobs, checking status...")
            wait_for_jobs(state.pending_jobs)
            
            # Analyze and continue
            var = state.get_ablation_vars()[state.current_var_idx]
            analysis = analyze_results(state.pending_jobs, var, state.num_seeds, use_wandb=state.use_wandb)
            print_analysis_report(analysis)
            
            best_value = analysis['best_value']
            state.current_cmd = apply_ablation(state.current_cmd, var, best_value)
            state.history.append({
                "round": state.round_num,
                "var_idx": state.current_var_idx,
                "variable": var.name,
                "best_value": best_value,
                "timestamp": datetime.now().isoformat(),
            })
            
            state.current_var_idx = (state.current_var_idx + 1) % len(state.get_ablation_vars())
            if state.current_var_idx == 0:
                # Check convergence at end of round
                if state.current_cmd == state.round_start_cmd:
                    print("*** CONVERGED: No changes made during this round. ***")
                else:
                    state.round_num += 1
                    state.seed_offset += state.num_seeds
            
            state.pending_jobs = []
            save_state(state)
    else:
        # New sweep
        if not args.cmd or not args.ablate:
            parser.error("--cmd and at least one --ablate are required for new sweeps")
        
        # Parse ablation variables
        ablation_vars = [parse_ablation_arg(a) for a in args.ablate]
        
        # Generate sweep name
        sweep_name = args.sweep_name or datetime.now().strftime("sweep_%Y%m%d_%H%M%S")
        sweep_dir = project_dir / "sweeps" / sweep_name
        sweep_dir.mkdir(parents=True, exist_ok=True)
        
        state = SweepState(
            base_cmd=args.cmd,
            ablation_vars=[asdict(v) for v in ablation_vars],
            num_seeds=args.num_seeds,
            sweep_name=sweep_name,
            sweep_dir=str(sweep_dir),
            current_cmd=args.cmd,
            partition=args.partition,
            gpu_type=args.gpu_type,
            num_gpus=args.num_gpus,
            cpus=args.cpus,
            mem=args.mem,
            time_limit=args.time,
            use_wandb=not args.no_wandb,
            skip_winner=args.skip_winner,
            fresh_seeds_per_var=args.fresh_seeds_per_var,
        )
        
        save_state(state)
        print(f"Created new sweep: {sweep_name}")
        print(f"State file: {sweep_dir}/state.json")
    
    # Run sweep loop
    print(f"\nStarting sweep with {len(state.get_ablation_vars())} ablation variables")
    print(f"Variables: {[v.name for v in state.get_ablation_vars()]}")
    
    # Show mode options
    modes = []
    if state.skip_winner:
        modes.append("skip_winner (reuse previous winner stats)")
    if state.fresh_seeds_per_var:
        modes.append("fresh_seeds_per_var (new seeds each variable)")
    if modes:
        print(f"Modes: {', '.join(modes)}")
    else:
        print("Modes: default (test all values, new seeds each round)")
    
    final_stats = None
    try:
        while True:
            state, should_continue, converged = run_sweep_step(state, project_dir)
            if converged:
                # Run final validation on fresh seeds
                state.seed_offset += state.num_seeds  # Use fresh seeds for validation
                final_stats = run_final_validation(state, project_dir)
                break
            if not should_continue:
                break
    except KeyboardInterrupt:
        print("\n\nSweep interrupted. State saved.")
        print(f"Resume with: python scripts/sweep.py --resume {state.sweep_dir}/state.json")
        save_state(state)
    
    # Final report
    print("\n" + "#" * 70)
    print("SWEEP COMPLETE")
    print("#" * 70)
    print(f"\nFinal command: {state.current_cmd}")
    
    if final_stats and final_stats['n'] > 0:
        print(f"\nFinal performance (on {final_stats['n']} fresh seeds):")
        print(f"  Mean: {final_stats['mean']:.2f} ± {final_stats['se']:.2f} (SE)")
        print(f"  Std:  {final_stats['std']:.2f}")
    
    print(f"\nHistory:")
    for h in state.history:
        print(f"  Round {h['round']}: {h['variable']} -> {h['best_value']}")


if __name__ == "__main__":
    main()
