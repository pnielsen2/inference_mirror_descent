#!/n/home09/pnielsen/.venvs/general/bin/python
"""
Thompson Sampling Sweep Controller

A Bayesian adaptive sweep strategy that explores "adjacent" configurations
using Thompson sampling with paired comparisons.

Adjacent configs differ from the base by exactly one variable:
- For flags: the opposite value
- For numerical values: base_value * factor or base_value / factor

Strategy:
1. Start with base command, run base + all adjacent configs on 2 seeds
2. Fit t-distribution to paired differences (adjacent - base)
3. Use Thompson sampling to select which configs to run more seeds on
4. When 99% confident an adjacent is better, it becomes new base
5. Stop when no adjacent reaches 99% confidence after max_seeds (default 10)

Example usage:
    ./scripts/sweep_thompson.py \\
        --cmd "python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4 ..." \\
        --var lr --var num_particles --var dpmd_long_lr_schedule flag \\
        --factor 2 \\
        --sweep-name thompson_sweep

Parallel sweeps (run multiple base configs):
    ./scripts/sweep_thompson.py \\
        --cmd "python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4 ..." \\
        --var lr --var num_particles \\
        --split dpmd_long_lr_schedule flag \\
        --sweep-name parallel_sweep
"""

import argparse
import json
import subprocess
import time
import re
import hashlib
import signal
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Set
from datetime import datetime
from scipy import stats

# Global state for signal handling
_active_jobs: List[Dict] = []
_sweep_state: Optional['ThompsonState'] = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class AblationVar:
    """An ablation variable specification."""
    name: str
    is_flag: bool = False
    factor: float = 2.0  # Adjacency factor for numerical values


@dataclass
class AdjacentConfig:
    """An adjacent configuration."""
    var_name: str
    direction: str  # "up", "down", or "toggle"
    value: str  # The actual value
    
    def key(self) -> str:
        return f"{self.var_name}_{self.direction}"


@dataclass
class ConfigResults:
    """Results for a configuration across seeds."""
    returns: Dict[int, float] = field(default_factory=dict)  # seed -> return
    
    def mean(self) -> float:
        if not self.returns:
            return float('-inf')
        return np.mean(list(self.returns.values()))
    
    def seeds(self) -> Set[int]:
        return set(self.returns.keys())


@dataclass
class ThompsonState:
    """Persistent state for Thompson sampling sweep."""
    # Configuration
    base_cmd: str
    ablation_vars: List[Dict]  # Serialized AblationVar list
    sweep_name: str
    sweep_dir: str
    factor: float = 2.0
    confidence_threshold: float = 0.99
    max_seeds: int = 10
    initial_seeds: int = 2
    min_seeds_for_promotion: int = 4
    
    # SLURM settings
    partition: str = "gpu_requeue"
    gpu_type: str = "nvidia_h200"
    num_gpus: int = 1
    cpus: int = 2
    mem: str = "32G"
    time_limit: str = "0-8:00"
    
    # Current state
    current_base_cmd: str = ""
    iteration: int = 0
    seed_offset: int = 0
    
    # Results tracking
    base_results: Dict[int, float] = field(default_factory=dict)  # seed -> return
    adjacent_results: Dict[str, Dict[int, float]] = field(default_factory=dict)  # config_key -> seed -> return
    
    # Job tracking
    pending_jobs: List[Dict] = field(default_factory=list)
    
    # History
    history: List[Dict] = field(default_factory=list)
    
    # W&B settings
    use_wandb: bool = True
    
    def get_ablation_vars(self) -> List[AblationVar]:
        return [AblationVar(**v) for v in self.ablation_vars]


@dataclass 
class ParallelSweepState:
    """State for parallel sweeps."""
    sweep_name: str
    sweep_dir: str
    split_var: Dict  # Serialized AblationVar for the split variable
    sub_sweeps: List[str]  # Paths to sub-sweep state files
    

# ============================================================================
# SLURM Job Management
# ============================================================================

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
echo "Iteration: {iteration}, Config: {config_key}, Seed: {seed}"
echo "Command: {cmd}"
echo "Started at: $(date)"

{cmd}

echo "Finished at: $(date)"
"""


def extract_default_from_script(script_path: str, var_name: str) -> Optional[str]:
    """Extract default value for an argument from a Python script's argparse definitions."""
    try:
        if not Path(script_path).exists():
            return None
            
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Look for patterns like: parser.add_argument("--var_name", ... default=VALUE ...)
        # Handle various formats: default=256, default=int(1e6), default=3e-4, default="string"
        # Use non-greedy match and look for the default= within the add_argument call
        patterns = [
            rf'add_argument\s*\(\s*["\']--{re.escape(var_name)}["\'][^)]*?default\s*=\s*([^,\)\s]+)',
            rf'add_argument\s*\(\s*["\']--{re.escape(var_name.replace("_", "-"))}["\'][^)]*?default\s*=\s*([^,\)\s]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content, re.DOTALL)
            if match:
                default_expr = match.group(1).strip()
                # Evaluate simple expressions
                try:
                    # Handle int(1e6), float expressions, etc.
                    value = eval(default_expr)
                    return str(value)
                except:
                    # Return as-is if can't evaluate
                    return default_expr.strip('"\'')
        
        return None
    except Exception as e:
        return None


def get_current_value(cmd: str, var: AblationVar, script_path: str = None) -> Optional[str]:
    """Get the current value of a variable in the command, falling back to script defaults."""
    if var.is_flag:
        return "on" if f"--{var.name}" in cmd else "off"
    else:
        match = re.search(rf'--{re.escape(var.name)}\s+(\S+)', cmd)
        if match:
            return match.group(1)
        
        # Try to extract default from script
        if script_path:
            default = extract_default_from_script(script_path, var.name)
            if default:
                return default
        
        return None


def apply_value(cmd: str, var: AblationVar, value: str) -> str:
    """Apply a value to the command."""
    if var.is_flag:
        if value == "on":
            if f"--{var.name}" not in cmd:
                cmd = cmd + f" --{var.name}"
        else:  # off
            cmd = re.sub(rf'\s*--{re.escape(var.name)}(?:\s|$)', ' ', cmd)
    else:
        if re.search(rf'--{re.escape(var.name)}\s+\S+', cmd):
            cmd = re.sub(rf'--{re.escape(var.name)}\s+\S+', f'--{var.name} {value}', cmd)
        else:
            cmd = cmd + f" --{var.name} {value}"
    return cmd.strip()


def format_number(value: float) -> str:
    """Format a number for command line arguments.
    
    - Large integers (>=1000): use integer format (e.g., 2000000 not 2e+06)
    - Small numbers (<0.001): use scientific notation (e.g., 1e-04)
    - Integers: no decimal point
    - Floats: use :g format for compact representation
    """
    if value >= 1000 and value == int(value):
        return str(int(value))
    elif value < 0.001:
        return f"{value:.0e}"
    elif value == int(value):
        return str(int(value))
    else:
        return f"{value:g}"


def get_adjacent_configs(cmd: str, vars: List[AblationVar], factor: float, project_dir: Path = None) -> List[AdjacentConfig]:
    """Generate all adjacent configurations."""
    adjacent = []
    
    # Extract script path from command for default value lookup
    script_path = None
    if project_dir:
        script_match = re.search(r'python\s+(\S+\.py)', cmd)
        if script_match:
            script_path = str(project_dir / script_match.group(1))
    
    for var in vars:
        current = get_current_value(cmd, var, script_path)
        
        if var.is_flag:
            # Toggle flag
            new_value = "off" if current == "on" else "on"
            adjacent.append(AdjacentConfig(
                var_name=var.name,
                direction="toggle",
                value=new_value
            ))
        else:
            # Numerical value: try both up and down
            try:
                current_num = float(current)
                
                # Up (multiply by factor)
                up_value = current_num * factor
                up_str = format_number(up_value)
                adjacent.append(AdjacentConfig(
                    var_name=var.name,
                    direction="up",
                    value=up_str
                ))
                
                # Down (divide by factor)
                down_value = current_num / factor
                down_str = format_number(down_value)
                adjacent.append(AdjacentConfig(
                    var_name=var.name,
                    direction="down",
                    value=down_str
                ))
            except (ValueError, TypeError):
                print(f"  Warning: Could not parse {var.name}={current} as number")
    
    return adjacent


def add_suffix(cmd: str, suffix: str) -> str:
    """Add or update suffix in command."""
    if "--suffix" in cmd:
        cmd = re.sub(r'--suffix\s+\S+', f'--suffix {suffix}', cmd)
    else:
        cmd = cmd + f" --suffix {suffix}"
    return cmd


def submit_job(state: ThompsonState, config_key: str, cmd: str, seed: int, 
               project_dir: Path) -> Optional[Dict]:
    """Submit a single job."""
    job_name = f"{state.sweep_name}_i{state.iteration}_{config_key}_s{seed}"
    log_file = Path(state.sweep_dir) / "logs" / job_name
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Add seed and suffix - MUST include seed for unique W&B identification
    cmd = re.sub(r'--seed\s+\d+', '', cmd)  # Remove existing seed
    cmd = cmd + f" --seed {seed}"
    cmd = add_suffix(cmd, f"{state.sweep_name}_i{state.iteration}_{config_key}_s{seed}")
    
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
        iteration=state.iteration,
        config_key=config_key,
        seed=seed,
        cmd=cmd,
    )
    
    script_path = Path(state.sweep_dir) / "scripts" / f"{job_name}.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script)
    
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        job_id = result.stdout.strip().split()[-1]
        return {
            "job_id": job_id,
            "config_key": config_key,
            "seed": seed,
            "cmd": cmd,
            "log_file": str(log_file),
            "status": "pending",
        }
    else:
        print(f"  FAILED to submit: {config_key}, seed={seed}: {result.stderr}")
        return None


def cancel_jobs(job_ids: List[str], graceful: bool = True) -> int:
    """Cancel SLURM jobs.
    
    If graceful=True, send SIGTERM first to allow W&B to clean up,
    then SIGKILL after a short delay if jobs are still running.
    """
    if not job_ids:
        return 0
    
    cancelled = 0
    
    if graceful:
        # First, send SIGTERM to allow graceful shutdown (W&B cleanup)
        for job_id in job_ids:
            subprocess.run(
                ["scancel", "--signal=TERM", job_id],
                capture_output=True,
                text=True
            )
        
        # Wait a few seconds for graceful termination
        print("  Waiting for graceful termination (W&B cleanup)...")
        time.sleep(5)
        
        # Then force kill any remaining jobs
        for job_id in job_ids:
            result = subprocess.run(
                ["scancel", job_id],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                cancelled += 1
    else:
        # Immediate cancellation
        for job_id in job_ids:
            result = subprocess.run(
                ["scancel", job_id],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                cancelled += 1
    
    return cancelled


def signal_handler(signum, frame):
    """Handle Ctrl+C by cancelling all active jobs."""
    global _active_jobs, _sweep_state
    print("\n\n*** Ctrl+C received. Cancelling all active jobs... ***")
    
    if _active_jobs:
        job_ids = [j["job_id"] for j in _active_jobs]
        cancelled = cancel_jobs(job_ids, graceful=True)
        print(f"Cancelled {cancelled} jobs.")
    
    if _sweep_state:
        print(f"State saved to: {_sweep_state.sweep_dir}/state.json")
    
    sys.exit(1)


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


# ============================================================================
# Result Extraction
# ============================================================================

def extract_final_return_from_log(log_file: str) -> Optional[float]:
    """Extract the final episode return from log file via tensorboard."""
    try:
        log_path = Path(f"{log_file}.out")
        if not log_path.exists():
            return None
        
        content = log_path.read_text()
        
        match = re.search(r'Git modification details logged to ([^\s]+)/dacer\.diff', content)
        if not match:
            return None
        
        log_dir = Path(match.group(1))
        if not log_dir.exists():
            return None
        
        # Try tensorboard
        try:
            from tensorboard.backend.event_processing import event_accumulator
            ea = event_accumulator.EventAccumulator(str(log_dir))
            ea.Reload()
            
            if 'sample/episode_return' in ea.Tags().get('scalars', []):
                events = ea.Scalars('sample/episode_return')
                if events:
                    values = [e.value for e in events[-10:]]
                    return np.mean(values)
        except Exception:
            pass
        
        return None
    except Exception:
        return None


def extract_final_return_from_wandb(suffix_pattern: str, project: str = "diffusion_online_rl") -> Optional[float]:
    """Extract final episode return from W&B."""
    if not WANDB_AVAILABLE:
        return None
    
    try:
        api = wandb.Api()
        runs = api.runs(project, order="-created_at")
        
        for run in runs:
            if suffix_pattern in run.name:
                if "sample/episode_return" in run.summary:
                    return run.summary["sample/episode_return"]
        return None
    except Exception:
        return None


def extract_result(job: Dict, use_wandb: bool = True) -> Optional[float]:
    """Extract result from a completed job."""
    result = extract_final_return_from_log(job.get("log_file", ""))
    
    if result is None and use_wandb and WANDB_AVAILABLE:
        cmd = job.get("cmd", "")
        suffix_match = re.search(r'--suffix\s+(\S+)', cmd)
        if suffix_match:
            result = extract_final_return_from_wandb(suffix_match.group(1))
    
    return result


# ============================================================================
# Statistical Analysis
# ============================================================================

def compute_paired_difference_stats(base_results: Dict[int, float], 
                                     adj_results: Dict[int, float]) -> Dict:
    """Compute t-distribution parameters for paired differences."""
    common_seeds = set(base_results.keys()) & set(adj_results.keys())
    
    if len(common_seeds) < 2:
        return {
            "n": len(common_seeds),
            "mean_diff": 0,
            "std_diff": float('inf'),
            "se_diff": float('inf'),
            "prob_better": 0.5,
        }
    
    diffs = [adj_results[s] - base_results[s] for s in common_seeds]
    n = len(diffs)
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs, ddof=1)
    se_diff = std_diff / np.sqrt(n)
    
    # Probability that adjacent is better (mean diff > 0)
    # Using t-distribution
    if se_diff > 0:
        t_stat = mean_diff / se_diff
        prob_better = 1 - stats.t.cdf(0, df=n-1, loc=mean_diff, scale=se_diff)
    else:
        prob_better = 1.0 if mean_diff > 0 else 0.0
    
    return {
        "n": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "se_diff": se_diff,
        "prob_better": prob_better,
        "diffs": diffs,
    }


def thompson_sample(base_results: Dict[int, float],
                    adjacent_results: Dict[str, Dict[int, float]]) -> str:
    """Sample from posterior and return config key with highest sample."""
    samples = {}
    
    for config_key, adj_res in adjacent_results.items():
        stats_dict = compute_paired_difference_stats(base_results, adj_res)
        
        if stats_dict["n"] < 2:
            # Prior: wide normal
            samples[config_key] = np.random.normal(0, 1000)
        else:
            # Posterior: t-distribution
            sample = stats.t.rvs(
                df=stats_dict["n"] - 1,
                loc=stats_dict["mean_diff"],
                scale=stats_dict["se_diff"]
            )
            samples[config_key] = sample
    
    return max(samples, key=samples.get)


# ============================================================================
# State Management
# ============================================================================

def save_state(state: ThompsonState):
    """Save state to JSON file."""
    state_path = Path(state.sweep_dir) / "state.json"
    with open(state_path, 'w') as f:
        json.dump(asdict(state), f, indent=2)


def load_state(state_file: str) -> ThompsonState:
    """Load state from JSON file."""
    with open(state_file, 'r') as f:
        data = json.load(f)
    return ThompsonState(**data)


# ============================================================================
# Main Thompson Sampling Loop
# ============================================================================

def run_thompson_sweep(state: ThompsonState, project_dir: Path) -> Dict:
    """Run the Thompson sampling sweep with parallel job submission."""
    global _active_jobs, _sweep_state
    _sweep_state = state
    
    while True:
        print(f"\n{'=' * 70}")
        print(f"ITERATION {state.iteration}")
        print(f"Base command: {state.current_base_cmd}")
        print(f"{'=' * 70}")
        
        # Generate adjacent configs
        vars = state.get_ablation_vars()
        adjacent_configs = get_adjacent_configs(state.current_base_cmd, vars, state.factor, project_dir)
        
        print(f"\nAdjacent configurations ({len(adjacent_configs)}):")
        for adj in adjacent_configs:
            print(f"  {adj.key()}: {adj.var_name} -> {adj.value}")
        
        # Initialize results for this iteration
        state.base_results = {}
        state.adjacent_results = {adj.key(): {} for adj in adjacent_configs}
        
        # Build command lookup for adjacent configs
        adj_cmds = {}
        for adj in adjacent_configs:
            adj_cmds[adj.key()] = apply_value(
                state.current_base_cmd,
                next(v for v in vars if v.name == adj.var_name),
                adj.value
            )
        
        # Track all active jobs and what seeds we've submitted
        all_jobs = []
        submitted_seeds = {"base": set()}
        for adj in adjacent_configs:
            submitted_seeds[adj.key()] = set()
        
        max_seed_used = state.seed_offset - 1
        
        # Phase 1: Submit initial exploration jobs (initial_seeds for all configs)
        print(f"\n--- Submitting initial jobs ({state.initial_seeds} seeds per config) ---")
        initial_seeds = list(range(state.seed_offset, state.seed_offset + state.initial_seeds))
        
        for seed in initial_seeds:
            # Submit base
            job = submit_job(state, "base", state.current_base_cmd, seed, project_dir)
            if job:
                all_jobs.append(job)
                submitted_seeds["base"].add(seed)
                print(f"  Submitted base, seed={seed}")
            
            # Submit all adjacent configs
            for adj in adjacent_configs:
                job = submit_job(state, adj.key(), adj_cmds[adj.key()], seed, project_dir)
                if job:
                    all_jobs.append(job)
                    submitted_seeds[adj.key()].add(seed)
                    print(f"  Submitted {adj.key()}, seed={seed}")
        
        max_seed_used = state.seed_offset + state.initial_seeds - 1
        
        # Also submit some random exploratory jobs at next seeds to keep pipeline full
        n_exploratory = min(len(adjacent_configs), 4)  # Submit a few extra
        next_seed = max_seed_used + 1
        print(f"\n--- Submitting {n_exploratory} exploratory jobs at seed {next_seed} ---")
        
        # Submit base at next seed
        job = submit_job(state, "base", state.current_base_cmd, next_seed, project_dir)
        if job:
            all_jobs.append(job)
            submitted_seeds["base"].add(next_seed)
            print(f"  Submitted base, seed={next_seed}")
        
        # Submit random adjacent configs at next seed
        import random
        random_adjs = random.sample(adjacent_configs, min(n_exploratory, len(adjacent_configs)))
        for adj in random_adjs:
            job = submit_job(state, adj.key(), adj_cmds[adj.key()], next_seed, project_dir)
            if job:
                all_jobs.append(job)
                submitted_seeds[adj.key()].add(next_seed)
                print(f"  Submitted {adj.key()}, seed={next_seed}")
        
        max_seed_used = next_seed
        
        state.pending_jobs = all_jobs
        _active_jobs = all_jobs  # Update global for signal handler
        save_state(state)
        
        # Main loop: process jobs as they complete, submit new ones via Thompson sampling
        print(f"\n--- Processing jobs ({len(all_jobs)} total) ---")
        collected_job_ids = set()
        import random
        
        def fill_pipeline():
            """Submit jobs to keep pipeline full with pending jobs.
            
            Strategy:
            - Target enough jobs to always have pending jobs in queue
            - For each config, use lowest unused seed for fair comparison
            - Thompson sampling requires initial_seeds base results
            - Mixed exploration/exploitation based on data availability
            """
            nonlocal max_seed_used, all_jobs
            
            statuses = get_job_statuses(all_jobs)
            n_active = len(statuses["pending"]) + len(statuses["running"])
            
            # Target: keep 30+ jobs active to ensure we always have pending jobs
            # This accounts for cluster scheduling delays
            target_active = 30
            max_seed = state.seed_offset + state.max_seeds
            
            # Can use Thompson sampling once we have initial_seeds base results
            use_thompson = len(state.base_results) >= state.initial_seeds
            
            jobs_submitted = 0
            
            while n_active < target_active:
                # Find the next config to submit - use lowest unused seed per config
                best_config = None
                best_seed = None
                best_is_base = False
                
                # Check base first - find lowest unused seed
                for seed in range(state.seed_offset, max_seed):
                    if seed not in submitted_seeds["base"]:
                        best_is_base = True
                        best_seed = seed
                        break
                
                # Check each adjacent config for lowest unused seed
                for adj in adjacent_configs:
                    for seed in range(state.seed_offset, max_seed):
                        if seed not in submitted_seeds[adj.key()]:
                            # This config needs this seed
                            if best_config is None or seed < best_seed:
                                best_config = adj
                                best_seed = seed
                                best_is_base = False
                            elif seed == best_seed and best_config is None:
                                best_config = adj
                                best_is_base = False
                            break  # Found lowest for this config
                
                # Nothing left to submit
                if best_seed is None and best_config is None:
                    break
                
                # Submit base job if it's the best choice or if we need base at this seed
                if best_is_base or (best_seed is not None and best_seed not in submitted_seeds["base"]):
                    seed_to_use = best_seed if best_is_base else best_seed
                    if seed_to_use not in submitted_seeds["base"]:
                        job = submit_job(state, "base", state.current_base_cmd, seed_to_use, project_dir)
                        if job:
                            all_jobs.append(job)
                            submitted_seeds["base"].add(seed_to_use)
                            n_active += 1
                            jobs_submitted += 1
                            max_seed_used = max(max_seed_used, seed_to_use)
                            print(f"  Submitted base, seed={seed_to_use}")
                        if best_is_base:
                            continue
                
                # Select config to submit using Thompson or exploration
                if best_config is None:
                    continue
                
                seed_to_use = best_seed
                
                # Categorize configs based on data availability
                configs_with_data = [a for a in adjacent_configs 
                                     if len(state.adjacent_results.get(a.key(), {})) >= 1]
                configs_without_data = [a for a in adjacent_configs if a not in configs_with_data]
                
                # Decide: Thompson sample or explore?
                selected = None
                if use_thompson and configs_with_data:
                    # Probability of exploration = proportion without data
                    explore_prob = len(configs_without_data) / len(adjacent_configs)
                    
                    if random.random() < explore_prob and configs_without_data:
                        # Random exploration - pick config without data that needs this seed
                        candidates = [a for a in configs_without_data 
                                      if seed_to_use not in submitted_seeds[a.key()]]
                        if candidates:
                            selected = random.choice(candidates)
                            mode = "Exploratory"
                    
                    if selected is None:
                        # Thompson sampling among configs with data
                        candidates = [a for a in configs_with_data 
                                      if seed_to_use not in submitted_seeds[a.key()]]
                        if candidates:
                            filtered_results = {a.key(): state.adjacent_results.get(a.key(), {}) 
                                                for a in candidates}
                            selected_key = thompson_sample(state.base_results, filtered_results)
                            selected = next((a for a in candidates if a.key() == selected_key), 
                                            random.choice(candidates) if candidates else None)
                            mode = "Thompson"
                else:
                    # Pure exploration - pick any config that needs lowest seed
                    candidates = [a for a in adjacent_configs 
                                  if seed_to_use not in submitted_seeds[a.key()]]
                    if candidates:
                        selected = random.choice(candidates)
                        mode = "Exploratory"
                
                if selected and seed_to_use not in submitted_seeds[selected.key()]:
                    job = submit_job(state, selected.key(), adj_cmds[selected.key()], seed_to_use, project_dir)
                    if job:
                        all_jobs.append(job)
                        submitted_seeds[selected.key()].add(seed_to_use)
                        n_active += 1
                        jobs_submitted += 1
                        max_seed_used = max(max_seed_used, seed_to_use)
                        print(f"  {mode}: Submitted {selected.key()}, seed={seed_to_use}")
                else:
                    # Can't find anything to submit
                    break
            
            if jobs_submitted > 0:
                state.pending_jobs = all_jobs
                _active_jobs[:] = all_jobs  # Update global for signal handler
                save_state(state)
            
            return jobs_submitted
        
        while True:
            # Wait for at least one job to complete
            newly_completed = wait_for_any_job(all_jobs)
            
            # Collect results from newly completed jobs
            for job in newly_completed:
                if job["job_id"] in collected_job_ids:
                    continue
                collected_job_ids.add(job["job_id"])
                
                result = extract_result(job, use_wandb=state.use_wandb)
                if result is not None:
                    if job["config_key"] == "base":
                        state.base_results[job["seed"]] = result
                        print(f"  Completed: base, seed={job['seed']}: {result:.2f}")
                    else:
                        state.adjacent_results[job["config_key"]][job["seed"]] = result
                        print(f"  Completed: {job['config_key']}, seed={job['seed']}: {result:.2f}")
                else:
                    print(f"  Completed: {job['config_key']}, seed={job['seed']}: MISSING")
            
            save_state(state)
            
            # Always try to fill the pipeline (Thompson or exploratory based on available data)
            fill_pipeline()
            
            # Check if we have enough data for statistics (need at least initial_seeds for base)
            if len(state.base_results) < state.initial_seeds:
                continue
            
            # Compute stats for all adjacent configs
            print("\n  Current statistics:")
            best_prob = 0
            best_config = None
            
            for adj in adjacent_configs:
                stats_dict = compute_paired_difference_stats(
                    state.base_results, 
                    state.adjacent_results[adj.key()]
                )
                if stats_dict['n'] >= 2:
                    print(f"    {adj.key()}: n={stats_dict['n']}, "
                          f"mean_diff={stats_dict['mean_diff']:.2f}, "
                          f"P(better)={stats_dict['prob_better']:.3f}")
                    
                    if stats_dict['prob_better'] > best_prob:
                        best_prob = stats_dict['prob_better']
                        best_config = adj
            
            # Check for convergence (found a better config)
            # Require both confidence threshold AND minimum number of paired comparisons
            best_n = 0
            if best_config:
                best_stats = compute_paired_difference_stats(
                    state.base_results, 
                    state.adjacent_results[best_config.key()]
                )
                best_n = best_stats['n']
            
            if best_prob >= state.confidence_threshold and best_n >= state.min_seeds_for_promotion:
                print(f"\n*** Found better config: {best_config.key()} "
                      f"(P={best_prob:.3f} >= {state.confidence_threshold}, n={best_n}) ***")
                
                # Cancel remaining jobs
                statuses = get_job_statuses(all_jobs)
                pending_ids = [j["job_id"] for j in statuses["pending"]]
                running_ids = [j["job_id"] for j in statuses["running"]]
                if pending_ids or running_ids:
                    print(f"  Cancelling {len(pending_ids)} pending and {len(running_ids)} running jobs...")
                    cancel_jobs(pending_ids + running_ids)
                
                # Update base command
                var = next(v for v in vars if v.name == best_config.var_name)
                state.current_base_cmd = apply_value(state.current_base_cmd, var, best_config.value)
                
                state.history.append({
                    "iteration": state.iteration,
                    "promoted": best_config.key(),
                    "new_base_cmd": state.current_base_cmd,
                    "prob_better": best_prob,
                    "timestamp": datetime.now().isoformat(),
                })
                
                state.iteration += 1
                state.seed_offset = max_seed_used + 1
                save_state(state)
                break  # Start new iteration
            
            # Check if we've hit max seeds
            n_seeds_used = len(state.base_results)
            if n_seeds_used >= state.max_seeds:
                # Wait for all remaining jobs
                statuses = get_job_statuses(all_jobs)
                if statuses["pending"] or statuses["running"]:
                    print(f"  Waiting for remaining {len(statuses['pending']) + len(statuses['running'])} jobs...")
                    wait_for_all_jobs(all_jobs)
                    # Collect any remaining results
                    for job in all_jobs:
                        if job["job_id"] not in collected_job_ids:
                            collected_job_ids.add(job["job_id"])
                            result = extract_result(job, use_wandb=state.use_wandb)
                            if result is not None:
                                if job["config_key"] == "base":
                                    state.base_results[job["seed"]] = result
                                else:
                                    state.adjacent_results[job["config_key"]][job["seed"]] = result
                
                # Recompute best_prob with all data
                best_prob = 0
                best_n = 0
                for adj in adjacent_configs:
                    stats_dict = compute_paired_difference_stats(
                        state.base_results, 
                        state.adjacent_results[adj.key()]
                    )
                    if stats_dict['prob_better'] > best_prob:
                        best_prob = stats_dict['prob_better']
                        best_config = adj
                        best_n = stats_dict['n']
                
                if best_prob >= state.confidence_threshold and best_n >= state.min_seeds_for_promotion:
                    # Found better config after collecting all results
                    print(f"\n*** Found better config: {best_config.key()} "
                          f"(P={best_prob:.3f} >= {state.confidence_threshold}, n={best_n}) ***")
                    var = next(v for v in vars if v.name == best_config.var_name)
                    state.current_base_cmd = apply_value(state.current_base_cmd, var, best_config.value)
                    state.history.append({
                        "iteration": state.iteration,
                        "promoted": best_config.key(),
                        "new_base_cmd": state.current_base_cmd,
                        "prob_better": best_prob,
                        "timestamp": datetime.now().isoformat(),
                    })
                    state.iteration += 1
                    state.seed_offset = max_seed_used + 1
                    save_state(state)
                    break
                
                print(f"\n*** No config reached {state.confidence_threshold} confidence "
                      f"after {state.max_seeds} seeds. Sweep complete. ***")
                
                # Final validation
                print("\n--- Final Validation ---")
                final_stats = run_final_validation(state, project_dir)
                
                state.history.append({
                    "iteration": state.iteration,
                    "converged": True,
                    "final_cmd": state.current_base_cmd,
                    "final_stats": final_stats,
                    "timestamp": datetime.now().isoformat(),
                })
                save_state(state)
                
                return {
                    "final_cmd": state.current_base_cmd,
                    "final_stats": final_stats,
                    "iterations": state.iteration,
                }
            
            # Check if there are still jobs running - if so, continue waiting
            # (fill_pipeline already handled above, so just continue the loop)
            statuses = get_job_statuses(all_jobs)
            if statuses["pending"] or statuses["running"]:
                continue


def get_job_statuses(jobs: List[Dict]) -> Dict[str, List[Dict]]:
    """Get statuses of all jobs, checking each job only once."""
    statuses = {"pending": [], "running": [], "completed": []}
    for job in jobs:
        status = check_job_status(job["job_id"])
        if status == "pending":
            statuses["pending"].append(job)
        elif status == "running":
            statuses["running"].append(job)
        else:
            statuses["completed"].append(job)
    return statuses


def wait_for_all_jobs(jobs: List[Dict], check_interval: int = 30):
    """Wait for all jobs to complete."""
    while True:
        statuses = get_job_statuses(jobs)
        n_pending = len(statuses["pending"])
        n_running = len(statuses["running"])
        n_completed = len(statuses["completed"])
        
        if n_pending == 0 and n_running == 0:
            break
        
        print(f"  Status: {n_pending} pending, {n_running} running, {n_completed} completed")
        time.sleep(check_interval)


def wait_for_any_job(jobs: List[Dict], check_interval: int = 10) -> List[Dict]:
    """Wait for at least one job to complete. Returns list of newly completed jobs."""
    previously_completed = set()
    for job in jobs:
        if check_job_status(job["job_id"]) == "completed":
            previously_completed.add(job["job_id"])
    
    while True:
        statuses = get_job_statuses(jobs)
        newly_completed = [j for j in statuses["completed"] 
                          if j["job_id"] not in previously_completed]
        
        if newly_completed:
            n_pending = len(statuses["pending"])
            n_running = len(statuses["running"])
            n_completed = len(statuses["completed"])
            print(f"  Status: {n_pending} pending, {n_running} running, {n_completed} completed")
            return newly_completed
        
        n_pending = len(statuses["pending"])
        n_running = len(statuses["running"])
        n_completed = len(statuses["completed"])
        print(f"  Status: {n_pending} pending, {n_running} running, {n_completed} completed")
        time.sleep(check_interval)


def collect_results(state: ThompsonState, jobs: List[Dict]):
    """Collect results from completed jobs."""
    for job in jobs:
        result = extract_result(job, use_wandb=state.use_wandb)
        
        if result is not None:
            if job["config_key"] == "base":
                state.base_results[job["seed"]] = result
                print(f"  base, seed={job['seed']}: {result:.2f}")
            else:
                if job["config_key"] not in state.adjacent_results:
                    state.adjacent_results[job["config_key"]] = {}
                state.adjacent_results[job["config_key"]][job["seed"]] = result
                print(f"  {job['config_key']}, seed={job['seed']}: {result:.2f}")
        else:
            print(f"  {job['config_key']}, seed={job['seed']}: MISSING")


def run_final_validation(state: ThompsonState, project_dir: Path, 
                         num_seeds: int = 5) -> Dict:
    """Run final validation on fresh seeds."""
    print(f"\nRunning final validation on {num_seeds} fresh seeds...")
    
    seeds = list(range(state.seed_offset, state.seed_offset + num_seeds))
    jobs = []
    
    for seed in seeds:
        job = submit_job(state, "final", state.current_base_cmd, seed, project_dir)
        if job:
            jobs.append(job)
            print(f"  Submitted final, seed={seed}")
    
    wait_for_all_jobs(jobs)
    
    returns = []
    for job in jobs:
        result = extract_result(job, use_wandb=state.use_wandb)
        if result is not None:
            returns.append(result)
            print(f"  seed={job['seed']}: {result:.2f}")
        else:
            print(f"  seed={job['seed']}: MISSING")
    
    if returns:
        return {
            "mean": np.mean(returns),
            "std": np.std(returns, ddof=1) if len(returns) > 1 else 0,
            "se": np.std(returns, ddof=1) / np.sqrt(len(returns)) if len(returns) > 1 else 0,
            "n": len(returns),
        }
    else:
        return {"mean": float('-inf'), "std": 0, "se": 0, "n": 0}


# ============================================================================
# Parallel Sweeps
# ============================================================================

def run_parallel_sweeps(base_cmd: str, vars: List[AblationVar], 
                        split_var: AblationVar, sweep_name: str,
                        project_dir: Path, args) -> List[Dict]:
    """Run multiple sweeps in parallel with different values of split_var."""
    
    sweep_dir = project_dir / "sweeps" / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine split values
    if split_var.is_flag:
        split_values = ["off", "on"]
    else:
        current = get_current_value(base_cmd, split_var)
        try:
            current_num = float(current)
            split_values = [
                str(current_num / split_var.factor),
                current,
                str(current_num * split_var.factor),
            ]
        except:
            split_values = [current]
    
    print(f"Running {len(split_values)} parallel sweeps for {split_var.name}:")
    for v in split_values:
        print(f"  {split_var.name}={v}")
    
    # Create sub-sweeps
    sub_sweep_states = []
    for i, split_val in enumerate(split_values):
        sub_name = f"{sweep_name}_{split_var.name}_{split_val}"
        sub_dir = sweep_dir / sub_name
        sub_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply split value to base command
        sub_cmd = apply_value(base_cmd, split_var, split_val)
        
        sub_state = ThompsonState(
            base_cmd=sub_cmd,
            ablation_vars=[asdict(v) for v in vars],
            sweep_name=sub_name,
            sweep_dir=str(sub_dir),
            factor=args.factor,
            confidence_threshold=args.confidence,
            max_seeds=args.max_seeds,
            initial_seeds=args.initial_seeds,
            min_seeds_for_promotion=args.min_seeds,
            current_base_cmd=sub_cmd,
            partition=args.partition,
            gpu_type=args.gpu_type,
            num_gpus=args.num_gpus,
            cpus=args.cpus,
            mem=args.mem,
            time_limit=args.time,
            use_wandb=not args.no_wandb,
            seed_offset=i * 100,  # Offset seeds for each parallel sweep
        )
        save_state(sub_state)
        sub_sweep_states.append(sub_state)
    
    # Run sub-sweeps (sequentially for now, could parallelize)
    results = []
    for sub_state in sub_sweep_states:
        print(f"\n{'#' * 70}")
        print(f"Starting sub-sweep: {sub_state.sweep_name}")
        print(f"{'#' * 70}")
        
        result = run_thompson_sweep(sub_state, project_dir)
        results.append({
            "sweep_name": sub_state.sweep_name,
            "split_value": get_current_value(sub_state.base_cmd, split_var),
            **result
        })
    
    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Thompson sampling sweep controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Main options
    parser.add_argument("--cmd", type=str,
                        help="Base python command to run")
    parser.add_argument("--var", action="append", nargs="+",
                        metavar=("NAME", "[flag]"),
                        help="Ablation variable. Add 'flag' for boolean flags.")
    parser.add_argument("--factor", type=float, default=2.0,
                        help="Adjacency factor for numerical values (default: 2)")
    parser.add_argument("--sweep-name", type=str, default=None,
                        help="Name for this sweep")
    
    # Thompson sampling options
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Confidence threshold for promotion (default: 0.95)")
    parser.add_argument("--max-seeds", type=int, default=10,
                        help="Maximum seeds per iteration (default: 10)")
    parser.add_argument("--initial-seeds", type=int, default=2,
                        help="Initial seeds for exploration (default: 2)")
    parser.add_argument("--min-seeds", type=int, default=4,
                        help="Minimum paired comparisons required for promotion (default: 4)")
    
    # Parallel sweep option
    parser.add_argument("--split", nargs="+", metavar=("NAME", "[flag]"),
                        help="Variable to split on for parallel sweeps")
    
    # Resume/status
    parser.add_argument("--resume", type=str, metavar="STATE_FILE",
                        help="Resume a previous sweep")
    parser.add_argument("--status", type=str, metavar="STATE_FILE",
                        help="Check status of a sweep")
    
    # SLURM options
    parser.add_argument("--partition", "-p", type=str, default="gpu_requeue")
    parser.add_argument("--gpu-type", type=str, default="nvidia_h200")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--cpus", "-c", type=int, default=2)
    parser.add_argument("--mem", type=str, default="32G")
    parser.add_argument("--time", "-t", type=str, default="0-6:00")
    
    # Result extraction
    parser.add_argument("--no-wandb", action="store_true",
                        help="Don't use W&B for result extraction")
    
    args = parser.parse_args()
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    project_dir = Path(__file__).parent.parent.resolve()
    
    # Handle status check
    if args.status:
        state = load_state(args.status)
        print(f"Sweep: {state.sweep_name}")
        print(f"Iteration: {state.iteration}")
        print(f"Base command: {state.current_base_cmd}")
        print(f"Seeds used: {len(state.base_results)}")
        print(f"History: {len(state.history)} events")
        return
    
    # Handle resume
    if args.resume:
        state = load_state(args.resume)
        print(f"Resuming sweep: {state.sweep_name}")
        result = run_thompson_sweep(state, project_dir)
    else:
        # New sweep
        if not args.cmd or not args.var:
            parser.error("--cmd and at least one --var are required for new sweeps")
        
        # Parse variables
        vars = []
        for var_spec in args.var:
            name = var_spec[0]
            is_flag = len(var_spec) > 1 and var_spec[1].lower() == "flag"
            vars.append(AblationVar(name=name, is_flag=is_flag, factor=args.factor))
        
        sweep_name = args.sweep_name or datetime.now().strftime("thompson_%Y%m%d_%H%M%S")
        
        # Check for parallel sweep
        if args.split:
            split_name = args.split[0]
            split_is_flag = len(args.split) > 1 and args.split[1].lower() == "flag"
            split_var = AblationVar(name=split_name, is_flag=split_is_flag, factor=args.factor)
            
            results = run_parallel_sweeps(args.cmd, vars, split_var, sweep_name,
                                          project_dir, args)
            
            print("\n" + "#" * 70)
            print("PARALLEL SWEEP COMPLETE")
            print("#" * 70)
            for r in results:
                print(f"\n{r['sweep_name']}:")
                print(f"  Final command: {r['final_cmd']}")
                if r['final_stats']['n'] > 0:
                    print(f"  Performance: {r['final_stats']['mean']:.2f} "
                          f"± {r['final_stats']['se']:.2f} (SE)")
        else:
            # Single sweep
            sweep_dir = project_dir / "sweeps" / sweep_name
            sweep_dir.mkdir(parents=True, exist_ok=True)
            
            state = ThompsonState(
                base_cmd=args.cmd,
                ablation_vars=[asdict(v) for v in vars],
                sweep_name=sweep_name,
                sweep_dir=str(sweep_dir),
                factor=args.factor,
                confidence_threshold=args.confidence,
                max_seeds=args.max_seeds,
                initial_seeds=args.initial_seeds,
                min_seeds_for_promotion=args.min_seeds,
                current_base_cmd=args.cmd,
                partition=args.partition,
                gpu_type=args.gpu_type,
                num_gpus=args.num_gpus,
                cpus=args.cpus,
                mem=args.mem,
                time_limit=args.time,
                use_wandb=not args.no_wandb,
            )
            save_state(state)
            
            print(f"Created new Thompson sweep: {sweep_name}")
            print(f"State file: {sweep_dir}/state.json")
            
            result = run_thompson_sweep(state, project_dir)
            
            print("\n" + "#" * 70)
            print("SWEEP COMPLETE")
            print("#" * 70)
            print(f"\nFinal command: {result['final_cmd']}")
            if result['final_stats']['n'] > 0:
                print(f"Performance: {result['final_stats']['mean']:.2f} "
                      f"± {result['final_stats']['se']:.2f} (SE)")
            print(f"Iterations: {result['iterations']}")


if __name__ == "__main__":
    main()
