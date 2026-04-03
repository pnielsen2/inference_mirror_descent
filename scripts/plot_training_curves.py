#!/usr/bin/env python3
"""
Plot mean and SE of episode reward curves for different experiment configurations.
Fetches data from wandb API on the fly.
"""

import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import wandb

# Define the 6 configurations with identifiable parameters (used for wandb filtering)
CONFIGS = {
    "DPMD baseline (long LR)": {
        "alg": "dpmd",
        "dpmd_constant_weight": False,
        "dpmd_long_lr_schedule": True,
    },
    "MALA-guided (const. weight)": {
        "alg": "dpmd",
        "dpmd_constant_weight": True,
        "dpmd_no_entropy_tuning": True,
        "num_particles": 1,
        "tfg_eta": 16.0,
        "mala_steps": 2,
        "buffer_size": 200000,
    },
    "Best-of-N BC (128 particles)": {
        "alg": "dpmd",
        "dpmd_constant_weight": True,
        "num_particles": 128,
        "particle_selection_lambda": 64,
    },
    "MB-PC (determ. dyn)": {
        "alg": "dpmd_mb_pc",
        "pc_deterministic_dyn": True,
        "pc_H_plan": 1,
    },
    "MB-PC (diffusion dyn)": {
        "alg": "dpmd_mb_pc",
        "pc_deterministic_dyn": False,
        "sprime_refresh_steps": 6,
        "bprop_refresh_steps": 6,
    },
    "MB-PC planner (H=2)": {
        "alg": "dpmd_mb_pc",
        "pc_deterministic_dyn": True,
        "pc_H_plan": 2,
        "pc_joint_seq": False,
        "beta_schedule_type": "cosine",
        "buffer_size": 200000,
        "tfg_eta": 2.0,
    },
}


def matches_config(run_config: dict, target_config: dict) -> bool:
    """Check if a run config matches the target config requirements."""
    for key, expected_value in target_config.items():
        if expected_value is None:
            continue
        run_value = run_config.get(key)
        
        # Handle numeric comparisons with tolerance
        if isinstance(expected_value, float):
            if run_value is None or abs(float(run_value) - expected_value) > 1e-6:
                return False
        elif run_value != expected_value:
            return False
    return True


def identify_config(run_config: dict) -> str:
    """Identify which of the 6 configurations a run belongs to."""
    alg = run_config.get("alg")
    
    if alg == "dpmd":
        if (run_config.get("dpmd_long_lr_schedule") and 
            not run_config.get("dpmd_constant_weight") and
            not run_config.get("fix_q_norm_bug")):
            return "DPMD baseline (long LR)"
        
        if (run_config.get("dpmd_constant_weight") and 
            run_config.get("num_particles") == 1 and
            run_config.get("dpmd_no_entropy_tuning") and
            abs(run_config.get("tfg_eta", 0) - 16.0) < 0.1 and
            run_config.get("mala_steps") == 2):
            return "MALA-guided (const. weight)"
        
        if (run_config.get("dpmd_constant_weight") and 
            run_config.get("num_particles") == 128 and
            run_config.get("particle_selection_lambda") == 64):
            return "Boltzmann selection (N=128)"
    
    elif alg == "dpmd_mb_pc":
        pc_det = run_config.get("pc_deterministic_dyn", False)
        pc_H = run_config.get("pc_H_plan", 1)
        pc_joint = run_config.get("pc_joint_seq", False)
        sprime_refresh = run_config.get("sprime_refresh_steps", 0)
        bprop_refresh = run_config.get("bprop_refresh_steps", 0)
        
        beta_type = run_config.get("beta_schedule_type", "linear")
        buffer_size = run_config.get("buffer_size", 100000)
        tfg_eta = run_config.get("tfg_eta", 0)
        if (pc_det and pc_H == 2 and not pc_joint and beta_type == "cosine" 
            and buffer_size == 200000 and abs(tfg_eta - 2.0) < 0.1):
            return "MB-PC planner (H=2)"
        
        if not pc_det and sprime_refresh == 6 and bprop_refresh == 6:
            return "MB-PC (diffusion dyn)"
        
        if pc_det and pc_H == 1:
            return "MB-PC (determ. dyn)"
    
    return None


def get_training_curve_from_wandb(run, env_name=None, max_steps=1000000, step_interval=5000):
    """Fetch training curve from wandb run history."""
    if env_name is None:
        env_name = run.config.get("env", "env")
    new_key = f"episode_return/{env_name}"
    legacy_key = "sample/episode_return"
    history = run.history(keys=[new_key, legacy_key, "_step"], samples=5000)

    if not history.empty and new_key in history.columns and history[new_key].notna().any():
        metric_key = new_key
    elif not history.empty and legacy_key in history.columns and history[legacy_key].notna().any():
        metric_key = legacy_key
    else:
        return None
    
    history = history.dropna(subset=[metric_key])
    if len(history) == 0:
        return None
    
    steps = history["_step"].values
    values = history[metric_key].values
    
    # Interpolate to regular intervals
    target_steps = np.arange(0, max_steps + step_interval, step_interval)
    interp_values = np.interp(target_steps, steps, values, left=np.nan, right=values[-1])
    
    return target_steps, interp_values


def main():
    api = wandb.Api(timeout=120)
    
    env_name = "HalfCheetah-v4"
    valid_seeds = {0, 1, 2, 3, 4}
    
    # Fetch all finished runs for this environment
    print(f"Fetching runs for {env_name} from wandb...")
    filters = {
        "config.env": env_name,
        "state": "finished",
    }
    runs = api.runs("diffusion_online_rl", filters=filters)
    
    # Group runs by configuration
    runs_by_config = defaultdict(dict)  # config_name -> {seed: run}
    
    for run in runs:
        config = run.config
        config_name = identify_config(config)
        if config_name is None:
            continue
        
        seed = config.get("seed")
        if seed is None or seed not in valid_seeds:
            continue
        
        # Keep most recent run per seed
        if seed not in runs_by_config[config_name] or run.created_at > runs_by_config[config_name][seed].created_at:
            runs_by_config[config_name][seed] = run
    
    # Print summary
    print("Found runs per configuration:")
    for config_name, seed_runs in runs_by_config.items():
        print(f"  {config_name}: {len(seed_runs)} runs, seeds={sorted(seed_runs.keys())}")
    
    # Fetch training curves
    curves_by_config = {}
    for config_name, seed_runs in runs_by_config.items():
        curves = []
        for seed in sorted(seed_runs.keys()):
            run = seed_runs[seed]
            print(f"  Fetching curve for {config_name} seed {seed}...")
            result = get_training_curve_from_wandb(run)
            if result is not None:
                curves.append(result)
        curves_by_config[config_name] = curves
    
    # Define order and colors for better visualization
    config_order = [
        "DPMD baseline (long LR)",
        "MALA-guided (const. weight)",
        "Boltzmann selection (N=128)",
        "MB-PC (determ. dyn)",
        "MB-PC (diffusion dyn)",
        "MB-PC planner (H=2)",
    ]
    
    colors = {
        "DPMD baseline (long LR)": "#1f77b4",
        "MALA-guided (const. weight)": "#ff7f0e",
        "Boltzmann selection (N=128)": "#2ca02c",
        "MB-PC (determ. dyn)": "#d62728",
        "MB-PC (diffusion dyn)": "#9467bd",
        "MB-PC planner (H=2)": "#8c564b",
    }
    
    # Create plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 5))
    
    for config_name in config_order:
        curves = curves_by_config.get(config_name, [])
        if len(curves) == 0:
            continue
        
        # All curves share the same target_steps from interpolation
        common_steps = curves[0][0]
        interpolated = np.array([v for _, v in curves])
        
        mean = np.nanmean(interpolated, axis=0)
        se = np.nanstd(interpolated, axis=0) / np.sqrt(len(interpolated))
        
        color = colors.get(config_name, "#333333")
        ax.plot(common_steps / 1e6, mean, label=config_name, color=color, linewidth=2)
        ax.fill_between(common_steps / 1e6, mean - se, mean + se, alpha=0.2, color=color)
    
    ax.set_xlabel("Environment Steps (millions)", fontsize=12)
    ax.set_ylabel("Average Episode Return", fontsize=12)
    ax.set_title("HalfCheetah-v4 Training Curves (Mean ± SE, n=5 seeds)", fontsize=14)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)
    
    ax.minorticks_on()
    ax.grid(which='minor', alpha=0.1)
    
    # Save plot
    output_dir = Path("/n/home09/pnielsen/inference_mirror_descent/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_curves.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Also show final returns
    print("\nFinal returns (mean ± SE):")
    for config_name, curves in sorted(curves_by_config.items()):
        if len(curves) == 0:
            continue
        final_returns = [v[-1] for _, v in curves]
        mean = np.mean(final_returns)
        se = np.std(final_returns) / np.sqrt(len(final_returns))
        print(f"  {config_name}: {mean:.1f} ± {se:.1f}")


if __name__ == "__main__":
    main()
