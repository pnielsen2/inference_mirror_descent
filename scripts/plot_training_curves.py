#!/usr/bin/env python3
"""
Plot mean and SE of episode reward curves for different experiment configurations.
"""

import os
from pathlib import Path
from collections import defaultdict
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tbparse import SummaryReader

# Define the 6 configurations with identifiable parameters
CONFIGS = {
    "DPMD baseline (long LR)": {
        "alg": "dpmd",
        "dpmd_constant_weight": False,
        "dpmd_long_lr_schedule": True,
        "num_particles": None,  # don't care
        "tfg_lambda": None,
        "mala_steps": None,
        "pc_deterministic_dyn": None,
        "pc_H_plan": None,
        "sprime_refresh_steps": None,
    },
    "MALA-guided (const. weight)": {
        "alg": "dpmd",
        "dpmd_constant_weight": True,
        "dpmd_no_entropy_tuning": True,
        "num_particles": 1,
        "tfg_lambda": 16.0,
        "mala_steps": 2,
        "buffer_size": 200000,
    },
    "Best-of-N BC (128 particles)": {
        "alg": "dpmd",
        "dpmd_constant_weight": True,
        "num_particles": 128,
        "particle_selection_lambda": 64,
        # No mala_steps or tfg_lambda requirements for this one
    },
    "MB-PC (determ. dyn)": {
        "alg": "dpmd_mb_pc",
        "pc_deterministic_dyn": True,
        "pc_H_plan": 1,
        # No sprime_refresh_steps requirement
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
        "pc_joint_seq": True,
    },
}


def load_config(run_dir: Path) -> dict:
    """Load config.yaml from a run directory."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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
        # Check for baseline DPMD with long LR schedule (WITHOUT fix_q_norm_bug)
        if (run_config.get("dpmd_long_lr_schedule") and 
            not run_config.get("dpmd_constant_weight") and
            not run_config.get("fix_q_norm_bug")):
            return "DPMD baseline (long LR)"
        
        # Check for MALA-guided (constant weight, 1 particle, tfg_lambda=16, mala_steps=2)
        if (run_config.get("dpmd_constant_weight") and 
            run_config.get("num_particles") == 1 and
            run_config.get("dpmd_no_entropy_tuning") and
            abs(run_config.get("tfg_lambda", 0) - 16.0) < 0.1 and
            run_config.get("mala_steps") == 2):
            return "MALA-guided (const. weight)"
        
        # Check for Boltzmann selection (128 particles, constant weight)
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
        
        # Check for planner H=2
        if pc_det and pc_H == 2 and pc_joint:
            return "MB-PC planner (H=2)"
        
        # Check for diffusion dynamics
        if not pc_det and sprime_refresh == 6 and bprop_refresh == 6:
            return "MB-PC (diffusion dyn)"
        
        # Check for deterministic dynamics H=1
        if pc_det and pc_H == 1:
            return "MB-PC (determ. dyn)"
    
    return None


def get_training_curve(run_dir: Path) -> pd.DataFrame:
    """Extract training curve from TensorBoard events."""
    try:
        reader = SummaryReader(str(run_dir), pivot=True)
        df = reader.scalars
        
        if df is None or df.empty:
            return None
        
        # Look for avg_ret or similar metric
        if 'avg_ret' in df.columns:
            result = df[['step', 'avg_ret']].dropna().copy()
        elif 'eval/avg_return' in df.columns:
            result = df[['step', 'eval/avg_return']].dropna().copy()
            result = result.rename(columns={'eval/avg_return': 'avg_ret'})
        else:
            # Try to find any return-related column
            found = False
            for col in df.columns:
                if 'ret' in col.lower() or 'return' in col.lower() or 'reward' in col.lower():
                    result = df[['step', col]].dropna().copy()
                    result = result.rename(columns={col: 'avg_ret'})
                    found = True
                    break
            if not found:
                return None
        
        # Convert to numeric, handling any object types
        result['step'] = pd.to_numeric(result['step'], errors='coerce')
        result['avg_ret'] = pd.to_numeric(result['avg_ret'], errors='coerce')
        result = result.dropna()
        
        # Sort by step
        result = result.sort_values('step').reset_index(drop=True)
        
        return result
    except Exception as e:
        print(f"Error reading {run_dir}: {e}")
        return None


def main():
    logs_dir = Path("/n/home09/pnielsen/inference_mirror_descent/logs/HalfCheetah-v4")
    
    # Group runs by configuration
    runs_by_config = defaultdict(list)
    
    # Only include runs from 2025-12-18 (today's runs)
    date_filter = "2025-12-18"
    
    for run_dir in logs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        # Filter by date
        if date_filter not in run_dir.name:
            continue
        
        config = load_config(run_dir)
        if config is None:
            continue
        
        config_name = identify_config(config)
        if config_name is not None:
            curve = get_training_curve(run_dir)
            if curve is not None and len(curve) > 0:
                runs_by_config[config_name].append({
                    'dir': run_dir,
                    'seed': config.get('seed'),
                    'curve': curve,
                    'timestamp': run_dir.name,  # for deduplication
                })
    
    # Deduplicate by seed - keep only the most recent run for each seed
    # Only include seeds 0-4 (the 5 intended runs)
    valid_seeds = {0, 1, 2, 3, 4}
    
    for config_name in runs_by_config:
        runs = runs_by_config[config_name]
        # Group by seed
        by_seed = defaultdict(list)
        for run in runs:
            if run['seed'] in valid_seeds:
                by_seed[run['seed']].append(run)
        
        # Keep only the most recent run for each seed
        deduped = []
        for seed, seed_runs in by_seed.items():
            # Sort by timestamp (descending) and take the first
            seed_runs.sort(key=lambda x: x['timestamp'], reverse=True)
            deduped.append(seed_runs[0])
        
        runs_by_config[config_name] = deduped
    
    # Print summary
    print("Found runs per configuration:")
    for config_name, runs in runs_by_config.items():
        seeds = [r['seed'] for r in runs]
        print(f"  {config_name}: {len(runs)} runs, seeds={seeds}")
    
    # Define order and colors for better visualization
    config_order = [
        "DPMD baseline (long LR)",
        "MALA-guided (const. weight)",
        "Boltzmann selection (N=128)",
        "MB-PC (determ. dyn)",
        "MB-PC (diffusion dyn)",
        "MB-PC planner (H=2)",
    ]
    
    # Nice color palette
    colors = {
        "DPMD baseline (long LR)": "#1f77b4",        # blue
        "MALA-guided (const. weight)": "#ff7f0e",    # orange
        "Boltzmann selection (N=128)": "#2ca02c",    # green
        "MB-PC (determ. dyn)": "#d62728",            # red
        "MB-PC (diffusion dyn)": "#9467bd",          # purple
        "MB-PC planner (H=2)": "#8c564b",            # brown
    }
    
    # Create plot with better styling (wider for full-width figure in paper)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 5))
    
    for config_name in config_order:
        if config_name not in runs_by_config:
            continue
        runs = runs_by_config[config_name]
        if len(runs) == 0:
            continue
        
        # Get all curves and interpolate to common steps
        all_curves = [r['curve'] for r in runs]
        
        # Find common step range
        min_step = max(curve['step'].min() for curve in all_curves)
        max_step = min(curve['step'].max() for curve in all_curves)
        
        # Create common step grid
        common_steps = np.linspace(min_step, max_step, 200)
        
        # Interpolate each curve to common steps
        interpolated = []
        for curve in all_curves:
            interp_values = np.interp(common_steps, curve['step'].values, curve['avg_ret'].values)
            interpolated.append(interp_values)
        
        interpolated = np.array(interpolated)
        
        # Calculate mean and SE
        mean = np.mean(interpolated, axis=0)
        se = np.std(interpolated, axis=0) / np.sqrt(len(interpolated))
        
        # Plot
        color = colors.get(config_name, "#333333")
        ax.plot(common_steps / 1e6, mean, label=config_name, color=color, linewidth=2)
        ax.fill_between(common_steps / 1e6, mean - se, mean + se, alpha=0.2, color=color)
    
    ax.set_xlabel("Environment Steps (millions)", fontsize=12)
    ax.set_ylabel("Average Episode Return", fontsize=12)
    ax.set_title("HalfCheetah-v4 Training Curves (Mean ± SE, n=5 seeds)", fontsize=14)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.0)
    
    # Add minor gridlines
    ax.minorticks_on()
    ax.grid(which='minor', alpha=0.1)
    
    # Save plot
    output_path = Path("/n/home09/pnielsen/inference_mirror_descent/training_curves.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Also show final returns
    print("\nFinal returns (mean ± SE):")
    for config_name, runs in sorted(runs_by_config.items()):
        if len(runs) == 0:
            continue
        final_returns = [r['curve']['avg_ret'].iloc[-1] for r in runs]
        mean = np.mean(final_returns)
        se = np.std(final_returns) / np.sqrt(len(final_returns))
        print(f"  {config_name}: {mean:.1f} ± {se:.1f}")


if __name__ == "__main__":
    main()
