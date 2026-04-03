#!/usr/bin/env python
"""
Plot training curves for model-free DPMD experiments across MuJoCo environments.
Queries wandb for runs matching specific command patterns and generates training curve plots.
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict
import pandas as pd
from datetime import datetime
import os

# MuJoCo environments to support
MUJOCO_ENVS = [
    "HalfCheetah-v4",
    "Hopper-v4",
    "Walker2d-v4",
    "Ant-v4",
    "Humanoid-v4",
]

# Target command pattern for model-free MALA-guided DPMD
# (without seed and tfg_eta which may vary)
MODEL_FREE_TARGET_ARGS = {
    "alg": "dpmd",
    "dpmd_constant_weight": True,
    "num_particles": 1,
    "mala_steps": 2,
    "q_critic_agg": "mean",
    "beta_schedule_type": "cosine",
    "beta_schedule_scale": 1,
    "dpmd_no_entropy_tuning": True,
    "buffer_size": 200000,
    "x0_hat_clip_radius": 3.0,
    "mala_adapt_rate": 0.2,
    "mala_per_level_eta": True,
    "q_td_huber_width": 30.0,
    "update_per_iteration": 4,
    "mala_guided_predictor": True,
    "lr_q": 0.00015,
}


def run_matches_target(run_config, target_args, env_name=None, tfg_eta=None):
    """Check if a run's config matches the target args."""
    for key, target_val in target_args.items():
        run_val = run_config.get(key)
        if isinstance(target_val, bool):
            if run_val != target_val:
                return False
        elif isinstance(target_val, (int, float)):
            if run_val is None:
                return False
            if abs(float(run_val) - float(target_val)) > 1e-6:
                return False
        elif run_val != target_val:
            return False
    
    # Check env if specified
    if env_name is not None:
        if run_config.get("env") != env_name:
            return False
    
    # Check tfg_eta if specified
    if tfg_eta is not None:
        run_tfg = run_config.get("tfg_eta")
        if run_tfg is None or abs(float(run_tfg) - float(tfg_eta)) > 1e-6:
            return False
    
    return True


def get_training_curve(run, env_name=None, max_steps=1000000, step_interval=10000):
    """
    Get training curve data from a run's history.
    Returns steps and values arrays, interpolated to regular intervals.
    """
    if env_name is None:
        env_name = run.config.get("env", "env")
    new_key = f"episode_return/{env_name}"
    legacy_key = "sample/episode_return"
    history = run.history(keys=[new_key, legacy_key, "_step"], samples=10000)

    if not history.empty and new_key in history.columns and history[new_key].notna().any():
        metric_key = new_key
    elif not history.empty and legacy_key in history.columns and history[legacy_key].notna().any():
        metric_key = legacy_key
    else:
        return None, None
    
    # Drop NaN values
    history = history.dropna(subset=[metric_key])
    if len(history) == 0:
        return None, None
    
    steps = history["_step"].values
    values = history[metric_key].values
    
    # Interpolate to regular intervals
    target_steps = np.arange(0, max_steps + step_interval, step_interval)
    interp_values = np.interp(target_steps, steps, values, left=np.nan, right=values[-1])
    
    return target_steps, interp_values


def find_runs_for_env(api, env_name, tfg_eta=16.0, seeds=None, most_recent_n=5):
    """
    Find runs matching the model-free target args for a specific environment.
    
    Args:
        api: wandb API object
        env_name: Environment name (e.g., "HalfCheetah-v4")
        tfg_eta: TFG lambda value to filter by
        seeds: List of seeds to include (if None, uses most recent runs)
        most_recent_n: Number of most recent runs to return per seed
    
    Returns:
        List of matching runs
    """
    runs = api.runs("diffusion_online_rl")
    
    matching_runs = []
    runs_by_seed = defaultdict(list)
    
    for run in runs:
        if run.state != "finished":
            continue
        
        config = run.config
        if not run_matches_target(config, MODEL_FREE_TARGET_ARGS, env_name, tfg_eta):
            continue
        
        seed = config.get("seed")
        if seed is None:
            continue
        
        # Filter by seeds if specified
        if seeds is not None and seed not in seeds:
            continue
        
        runs_by_seed[seed].append(run)
    
    # Get most recent runs for each seed
    for seed in sorted(runs_by_seed.keys()):
        seed_runs = runs_by_seed[seed]
        # Sort by created_at (most recent first)
        seed_runs.sort(key=lambda r: r.created_at, reverse=True)
        # Take most recent
        if len(seed_runs) > 0:
            matching_runs.append(seed_runs[0])
            print(f"  Found run for seed {seed}: {seed_runs[0].name} (created: {seed_runs[0].created_at})")
    
    return matching_runs


def plot_training_curves(env_curves_dict, output_path, title=None):
    """
    Plot training curves with mean and 95% CI for multiple methods.
    
    Args:
        env_curves_dict: Dict mapping method names to lists of (steps, values) tuples
        output_path: Path to save the figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx, (method_name, curves) in enumerate(env_curves_dict.items()):
        if not curves:
            continue
        
        # Stack all curves (they should have the same steps)
        all_values = []
        steps = None
        for s, v in curves:
            if s is not None and v is not None:
                if steps is None:
                    steps = s
                all_values.append(v)
        
        if not all_values or steps is None:
            continue
        
        all_values = np.array(all_values)
        n_seeds = all_values.shape[0]
        
        # Calculate mean and 95% CI using t-distribution
        mean = np.nanmean(all_values, axis=0)
        
        if n_seeds > 1:
            sem = stats.sem(all_values, axis=0, nan_policy='omit')
            t_crit = stats.t.ppf(0.975, df=n_seeds - 1)
            ci = t_crit * sem
        else:
            ci = np.zeros_like(mean)
        
        color = colors[idx % len(colors)]
        ax.plot(steps, mean, label=f"{method_name} (n={n_seeds})", color=color, linewidth=2)
        ax.fill_between(steps, mean - ci, mean + ci, alpha=0.2, color=color)
    
    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_ylabel('Episode Return', fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis with K/M suffixes
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")
    plt.close()


def get_final_returns(curves):
    """Get final returns from a list of (steps, values) curves."""
    returns = []
    for steps, values in curves:
        if values is not None and len(values) > 0:
            # Use mean of last 10 values for stability
            last_n = min(10, len(values))
            final_return = np.nanmean(values[-last_n:])
            if not np.isnan(final_return):
                returns.append(final_return)
    return returns


def main():
    api = wandb.Api(timeout=60)
    
    # Configuration
    tfg_eta = 16.0
    target_seeds = [0, 1, 2, 3, 4]
    
    figures_dir = "/n/home09/pnielsen/inference_mirror_descent/figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    results_summary = {}
    
    for env_name in MUJOCO_ENVS:
        print(f"\n{'='*60}")
        print(f"Processing {env_name}...")
        print(f"{'='*60}")
        
        # Find runs for this environment
        runs = find_runs_for_env(
            api, 
            env_name, 
            tfg_eta=tfg_eta, 
            seeds=target_seeds,
            most_recent_n=1
        )
        
        if not runs:
            print(f"  No matching runs found for {env_name}")
            results_summary[env_name] = {"status": "NO_DATA", "n_seeds": 0}
            continue
        
        # Get training curves
        curves = []
        for run in runs:
            steps, values = get_training_curve(run)
            if steps is not None:
                curves.append((steps, values))
                print(f"    Got curve with {len(steps)} points")
        
        if not curves:
            print(f"  No training curve data for {env_name}")
            results_summary[env_name] = {"status": "NO_CURVE_DATA", "n_seeds": 0}
            continue
        
        # Calculate final return statistics
        final_returns = get_final_returns(curves)
        n_seeds = len(final_returns)
        
        if n_seeds > 0:
            mean_return = np.mean(final_returns)
            if n_seeds > 1:
                se = stats.sem(final_returns)
            else:
                se = 0
            
            results_summary[env_name] = {
                "status": "COMPLETE" if n_seeds >= 5 else "IN_PROGRESS",
                "n_seeds": n_seeds,
                "mean": mean_return,
                "se": se,
                "final_returns": final_returns,
            }
            print(f"  {env_name}: n={n_seeds}, mean={mean_return:.1f} ± {se:.1f}")
        
        # Plot training curve for this environment
        env_curves_dict = {f"MALA-Guided DPMD (λ={tfg_eta})": curves}
        output_path = os.path.join(figures_dir, f"training_curve_{env_name.replace('-', '_')}.png")
        plot_training_curves(
            env_curves_dict, 
            output_path, 
            title=f"{env_name}: MALA-Guided DPMD Training Curve"
        )
    
    # Print summary table
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Environment':<20} {'Status':<15} {'n':<5} {'Return':<20}")
    print("-"*60)
    for env_name in MUJOCO_ENVS:
        result = results_summary.get(env_name, {"status": "NOT_RUN", "n_seeds": 0})
        status = result["status"]
        n = result["n_seeds"]
        if status in ["COMPLETE", "IN_PROGRESS"] and "mean" in result:
            return_str = f"{result['mean']:.1f} ± {result['se']:.1f}"
        else:
            return_str = "---"
        print(f"{env_name:<20} {status:<15} {n:<5} {return_str:<20}")
    
    return results_summary


if __name__ == "__main__":
    main()
