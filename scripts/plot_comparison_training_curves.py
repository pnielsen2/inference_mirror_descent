#!/usr/bin/env python
"""
Plot training curves comparing MALA-guided DPMD with baseline methods (SAC, DPMD, LSAC).
Supports all MuJoCo environments and can incorporate external baseline data.

Usage:
    python scripts/plot_comparison_training_curves.py --env HalfCheetah-v4
    python scripts/plot_comparison_training_curves.py --env all
"""

import argparse
import wandb
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os
import json

# MuJoCo environments
MUJOCO_ENVS = [
    "HalfCheetah-v4",
    "Hopper-v4", 
    "Walker2d-v4",
    "Ant-v4",
    "Humanoid-v4",
]

# Target hyperparameters for MALA-guided DPMD
MALA_GUIDED_CONFIG = {
    "alg": "dpmd",
    "dpmd_constant_weight": True,
    "tfg_lambda": 16.0,
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

# Placeholder baseline data (to be replaced with actual data from LSAC repo or other sources)
# Format: {env: {method: {"mean": [...], "std": [...], "steps": [...]}}}
BASELINE_DATA = {
    "HalfCheetah-v4": {
        # Placeholder - replace with actual LSAC/SAC data
        # Data should be: mean episode return every 10K steps for 1M steps (101 points)
        "SAC": None,  # TBD
        "DPMD": None,  # TBD
        "LSAC": None,  # TBD
    },
    "Hopper-v4": {"SAC": None, "DPMD": None, "LSAC": None},
    "Walker2d-v4": {"SAC": None, "DPMD": None, "LSAC": None},
    "Ant-v4": {"SAC": None, "DPMD": None, "LSAC": None},
    "Humanoid-v4": {"SAC": None, "DPMD": None, "LSAC": None},
}


def fetch_mala_guided_runs(api, env_name, seeds=[0, 1, 2, 3, 4]):
    """Fetch MALA-guided DPMD runs from wandb."""
    filters = {
        "config.alg": "dpmd",
        "config.env": env_name,
        "config.dpmd_constant_weight": True,
        "config.tfg_lambda": MALA_GUIDED_CONFIG["tfg_lambda"],
        "config.num_particles": 1,
        "config.mala_steps": 2,
        "config.q_critic_agg": "mean",
        "config.mala_guided_predictor": True,
        "config.dpmd_no_entropy_tuning": True,
        "state": "finished",
    }
    
    print(f"Fetching MALA-guided runs for {env_name}...")
    runs = api.runs("diffusion_online_rl", filters=filters)
    
    runs_by_seed = {}
    for run in runs:
        config = run.config
        seed = config.get("seed")
        if seed is None or seed not in seeds:
            continue
        
        # Additional config checks
        if config.get("buffer_size") != 200000:
            continue
        if abs(config.get("x0_hat_clip_radius", 0) - 3.0) > 0.01:
            continue
        if abs(config.get("mala_adapt_rate", 0) - 0.2) > 0.01:
            continue
        if not config.get("mala_per_level_eta"):
            continue
        if abs(config.get("q_td_huber_width", 0) - 30.0) > 0.1:
            continue
        if config.get("update_per_iteration") != 4:
            continue
        if abs(config.get("lr_q", 0) - 0.00015) > 1e-6:
            continue
        
        if seed not in runs_by_seed or run.created_at > runs_by_seed[seed].created_at:
            runs_by_seed[seed] = run
    
    print(f"  Found {len(runs_by_seed)} runs: seeds {sorted(runs_by_seed.keys())}")
    return runs_by_seed


def get_training_curve(run, max_steps=1000000, step_interval=10000):
    """Get interpolated training curve from run history."""
    history = run.history(keys=["sample/episode_return", "_step"], samples=5000)
    
    if history.empty or "sample/episode_return" not in history.columns:
        return None, None
    
    history = history.dropna(subset=["sample/episode_return"])
    if len(history) == 0:
        return None, None
    
    steps = history["_step"].values
    values = history["sample/episode_return"].values
    
    target_steps = np.arange(0, max_steps + step_interval, step_interval)
    interp_values = np.interp(target_steps, steps, values, left=np.nan, right=values[-1])
    
    return target_steps, interp_values


def load_baseline_data(env_name, method):
    """
    Load baseline data from external source.
    
    For LSAC: Clone https://github.com/hmishfaq/LSAC and look in logs/ directory
    Format expected: CSV with columns for step and return, or JSON with mean/std arrays
    """
    baseline_dir = "/n/home09/pnielsen/inference_mirror_descent/baselines"
    
    # Try to load from local baseline cache
    json_path = os.path.join(baseline_dir, f"{method}_{env_name.replace('-', '_')}.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data.get("steps"), data.get("mean"), data.get("std")
    
    # Placeholder return
    return None, None, None


def plot_comparison(env_name, mala_curves, baseline_data, output_path):
    """Plot comparison training curves with 95% CI."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        "MALA-Guided (Ours)": "#1f77b4",
        "SAC": "#ff7f0e",
        "DPMD": "#2ca02c", 
        "LSAC": "#d62728",
    }
    
    # Plot MALA-guided curves
    if mala_curves:
        all_values = np.array([v for s, v in mala_curves if v is not None])
        steps = mala_curves[0][0] if mala_curves else None
        
        if len(all_values) > 0 and steps is not None:
            n = all_values.shape[0]
            mean = np.nanmean(all_values, axis=0)
            
            if n > 1:
                sem = stats.sem(all_values, axis=0, nan_policy='omit')
                t_crit = stats.t.ppf(0.975, df=n - 1)
                ci = t_crit * sem
            else:
                ci = np.zeros_like(mean)
            
            color = colors["MALA-Guided (Ours)"]
            ax.plot(steps, mean, label=f"MALA-Guided (Ours, n={n})", color=color, linewidth=2)
            ax.fill_between(steps, mean - ci, mean + ci, alpha=0.2, color=color)
    
    # Plot baseline methods
    for method in ["SAC", "DPMD", "LSAC"]:
        b_steps, b_mean, b_std = load_baseline_data(env_name, method)
        
        if b_mean is not None and b_steps is not None:
            color = colors.get(method, "#888888")
            ax.plot(b_steps, b_mean, label=method, color=color, linewidth=2, linestyle='--')
            if b_std is not None:
                ax.fill_between(b_steps, 
                               np.array(b_mean) - np.array(b_std),
                               np.array(b_mean) + np.array(b_std),
                               alpha=0.15, color=color)
    
    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_ylabel('Episode Return', fontsize=12)
    ax.set_title(f'{env_name}: Training Curves Comparison', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    return fig


def compute_final_stats(curves):
    """Compute final return statistics from curves."""
    final_returns = []
    for steps, values in curves:
        if values is not None and len(values) > 0:
            final_returns.append(np.nanmean(values[-10:]))
    
    if not final_returns:
        return None, None, 0
    
    mean = np.mean(final_returns)
    se = stats.sem(final_returns) if len(final_returns) > 1 else 0
    return mean, se, len(final_returns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="HalfCheetah-v4",
                       help="Environment name or 'all' for all MuJoCo envs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    
    api = wandb.Api(timeout=120)
    
    figures_dir = "/n/home09/pnielsen/inference_mirror_descent/figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    # Create baselines directory for caching baseline data
    baselines_dir = "/n/home09/pnielsen/inference_mirror_descent/baselines"
    os.makedirs(baselines_dir, exist_ok=True)
    
    envs = MUJOCO_ENVS if args.env == "all" else [args.env]
    
    results = {}
    
    for env_name in envs:
        print(f"\n{'='*60}")
        print(f"Processing {env_name}")
        print(f"{'='*60}")
        
        # Fetch MALA-guided runs
        runs_by_seed = fetch_mala_guided_runs(api, env_name, args.seeds)
        
        # Get training curves
        curves = []
        for seed in sorted(runs_by_seed.keys()):
            run = runs_by_seed[seed]
            print(f"  Fetching curve for seed {seed}...")
            steps, values = get_training_curve(run)
            if steps is not None:
                curves.append((steps, values))
        
        # Compute final stats
        mean, se, n = compute_final_stats(curves)
        results[env_name] = {"mean": mean, "se": se, "n": n}
        
        if mean is not None:
            print(f"  Final: {mean:.1f} ± {se:.1f} (n={n})")
        else:
            print(f"  No data available")
        
        # Plot comparison
        if curves:
            output_path = os.path.join(figures_dir, 
                f"comparison_{env_name.replace('-', '_')}.png")
            plot_comparison(env_name, curves, BASELINE_DATA.get(env_name, {}), output_path)
    
    # Print summary table
    print("\n" + "="*60)
    print("RESULTS SUMMARY (MALA-Guided DPMD)")
    print("="*60)
    print(f"{'Environment':<20} {'Return':<25} {'n':<5}")
    print("-"*50)
    for env_name in envs:
        r = results.get(env_name, {})
        if r.get("mean") is not None:
            return_str = f"{r['mean']:.1f} ± {r['se']:.1f}"
        else:
            return_str = "in progress"
        print(f"{env_name:<20} {return_str:<25} {r.get('n', 0):<5}")
    
    # Save results to JSON
    results_path = os.path.join(figures_dir, "mala_guided_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
