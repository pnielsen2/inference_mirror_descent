#!/usr/bin/env python
"""
Efficiently fetch training curves from wandb using server-side filtering.
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os
import json

# Use longer timeout
api = wandb.Api(timeout=120)

def fetch_model_free_runs(env_name="HalfCheetah-v4", tfg_eta=16.0, seeds=[0,1,2,3,4]):
    """
    Fetch runs using wandb filters for efficiency.
    """
    # Use wandb filters to query on server side
    filters = {
        "config.alg": "dpmd",
        "config.env": env_name,
        "config.dpmd_constant_weight": True,
        "config.tfg_eta": tfg_eta,
        "config.num_particles": 1,
        "config.mala_steps": 2,
        "config.q_critic_agg": "mean",
        "config.mala_guided_predictor": True,
        "config.dpmd_no_entropy_tuning": True,
        "state": "finished",
    }
    
    print(f"Querying wandb with filters for {env_name}, tfg_eta={tfg_eta}...")
    runs = api.runs("diffusion_online_rl", filters=filters)
    
    # Group by seed and get most recent
    runs_by_seed = {}
    for run in runs:
        seed = run.config.get("seed")
        if seed is None or seed not in seeds:
            continue
        
        # Check additional config requirements
        config = run.config
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
    
    print(f"Found {len(runs_by_seed)} matching runs")
    for seed, run in sorted(runs_by_seed.items()):
        print(f"  Seed {seed}: {run.name} (created: {run.created_at})")
    
    return runs_by_seed


def get_training_curve(run, env_name=None, max_steps=1000000, step_interval=10000):
    """Get training curve from run history."""
    print(f"  Fetching history for {run.name}...")
    # Determine metric key: new format uses episode_return/{env},
    # fall back to legacy sample/episode_return for old runs.
    if env_name is None:
        env_name = run.config.get("env", "env")
    new_key = f"episode_return/{env_name}"
    legacy_key = "sample/episode_return"
    history = run.history(keys=[new_key, legacy_key, "_step"], samples=5000)
    
    # Prefer new key, fall back to legacy
    if not history.empty and new_key in history.columns and history[new_key].notna().any():
        metric_key = new_key
    elif not history.empty and legacy_key in history.columns and history[legacy_key].notna().any():
        metric_key = legacy_key
    else:
        return None, None
    
    history = history.dropna(subset=[metric_key])
    if len(history) == 0:
        return None, None
    
    steps = history["_step"].values
    values = history[metric_key].values
    
    # Interpolate to regular intervals
    target_steps = np.arange(0, max_steps + step_interval, step_interval)
    interp_values = np.interp(target_steps, steps, values, left=np.nan, right=values[-1])
    
    return target_steps, interp_values


def plot_training_curve(curves_dict, env_name, output_path):
    """Plot training curves with 95% CI."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, (method_name, curves) in enumerate(curves_dict.items()):
        if not curves:
            continue
        
        all_values = np.array([v for s, v in curves if v is not None])
        steps = curves[0][0] if curves else None
        
        if len(all_values) == 0 or steps is None:
            continue
        
        n = all_values.shape[0]
        mean = np.nanmean(all_values, axis=0)
        
        if n > 1:
            sem = stats.sem(all_values, axis=0, nan_policy='omit')
            t_crit = stats.t.ppf(0.975, df=n-1)
            ci = t_crit * sem
        else:
            ci = np.zeros_like(mean)
        
        color = colors[idx % len(colors)]
        ax.plot(steps, mean, label=f"{method_name} (n={n})", color=color, linewidth=2)
        ax.fill_between(steps, mean - ci, mean + ci, alpha=0.2, color=color)
    
    ax.set_xlabel('Environment Steps', fontsize=12)
    ax.set_ylabel('Episode Return', fontsize=12)
    ax.set_title(f'{env_name}: Training Curves', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M' if x >= 1e6 else f'{x/1e3:.0f}K'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    figures_dir = "/n/home09/pnielsen/inference_mirror_descent/figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    env_name = "HalfCheetah-v4"
    tfg_eta = 16.0
    seeds = [0, 1, 2, 3, 4]
    
    # Fetch runs
    runs_by_seed = fetch_model_free_runs(env_name, tfg_eta, seeds)
    
    if not runs_by_seed:
        print("No matching runs found!")
        return
    
    # Get training curves
    curves = []
    for seed in sorted(runs_by_seed.keys()):
        run = runs_by_seed[seed]
        steps, values = get_training_curve(run)
        if steps is not None:
            curves.append((steps, values))
    
    # Calculate final statistics
    final_returns = []
    for steps, values in curves:
        if values is not None:
            final_returns.append(np.nanmean(values[-10:]))
    
    if final_returns:
        mean_return = np.mean(final_returns)
        se = stats.sem(final_returns) if len(final_returns) > 1 else 0
        print(f"\nFinal Return: {mean_return:.1f} ± {se:.1f} (n={len(final_returns)})")
    
    # Plot
    curves_dict = {f"MALA-Guided DPMD (λ={tfg_eta})": curves}
    output_path = os.path.join(figures_dir, f"model_free_training_curve_{env_name.replace('-','_')}.png")
    plot_training_curve(curves_dict, env_name, output_path)
    
    # Save data to JSON for later use
    data = {
        "env": env_name,
        "tfg_eta": tfg_eta,
        "seeds": list(runs_by_seed.keys()),
        "final_returns": final_returns,
        "mean_return": mean_return if final_returns else None,
        "se": se if final_returns else None,
    }
    json_path = os.path.join(figures_dir, f"model_free_results_{env_name.replace('-','_')}.json")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved data: {json_path}")


if __name__ == "__main__":
    main()
