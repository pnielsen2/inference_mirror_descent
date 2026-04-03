#!/usr/bin/env python
"""
Query wandb for training runs matching a specific command pattern and plot
average final episode return as a function of tfg_eta with 95% CI.
"""

import wandb
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from collections import defaultdict

# Target command pattern (without seed and tfg_eta which vary)
TARGET_ARGS = {
    "alg": "dpmd",
    "env": "HalfCheetah-v4",
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


def run_matches_target(run_config):
    """Check if a run's config matches the target args."""
    for key, target_val in TARGET_ARGS.items():
        run_val = run_config.get(key)
        # Handle type mismatches (wandb may store as different types)
        if isinstance(target_val, bool):
            if run_val != target_val:
                return False
        elif isinstance(target_val, (int, float)):
            if run_val is None:
                return False
            # Use approximate comparison for floats
            if abs(float(run_val) - float(target_val)) > 1e-6:
                return False
        elif run_val != target_val:
            return False
    return True


def get_final_episode_return(run):
    """Get the final episode return from a run's history."""
    env_name = run.config.get("env", "env")
    new_key = f"episode_return/{env_name}"
    legacy_key = "sample/episode_return"
    history = run.history(keys=[new_key, legacy_key], samples=5000)

    if not history.empty and new_key in history.columns and history[new_key].notna().any():
        metric_key = new_key
    elif not history.empty and legacy_key in history.columns and history[legacy_key].notna().any():
        metric_key = legacy_key
    else:
        return None
    
    # Get the last non-NaN value
    returns = history[metric_key].dropna()
    if len(returns) == 0:
        return None
    
    # Use the mean of the last few values for stability
    last_n = min(10, len(returns))
    return returns.iloc[-last_n:].mean()


def main():
    # Initialize wandb API
    api = wandb.Api()
    
    # Get all runs from the project
    print("Fetching runs from wandb project 'diffusion_online_rl'...")
    runs = api.runs("diffusion_online_rl")
    
    # Collect data: tfg_eta -> list of final returns
    tfg_eta_returns = defaultdict(list)
    matched_runs = 0
    
    for run in runs:
        if run.state != "finished":
            continue
            
        config = run.config
        
        if not run_matches_target(config):
            continue
        
        tfg_eta = config.get("tfg_eta")
        if tfg_eta is None:
            continue
            
        final_return = get_final_episode_return(run)
        if final_return is None:
            continue
        
        tfg_eta_returns[tfg_eta].append(final_return)
        matched_runs += 1
        print(f"  Matched run: tfg_eta={tfg_eta}, seed={config.get('seed')}, return={final_return:.1f}")
    
    print(f"\nTotal matched runs: {matched_runs}")
    
    if not tfg_eta_returns:
        print("No matching runs found!")
        return
    
    # Sort by tfg_eta
    tfg_etas = sorted(tfg_eta_returns.keys())
    
    # Calculate statistics
    means = []
    ci_lower = []
    ci_upper = []
    
    for lam in tfg_etas:
        returns = np.array(tfg_eta_returns[lam])
        n = len(returns)
        mean = np.mean(returns)
        means.append(mean)
        
        if n > 1:
            # 95% CI using t-distribution
            sem = stats.sem(returns)
            t_crit = stats.t.ppf(0.975, df=n-1)
            ci = t_crit * sem
            ci_lower.append(mean - ci)
            ci_upper.append(mean + ci)
        else:
            # Single sample, no CI
            ci_lower.append(mean)
            ci_upper.append(mean)
        
        print(f"tfg_eta={lam}: n={n}, mean={mean:.1f}, 95% CI=[{ci_lower[-1]:.1f}, {ci_upper[-1]:.1f}]")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    tfg_etas = np.array(tfg_etas)
    means = np.array(means)
    ci_lower = np.array(ci_lower)
    ci_upper = np.array(ci_upper)
    
    # Plot with error bars
    ax.errorbar(
        tfg_etas, means,
        yerr=[means - ci_lower, ci_upper - means],
        fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8,
        color='#1f77b4', ecolor='#1f77b4', alpha=0.8
    )
    
    # Fill between for CI visualization
    ax.fill_between(tfg_etas, ci_lower, ci_upper, alpha=0.2, color='#1f77b4')
    
    ax.set_xscale('log')
    ax.set_xlabel('tfg_eta', fontsize=12)
    ax.set_ylabel('Average Final Episode Return', fontsize=12)
    ax.set_title('HalfCheetah-v4: DPMD with Constant Weight\nFinal Return vs. TFG Lambda', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add sample size annotations
    for i, (lam, n) in enumerate(zip(tfg_etas, [len(tfg_eta_returns[l]) for l in tfg_etas])):
        ax.annotate(f'n={n}', (lam, means[i]), textcoords="offset points",
                   xytext=(0, 10), ha='center', fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    
    # Save figure
    output_path = "/n/home09/pnielsen/inference_mirror_descent/figures/tfg_eta_sweep.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    plt.show()


if __name__ == "__main__":
    main()
