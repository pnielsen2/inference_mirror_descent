#!/usr/bin/env python
"""
Generate paper figures and table data from wandb API and LSAC baseline data.
Fetches MGMD training curves from wandb on the fly.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from scipy import stats
from collections import defaultdict
from pathlib import Path
import wandb

# Configuration
FIGURES_DIR = "/n/home09/pnielsen/inference_mirror_descent/figures"
LSAC_DATA_DIR = "/n/home09/pnielsen/LSAC/data"

# MGMD config filters for wandb queries
MGMD_CONFIG = {
    "alg": "dpmd",
    "dpmd_constant_weight": True,
    "dpmd_no_entropy_tuning": True,
    "num_particles": 1,
    "mala_steps": 2,
    "q_critic_agg": "mean",
    "mala_guided_predictor": True,
    "x0_hat_clip_radius": 3.0,
    "mala_adapt_rate": 0.2,
    "mala_per_level_eta": True,
    "q_td_huber_width": 30.0,
    "lr_q": 0.00015,
}

# LSAC baseline methods mapping
LSAC_ALGO_MAP = {
    'OURS': 'LSAC', 'DSAC': 'DSAC', 'SAC': 'SAC', 'TD3': 'TD3',
    'DIPO': 'DIPO', 'PPO': 'PPO', 'TRPO': 'TRPO'
}

# Environments
ENVS = {
    'halfcheetah': 'HalfCheetah',
    'ant': 'Ant',
    'swimmer': 'Swimmer',
    'walker2d': 'Walker2d',
    'hopper': 'Hopper',
    'humanoid': 'Humanoid'
}

# Fixed color mapping for consistent colors across all plots
METHOD_COLORS = {
    'MGMD (Ours)': 'C0',  # Blue
    'LSAC': 'C1',         # Orange
    'DSAC': 'C2',         # Green
    'SAC': 'C3',          # Red
    'TD3': 'C4',          # Purple
    'DIPO': 'C5',         # Brown
    'PPO': 'C6',          # Pink
    'TRPO': 'C7',         # Gray
}


def fetch_mgmd_curves(api, env_name, seeds=[0, 1, 2, 3, 4], max_steps=1000000, step_interval=10000):
    """Fetch MGMD training curves from wandb API."""
    filters = {
        "config.env": env_name,
        "config.alg": "dpmd",
        "config.dpmd_constant_weight": True,
        "config.mala_steps": 2,
        "config.num_particles": 1,
        "config.q_critic_agg": "mean",
        "config.mala_guided_predictor": True,
        "config.dpmd_no_entropy_tuning": True,
        "state": "finished",
    }
    
    print(f"  Fetching MGMD runs for {env_name} from wandb...")
    runs = api.runs("diffusion_online_rl", filters=filters)
    
    # Group by seed, keep most recent
    runs_by_seed = {}
    for run in runs:
        config = run.config
        seed = config.get("seed")
        if seed is None or seed not in seeds:
            continue
        # Additional config checks (handle None values from wandb config)
        x0_clip = config.get("x0_hat_clip_radius")
        if x0_clip is None or abs(float(x0_clip) - 3.0) > 0.01:
            continue
        adapt_rate = config.get("mala_adapt_rate")
        if adapt_rate is None or abs(float(adapt_rate) - 0.2) > 0.01:
            continue
        if not config.get("mala_per_level_eta"):
            continue
        huber = config.get("q_td_huber_width")
        if huber is None or abs(float(huber) - 30.0) > 0.1:
            continue
        lr_q = config.get("lr_q")
        if lr_q is None or abs(float(lr_q) - 0.00015) > 1e-6:
            continue
        
        if seed not in runs_by_seed or run.created_at > runs_by_seed[seed].created_at:
            runs_by_seed[seed] = run
    
    if not runs_by_seed:
        return None, None
    
    target_steps = np.arange(0, max_steps + step_interval, step_interval)
    all_curves = []
    
    for seed in sorted(runs_by_seed.keys()):
        run = runs_by_seed[seed]
        env_name = run.config.get("env", "env")
        new_key = f"episode_return/{env_name}"
        legacy_key = "sample/episode_return"
        history = run.history(keys=[new_key, legacy_key, "_step"], samples=5000)
        if not history.empty and new_key in history.columns and history[new_key].notna().any():
            metric_key = new_key
        elif not history.empty and legacy_key in history.columns and history[legacy_key].notna().any():
            metric_key = legacy_key
        else:
            continue
        history = history.dropna(subset=[metric_key])
        if len(history) == 0:
            continue
        
        steps = history["_step"].values
        values = history[metric_key].values
        interp_values = np.interp(target_steps, steps, values, left=np.nan, right=values[-1])
        all_curves.append(interp_values)
        print(f"    Loaded seed {seed}: {run.name} ({len(steps)} points)")
    
    if not all_curves:
        return None, None
    
    return target_steps, np.array(all_curves)


def load_lsac_baseline(env_key):
    """Load LSAC baseline data from pickle files."""
    pkl_path = os.path.join(LSAC_DATA_DIR, f"all_data_{env_key}.pkl")
    if not os.path.exists(pkl_path):
        return None
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return pd.read_csv(StringIO(data))


def compute_90_ci(values, axis=0):
    """Compute 90% CI using t-distribution."""
    n = values.shape[axis]
    if n <= 1:
        return np.zeros(values.shape[1] if axis == 0 else values.shape[0])
    
    mean = np.nanmean(values, axis=axis)
    std = np.nanstd(values, axis=axis, ddof=1)
    t_crit = stats.t.ppf(0.95, df=n-1)
    ci = t_crit * std / np.sqrt(n)
    return ci


def generate_training_curves_figure(api, output_path):
    """Generate 6-panel training curves figure."""
    plt.style.use('default')
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    
    # t-critical values for 90% CI
    t_crit_10 = stats.t.ppf(0.95, df=9)  # LSAC: 10 seeds
    t_crit_5 = stats.t.ppf(0.95, df=4)   # MGMD: 5 seeds
    
    table_data = {}
    
    for idx, (env_key, env_name) in enumerate(ENVS.items()):
        ax = axes[idx]
        env_v4 = f"{env_name}-v4"
        table_data[env_name] = {}
        
        print(f"\nProcessing {env_name}...")
        
        # Fetch MGMD data from wandb
        steps, curves = fetch_mgmd_curves(api, env_v4)
        if curves is not None and len(curves) > 0:
            mgmd_mean = np.nanmean(curves, axis=0)
            mgmd_std = np.nanstd(curves, axis=0, ddof=1)
            n_seeds = curves.shape[0]
            t_crit = stats.t.ppf(0.95, df=max(1, n_seeds-1))
            mgmd_ci = t_crit * mgmd_std / np.sqrt(n_seeds)
            
            color = METHOD_COLORS['MGMD (Ours)']
            ax.plot(steps/1e6, mgmd_mean, label='MGMD (Ours)', linewidth=1.5, color=color)
            ax.fill_between(steps/1e6, mgmd_mean - mgmd_ci, mgmd_mean + mgmd_ci, alpha=0.2, color=color)
            
            # Final value for table
            final_mean = np.nanmean(curves[:, -1])
            final_std = np.nanstd(curves[:, -1], ddof=1)
            final_ci = t_crit * final_std / np.sqrt(n_seeds)
            table_data[env_name]['MGMD'] = (final_mean, final_ci, n_seeds)
        
        # Load LSAC baselines
        df = load_lsac_baseline(env_key)
        if df is not None:
            for lsac_name, display_name in LSAC_ALGO_MAP.items():
                if lsac_name not in df['algo'].values:
                    continue
                algo_df = df[df['algo'] == lsac_name].sort_values('steps')
                steps = algo_df['steps'].values / 1e6
                means = algo_df['rew_mean'].values
                stds = algo_df['rew_std'].values
                # 90% CI for LSAC baselines (10 seeds)
                cis = t_crit_10 * stds / np.sqrt(10)
                
                color = METHOD_COLORS[display_name]
                ax.plot(steps, means, label=display_name, linewidth=1.2, color=color)
                ax.fill_between(steps, means - cis, means + cis, alpha=0.15, color=color)
                
                # Final value for table
                final_row = algo_df[algo_df['steps'] == algo_df['steps'].max()]
                final_mean = final_row['rew_mean'].values[0]
                final_std = final_row['rew_std'].values[0]
                final_ci = t_crit_10 * final_std / np.sqrt(10)
                table_data[env_name][display_name] = (final_mean, final_ci, 10)
        
        ax.set_title(env_name, fontsize=11)
        ax.set_xlabel('Steps (M)', fontsize=9)
        ax.set_ylabel('Episode Return', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=8)
    
    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=8, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure to: {output_path}")
    plt.close()
    
    return table_data


def print_latex_table(table_data):
    """Print LaTeX table with results."""
    methods = ['MGMD', 'LSAC', 'DSAC', 'DIPO', 'SAC', 'TD3', 'PPO', 'TRPO']
    
    print("\n" + "="*80)
    print("LATEX TABLE DATA (mean ± 90% CI)")
    print("="*80)
    
    for env_name in ENVS.values():
        row = [env_name]
        for method in methods:
            if method in table_data.get(env_name, {}):
                mean, ci, n = table_data[env_name][method]
                row.append(f"${mean:.0f}{{\\scriptstyle\\pm{ci:.0f}}}$")
            else:
                row.append("$^\\dagger$")
        print(" & ".join(row) + " \\\\")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    api = wandb.Api(timeout=120)
    
    # Generate training curves figure
    output_path = os.path.join(FIGURES_DIR, "model_free_training_curves_all_envs.png")
    table_data = generate_training_curves_figure(api, output_path)
    
    # Print table data
    print_latex_table(table_data)
    
    # Save table data for later use
    table_path = os.path.join(FIGURES_DIR, "table_data.pkl")
    with open(table_path, 'wb') as f:
        pickle.dump(table_data, f)
    print(f"\nSaved table data to: {table_path}")


if __name__ == "__main__":
    main()
