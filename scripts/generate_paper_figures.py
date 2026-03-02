#!/usr/bin/env python
"""
Generate paper figures and table data from local tensorboard logs and LSAC baseline data.
No wandb API required - reads directly from local event files.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from scipy import stats
from tensorboard.backend.event_processing import event_accumulator
from collections import defaultdict
from pathlib import Path

# Configuration
FIGURES_DIR = "/n/home09/pnielsen/inference_mirror_descent/figures"
LOGS_DIR = "/n/home09/pnielsen/inference_mirror_descent/logs"
LSAC_DATA_DIR = "/n/home09/pnielsen/LSAC/data"

# Target MGMD runs (known good runs for HalfCheetah)
MGMD_TARGET_RUNS = {
    "HalfCheetah-v4": {
        0: "dpmd_2026-01-15_08-02-43_s0_",
        1: "dpmd_2026-01-15_11-05-52_s1_",
        2: "dpmd_2026-01-15_11-05-52_s2_",
        3: "dpmd_2026-01-15_11-06-05_s3_",
        4: "dpmd_2026-01-15_11-07-06_s4_",
    }
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


def load_tensorboard_scalar(log_dir, tag="sample/episode_return"):
    """Load scalar data from tensorboard event files."""
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={event_accumulator.SCALARS: 0}  # Load all
    )
    ea.Reload()
    
    if tag not in ea.Tags()['scalars']:
        return None, None
    
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def load_mgmd_curves(env_name, target_runs, max_steps=1000000, step_interval=10000):
    """Load MGMD training curves from local tensorboard logs."""
    env_log_dir = os.path.join(LOGS_DIR, env_name)
    if not os.path.exists(env_log_dir):
        return None, None
    
    target_steps = np.arange(0, max_steps + step_interval, step_interval)
    all_curves = []
    
    for seed, run_name in target_runs.items():
        run_dir = os.path.join(env_log_dir, run_name)
        if not os.path.exists(run_dir):
            print(f"  Warning: Run dir not found: {run_dir}")
            continue
        
        steps, values = load_tensorboard_scalar(run_dir)
        if steps is None:
            print(f"  Warning: No data for {run_name}")
            continue
        
        # Interpolate to regular intervals
        interp_values = np.interp(target_steps, steps, values, left=np.nan, right=values[-1])
        all_curves.append(interp_values)
        print(f"  Loaded seed {seed}: {run_name} ({len(steps)} points)")
    
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


def generate_training_curves_figure(output_path):
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
        
        # Load MGMD data if available
        if env_v4 in MGMD_TARGET_RUNS:
            steps, curves = load_mgmd_curves(env_v4, MGMD_TARGET_RUNS[env_v4])
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
    
    # Generate training curves figure
    output_path = os.path.join(FIGURES_DIR, "model_free_training_curves_all_envs.png")
    table_data = generate_training_curves_figure(output_path)
    
    # Print table data
    print_latex_table(table_data)
    
    # Save table data for later use
    table_path = os.path.join(FIGURES_DIR, "table_data.pkl")
    with open(table_path, 'wb') as f:
        pickle.dump(table_data, f)
    print(f"\nSaved table data to: {table_path}")


if __name__ == "__main__":
    main()
