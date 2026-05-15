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
WANDB_PROJECT = "pnielsen2-harvard/diffusion_online_rl"

# ── MGMD run IDs ────────────────────────────────────────────────────────────
# HalfCheetah: 10 KL-Budget seeds (March 25, seeds 5-14)
MGMD_RUN_IDS = {
    "HalfCheetah-v4": {
        5: "bscdb6k6",
        6: "s77ywsoz",
        7: "hurp23vk",
        8: "8xcmblzz",
        9: "qlcbvy0i",
        10: "8dw3rm98",
        11: "j74xbjv4",
        12: "5bu21100",
        13: "jb8f66ci",
        14: "6ljcurqt",
    },
    # Other envs: single dist_shift_eta runs (March 30)
    "Hopper-v4": {1: "bk8pkd0j"},
    "Walker2d-v4": {1: "xe358pb3"},
    "Ant-v4": {1: "6mdw2hrs"},
    "Swimmer-v4": {1: "mx6j3c41"},
    "Humanoid-v4": {1: "xbxyxj3p"},
}

# LSAC baseline methods mapping
LSAC_ALGO_MAP = {
    'SAC': 'SAC', 'TD3': 'TD3',
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
    'MGMD': 'C0',  # Blue
    'SAC': 'C1',          # Orange
    'TD3': 'C2',          # Green
    'DIPO': 'C3',         # Red
    'PPO': 'C4',          # Purple
    'TRPO': 'C5',         # Brown
}


MAX_STEPS = 1_000_000
STEP_INTERVAL = 10_000
TARGET_STEPS = np.arange(0, MAX_STEPS + STEP_INTERVAL, STEP_INTERVAL)


def fetch_curve(run):
    """Fetch and interpolate a single training curve from a wandb run."""
    env_name = run.config.get("env", "")
    new_key = f"episode_return/{env_name}"
    legacy_key = "sample/episode_return"

    # Try new key first, fall back to legacy
    for metric_key in (new_key, legacy_key):
        history = run.history(keys=[metric_key, "_step"], samples=10000)
        if not history.empty and metric_key in history.columns:
            history = history.dropna(subset=[metric_key])
            if len(history) > 0:
                steps = history["_step"].values
                values = history[metric_key].values
                return np.interp(TARGET_STEPS, steps, values, left=np.nan, right=values[-1])
    return None


def fetch_mgmd_curves_by_id(api, env_name):
    """Fetch MGMD training curves by explicit run IDs."""
    seed_ids = MGMD_RUN_IDS.get(env_name, {})
    if not seed_ids:
        return None, None

    print(f"  Fetching MGMD runs for {env_name} ({len(seed_ids)} seeds)...")
    all_curves = []
    for seed, run_id in sorted(seed_ids.items()):
        run = api.run(f"{WANDB_PROJECT}/{run_id}")
        curve = fetch_curve(run)
        if curve is not None:
            all_curves.append(curve)
            print(f"    seed {seed}: OK ({run.name})")
        else:
            print(f"    seed {seed}: FAILED ({run_id})")

    if not all_curves:
        return None, None
    return TARGET_STEPS, np.array(all_curves)


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
        
        # Fetch MGMD data from wandb (by run ID)
        steps, curves = fetch_mgmd_curves_by_id(api, env_v4)
        if curves is not None and len(curves) > 0:
            mgmd_mean = np.nanmean(curves, axis=0)
            mgmd_std = np.nanstd(curves, axis=0, ddof=1)
            n_seeds = curves.shape[0]
            t_crit = stats.t.ppf(0.95, df=max(1, n_seeds-1))
            mgmd_ci = t_crit * mgmd_std / np.sqrt(n_seeds)
            
            color = METHOD_COLORS['MGMD']
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
    methods = ['MGMD', 'DIPO', 'SAC', 'TD3', 'PPO', 'TRPO']
    
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
