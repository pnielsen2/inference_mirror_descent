from typing import Dict

import re
import numpy as np
import pandas as pd
import wandb

from matplotlib import pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
sns.set_context(font_scale=1.2)


def _fetch_runs_by_name_pattern(api, env_name, pattern):
    """Fetch runs from wandb whose name matches a regex pattern."""
    filters = {
        "config.env": env_name,
        "state": "finished",
    }
    runs = api.runs("diffusion_online_rl", filters=filters)
    
    matched = []
    for run in runs:
        if re.match(pattern, run.name):
            matched.append(run)
    return matched


def _run_history_to_df(run):
    """Download a single run's history and return a DataFrame with step + avg_ret."""
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
    df = pd.DataFrame({
        "step": history["_step"].values,
        "avg_ret": history[metric_key].values,
    })
    return df


def plot_mean(patterns_dict: Dict, env_name, fig_name=None,
              max_steps=None):
    """Plot mean training curves for runs matching name patterns, fetched from wandb."""
    api = wandb.Api(timeout=120)
    plt.figure(figsize=(4, 3))
    dfs = []
    for alg, pattern in patterns_dict.items():
        print(f"Fetching runs for '{alg}' matching pattern in {env_name}...")
        matched_runs = _fetch_runs_by_name_pattern(api, env_name, pattern)
        for run in matched_runs:
            df = _run_history_to_df(run)
            if df is None:
                continue
            seed = run.config.get("seed", "?")
            df["seed"] = str(seed)
            df["alg"] = alg
            dfs.append(df)
            print(f"  Loaded {run.name} (seed={seed})")

    if not dfs:
        print("No matching runs found!")
        return

    total_df = pd.concat(dfs, ignore_index=True)
    if max_steps is not None:
        total_df = total_df[total_df['step'] < max_steps]
    sns.lineplot(data=total_df, x='step', y='avg_ret', hue='alg')
    if fig_name is not None:
        plt.savefig(fig_name)
    else:
        plt.show()


def load_best_results(pattern, env_name, show_df=False,
              max_steps=None):
    """Load best results for runs matching a name pattern, fetched from wandb."""
    api = wandb.Api(timeout=120)
    matched_runs = _fetch_runs_by_name_pattern(api, env_name, pattern)
    
    dfs = []
    for run in matched_runs:
        df = _run_history_to_df(run)
        if df is None:
            continue
        if max_steps is not None:
            df = df[df['step'] < max_steps]
        best_row = df.loc[df['avg_ret'].idxmax()].copy()
        best_row['seed'] = str(run.config.get("seed", "?"))
        dfs.append(best_row)
    
    if not dfs:
        print("No matching runs found!")
        return None
    
    total_df = pd.concat(dfs, ignore_index=True, axis=1).T
    if show_df:
        print(total_df.to_markdown())
    print(f"${total_df['avg_ret'].astype(float).mean():.3f} \pm {total_df['avg_ret'].astype(float).std():.3f}$")
    return total_df

if __name__ == "__main__":
    # patterns_dict = {
    #     'sampling_ema': r".*diffv2.*01.*diffv2_sampling_with_ema$",
    #     # 'qsm': r".*qsm.*01.*atp1$",
    #     # 'sac': r".*sac.*01.*atp1$"
    # }
    # plot_mean(patterns_dict, 'Ant-v4')
    pass
