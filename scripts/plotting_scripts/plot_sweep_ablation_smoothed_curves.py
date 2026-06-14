#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CI_LEVEL = 0.90
PALETTE = plt.cm.tab10.colors
_STATS_MODULE: Any | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep-id', type=int, required=True)
    parser.add_argument('--out-dir', type=Path, default=None)
    parser.add_argument('--ci-level', type=float, default=CI_LEVEL)
    return parser.parse_args()


def level_sort_key(value):
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        return (0, 1 if math.isinf(value) else 0, value)
    return (1, 0, str(value))


def level_label(value) -> str:
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        return 'inf' if math.isinf(value) else f'{value:g}'
    return str(value)


def subplot_shape(n: int) -> tuple[int, int]:
    if n <= 1:
        return 1, 1
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def fit_y_axis(ax, y_parts: list[np.ndarray]) -> None:
    arrays = []
    for part in y_parts:
        arr = np.asarray(part, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            arrays.append(arr)
    if not arrays:
        return
    values = np.concatenate(arrays)
    ymin = float(values.min())
    ymax = float(values.max())
    if ymin == ymax:
        pad = max(1.0, 0.05 * max(abs(ymin), 1.0))
    else:
        pad = 0.05 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)


def get_stats_module():
    global _STATS_MODULE
    if _STATS_MODULE is None:
        print('importing scipy.stats...', flush=True)
        from scipy import stats
        _STATS_MODULE = stats
        print('imported scipy.stats', flush=True)
    return _STATS_MODULE


def load_metadata(sweep_id: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    index_path = REPO_ROOT / 'data' / 'modelling_data' / str(sweep_id) / 'sweep_config_index.csv'
    print(f'loading metadata from {index_path}', flush=True)
    df = pd.read_csv(index_path)
    if 'run_id' not in df.columns:
        raise ValueError(f'missing run_id column in {index_path}')
    envs = sorted(df['env'].dropna().astype(str).unique().tolist()) if 'env' in df.columns else ['all']
    ablations = [c for c in df.columns if c not in {'run_id', 'env', 'smoothed_episode_return'}]
    if not ablations:
        raise ValueError(f'no ablation columns found in {index_path}')
    df['run_id'] = df['run_id'].astype(str)
    if 'env' in df.columns:
        df['env'] = df['env'].astype(str)
    print(f'loaded metadata: {len(df)} runs, {len(envs)} envs, ablations={ablations}', flush=True)
    return df, envs, ablations


def load_smoothed_run(sweep_id: int, run_id: str) -> pd.DataFrame:
    path = (
        REPO_ROOT / 'data' / 'plotting_data' / str(sweep_id)
        / 'processed_episode_returns' / run_id / f'{run_id}_smoothed_returns.csv'
    )
    df = pd.read_csv(path)
    required = {'_step', 'mean', 'mean_var'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'missing columns {sorted(missing)} in {path}')
    out = df[['_step', 'mean', 'mean_var']].copy()
    out = out.dropna(subset=['_step', 'mean', 'mean_var']).sort_values('_step').reset_index(drop=True)
    out['_step'] = out['_step'].astype(float)
    out['mean'] = out['mean'].astype(float)
    out['mean_var'] = out['mean_var'].astype(float).clip(lower=0.0)
    return out


def build_run_table(sweep_id: int, meta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(meta_df)
    print(f'building run table for {total} runs', flush=True)
    for i, row in enumerate(meta_df.itertuples(index=False), start=1):
        run_id = str(row.run_id)
        print(f'[{i}/{total}] loading {run_id}', flush=True)
        smoothed = load_smoothed_run(sweep_id, run_id)
        record = row._asdict()
        record['smoothed_df'] = smoothed
        rows.append(record)
    print('finished building run table', flush=True)
    return pd.DataFrame(rows)


def aggregate_group(run_dfs: list[pd.DataFrame], ci_level: float) -> pd.DataFrame:
    stats = get_stats_module()
    merged = None
    for idx, run_df in enumerate(run_dfs):
        run_frame = run_df.rename(
            columns={
                'mean': f'mean_{idx}',
                'mean_var': f'mean_var_{idx}',
            }
        )
        if merged is None:
            merged = run_frame
        else:
            merged = merged.merge(run_frame, on='_step', how='outer')
    if merged is None:
        return pd.DataFrame(columns=['_step', 'mean', 'ci_lo', 'ci_hi', 'n'])
    merged = merged.sort_values('_step').reset_index(drop=True)
    mean_cols = [c for c in merged.columns if c.startswith('mean_') and not c.startswith('mean_var_')]
    var_cols = [c for c in merged.columns if c.startswith('mean_var_')]
    means = merged[mean_cols].to_numpy(dtype=float)
    variances = merged[var_cols].to_numpy(dtype=float)
    counts = np.sum(~np.isnan(means), axis=1)
    point_mean = np.nanmean(means, axis=1)
    point_mean[counts == 0] = np.nan
    total_var = np.full(len(merged), np.nan)
    one_mask = counts == 1
    if np.any(one_mask):
        total_var[one_mask] = np.nansum(variances[one_mask], axis=1)
    many_mask = counts > 1
    if np.any(many_mask):
        between = np.nanvar(means[many_mask], axis=1, ddof=1) / counts[many_mask]
        within = np.nansum(variances[many_mask], axis=1) / (counts[many_mask] ** 2)
        total_var[many_mask] = between + within
    se = np.sqrt(np.clip(total_var, 0.0, None))
    half_width = np.full(len(merged), np.nan)
    z_crit = stats.norm.ppf(1.0 - (1.0 - ci_level) / 2.0)
    if np.any(one_mask):
        half_width[one_mask] = z_crit * se[one_mask]
    for n in sorted({int(v) for v in counts if v > 1}):
        mask = counts == n
        t_crit = stats.t.ppf(1.0 - (1.0 - ci_level) / 2.0, df=n - 1)
        half_width[mask] = t_crit * se[mask]
    return pd.DataFrame({
        '_step': merged['_step'].to_numpy(dtype=float),
        'mean': point_mean,
        'ci_lo': point_mean - half_width,
        'ci_hi': point_mean + half_width,
        'n': counts.astype(int),
    })


def plot_ablation(run_table: pd.DataFrame, axis: str, envs: list[str], out_dir: Path, ci_level: float) -> None:
    levels = sorted(run_table[axis].dropna().unique().tolist(), key=level_sort_key)
    print(f'plotting ablation axis {axis} with {len(levels)} levels across {len(envs)} envs', flush=True)
    rows, cols = subplot_shape(len(envs))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.8 * rows), sharex=True)
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)
    axes_flat = list(axes_arr.ravel())
    x_parts = []
    for env_idx, env in enumerate(envs):
        ax = axes_flat[env_idx]
        y_parts: list[np.ndarray] = []
        env_table = run_table[run_table['env'] == env] if 'env' in run_table.columns else run_table
        print(f'  env {env} ({env_idx + 1}/{len(envs)}): {len(env_table)} runs', flush=True)
        for level_idx, level in enumerate(levels):
            group = env_table[env_table[axis] == level]
            if group.empty:
                continue
            print(
                f'    level {level_label(level)} ({level_idx + 1}/{len(levels)}): '
                f'aggregating {len(group)} runs',
                flush=True,
            )
            agg = aggregate_group(group['smoothed_df'].tolist(), ci_level)
            valid = np.isfinite(agg['mean'].to_numpy(dtype=float))
            if not np.any(valid):
                print(f'    level {level_label(level)}: no valid aggregated points', flush=True)
                continue
            x = agg['_step'].to_numpy(dtype=float) / 1e6
            mean = agg['mean'].to_numpy(dtype=float)
            ci_lo = agg['ci_lo'].to_numpy(dtype=float)
            ci_hi = agg['ci_hi'].to_numpy(dtype=float)
            color = PALETTE[level_idx % len(PALETTE)]
            label = f'{level_label(level)} (n={len(group)})'
            ax.plot(x[valid], mean[valid], color=color, linewidth=1.6, label=label)
            band = valid & np.isfinite(ci_lo) & np.isfinite(ci_hi)
            if np.any(band):
                ax.fill_between(x[band], ci_lo[band], ci_hi[band], color=color, alpha=0.18)
                y_parts.extend([ci_lo[band], ci_hi[band]])
            y_parts.append(mean[valid])
            x_parts.append(x[valid])
            print(
                f'    level {level_label(level)}: plotted {int(np.sum(valid))} points',
                flush=True,
            )
        ax.set_title(env.replace('-v3', ''), fontsize=11)
        ax.set_xlabel('Env steps (M)', fontsize=10)
        ax.set_ylabel('Episode return', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=9)
        if y_parts:
            fit_y_axis(ax, y_parts)
        print(f'  finished env {env}', flush=True)
    for ax in axes_flat[len(envs):]:
        ax.set_visible(False)
    if x_parts:
        xmax = max(float(np.nanmax(part)) for part in x_parts if len(part) > 0)
        for ax in axes_flat[:len(envs)]:
            ax.set_xlim(0.0, xmax)
    handles, labels = [], []
    for ax in axes_flat[:len(envs)]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=max(1, min(len(handles), 6)), fontsize=9, bbox_to_anchor=(0.5, 0.0))
    ci_pct = int(round(100 * ci_level))
    fig.suptitle(f'Sweep {int(run_table["sweep_id"].iloc[0])} — ablation axis: {axis} ({ci_pct}% CI)', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    out_path = out_dir / f'sweep_{int(run_table["sweep_id"].iloc[0])}_ablation_smoothed_curves_{axis}.png'
    print(f'saving figure to {out_path}', flush=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {out_path}', flush=True)


def main() -> None:
    args = parse_args()
    print(f'starting plot_sweep_ablation_smoothed_curves for sweep {args.sweep_id}', flush=True)
    meta_df, envs, ablations = load_metadata(args.sweep_id)
    out_dir = args.out_dir or (REPO_ROOT / 'figures' / f'sweep_{args.sweep_id}_smoothed_ablations')
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'output directory: {out_dir}', flush=True)
    run_table = build_run_table(args.sweep_id, meta_df)
    run_table['sweep_id'] = int(args.sweep_id)
    print(f'loaded {len(run_table)} runs', flush=True)
    print(f'ablations: {ablations}', flush=True)
    for axis_idx, axis in enumerate(ablations, start=1):
        print(f'[{axis_idx}/{len(ablations)}] starting ablation plot for {axis}', flush=True)
        plot_ablation(run_table, axis, envs, out_dir, args.ci_level)
        print(f'[{axis_idx}/{len(ablations)}] finished ablation plot for {axis}', flush=True)
    print('all plots finished', flush=True)


if __name__ == '__main__':
    main()
