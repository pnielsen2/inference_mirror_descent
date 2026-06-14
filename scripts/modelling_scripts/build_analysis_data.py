#!/usr/bin/env python3
"""Build the final analysis CSV by appending smoothed_episode_return to the
sweep config index.

Reads:
    data/modelling_data/{sweep_id}/sweep_config_index.csv
    data/plotting_data/{sweep_id}/processed_episode_returns/{run_id}/{run_id}_smoothed_returns.csv

Writes:
    data/modelling_data/{sweep_id}/{sweep_id}_analysis_data.csv

Usage:
    python build_analysis_data.py --sweep-id 83
"""

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep-id', type=int, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(REPO_ROOT / 'data' / 'modelling_data' / str(args.sweep_id)
                     / 'sweep_config_index.csv')
    smoothed_returns = []

    for i, row in df.iterrows():
        run_id = str(row['run_id'])
        print(f'[{i + 1}/{len(df)}] {row.to_dict()}')
        smoothed_path = (REPO_ROOT / 'data' / 'plotting_data' / str(args.sweep_id)
                         / 'processed_episode_returns' / run_id
                         / f'{run_id}_smoothed_returns.csv')
        smoothed_df = pd.read_csv(smoothed_path)
        smoothed_returns.append(float(smoothed_df['mean'].iloc[-1]))

    df['smoothed_episode_return'] = smoothed_returns

    out_path = (REPO_ROOT / 'data' / 'modelling_data' / str(args.sweep_id)
                / f'{args.sweep_id}_analysis_data.csv')
    df.to_csv(out_path, index=False)
    print(f'saved {out_path}')


if __name__ == '__main__':
    main()
