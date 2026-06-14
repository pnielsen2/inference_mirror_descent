#!/usr/bin/env python3
"""Add or update the row for a single run in the sweep config index CSV.

Usage:
    python update_config_index.py --sweep-id 83 --run-id 9l6nv84u --run-dir /path/to/offline-run-...
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def update_config_index(sweep_id: int, run_id: str, run_dir: Path) -> None:
    index_path = (REPO_ROOT / 'data' / 'modelling_data' / str(sweep_id)
                  / 'sweep_config_index.csv')
    if index_path.exists():
        df_check = pd.read_csv(index_path)
        if run_id in df_check['run_id'].astype(str).values:
            print(f'skipping {run_id} (already in config_index)')
            return
    print(f'adding {run_id} to the config_index')

    config = yaml.safe_load((run_dir / 'files' / 'config.yaml').read_text())
    ablation_keys = [k.strip() for k in config['config_tag_keys']['value'].split(',')]

    row = {'run_id': run_id, 'env': config['env']['value']}
    for key in ablation_keys:
        if key != 'env':
            row[key] = config[key]['value']

    index_path = (REPO_ROOT / 'data' / 'modelling_data' / str(sweep_id)
                  / 'sweep_config_index.csv')
    index_path.parent.mkdir(parents=True, exist_ok=True)

    if index_path.exists():
        df = pd.read_csv(index_path)
        df = df[df['run_id'] != run_id].reset_index(drop=True)
    else:
        df = pd.DataFrame()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(index_path, index=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep-id', type=int, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    update_config_index(args.sweep_id, args.run_id, args.run_dir)


if __name__ == '__main__':
    main()
