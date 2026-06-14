#!/usr/bin/env python3
"""Copy the episode returns for a single run from its W&B offline bundle
to the local data directory.

Usage:
    python copy_raw_returns.py --sweep-id 83 --run-id 9l6nv84u --run-dir /path/to/offline-run-...
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def copy_raw_returns(sweep_id: int, run_id: str, run_dir: Path) -> None:
    out_path = (REPO_ROOT / 'data' / 'plotting_data' / str(sweep_id)
                / 'raw_episode_returns' / f'{run_id}.csv')
    if out_path.exists():
        print(f'skipping raw {run_id} episode returns (already exists)')
        return
    print(f'copying raw {run_id} episode returns')

    config = yaml.safe_load((run_dir / 'files' / 'config.yaml').read_text())
    env_name = config['env']['value']
    metric_name = f'episode_return/{env_name}'

    wandb_bundle = run_dir / f'run-{run_id}.wandb'

    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal import datastore

    store = datastore.DataStore()
    store.open_for_scan(str(wandb_bundle))
    rows = []

    while True:
        data = store.scan_data()
        if data is None:
            break
        record = wandb_internal_pb2.Record()
        record.ParseFromString(data)
        if record.WhichOneof('record_type') != 'history':
            continue
        row = {}
        for item in record.history.item:
            key = '.'.join(item.nested_key) if item.nested_key else item.key
            if item.value_json:
                row[key] = json.loads(item.value_json)
        if metric_name in row:
            rows.append({
                '_step': row.get('_step'),
                metric_name: row[metric_name],
                '_runtime': row.get('_runtime'),
            })

    df = pd.DataFrame(rows).sort_values('_step').reset_index(drop=True)

    out_path = (REPO_ROOT / 'data' / 'plotting_data' / str(sweep_id)
                / 'raw_episode_returns' / f'{run_id}.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep-id', type=int, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    copy_raw_returns(args.sweep_id, args.run_id, args.run_dir)


if __name__ == '__main__':
    main()
