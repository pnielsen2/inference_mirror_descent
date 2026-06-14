#!/usr/bin/env python3
"""Append last_50k_mean to the sweep analysis CSV.

last_50k_mean is the mean episode return across all episodes whose final
step fell in [950000, 1000000] (the last 50k steps of a 1M-step run).

Reads/writes:
    data/modelling_data/{sweep_id}/{sweep_id}_analysis_data.csv
    data/plotting_data/{sweep_id}/raw_episode_returns/{run_id}.csv

Usage:
    python add_last_50k_mean.py --sweep-id 83
"""

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
THRESHOLD = 950_000  # last 50k of 1M steps


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep-id', type=int, required=True)
    return p.parse_args()


def csv_mean_above_threshold(path: Path, step_col: str, val_col: str, threshold: int) -> float:
    total, count = 0.0, 0
    with open(path) as f:
        for row in csv.DictReader(f):
            if float(row[step_col]) >= threshold:
                total += float(row[val_col])
                count += 1
    return total / count


def main():
    args = parse_args()
    analysis_path = (REPO_ROOT / 'data' / 'modelling_data' / str(args.sweep_id)
                     / f'{args.sweep_id}_analysis_data.csv')

    with open(analysis_path) as f:
        rows = list(csv.DictReader(f))
    fieldnames = list(rows[0].keys())
    if 'last_50k_mean' not in fieldnames:
        fieldnames.append('last_50k_mean')

    for i, row in enumerate(rows):
        run_id = row['run_id']
        print(f'[{i + 1}/{len(rows)}] computing last_50k_mean for {run_id}')
        raw_path = (REPO_ROOT / 'data' / 'plotting_data' / str(args.sweep_id)
                    / 'raw_episode_returns' / f'{run_id}.csv')
        with open(raw_path) as f:
            raw_header = next(csv.reader(f))
        ep_col = next(c for c in raw_header if c not in ('_step', '_runtime'))
        val = csv_mean_above_threshold(raw_path, '_step', ep_col, THRESHOLD)
        row['last_50k_mean'] = f'{val}'
        print(f'[{i + 1}/{len(rows)}] {run_id} last_50k_mean={val:.2f}')

    with open(analysis_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'saved {analysis_path}')


if __name__ == '__main__':
    main()
