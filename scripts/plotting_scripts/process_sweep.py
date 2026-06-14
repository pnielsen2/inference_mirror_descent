#!/usr/bin/env python3
"""Orchestrate full sweep processing: copy raw returns, build config index,
fit Kalman models, and assemble analysis data.

Usage:
    python process_sweep.py --sweep-id 83
    python process_sweep.py --sweep-id 83 --interval 5000
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

DEFAULT_WANDB_ROOT = Path('/n/netscratch/kdbrantley_lab/Lab/pnielsen/wandb')
PLOTTING_DIR = Path(__file__).resolve().parent
MODELLING_DIR = Path(__file__).resolve().parent.parent / 'modelling_scripts'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep-id', type=int, required=True)
    p.add_argument('--wandb-root', type=Path, default=DEFAULT_WANDB_ROOT)
    p.add_argument('--interval', type=float, default=5000.0)
    return p.parse_args()


def call_script(path, *args):
    cmd = [sys.executable, str(path)] + [str(a) for a in args]
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    sweep_root = args.wandb_root / f'sweep_{args.sweep_id}'

    # Read ablations from the first config.yaml found.
    first_config = None
    for job_dir in sweep_root.iterdir():
        if not job_dir.name.startswith('job_'):
            continue
        for run_dir in (job_dir / 'wandb').iterdir():
            if run_dir.name.startswith('offline-run-'):
                first_config = yaml.safe_load((run_dir / 'files' / 'config.yaml').read_text())
                break
        if first_config is not None:
            break

    ablation_keys = [k.strip() for k in first_config['config_tag_keys']['value'].split(',')]
    print(f'[sweep {args.sweep_id}] ablation keys: {ablation_keys}')

    total = 0
    for job_dir in sweep_root.iterdir():
        if not job_dir.name.startswith('job_'):
            continue
        for run_dir in (job_dir / 'wandb').iterdir():
            if not run_dir.name.startswith('offline-run-'):
                continue
            run_id = run_dir.name.rsplit('-', 1)[1]
            total += 1
            print(f'[{total}] run_id={run_id}')

            call_script(PLOTTING_DIR / 'copy_raw_returns.py',
                '--sweep-id', args.sweep_id,
                '--run-id', run_id,
                '--run-dir', run_dir)

            call_script(PLOTTING_DIR / 'update_config_index.py',
                '--sweep-id', args.sweep_id,
                '--run-id', run_id,
                '--run-dir', run_dir)

    print(f'[done] processed {total} runs, now fitting Kalman models')
    call_script(PLOTTING_DIR / 'fit_kalman_runs.py',
        '--sweep-id', args.sweep_id,
        '--interval', args.interval)

    print(f'[done] fitting complete, now building analysis data')
    call_script(MODELLING_DIR / 'build_analysis_data.py',
        '--sweep-id', args.sweep_id)

    print(f'[done] analysis data built, now computing last_50k_mean')
    call_script(MODELLING_DIR / 'add_last_50k_mean.py',
        '--sweep-id', args.sweep_id)

    print(f'[done] all done')


if __name__ == '__main__':
    main()
