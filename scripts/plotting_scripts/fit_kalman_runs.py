#!/usr/bin/env python3
"""Fit Kalman filter parameters for every run in the sweep config index,
then compute filtered and smoothed outputs on a regular grid.

Reads:
    data/modelling_data/{sweep_id}/sweep_config_index.csv
    data/plotting_data/{sweep_id}/raw_episode_returns/{run_id}.csv

Writes:
    data/plotting_data/{sweep_id}/processed_episode_returns/{run_id}/{run_id}_filtered_returns.csv
    data/plotting_data/{sweep_id}/processed_episode_returns/{run_id}/{run_id}_smoothed_returns.csv

Each CSV has columns: _step, mean, mean_var, obs_var
    mean     - posterior mean of the latent state (= observation distribution mean)
    mean_var - posterior variance of the latent state
    obs_var  - posterior predictive variance of a single observation (mean_var + r)

Filtered grid: [interval, 2*interval, ..., smallest multiple of interval >= max(_step)]
Smoothed grid: [0, interval, 2*interval, ..., same max]

Usage:
    python fit_kalman_runs.py --sweep-id 83
    python fit_kalman_runs.py --sweep-id 83 --interval 5000
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Kalman filter / smoother
# ---------------------------------------------------------------------------

@dataclass
class Kalman1DParams:
    a: float
    q: float
    r: float
    m0: float
    p0: float
    b: float = 0.0


class Kalman1D:
    def __init__(self, params: Kalman1DParams):
        self.params = params

    def _transition_components(self, dt):
        a = float(self.params.a)
        b = float(self.params.b)
        q = float(self.params.q)
        dt = float(dt)
        if dt < 0:
            raise ValueError('times must be nondecreasing')
        a_dt = a ** dt
        if np.isclose(a, 1.0):
            drift = b * dt
        else:
            drift = b * (1.0 - a_dt) / (1.0 - a)
        if np.isclose(a * a, 1.0):
            process_var = q * dt
        else:
            process_var = q * (1.0 - (a * a) ** dt) / (1.0 - a * a)
        return a_dt, drift, process_var

    def _transition(self, mean, var, dt):
        a_dt, drift, process_var = self._transition_components(dt)
        return a_dt * mean + drift, (a_dt ** 2) * var + process_var

    def filter_timeline(self, y, observation_times, output_times):
        y = np.asarray(y, dtype=float)
        observation_times = np.asarray(observation_times, dtype=float)
        output_times = np.asarray(output_times, dtype=float)
        timeline = np.unique(np.concatenate([observation_times, output_times]))

        r = float(self.params.r)
        m = float(self.params.m0)
        p = float(self.params.p0)
        obs_idx = 0
        total_loglik = 0.0
        rows = []
        previous_time = None

        for time in timeline:
            dt = 0.0 if previous_time is None else float(time - previous_time)
            predicted_mean, predicted_var = self._transition(m, p, dt)
            filtered_mean = predicted_mean
            filtered_var = predicted_var

            if obs_idx < len(observation_times) and np.isclose(observation_times[obs_idx], time):
                obs = float(y[obs_idx])
                predictive_var = predicted_var + r
                if predictive_var > 0.0:
                    gain = predicted_var / predictive_var
                    filtered_mean = predicted_mean + gain * (obs - predicted_mean)
                    filtered_var = (1.0 - gain) * predicted_var
                    total_loglik += -0.5 * (
                        np.log(2.0 * np.pi * predictive_var)
                        + (obs - predicted_mean) ** 2 / predictive_var
                    )
                else:
                    total_loglik = -np.inf
                obs_idx += 1

            rows.append({
                'time': float(time),
                'predicted_mean': predicted_mean,
                'predicted_var': predicted_var,
                'filtered_mean': filtered_mean,
                'filtered_var': filtered_var,
            })
            m = filtered_mean
            p = filtered_var
            previous_time = float(time)

        full_df = pd.DataFrame(rows)
        return full_df, total_loglik

    def smooth_timeline(self, filtered_full_df, output_times):
        full_df = filtered_full_df.copy().reset_index(drop=True)
        times = full_df['time'].to_numpy(dtype=float)
        f_mean = full_df['filtered_mean'].to_numpy(dtype=float)
        f_var = full_df['filtered_var'].to_numpy(dtype=float)
        p_mean = full_df['predicted_mean'].to_numpy(dtype=float)
        p_var = full_df['predicted_var'].to_numpy(dtype=float)

        s_mean = f_mean.copy()
        s_var = f_var.copy()

        for i in range(len(full_df) - 2, -1, -1):
            dt_next = float(times[i + 1] - times[i])
            a_dt, _, _ = self._transition_components(dt_next)
            next_p_var = float(p_var[i + 1])
            gain = float(f_var[i] * a_dt / next_p_var) if next_p_var > 0.0 else 0.0
            s_mean[i] = f_mean[i] + gain * (s_mean[i + 1] - p_mean[i + 1])
            s_var[i] = max(f_var[i] + gain ** 2 * (s_var[i + 1] - next_p_var), 0.0)

        full_df['smoothed_mean'] = s_mean
        full_df['smoothed_var'] = s_var

        output_times_arr = np.asarray(output_times, dtype=float)
        output_df = full_df[full_df['time'].isin(output_times_arr)].copy().reset_index(drop=True)
        return output_df


# ---------------------------------------------------------------------------
# Parameter fitting
# ---------------------------------------------------------------------------

def fit_params(times: np.ndarray, y: np.ndarray) -> Kalman1DParams:
    y_var = float(max(np.var(y), 1.0))
    window = y[:min(len(y), 100)]
    diff_window = np.diff(window) if len(window) > 1 else np.array([1.0])

    initial = Kalman1DParams(
        a=0.9999,
        q=float(max(np.var(diff_window), 1e-6)),
        r=float(max(0.05 * y_var, 1e-6)),
        m0=float(y[0]),
        p0=y_var,
        b=0.0,
    )

    def pack(a_raw, log_q, log_r, b, m0, log_p0):
        return Kalman1DParams(
            a=1.0 / (1.0 + np.exp(-a_raw)),
            q=np.exp(log_q),
            r=np.exp(log_r),
            m0=m0,
            p0=np.exp(log_p0),
            b=b,
        )

    def objective(theta):
        params = pack(*theta)
        model = Kalman1D(params)
        _, loglik = model.filter_timeline(y, observation_times=times, output_times=times)
        return -loglik

    x0 = np.array([
        np.log(initial.a / (1.0 - initial.a)),
        np.log(initial.q),
        np.log(initial.r),
        initial.b,
        initial.m0,
        np.log(initial.p0),
    ], dtype=float)

    log_min = -20.0
    bounds = [
        (None, None),   # a_raw
        (log_min, None),  # log_q
        (log_min, None),  # log_r
        (None, None),   # b
        (None, None),   # m0
        (log_min, None),  # log_p0
    ]
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    return pack(*result.x)


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------

def process_run(run_id: str, sweep_id: int, interval: float, idx: int = 0, total: int = 0) -> None:
    tag = f'[{idx}/{total}] ' if total else ''
    out_dir = (REPO_ROOT / 'data' / 'plotting_data' / str(sweep_id)
               / 'processed_episode_returns' / run_id)
    if (out_dir / f'{run_id}_smoothed_returns.csv').exists():
        print(f'{tag}skipping {run_id} (already processed)')
        return
    raw_path = (REPO_ROOT / 'data' / 'plotting_data' / str(sweep_id)
                / 'raw_episode_returns' / f'{run_id}.csv')
    df = pd.read_csv(raw_path)
    ep_col = next(c for c in df.columns if c not in ('_step', '_runtime', '_timestamp'))
    df = df.dropna(subset=[ep_col]).sort_values('_step').reset_index(drop=True)
    times = df['_step'].to_numpy(dtype=float)
    y = df[ep_col].to_numpy(dtype=float)

    print(f'{tag}fitting {run_id}')
    params = fit_params(times, y)
    print(f'{tag}done fitting {run_id}')

    model = Kalman1D(params)
    r = float(params.r)

    max_step = float(times[-1])
    max_grid = math.ceil(max_step / interval) * interval
    regular_grid = np.arange(interval, max_grid + 0.5, interval, dtype=float)
    smoothed_grid = np.concatenate([[0.0], regular_grid])

    print(f'{tag}filtering {run_id}')
    full_df, _ = model.filter_timeline(y, observation_times=times, output_times=smoothed_grid)

    filtered_rows = full_df[full_df['time'].isin(regular_grid)].reset_index(drop=True)
    filtered_csv = pd.DataFrame({
        '_step': filtered_rows['time'].astype(int),
        'mean': filtered_rows['filtered_mean'],
        'mean_var': filtered_rows['filtered_var'],
        'obs_var': filtered_rows['filtered_var'] + r,
    })
    print(f'{tag}filtered {run_id}')

    out_dir = (REPO_ROOT / 'data' / 'plotting_data' / str(sweep_id)
               / 'processed_episode_returns' / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    filtered_csv.to_csv(out_dir / f'{run_id}_filtered_returns.csv', index=False)
    print(f'{tag}stored filtered {run_id}')

    print(f'{tag}smoothing {run_id}')
    smoothed_rows = model.smooth_timeline(full_df, output_times=smoothed_grid)
    smoothed_csv = pd.DataFrame({
        '_step': smoothed_rows['time'].astype(int),
        'mean': smoothed_rows['smoothed_mean'],
        'mean_var': smoothed_rows['smoothed_var'],
        'obs_var': smoothed_rows['smoothed_var'] + r,
    })
    print(f'{tag}smoothed {run_id}')

    print(f'{tag}storing smoothed {run_id}')
    smoothed_csv.to_csv(out_dir / f'{run_id}_smoothed_returns.csv', index=False)
    print(f'{tag}stored smoothed {run_id}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--sweep-id', type=int, required=True)
    p.add_argument('--interval', type=float, default=5000.0)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(REPO_ROOT / 'data' / 'modelling_data' / str(args.sweep_id)
                     / 'sweep_config_index.csv')
    total = len(df)
    for i, row in enumerate(df.itertuples(), start=1):
        process_run(str(row.run_id), args.sweep_id, args.interval, i, total)


if __name__ == '__main__':
    main()
