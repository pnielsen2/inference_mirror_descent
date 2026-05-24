from argparse import ArgumentParser
from dataclasses import asdict, dataclass
from pathlib import Path
import json


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_WANDB_ROOT = Path('/n/netscratch/kdbrantley_lab/Lab/pnielsen/wandb')
MODEL_CACHE_PATH = SCRIPT_DIR / 'plotting_data' / 'kalman_models.csv'
SMOOTHED_RUNS_DIR = SCRIPT_DIR / 'plotting_data' / 'smoothed_runs'
ANALYSIS_CACHE_DIR = PROJECT_ROOT / 'analysis_cache'
CACHE_COLUMNS = [
    'run_id',
    'sweep_id',
    'run_dir',
    'env_name',
    'metric_name',
    'num_observations',
    'total_loglik',
    'a',
    'q',
    'r',
    'm0',
    'p0',
    'b',
]
SMOOTHED_RUN_COLUMNS = [
    '_step',
    'filtered_mean',
    'filtered_mean_var',
    'filtered_obs_var',
    'smoothed_mean',
    'smoothed_mean_var',
    'smoothed_obs_var',
]
REGULAR_OUTPUT_COLUMN_MAP = {
    'filtered': ['filtered_mean', 'filtered_mean_var', 'filtered_obs_var'],
    'smoothed': ['smoothed_mean', 'smoothed_mean_var', 'smoothed_obs_var'],
    'both': [
        'filtered_mean',
        'filtered_mean_var',
        'filtered_obs_var',
        'smoothed_mean',
        'smoothed_mean_var',
        'smoothed_obs_var',
    ],
}


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
        import numpy as np
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
        pred_mean = a_dt * mean + drift
        pred_var = (a_dt ** 2) * var + process_var
        return pred_mean, pred_var

    def filter_timeline(self, y, observation_times=None, output_times=None):
        import numpy as np
        import pandas as pd
        y = np.asarray(y, dtype=float)
        if observation_times is None:
            observation_times = np.arange(len(y), dtype=float)
        else:
            observation_times = np.asarray(observation_times, dtype=float)
        if len(observation_times) != len(y):
            raise ValueError('observation_times must have the same length as y')
        if len(observation_times) > 1 and np.any(np.diff(observation_times) <= 0):
            raise ValueError('observation_times must be strictly increasing')
        if output_times is None:
            output_times = observation_times
        else:
            output_times = np.unique(np.asarray(output_times, dtype=float))
            if len(output_times) > 1 and np.any(np.diff(output_times) < 0):
                raise ValueError('output_times must be nondecreasing')
        timeline = np.unique(np.concatenate([observation_times, output_times])).astype(float)
        r = float(self.params.r)
        m = float(self.params.m0)
        p = float(self.params.p0)
        obs_idx = 0
        rows = []
        total_loglik = 0.0
        previous_time = None

        for timeline_index, time in enumerate(timeline):
            dt = 0.0 if previous_time is None else float(time - previous_time)
            predicted_mean, predicted_var = self._transition(m, p, dt)
            predictive_mean = predicted_mean
            predictive_var = predicted_var + r
            filtered_mean = predicted_mean
            filtered_var = predicted_var
            observation = np.nan
            loglik = np.nan
            is_observation = False

            if obs_idx < len(observation_times) and np.isclose(observation_times[obs_idx], time):
                observation = float(y[obs_idx])
                innovation = observation - predictive_mean
                gain = predicted_var / predictive_var
                filtered_mean = predicted_mean + gain * innovation
                filtered_var = (1.0 - gain) * predicted_var
                loglik = -0.5 * (np.log(2.0 * np.pi * predictive_var) + (innovation ** 2) / predictive_var)
                total_loglik += float(loglik)
                is_observation = True
                obs_idx += 1

            rows.append({
                'timeline_index': timeline_index,
                'time': float(time),
                'dt': dt,
                'is_observation': is_observation,
                'observation': observation,
                'predicted_mean': predicted_mean,
                'predicted_var': predicted_var,
                'predictive_mean': predictive_mean,
                'predictive_var': predictive_var,
                'filtered_mean': filtered_mean,
                'filtered_var': filtered_var,
                'loglik': float(loglik) if is_observation else np.nan,
            })
            m = filtered_mean
            p = filtered_var
            previous_time = float(time)

        full_df = pd.DataFrame(rows)
        output_df = full_df.loc[full_df['time'].isin(output_times)].copy().reset_index(drop=True)
        output_df.insert(0, 't', np.arange(len(output_df)))
        return full_df, output_df, total_loglik

    def filter(self, y, times=None):
        _, output_df, total_loglik = self.filter_timeline(y, observation_times=times, output_times=times)
        return output_df, total_loglik

    def smooth_timeline(self, filtered_full_df, output_times=None):
        import numpy as np
        full_df = filtered_full_df.copy().reset_index(drop=True)
        times = full_df['time'].to_numpy(dtype=float)
        filtered_mean = full_df['filtered_mean'].to_numpy(dtype=float)
        filtered_var = full_df['filtered_var'].to_numpy(dtype=float)
        predicted_mean = full_df['predicted_mean'].to_numpy(dtype=float)
        predicted_var = full_df['predicted_var'].to_numpy(dtype=float)

        smoothed_mean = filtered_mean.copy()
        smoothed_var = filtered_var.copy()

        for idx in range(len(full_df) - 2, -1, -1):
            dt_next = float(times[idx + 1] - times[idx])
            transition_coeff, _, _ = self._transition_components(dt_next)
            next_predicted_var = float(predicted_var[idx + 1])
            if next_predicted_var <= 0.0:
                smoothing_gain = 0.0
            else:
                smoothing_gain = float(filtered_var[idx] * transition_coeff / next_predicted_var)
            smoothed_mean[idx] = filtered_mean[idx] + smoothing_gain * (smoothed_mean[idx + 1] - predicted_mean[idx + 1])
            smoothed_var[idx] = max(
                filtered_var[idx] + (smoothing_gain ** 2) * (smoothed_var[idx + 1] - next_predicted_var),
                0.0,
            )

        full_df['smoothed_mean'] = smoothed_mean
        full_df['smoothed_mean_var'] = smoothed_var
        full_df['smoothed_obs_var'] = smoothed_var + float(self.params.r)

        if output_times is None:
            output_times = times
        else:
            output_times = np.unique(np.asarray(output_times, dtype=float))
        output_df = full_df.loc[full_df['time'].isin(output_times)].copy().reset_index(drop=True)
        output_df.insert(0, 't', np.arange(len(output_df)))
        return full_df, output_df


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--sweep-id', type=int, default=83)
    parser.add_argument('--wandb-root', type=Path, default=DEFAULT_WANDB_ROOT)
    parser.add_argument('--regular-output', choices=['none', 'filtered', 'smoothed', 'both'], default='none')
    parser.add_argument('--interval', type=float, default=5000.0)
    return parser.parse_args()


def resolve_run_dir(run_id: str, sweep_id: int, wandb_root: Path) -> Path:
    sweep_root = wandb_root / f'sweep_{sweep_id}'
    if not sweep_root.exists():
        raise FileNotFoundError(f'Could not find sweep directory {sweep_root}')
    matches = sorted(sweep_root.glob(f'job_*/wandb/offline-run-*-{run_id}'))
    if not matches:
        raise FileNotFoundError(
            f'Could not find a local offline W&B directory for run_id={run_id!r} under {sweep_root}'
        )
    if len(matches) > 1:
        raise ValueError(f'Found multiple offline run directories for run_id={run_id!r}: {matches}')
    return matches[0]


def infer_env_name(run_dir: Path) -> str:
    import yaml
    summary_path = run_dir / 'files' / 'wandb-summary.json'
    config_path = run_dir / 'files' / 'config.yaml'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    env_name = summary.get('env')
    if env_name is None and config_path.exists():
        config = yaml.safe_load(config_path.read_text())
        env_entry = config.get('env')
        env_name = env_entry.get('value') if isinstance(env_entry, dict) and 'value' in env_entry else env_entry
    if env_name is None:
        raise KeyError(f'Could not infer env from {summary_path} or {config_path}')
    return env_name


def load_episode_return_df(run_id: str, run_dir: Path, metric_name: str):
    import pandas as pd
    ANALYSIS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    metric_slug = metric_name.replace('/', '_')
    csv_cache = ANALYSIS_CACHE_DIR / f'{run_id}_{metric_slug}.csv'
    if csv_cache.exists():
        return pd.read_csv(csv_cache)

    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal import datastore

    wandb_bundle = run_dir / f'run-{run_id}.wandb'
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
            if not item.value_json:
                continue
            row[key] = json.loads(item.value_json)

        if metric_name in row:
            rows.append({
                '_step': row.get('_step'),
                metric_name: row[metric_name],
                '_runtime': row.get('_runtime'),
                '_timestamp': row.get('_timestamp'),
            })

    import pandas as pd
    episode_return_df = pd.DataFrame(rows).sort_values('_step').reset_index(drop=True)
    episode_return_df.to_csv(csv_cache, index=False)
    return episode_return_df


def _read_cached_row(run_id: str):
    import csv
    if not MODEL_CACHE_PATH.exists():
        return None
    with MODEL_CACHE_PATH.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('run_id') == run_id:
                return row
    return None


def _model_params_from_row(row) -> Kalman1DParams:
    return Kalman1DParams(
        a=float(row['a']),
        q=float(row['q']),
        r=float(row['r']),
        m0=float(row['m0']),
        p0=float(row['p0']),
        b=float(row['b']),
    )


def load_model_cache():
    import numpy as np
    import pandas as pd
    MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_CACHE_PATH.exists():
        model_cache_df = pd.read_csv(MODEL_CACHE_PATH)
        missing_columns = [column for column in CACHE_COLUMNS if column not in model_cache_df.columns]
        for column in missing_columns:
            model_cache_df[column] = np.nan
        return model_cache_df[CACHE_COLUMNS]
    return pd.DataFrame(columns=CACHE_COLUMNS)


def save_model_cache(model_cache_df) -> None:
    MODEL_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_cache_df[CACHE_COLUMNS].to_csv(MODEL_CACHE_PATH, index=False)


def pack_params(a_raw, log_q, log_r, b, m0, log_p0):
    import numpy as np
    a = 1.0 / (1.0 + np.exp(-a_raw))
    q = np.exp(log_q)
    r = np.exp(log_r)
    p0 = np.exp(log_p0)
    return Kalman1DParams(a=a, q=q, r=r, m0=m0, p0=p0, b=b)


def fit_or_load_params(run_id: str, y, times):
    import numpy as np
    window = y[: min(len(y), 100)]
    diff_window = np.diff(window) if len(window) > 1 else np.array([1.0])
    y_var = float(max(np.var(y), 1.0))
    initial_params = Kalman1DParams(
        a=0.9999,
        q=float(max(np.var(diff_window), 1e-6)),
        r=float(max(0.05 * y_var, 1e-6)),
        m0=float(y[0]),
        p0=y_var,
        b=0.0,
    )

    def objective(theta):
        params = pack_params(*theta)
        model = Kalman1D(params)
        _, total_loglik = model.filter(y, times=times)
        return -total_loglik

    from scipy.optimize import minimize

    x0 = np.array([
        np.log(initial_params.a / (1.0 - initial_params.a)),
        np.log(initial_params.q),
        np.log(initial_params.r),
        initial_params.b,
        initial_params.m0,
        np.log(initial_params.p0),
    ], dtype=float)
    result = minimize(objective, x0, method='L-BFGS-B')
    fitted_params = pack_params(*result.x)
    return fitted_params, 'fit', str(result.message)


def _smoothed_run_cache_path(run_id: str) -> Path:
    return SMOOTHED_RUNS_DIR / f'{run_id}.csv'


def _regular_grid(times, interval):
    import numpy as np
    if interval <= 0:
        raise ValueError('interval must be positive')
    max_time = float(np.max(times)) if len(times) else 0.0
    return np.arange(0.0, max_time + interval, interval, dtype=float)


def _load_smoothed_run_cache(run_id: str):
    import numpy as np
    import pandas as pd
    cache_path = _smoothed_run_cache_path(run_id)
    SMOOTHED_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        cache_df = pd.read_csv(cache_path)
    else:
        cache_df = pd.DataFrame(columns=SMOOTHED_RUN_COLUMNS)
    for column in SMOOTHED_RUN_COLUMNS:
        if column not in cache_df.columns:
            cache_df[column] = np.nan
    return cache_df[SMOOTHED_RUN_COLUMNS]


def _save_smoothed_run_cache(run_id: str, cache_df) -> None:
    SMOOTHED_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    cache_df[SMOOTHED_RUN_COLUMNS].sort_values('_step').reset_index(drop=True).to_csv(
        _smoothed_run_cache_path(run_id),
        index=False,
    )


def _has_regular_output(cache_df, requested_steps, requested_columns):
    import pandas as pd
    if cache_df.empty or '_step' not in cache_df.columns:
        return False
    for column in requested_columns:
        if column not in cache_df.columns:
            return False
    requested_df = pd.DataFrame({'_step': requested_steps.astype(float)})
    lookup_df = cache_df[['_step'] + requested_columns].copy()
    lookup_df['_step'] = lookup_df['_step'].astype(float)
    merged_df = requested_df.merge(lookup_df, on='_step', how='left')
    return all(merged_df[column].notna().all() for column in requested_columns)


def _merge_smoothed_run_cache(cache_df, new_output_df):
    import numpy as np
    merged_df = cache_df.merge(new_output_df, on='_step', how='outer', suffixes=('', '__new'))
    for column in SMOOTHED_RUN_COLUMNS[1:]:
        new_column = f'{column}__new'
        if new_column in merged_df.columns:
            merged_df[column] = merged_df[new_column].where(merged_df[new_column].notna(), merged_df[column])
            merged_df = merged_df.drop(columns=[new_column])
    for column in SMOOTHED_RUN_COLUMNS:
        if column not in merged_df.columns:
            merged_df[column] = np.nan
    return merged_df[SMOOTHED_RUN_COLUMNS].sort_values('_step').reset_index(drop=True)


def _compute_regular_output(model, y, times, interval, regular_output):
    import numpy as np
    import pandas as pd
    requested_steps = _regular_grid(times, interval)
    regular_filter_full_df, regular_filter_df, total_loglik = model.filter_timeline(
        y,
        observation_times=times,
        output_times=requested_steps,
    )
    output_df = pd.DataFrame({'_step': regular_filter_df['time'].to_numpy(dtype=float)})
    if regular_output in {'filtered', 'both'}:
        filtered_mean_var = regular_filter_df['filtered_var'].to_numpy(dtype=float)
        output_df['filtered_mean'] = regular_filter_df['filtered_mean'].to_numpy(dtype=float)
        output_df['filtered_mean_var'] = filtered_mean_var
        output_df['filtered_obs_var'] = filtered_mean_var + float(model.params.r)
    if regular_output in {'smoothed', 'both'}:
        _, regular_smoothed_df = model.smooth_timeline(regular_filter_full_df, output_times=requested_steps)
        smoothed_mean_var = regular_smoothed_df['smoothed_mean_var'].to_numpy(dtype=float)
        output_df['smoothed_mean'] = regular_smoothed_df['smoothed_mean'].to_numpy(dtype=float)
        output_df['smoothed_mean_var'] = smoothed_mean_var
        output_df['smoothed_obs_var'] = regular_smoothed_df['smoothed_obs_var'].to_numpy(dtype=float)
    for column in SMOOTHED_RUN_COLUMNS:
        if column not in output_df.columns:
            output_df[column] = np.nan
    return output_df[SMOOTHED_RUN_COLUMNS], requested_steps, total_loglik


def _upsert_model_cache_row(
    run_id: str,
    sweep_id: int,
    run_dir: Path,
    env_name: str,
    metric_name: str,
    num_observations: int,
    total_loglik,
    fitted_params: Kalman1DParams,
):
    import pandas as pd
    model_cache_df = load_model_cache()
    fitted_row = pd.DataFrame([
        {
            'run_id': run_id,
            'sweep_id': sweep_id,
            'run_dir': str(run_dir),
            'env_name': env_name,
            'metric_name': metric_name,
            'num_observations': int(num_observations),
            'total_loglik': None if total_loglik is None else float(total_loglik),
            'a': float(fitted_params.a),
            'q': float(fitted_params.q),
            'r': float(fitted_params.r),
            'm0': float(fitted_params.m0),
            'p0': float(fitted_params.p0),
            'b': float(fitted_params.b),
        }
    ])
    model_cache_df = pd.concat([model_cache_df.loc[model_cache_df['run_id'] != run_id], fitted_row], ignore_index=True)
    save_model_cache(model_cache_df)


def _coerce(value, cast):
    try:
        return cast(value) if value not in (None, '', 'nan') else None
    except (ValueError, TypeError):
        return None


def main():
    args = parse_args()
    if args.interval <= 0:
        raise ValueError('interval must be positive')

    cached_row = _read_cached_row(args.run_id)

    if args.regular_output == 'none' and cached_row is not None:
        fitted_params = _model_params_from_row(cached_row)
        result = {
            'run_id': args.run_id,
            'sweep_id': _coerce(cached_row.get('sweep_id'), int) or args.sweep_id,
            'run_dir': cached_row.get('run_dir') or None,
            'env_name': cached_row.get('env_name') or None,
            'metric_name': cached_row.get('metric_name') or None,
            'parameter_source': 'cache',
            'status_message': 'loaded cached model',
            'model_cache_path': str(MODEL_CACHE_PATH),
            'total_loglik': _coerce(cached_row.get('total_loglik'), float),
            'num_observations': _coerce(cached_row.get('num_observations'), int),
            'regular_output': 'none',
            'interval': None,
            'smoothed_run_cache_path': None,
            'smoothed_run_cache_source': None,
            'smoothed_run_output_columns': [],
            'params': {key: float(value) for key, value in asdict(fitted_params).items()},
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if cached_row is not None and cached_row.get('run_dir'):
        run_dir = Path(cached_row['run_dir'])
    else:
        run_dir = resolve_run_dir(args.run_id, args.sweep_id, args.wandb_root)

    env_name = cached_row.get('env_name') if cached_row is not None else None
    if not env_name:
        env_name = infer_env_name(run_dir)

    metric_name = cached_row.get('metric_name') if cached_row is not None else None
    if not metric_name:
        metric_name = f'episode_return/{env_name}'

    episode_return_df = load_episode_return_df(args.run_id, run_dir, metric_name)
    import numpy as np

    times = episode_return_df['_step'].to_numpy(dtype=float)
    y = episode_return_df[metric_name].to_numpy(dtype=float)

    if cached_row is not None:
        fitted_params = _model_params_from_row(cached_row)
        parameter_source = 'cache'
        status_message = 'loaded cached model'
        cached_total_loglik = _coerce(cached_row.get('total_loglik'), float)
    else:
        fitted_params, parameter_source, status_message = fit_or_load_params(args.run_id, y, times)
        cached_total_loglik = None

    model = Kalman1D(fitted_params)
    total_loglik = cached_total_loglik
    smoothed_run_cache_path = None
    smoothed_run_cache_source = None
    smoothed_run_output_columns = []

    if args.regular_output != 'none':
        smoothed_run_cache_path = _smoothed_run_cache_path(args.run_id)
        smoothed_run_output_columns = REGULAR_OUTPUT_COLUMN_MAP[args.regular_output]
        requested_steps = _regular_grid(times, args.interval)
        smoothed_run_cache_df = _load_smoothed_run_cache(args.run_id)
        if _has_regular_output(smoothed_run_cache_df, requested_steps, smoothed_run_output_columns):
            smoothed_run_cache_source = 'cache'
        else:
            computed_output_df, requested_steps, total_loglik = _compute_regular_output(
                model,
                y,
                times,
                args.interval,
                args.regular_output,
            )
            smoothed_run_cache_df = _merge_smoothed_run_cache(smoothed_run_cache_df, computed_output_df)
            _save_smoothed_run_cache(args.run_id, smoothed_run_cache_df)
            smoothed_run_cache_source = 'computed'

    if total_loglik is None:
        _, total_loglik = model.filter(y, times=times)

    _upsert_model_cache_row(
        run_id=args.run_id,
        sweep_id=args.sweep_id,
        run_dir=run_dir,
        env_name=env_name,
        metric_name=metric_name,
        num_observations=len(y),
        total_loglik=total_loglik,
        fitted_params=fitted_params,
    )

    result = {
        'run_id': args.run_id,
        'sweep_id': args.sweep_id,
        'run_dir': str(run_dir),
        'env_name': env_name,
        'metric_name': metric_name,
        'parameter_source': parameter_source,
        'status_message': status_message,
        'model_cache_path': str(MODEL_CACHE_PATH),
        'total_loglik': float(total_loglik),
        'num_observations': int(len(y)),
        'regular_output': args.regular_output,
        'interval': None if args.regular_output == 'none' else float(args.interval),
        'smoothed_run_cache_path': None if smoothed_run_cache_path is None else str(smoothed_run_cache_path),
        'smoothed_run_cache_source': smoothed_run_cache_source,
        'smoothed_run_output_columns': smoothed_run_output_columns,
        'smoothed_run_num_rows': None if args.regular_output == 'none' else int(len(_regular_grid(times, args.interval))),
        'params': {key: float(value) for key, value in asdict(fitted_params).items()},
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
