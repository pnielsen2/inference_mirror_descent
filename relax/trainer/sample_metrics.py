"""Host-side per-step bookkeeping + periodic wandb metrics for the vmapped
multi-run trainer.

Encapsulates everything that's "what wandb sees", not "what the algorithm
does":

* per-run buffer-add scalars (action_mean/var/clip_frac, q_agg, v_value,
  q_var) computed from the rollout outputs, buffer.add_batch, SampleLog.add;
* per-run episode-return drain (wandb scalar + local CSV mirror that
  survives wandb rate-limit drops);
* periodic ``Global_EMAs/*`` flush of η / advantage moments / one-step
  dist-shift estimates;
* the per-step Q-ensemble variance diagnostic (vmapped+jitted across
  runs, lazily compiled on first call).

The trainer's ``gather_transitions()`` collapses to: rollout → ema update → env.step →
``recorder.record(...)``.
"""
from pathlib import Path
from typing import List

import jax
import jax.numpy as jnp
import numpy as np

from relax.trainer.accumulator import Interval, SampleLog
from relax.trainer.wandb_logging import WandbMultiSeedLogger
from relax.utils.experience import Experience


def _q_ensemble_var_vmap(algorithm, obs_nm: np.ndarray, action_nm: np.ndarray,
                         jit_cache: dict) -> np.ndarray:
    """Per-run mean Var(Q_i) across the Q ensemble for [num_runs, envs_per_run] (obs, action).

    Returns an ``np.ndarray`` of shape ``[num_runs]``. Vmapped+jitted so a single
    XLA call replaces the host-side per-run Python loop. Only valid for
    ensembles of size >= 2 (returns zeros otherwise). The compiled fn is
    cached on the recorder via ``jit_cache``.
    """
    q_params = algorithm.state.params.q  # tuple of N_q [N_runs, ...] pytrees
    if len(q_params) < 2:
        return np.zeros((obs_nm.shape[0],), dtype=np.float32)
    if jit_cache.get("fn") is None:
        q_fn = algorithm.model.q

        def _per_run(q_params_run, s, a):
            means = [q_fn(qp, s, a) for qp in q_params_run]
            stacked = jnp.stack(means, axis=0)
            return jnp.mean(jnp.var(stacked, axis=0))

        jit_cache["fn"] = jax.jit(jax.vmap(_per_run))
    return np.asarray(
        jit_cache["fn"](q_params, jnp.asarray(obs_nm), jnp.asarray(action_nm))
    )


class SampleMetricsRecorder:
    def __init__(
        self,
        *,
        algorithm,
        buffers: list,
        sample_logs: List[SampleLog],
        logger: WandbMultiSeedLogger,
        env_name: str,
        log_path: Path,
        sample_log_interval: Interval,
    ):
        self.algorithm = algorithm
        self.buffers = buffers
        self.sample_logs = sample_logs
        self.logger = logger
        self.env_name = env_name
        self.log_path = log_path
        self.sample_log_interval = sample_log_interval
        self.num_runs = len(buffers)
        self._q_var_jit_cache: dict = {}
        self._local_return_path: Path | None = None

    # ------------------------------------------------------------------
    # One-time setup: open the local episode-return mirror CSV.
    # ------------------------------------------------------------------
    def init(self):
        self.log_path.mkdir(parents=True, exist_ok=True)
        self._local_return_path = self.log_path / "episode_returns.csv"
        if not self._local_return_path.exists():
            with open(self._local_return_path, "w") as f:
                f.write(f"seed,step,episode_return/{self.env_name}\n")

    # ------------------------------------------------------------------
    # Per-step record: buffer-add scalars, buffer.add_batch, SampleLog.add,
    # episode-return drain, and the periodic Global_EMAs/* flush.
    # ------------------------------------------------------------------
    def record(
        self,
        *,
        obs_nm: np.ndarray,
        action_nm: np.ndarray,
        q_per_env: np.ndarray,
        v_per_env: np.ndarray | None,
        rew_nm: np.ndarray,
        term_nm: np.ndarray,
        trunc_nm: np.ndarray,
        nxt_nm: np.ndarray,
    ):
        action_np = np.asarray(action_nm)
        action_abs = np.abs(action_np)
        action_mean = np.mean(action_np, axis=-1)                                    # [num_runs, envs_per_run]
        action_var = np.var(action_np, axis=-1)                                      # [num_runs, envs_per_run]
        action_clip_frac = np.mean((action_abs > 0.99).astype(np.float32), axis=-1)  # [num_runs, envs_per_run]
        q_mean_per_run = np.mean(q_per_env, axis=1)                                  # [num_runs]
        v_mean_per_run = np.mean(v_per_env, axis=1) if v_per_env is not None else None

        q_var_per_run = None
        if not getattr(self.algorithm, "on_policy_ema", False):
            q_var_per_run = _q_ensemble_var_vmap(
                self.algorithm, obs_nm, action_np, self._q_var_jit_cache,
            )

        for s in range(self.num_runs):
            run_info = {
                "action_mean": action_mean[s],
                "action_var": action_var[s],
                "action_clip_frac": action_clip_frac[s],
                "q_agg": float(q_mean_per_run[s]),
            }
            if v_mean_per_run is not None:
                run_info["v_value"] = float(v_mean_per_run[s])
            if q_var_per_run is not None:
                run_info["q_var"] = float(q_var_per_run[s])
            exp_s = Experience.create(
                obs_nm[s], action_np[s], rew_nm[s], term_nm[s], trunc_nm[s], nxt_nm[s], run_info,
            )
            self.buffers[s].add_batch(exp_s)
            self.sample_logs[s].add(rew_nm[s], term_nm[s], trunc_nm[s], run_info)

        # Drain pending episode returns to wandb (per run) and mirror to a
        # local CSV so the return curve survives wandb rate-limit drops.
        ep_key = f"episode_return/{self.env_name}"
        with open(self._local_return_path, "a", buffering=1) as f_local:
            for s in range(self.num_runs):
                for env_step, ret in self.sample_logs[s].take_pending_episode_returns():
                    self.logger.add_scalar_per_run(s, ep_key, ret, step=env_step)
                    f_local.write(f"{s},{int(env_step)},{float(ret)}\n")

        # Periodic sample-interval flush.
        # All runs advance sample_step by envs_per_run in lockstep; check run 0.
        if self.sample_log_interval.check(self.sample_logs[0].sample_step):
            self.flush_periodic()

    # ------------------------------------------------------------------
    # Periodic Global_EMAs/* metrics + sample-log accumulator drain.
    # ------------------------------------------------------------------
    def flush_periodic(self):
        alg = self.algorithm
        state = alg.state
        log = self.logger.add_scalar_per_run
        for s in range(self.num_runs):
            sstep = int(self.sample_logs[s].sample_step)
            self.sample_logs[s].log_accumulator(
                lambda k, v, _step, _s=s, _sstep=sstep: log(_s, k, v, step=_sstep)
            )
            tfg_eta = float(np.asarray(state.tfg_eta)[s])
            log(s, "Global_EMAs/tfg_eta", tfg_eta, step=sstep)
            if getattr(alg, "on_policy_ema", False):
                m2 = float(np.asarray(state.advantage_second_moment_ema)[s])
                kl_budget = float(np.asarray(state.hp.kl_budget_val)[s])
                m3 = float(np.asarray(state.advantage_third_moment_ema)[s])
                cov = float(np.asarray(state.dist_shift_covariance_ema)[s])
                shape = float(np.asarray(state.dist_shift_shape_ema)[s])
                eta = tfg_eta / float(np.sqrt(max(m2, 1e-8)))
                eta_kl = float(np.sqrt(2.0 * kl_budget / max(m2, 1e-8)))
                log(s, "Global_EMAs/Advantage_second_moment", m2, step=sstep)
                log(s, "Global_EMAs/Advantage_third_moment", m3, step=sstep)
                log(s, "Global_EMAs/Distribution_shift_covariance", cov, step=sstep)
                log(s, "Global_EMAs/Distribution_shift_shape", shape, step=sstep)
                log(s, "Global_EMAs/eta", eta, step=sstep)
                log(s, "Global_EMAs/eta_kl_ceiling", eta_kl, step=sstep)
                log(s, "Global_EMAs/eta_kl_budget", eta_kl, step=sstep)
                if bool(getattr(alg, "one_step_dist_shift_eta", False)):
                    eta_one_step = -1.0 / (float(np.sqrt(max(m2, 1e-8))) * shape) if shape < -1e-8 else float("nan")
                    if not np.isnan(eta_one_step):
                        log(s, "Global_EMAs/eta_quadratic", eta_one_step, step=sstep)
                        log(s, "Global_EMAs/eta_one_step_dist_shift", eta_one_step, step=sstep)
            else:
                log(s, "Global_EMAs/eta", tfg_eta, step=sstep)
        self.logger.flush_all()
