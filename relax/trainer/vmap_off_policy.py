"""Vmap-parallel multi-seed trainer (Step 1 of the vmap refactor).

Trains N independent RL seeds in parallel on a single device via ``jax.vmap``
over the algorithm's ``stateless_update`` and ``stateless_get_action``.
Currently supports the DPMD packed multi-seed path for both the
KL-budget/on-policy-EMA setting and fixed-tfg_eta mode. Unsupported features
still raise at construction time so failures are loud, not silent.

Layout (Option B):
  * One ``env`` VectorEnv of total size ``N * M`` (N seeds × M per-seed envs).
  * Inbound obs reshape ``[N*M, obs_dim] -> [N, M, obs_dim]``.
  * ``algorithm.state`` has a leading [N] seed axis on every leaf.
  * ``buffers`` is a list of N independent TreeBuffers.
  * Each seed has its own wandb run and its own SampleLog.
"""
from pathlib import Path
from typing import List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import Env
from tqdm import tqdm

from relax.algorithm import Algorithm
from relax.algorithm import ema_eta
from relax.buffer import TreeBuffer
from relax.env.vector import VectorEnv
from relax.trainer.accumulator import Interval, SampleLog, UpdateLog
from relax.trainer.sample_metrics import SampleMetricsRecorder
from relax.trainer.wandb_logging import WandbMultiSeedLogger, build_config_tag  # noqa: F401  (build_config_tag re-exported for analysis scripts)
from relax.utils.experience import Experience


def _detect_env_can_terminate(env_name: str) -> bool:
    """Probe a gym(nasium) env id to decide if it ever sets ``terminated=True``.

    Used by ``VmapOffPolicyTrainer`` to skip episode-length / reward-mean
    metrics for envs that only ever time out (e.g. v5 mujoco envs with
    ``_terminate_when_unhealthy=False``). Falls back to ``True`` (treat as
    terminating) when the probe fails for any reason.
    """
    try:
        import gymnasium
        probe = gymnasium.make(env_name)
        inner = probe.unwrapped
        can = getattr(inner, "_terminate_when_unhealthy", None)
        probe.close()
        if can is not None:
            return bool(can)
        import inspect
        src = inspect.getsource(inner.step)
        if "return observation, reward, False, False" in src:
            return False
        return True
    except Exception:
        return True


class VmapOffPolicyTrainer:
    def __init__(
        self,
        env: Env,
        algorithm: Algorithm,
        buffers: List[TreeBuffer],
        log_path: Path,
        *,
        parallel_seeds: int,
        per_seed_envs: int,
        batch_size: int = 256,
        start_step: int = 1000,
        total_step: int = int(1e6),
        update_per_iteration: int = 1,
        sample_log_n_env_step: int = 1000,
        update_log_n_env_steps: int = 5000,
        hparams: Optional[dict] = None,
        wandb_names: Optional[List[str]] = None,
        hp_pack_dict: Optional[dict] = None,
        sweep_id: Optional[int] = None,
        config_tag_keys: Optional[str] = None,
    ):
        self.env = env
        self.algorithm = algorithm
        self.buffers = buffers
        self.log_path = log_path
        self.N = int(parallel_seeds)
        self.M = int(per_seed_envs)
        self.batch_size = int(batch_size)
        self.start_step = int(start_step)
        self.total_step = int(total_step)
        self.update_per_iteration = int(update_per_iteration)
        self.sample_log_n_env_step = int(sample_log_n_env_step)
        self.update_log_n_env_steps = int(update_log_n_env_steps)
        self.hparams = hparams or {}
        self._wandb_names = wandb_names
        self.sweep_id = sweep_id
        # Parse the comma-separated list of tag keys (set by launch.py from
        # the union of hard+easy ablation axes, minus env/seed). None means
        # the config_tag falls back to "single" -- only meaningful when
        # bypassing launch.py (e.g. ad-hoc single-config runs).
        self.config_tag_keys = None
        if config_tag_keys:
            self.config_tag_keys = [
                k.strip() for k in config_tag_keys.split(",") if k.strip()
            ]
        # Pre-parsed hp_pack (from --hp_pack_inline) so the wandb logger can
        # overwrite the shared-hparams scalars with each vmap slot's per-slot
        # value. Pack keys follow argparse attribute names (see
        # scripts/launch.py FLAG_TO_HP_KEY); train_mujoco.py has already
        # translated those to Diffv2TrainState field names for its own
        # override step.
        self._hp_pack = hp_pack_dict

        if len(buffers) != self.N:
            raise ValueError(f"Expected {self.N} buffers, got {len(buffers)}")
        if not isinstance(env.unwrapped, VectorEnv):
            raise ValueError("VmapOffPolicyTrainer requires a VectorEnv.")
        total = env.unwrapped.num_envs
        if total != self.N * self.M:
            raise ValueError(
                f"env.num_envs={total} but expected N*M = {self.N}*{self.M} = {self.N * self.M}"
            )

        self.env_name = env.spec.id if env.spec is not None else "env"
        _hp = getattr(self.algorithm.state, "hp", None)
        _gamma_src = getattr(_hp, "gamma", None) if _hp is not None else getattr(self.algorithm, "gamma", 0.99)
        _gamma = np.asarray(_gamma_src, dtype=np.float64)
        if _gamma.ndim == 0:
            _gamma = np.broadcast_to(_gamma, (self.N,)).astype(np.float64)
        _q_label = "Q"

        _can_terminate = _detect_env_can_terminate(self.env_name)
        self.sample_logs = [
            SampleLog(
                num_envs=self.M,
                env_name=self.env_name,
                gamma=float(_gamma[s]),
                q_label=_q_label,
                env_can_terminate=_can_terminate,
            )
            for s in range(self.N)
        ]
        self.update_log = UpdateLog()
        self.sample_log_interval = Interval(self.sample_log_n_env_step)
        self._last_update_log_env_step = 0

        self.logger = WandbMultiSeedLogger(
            N=self.N,
            env_name=self.env_name,
            log_path=self.log_path,
            wandb_names=self._wandb_names,
            hp_pack=self._hp_pack,
            sweep_id=self.sweep_id,
            hparams=self.hparams,
            config_tag_keys=self.config_tag_keys,
        )
        self.recorder = SampleMetricsRecorder(
            algorithm=self.algorithm,
            buffers=self.buffers,
            sample_logs=self.sample_logs,
            logger=self.logger,
            env_name=self.env_name,
            log_path=self.log_path,
            sample_log_interval=self.sample_log_interval,
        )

        # Per-seed host-side buffers for dist-shift covariance (shape [N, M]).
        self._prev_adv_per_env: Optional[np.ndarray] = None
        self._prev_valid: Optional[np.ndarray] = None

        # Pick the on-policy EMA update path once at construction time. Off
        # by default; the rollout block in sample() invokes this only when
        # the algorithm advertises on_policy_ema=True (i.e. --kl_budget set).
        if bool(getattr(self.algorithm, "one_step_dist_shift_eta", False)):
            self._on_policy_ema_update = self._ema_update_one_step
        else:
            self._on_policy_ema_update = self._ema_update_kl_only

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self, dummy_experience: Experience):
        # Trigger JIT tracing with a vmap-shaped dummy batch.
        def add_seed_axis(x):
            return np.broadcast_to(np.asarray(x), (self.N,) + np.shape(x)).copy()
        stacked = jax.tree.map(add_seed_axis, dummy_experience)
        self.algorithm.warmup_vmap(stacked, self.N)
        self.progress = tqdm(total=self.total_step, desc="Sample Step (per seed)", disable=None, dynamic_ncols=True)

        self.recorder.init()
        self.logger.set_snr(getattr(self.algorithm, "_snr", None))
        self.logger.init_runs()

    # ------------------------------------------------------------------
    # Warmup (random actions)
    # ------------------------------------------------------------------
    def warmup(self, key: jax.Array):
        train_obs_flat, _ = self.env.reset()
        # obs_flat: [N*M, obs_dim]
        # Each buffer fills to start_step transitions independently.
        # Since we step all N*M envs in sync, per-buffer per-step add is M.
        while any(len(b) < self.start_step for b in self.buffers):
            action_env = self.env.action_space.sample()  # [N*M, act_dim]
            next_obs_flat, reward_flat, term_flat, trunc_flat, info = self.env.step(action_env)

            obs_nm = np.asarray(train_obs_flat).reshape(self.N, self.M, -1)
            action_nm = np.asarray(action_env).reshape(self.N, self.M, -1)
            nxt_nm = np.asarray(next_obs_flat).reshape(self.N, self.M, -1)
            rew_nm = np.asarray(reward_flat).reshape(self.N, self.M)
            term_nm = np.asarray(term_flat).reshape(self.N, self.M)
            trunc_nm = np.asarray(trunc_flat).reshape(self.N, self.M)

            for s in range(self.N):
                exp_s = Experience.create(
                    obs_nm[s], action_nm[s], rew_nm[s], term_nm[s], trunc_nm[s], nxt_nm[s], {},
                )
                self.buffers[s].add_batch(exp_s)

            if np.any(term_flat) or np.any(trunc_flat):
                train_obs_flat, _ = self.env.reset()
            else:
                train_obs_flat = next_obs_flat
        return train_obs_flat

    # ------------------------------------------------------------------
    # Sample step
    # ------------------------------------------------------------------
    def sample(self, sample_key: jax.Array, obs_flat: np.ndarray) -> np.ndarray:
        # obs_flat: [N*M, obs_dim] (from prior env.step)
        obs_nm = np.asarray(obs_flat).reshape(self.N, self.M, -1)

        # One PRNG key per seed (sample_key already has leading [N] seed axis).
        keys = sample_key

        # Vmapped policy rollout. Returns (action [N,M,A], q [N,M], v [N,M] or None).
        action_nm, q_per_env, v_per_env = self.algorithm.get_action_vmap(keys, obs_nm)

        # Host-side seed-axis-vectorized EMA + eta update.
        # NOTE: _prev_* roll-forward happens AFTER env.step (below) so we know
        # which envs terminated/truncated this step.
        adv_per_env_now = None
        if getattr(self.algorithm, "on_policy_ema", False) and v_per_env is not None:
            adv_per_env_now = self._on_policy_ema_update(q_per_env, v_per_env)

        # Env step (flatten for Option B).
        action_flat = action_nm.reshape(self.N * self.M, -1)
        next_obs_flat, reward_flat, term_flat, trunc_flat, info = self.env.step(action_flat)

        # Reshape all outputs back to [N, M, ...].
        nxt_nm = np.asarray(next_obs_flat).reshape(self.N, self.M, -1)
        rew_nm = np.asarray(reward_flat).reshape(self.N, self.M)
        term_nm = np.asarray(term_flat).reshape(self.N, self.M)
        trunc_nm = np.asarray(trunc_flat).reshape(self.N, self.M)

        # Roll one-step covariance buffer using this-step done mask.
        if bool(getattr(self.algorithm, "one_step_dist_shift_eta", False)) and adv_per_env_now is not None:
            done_nm = term_nm | trunc_nm
            self._prev_adv_per_env = adv_per_env_now.copy()
            self._prev_valid = ~done_nm

        self.recorder.record(
            obs_nm=obs_nm, action_nm=action_nm,
            q_per_env=q_per_env, v_per_env=v_per_env,
            rew_nm=rew_nm, term_nm=term_nm, trunc_nm=trunc_nm, nxt_nm=nxt_nm,
        )

        if np.any(term_flat) or np.any(trunc_flat):
            obs_flat, _ = self.env.reset()
        else:
            obs_flat = next_obs_flat

        return obs_flat

    # ------------------------------------------------------------------
    # On-policy EMA + adaptive-η update (host-side, seed-axis-vectorized).
    # The two pure paths live in ``relax.algorithm.ema_eta``; we just
    # dispatch and replace the train state.
    # ------------------------------------------------------------------
    def _ema_update_kl_only(self, q_per_env: np.ndarray, v_per_env: np.ndarray) -> np.ndarray:
        new_state, adv = ema_eta.update_state_kl_only(
            self.algorithm.state, q_per_env, v_per_env, self.N,
        )
        self.algorithm.state = new_state
        return adv

    def _ema_update_one_step(self, q_per_env: np.ndarray, v_per_env: np.ndarray) -> np.ndarray:
        new_state, adv = ema_eta.update_state_one_step(
            self.algorithm.state, q_per_env, v_per_env,
            self._prev_adv_per_env, self._prev_valid, self.N,
        )
        self.algorithm.state = new_state
        return adv

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------
    def update(self, update_key: jax.Array):
        keys = update_key
        batches = [self.buffers[s].sample(self.batch_size) for s in range(self.N)]

        def stack_leaves(*xs):
            return jnp.stack([jnp.asarray(x) for x in xs], axis=0)

        stacked = jax.tree.map(stack_leaves, *batches)
        info, array_info = self.algorithm.update_vmap(keys, stacked)
        self.logger.accumulate_arrays(array_info)

        # info: dict tag -> np.ndarray[N]
        # Log per-seed; use per-seed update step = UpdateLog.update_step * 5
        # (existing convention from UpdateLog.log at line 291 of accumulator.py).
        self.update_log.update_step += 1
        current_env_step = self.sample_logs[0].sample_step
        log_this_step = (
            current_env_step - self._last_update_log_env_step >= self.update_log_n_env_steps
        )
        if log_this_step:
            self._last_update_log_env_step = current_env_step
            sample_steps = [int(self.sample_logs[s].sample_step) for s in range(self.N)]
            self.logger.flush_accumulated_arrays(sample_steps)
            for tag, vals in info.items():
                arr = np.asarray(vals)
                for s in range(self.N):
                    self.logger.add_scalar_per_seed(s, tag, float(arr[s]),
                                                   step=sample_steps[s])

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, key: jax.Array):
        try:
            # key always has leading [N] seed axis (set by train_mujoco.py).
            split_keys = jax.vmap(lambda k: jax.random.split(k, 2))(key)
            train_key = split_keys[:, 0]
            warmup_key = split_keys[:, 1]
            obs = self.warmup(warmup_key)
            self._train_standard(train_key, obs)
        except KeyboardInterrupt:
            pass
        finally:
            self.finish()

    def _train_standard(self, key: jax.Array, obs):
        while self.sample_logs[0].sample_step <= self.total_step:
            step = self.sample_logs[0].sample_step
            seed_keys = jax.vmap(lambda k: jax.random.fold_in(k, step))(key)
            split_keys = jax.vmap(lambda k: jax.random.split(k, 2))(seed_keys)
            sample_key, update_key = split_keys[:, 0], split_keys[:, 1]
            obs = self.sample(sample_key, obs)
            if self.update_per_iteration > 1:
                update_keys = jax.vmap(
                    lambda k: jax.random.split(k, self.update_per_iteration)
                )(update_key)
                for i in range(self.update_per_iteration):
                    self.update(update_keys[:, i])
            else:
                self.update(update_key)
            self.progress.n = self.sample_logs[0].sample_step
            self.progress.refresh()
            self.logger.flush_all()
        return obs

    def finish(self):
        self.logger.flush_all()
        try:
            self.env.close()
        except Exception:
            pass
        if hasattr(self, "progress"):
            self.progress.close()
        self.logger.finish()
