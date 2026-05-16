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
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import Env
from tqdm import tqdm

from relax.algorithm import Algorithm
from relax.buffer import TreeBuffer
from relax.env.vector import VectorEnv
from relax.trainer.accumulator import Interval, SampleLog, UpdateLog
from relax.trainer.wandb_logging import WandbMultiSeedLogger, build_config_tag  # noqa: F401  (build_config_tag re-exported for analysis scripts)
from relax.utils.experience import Experience


def _detect_env_can_terminate(env_name: str) -> bool:
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

        self._check_supported_config()

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
        _gamma = np.asarray(getattr(self.algorithm.state, "gamma", getattr(self.algorithm, "gamma", 0.99)), dtype=np.float64)
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

    def _check_supported_config(self):
        alg = self.algorithm
        if getattr(alg, "supervised_steps", 1) > 1:
            raise NotImplementedError("supervised_steps > 1 is not supported under vmap.")

    def _per_seed_keys(self, key: jax.Array) -> jax.Array:
        # train_mujoco.py always hands in a batched key (shape [N] of typed
        # keys or [N, 2] of raw uint32 pairs). The single-master-key fallback
        # has been removed.
        return key

    def _iter_keys(self, key: jax.Array, step: int) -> Tuple[jax.Array, jax.Array]:
        seed_keys = jax.vmap(lambda k: jax.random.fold_in(k, step))(key)
        split_keys = jax.vmap(lambda k: jax.random.split(k, 2))(seed_keys)
        return split_keys[:, 0], split_keys[:, 1]

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

        # Local backup: episode-return curves per seed, so the metric survives
        # even when wandb's per-project filestream rate limit silently drops
        # logs. Only episode_return is stored locally (other metrics would be
        # too large on disk). Metric key mirrors the wandb key:
        # f"episode_return/{env_name}".
        self.log_path.mkdir(parents=True, exist_ok=True)
        self._local_return_path = self.log_path / "episode_returns.csv"
        if not self._local_return_path.exists():
            with open(self._local_return_path, "w") as f:
                f.write(f"seed,step,episode_return/{self.env_name}\n")

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

        # One PRNG key per seed.
        keys = self._per_seed_keys(sample_key)

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

        # Per-seed buffer add + SampleLog update.
        action_np = np.asarray(action_nm)
        action_abs = np.abs(action_np)
        action_mean = np.mean(action_np, axis=-1)                  # [N, M]
        action_var = np.var(action_np, axis=-1)                    # [N, M]
        action_clip_frac = np.mean((action_abs > 0.99).astype(np.float32), axis=-1)  # [N, M]
        q_mean_per_seed = np.mean(q_per_env, axis=1)                # [N]
        v_mean_per_seed = np.mean(v_per_env, axis=1) if v_per_env is not None else None

        q_var_per_seed = None
        if not getattr(self.algorithm, "on_policy_ema", False):
            # Single vmapped+jitted call across all seeds; shape [N].
            q_var_per_seed = self.algorithm.compute_q_ensemble_var_vmap(obs_nm, action_np)

        for s in range(self.N):
            seed_info = {
                "action_mean": action_mean[s],
                "action_var": action_var[s],
                "action_clip_frac": action_clip_frac[s],
                "q_agg": float(q_mean_per_seed[s]),
            }
            if v_mean_per_seed is not None:
                seed_info["v_value"] = float(v_mean_per_seed[s])
            if q_var_per_seed is not None:
                seed_info["q_var"] = float(q_var_per_seed[s])
            exp_s = Experience.create(
                obs_nm[s], action_np[s], rew_nm[s], term_nm[s], trunc_nm[s], nxt_nm[s], seed_info,
            )
            self.buffers[s].add_batch(exp_s)
            self.sample_logs[s].add(rew_nm[s], term_nm[s], trunc_nm[s], seed_info)

        # Drain pending episode returns to wandb (per seed) and mirror to a
        # local CSV so the return curve survives wandb rate-limit drops.
        ep_key = f"episode_return/{self.env_name}"
        with open(self._local_return_path, "a", buffering=1) as f_local:
            for s in range(self.N):
                for env_step, ret in self.sample_logs[s].take_pending_episode_returns():
                    self.logger.add_scalar_per_seed(s, ep_key, ret, step=env_step)
                    f_local.write(f"{s},{int(env_step)},{float(ret)}\n")

        # Periodic sample-interval flush.
        # All seeds advance sample_step by M in lockstep; check seed 0.
        sl0 = self.sample_logs[0]
        if self.sample_log_interval.check(sl0.sample_step):
            self._log_periodic_sample_metrics()

        if np.any(term_flat) or np.any(trunc_flat):
            obs_flat, _ = self.env.reset()
        else:
            obs_flat = next_obs_flat

        return obs_flat

    # ------------------------------------------------------------------
    # On-policy EMA + adaptive-η update (host-side, seed-axis-vectorized).
    # Two single-purpose paths, dispatched once at construction time:
    #
    #   * _ema_update_kl_only       -- KL-budget-only η selection.
    #   * _ema_update_one_step      -- KL ceiling AND one-step distribution-
    #                                  shift η* (requires --one_step_dist_shift_eta).
    #
    # Every reduction that was over [M] per-env is over axis=1 of [N, M]
    # arrays, yielding [N]-shaped per-seed quantities. State EMA fields are
    # [N] float32 arrays. The numerical sequence (float64 promotion, the
    # 1e-8 clamp, the 0.574 acceptance target, etc.) is preserved verbatim
    # from the legacy combined implementation.
    # ------------------------------------------------------------------
    def _broadcast_state_scalar(self, value) -> np.ndarray:
        """Promote a state scalar/[N] field to a [N] float64 ndarray."""
        arr = np.asarray(value).astype(np.float64)
        if arr.ndim == 0:
            arr = np.broadcast_to(arr, (self.N,)).astype(np.float64)
        return arr

    def _ema_update_kl_only(self, q_per_env: np.ndarray, v_per_env: np.ndarray) -> np.ndarray:
        """KL-budget-only path: update E[A^2] EMA and set η = sqrt(2δ/E[A^2])."""
        alg = self.algorithm
        state = alg.state
        adv_per_env = q_per_env - v_per_env                # [N, M]
        m2_batch = np.mean(adv_per_env ** 2, axis=1)        # [N]

        tau_v = self._broadcast_state_scalar(state.adv_ema_tau)
        cur_m2 = np.asarray(state.advantage_second_moment_ema)
        new_m2 = (1 - tau_v) * cur_m2 + tau_v * m2_batch

        kl_budget = self._broadcast_state_scalar(state.kl_budget_val)
        m2_safe = np.maximum(new_m2, 1e-8)
        sqrt_v = np.sqrt(m2_safe)
        eta_kl_raw = np.sqrt(2.0 * kl_budget / m2_safe)
        new_eta = eta_kl_raw * sqrt_v

        alg.state = state._replace(
            advantage_second_moment_ema=jnp.asarray(new_m2.astype(np.float32)),
            tfg_eta=jnp.asarray(new_eta.astype(np.float32)),
        )
        return adv_per_env

    def _ema_update_one_step(self, q_per_env: np.ndarray, v_per_env: np.ndarray) -> np.ndarray:
        """One-step distribution-shift path: update {E[A^2], E[A^3], cov, shape}
        EMAs and pick η = min(η_KL, η*) where η* comes from the second-order
        expansion using the one-step covariance estimate."""
        alg = self.algorithm
        state = alg.state
        adv_per_env = q_per_env - v_per_env                # [N, M]
        m2_batch = np.mean(adv_per_env ** 2, axis=1)        # [N]
        m3_batch = np.mean(adv_per_env ** 3, axis=1)        # [N]

        tau_v = self._broadcast_state_scalar(state.adv_ema_tau)
        cur_m2 = np.asarray(state.advantage_second_moment_ema)
        cur_m3 = np.asarray(state.advantage_third_moment_ema)
        cur_c = np.asarray(state.dist_shift_covariance_ema)
        cur_shape = np.asarray(state.dist_shift_shape_ema)

        new_m2 = (1 - tau_v) * cur_m2 + tau_v * m2_batch
        new_m3 = (1 - tau_v) * cur_m3 + tau_v * m3_batch
        new_c = cur_c
        new_shape = cur_shape

        # One-step covariance c_batch is only valid for env-steps where the
        # previous step did not terminate; seeds with no valid samples this
        # batch keep the prior EMA values unchanged.
        if self._prev_adv_per_env is not None and self._prev_valid is not None:
            valid = self._prev_valid                           # [N, M] bool
            valid_count = np.sum(valid, axis=1)                # [N]
            prod = valid.astype(np.float64) * (adv_per_env ** 2) * self._prev_adv_per_env
            sums = np.sum(prod, axis=1)
            c_batch = np.where(valid_count > 0, sums / np.maximum(valid_count, 1), 0.0)
            c_batch_valid = valid_count > 0

            new_c_candidate = (1 - tau_v) * cur_c + tau_v * c_batch
            new_c = np.where(c_batch_valid, new_c_candidate, cur_c)

            gamma = self._broadcast_state_scalar(state.gamma)
            tau_s = self._broadcast_state_scalar(state.shape_ema_tau)
            v_raw_safe = np.maximum(m2_batch, 1e-8)
            b_batch = 2.0 * gamma * c_batch + m3_batch
            s_batch = b_batch / v_raw_safe ** 1.5
            new_shape_candidate = (1 - tau_s) * cur_shape + tau_s * s_batch
            new_shape = np.where(c_batch_valid, new_shape_candidate, cur_shape)

        # η = min(η_KL, η*); η* is +inf when shape is non-negative.
        kl_budget = self._broadcast_state_scalar(state.kl_budget_val)
        m2_safe = np.maximum(new_m2, 1e-8)
        sqrt_v = np.sqrt(m2_safe)
        eta_kl_raw = np.sqrt(2.0 * kl_budget / m2_safe)
        eta_star_raw = np.where(new_shape < -1e-8, -1.0 / (sqrt_v * new_shape), np.inf)
        eta_raw = np.minimum(eta_star_raw, eta_kl_raw)
        new_eta = eta_raw * sqrt_v

        alg.state = state._replace(
            advantage_second_moment_ema=jnp.asarray(new_m2.astype(np.float32)),
            advantage_third_moment_ema=jnp.asarray(new_m3.astype(np.float32)),
            dist_shift_covariance_ema=jnp.asarray(new_c.astype(np.float32)),
            dist_shift_shape_ema=jnp.asarray(new_shape.astype(np.float32)),
            tfg_eta=jnp.asarray(new_eta.astype(np.float32)),
        )
        return adv_per_env

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------
    def update(self, update_key: jax.Array):
        keys = self._per_seed_keys(update_key)
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
    # Periodic sample-interval metrics
    # ------------------------------------------------------------------
    def _log_periodic_sample_metrics(self):
        alg = self.algorithm
        state = alg.state
        log = self.logger.add_scalar_per_seed
        for s in range(self.N):
            sstep = int(self.sample_logs[s].sample_step)
            self.sample_logs[s].log_accumulator(
                lambda k, v, _step, _s=s, _sstep=sstep: log(_s, k, v, step=_sstep)
            )
            tfg_eta = float(np.asarray(state.tfg_eta)[s])
            log(s, "Global_EMAs/tfg_eta", tfg_eta, step=sstep)
            if getattr(alg, "on_policy_ema", False):
                m2 = float(np.asarray(state.advantage_second_moment_ema)[s])
                kl_budget = float(np.asarray(state.kl_budget_val)[s])
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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, key: jax.Array):
        try:
            # key always has leading [N] seed axis (see _per_seed_keys).
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
            sample_key, update_key = self._iter_keys(key, self.sample_logs[0].sample_step)
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
