"""Vmap-parallel multi-seed trainer (Step 1 of the vmap refactor).

Trains N independent RL seeds in parallel on a single device via ``jax.vmap``
over the algorithm's ``stateless_update`` and ``stateless_get_action``.
Currently restricted to the DPMD + on_policy_ema ("kl_budget") config path —
this matches all the recent Humanoid MGMD experiments. Unsupported features
raise at construction time so failures are loud, not silent.

Layout (Option B):
  * One ``env`` VectorEnv of total size ``N * M`` (N seeds × M per-seed envs).
  * Inbound obs reshape ``[N*M, obs_dim] -> [N, M, obs_dim]``.
  * ``algorithm.state`` has a leading [N] seed axis on every leaf.
  * ``buffers`` is a list of N independent TreeBuffers.
  * Each seed has its own wandb run and its own SampleLog.
"""
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import wandb
from gymnasium import Env
from tqdm import tqdm

_WANDB_INIT_STAGGER_MIN_SECONDS = float(
    os.environ.get("WANDB_INIT_STAGGER_MIN_SECONDS", "0.0")
)
_WANDB_INIT_STAGGER_MAX_SECONDS = float(
    os.environ.get("WANDB_INIT_STAGGER_MAX_SECONDS", "3.0")
)

from relax.algorithm import Algorithm
from relax.buffer import TreeBuffer
from relax.env.vector import VectorEnv
from relax.trainer.accumulator import Interval, SampleLog, UpdateLog
from relax.utils.experience import Experience


def _format_tag_value(v) -> str:
    """Deterministic stringification of a tag value.

    Booleans: True / False (checked before int because bool is an int
    subclass in Python). Ints/floats: '%g'. Everything else: str()."""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        return f"{v:g}"
    return str(v)


def build_config_tag(hp_pack: Optional[dict], seed_index: int,
                     sweep_id: Optional[int],
                     hparams: Optional[dict] = None,
                     tag_keys: Optional[list] = None) -> str:
    """Deterministic wandb-pasteable identifier for (sweep, per-slot config).

    When ``tag_keys`` is given, the tag includes one key=value pair for every
    listed key -- drawing per-slot values from hp_pack when the key is a
    pack key, else the shared value from ``hparams`` (typically the CLI
    argparse attributes for this slurm job). This lets the tag capture both
    easy (vmappable, per-slot) and hard (per-job, shared across the vmap)
    ablation axes. Seed and env are conventionally kept out of the tag at
    the call site so the tag groups runs across envs and seed replicas.

    When ``tag_keys`` is None, falls back to legacy behavior: iterate every
    pack key except 'seed', sorted.
    """
    parts = []
    if tag_keys:
        for k in sorted(tag_keys):
            if hp_pack is not None and k in hp_pack:
                v = hp_pack[k][seed_index]
            elif hparams is not None and k in hparams:
                v = hparams[k]
            else:
                continue  # key not known in either source; skip silently
            parts.append(f"{k}={_format_tag_value(v)}")
    elif hp_pack is not None:
        for k in sorted(k for k in hp_pack if k != "seed"):
            parts.append(f"{k}={_format_tag_value(hp_pack[k][seed_index])}")
    body = "_".join(parts) or "single"
    return f"sweep{sweep_id}_{body}" if sweep_id is not None else body


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
        hp_pack_path: Optional[str] = None,
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
        # "fall back to legacy: iterate hp_pack keys except seed".
        self.config_tag_keys = None
        if config_tag_keys:
            self.config_tag_keys = [
                k.strip() for k in config_tag_keys.split(",") if k.strip()
            ]
        # Load the hp_pack once so _init_wandb_runs can overwrite the
        # shared-hparams scalars with each vmap slot's per-slot value. Pack
        # keys follow argparse attribute names (see scripts/launch.py
        # FLAG_TO_HP_KEY); train_mujoco.py has already translated those to
        # Diffv2TrainState field names for its own override step. Prefer the
        # dict form (already parsed from --hp_pack_inline or --hp_pack by the
        # caller) to avoid re-reading from disk; fall back to hp_pack_path.
        self._hp_pack = None
        if hp_pack_dict is not None:
            self._hp_pack = hp_pack_dict
        elif hp_pack_path is not None:
            import json
            with open(hp_pack_path) as f:
                self._hp_pack = json.load(f)

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
        _q_label = "Q_η" if getattr(self.algorithm, "use_entropic_q", False) else "Q"
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

        self._wandb_runs: List = []
        self._wandb_pending: List[dict] = [{} for _ in range(self.N)]
        self._wandb_pending_step: List[Optional[int]] = [None] * self.N
        self._array_accum = {}

        # Per-seed host-side buffers for dist-shift covariance (shape [N, M]).
        self._prev_adv_per_env: Optional[np.ndarray] = None
        self._prev_valid: Optional[np.ndarray] = None

    def _is_typed_key(self, key: jax.Array) -> bool:
        dtype = getattr(key, "dtype", None)
        if dtype is None:
            return False
        try:
            return jax.dtypes.issubdtype(dtype, jax.dtypes.prng_key)
        except Exception:
            return False

    def _check_supported_config(self):
        alg = self.algorithm
        if not getattr(alg, "on_policy_ema", False):
            raise NotImplementedError(
                "VmapOffPolicyTrainer requires on_policy_ema (pass --kl_budget)."
            )
        if getattr(alg, "two_step_dist_shift_eta", False):
            raise NotImplementedError("two_step_dist_shift_eta not yet supported under vmap.")
        if getattr(alg, "dist_shift_eta", False) and not getattr(alg, "one_step_dist_shift_eta", False):
            raise NotImplementedError(
                "Plain dist_shift_eta (D_ψ covariance) not yet supported under vmap. "
                "Use --one_step_dist_shift_eta."
            )
        if getattr(alg, "soft_pi_mode", False):
            raise NotImplementedError("soft_pi_mode is not supported under vmap.")
        if getattr(alg, "latent_action_space", False):
            raise NotImplementedError("latent_action_space is not supported under vmap.")
        if getattr(alg, "supervised_steps", 1) > 1:
            raise NotImplementedError("supervised_steps > 1 is not supported under vmap.")
        if getattr(alg, "tfg_eta_schedule", "constant") not in ("constant",):
            raise NotImplementedError("tfg_eta_schedule != 'constant' is not supported under vmap.")
        if getattr(alg, "equal_episode_weighting", False):
            raise NotImplementedError("equal_episode_weighting is not supported under vmap.")
        if getattr(alg, "entropic_critic_param", "none") != "none":
            raise NotImplementedError(
                "entropic_critic_param != 'none' is not supported under vmap "
                "(would need to thread the mutable tfg_eta closure through vmap)."
            )

    def _is_single_key(self, key: jax.Array) -> bool:
        shape = tuple(np.shape(key))
        if self._is_typed_key(key):
            return shape == ()
        return shape == (2,)

    def _is_batched_key(self, key: jax.Array) -> bool:
        shape = tuple(np.shape(key))
        if self._is_typed_key(key):
            return shape == (self.N,)
        return shape == (self.N, 2)

    def _per_seed_keys(self, key: jax.Array) -> jax.Array:
        if self._is_single_key(key):
            return jax.random.split(key, self.N)
        if self._is_batched_key(key):
            return key
        raise ValueError(f"Unexpected key shape {np.shape(key)} for parallel_seeds={self.N}")

    def _iter_keys(self, key: jax.Array, step: int) -> Tuple[jax.Array, jax.Array]:
        if self._is_single_key(key):
            ikey = jax.random.fold_in(key, step)
            return jax.random.split(ikey)
        if self._is_batched_key(key):
            seed_keys = jax.vmap(lambda k: jax.random.fold_in(k, step))(key)
            split_keys = jax.vmap(lambda k: jax.random.split(k, 2))(seed_keys)
            return split_keys[:, 0], split_keys[:, 1]
        raise ValueError(f"Unexpected key shape {np.shape(key)} for parallel_seeds={self.N}")

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

        self._init_wandb_runs()

    def _init_wandb_runs(self):
        self._wandb_runs = []
        base_name = self.log_path.name
        # Keep group = env name (its original semantics). sweep_id is logged as
        # a regular config field so filtering in wandb is config.sweep_id == N.
        group = self.env_name
        for s in range(self.N):
            if s > 0 and _WANDB_INIT_STAGGER_MAX_SECONDS > 0:
                time.sleep(random.uniform(_WANDB_INIT_STAGGER_MIN_SECONDS,
                                          _WANDB_INIT_STAGGER_MAX_SECONDS))
            name = self._wandb_names[s] if self._wandb_names else f"{base_name}-s{s}"
            cfg = dict(self.hparams)
            cfg["seed_index"] = s
            cfg["parallel_seeds"] = self.N
            # Overwrite each hp_pack key's shared-CLI-default scalar with the
            # per-slot value, so wandb's filter / parallel-coordinates UI
            # reflects the actual hyperparameter this vmap slot is running.
            if self._hp_pack is not None:
                for k, values in self._hp_pack.items():
                    cfg[k] = values[s]
            # Drop the pack transport fields from wandb config -- the per-slot
            # key overwrites above already give wandb exactly the same info
            # and this keeps the logged config clean (no ~1KB JSON blob on
            # every run, no scratch-file paths).
            cfg.pop("hp_pack", None)
            cfg.pop("hp_pack_inline", None)
            if self.sweep_id is not None:
                cfg["sweep_id"] = int(self.sweep_id)
            # config_tag: automatic from whatever hps are in the pack
            # (excluding seed). Used as both the wandb-filter paste string
            # and the internal join key in analysis scripts. Unique per
            # (sweep, config).
            cfg["config_tag"] = build_config_tag(
                self._hp_pack, s, self.sweep_id,
                hparams=self.hparams, tag_keys=self.config_tag_keys,
            )
            # Honor WANDB_DIR when set by the sbatch wrapper (offline-mode
            # sweeps write to per-job netscratch dirs that a background
            # sync loop uploads to wandb.ai). Fall back to /tmp for the
            # local / online-mode path so unchanged setups still work.
            run = wandb.init(
                project="diffusion_online_rl",
                name=name,
                dir=os.environ.get("WANDB_DIR", "/tmp"),
                group=group,
                config=cfg,
                reinit="create_new",
                settings=wandb.Settings(console="off"),
            )
            self._wandb_runs.append(run)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _set_pending_step(self, step: int):
        # Unified step for the current flush cycle. Each seed's wandb run uses
        # its own seed-local step, but we record a single step for "now"; all
        # per-seed values logged under the same flush share that step.
        # We store per-seed pending steps so each seed can use its own
        # sample_step as the wandb x-axis.
        for s in range(self.N):
            if self._wandb_pending_step[s] is None:
                self._wandb_pending_step[s] = int(self.sample_logs[s].sample_step)

    def add_scalar_per_seed(self, seed: int, tag: str, value: float, step: Optional[int] = None):
        if step is not None:
            # Flush if the step changed.
            if self._wandb_pending_step[seed] is not None and step != self._wandb_pending_step[seed]:
                self._flush_wandb_seed(seed)
            self._wandb_pending_step[seed] = int(step)
        self._wandb_pending[seed][tag] = float(value)

    def add_scalar_all(self, tag: str, value: float, step: Optional[int] = None):
        for s in range(self.N):
            self.add_scalar_per_seed(s, tag, value, step)

    def _buffer_wandb_per_seed(self, seed: int, data: dict, step: int):
        if self._wandb_pending_step[seed] is not None and step != self._wandb_pending_step[seed]:
            self._flush_wandb_seed(seed)
        self._wandb_pending[seed].update(data)
        self._wandb_pending_step[seed] = int(step)

    def add_arrays_vmap(self, array_info: dict):
        if not array_info:
            return

        snr = getattr(self.algorithm, "_snr", None)
        if snr is not None:
            log2_snr = np.log2(np.maximum(snr, 1e-12))

        mala_keys = ("MALA/eta_scale", "MALA/acceptance_rate", "MALA/clip_frac")
        mala_key_set = set(mala_keys)

        for s in range(self.N):
            step = int(self.sample_logs[s].sample_step)
            seed_arrays = {tag: np.asarray(value)[s] for tag, value in array_info.items()}

            has_mala = any(k in seed_arrays for k in mala_keys)
            if has_mala and snr is not None:
                log_dict = {}
                for metric_name in mala_keys:
                    values = np.asarray(seed_arrays.get(metric_name, np.full_like(log2_snr, np.nan)))
                    if np.all(np.isnan(values)):
                        continue
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(log2_snr, values)
                    ax.set_xlabel("log2(SNR)")
                    ax.set_ylabel(metric_name.split("/")[-1])
                    ax.set_title(f"{metric_name} ({self.env_name})")
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    log_dict[metric_name] = wandb.Image(fig)
                    plt.close(fig)
                if log_dict:
                    self._buffer_wandb_per_seed(s, log_dict, step)

            for tag, value in seed_arrays.items():
                if tag in mala_key_set:
                    continue
                arr = np.asarray(value)
                if snr is not None and len(arr) == len(snr):
                    table = wandb.Table(
                        columns=["log2_snr", "value"],
                        data=[[float(log2_snr[i]), float(arr[i])] for i in range(len(arr))],
                    )
                else:
                    table = wandb.Table(
                        columns=["level", "value"],
                        data=[[int(i), float(arr[i])] for i in range(len(arr))],
                    )
                self._buffer_wandb_per_seed(s, {tag: table}, step)

    def _flush_wandb_seed(self, s: int):
        if self._wandb_pending[s]:
            self._wandb_runs[s].log(self._wandb_pending[s], step=self._wandb_pending_step[s])
            self._wandb_pending[s] = {}
        self._wandb_pending_step[s] = None

    def _flush_wandb(self):
        for s in range(self.N):
            self._flush_wandb_seed(s)

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

        # Vmapped policy rollout. Returns (action [N,M,A], q [N,M], v [N,M]).
        action_nm, q_per_env, v_per_env = self.algorithm.get_action_vmap(keys, obs_nm)

        # Host-side seed-axis-vectorized EMA + eta update.
        # NOTE: _prev_* roll-forward happens AFTER env.step (below) so we know
        # which envs terminated/truncated this step.
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
        if bool(getattr(self.algorithm, "one_step_dist_shift_eta", False)):
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
        v_mean_per_seed = np.mean(v_per_env, axis=1)                # [N]

        for s in range(self.N):
            seed_info = {
                "action_mean": action_mean[s],
                "action_var": action_var[s],
                "action_clip_frac": action_clip_frac[s],
                "q_agg": float(q_mean_per_seed[s]),
                "v_value": float(v_mean_per_seed[s]),
            }
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
                    self.add_scalar_per_seed(s, ep_key, ret, step=env_step)
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
    # EMA update (host-side, seed-axis-vectorized)
    # ------------------------------------------------------------------
    def _on_policy_ema_update(self, q_per_env: np.ndarray, v_per_env: np.ndarray):
        """Mirror of OffPolicyTrainer's on_policy_ema block for vmap mode.

        Supports: kl-only, and one_step_dist_shift_eta. Every reduction that
        was over [M] per-env is now over axis=1 of [N, M] arrays, yielding
        [N]-shaped per-seed quantities. State EMA fields are [N] floats.
        """
        alg = self.algorithm
        adv_per_env = q_per_env - v_per_env          # [N, M]
        m2_batch = np.mean(adv_per_env ** 2, axis=1)  # [N]
        state = alg.state
        # Per-seed adv EMA rate: [N]-shaped array from state so this ablation can vmap.
        tau_v = np.asarray(state.adv_ema_tau).astype(np.float64)
        if tau_v.ndim == 0:
            tau_v = np.broadcast_to(tau_v, (self.N,)).astype(np.float64)
        cur_m2 = np.asarray(state.advantage_second_moment_ema)     # [N]
        cur_m3 = np.asarray(state.advantage_third_moment_ema)      # [N]
        cur_c = np.asarray(state.dist_shift_covariance_ema)        # [N]
        cur_shape = np.asarray(state.dist_shift_shape_ema)         # [N]
        cur_coeff = np.asarray(state.dist_shift_coeff_ema)         # [N]
        cur_shape3 = np.asarray(state.dist_shift_shape3_ema)       # [N]
        new_m2 = (1 - tau_v) * cur_m2 + tau_v * m2_batch

        _one_step = bool(getattr(alg, "one_step_dist_shift_eta", False))
        _dist_shift = bool(getattr(alg, "dist_shift_eta", False))
        _any_dist_shift = _one_step or _dist_shift

        new_m3 = cur_m3
        new_c = cur_c
        new_shape = cur_shape
        new_coeff = cur_coeff
        new_shape3 = cur_shape3

        if _any_dist_shift:
            m3_batch = np.mean(adv_per_env ** 3, axis=1)              # [N]
            new_m3 = (1 - tau_v) * cur_m3 + tau_v * m3_batch

        c_batch = None
        c_batch_valid = None
        if _one_step and self._prev_adv_per_env is not None and self._prev_valid is not None:
            valid = self._prev_valid  # [N, M] bool
            valid_count = np.sum(valid, axis=1)  # [N]
            prod = valid.astype(np.float64) * (adv_per_env ** 2) * self._prev_adv_per_env
            sums = np.sum(prod, axis=1)  # [N]
            c_batch = np.where(valid_count > 0, sums / np.maximum(valid_count, 1), 0.0)
            c_batch_valid = valid_count > 0

        if _any_dist_shift and c_batch is not None:
            # Seeds with c_batch_valid get an EMA update; others keep prior values.
            new_c_candidate = (1 - tau_v) * cur_c + tau_v * c_batch
            new_c = np.where(c_batch_valid, new_c_candidate, cur_c)

            gamma = np.asarray(state.gamma).astype(np.float64)
            if gamma.ndim == 0:
                gamma = np.broadcast_to(gamma, (self.N,)).astype(np.float64)
            tau_s = np.asarray(state.shape_ema_tau).astype(np.float64)
            if tau_s.ndim == 0:
                tau_s = np.broadcast_to(tau_s, (self.N,)).astype(np.float64)
            v_raw_safe = np.maximum(m2_batch, 1e-8)
            b_batch = 2.0 * gamma * c_batch + m3_batch
            s_batch = b_batch / v_raw_safe ** 1.5
            new_shape_candidate = (1 - tau_s) * cur_shape + tau_s * s_batch
            new_shape = np.where(c_batch_valid, new_shape_candidate, cur_shape)
            new_coeff_candidate = (1 - tau_s) * cur_coeff + tau_s * b_batch
            new_coeff = np.where(c_batch_valid, new_coeff_candidate, cur_coeff)

        # η selection, per-seed.
        kl_budget = np.asarray(state.kl_budget_val).astype(np.float64)
        if kl_budget.ndim == 0:
            kl_budget = np.broadcast_to(kl_budget, (self.N,)).astype(np.float64)
        m2_safe = np.maximum(new_m2, 1e-8)
        eta_kl_raw = np.sqrt(2.0 * kl_budget / m2_safe)                # [N]
        sqrt_v = np.sqrt(m2_safe)

        if _one_step:
            if bool(getattr(alg, "direct_eta_coeff_ema", False)):
                eta_star_raw = np.where(new_coeff < -1e-8, -new_m2 / new_coeff, np.inf)
            else:
                eta_star_raw = np.where(new_shape < -1e-8, -1.0 / (sqrt_v * new_shape), np.inf)
            eta_raw = np.minimum(eta_star_raw, eta_kl_raw)
        else:
            eta_star_raw = np.full_like(eta_kl_raw, np.inf)
            eta_raw = eta_kl_raw

        new_eta = eta_raw * sqrt_v  # code-space η (per-seed)

        # Write back EMA + eta state.
        alg.state = state._replace(
            advantage_second_moment_ema=jnp.asarray(new_m2.astype(np.float32)),
            advantage_third_moment_ema=jnp.asarray(new_m3.astype(np.float32)),
            dist_shift_covariance_ema=jnp.asarray(new_c.astype(np.float32)),
            dist_shift_shape_ema=jnp.asarray(new_shape.astype(np.float32)),
            dist_shift_coeff_ema=jnp.asarray(new_coeff.astype(np.float32)),
            dist_shift_shape3_ema=jnp.asarray(new_shape3.astype(np.float32)),
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
        for tag, vals in array_info.items():
            self._array_accum.setdefault(tag, []).append(np.asarray(vals))

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
            if self._array_accum:
                averaged_arrays = {
                    tag: sum(v_list) / len(v_list)
                    for tag, v_list in self._array_accum.items()
                }
                self._array_accum.clear()
                self.add_arrays_vmap(averaged_arrays)
            for tag, vals in info.items():
                arr = np.asarray(vals)
                for s in range(self.N):
                    self.add_scalar_per_seed(s, tag, float(arr[s]),
                                             step=int(self.sample_logs[s].sample_step))

    # ------------------------------------------------------------------
    # Periodic sample-interval metrics
    # ------------------------------------------------------------------
    def _log_periodic_sample_metrics(self):
        alg = self.algorithm
        state = alg.state
        for s in range(self.N):
            sstep = int(self.sample_logs[s].sample_step)
            self.sample_logs[s].log_accumulator(
                lambda k, v, _step, _s=s, _sstep=sstep: self.add_scalar_per_seed(_s, k, v, step=_sstep)
            )
            m2 = float(np.asarray(state.advantage_second_moment_ema)[s])
            kl_budget = float(np.asarray(state.kl_budget_val)[s])
            m3 = float(np.asarray(state.advantage_third_moment_ema)[s])
            cov = float(np.asarray(state.dist_shift_covariance_ema)[s])
            shape = float(np.asarray(state.dist_shift_shape_ema)[s])
            coeff = float(np.asarray(state.dist_shift_coeff_ema)[s])
            shape3 = float(np.asarray(state.dist_shift_shape3_ema)[s])
            tfg_eta = float(np.asarray(state.tfg_eta)[s])
            eta = tfg_eta / float(np.sqrt(max(m2, 1e-8)))
            eta_kl = float(np.sqrt(2.0 * kl_budget / max(m2, 1e-8)))
            self.add_scalar_per_seed(s, "Global_EMAs/Advantage_second_moment", m2, step=sstep)
            self.add_scalar_per_seed(s, "Global_EMAs/Advantage_third_moment", m3, step=sstep)
            self.add_scalar_per_seed(s, "Global_EMAs/Distribution_shift_covariance", cov, step=sstep)
            self.add_scalar_per_seed(s, "Global_EMAs/Distribution_shift_shape", shape, step=sstep)
            self.add_scalar_per_seed(s, "Global_EMAs/Distribution_shift_coefficient", coeff, step=sstep)
            self.add_scalar_per_seed(s, "Global_EMAs/Distribution_shift_shape3", shape3, step=sstep)
            self.add_scalar_per_seed(s, "Global_EMAs/eta", eta, step=sstep)
            self.add_scalar_per_seed(s, "Global_EMAs/tfg_eta", tfg_eta, step=sstep)
            self.add_scalar_per_seed(s, "Global_EMAs/eta_kl_ceiling", eta_kl, step=sstep)
            self.add_scalar_per_seed(s, "Global_EMAs/eta_kl_budget", eta_kl, step=sstep)
            if bool(getattr(alg, "one_step_dist_shift_eta", False)):
                if bool(getattr(alg, "direct_eta_coeff_ema", False)):
                    eta_one_step = -m2 / coeff if coeff < -1e-8 else float("nan")
                else:
                    eta_one_step = -1.0 / (float(np.sqrt(max(m2, 1e-8))) * shape) if shape < -1e-8 else float("nan")
                if not np.isnan(eta_one_step):
                    self.add_scalar_per_seed(s, "Global_EMAs/eta_quadratic", eta_one_step, step=sstep)
                    self.add_scalar_per_seed(s, "Global_EMAs/eta_one_step_dist_shift", eta_one_step, step=sstep)
        self._flush_wandb()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self, key: jax.Array):
        try:
            if self._is_single_key(key):
                train_key, warmup_key = jax.random.split(key)
            elif self._is_batched_key(key):
                split_keys = jax.vmap(lambda k: jax.random.split(k, 2))(key)
                train_key = split_keys[:, 0]
                warmup_key = split_keys[:, 1]
            else:
                raise ValueError(f"Unexpected key shape {np.shape(key)} for parallel_seeds={self.N}")
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
                if self._is_single_key(update_key):
                    update_keys = jax.random.split(update_key, self.update_per_iteration)
                    for i in range(self.update_per_iteration):
                        self.update(update_keys[i])
                else:
                    update_keys = jax.vmap(
                        lambda k: jax.random.split(k, self.update_per_iteration)
                    )(update_key)
                    for i in range(self.update_per_iteration):
                        self.update(update_keys[:, i])
            else:
                self.update(update_key)
            self.progress.n = self.sample_logs[0].sample_step
            self.progress.refresh()
            self._flush_wandb()
        return obs

    def finish(self):
        self._flush_wandb()
        try:
            self.env.close()
        except Exception:
            pass
        if hasattr(self, "progress"):
            self.progress.close()
        for run in self._wandb_runs:
            try:
                run.finish()
            except Exception:
                pass
