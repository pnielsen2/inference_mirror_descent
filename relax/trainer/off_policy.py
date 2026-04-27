from pathlib import Path
from typing import Callable, Optional, Tuple
import math
import numbers
import os
import pickle
import signal
import time

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import Env
from tqdm import tqdm
import wandb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from relax.algorithm import Algorithm
from relax.buffer import ExperienceBuffer, TreeBuffer
from relax.env.vector import VectorEnv
from relax.trainer.accumulator import Accumulator, SampleLog, UpdateLog, Interval
from relax.utils.experience import Experience
from relax.utils.jax_utils import action_to_latent_normalcdf, latent_to_action_normalcdf


_REQUEUE_PARTITIONS = {"kempner_requeue", "seas_requeue"}


def _detect_env_can_terminate(env_name: str) -> bool:
    """Probe whether a gymnasium env can emit terminated=True.

    Creates a temporary env instance and checks for a
    ``_terminate_when_unhealthy`` attribute (MuJoCo convention).  Envs that
    lack any termination mechanism (e.g. Swimmer, HalfCheetah) always return
    terminated=False from step(), so episode-length / reward-mean logging is
    not meaningful for them.
    """
    try:
        import gymnasium
        probe = gymnasium.make(env_name)
        inner = probe.unwrapped
        can = getattr(inner, "_terminate_when_unhealthy", None)
        probe.close()
        if can is not None:
            return bool(can)
        # If the attribute doesn't exist, check if step ever hardcodes
        # terminated=False by inspecting the source. Fall back to True
        # (assume termination is possible) to avoid silently dropping data.
        import inspect
        src = inspect.getsource(inner.step)
        if "return observation, reward, False, False" in src:
            return False
        return True
    except Exception:
        return True


def _detect_slurm_requeue() -> bool:
    """Return True iff we're running on a requeue-capable SLURM partition.

    Checks ``$SLURM_JOB_PARTITION`` against a known set of requeue
    partitions rather than probing the job's Requeue flag, because
    non-requeue partitions (e.g. kempner_h100) can have Requeue=1 set
    by default even though they are never actually preempted.
    """
    partition = os.environ.get("SLURM_JOB_PARTITION", "")
    return partition in _REQUEUE_PARTITIONS


class OffPolicyTrainer:
    def __init__(
        self,
        env: Env,
        algorithm: Algorithm,
        buffer: ExperienceBuffer,
        log_path: Path,
        batch_size: int = 256,
        val_batch_size: Optional[int] = None,
        start_step: int = 1000,
        total_step: int = int(1e6),
        sample_per_iteration: int = 1,
        update_per_iteration: int = 1,
        evaluate_env: Optional[Env] = None,
        evaluate_every: int = 10000,
        evaluate_n_episode: int = 20,
        sample_log_n_env_step: int = 1000,
        update_log_n_env_steps: int = 5000,
        debug: bool = False,
        timing_log_every: int = 0,
        done_info_keys: Tuple[str, ...] = (),
        save_policy_every: int = 10000,
        save_value: bool = True,
        hparams: Optional[dict] = None,
        policy_pkl_template: str = "policy-{sample_step}-{update_step}.pkl",
        warmup_with: str = "random",  # "policy" or "random"
        use_validation: bool = False,
        validation_size: int = 5000,
        use_hypergrad: bool = False,
        hypergrad_period: int = 100,
        # Optional dedicated validation environment and buffer. When provided,
        # validation data is drawn from this buffer instead of reserving the
        # most recent transitions in the training buffer.
        val_env: Optional[Env] = None,
        val_buffer: Optional[ExperienceBuffer] = None,
        validation_ratio: float = 0.0,
        track_next_action: bool = False,  # Enable SARSA-style next_action tracking
        latent_action_space: bool = False,
        latent_action_eps: float = 1e-6,
        tfg_patience: float = float("inf"),
        tfg_reduction_factor: float = 1.0,
        tfg_eta_start: float = 0.0,
        tfg_eta_end: Optional[float] = None,
        log_md_kl_every: int = int(1e9),
        # Soft policy iteration
        soft_pi_mode: bool = False,
        iterations_per_pi_step: int = 100000,
        num_pi_steps: int = 10,
        equal_episode_weighting: bool = False,
    ):
        self.env = env
        self.algorithm = algorithm
        self.buffer = buffer
        self.batch_size = int(batch_size)
        # Use val_batch_size if provided; otherwise mirror the training
        # batch size for validation.
        self.val_batch_size = int(val_batch_size) if val_batch_size is not None else int(batch_size)
        self.start_step = start_step
        self.total_step = total_step
        self.sample_per_iteration = sample_per_iteration
        self.update_per_iteration = update_per_iteration
        self.log_path = log_path
        self.policy_pkl_template = policy_pkl_template
        self.evaluate_env = evaluate_env
        self.evaluate_every = evaluate_every
        self.evaluate_n_episode = evaluate_n_episode
        self.sample_log_n_env_step = sample_log_n_env_step
        self.update_log_n_env_steps = update_log_n_env_steps
        self._last_update_log_env_step = 0
        self._array_accum = {}  # accumulate per-update arrays for averaging at log time
        self.debug = bool(debug)
        self.timing_log_every = int(timing_log_every)
        self.done_info_keys = done_info_keys
        self.save_policy_every = save_policy_every
        self.hparams = hparams
        self.warmup_with = warmup_with
        self.save_value = save_value
        self.use_validation = bool(use_validation)
        self.validation_size = int(validation_size)
        self.use_hypergrad = bool(use_hypergrad)
        self.hypergrad_period = int(hypergrad_period)
        # Optional dedicated validation env / buffer driven by a validation
        # ratio. When these are provided, validation batches are drawn from
        # the validation buffer instead of the within-buffer temporal split.
        self.val_env = val_env
        self.val_buffer = val_buffer
        self.validation_ratio = float(validation_ratio)
        self.track_next_action = track_next_action
        self.latent_action_space = bool(latent_action_space)
        self.latent_action_eps = float(latent_action_eps)

        self.tfg_patience = float(tfg_patience)
        self.tfg_reduction_factor = float(tfg_reduction_factor)
        self._tfg_plateau_lambda: Optional[float] = None
        self._tfg_plateau_best_return: float = -float("inf")
        self._tfg_plateau_best_step: int = 0
        self._tfg_plateau_bad_count: int = 0

        # Log-linear tfg_eta schedule over training
        self.tfg_eta_start = float(tfg_eta_start)
        self.tfg_eta_end = float(tfg_eta_end) if tfg_eta_end is not None else self.tfg_eta_start
        self._tfg_schedule_enabled = (self.tfg_eta_end != self.tfg_eta_start)

        # Accumulator for averaging eta between log steps
        self._eta_accum: list = []
        self._tfg_eta_accum: list = []
        # Accumulators for eta candidate breakdown (raw space)
        self._eta_kl_accum: list = []
        self._eta_quad_accum: list = []
        self._eta_cubic_accum: list = []

        # One-step dist-shift covariance: buffer previous advantages and done mask
        self._prev_adv_per_env: "np.ndarray | None" = None
        self._prev_valid: "np.ndarray | None" = None
        # Two-step dist-shift: buffer advantages from two steps ago
        self._prev2_adv_per_env: "np.ndarray | None" = None
        self._prev2_valid: "np.ndarray | None" = None

        # Equal episode weighting: episode-weighted buffer sampling + episode-averaged EMAs
        self.equal_episode_weighting = bool(equal_episode_weighting)

        # Periodic KL(π_tilt || π_0) logging
        self.log_md_kl_every = int(log_md_kl_every)

        # Soft policy iteration
        self.soft_pi_mode = bool(soft_pi_mode)
        self.iterations_per_pi_step = int(iterations_per_pi_step)
        self.num_pi_steps = int(num_pi_steps)
        if self.soft_pi_mode:
            self.total_step = self.num_pi_steps * self.iterations_per_pi_step
            self._spi_accumulator = Accumulator()
            self._spi_pi_step = 0  # current PI step (0 = Q-only)
            self._spi_prev = None  # prev-step data for on-policy Bellman MSE

        if self.latent_action_space and not (self.latent_action_eps > 0.0):
            raise ValueError(
                f"latent_action_eps must be > 0 when latent_action_space is enabled (got {self.latent_action_eps}). "
                "Setting eps=0 can yield infinite latents when inverting NormalCDF near action boundaries."
            )
        # State for tracking next_action in SARSA-style buffer
        self._prev_buffer_indices: Optional[np.ndarray] = None
        self._prev_dones: Optional[np.ndarray] = None
        # TODO: make EpisodeLog and Experience configurable
        # TODO: re-add done_info_keys support
        # TODO: re-add evaluation support

        self.env_name = env.spec.id if env.spec is not None else "env"
        _gamma = getattr(self.algorithm, "gamma", 0.99)
        _q_label = "Q_η" if getattr(self.algorithm, "use_entropic_q", False) else "Q"
        if isinstance(self.env.unwrapped, VectorEnv):
            self.is_vec = True
            _num_envs = self.env.unwrapped.num_envs
        else:
            self.is_vec = False
            _num_envs = 1
        self._num_envs = _num_envs
        _can_terminate = _detect_env_can_terminate(self.env_name)
        self.sample_log = SampleLog(num_envs=_num_envs, env_name=self.env_name, gamma=_gamma, q_label=_q_label, env_can_terminate=_can_terminate)
        self.update_log = UpdateLog()
        self.last_metrics = {}
        self._wandb_pending = {}
        self._wandb_pending_step = None

        # Episode tracking for equal_episode_weighting
        if self.equal_episode_weighting:
            self._init_episode_tracking(_num_envs)

        self._timing_sample_s = 0.0
        self._timing_update_s = 0.0
        self._timing_get_action_s = 0.0
        self._timing_env_step_s = 0.0
        self._timing_batch_s = 0.0
        self._timing_alg_update_s = 0.0
        self._timing_sample_n = 0
        self._timing_update_n = 0
        self._timing_last_logged_sample_step = 0
        # The following two depends on sample_step, which may not update by one only
        self.sample_log_interval = Interval(self.sample_log_n_env_step)
        self.save_policy_interval = Interval(self.save_policy_every)

        # Checkpointing for SLURM preemption resilience.
        # Only enabled when the job is actually requeueable — otherwise
        # a SIGTERM from `scancel` or the wandb stop button gets trapped
        # and forces a multi-minute buffer pickle before the process can
        # die, leaving the job stuck in SLURM CG state.
        self._checkpoint_dir = self.log_path / "checkpoints"
        self._preempted = False
        self._checkpoint_enabled = _detect_slurm_requeue()
        if self._checkpoint_enabled:
            print("[Checkpoint] SLURM job has Requeue=1 — checkpointing enabled.")
        else:
            print("[Checkpoint] Not a requeueable SLURM job — checkpointing disabled.")

        # Check for existing checkpoint to resume from (only on requeueable
        # jobs — on a fresh run we never want to silently pick up a stale
        # checkpoint from a previous experiment in the same log_path).
        # The wandb run ID is also persisted at the log_path level
        # (outside the atomic checkpoint swap) so it survives even if
        # a checkpoint save was interrupted between renames.
        wandb_id_file = self.log_path / "wandb_run_id.txt"
        wandb_dir = os.environ.get("WANDB_DIR", str(log_path))
        ckpt = self._find_checkpoint() if self._checkpoint_enabled else None
        if ckpt is not None and wandb_id_file.exists():
            with open(wandb_id_file) as f:
                resume_id = f.read().strip()
            wandb.init(
                project="diffusion_online_rl",
                name=log_path.name,
                dir=wandb_dir,
                group=env.spec.id,
                config=self.hparams,
                id=resume_id,
                resume="must",
            )
            print(f"[Checkpoint] Resuming wandb run {resume_id}")
        else:
            wandb.init(
                project="diffusion_online_rl",
                name=log_path.name,
                dir=wandb_dir,
                group=env.spec.id,
                config=self.hparams,
            )

    # ------------------------------------------------------------------
    # Equal episode weighting helpers
    # ------------------------------------------------------------------

    def _init_episode_tracking(self, num_envs: int):
        """Initialize state for episode-weighted buffer sampling and episode-averaged EMAs."""
        # --- Buffer sampling: track completed episodes ---
        self._ep_current = [[] for _ in range(num_envs)]  # per-env current episode buffer indices
        self._ep_completed = []  # list of np.array of buffer indices for completed episodes
        self._ep_total_added = 0  # monotonic counter of total transitions added
        self._ep_start_seq = np.zeros(num_envs, dtype=np.int64)  # per-env episode start sequence

        # --- Adjusted EMA rates for per-episode updates ---
        # Convert per-step tau to per-episode tau: tau_ep = 1 - (1-tau)^L
        # using L=1000 (standard fixed-length MuJoCo episode length).
        L = 1000
        tau_v = float(getattr(self.algorithm, 'advantage_ema_tau', 0.0005))
        tau_s = float(getattr(self.algorithm, 'shape_ema_tau', 0.0001))
        tau_s3 = float(getattr(self.algorithm, 'shape3_ema_tau', 0.00005))
        self._ep_tau_v = 1.0 - (1.0 - tau_v) ** L
        self._ep_tau_s = 1.0 - (1.0 - tau_s) ** L
        self._ep_tau_s3 = 1.0 - (1.0 - tau_s3) ** L

        # --- Episode-averaged EMAs: per-env accumulators ---
        self._eema_sum_a2 = np.zeros(num_envs)
        self._eema_sum_a3 = np.zeros(num_envs)
        self._eema_sum_a4 = np.zeros(num_envs)
        self._eema_count = np.zeros(num_envs, dtype=np.int64)
        # One-step covariance: A_curr^2 * A_prev
        self._eema_sum_c = np.zeros(num_envs)
        self._eema_c_count = np.zeros(num_envs, dtype=np.int64)
        # D_psi covariance: D_psi * A
        self._eema_sum_dpsi_a = np.zeros(num_envs)
        # Two-step quantities
        self._eema_sum_d = np.zeros(num_envs)  # A_prev2 * A_curr^2 * A_prev
        self._eema_d_count = np.zeros(num_envs, dtype=np.int64)
        self._eema_sum_e = np.zeros(num_envs)  # A_prev * A_curr^3
        self._eema_sum_aprev2_acurr2 = np.zeros(num_envs)  # A_prev^2 * A_curr^2
        self._eema_ef_count = np.zeros(num_envs, dtype=np.int64)
        # Per-env prev tracking (within episode)
        self._eema_prev_adv = np.zeros(num_envs)
        self._eema_prev_valid = np.zeros(num_envs, dtype=bool)
        self._eema_prev2_adv = np.zeros(num_envs)
        self._eema_prev2_valid = np.zeros(num_envs, dtype=bool)

    def _ep_record_transition(self, old_ptr: int, batch_size: int, terminated, truncated):
        """Record buffer indices for the current episode and finalize on done."""
        indices = np.arange(old_ptr, old_ptr + batch_size) % self.buffer.max_len
        done = np.atleast_1d(np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool))
        self._ep_total_added += batch_size

        for j in range(batch_size):
            self._ep_current[j].append(indices[j])
            if done[j]:
                if len(self._ep_current[j]) > 0:
                    self._ep_completed.append(
                        (np.array(self._ep_current[j]), int(self._ep_start_seq[j]))
                    )
                self._ep_current[j] = []
                self._ep_start_seq[j] = self._ep_total_added

    def _ep_record_transition_single(self, old_ptr: int, terminated, truncated):
        """Record buffer index for single-env episode tracking."""
        done = bool(terminated) or bool(truncated)
        self._ep_total_added += 1
        self._ep_current[0].append(old_ptr)
        if done:
            if len(self._ep_current[0]) > 0:
                self._ep_completed.append(
                    (np.array(self._ep_current[0]), int(self._ep_start_seq[0]))
                )
            self._ep_current[0] = []
            self._ep_start_seq[0] = self._ep_total_added

    def _ep_prune_stale(self):
        """Remove episodes whose transitions have been overwritten by the ring buffer."""
        if self._ep_total_added <= self.buffer.max_len:
            return  # buffer hasn't wrapped yet
        oldest_valid_seq = self._ep_total_added - self.buffer.max_len
        self._ep_completed = [(indices, start_seq) for indices, start_seq in self._ep_completed
                              if start_seq >= oldest_valid_seq]

    def _ep_sample_indices(self, size: int) -> np.ndarray:
        """Sample transition indices with equal weight per episode.

        Picks an episode uniformly, then a transition uniformly within it.
        Falls back to uniform sampling if no completed episodes are available.
        """
        self._ep_prune_stale()
        if not self._ep_completed:
            # Fallback: uniform sampling
            return self.buffer.rng.integers(0, len(self.buffer), size=size)

        n_episodes = len(self._ep_completed)
        ep_choices = self.buffer.rng.integers(0, n_episodes, size=size)
        indices = np.empty(size, dtype=np.intp)
        for i, ep_idx in enumerate(ep_choices):
            ep_indices, _ = self._ep_completed[ep_idx]
            indices[i] = ep_indices[self.buffer.rng.integers(0, len(ep_indices))]
        return indices

    def _eema_accumulate_step(self, adv_per_env, obs=None, action=None):
        """Accumulate per-env advantage statistics for episode-averaged EMAs."""
        adv = np.atleast_1d(adv_per_env)
        n = len(adv)

        self._eema_sum_a2[:n] += adv ** 2
        self._eema_sum_a3[:n] += adv ** 3
        self._eema_sum_a4[:n] += adv ** 4
        self._eema_count[:n] += 1

        _one_step = getattr(self.algorithm, 'one_step_dist_shift_eta', False)
        _two_step = getattr(self.algorithm, 'two_step_dist_shift_eta', False)
        _dist_shift = getattr(self.algorithm, 'dist_shift_eta', False)

        if _one_step or _two_step:
            # One-step: c = A_curr^2 * A_prev
            for j in range(n):
                if self._eema_prev_valid[j]:
                    self._eema_sum_c[j] += adv[j] ** 2 * self._eema_prev_adv[j]
                    self._eema_c_count[j] += 1
                    # e = A_prev * A_curr^3
                    self._eema_sum_e[j] += self._eema_prev_adv[j] * adv[j] ** 3
                    # A_prev^2 * A_curr^2 (for f computation at finalize)
                    self._eema_sum_aprev2_acurr2[j] += self._eema_prev_adv[j] ** 2 * adv[j] ** 2
                    self._eema_ef_count[j] += 1

            if _two_step:
                for j in range(n):
                    if self._eema_prev2_valid[j]:
                        self._eema_sum_d[j] += (self._eema_prev2_adv[j]
                                                 * adv[j] ** 2
                                                 * self._eema_prev_adv[j])
                        self._eema_d_count[j] += 1

            # Shift prev -> prev2, update prev
            self._eema_prev2_adv[:n] = self._eema_prev_adv[:n].copy()
            self._eema_prev2_valid[:n] = self._eema_prev_valid[:n].copy()
            self._eema_prev_adv[:n] = adv
            self._eema_prev_valid[:n] = True

        elif _dist_shift:
            # D_psi-based covariance
            d_psi_per_env = self.algorithm.evaluate_d_psi(obs, action)
            if d_psi_per_env is not None:
                d_psi = np.asarray(d_psi_per_env).ravel()
                self._eema_sum_dpsi_a[:n] += d_psi * adv

    def _eema_finalize_episode(self, env_idx: int):
        """Compute episode averages and perform one EMA update for the finished episode."""
        count = int(self._eema_count[env_idx])
        if count == 0:
            self._eema_reset_env(env_idx)
            return

        _dist_shift = getattr(self.algorithm, 'dist_shift_eta', False)
        _one_step = getattr(self.algorithm, 'one_step_dist_shift_eta', False)
        _two_step = getattr(self.algorithm, 'two_step_dist_shift_eta', False)
        _any_dist_shift = _dist_shift or _one_step or _two_step

        # Episode-averaged quantities
        m2_ep = self._eema_sum_a2[env_idx] / count
        m3_ep = self._eema_sum_a3[env_idx] / count if _any_dist_shift else None

        c_ep = None
        if _one_step or _two_step:
            if self._eema_c_count[env_idx] > 0:
                c_ep = self._eema_sum_c[env_idx] / self._eema_c_count[env_idx]
        elif _dist_shift:
            if count > 0:
                c_ep = self._eema_sum_dpsi_a[env_idx] / count

        d_ep, e_ep, f_ep, rho_ep = None, None, None, None
        d_valid = False
        if _two_step and c_ep is not None:
            m4_ep = self._eema_sum_a4[env_idx] / count
            rho_ep = m4_ep - 3.0 * m2_ep ** 2

            if self._eema_d_count[env_idx] > 0:
                d_ep = self._eema_sum_d[env_idx] / self._eema_d_count[env_idx]
                d_valid = True

            if self._eema_ef_count[env_idx] > 0:
                ef_count = self._eema_ef_count[env_idx]
                e_ep = self._eema_sum_e[env_idx] / ef_count
                # f = E[(A_prev^2 - E[A^2]) * A_curr^2]
                #   = E[A_prev^2 * A_curr^2] - E[A^2] * E[A_curr^2 over valid pairs]
                # Approximate E[A_curr^2 over valid pairs] ≈ m2_ep
                f_ep = self._eema_sum_aprev2_acurr2[env_idx] / ef_count - m2_ep * m2_ep

        # Perform EMA update
        self._do_ema_update_and_eta(
            m2_batch=float(m2_ep),
            m3_batch=float(m3_ep) if m3_ep is not None else None,
            c_batch=float(c_ep) if c_ep is not None else None,
            d_batch=float(d_ep) if d_ep is not None else None,
            e_batch=float(e_ep) if e_ep is not None else None,
            f_batch=float(f_ep) if f_ep is not None else None,
            rho_batch=float(rho_ep) if rho_ep is not None else None,
            d_valid=d_valid,
        )
        self._eema_reset_env(env_idx)

    def _eema_reset_env(self, env_idx: int):
        """Reset episode EMA accumulators for a single environment."""
        self._eema_sum_a2[env_idx] = 0.0
        self._eema_sum_a3[env_idx] = 0.0
        self._eema_sum_a4[env_idx] = 0.0
        self._eema_count[env_idx] = 0
        self._eema_sum_c[env_idx] = 0.0
        self._eema_c_count[env_idx] = 0
        self._eema_sum_dpsi_a[env_idx] = 0.0
        self._eema_sum_d[env_idx] = 0.0
        self._eema_d_count[env_idx] = 0
        self._eema_sum_e[env_idx] = 0.0
        self._eema_sum_aprev2_acurr2[env_idx] = 0.0
        self._eema_ef_count[env_idx] = 0
        self._eema_prev_adv[env_idx] = 0.0
        self._eema_prev_valid[env_idx] = False
        self._eema_prev2_adv[env_idx] = 0.0
        self._eema_prev2_valid[env_idx] = False

    def _do_ema_update_and_eta(self, m2_batch, m3_batch=None, c_batch=None,
                                d_batch=None, e_batch=None, f_batch=None,
                                rho_batch=None, d_valid=False):
        """Perform one EMA update step with the given batch quantities and recompute eta.

        Uses adjusted per-episode EMA rates (computed in _init_episode_tracking).
        """
        _dist_shift = getattr(self.algorithm, 'dist_shift_eta', False)
        _one_step = getattr(self.algorithm, 'one_step_dist_shift_eta', False)
        _two_step = getattr(self.algorithm, 'two_step_dist_shift_eta', False)
        _any_dist_shift = _dist_shift or _one_step or _two_step

        tau_v = self._ep_tau_v
        tau_s = self._ep_tau_s
        gamma = self.algorithm.gamma

        new_m2 = (1 - tau_v) * float(self.algorithm.state.advantage_second_moment_ema) + tau_v * m2_batch
        new_m3 = float(self.algorithm.state.advantage_third_moment_ema)
        new_c = float(self.algorithm.state.dist_shift_covariance_ema)
        new_shape = float(self.algorithm.state.dist_shift_shape_ema)
        new_coeff = float(self.algorithm.state.dist_shift_coeff_ema)
        new_shape3 = float(self.algorithm.state.dist_shift_shape3_ema)

        if _any_dist_shift and m3_batch is not None:
            new_m3 = (1 - tau_v) * new_m3 + tau_v * m3_batch

            if c_batch is not None:
                new_c = (1 - tau_v) * new_c + tau_v * c_batch

                v_raw_safe = max(m2_batch, 1e-8)
                b_batch = 2.0 * gamma * c_batch + m3_batch
                s_batch = b_batch / (v_raw_safe ** 1.5)
                new_shape = (1 - tau_s) * new_shape + tau_s * s_batch
                new_coeff = (1 - tau_s) * new_coeff + tau_s * b_batch

            if _two_step and c_batch is not None and rho_batch is not None:
                tau_s3 = self._ep_tau_s3
                v_raw_safe = max(m2_batch, 1e-8)

                tau_batch = (1.0 / 6.0) * rho_batch
                if e_batch is not None:
                    tau_batch += 0.5 * gamma * e_batch
                if f_batch is not None:
                    tau_batch += 0.5 * gamma * f_batch
                if d_valid and d_batch is not None:
                    tau_batch += gamma ** 2 * d_batch

                s3_batch = tau_batch / (v_raw_safe ** 2)
                new_shape3 = (1 - tau_s3) * new_shape3 + tau_s3 * s3_batch

        # Compute eta
        kl_budget = self.algorithm.kl_budget
        m2_safe = max(new_m2, 1e-8)
        eta_kl_raw = float(np.sqrt(2.0 * kl_budget / m2_safe))
        sqrt_v = float(np.sqrt(m2_safe))

        if _two_step:
            S = new_shape
            s3 = new_shape3
            eta_cubic_raw = float('inf')
            disc = S ** 2 - 12.0 * s3
            if disc > 0 and abs(s3) > 1e-12:
                sqrt_disc = float(np.sqrt(disc))
                x_star = (-S + float(np.sign(S)) * sqrt_disc) / (6.0 * s3)
                if x_star > 0:
                    eta_cubic_raw = x_star / sqrt_v
            elif abs(s3) <= 1e-12 and S < -1e-8:
                eta_cubic_raw = -1.0 / (sqrt_v * S)

            eta_quad_raw = float('inf')
            if new_shape < -1e-8:
                eta_quad_raw = -1.0 / (sqrt_v * new_shape)

            eta_raw = min(eta_cubic_raw, eta_quad_raw, eta_kl_raw)

            self._eta_kl_accum.append(eta_kl_raw)
            self._eta_quad_accum.append(eta_quad_raw if eta_quad_raw != float('inf') else float('nan'))
            self._eta_cubic_accum.append(eta_cubic_raw if eta_cubic_raw != float('inf') else float('nan'))
        elif _any_dist_shift:
            if getattr(self.algorithm, 'direct_eta_coeff_ema', False):
                if new_coeff < -1e-8:
                    eta_star_raw = -new_m2 / new_coeff
                else:
                    eta_star_raw = float('inf')
            else:
                if new_shape < -1e-8:
                    eta_star_raw = -1.0 / (sqrt_v * new_shape)
                else:
                    eta_star_raw = float('inf')
            eta_raw = min(eta_star_raw, eta_kl_raw)

            self._eta_kl_accum.append(eta_kl_raw)
            self._eta_quad_accum.append(eta_star_raw if eta_star_raw != float('inf') else float('nan'))
        else:
            eta_raw = eta_kl_raw

        new_eta = eta_raw * float(np.sqrt(m2_safe))

        self.algorithm.state = self.algorithm.state._replace(
            advantage_second_moment_ema=jnp.float32(new_m2),
            advantage_third_moment_ema=jnp.float32(new_m3),
            dist_shift_covariance_ema=jnp.float32(new_c),
            dist_shift_shape_ema=jnp.float32(new_shape),
            dist_shift_coeff_ema=jnp.float32(new_coeff),
            dist_shift_shape3_ema=jnp.float32(new_shape3),
            tfg_eta=jnp.float32(new_eta),
        )

    def _current_eta_metrics(self):
        if not hasattr(self.algorithm, 'state'):
            return float('nan'), float('nan')
        st = self.algorithm.state
        tfg_eta = float(st.tfg_eta)
        if getattr(self.algorithm, 'on_policy_ema', False) and getattr(self.algorithm, 'critic_normalization', 'none') == 'ema':
            m2 = float(st.advantage_second_moment_ema)
            eta = tfg_eta / math.sqrt(max(m2, 1e-8))
            return eta, tfg_eta
        return tfg_eta, tfg_eta

    def _get_train_val_index_sets(self):
        """Return (train_indices, val_indices) for the replay buffer.

        When validation is enabled, we treat the last ``validation_size``
        transitions in temporal order as the validation set and exclude them
        from training batches. This implementation assumes a TreeBuffer
        backend and falls back to no split otherwise.
        """

        # If a dedicated validation buffer is provided, we no longer split the
        # main training buffer; callers are expected to sample validation
        # batches from self.val_buffer instead.
        if self.val_buffer is not None:
            return None, None

        if (not self.use_validation) or self.validation_size <= 0:
            return None, None

        if not isinstance(self.buffer, TreeBuffer):
            return None, None

        buf = self.buffer
        N = len(buf)
        if N <= 0:
            return None, None

        # If the buffer is too small relative to the requested validation
        # size and batch size, skip validation for now to avoid starving
        # training.
        if N <= self.validation_size + self.batch_size:
            return None, None

        val_size = min(self.validation_size, N)
        train_size = N - val_size

        max_len = buf.max_len
        ptr = buf.ptr

        import numpy as np

        if N < max_len:
            # No wrap-around yet: indices 0..N-1 are in temporal order.
            train_idx = np.arange(0, train_size, dtype=np.int64)
            val_idx = np.arange(train_size, N, dtype=np.int64)
        else:
            # Ring buffer is full: reconstruct temporal order starting from
            # the oldest element at (ptr - N) mod max_len.
            oldest = (ptr - N) % max_len
            order = (oldest + np.arange(N, dtype=np.int64)) % max_len
            train_idx = order[:train_size]
            val_idx = order[train_size:]

        return train_idx, val_idx

    def _sample_from_index_set(self, indices, batch_size: int):
        """Sample a batch uniformly from the given index set.

        Requires a TreeBuffer backend. Falls back to random sampling from the
        entire buffer if indices is None.
        """

        if indices is None or not isinstance(self.buffer, TreeBuffer):
            return self.buffer.sample(batch_size)

        import numpy as np

        buf = self.buffer
        if indices.size == 0:
            return self.buffer.sample(batch_size)

        replace = indices.size < batch_size
        chosen = buf.rng.choice(indices, size=batch_size, replace=replace)
        return buf.gather_indices(chosen, to_jax=True)

    def setup(self, dummy_data: Experience):
        self.algorithm.warmup(dummy_data)

        self.progress = tqdm(total=self.total_step, desc="Sample Step", disable=None, dynamic_ncols=True)

    def _array_nonfinite_summary(self, x: np.ndarray) -> str:
        if x.size == 0:
            return f"shape={x.shape} dtype={x.dtype} empty"
        finite = np.isfinite(x)
        n_bad = int((~finite).sum())
        n_total = int(x.size)
        if n_bad == 0:
            x_min = float(np.nanmin(x))
            x_max = float(np.nanmax(x))
            x_mean = float(np.nanmean(x))
            return f"shape={x.shape} dtype={x.dtype} min={x_min:.6g} max={x_max:.6g} mean={x_mean:.6g}"

        finite_vals = x[finite]
        if finite_vals.size > 0:
            x_min = float(np.min(finite_vals))
            x_max = float(np.max(finite_vals))
            x_mean = float(np.mean(finite_vals))
            finite_stats = f"finite_min={x_min:.6g} finite_max={x_max:.6g} finite_mean={x_mean:.6g}"
        else:
            finite_stats = "no_finite_values"

        flat_bad = np.flatnonzero((~finite).reshape(-1))
        bad_preview = flat_bad[:8].tolist()
        return (
            f"shape={x.shape} dtype={x.dtype} nonfinite={n_bad}/{n_total} "
            f"bad_flat_idx={bad_preview} {finite_stats}"
        )

    def _is_jax_array(self, x) -> bool:
        # Avoid importing jaxlib types directly; use module names for robustness.
        t = type(x)
        mod = getattr(t, "__module__", "")
        return mod.startswith("jax") or mod.startswith("jaxlib")

    def _assert_finite(self, name: str, value, *, sample_step: Optional[int] = None, update_step: Optional[int] = None) -> None:
        if not self.debug:
            return

        if value is None:
            return

        if isinstance(value, numbers.Number):
            if not np.isfinite(value):
                raise FloatingPointError(
                    f"Non-finite scalar detected in {name}: {value} (sample_step={sample_step}, update_step={update_step})"
                )
            return

        if isinstance(value, dict):
            for k, v in value.items():
                self._assert_finite(f"{name}.{k}", v, sample_step=sample_step, update_step=update_step)
            return

        if hasattr(value, "_fields"):
            for field_name in value._fields:
                self._assert_finite(
                    f"{name}.{field_name}",
                    getattr(value, field_name),
                    sample_step=sample_step,
                    update_step=update_step,
                )
            return

        if self._is_jax_array(value):
            x = value
            finite = jnp.all(jnp.isfinite(x))
            if not bool(jax.device_get(finite)):
                x_host = np.asarray(jax.device_get(x))
                summary = self._array_nonfinite_summary(x_host)
                raise FloatingPointError(
                    f"Non-finite tensor detected in {name} (sample_step={sample_step}, update_step={update_step}): {summary}"
                )
            return

        try:
            x_np = np.asarray(value)
        except Exception:
            return

        if not np.all(np.isfinite(x_np)):
            summary = self._array_nonfinite_summary(x_np)
            raise FloatingPointError(
                f"Non-finite tensor detected in {name} (sample_step={sample_step}, update_step={update_step}): {summary}"
            )

    def warmup(self, key: jax.Array):
        """Warm up the replay buffers with random (or policy) experience.

        When a dedicated validation buffer is provided, we also warm up the
        validation environment so that an initial validation set is available
        before training and hypergradient updates begin.
        """

        # Warmup for the main training buffer.
        train_obs, _ = self.env.reset()
        step = 0
        key_fn = jax.jit(lambda step: jax.random.fold_in(key, step))
        # State for SARSA-style next_action tracking during warmup
        warmup_prev_indices: Optional[np.ndarray] = None
        warmup_prev_dones: Optional[np.ndarray] = None
        while len(self.buffer) < self.start_step:
            step += 1
            if self.warmup_with == "random":
                action_env = self.env.action_space.sample()
                action = action_env
            elif self.warmup_with == "policy":
                _action_out = self.algorithm.get_action(key_fn(step), train_obs)
                action = _action_out[0] if isinstance(_action_out, tuple) else _action_out
                action_env = action
            else:
                raise ValueError(f"Invalid warmup_with {self.warmup_with}!")

            # If using latent action space, convert env-space action to latent for storage
            # and squash latent to env-space for interaction.
            if self.latent_action_space:
                if self.warmup_with == "random":
                    action_latent = np.asarray(action_to_latent_normalcdf(jnp.asarray(action_env), eps=self.latent_action_eps))
                    action_env = np.asarray(latent_to_action_normalcdf(jnp.asarray(action_latent), eps=self.latent_action_eps))
                else:
                    action_latent = np.asarray(action)
                    action_env = np.asarray(latent_to_action_normalcdf(jnp.asarray(action_latent), eps=self.latent_action_eps))
            else:
                action_latent = action
            
            # SARSA-style tracking: update previous experience's next_action
            if self.track_next_action and warmup_prev_indices is not None:
                if warmup_prev_dones is not None:
                    prev_dones = np.atleast_1d(warmup_prev_dones)
                    prev_indices = np.atleast_1d(warmup_prev_indices)
                    action_arr = np.atleast_2d(action_latent) if np.asarray(action_latent).ndim == 1 else action_latent
                    non_terminal_mask = ~prev_dones
                    if np.any(non_terminal_mask):
                        indices_to_update = prev_indices[non_terminal_mask]
                        actions_to_set = action_arr[non_terminal_mask] if self.is_vec else action_latent
                        update_exp = Experience(
                            obs=None, action=None, reward=None, done=None, next_obs=None,
                            next_action=actions_to_set,
                        )
                        self.buffer.replace(indices_to_update, update_exp)

            next_obs, reward, terminated, truncated, info = self.env.step(action_env)

            self._assert_finite(
                "warmup.next_obs",
                next_obs,
                sample_step=int(step),
                update_step=int(self.update_log.update_step),
            )
            self._assert_finite(
                "warmup.reward",
                reward,
                sample_step=int(step),
                update_step=int(self.update_log.update_step),
            )

            if self.latent_action_space:
                a_env_arr = np.asarray(action_env)
                info = dict(info)
                if self.is_vec:
                    info["action_env_mean_abs"] = np.mean(np.abs(a_env_arr), axis=-1)
                    info["action_env_std"] = np.std(a_env_arr, axis=-1)
                else:
                    info["action_env_mean_abs"] = float(np.mean(np.abs(a_env_arr)))
                    info["action_env_std"] = float(np.std(a_env_arr))

            experience = Experience.create(train_obs, action_latent, reward, terminated, truncated, next_obs, info)

            # Track buffer indices before adding (for next iteration's next_action update)
            if self.track_next_action:
                if self.is_vec:
                    batch_size = np.atleast_1d(reward).shape[0]
                    warmup_prev_indices = np.arange(self.buffer.ptr, self.buffer.ptr + batch_size) % self.buffer.max_len
                else:
                    warmup_prev_indices = np.array([self.buffer.ptr])
                warmup_prev_dones = np.atleast_1d(terminated) | np.atleast_1d(truncated)

            # Record buffer position before adding (for episode tracking)
            _warmup_old_ptr = self.buffer.ptr

            if self.is_vec:
                self.buffer.add_batch(experience)
            else:
                self.buffer.add(experience)

            # Episode tracking for equal_episode_weighting
            if self.equal_episode_weighting:
                if self.is_vec:
                    batch_size = np.atleast_1d(reward).shape[0]
                    self._ep_record_transition(_warmup_old_ptr, batch_size, terminated, truncated)
                else:
                    self._ep_record_transition_single(_warmup_old_ptr, terminated, truncated)

            if np.any(terminated) or np.any(truncated):
                train_obs, _ = self.env.reset()
            else:
                train_obs = next_obs

        # Initialize tracking state for main sample loop to continue from warmup
        if self.track_next_action:
            self._prev_buffer_indices = warmup_prev_indices
            self._prev_dones = warmup_prev_dones

        # Optional warmup for the dedicated validation buffer.
        if self.val_env is not None and self.val_buffer is not None and self.validation_ratio > 0.0:
            # Target an initial validation buffer size proportional to the
            # training warmup size.
            target_val_size = max(1, int(self.start_step * self.validation_ratio))
            val_obs, _ = self.val_env.reset()
            val_step = 0
            val_key_fn = jax.jit(lambda step: jax.random.fold_in(jax.random.fold_in(key, 1), step))
            while len(self.val_buffer) < target_val_size:
                val_step += 1
                if self.warmup_with == "random":
                    val_action_env = self.val_env.action_space.sample()
                    val_action = val_action_env
                elif self.warmup_with == "policy":
                    _val_action_out = self.algorithm.get_action(val_key_fn(val_step), val_obs)
                    val_action = _val_action_out[0] if isinstance(_val_action_out, tuple) else _val_action_out
                    val_action_env = val_action
                else:
                    raise ValueError(f"Invalid warmup_with {self.warmup_with}!")

                if self.latent_action_space:
                    if self.warmup_with == "random":
                        val_action_latent = np.asarray(action_to_latent_normalcdf(jnp.asarray(val_action_env), eps=self.latent_action_eps))
                        val_action_env = np.asarray(latent_to_action_normalcdf(jnp.asarray(val_action_latent), eps=self.latent_action_eps))
                    else:
                        val_action_latent = np.asarray(val_action)
                        val_action_env = np.asarray(latent_to_action_normalcdf(jnp.asarray(val_action_latent), eps=self.latent_action_eps))
                else:
                    val_action_latent = val_action
                    val_action_env = val_action

                val_next_obs, val_reward, val_terminated, val_truncated, val_info = self.val_env.step(val_action_env)

                val_experience = Experience.create(val_obs, val_action_latent, val_reward, val_terminated, val_truncated, val_next_obs, val_info)
                if isinstance(self.val_env.unwrapped, VectorEnv):
                    self.val_buffer.add_batch(val_experience)
                else:
                    self.val_buffer.add(val_experience)

                if np.any(val_terminated) or np.any(val_truncated):
                    val_obs, _ = self.val_env.reset()
                else:
                    val_obs = val_next_obs

            # Persist the latest validation observations for use during
            # subsequent sampling steps.
            self.val_obs = val_obs

        return train_obs

    def sample(self, sample_key: jax.Array, obs: np.ndarray):
        sl = self.sample_log

        do_timing = self.timing_log_every > 0
        if do_timing:
            t0 = time.perf_counter()
            t_action0 = t0

        action_out = self.algorithm.get_action(sample_key, obs)
        if isinstance(action_out, tuple) and len(action_out) == 3:
            action, q_agg, q_var = action_out
        elif isinstance(action_out, tuple) and len(action_out) == 2:
            action, q_agg = action_out
            q_var = None
        else:
            action, q_agg, q_var = action_out, None, None

        # --- On-policy EMA updates (--kl_budget mode) ---
        _on_policy_ema = getattr(self.algorithm, 'on_policy_ema', False)
        _v_per_env_mean = None
        _adv_for_one_step = None  # saved for buffering after env.step
        if _on_policy_ema and q_agg is not None and q_var is not None:
            # q_agg and q_var are actually per-env Q and V arrays
            q_per_env = np.asarray(q_agg).ravel()   # [num_envs]
            v_per_env = np.asarray(q_var).ravel()    # [num_envs]
            adv_per_env = q_per_env - v_per_env

            if not self.equal_episode_weighting:
                # --- Per-step EMA updates (default path) ---
                _dist_shift = getattr(self.algorithm, 'dist_shift_eta', False)
                _one_step = getattr(self.algorithm, 'one_step_dist_shift_eta', False)
                _two_step = getattr(self.algorithm, 'two_step_dist_shift_eta', False)
                _any_dist_shift = _dist_shift or _one_step or _two_step
                if _one_step or _two_step:
                    _adv_for_one_step = adv_per_env

                tau_v = float(self.algorithm.advantage_ema_tau)
                tau_s = float(self.algorithm.shape_ema_tau)
                m2_batch = float(np.mean(adv_per_env ** 2))
                new_m2 = (1 - tau_v) * float(self.algorithm.state.advantage_second_moment_ema) + tau_v * m2_batch
                new_m3 = float(self.algorithm.state.advantage_third_moment_ema)
                new_c = float(self.algorithm.state.dist_shift_covariance_ema)
                new_shape = float(self.algorithm.state.dist_shift_shape_ema)
                new_coeff = float(self.algorithm.state.dist_shift_coeff_ema)
                new_shape3 = float(self.algorithm.state.dist_shift_shape3_ema)

                if _any_dist_shift:
                    m3_batch = float(np.mean(adv_per_env ** 3))
                    new_m3 = (1 - tau_v) * new_m3 + tau_v * m3_batch

                    c_batch = None
                    if _one_step or _two_step:
                        # One-step MC covariance: c ≈ E[A(s',a')² · A(s,a)]
                        # Uses buffered advantages from previous sample step
                        if self._prev_adv_per_env is not None and self._prev_valid is not None:
                            valid = self._prev_valid
                            if np.any(valid):
                                c_batch = float(np.mean(
                                    adv_per_env[valid] ** 2 * self._prev_adv_per_env[valid]
                                ))
                    else:
                        # D_ψ-based covariance: c = E[D_ψ(s,a) · A(s,a)]
                        d_psi_per_env = self.algorithm.evaluate_d_psi(obs, action)
                        if d_psi_per_env is not None:
                            d_psi_per_env = np.asarray(d_psi_per_env).ravel()
                            c_batch = float(np.mean(d_psi_per_env * adv_per_env))

                    # Update individual c/m3 EMAs (for diagnostic logging)
                    if c_batch is not None:
                        new_c = (1 - tau_v) * new_c + tau_v * c_batch

                    # Update dimensionless shape EMA: s₂ = (2γc + κ₃) / v^(3/2)
                    # Tracked with slower EMA; only update when c_batch available
                    gamma = self.algorithm.gamma
                    if c_batch is not None:
                        v_raw_safe = max(m2_batch, 1e-8)
                        b_batch = 2.0 * gamma * c_batch + m3_batch
                        s_batch = b_batch / (v_raw_safe ** 1.5)
                        new_shape = (1 - tau_s) * new_shape + tau_s * s_batch
                        new_coeff = (1 - tau_s) * new_coeff + tau_s * b_batch

                    # Two-step: compute τ = γ²d + (γ/2)e + (γ/2)f + (1/6)ρ
                    # then update s₃ = τ / v²
                    if _two_step and c_batch is not None:
                        tau_s3 = float(self.algorithm.shape3_ema_tau)
                        v_raw_safe = max(m2_batch, 1e-8)

                        # d ≈ E[A_{t-2} · A_t² · A_{t-1}]  (two-step buffer)
                        d_batch_val = 0.0
                        d_valid = False
                        if self._prev2_adv_per_env is not None and self._prev2_valid is not None:
                            valid2 = self._prev2_valid
                            if np.any(valid2):
                                A_curr = adv_per_env[valid2]
                                A_prev = self._prev_adv_per_env[valid2]
                                A_prev2 = self._prev2_adv_per_env[valid2]
                                d_batch_val = float(np.mean(A_prev2 * A_curr ** 2 * A_prev))
                                d_valid = True

                        # e ≈ E[A_prev · A_curr³]  (one-step buffer)
                        e_batch = 0.0
                        if self._prev_adv_per_env is not None and self._prev_valid is not None:
                            valid1 = self._prev_valid
                            if np.any(valid1):
                                e_batch = float(np.mean(
                                    self._prev_adv_per_env[valid1] * adv_per_env[valid1] ** 3
                                ))

                        # f ≈ E[(A_prev² - m2_batch) · A_curr²]  (one-step buffer)
                        f_batch = 0.0
                        if self._prev_adv_per_env is not None and self._prev_valid is not None:
                            valid1 = self._prev_valid
                            if np.any(valid1):
                                A_prev = self._prev_adv_per_env[valid1]
                                A_curr = adv_per_env[valid1]
                                f_batch = float(np.mean((A_prev ** 2 - m2_batch) * A_curr ** 2))

                        # ρ = E[A⁴] - 3·E[A²]²  (global excess kurtosis, no buffer)
                        rho_batch = float(np.mean(adv_per_env ** 4)) - 3.0 * m2_batch ** 2

                        # τ = γ²d + (γ/2)e + (γ/2)f + (1/6)ρ
                        tau_batch = (1.0 / 6.0) * rho_batch + 0.5 * gamma * e_batch + 0.5 * gamma * f_batch
                        if d_valid:
                            tau_batch += gamma ** 2 * d_batch_val

                        s3_batch = tau_batch / (v_raw_safe ** 2)
                        new_shape3 = (1 - tau_s3) * new_shape3 + tau_s3 * s3_batch

                # Compute η in raw (unnormalized) space, then convert to code
                # space.  The guidance divides advantages by sqrt(m2), so
                # η_code = η_raw × sqrt(m2).
                kl_budget = self.algorithm.kl_budget
                m2_safe = max(new_m2, 1e-8)

                # KL ceiling (paper §3.1): η_raw ≤ sqrt(2Δ / E[A²])
                eta_kl_raw = float(np.sqrt(2.0 * kl_budget / m2_safe))

                sqrt_v = float(np.sqrt(m2_safe))

                if _two_step:
                    # Compute all three eta candidates independently, take min.

                    # Cubic: 1 + S·x + 3·s₃·x² = 0 where x = η·√v
                    # S = new_shape = (2γc+κ₃)/v^{3/2} (= 2b/v^{3/2})
                    S = new_shape
                    s3 = new_shape3
                    eta_cubic_raw = float('inf')
                    disc = S ** 2 - 12.0 * s3
                    if disc > 0 and abs(s3) > 1e-12:
                        sqrt_disc = float(np.sqrt(disc))
                        x_star = (-S + float(np.sign(S)) * sqrt_disc) / (6.0 * s3)
                        if x_star > 0:
                            eta_cubic_raw = x_star / sqrt_v
                    elif abs(s3) <= 1e-12 and S < -1e-8:
                        # s₃ ≈ 0: cubic degenerates to quadratic
                        eta_cubic_raw = -1.0 / (sqrt_v * S)

                    # Quadratic: η* = -1/(√v · S)
                    eta_quad_raw = float('inf')
                    if new_shape < -1e-8:
                        eta_quad_raw = -1.0 / (sqrt_v * new_shape)

                    eta_raw = min(eta_cubic_raw, eta_quad_raw, eta_kl_raw)

                    # Log eta candidates (convert to code space)
                    self._eta_kl_accum.append(eta_kl_raw * sqrt_v)
                    self._eta_quad_accum.append(eta_quad_raw * sqrt_v if eta_quad_raw != float('inf') else float('nan'))
                    self._eta_cubic_accum.append(eta_cubic_raw * sqrt_v if eta_cubic_raw != float('inf') else float('nan'))
                elif _any_dist_shift:
                    # Quadratic (one-step): η* = -v / B  (direct)  or  -1 / (√v · s₂)  (shape)
                    if getattr(self.algorithm, 'direct_eta_coeff_ema', False):
                        if new_coeff < -1e-8:
                            eta_star_raw = -new_m2 / new_coeff
                        else:
                            eta_star_raw = float('inf')
                    else:
                        if new_shape < -1e-8:
                            eta_star_raw = -1.0 / (sqrt_v * new_shape)
                        else:
                            eta_star_raw = float('inf')
                    eta_raw = min(eta_star_raw, eta_kl_raw)

                    # Log eta candidates (convert to code space)
                    self._eta_kl_accum.append(eta_kl_raw * sqrt_v)
                    self._eta_quad_accum.append(eta_star_raw * sqrt_v if eta_star_raw != float('inf') else float('nan'))
                else:
                    eta_raw = eta_kl_raw

                # Convert to code space (guidance normalises A by sqrt(m2))
                new_eta = eta_raw * float(np.sqrt(m2_safe))

                # Update algorithm state (outside jit)
                self.algorithm.state = self.algorithm.state._replace(
                    advantage_second_moment_ema=jnp.float32(new_m2),
                    advantage_third_moment_ema=jnp.float32(new_m3),
                    dist_shift_covariance_ema=jnp.float32(new_c),
                    dist_shift_shape_ema=jnp.float32(new_shape),
                    dist_shift_coeff_ema=jnp.float32(new_coeff),
                    dist_shift_shape3_ema=jnp.float32(new_shape3),
                    tfg_eta=jnp.float32(new_eta),
                )
            else:
                # --- Episode-averaged EMA: accumulate per-env, update at episode end ---
                self._eema_accumulate_step(adv_per_env, obs, action)

            # Convert per-env to scalars for downstream logging
            _v_per_env_mean = float(np.mean(v_per_env))
            q_agg = float(np.mean(q_per_env))
            q_var = self.algorithm._compute_q_ensemble_var(action, obs)

        if do_timing:
            t_action1 = time.perf_counter()
            self._timing_get_action_s += (t_action1 - t_action0)

        if self.latent_action_space:
            action_latent = np.asarray(action)
            action_env = np.asarray(latent_to_action_normalcdf(jnp.asarray(action_latent), eps=self.latent_action_eps))
        else:
            action_latent = action
            action_env = action

        # In SPI mode, compute all diagnostics on actual env (obs, action) pairs.
        # Accumulated and logged at episode boundaries in _train_soft_pi().
        if self.soft_pi_mode:
            q_live, q_frozen = self.algorithm.spi_q_values(obs, action_latent)
            q_live_mean = float(np.mean(q_live))
            q_frozen_mean = float(np.mean(q_frozen))
            self._spi_accumulator.add("spi/E_Q_live", q_live_mean)
            self._spi_accumulator.add("spi/E_Q_frozen", q_frozen_mean)
            self._spi_accumulator.add("spi/q_shift_delta", q_live_mean - q_frozen_mean)

            # On-policy Bellman MSE (one-step delayed: use prev transition + current Q)
            if self._spi_prev is not None:
                prev = self._spi_prev
                gamma = self.algorithm.gamma
                done_mask = 1.0 - prev["done"].astype(np.float32)
                # TD target = r + γ Q(s', a') * (1 - done)
                td_target_live = prev["reward"] + gamma * q_live * done_mask
                td_target_frozen = prev["reward"] + gamma * q_frozen * done_mask
                self._spi_accumulator.add(
                    "spi/bellman_mse_live",
                    float(np.mean((prev["q_live"] - td_target_live) ** 2)),
                )
                self._spi_accumulator.add(
                    "spi/bellman_mse_frozen",
                    float(np.mean((prev["q_frozen"] - td_target_frozen) ** 2)),
                )

            # Policy distillation loss on actual guided actions (PI step 1+ only)
            if self._spi_pi_step > 0:
                ploss_key = jax.random.fold_in(sample_key, 0x5D1)
                ploss = self.algorithm.evaluate_policy_loss(ploss_key, obs, action_latent)
                self._spi_accumulator.add("spi/policy_loss", ploss)

        self._assert_finite(
            "sample.obs",
            obs,
            sample_step=int(sl.sample_step),
            update_step=int(self.update_log.update_step),
        )
        self._assert_finite(
            "sample.action_latent",
            action_latent,
            sample_step=int(sl.sample_step),
            update_step=int(self.update_log.update_step),
        )
        self._assert_finite(
            "sample.action_env",
            action_env,
            sample_step=int(sl.sample_step),
            update_step=int(self.update_log.update_step),
        )
        
        # SARSA-style tracking: update previous experience's next_action with current action
        if self.track_next_action and self._prev_buffer_indices is not None:
            # Only update for non-terminal previous transitions
            if self._prev_dones is not None:
                # For vectorized envs, _prev_dones is a boolean array
                # For single env, _prev_dones is a scalar boolean
                prev_dones = np.atleast_1d(self._prev_dones)
                prev_indices = np.atleast_1d(self._prev_buffer_indices)
                action_arr = np.atleast_2d(action_latent) if np.asarray(action_latent).ndim == 1 else action_latent
                
                # Update next_action only for non-terminal transitions
                non_terminal_mask = ~prev_dones
                if np.any(non_terminal_mask):
                    indices_to_update = prev_indices[non_terminal_mask]
                    actions_to_set = action_arr[non_terminal_mask] if self.is_vec else action_latent
                    # Use buffer's replace method to update only next_action field
                    # Create a partial Experience with only next_action set
                    update_exp = Experience(
                        obs=None, action=None, reward=None, done=None, next_obs=None,
                        next_action=actions_to_set if self.is_vec else action_latent,
                    )
                    self.buffer.replace(indices_to_update, update_exp)
        
        next_obs, reward, terminated, truncated, info = self.env.step(action_env)

        self._assert_finite(
            "sample.next_obs",
            next_obs,
            sample_step=int(sl.sample_step),
            update_step=int(self.update_log.update_step),
        )
        self._assert_finite(
            "sample.reward",
            reward,
            sample_step=int(sl.sample_step),
            update_step=int(self.update_log.update_step),
        )

        # SPI: store transition data for next-step Bellman MSE computation.
        # q_live/q_frozen were computed above (before env.step); reward/done come
        # from env.step.  At episode boundaries we reset to avoid cross-episode
        # Bellman errors.
        if self.soft_pi_mode:
            done_arr = np.asarray(terminated) | np.asarray(truncated)
            self._spi_prev = {
                "q_live": q_live,
                "q_frozen": q_frozen,
                "reward": np.asarray(reward, dtype=np.float32),
                "done": done_arr,
            }

        a_env_arr = np.asarray(action_env)
        info = dict(info)
        if self.is_vec:
            info["action_mean"] = np.mean(a_env_arr, axis=-1)
            info["action_var"] = np.var(a_env_arr, axis=-1)
            info["action_clip_frac"] = np.mean((np.abs(a_env_arr) > 0.99).astype(np.float32), axis=-1)
        else:
            info["action_mean"] = float(np.mean(a_env_arr))
            info["action_var"] = float(np.var(a_env_arr))
            info["action_clip_frac"] = float(np.mean(np.abs(a_env_arr) > 0.99))
        if q_agg is not None:
            info["q_agg"] = float(q_agg)
        if q_var is not None:
            info["q_var"] = float(q_var)
        if _v_per_env_mean is not None:
            info["v_value"] = _v_per_env_mean
        elif hasattr(self.algorithm, 'evaluate_value'):
            v_value = self.algorithm.evaluate_value(obs)
            if v_value is not None:
                info["v_value"] = v_value

        experience = Experience.create(obs, action_latent, reward, terminated, truncated, next_obs, info)
        
        # Track buffer indices before adding (for next iteration's next_action update)
        if self.track_next_action:
            if self.is_vec:
                batch_size = np.atleast_1d(reward).shape[0]
                # Indices where this batch will be stored
                current_indices = np.arange(self.buffer.ptr, self.buffer.ptr + batch_size) % self.buffer.max_len
            else:
                current_indices = np.array([self.buffer.ptr])
            self._prev_buffer_indices = current_indices
            self._prev_dones = np.atleast_1d(terminated) | np.atleast_1d(truncated)
        
        # Record buffer position before adding (for episode tracking)
        _old_ptr = self.buffer.ptr

        if self.is_vec:
            self.buffer.add_batch(experience)
        else:
            self.buffer.add(experience)

        # Episode tracking for equal_episode_weighting (buffer sampling)
        if self.equal_episode_weighting:
            if self.is_vec:
                _bs = np.atleast_1d(reward).shape[0]
                self._ep_record_transition(_old_ptr, _bs, terminated, truncated)
            else:
                self._ep_record_transition_single(_old_ptr, terminated, truncated)

        # In parallel, advance the validation environment (if any) and record
        # its transitions into the dedicated validation buffer. This maintains
        # a stream of validation experience generated under the current
        # policy, without contaminating the training buffer.
        if self.val_env is not None and self.val_buffer is not None and self.validation_ratio > 0.0:
            # Lazily initialize validation observations if warmup did not run
            # or if they were reset.
            if not hasattr(self, "val_obs"):
                self.val_obs, _ = self.val_env.reset()

            # Use an independent policy action for the validation environment
            # based on its own observations to respect potential differences
            # in vectorization layouts.
            val_key = jax.random.fold_in(sample_key, 1)
            _val_action_out = self.algorithm.get_action(val_key, self.val_obs)
            val_action = _val_action_out[0] if isinstance(_val_action_out, tuple) else _val_action_out
            if self.latent_action_space:
                val_action_latent = np.asarray(val_action)
                val_action_env = np.asarray(latent_to_action_normalcdf(jnp.asarray(val_action_latent), eps=self.latent_action_eps))
            else:
                val_action_latent = val_action
                val_action_env = val_action
            val_next_obs, val_reward, val_terminated, val_truncated, val_info = self.val_env.step(val_action_env)

            val_experience = Experience.create(
                self.val_obs,
                val_action_latent,
                val_reward,
                val_terminated,
                val_truncated,
                val_next_obs,
                val_info,
            )
            if isinstance(self.val_env.unwrapped, VectorEnv):
                self.val_buffer.add_batch(val_experience)
            else:
                self.val_buffer.add(val_experience)

            if np.any(val_terminated) or np.any(val_truncated):
                self.val_obs, _ = self.val_env.reset()
            else:
                self.val_obs = val_next_obs

        any_done = sl.add(reward, terminated, truncated, info)

        # Episodes are logged the moment they finish, at their per-env
        # sub-step env_step = num_envs * (num_batched_env_steps - 1) + env_idx + 1
        # (computed inside sl.add). One wandb.log call per finished episode,
        # with the wandb internal step set to that env_step.
        ep_return_key = f"episode_return/{sl.env_name}"
        for env_step, ret in sl.take_pending_episode_returns():
            self.add_scalar(ep_return_key, ret, env_step)

        # Periodic flush of averaged accumulator metrics + global EMAs at
        # the current sample_step. Independent of whether episodes ended.
        if self.sample_log_interval.check(sl.sample_step):
            sl.log_accumulator(self.add_scalar)
            step = sl.sample_step
            if (
                hasattr(self.algorithm, "on_policy_ema")
                and self.algorithm.on_policy_ema
                and hasattr(self.algorithm, "state")
            ):
                st = self.algorithm.state
                self.add_scalar("Global_EMAs/Advantage_second_moment", float(st.advantage_second_moment_ema), step)
                self.add_scalar("Global_EMAs/Advantage_third_moment", float(st.advantage_third_moment_ema), step)
                self.add_scalar("Global_EMAs/Distribution_shift_covariance", float(st.dist_shift_covariance_ema), step)
                self.add_scalar("Global_EMAs/Distribution_shift_shape", float(st.dist_shift_shape_ema), step)
                self.add_scalar("Global_EMAs/Distribution_shift_coefficient", float(st.dist_shift_coeff_ema), step)
                self.add_scalar("Global_EMAs/Distribution_shift_shape3", float(st.dist_shift_shape3_ema), step)
            if self._eta_accum:
                avg_eta = sum(self._eta_accum) / len(self._eta_accum)
                self.add_scalar("Global_EMAs/eta", avg_eta, step)
                self._eta_accum.clear()
            if self._tfg_eta_accum:
                avg_tfg_eta = sum(self._tfg_eta_accum) / len(self._tfg_eta_accum)
                self.add_scalar("Global_EMAs/tfg_eta", avg_tfg_eta, step)
                self._tfg_eta_accum.clear()
            if self._eta_kl_accum:
                avg_eta_kl = sum(self._eta_kl_accum) / len(self._eta_kl_accum)
                self.add_scalar("Global_EMAs/eta_kl_ceiling", avg_eta_kl, step)
                self.add_scalar("Global_EMAs/eta_kl_budget", avg_eta_kl, step)
                self._eta_kl_accum.clear()
            if self._eta_quad_accum:
                vals = [v for v in self._eta_quad_accum if not math.isnan(v)]
                if vals:
                    avg_eta_quad = sum(vals) / len(vals)
                    self.add_scalar("Global_EMAs/eta_quadratic", avg_eta_quad, step)
                    self.add_scalar("Global_EMAs/eta_one_step_dist_shift", avg_eta_quad, step)
                self._eta_quad_accum.clear()
            if self._eta_cubic_accum:
                vals = [v for v in self._eta_cubic_accum if not math.isnan(v)]
                if vals:
                    avg_eta_cubic = sum(vals) / len(vals)
                    self.add_scalar("Global_EMAs/eta_cubic", avg_eta_cubic, step)
                    self.add_scalar("Global_EMAs/eta_two_step_dist_shift", avg_eta_cubic, step)
                self._eta_cubic_accum.clear()

        self.progress.update(sl.sample_step - self.progress.n)

        # Episode-averaged EMA: finalize EMA for envs whose episodes just ended
        if self.equal_episode_weighting and any_done and _on_policy_ema:
            _done_arr = np.atleast_1d(np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool))
            for _j in range(len(_done_arr)):
                if _done_arr[_j]:
                    self._eema_finalize_episode(_j)

        if any_done:
            obs, _ = self.env.reset()
            # Reset open-loop execution state at episode boundaries
            if hasattr(self.algorithm, 'reset_open_loop'):
                self.algorithm.reset_open_loop()
        else:
            obs = next_obs

        # Buffer advantages for one/two-step dist-shift covariance estimation.
        # Must happen after env.step so we know which envs reset.
        if _adv_for_one_step is not None:
            done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
            valid_now = ~np.atleast_1d(done)
            # Shift: prev → prev2 (AND validity masks across both steps)
            self._prev2_adv_per_env = self._prev_adv_per_env
            self._prev2_valid = (self._prev_valid & valid_now) if self._prev_valid is not None else None
            self._prev_adv_per_env = _adv_for_one_step.copy()
            self._prev_valid = valid_now

        # Accumulate effective eta for averaged logging
        if hasattr(self.algorithm, 'get_current_tfg_eta'):
            eta, tfg_eta = self._current_eta_metrics()
            self._eta_accum.append(float(eta))
            self._tfg_eta_accum.append(float(tfg_eta))

        if do_timing:
            t1 = time.perf_counter()
            self._timing['sample_total'] += (t1 - t0)
            self._timing_sample_n += 1

        return obs

    def update(self, update_key: jax.Array):
        ul = self.update_log

        do_timing = self.timing_log_every > 0
        if do_timing:
            t0 = time.perf_counter()

        # The UpdateLog increments its internal step counter when we call
        # ul.add(info). For gating logic (e.g., how often to log or run
        # hypergradient updates), we base decisions on the *next* update step
        # index, corresponding to the update we are about to perform.
        current_update_step = ul.update_step
        next_update_step = current_update_step + 1

        supervised_steps = getattr(self.algorithm, "supervised_steps", 1)

        key = update_key

        # Sample training batch from the main replay buffer.
        # For H-step policy training, sample sequences instead of individual transitions.
        H_train = getattr(self.algorithm, "H_train", 1)
        if do_timing:
            t_batch0 = time.perf_counter()
        train_idx, val_idx = self._get_train_val_index_sets()
        if H_train > 1 and isinstance(self.buffer, TreeBuffer):
            # Sample H-step sequences for policy training
            train_data = self.buffer.sample_sequences(self.batch_size, H_train)
        elif self.equal_episode_weighting and isinstance(self.buffer, TreeBuffer):
            # Episode-weighted: sample episode uniformly, then transition within
            ep_indices = self._ep_sample_indices(self.batch_size)
            train_data = self.buffer.gather_indices(ep_indices)
        elif train_idx is None:
            train_data = self.buffer.sample(self.batch_size)
        else:
            train_data = self._sample_from_index_set(train_idx, self.batch_size)
        if do_timing:
            t_batch1 = time.perf_counter()
            self._timing_batch_s += (t_batch1 - t_batch0)

        self._assert_finite(
            "update.train_data",
            train_data,
            sample_step=int(self.sample_log.sample_step),
            update_step=int(next_update_step),
        )

        # Decide whether we need a validation batch for this step. We only
        # draw validation samples when they are actually required, either for
        # a hypergradient update or for periodic validation logging. When a
        # dedicated validation buffer is present, validation batches are drawn
        # from that buffer; otherwise we fall back to the within-buffer
        # temporal split defined by _get_train_val_index_sets.
        val_data = None
        if self.val_buffer is not None:
            have_val_indices = self.use_validation and len(self.val_buffer) > 0
        else:
            have_val_indices = (
                self.use_validation
                and val_idx is not None
                and len(val_idx) > 0
            )
        need_val_for_hyper = (
            self.use_hypergrad
            and have_val_indices
            and (next_update_step % self.hypergrad_period == 0)
        )
        current_env_step = self.sample_log.sample_step
        log_this_step = (current_env_step - self._last_update_log_env_step >= self.update_log_n_env_steps)

        need_val_for_logging = have_val_indices and log_this_step
        if need_val_for_hyper or need_val_for_logging:
            if self.val_buffer is not None:
                # Sample from the dedicated validation buffer.
                val_data = self.val_buffer.sample(self.val_batch_size)
            else:
                val_data = self._sample_from_index_set(val_idx, self.val_batch_size)

        if val_data is not None:
            self._assert_finite(
                "update.val_data",
                val_data,
                sample_step=int(self.sample_log.sample_step),
                update_step=int(next_update_step),
            )

        # Main training / hypergradient path
        if do_timing:
            t_alg0 = time.perf_counter()
        if (
            self.use_hypergrad
            and val_data is not None
            and hasattr(self.algorithm, "hyper_update")
            and (next_update_step % self.hypergrad_period == 0)
        ):
            # Let the algorithm perform a hypergradient-aware update that
            # consumes both train and validation batches. Any validation loss
            # used internally for hypergradients is returned here but we do
            # not rely on it for logging; logging always uses
            # compute_validation_metrics below for consistency.
            info, array_info, _ = self.algorithm.hyper_update(update_key, train_data, val_data)
        else:
            # Standard training update (optionally with supervised warmup)
            if supervised_steps > 1 and hasattr(self.algorithm, "update_supervised"):
                for _ in range(supervised_steps - 1):
                    key, subkey = jax.random.split(key)
                    self.algorithm.update_supervised(subkey, train_data)

                key, subkey = jax.random.split(key)
                info, array_info = self.algorithm.update(subkey, train_data)
            else:
                info, array_info = self.algorithm.update(update_key, train_data)
        if do_timing:
            t_alg1 = time.perf_counter()
            self._timing_alg_update_s += (t_alg1 - t_alg0)

        self._assert_finite(
            "update.info",
            info,
            sample_step=int(self.sample_log.sample_step),
            update_step=int(next_update_step),
        )
        self._assert_finite(
            "update.array_info",
            array_info,
            sample_step=int(self.sample_log.sample_step),
            update_step=int(next_update_step),
        )

        # Separately compute validation metrics if we have a validation batch
        # and the algorithm exposes a validation hook. To avoid doubling the
        # per-step training cost, we only compute validation metrics at the
        # same interval as other logged scalars, and we always use
        # compute_validation_metrics so that validation logging behaves the
        # same regardless of whether hypergrad is enabled.
        val_info = None
        if val_data is not None and log_this_step:
            val_key = jax.random.fold_in(update_key, 1)
            val_info = self.algorithm.compute_validation_metrics(val_key, val_data)

        if val_info is not None:
            self._assert_finite(
                "update.val_info",
                val_info,
                sample_step=int(self.sample_log.sample_step),
                update_step=int(next_update_step),
            )

        if self.soft_pi_mode:
            # In SPI mode, all diagnostics are computed on actual env actions
            # in sample() and accumulated in _spi_accumulator.  The regular
            # UpdateLog is bypassed (empty dict keeps step counter advancing).
            ul.add({})

            if do_timing:
                t1 = time.perf_counter()
                self._timing_update_s += (t1 - t0)
                self._timing_update_n += 1
            return

        # Accumulate array metrics for averaging at log time
        for k, v in array_info.items():
            self._array_accum.setdefault(k, []).append(np.asarray(v))

        ul.add(info)

        if do_timing:
            t1 = time.perf_counter()
            self._timing_update_s += (t1 - t0)
            self._timing_update_n += 1

        if log_this_step:
            self._last_update_log_env_step = current_env_step
            current_step = self.sample_log.sample_step

            # Average accumulated arrays over updates since last log
            averaged_arrays = {}
            for k, v_list in self._array_accum.items():
                averaged_arrays[k] = sum(v_list) / len(v_list)
            self._array_accum.clear()
            self.add_arrays(averaged_arrays, current_step)

            ul.log(
                lambda tag, value, _step: self.add_scalar(
                    tag,
                    value,
                    current_step,
                )
            )

            # Log validation metrics, if available, under "validation/".
            if val_data is not None and val_info is not None:
                for tag, value in val_info.items():
                    self.add_scalar(f"validation/{tag}", value, current_step)

            # When hypergradient adaptation is enabled and the underlying
            # algorithm exposes lr-scale hyperparameters, log the total
            # effective learning rates under a separate "hyperparameters/"
            # section for easier monitoring.
            if (
                self.use_hypergrad
                and hasattr(self.algorithm, "state")
                and hasattr(self.algorithm.state, "hyper")
            ):
                hyper = self.algorithm.state.hyper
                try:
                    log_scale_policy = float(hyper.log_lr_scale_policy)
                    log_scale_dyn = float(hyper.log_lr_scale_dynamics)
                    log_scale_reward = float(hyper.log_lr_scale_reward)
                    log_scale_value = float(hyper.log_lr_scale_value)

                    # If the algorithm exposes base learning rates, log the
                    # total effective learning rate for each head.
                    if hasattr(self.algorithm, "policy_lr"):
                        eff_policy = float(self.algorithm.policy_lr) * math.exp(log_scale_policy)
                        self.add_scalar(
                            "hyperparameters/eff_lr_policy",
                            eff_policy,
                            current_step,
                        )
                    if hasattr(self.algorithm, "dyn_lr"):
                        eff_dyn = float(self.algorithm.dyn_lr) * math.exp(log_scale_dyn)
                        self.add_scalar(
                            "hyperparameters/eff_lr_dynamics",
                            eff_dyn,
                            current_step,
                        )
                    if hasattr(self.algorithm, "reward_lr"):
                        eff_reward = float(self.algorithm.reward_lr) * math.exp(log_scale_reward)
                        self.add_scalar(
                            "hyperparameters/eff_lr_reward",
                            eff_reward,
                            current_step,
                        )
                    if hasattr(self.algorithm, "value_lr"):
                        eff_value = float(self.algorithm.value_lr) * math.exp(log_scale_value)
                        self.add_scalar(
                            "hyperparameters/eff_lr_value",
                            eff_value,
                            current_step,
                        )
                except TypeError:
                    # In case any of the hyper fields are not simple
                    # scalars, skip logging rather than failing.
                    pass


    def train(self, key: jax.Array):
        key, warmup_key = jax.random.split(key)

        # Try to resume from checkpoint first (requeueable jobs only)
        if self._checkpoint_enabled and self.load_checkpoint():
            # Reset environment to get fresh obs — we can't serialize env
            # state, so the first transition after resume will be from a
            # reset state.  For off-policy with a large buffer this is
            # negligible (one transition out of ~1M).
            obs, _ = self.env.reset()
            # Advance progress bar to match restored state
            self.progress.update(self.sample_log.sample_step)
            print(f"[Checkpoint] Skipping warmup — resuming training from step {self.sample_log.sample_step}")
        else:
            # Warmup internally resets environments and returns the latest
            # training observations after the replay buffer has been populated.
            obs = self.warmup(warmup_key)

        if self.soft_pi_mode:
            obs = self._train_soft_pi(key, obs)
        else:
            obs = self._train_standard(key, obs)

    def _train_standard(self, key: jax.Array, obs):
        """Original training loop (no soft-PI)."""
        iter_key_fn = create_iter_key_fn(key, self.sample_per_iteration, self.update_per_iteration)
        sl, ul = self.sample_log, self.update_log

        # Checkpoint every 50K env steps
        checkpoint_every = 50_000
        last_checkpoint_step = sl.sample_step

        self.progress.unpause()
        while sl.sample_step <= self.total_step:
            # Check if preempted (SIGTERM received on a requeueable job)
            if self._checkpoint_enabled and self._preempted:
                print(f"[Checkpoint] Preemption detected at step {sl.sample_step} — checkpointing...")
                self.save_checkpoint()
                return obs

            sample_keys, update_keys = iter_key_fn(sl.sample_step)

            for i in range(self.sample_per_iteration):
                obs = self.sample(sample_keys[i], obs)

            for i in range(self.update_per_iteration):
                self.update(update_keys[i])

            # Update tfg_eta according to log-linear schedule
            self._update_tfg_eta_schedule(sl.sample_step)

            # Periodic KL(π_tilt || π_0) logging
            if sl.sample_step % self.log_md_kl_every == 0:
                self._log_md_kl(sample_keys[0], obs, sl.sample_step)

            if self.timing_log_every > 0 and (sl.sample_step % self.timing_log_every == 0):
                self._log_timing(sl.sample_step)

            self._flush_wandb()

            # Periodic checkpoint (requeueable jobs only)
            if self._checkpoint_enabled and sl.sample_step - last_checkpoint_step >= checkpoint_every:
                self.save_checkpoint()
                last_checkpoint_step = sl.sample_step

        return obs

    def _train_soft_pi(self, key: jax.Array, obs):
        """Soft policy iteration training loop.

        PI step 0: Q-only (policy frozen, base policy collects data).
        PI step 1+: Q + policy training (policy distills guided actions).

        iterations_per_pi_step counts training iterations (each iteration =
        sample_per_iteration env steps + update_per_iteration gradient steps).
        SPI diagnostic metrics are accumulated and logged at episode boundaries.
        """
        iter_key_fn = create_iter_key_fn(key, self.sample_per_iteration, self.update_per_iteration)
        sl, ul = self.sample_log, self.update_log

        prev_episode_count = sl.sample_episode

        self.progress.unpause()
        for pi_step in range(self.num_pi_steps):
            # --- Start of policy improvement step ---
            self.algorithm.freeze_guidance_q()
            pi_step_start_sample = sl.sample_step

            # PI step 0: Q-only training (policy frozen).
            # PI step 1+: enable policy training (distill guided actions).
            self._spi_pi_step = pi_step
            if pi_step == 0:
                self.algorithm.set_train_policy(False)
            else:
                self.algorithm.set_train_policy(True)

            print(f"[soft-PI] PI step {pi_step}/{self.num_pi_steps}, "
                  f"policy={'ON' if pi_step > 0 else 'OFF'}, "
                  f"sample_step={sl.sample_step}")

            self.add_scalar("pi_step/id", pi_step, sl.sample_step)

            for iteration_in_step in range(self.iterations_per_pi_step):
                sample_keys, update_keys = iter_key_fn(sl.sample_step)

                for i in range(self.sample_per_iteration):
                    obs = self.sample(sample_keys[i], obs)

                for i in range(self.update_per_iteration):
                    self.update(update_keys[i])

                # At episode boundaries, log accumulated SPI diagnostics
                if sl.sample_episode > prev_episode_count:
                    current_step = sl.sample_step
                    self._spi_accumulator.log(
                        lambda k, v: self.add_scalar(k, v, current_step)
                    )
                    self._spi_accumulator.reset()
                    self.add_scalar("pi_step/id", pi_step, current_step)
                    self.add_scalar("pi_step/iteration", iteration_in_step, current_step)
                    prev_episode_count = sl.sample_episode

                # Update tfg_eta according to log-linear schedule
                self._update_tfg_eta_schedule(sl.sample_step)

                # Periodic KL(π_tilt || π_0) logging
                if sl.sample_step % self.log_md_kl_every == 0:
                    self._log_md_kl(sample_keys[0], obs, sl.sample_step)

                if self.timing_log_every > 0 and (sl.sample_step % self.timing_log_every == 0):
                    self._log_timing(sl.sample_step)

                self._flush_wandb()


            # --- End of policy improvement step ---
            env_steps_this_pi = sl.sample_step - pi_step_start_sample
            self.add_scalar("pi_step/total_env_steps", env_steps_this_pi, sl.sample_step)
            self._flush_wandb()
            print(f"[soft-PI] Completed PI step {pi_step}, "
                  f"ran {env_steps_this_pi} env steps ({iteration_in_step} iters), "
                  f"sample_step={sl.sample_step}")

        return obs

    def _log_timing(self, step: int) -> None:
        if self._timing_sample_n <= 0 and self._timing_update_n <= 0:
            return

        sample_ms = 1000.0 * self._timing_sample_s / max(1, self._timing_sample_n)
        update_ms = 1000.0 * self._timing_update_s / max(1, self._timing_update_n)
        action_ms = 1000.0 * self._timing_get_action_s / max(1, self._timing_sample_n)
        env_ms = 1000.0 * self._timing_env_step_s / max(1, self._timing_sample_n)
        batch_ms = 1000.0 * self._timing_batch_s / max(1, self._timing_update_n)
        alg_ms = 1000.0 * self._timing_alg_update_s / max(1, self._timing_update_n)

        total = self._timing_sample_s + self._timing_update_s
        sample_frac = float(self._timing_sample_s / total) if total > 0 else 0.0
        update_frac = float(self._timing_update_s / total) if total > 0 else 0.0

        self.add_scalar("timing/sample_ms", float(sample_ms), int(step))
        self.add_scalar("timing/update_ms", float(update_ms), int(step))
        self.add_scalar("timing/get_action_ms", float(action_ms), int(step))
        self.add_scalar("timing/env_step_ms", float(env_ms), int(step))
        self.add_scalar("timing/batch_sample_ms", float(batch_ms), int(step))
        self.add_scalar("timing/alg_update_ms", float(alg_ms), int(step))
        self.add_scalar("timing/sample_frac", float(sample_frac), int(step))
        self.add_scalar("timing/update_frac", float(update_frac), int(step))

        self._timing_sample_s = 0.0
        self._timing_update_s = 0.0
        self._timing_get_action_s = 0.0
        self._timing_env_step_s = 0.0
        self._timing_batch_s = 0.0
        self._timing_alg_update_s = 0.0
        self._timing_sample_n = 0
        self._timing_update_n = 0
        self._timing_last_logged_sample_step = int(step)

    def _buffer_wandb(self, data: dict, step: int):
        """Add metrics to the pending wandb buffer, auto-flushing if step changes."""
        if self._wandb_pending_step is not None and self._wandb_pending_step != step:
            self._flush_wandb()
        self._wandb_pending.update(data)
        self._wandb_pending_step = step

    def _flush_wandb(self):
        """Flush all pending wandb metrics in a single wandb.log call."""
        if self._wandb_pending and self._wandb_pending_step is not None:
            wandb.log(self._wandb_pending, step=self._wandb_pending_step)
            self._wandb_pending = {}
            self._wandb_pending_step = None

    def add_scalar(self, tag: str, value: float, step: int):
        if tag.startswith("episode_return/"):
            self._maybe_reduce_tfg_eta_on_plateau(float(value), int(step))
        self.last_metrics[tag] = value
        # All metrics — episode returns and others alike — go through the
        # same step-keyed buffer. wandb's internal _step ends up tracking
        # the env step we pass: per-env sub-step for episode_return, full
        # sample_step for everything else. Monotonicity is preserved as
        # long as episodes are flushed before per-iteration metrics, which
        # is enforced in the sample() loop.
        self._buffer_wandb({tag: value}, step)

    def _maybe_reduce_tfg_eta_on_plateau(self, episode_return: float, step: int) -> None:
        if not np.isfinite(self.tfg_patience):
            return
        if not (self.tfg_reduction_factor < 1.0):
            return
        if not hasattr(self.algorithm, "get_current_tfg_eta"):
            return
        if not hasattr(self.algorithm, "set_tfg_eta"):
            return

        current_lambda = float(self.algorithm.get_current_tfg_eta())
        if self._tfg_plateau_lambda is None or (current_lambda != self._tfg_plateau_lambda):
            self._tfg_plateau_lambda = current_lambda
            self._tfg_plateau_best_return = -float("inf")
            self._tfg_plateau_best_step = step
            self._tfg_plateau_bad_count = 0

        if episode_return > self._tfg_plateau_best_return:
            self._tfg_plateau_best_return = float(episode_return)
            self._tfg_plateau_best_step = int(step)
            self._tfg_plateau_bad_count = 0
            return

        self._tfg_plateau_bad_count += 1

        if self._tfg_plateau_bad_count < int(self.tfg_patience):
            return

        new_lambda = max(0.0, current_lambda * float(self.tfg_reduction_factor))
        if new_lambda == current_lambda:
            return

        self.algorithm.set_tfg_eta(new_lambda)

        tag = "hyperparameters/tfg_eta"
        self.last_metrics[tag] = float(new_lambda)
        self._buffer_wandb({tag: float(new_lambda)}, step)

        self._tfg_plateau_lambda = float(new_lambda)
        self._tfg_plateau_best_return = -float("inf")
        self._tfg_plateau_best_step = step
        self._tfg_plateau_bad_count = 0

    def _update_tfg_eta_schedule(self, step: int) -> None:
        """Update tfg_eta according to log-linear schedule over training."""
        if not self._tfg_schedule_enabled:
            return
        if not hasattr(self.algorithm, "set_tfg_eta"):
            return

        # Compute progress t in [0, 1] from start_step to total_step
        effective_start = max(self.start_step, 1)
        if step <= effective_start:
            t = 0.0
        elif step >= self.total_step:
            t = 1.0
        else:
            t = (step - effective_start) / max(self.total_step - effective_start, 1)

        # Log-linear interpolation: log(lambda_t) = (1-t)*log(start) + t*log(end)
        # lambda_t = start^(1-t) * end^t
        # Handle edge case where either endpoint is 0
        start = self.tfg_eta_start
        end = self.tfg_eta_end
        if start > 0 and end > 0:
            new_lambda = (start ** (1.0 - t)) * (end ** t)
        elif start == 0 and end == 0:
            new_lambda = 0.0
        else:
            # Fallback to linear interpolation if one endpoint is 0
            new_lambda = (1.0 - t) * start + t * end

        self.algorithm.set_tfg_eta(float(new_lambda))

    def _log_md_kl(self, key: jax.Array, obs: np.ndarray, step: int) -> None:
        """Log KL(π_tilt || π_0) for mirror descent monitoring."""
        if not hasattr(self.algorithm, "estimate_md_kl"):
            return

        kl_key = jax.random.fold_in(key, step)
        kl = self.algorithm.estimate_md_kl(kl_key, obs, n_samples=16)

        tag = "mirror_descent/kl_tilt_base"
        self.last_metrics[tag] = float(kl)
        self._buffer_wandb({tag: float(kl)}, step)

    def add_arrays(self, array_info, step):
        if not array_info:
            return

        snr = getattr(self.algorithm, "_snr", None)
        if snr is not None:
            log2_snr = np.log2(np.maximum(snr, 1e-12))

        # Log MALA per-level metrics as matplotlib images so that wandb
        # provides a step slider for each chart.  Enable "Sync slider by
        # key (Step)" in workspace settings to compare across runs.
        mala_keys = ("MALA/eta_scale", "MALA/acceptance_rate", "MALA/clip_frac")
        has_mala = any(k in array_info for k in mala_keys)
        if has_mala and snr is not None:
            mala_metrics = {
                "MALA/eta_scale": np.asarray(array_info.get("MALA/eta_scale", np.full_like(log2_snr, np.nan))),
                "MALA/acceptance_rate": np.asarray(array_info.get("MALA/acceptance_rate", np.full_like(log2_snr, np.nan))),
                "MALA/clip_frac": np.asarray(array_info.get("MALA/clip_frac", np.full_like(log2_snr, np.nan))),
            }
            log_dict = {}
            for metric_name, values in mala_metrics.items():
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
                self._buffer_wandb(log_dict, step)

        # Log remaining non-MALA arrays as raw tables
        mala_key_set = set(mala_keys)
        for tag, value in array_info.items():
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
            self._buffer_wandb({tag: table}, step)

    # ------------------------------------------------------------------
    # Checkpointing for SLURM preemption resilience
    # ------------------------------------------------------------------

    def _find_checkpoint(self) -> Optional[Path]:
        """Return the checkpoint directory if a valid checkpoint exists.

        Falls back to checkpoints_old/ or checkpoints_tmp/ in case a
        prior save was interrupted during the atomic swap.
        """
        for candidate in [self._checkpoint_dir,
                          self._checkpoint_dir.parent / "checkpoints_old",
                          self._checkpoint_dir.parent / "checkpoints_tmp"]:
            if candidate.exists() and (candidate / "meta.pkl").exists():
                return candidate
        return None

    def _install_signal_handler(self):
        """Install SIGTERM handler that triggers a checkpoint on preemption."""
        def _handle_sigterm(signum, frame):
            print("[Checkpoint] SIGTERM received — saving checkpoint before exit...")
            self._preempted = True
        signal.signal(signal.SIGTERM, _handle_sigterm)

    def save_checkpoint(self):
        """Save full trainer state for resumption after preemption.

        Uses atomic write: saves to a temp directory first, then swaps
        into the final location.  This prevents a partial write from
        corrupting an existing valid checkpoint.
        """
        import shutil

        final_dir = self._checkpoint_dir
        tmp_dir = final_dir.parent / "checkpoints_tmp"
        old_dir = final_dir.parent / "checkpoints_old"

        # Clean up any leftover temp dir from a prior interrupted save
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # 1. wandb run ID — persist both inside the checkpoint and at the
        #    log_path level so it survives even if the atomic swap is
        #    interrupted between renames.
        run_id = wandb.run.id
        with open(tmp_dir / "wandb_run_id.txt", "w") as f:
            f.write(run_id)
        with open(self.log_path / "wandb_run_id.txt", "w") as f:
            f.write(run_id)

        # 2. Algorithm state (params + optimizer)
        self.algorithm.save(str(tmp_dir / "algorithm.pkl"))

        # 3. Replay buffer
        self.buffer.save(tmp_dir / "buffer.pkl")

        # 4. Trainer metadata — written last as "commit" marker
        meta = {
            "sample_step": self.sample_log.sample_step,
            "sample_episode": self.sample_log.sample_episode,
            "update_step": self.update_log.update_step,
            "last_update_log_env_step": self._last_update_log_env_step,
            "eta_accum": list(self._eta_accum),
            "tfg_eta_accum": list(self._tfg_eta_accum),
            "eta_kl_accum": list(self._eta_kl_accum),
            "eta_quad_accum": list(self._eta_quad_accum),
            "eta_cubic_accum": list(self._eta_cubic_accum),
            "tfg_plateau_lambda": self._tfg_plateau_lambda,
            "tfg_plateau_best_return": self._tfg_plateau_best_return,
            "tfg_plateau_best_step": self._tfg_plateau_best_step,
            "tfg_plateau_bad_count": self._tfg_plateau_bad_count,
            "prev_adv_per_env": self._prev_adv_per_env,
            "prev_valid": self._prev_valid,
        }
        with open(tmp_dir / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)

        # Atomic swap: old → delete, current → old, tmp → current
        if old_dir.exists():
            shutil.rmtree(old_dir)
        if final_dir.exists():
            final_dir.rename(old_dir)
        tmp_dir.rename(final_dir)
        if old_dir.exists():
            shutil.rmtree(old_dir)

        print(f"[Checkpoint] Saved at sample_step={self.sample_log.sample_step}, "
              f"update_step={self.update_log.update_step}")

    def load_checkpoint(self) -> bool:
        """Load trainer state from checkpoint. Returns True if checkpoint was loaded."""
        ckpt_dir = self._find_checkpoint()
        if ckpt_dir is None:
            return False

        print("[Checkpoint] Loading checkpoint...")

        # 1. Algorithm state
        self.algorithm.load(str(ckpt_dir / "algorithm.pkl"))

        # 2. Replay buffer
        self.buffer.load(ckpt_dir / "buffer.pkl")

        # 3. Trainer metadata
        with open(ckpt_dir / "meta.pkl", "rb") as f:
            meta = pickle.load(f)

        self.sample_log.sample_step = meta["sample_step"]
        self.sample_log.sample_episode = meta["sample_episode"]
        self.update_log.update_step = meta["update_step"]
        self._last_update_log_env_step = meta["last_update_log_env_step"]
        self._eta_accum = meta.get("eta_accum", [])
        self._tfg_eta_accum = meta.get("tfg_eta_accum", [])
        self._eta_kl_accum = meta.get("eta_kl_accum", [])
        self._eta_quad_accum = meta.get("eta_quad_accum", [])
        self._eta_cubic_accum = meta.get("eta_cubic_accum", [])
        self._tfg_plateau_lambda = meta.get("tfg_plateau_lambda")
        self._tfg_plateau_best_return = meta.get("tfg_plateau_best_return", -float("inf"))
        self._tfg_plateau_best_step = meta.get("tfg_plateau_best_step", 0)
        self._tfg_plateau_bad_count = meta.get("tfg_plateau_bad_count", 0)
        self._prev_adv_per_env = meta.get("prev_adv_per_env")
        self._prev_valid = meta.get("prev_valid")

        print(f"[Checkpoint] Restored at sample_step={self.sample_log.sample_step}, "
              f"update_step={self.update_log.update_step}, "
              f"buffer_len={len(self.buffer)}")
        return True

    def delete_checkpoint(self):
        """Remove checkpoint files after successful completion."""
        import shutil
        for d in [self._checkpoint_dir,
                  self._checkpoint_dir.parent / "checkpoints_tmp",
                  self._checkpoint_dir.parent / "checkpoints_old"]:
            if d.exists():
                shutil.rmtree(d)
        wandb_id_file = self.log_path / "wandb_run_id.txt"
        if wandb_id_file.exists():
            wandb_id_file.unlink()
        print("[Checkpoint] Cleaned up checkpoint files")

    # ------------------------------------------------------------------

    def run(self, key: jax.Array):
        if self._checkpoint_enabled:
            self._install_signal_handler()
        try:
            self.train(key)
        except KeyboardInterrupt:
            pass
        finally:
            if self._checkpoint_enabled and self._preempted:
                # Don't delete checkpoint or finish wandb cleanly —
                # we want to resume from it.
                self._flush_wandb()
                self.progress.close()
                wandb.finish(exit_code=1)
            else:
                if self._checkpoint_enabled:
                    self.delete_checkpoint()
                self.finish()

    def finish(self):
        self._flush_wandb()
        self.env.close()
        self.progress.close()
        wandb.finish()

def create_iter_key_fn(key: jax.Array, sample_per_iteration: int, update_per_iteration: int) -> Callable[[int], Tuple[jax.Array, jax.Array]]:
    def iter_key_fn(step: int):
        iter_key = jax.random.fold_in(key, step)
        sample_key, update_key = jax.random.split(iter_key)
        if sample_per_iteration > 1:
            sample_key = jax.random.split(sample_key, sample_per_iteration)
        else:
            sample_key = (sample_key,)
        if update_per_iteration > 1:
            update_key = jax.random.split(update_key, update_per_iteration)
        else:
            update_key = (update_key,)
        return sample_key, update_key

    iter_key_fn = jax.jit(iter_key_fn)
    iter_key_fn(0)  # Warm up
    return iter_key_fn
