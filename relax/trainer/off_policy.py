from pathlib import Path
import subprocess
import sys
from typing import Callable, Optional, Tuple
import math

import jax
import numpy as np
from gymnasium import Env
from tqdm import tqdm
from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams
import wandb

from relax.algorithm import Algorithm
from relax.buffer import ExperienceBuffer, TreeBuffer
from relax.env.vector import VectorEnv
from relax.trainer.accumulator import SampleLog, VectorSampleLog, UpdateLog, Interval
from relax.utils.experience import Experience


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
        sample_log_n_episode: int = 10,
        update_log_n_step: int = 1000,
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
        self.sample_log_n_episode = sample_log_n_episode
        self.update_log_n_step = update_log_n_step
        self.done_info_keys = done_info_keys
        self.save_policy_every = save_policy_every
        self.hparams = hparams
        self.warmup_with = warmup_with
        self.save_value = save_value
        self.evaluator = None
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
        # State for tracking next_action in SARSA-style buffer
        self._prev_buffer_indices: Optional[np.ndarray] = None
        self._prev_dones: Optional[np.ndarray] = None
        # TODO: make EpisodeLog and Experience configurable
        # TODO: re-add done_info_keys support
        # TODO: re-add evaluation support

        if isinstance(self.env.unwrapped, VectorEnv):
            self.is_vec = True
            self.sample_log = VectorSampleLog(self.env.unwrapped.num_envs)
        else:
            self.is_vec = False
            self.sample_log = SampleLog()
        self.update_log = UpdateLog()
        self.last_metrics = {}
        # The following two depends on sample_step, which may not update by one only
        self.sample_log_interval = Interval(self.sample_log_n_episode)
        self.save_policy_interval = Interval(self.save_policy_every)
        # self.eval_interval = Interval()
        wandb.init(
            project="diffusion_online_rl",
            name=log_path.name,
            dir=log_path,
            group=env.spec.id,
            config=self.hparams,
        )

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

        # Setup logger
        self.logger = SummaryWriter(str(self.log_path))
        self.progress = tqdm(total=self.total_step, desc="Sample Step", disable=None, dynamic_ncols=True)

        self.algorithm.save_policy_structure(self.log_path, dummy_data.obs[0])
        if self.save_value:
            self.algorithm.save_q_structure(self.log_path, dummy_obs=dummy_data.obs[0], dummy_action=dummy_data.action[0])
        # Only launch an evaluator subprocess if an evaluation environment was provided.
        if self.evaluate_env is not None:
            self.evaluator = subprocess.Popen(
                [
                    sys.executable,
                    "-m", "relax.trainer.evaluator",
                    str(self.log_path),
                    "--env", self.env.spec.id,
                    "--num_episodes", str(self.evaluate_n_episode),
                    "--seed", str(0),
                ],
                stdin=subprocess.PIPE,
                bufsize=0,
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
                action = self.env.action_space.sample()
            elif self.warmup_with == "policy":
                action = self.algorithm.get_action(key_fn(step), train_obs)
            else:
                raise ValueError(f"Invalid warmup_with {self.warmup_with}!")
            
            # SARSA-style tracking: update previous experience's next_action
            if self.track_next_action and warmup_prev_indices is not None:
                if warmup_prev_dones is not None:
                    prev_dones = np.atleast_1d(warmup_prev_dones)
                    prev_indices = np.atleast_1d(warmup_prev_indices)
                    action_arr = np.atleast_2d(action) if np.asarray(action).ndim == 1 else action
                    non_terminal_mask = ~prev_dones
                    if np.any(non_terminal_mask):
                        indices_to_update = prev_indices[non_terminal_mask]
                        actions_to_set = action_arr[non_terminal_mask] if self.is_vec else action
                        update_exp = Experience(
                            obs=None, action=None, reward=None, done=None, next_obs=None,
                            next_action=actions_to_set,
                        )
                        self.buffer.replace(indices_to_update, update_exp)
            
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            experience = Experience.create(train_obs, action, reward, terminated, truncated, next_obs, info)
            
            # Track buffer indices before adding (for next iteration's next_action update)
            if self.track_next_action:
                if self.is_vec:
                    batch_size = np.atleast_1d(reward).shape[0]
                    warmup_prev_indices = np.arange(self.buffer.ptr, self.buffer.ptr + batch_size) % self.buffer.max_len
                else:
                    warmup_prev_indices = np.array([self.buffer.ptr])
                warmup_prev_dones = np.atleast_1d(terminated) | np.atleast_1d(truncated)
            
            if self.is_vec:
                self.buffer.add_batch(experience)
            else:
                self.buffer.add(experience)

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
                    val_action = self.val_env.action_space.sample()
                elif self.warmup_with == "policy":
                    val_action = self.algorithm.get_action(val_key_fn(val_step), val_obs)
                else:
                    raise ValueError(f"Invalid warmup_with {self.warmup_with}!")
                val_next_obs, val_reward, val_terminated, val_truncated, val_info = self.val_env.step(val_action)

                val_experience = Experience.create(val_obs, val_action, val_reward, val_terminated, val_truncated, val_next_obs, val_info)
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

        action = self.algorithm.get_action(sample_key, obs)
        
        # SARSA-style tracking: update previous experience's next_action with current action
        if self.track_next_action and self._prev_buffer_indices is not None:
            # Only update for non-terminal previous transitions
            if self._prev_dones is not None:
                # For vectorized envs, _prev_dones is a boolean array
                # For single env, _prev_dones is a scalar boolean
                prev_dones = np.atleast_1d(self._prev_dones)
                prev_indices = np.atleast_1d(self._prev_buffer_indices)
                action_arr = np.atleast_2d(action) if action.ndim == 1 else action
                
                # Update next_action only for non-terminal transitions
                non_terminal_mask = ~prev_dones
                if np.any(non_terminal_mask):
                    indices_to_update = prev_indices[non_terminal_mask]
                    actions_to_set = action_arr[non_terminal_mask] if self.is_vec else action_arr[0]
                    # Use buffer's replace method to update only next_action field
                    # Create a partial Experience with only next_action set
                    update_exp = Experience(
                        obs=None, action=None, reward=None, done=None, next_obs=None,
                        next_action=actions_to_set if self.is_vec else action,
                    )
                    self.buffer.replace(indices_to_update, update_exp)
        
        next_obs, reward, terminated, truncated, info = self.env.step(action)

        experience = Experience.create(obs, action, reward, terminated, truncated, next_obs, info)
        
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
        
        if self.is_vec:
            self.buffer.add_batch(experience)
        else:
            self.buffer.add(experience)

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
            val_action = self.algorithm.get_action(val_key, self.val_obs)
            val_next_obs, val_reward, val_terminated, val_truncated, val_info = self.val_env.step(val_action)

            val_experience = Experience.create(
                self.val_obs,
                val_action,
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

        if any_done:
            if self.sample_log_interval.check(sl.sample_episode):
                sl.log(self.add_scalar)
            self.progress.update(sl.sample_step - self.progress.n)

            obs, _ = self.env.reset()
            # Reset open-loop execution state at episode boundaries
            if hasattr(self.algorithm, 'reset_open_loop'):
                self.algorithm.reset_open_loop()
        else:
            obs = next_obs

        return obs

    def update(self, update_key: jax.Array):
        ul = self.update_log

        # The UpdateLog increments its internal step counter when we call
        # ul.add(info). For gating logic (e.g., how often to log or run
        # hypergradient updates), we base decisions on the *next* update step
        # index, corresponding to the update we are about to perform.
        current_update_step = ul.update_step
        next_update_step = current_update_step + 1

        supervised_steps = getattr(self.algorithm, "supervised_steps", 1)

        key = update_key

        # Sample training batch uniformly from the main replay buffer.
        # For H-step policy training, sample sequences instead of individual transitions.
        H_train = getattr(self.algorithm, "H_train", 1)
        train_idx, val_idx = self._get_train_val_index_sets()
        if H_train > 1 and isinstance(self.buffer, TreeBuffer):
            # Sample H-step sequences for policy training
            train_data = self.buffer.sample_sequences(self.batch_size, H_train)
        elif train_idx is None:
            train_data = self.buffer.sample(self.batch_size)
        else:
            train_data = self._sample_from_index_set(train_idx, self.batch_size)

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
        log_this_step = (next_update_step % self.update_log_n_step == 0)

        need_val_for_logging = have_val_indices and log_this_step
        if need_val_for_hyper or need_val_for_logging:
            if self.val_buffer is not None:
                # Sample from the dedicated validation buffer.
                val_data = self.val_buffer.sample(self.val_batch_size)
            else:
                val_data = self._sample_from_index_set(val_idx, self.val_batch_size)

        # Main training / hypergradient path
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
            info, dist_info, _ = self.algorithm.hyper_update(update_key, train_data, val_data)
        else:
            # Standard training update (optionally with supervised warmup)
            if supervised_steps > 1 and hasattr(self.algorithm, "update_supervised"):
                for _ in range(supervised_steps - 1):
                    key, subkey = jax.random.split(key)
                    self.algorithm.update_supervised(subkey, train_data)

                key, subkey = jax.random.split(key)
                info, dist_info = self.algorithm.update(subkey, train_data)
            else:
                info, dist_info = self.algorithm.update(update_key, train_data)

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

        ul.add(info)

        if log_this_step:
            current_step = self.sample_log.sample_step
            self.add_hist(dist_info, current_step)

            # Log training metrics under a dedicated "training/" section so
            # they appear in a separate tab analogous to the validation tab.
            ul.log(
                lambda tag, value, _step: self.add_scalar(
                    f"training/{tag}", value, current_step
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

        # Warmup internally resets environments and returns the latest
        # training observations after the replay buffer has been populated.
        obs = self.warmup(warmup_key)

        iter_key_fn = create_iter_key_fn(key, self.sample_per_iteration, self.update_per_iteration)
        sl, ul = self.sample_log, self.update_log

        self.progress.unpause()
        while sl.sample_step <= self.total_step:
            sample_keys, update_keys = iter_key_fn(sl.sample_step)

            for i in range(self.sample_per_iteration):
                obs = self.sample(sample_keys[i], obs)

            for i in range(self.update_per_iteration):
                self.update(update_keys[i])

            if self.save_policy_interval.check(sl.sample_step):
                policy_pkl_name = self.policy_pkl_template.format(
                    sample_step=sl.sample_step,
                    update_step=ul.update_step,
                )
                self.algorithm.save_policy(self.log_path / policy_pkl_name)
                
                if self.save_value:
                    self.algorithm.save_q(self.log_path / policy_pkl_name.replace('policy', 'value'))
                
                if self.evaluator is not None:
                    command = f"{sl.sample_step},{self.log_path / policy_pkl_name}\n"
                    self.evaluator.stdin.write(command.encode())

    def add_scalar(self, tag: str, value: float, step: int):
        self.last_metrics[tag] = value
        wandb.log({tag: value}, step=step)
        self.logger.add_scalar(tag, value, step)
        self.logger.flush()
        
    def add_hist(self, info_hist, step):
        for tag, value in info_hist.items():
            self.logger.add_histogram(tag, np.array(value), step)
            wandb.log({tag: wandb.Histogram(np.array(value))}, step=step)
        self.logger.flush()

    def run(self, key: jax.Array):
        try:
            self.train(key)
        except KeyboardInterrupt:
            pass
        finally:
            self.finish()

    def finish(self):
        self.env.close()
        self.algorithm.save(self.log_path / "state.pkl")
        if self.hparams is not None and len(self.last_metrics) > 0:
            exp, ssi, sei = hparams(self.hparams, self.last_metrics)
            self.logger.file_writer.add_summary(exp)
            self.logger.file_writer.add_summary(ssi)
            self.logger.file_writer.add_summary(sei)
        self.logger.close()
        self.progress.close()
        if self.evaluator is not None:
            self.evaluator.stdin.close()
            self.evaluator.wait()

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
