from collections import defaultdict
from typing import Callable

from numba import njit, types as nt
import numpy as np

class Accumulator:
    __slots__ = ("prefix", "buffer")

    def __init__(self, prefix=""):
        self.prefix = prefix
        self.buffer = defaultdict(list)

    def add(self, key, value):
        self.buffer[key].append(value)

    def add_vec(self, key, value):
        self.buffer[key].extend(value)

    def add_all(self, data: dict):
        for key, value in data.items():
            self.add(key, value)

    def reset(self):
        self.buffer.clear()

    def log(self, log_fn: Callable[[str, float], None]):
        for key, values in self.buffer.items():
            key = key if not self.prefix else f"{self.prefix}/{key}"
            value = sum(values) / len(values)
            log_fn(key, value)

class SampleLog:
    """Unified episode tracker for 1..N parallel environments.

    Each call to ``add()`` corresponds to one synchronous batch step across all
    ``num_envs`` environments.  ``sample_step`` advances by ``num_envs`` per
    call, so that the per-env sub-step for env *i* finishing on batch step *N* is

        env_step = (N - 1) * num_envs + (i + 1)

    Episode returns are recorded in ``_pending_episode_returns`` and drained by
    the trainer via ``take_pending_episode_returns()``.
    """
    __slots__ = (
        "num_envs",
        "env_name",
        "gamma",
        "q_label",
        "env_can_terminate",
        "sample_step",
        "sample_episode",
        "episode_return",
        "episode_length",
        "episode_reward_sum",
        "episode_action_mean_sum",
        "episode_action_var_sum",
        "episode_action_clip_frac_sum",
        "episode_q_var_sum",
        "_has_q_agg",
        "_has_q_var",
        "_episode_q_agg_lists",
        "_episode_reward_lists",
        "_episode_terminated_lists",
        "_pending_episode_returns",
        "accumulator",
    )

    def __init__(self, num_envs: int = 1, env_name: str = "env", gamma: float = 0.99, q_label: str = "Q", env_can_terminate: bool = True):
        self.num_envs = num_envs
        self.env_name = env_name
        self.gamma = gamma
        self.q_label = q_label
        self.env_can_terminate = env_can_terminate
        self.sample_step = 0
        self.sample_episode = 0
        self.episode_return = np.zeros((num_envs,), dtype=np.float64)
        self.episode_length = np.zeros((num_envs,), dtype=np.int64)
        self.episode_reward_sum = np.zeros((num_envs,), dtype=np.float64)
        self.episode_action_mean_sum = np.zeros((num_envs,), dtype=np.float64)
        self.episode_action_var_sum = np.zeros((num_envs,), dtype=np.float64)
        self.episode_action_clip_frac_sum = np.zeros((num_envs,), dtype=np.float64)
        self.episode_q_var_sum = np.zeros((num_envs,), dtype=np.float64)
        self._has_q_agg = np.zeros((num_envs,), dtype=np.bool_)
        self._has_q_var = np.zeros((num_envs,), dtype=np.bool_)
        self._episode_q_agg_lists = [[] for _ in range(num_envs)]
        self._episode_reward_lists = [[] for _ in range(num_envs)]
        self._episode_terminated_lists = [[] for _ in range(num_envs)]
        self._pending_episode_returns = []
        self.accumulator = Accumulator("")

    def add(self, reward, terminated, truncated, info: dict):
        """Record one synchronous step across all envs.

        ``reward``, ``terminated``, ``truncated`` are arrays of shape
        ``(num_envs,)`` (or scalars when ``num_envs == 1``).
        """
        reward = np.atleast_1d(np.asarray(reward, dtype=np.float64))
        terminated = np.atleast_1d(np.asarray(terminated, dtype=np.bool_))
        truncated = np.atleast_1d(np.asarray(truncated, dtype=np.bool_))

        self.episode_return += reward
        self.episode_reward_sum += reward
        self.episode_length += 1
        self.sample_step += self.num_envs

        if "action_mean" in info:
            self.episode_action_mean_sum += np.atleast_1d(np.asarray(info["action_mean"], dtype=np.float64))
        if "action_var" in info:
            self.episode_action_var_sum += np.atleast_1d(np.asarray(info["action_var"], dtype=np.float64))
        if "action_clip_frac" in info:
            self.episode_action_clip_frac_sum += np.atleast_1d(np.asarray(info["action_clip_frac"], dtype=np.float64))
        if "q_agg" in info:
            q_agg_val = float(info["q_agg"])
            self._has_q_agg[:] = True
            for i in range(self.num_envs):
                self._episode_q_agg_lists[i].append(q_agg_val)
                self._episode_reward_lists[i].append(float(reward[i]))
                self._episode_terminated_lists[i].append(bool(terminated[i]))
            self.accumulator.add(f"Critic/average_{self.q_label}", q_agg_val)
        if "q_var" in info:
            self.episode_q_var_sum += float(info["q_var"])
            self._has_q_var[:] = True
        if "v_value" in info:
            self.accumulator.add("Critic/average_V", float(info["v_value"]))

        done = terminated | truncated
        done_indices = np.flatnonzero(done)
        done_count = len(done_indices)

        self.sample_episode += done_count

        if done_count > 0:
            for env_idx in done_indices:
                env_step = self.sample_step - (self.num_envs - 1 - int(env_idx))
                self._pending_episode_returns.append(
                    (env_step, float(self.episode_return[env_idx]))
                )

            denom = np.maximum(self.episode_length[done_indices], 1).astype(np.float64)
            if self.env_can_terminate:
                self.accumulator.add_vec("episodes/reward_mean", (self.episode_reward_sum[done_indices] / denom).tolist())
                self.accumulator.add_vec("episodes/episode_length", self.episode_length[done_indices].astype(np.float64).tolist())
            self.accumulator.add_vec("actions/action_mean", (self.episode_action_mean_sum[done_indices] / denom).tolist())
            self.accumulator.add_vec("actions/action_var", (self.episode_action_var_sum[done_indices] / denom).tolist())
            self.accumulator.add_vec("actions/final_action_clip_frac", (self.episode_action_clip_frac_sum[done_indices] / denom).tolist())
            ql = self.q_label
            if np.any(self._has_q_var[done_indices]):
                q_var_mask = self._has_q_var[done_indices]
                self.accumulator.add_vec(f"Critic/E(Var({{{ql}_i}})_env)", (self.episode_q_var_sum[done_indices][q_var_mask] / denom[q_var_mask]).tolist())
            for idx in done_indices:
                if self._has_q_agg[idx] and len(self._episode_q_agg_lists[idx]) > 0:
                    bias = _compute_q_agg_tilt_bias(
                        self._episode_q_agg_lists[idx],
                        self._episode_reward_lists[idx],
                        self._episode_terminated_lists[idx],
                        self.gamma,
                        bool(terminated[idx]),
                    )
                    if bias is not None:
                        self.accumulator.add(f"Critic/{ql}_agg_tilt_bias", bias)

            # Reset done envs
            self.episode_return[done_indices] = 0.0
            self.episode_length[done_indices] = 0
            self.episode_reward_sum[done_indices] = 0.0
            self.episode_action_mean_sum[done_indices] = 0.0
            self.episode_action_var_sum[done_indices] = 0.0
            self.episode_action_clip_frac_sum[done_indices] = 0.0
            self.episode_q_var_sum[done_indices] = 0.0
            self._has_q_agg[done_indices] = False
            self._has_q_var[done_indices] = False
            for idx in done_indices:
                self._episode_q_agg_lists[idx].clear()
                self._episode_reward_lists[idx].clear()
                self._episode_terminated_lists[idx].clear()

        return done_count > 0

    def take_pending_episode_returns(self):
        """Return and clear pending (env_step, return) tuples for episodes
        that finished since the last call."""
        pending = self._pending_episode_returns
        self._pending_episode_returns = []
        return pending

    def log_accumulator(self, log_fn: Callable[[str, float, int], None]):
        """Flush averaged accumulator metrics at the current sample_step.
        Episode returns are drained separately via
        ``take_pending_episode_returns``."""
        self.accumulator.log(lambda k, v: log_fn(k, v, self.sample_step))
        self.accumulator.reset()


def _compute_q_agg_tilt_bias(q_list, reward_list, terminated_list, gamma, episode_terminated):
    """Compute weighted average Q_agg tilt bias for a single episode.

    bias_t = Q_agg_t - sum_{k=t}^{T-1} gamma^{k-t} * r_k

    No tail correction is applied regardless of how the episode ended, so that
    the metric is comparable across environments with different episode lengths.

    Weights: w_t proportional to min-variance, approximated as (1 - gamma^{2*(T-t)}).
    """
    T = len(q_list)
    if T == 0:
        return None
    q_arr = np.array(q_list, dtype=np.float64)
    r_arr = np.array(reward_list, dtype=np.float64)

    # Compute discounted remaining return from each step
    remaining_return = np.zeros(T, dtype=np.float64)
    cumulative = 0.0
    for t in range(T - 1, -1, -1):
        cumulative = r_arr[t] + gamma * cumulative
        remaining_return[t] = cumulative

    bias = q_arr - remaining_return

    return float(np.mean(bias))


class VectorFragmentSampleLog:
    __slots__ = ("num_envs", "fragment_length", "sample_step", "sample_episode", "episode_return", "episode_length", "accumulator")

    def __init__(self, num_envs: int, fragment_length: int):
        self.num_envs = num_envs
        self.fragment_length = fragment_length
        self.sample_step = 0
        self.sample_episode = 0
        self.episode_return = np.zeros((num_envs,), dtype=np.float64)
        self.episode_length = np.zeros((num_envs,), dtype=np.int64)
        self.accumulator = Accumulator("sample")

    def add(self, reward: np.ndarray, terminated: np.ndarray, truncated: np.ndarray, info: dict):
        done_count, complete_episode_return, complete_episode_length = process_fragment(reward, terminated, truncated, self.episode_return, self.episode_length, self.num_envs, self.fragment_length)
        self.sample_step += self.num_envs * self.fragment_length
        self.sample_episode += done_count
        self.accumulator.add_vec("episode_return", complete_episode_return.tolist())
        self.accumulator.add_vec("episode_length", complete_episode_length.tolist())
        return done_count > 0

    def log(self, log_fn: Callable[[str, float, int], None]):
        self.accumulator.log(lambda k, v: log_fn(k, v, self.sample_step))
        self.accumulator.reset()

@njit([(nt.float64[:, ::1], nt.boolean[:, ::1], nt.boolean[:, ::1], nt.float64[::1], nt.int64[::1], nt.int64, nt.int64)], cache=True)
def process_fragment(reward: np.ndarray, terminated: np.ndarray, truncated: np.ndarray, episode_return: np.ndarray, episode_length: np.ndarray, num_envs: int, fragment_length: int):
    assert reward.shape == terminated.shape == truncated.shape == (num_envs, fragment_length)

    done = terminated | truncated
    done_count = np.count_nonzero(done)

    if done_count > 0:
        complete_episode_return = np.empty((done_count,), dtype=np.float64)
        complete_episode_length = np.empty((done_count,), dtype=np.int64)
        ptr = 0
        for i in range(num_envs):
            initial_return = episode_return[i]
            initial_length = episode_length[i]
            left = 0
            for j in range(fragment_length):
                if done[i, j]:
                    right = j + 1
                    complete_episode_return[ptr] = reward[i, left:right].sum() + initial_return
                    complete_episode_length[ptr] = right - left + initial_length
                    ptr += 1
                    left = right
                    initial_return = 0.0
                    initial_length = 0
            episode_return[i] = initial_return + reward[i, left:].sum()
            episode_length[i] = fragment_length - left + initial_length
    else:
        episode_return += reward.sum(axis=-1)
        episode_length += fragment_length

    return done_count, complete_episode_return, complete_episode_length

class UpdateLog:
    __slots__ = ("update_step", "accumulator")

    def __init__(self):
        self.update_step = 0
        self.accumulator = Accumulator("")

    def add(self, metrics: dict):
        self.update_step += 1
        self.accumulator.add_all(metrics)

    def log(self, log_fn: Callable[[str, float, int], None]):
        self.accumulator.log(lambda k, v: log_fn(k, v, self.update_step * 5))
        self.accumulator.reset()


class Interval:
    __slots__ = ("interval", "last_step")

    def __init__(self, interval: int):
        self.interval = interval
        self.last_step = 0

    def check(self, step: int) -> bool:
        if step - self.last_step >= self.interval:
            self.last_step = step
            return True
        return False
