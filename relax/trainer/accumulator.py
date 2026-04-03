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
    __slots__ = (
        "sample_step",
        "sample_episode",
        "env_name",
        "gamma",
        "episode_return",
        "episode_length",
        "episode_reward_sum",
        "episode_action_mean_sum",
        "episode_action_var_sum",
        "episode_action_clip_frac_sum",
        "episode_q_agg_sum",
        "episode_q_var_sum",
        "_has_q_agg",
        "_has_q_var",
        "_episode_q_agg_list",
        "_episode_reward_list",
        "_episode_terminated_list",
        "accumulator",
    )

    def __init__(self, env_name: str = "env", gamma: float = 0.99):
        self.sample_step = 0
        self.sample_episode = 0
        self.env_name = env_name
        self.gamma = gamma
        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_reward_sum = 0.0
        self.episode_action_mean_sum = 0.0
        self.episode_action_var_sum = 0.0
        self.episode_action_clip_frac_sum = 0.0
        self.episode_q_agg_sum = 0.0
        self.episode_q_var_sum = 0.0
        self._has_q_agg = False
        self._has_q_var = False
        self._episode_q_agg_list = []
        self._episode_reward_list = []
        self._episode_terminated_list = []
        self.accumulator = Accumulator("")

    def add(self, reward: float, terminated: bool, truncated: bool, info: dict):
        self.episode_return += reward
        self.episode_reward_sum += reward
        self.episode_length += 1
        self.sample_step += 1

        if "action_mean" in info:
            self.episode_action_mean_sum += float(info["action_mean"])
        if "action_var" in info:
            self.episode_action_var_sum += float(info["action_var"])
        if "action_clip_frac" in info:
            self.episode_action_clip_frac_sum += float(info["action_clip_frac"])
        if "q_agg" in info:
            self.episode_q_agg_sum += float(info["q_agg"])
            self._has_q_agg = True
            self._episode_q_agg_list.append(float(info["q_agg"]))
            self._episode_reward_list.append(float(reward))
            self._episode_terminated_list.append(bool(terminated))
            self.accumulator.add("Critic/average_Q", float(info["q_agg"]))
        if "q_var" in info:
            self.episode_q_var_sum += float(info["q_var"])
            self._has_q_var = True
        if "v_value" in info:
            self.accumulator.add("Critic/average_V", float(info["v_value"]))

        done = terminated or truncated
        if done:
            self.sample_episode += 1
            ep_return_key = f"episode_return/{self.env_name}"
            self.accumulator.add(ep_return_key, float(self.episode_return))
            L = max(self.episode_length, 1)
            if terminated:
                self.accumulator.add("episodes/reward_mean", self.episode_reward_sum / L)
                self.accumulator.add("episodes/episode_length", float(self.episode_length))
            self.accumulator.add("actions/action_mean", self.episode_action_mean_sum / L)
            self.accumulator.add("actions/action_var", self.episode_action_var_sum / L)
            self.accumulator.add("actions/final_action_clip_frac", self.episode_action_clip_frac_sum / L)
            if self._has_q_var:
                self.accumulator.add("Critic/E(Var({Q_i})_env)", self.episode_q_var_sum / L)
            if self._has_q_agg and len(self._episode_q_agg_list) > 0:
                bias = _compute_q_agg_tilt_bias(
                    self._episode_q_agg_list,
                    self._episode_reward_list,
                    self._episode_terminated_list,
                    self.gamma,
                    terminated,
                )
                if bias is not None:
                    self.accumulator.add("Critic/Q_agg_tilt_bias", bias)
            self._reset_episode()

        return done

    def _reset_episode(self):
        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_reward_sum = 0.0
        self.episode_action_mean_sum = 0.0
        self.episode_action_var_sum = 0.0
        self.episode_action_clip_frac_sum = 0.0
        self.episode_q_agg_sum = 0.0
        self.episode_q_var_sum = 0.0
        self._has_q_agg = False
        self._has_q_var = False
        self._episode_q_agg_list.clear()
        self._episode_reward_list.clear()
        self._episode_terminated_list.clear()

    def log(self, log_fn: Callable[[str, float, int], None]):
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


class VectorSampleLog:
    __slots__ = (
        "num_envs",
        "env_name",
        "gamma",
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
        "accumulator",
    )

    def __init__(self, num_envs: int, env_name: str = "env", gamma: float = 0.99):
        self.num_envs = num_envs
        self.env_name = env_name
        self.gamma = gamma
        self.sample_step = 0
        self.sample_episode = 0
        self.episode_return = np.zeros((num_envs,), dtype=np.float64)
        self.episode_length = np.zeros((num_envs,), dtype=np.int64)
        self.episode_reward_sum = np.zeros((num_envs,), dtype=np.float64)
        self.episode_action_mean_sum = np.zeros((num_envs,), dtype=np.float64)
        self.episode_action_var_sum = np.zeros((num_envs,), dtype=np.float64)
        self.episode_action_clip_frac_sum = np.zeros((num_envs,), dtype=np.float64)
        self.episode_q_var_sum = np.zeros((num_envs,), dtype=np.float64)
        self._has_q_agg = False
        self._has_q_var = False
        self._episode_q_agg_lists = [[] for _ in range(num_envs)]
        self._episode_reward_lists = [[] for _ in range(num_envs)]
        self._episode_terminated_lists = [[] for _ in range(num_envs)]
        self.accumulator = Accumulator("")

    def add(self, reward: np.ndarray, terminated: np.ndarray, truncated: np.ndarray, info: dict):
        self.episode_return += reward
        self.episode_reward_sum += reward
        self.episode_length += 1
        self.sample_step += self.num_envs

        if "action_mean" in info:
            self.episode_action_mean_sum += np.asarray(info["action_mean"], dtype=np.float64)
        if "action_var" in info:
            self.episode_action_var_sum += np.asarray(info["action_var"], dtype=np.float64)
        if "action_clip_frac" in info:
            self.episode_action_clip_frac_sum += np.asarray(info["action_clip_frac"], dtype=np.float64)
        if "q_agg" in info:
            q_agg_val = float(info["q_agg"])
            self._has_q_agg = True
            for i in range(self.num_envs):
                self._episode_q_agg_lists[i].append(q_agg_val)
                self._episode_reward_lists[i].append(float(reward[i]))
                self._episode_terminated_lists[i].append(bool(terminated[i]))
            self.accumulator.add("Critic/average_Q", q_agg_val)
        if "q_var" in info:
            self.episode_q_var_sum += float(info["q_var"])
            self._has_q_var = True
        if "v_value" in info:
            self.accumulator.add("Critic/average_V", float(info["v_value"]))

        done = terminated | truncated
        done_count = np.count_nonzero(done)

        ep_return_key = f"episode_return/{self.env_name}"
        self.sample_episode += done_count

        if done_count > 0:
            denom = np.maximum(self.episode_length[done], 1).astype(np.float64)
            self.accumulator.add_vec(ep_return_key, self.episode_return[done].tolist())
            term_done = terminated[done]
            if np.any(term_done):
                term_denom = np.maximum(self.episode_length[done][term_done], 1).astype(np.float64)
                self.accumulator.add_vec("episodes/reward_mean", (self.episode_reward_sum[done][term_done] / term_denom).tolist())
                self.accumulator.add_vec("episodes/episode_length", self.episode_length[done][term_done].astype(np.float64).tolist())
            self.accumulator.add_vec("actions/action_mean", (self.episode_action_mean_sum[done] / denom).tolist())
            self.accumulator.add_vec("actions/action_var", (self.episode_action_var_sum[done] / denom).tolist())
            self.accumulator.add_vec("actions/final_action_clip_frac", (self.episode_action_clip_frac_sum[done] / denom).tolist())
            if self._has_q_var:
                self.accumulator.add_vec("Critic/E(Var({Q_i})_env)", (self.episode_q_var_sum[done] / denom).tolist())
            if self._has_q_agg:
                done_indices = np.flatnonzero(done)
                for idx in done_indices:
                    if len(self._episode_q_agg_lists[idx]) > 0:
                        bias = _compute_q_agg_tilt_bias(
                            self._episode_q_agg_lists[idx],
                            self._episode_reward_lists[idx],
                            self._episode_terminated_lists[idx],
                            self.gamma,
                            bool(terminated[idx]),
                        )
                        if bias is not None:
                            self.accumulator.add("Critic/Q_agg_tilt_bias", bias)

        # Reset done envs
        self.episode_return[done] = 0.0
        self.episode_length[done] = 0
        self.episode_reward_sum[done] = 0.0
        self.episode_action_mean_sum[done] = 0.0
        self.episode_action_var_sum[done] = 0.0
        self.episode_action_clip_frac_sum[done] = 0.0
        self.episode_q_var_sum[done] = 0.0
        if done_count > 0:
            done_indices = np.flatnonzero(done)
            for idx in done_indices:
                self._episode_q_agg_lists[idx].clear()
                self._episode_reward_lists[idx].clear()
                self._episode_terminated_lists[idx].clear()
            self._has_q_agg = False
            self._has_q_var = False

        return done_count > 0

    def log(self, log_fn: Callable[[str, float, int], None]):
        self.accumulator.log(lambda k, v: log_fn(k, v, self.sample_step))
        self.accumulator.reset()

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
