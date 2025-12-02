from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Any

import numpy as np


@dataclass
class Interval:
    """Simple interval helper used to decide when to log / save.

    OffPolicyTrainer expects:
      - .check(step: int) -> bool
    """

    interval: int
    _last_step: int = 0

    def check(self, step: int) -> bool:
        if self.interval <= 0:
            return False
        if step - self._last_step >= self.interval:
            self._last_step = step
            return True
        return False


@dataclass
class SampleLog:
    """Tracks sampling statistics over environment steps.

    OffPolicyTrainer expects:
      - attributes: sample_step, sample_episode
      - add(reward, terminated, truncated, info) -> bool (any_done)
      - log(add_scalar_fn)

    Semantics:
      - "episode_return" should reflect the *average per-episode return* over
        the last N completed episodes, where N is controlled by
        OffPolicyTrainer.sample_log_n_episode via Interval.
    """

    sample_step: int = 0
    sample_episode: int = 0
    _current_return: float = 0.0
    _pending_returns: list[float] = field(default_factory=list)

    def add(self, reward, terminated, truncated, info) -> bool:
        """Accumulate rewards for the current episode and detect termination.

        For vector environments, we aggregate by taking the mean reward across
        envs at each step and summing that over the episode; this matches the
        original behavior where a single scalar "episode_return" is reported.
        """
        self.sample_step += 1

        # Reward may be scalar or array-like over vector envs.
        r = np.asarray(reward, dtype=float)
        self._current_return += float(r.mean())

        term = np.asarray(terminated, dtype=bool)
        truc = np.asarray(truncated, dtype=bool)
        any_done = bool(np.any(term) or np.any(truc))

        if any_done:
            # Episode just finished; stash its return and reset accumulator.
            self.sample_episode += 1
            self._pending_returns.append(self._current_return)
            self._current_return = 0.0

        return any_done

    def log(self, add_scalar: Callable[[str, float, int], None]) -> None:
        """Log the mean return over all episodes completed since last log()."""
        if not self._pending_returns:
            return

        avg_return = float(np.mean(self._pending_returns))
        add_scalar("sample/episode_return", avg_return, self.sample_step)

        # Clear pending returns; subsequent episodes will accumulate separately.
        self._pending_returns.clear()


@dataclass
class VectorSampleLog(SampleLog):
    """Vectorized version for VectorEnv.

    OffPolicyTrainer only relies on the same interface as SampleLog, but this
    class allows for future per-env extensions. For now we aggregate across
    envs just like SampleLog.
    """

    num_envs: int = 1


@dataclass
class UpdateLog:
    """Tracks update statistics for training.

    OffPolicyTrainer expects:
      - attribute: update_step
      - add(info: dict)
      - log(log_fn: Callable[[tag, value, step], None])
    """

    update_step: int = 0
    _last_info: Dict[str, Any] = field(default_factory=dict)

    def add(self, info: Dict[str, Any]) -> None:
        self.update_step += 1
        self._last_info = dict(info)

    def log(self, log_fn: Callable[[str, float, int], None]) -> None:
        for k, v in self._last_info.items():
            # v may be JAX array, numpy array, or scalar.
            try:
                val = float(np.asarray(v))
            except Exception:
                continue
            log_fn(k, val, self.update_step)
