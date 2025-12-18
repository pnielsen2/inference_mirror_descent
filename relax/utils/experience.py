from typing import NamedTuple, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import jax

def probe_batch_size(reward: "jax.Array") -> Optional[int]:
    try:
        if reward.ndim > 0:
            return reward.shape[0]
        else:
            return None
    except AttributeError:
        return None

class Experience(NamedTuple):
    obs: "jax.Array"
    action: "jax.Array"
    reward: "jax.Array"
    done: "jax.Array"
    next_obs: "jax.Array"
    next_action: Optional["jax.Array"] = None  # Action taken at next_obs (for SARSA-style TD)

    def batch_size(self) -> Optional[int]:
        return probe_batch_size(self.reward)

    def __repr__(self):
        return f"Experience(size={self.batch_size()})"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None, include_next_action: bool = False):
        leading_dims = (batch_size,) if batch_size is not None else ()
        # Always include next_action in the example (for consistent buffer structure)
        # The field will only be populated when track_next_action is enabled
        return Experience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=np.zeros(leading_dims, dtype=np.float32),
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            done=np.zeros(leading_dims, dtype=np.bool_),
            next_action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
        )

    @staticmethod
    def create(obs, action, reward, terminated, truncated, next_obs, info=None, next_action=None):
        return Experience(obs=obs, action=action, reward=reward, done=terminated, next_obs=next_obs, next_action=next_action)

class GAEExperience(NamedTuple):
    obs: "jax.Array"
    action: "jax.Array"
    reward: "jax.Array"
    done: "jax.Array"
    next_obs: "jax.Array"
    ret: "jax.Array"
    adv: "jax.Array"

    def batch_size(self) -> Optional[int]:
        return probe_batch_size(self.reward)

    def __repr__(self):
        return f"GAEExperience(size={self.batch_size()})"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return GAEExperience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=np.zeros(leading_dims, dtype=np.float32),
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            done=np.zeros(leading_dims, dtype=np.bool_),
            ret=np.zeros(leading_dims, dtype=np.float32),
            adv=np.zeros(leading_dims, dtype=np.float32),
        )

class SafeExperience(NamedTuple):
    obs: "jax.Array"
    action: "jax.Array"
    reward: "jax.Array"
    done: "jax.Array"
    next_obs: "jax.Array"
    cost: "jax.Array"
    feasible: "jax.Array"
    infeasible: "jax.Array"
    barrier: "jax.Array"
    next_barrier: "jax.Array"

    def batch_size(self) -> Optional[int]:
        try:
            if self.reward.ndim > 0:
                return self.reward.shape[0]
            else:
                return None
        except AttributeError:
            return None

    def __repr__(self):
        return f"SafeExperience(size={self.batch_size()})"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return SafeExperience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=np.zeros(leading_dims, dtype=np.float32),
            done=np.zeros(leading_dims, dtype=np.bool_),
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            cost=np.zeros(leading_dims, dtype=np.float32),
            feasible=np.zeros(leading_dims, dtype=np.bool_),
            infeasible=np.zeros(leading_dims, dtype=np.bool_),
            barrier=np.zeros(leading_dims, dtype=np.float32),
            next_barrier=np.zeros(leading_dims, dtype=np.float32),
        )

    @staticmethod
    def create(obs, action, reward, terminated, truncated, next_obs, info: dict):
        cost = info.get("cost", 0.0)
        feasible = info.get("feasible", False)
        infeasible = info.get("infeasible", False)
        barrier = info.get("barrier", 0.0)
        next_barrier = info.get("next_barrier", 0.0)
        return SafeExperience(
            obs=obs,
            action=action,
            reward=reward,
            done=terminated,
            next_obs=next_obs,
            cost=cost,
            feasible=feasible,
            infeasible=infeasible,
            barrier=barrier,
            next_barrier=next_barrier,
        )

class ObsActionPair(NamedTuple):
    obs: "jax.Array"
    action: "jax.Array"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return ObsActionPair(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
        )


class SequenceExperience(NamedTuple):
    """H-step sequence of experiences starting from obs.

    Used for training policies that denoise H future actions conditioned on
    the current state.

    Attributes:
        obs: Starting observation, shape [B, obs_dim].
        actions: H consecutive actions, shape [B, H, act_dim].
        rewards: H consecutive rewards, shape [B, H].
        dones: H consecutive done flags, shape [B, H].
        next_obs_seq: H consecutive next observations, shape [B, H, obs_dim].
            next_obs_seq[:, h] is the state after taking actions[:, h].
    """

    obs: "jax.Array"
    actions: "jax.Array"
    rewards: "jax.Array"
    dones: "jax.Array"
    next_obs_seq: "jax.Array"

    def batch_size(self) -> Optional[int]:
        return probe_batch_size(self.rewards[:, 0] if self.rewards.ndim > 1 else self.rewards)

    def horizon(self) -> int:
        return self.actions.shape[1] if self.actions.ndim > 1 else 1

    def __repr__(self):
        return f"SequenceExperience(batch={self.batch_size()}, H={self.horizon()})"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, horizon: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return SequenceExperience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            actions=np.zeros((*leading_dims, horizon, action_dim), dtype=np.float32),
            rewards=np.zeros((*leading_dims, horizon), dtype=np.float32),
            dones=np.zeros((*leading_dims, horizon), dtype=np.bool_),
            next_obs_seq=np.zeros((*leading_dims, horizon, obs_dim), dtype=np.float32),
        )
