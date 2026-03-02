import numpy as np
from gymnasium import Env, Wrapper, make
from gymnasium.spaces import Box

from relax.env.vector import VectorEnv, SerialVectorEnv, GymProcessVectorEnv, PipeProcessVectorEnv, SpinlockProcessVectorEnv, FutexProcessVectorEnv


class DummyActionPenaltyWrapper(Wrapper):
    def __init__(self, env: Env, *, dummy_action_dim: int, dummy_action_alpha: float):
        super().__init__(env)
        assert isinstance(env.action_space, Box) and env.action_space.is_bounded()
        assert len(env.action_space.shape) == 1

        self.base_act_dim = int(env.action_space.shape[0])
        self.dummy_action_dim = int(dummy_action_dim)
        self.dummy_action_alpha = float(dummy_action_alpha)

        if self.dummy_action_dim < self.base_act_dim:
            raise ValueError(
                f"dummy_action_dim ({self.dummy_action_dim}) must be >= base action dim ({self.base_act_dim})."
            )

        if self.dummy_action_dim == self.base_act_dim:
            self.action_space = env.action_space
        else:
            low = np.concatenate(
                [
                    np.asarray(env.action_space.low, dtype=env.action_space.dtype),
                    -np.ones((self.dummy_action_dim - self.base_act_dim,), dtype=env.action_space.dtype),
                ],
                axis=0,
            )
            high = np.concatenate(
                [
                    np.asarray(env.action_space.high, dtype=env.action_space.dtype),
                    np.ones((self.dummy_action_dim - self.base_act_dim,), dtype=env.action_space.dtype),
                ],
                axis=0,
            )
            self.action_space = Box(low=low, high=high, dtype=env.action_space.dtype)

    def step(self, action: np.ndarray):
        action = np.asarray(action)
        if self.dummy_action_dim == self.base_act_dim:
            return self.env.step(action)
        a_env = action[: self.base_act_dim]

        penalty = 0.0
        if self.dummy_action_dim > self.base_act_dim and self.dummy_action_alpha != 0.0:
            a_dummy = action[self.base_act_dim : self.dummy_action_dim]
            penalty = self.dummy_action_alpha * float(np.sum(np.square(a_dummy)))

        obs, reward, terminated, truncated, info = self.env.step(a_env)
        info = dict(info)
        info["dummy_action_penalty"] = penalty
        info["reward_env"] = float(reward)
        return obs, reward - penalty, terminated, truncated, info


class DummyActionPenaltyVectorWrapper(Wrapper, VectorEnv):
    def __init__(self, env: VectorEnv, *, dummy_action_dim: int, dummy_action_alpha: float):
        super().__init__(env)
        assert isinstance(env.action_space, Box) and env.action_space.is_bounded()
        assert len(env.action_space.shape) == 2
        assert hasattr(env, "single_action_space")
        assert isinstance(env.single_action_space, Box) and env.single_action_space.is_bounded()
        assert len(env.single_action_space.shape) == 1

        self.num_envs = int(env.num_envs)
        self.base_act_dim = int(env.single_action_space.shape[0])
        self.dummy_action_dim = int(dummy_action_dim)
        self.dummy_action_alpha = float(dummy_action_alpha)

        if self.dummy_action_dim < self.base_act_dim:
            raise ValueError(
                f"dummy_action_dim ({self.dummy_action_dim}) must be >= base action dim ({self.base_act_dim})."
            )

        self.single_observation_space = env.single_observation_space
        self.observation_space = env.observation_space

        if self.dummy_action_dim == self.base_act_dim:
            self.single_action_space = env.single_action_space
            self.action_space = env.action_space
        else:
            low_single = np.concatenate(
                [
                    np.asarray(env.single_action_space.low, dtype=env.single_action_space.dtype),
                    -np.ones((self.dummy_action_dim - self.base_act_dim,), dtype=env.single_action_space.dtype),
                ],
                axis=0,
            )
            high_single = np.concatenate(
                [
                    np.asarray(env.single_action_space.high, dtype=env.single_action_space.dtype),
                    np.ones((self.dummy_action_dim - self.base_act_dim,), dtype=env.single_action_space.dtype),
                ],
                axis=0,
            )
            self.single_action_space = Box(low=low_single, high=high_single, dtype=env.single_action_space.dtype)

            low = np.broadcast_to(low_single, (self.num_envs, self.dummy_action_dim))
            high = np.broadcast_to(high_single, (self.num_envs, self.dummy_action_dim))
            self.action_space = Box(low=low, high=high, dtype=env.action_space.dtype)

        self.obs_dim = int(env.obs_dim)
        self.act_dim = int(self.dummy_action_dim)

    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action: np.ndarray):
        action = np.asarray(action)
        if self.dummy_action_dim == self.base_act_dim:
            return self.env.step(action)
        a_env = action[:, : self.base_act_dim]

        penalty = 0.0
        if self.dummy_action_dim > self.base_act_dim and self.dummy_action_alpha != 0.0:
            a_dummy = action[:, self.base_act_dim : self.dummy_action_dim]
            penalty = self.dummy_action_alpha * np.sum(np.square(a_dummy), axis=-1)

        obs, reward, terminated, truncated, info = self.env.step(a_env)
        info = dict(info)
        info["dummy_action_penalty"] = penalty
        info["reward_env"] = reward
        return obs, reward - penalty, terminated, truncated, info

class RelaxWrapper(Wrapper):
    def __init__(self, env: Env, action_seed: int = 0):
        super().__init__(env)
        self.env: Env[np.ndarray, np.ndarray]

        assert isinstance(env.observation_space, Box)
        assert isinstance(env.action_space, Box) and env.action_space.is_bounded()
        if isinstance(env, VectorEnv):
            _, self.obs_dim = env.observation_space.shape
            _, self.act_dim = env.action_space.shape
            single_action_space = env.single_action_space
        else:
            self.obs_dim, = env.observation_space.shape
            self.act_dim, = env.action_space.shape
            single_action_space = env.action_space

        if np.any(single_action_space.low != -1.0) or np.any(single_action_space.high != 1.0):
            print(f"NOTE: The action space is not normalized, but {single_action_space.low} to {single_action_space.high}, will be rescaled.")
            self.needs_rescale = True
            self.original_action_center = (single_action_space.low + single_action_space.high) * 0.5
            self.original_action_half_range = (single_action_space.high - single_action_space.low) * 0.5
        else:
            self.needs_rescale = False
        self.original_action_dtype = env.action_space.dtype

        self._action_space = Box(
            low=-1,
            high=1,
            shape=env.action_space.shape,
            dtype=np.float32,
            seed=action_seed
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs.astype(np.float32, copy=False), info

    def step(self, action: np.ndarray):
        action = action.astype(self.original_action_dtype)
        if self.needs_rescale:
            action *= self.original_action_half_range
            action += self.original_action_center
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs.astype(np.float32, copy=False), reward, terminated, truncated, info

def create_env(
    name: str,
    seed: int,
    action_seed: int = 0,
    *,
    dummy_action_dim: int = 0,
    dummy_action_alpha: float = 0.0,
    backend: str = "gymnasium",
):
    if backend == "mjx":
        from relax.env.mjx_wrapper import BraxGymnasiumWrapper
        env = BraxGymnasiumWrapper(name, seed=seed)
        if dummy_action_dim and dummy_action_dim > 0:
            env = DummyActionPenaltyWrapper(env, dummy_action_dim=int(dummy_action_dim), dummy_action_alpha=float(dummy_action_alpha))
        env = RelaxWrapper(env, action_seed)
        return env, env.obs_dim, env.act_dim
    env = make(name)
    if dummy_action_dim and dummy_action_dim > 0:
        env = DummyActionPenaltyWrapper(env, dummy_action_dim=int(dummy_action_dim), dummy_action_alpha=float(dummy_action_alpha))
    env.reset(seed=seed)
    env = RelaxWrapper(env, action_seed)
    return env, env.obs_dim, env.act_dim

def create_vector_env(name: str, num_envs: int, seed: int, action_seed: int = 0, mode: str = "serial", backend: str = "gymnasium", **kwargs):
    if backend == "mjx":
        from relax.env.mjx_wrapper import BraxVectorEnv
        dummy_action_dim = int(kwargs.pop("dummy_action_dim", 0) or 0)
        dummy_action_alpha = float(kwargs.pop("dummy_action_alpha", 0.0) or 0.0)
        env = BraxVectorEnv(name, num_envs, seed)
        if dummy_action_dim and dummy_action_dim > 0:
            env = DummyActionPenaltyVectorWrapper(env, dummy_action_dim=dummy_action_dim, dummy_action_alpha=dummy_action_alpha)
        env = RelaxWrapper(env, action_seed)
        return env, env.obs_dim, env.act_dim
    Impl = {
        "serial": SerialVectorEnv,
        "gym": GymProcessVectorEnv,
        "pipe": PipeProcessVectorEnv,
        "spinlock": SpinlockProcessVectorEnv,
        "futex": FutexProcessVectorEnv,
    }[mode]
    dummy_action_dim = int(kwargs.pop("dummy_action_dim", 0) or 0)
    dummy_action_alpha = float(kwargs.pop("dummy_action_alpha", 0.0) or 0.0)
    env = Impl(name, num_envs, seed, **kwargs)
    if dummy_action_dim and dummy_action_dim > 0:
        env = DummyActionPenaltyVectorWrapper(env, dummy_action_dim=dummy_action_dim, dummy_action_alpha=dummy_action_alpha)
    env = RelaxWrapper(env, action_seed)
    return env, env.obs_dim, env.act_dim
