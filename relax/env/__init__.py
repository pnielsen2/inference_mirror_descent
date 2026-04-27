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

class _PerEntrySampledBox:
    """Duck-typed stand-in for gym.spaces.Box whose ``sample()`` concatenates
    samples from N independent sub-Boxes (one per vmap entry), each with its
    own RNG seeded from a per-entry master. Only used for the normalized
    [-1, 1] action space built by ``RelaxWrapper`` -- its only consumer is
    ``self.env.action_space.sample()`` in the trainers, so we implement just
    enough of the Box API for that.
    """
    def __init__(self, total_envs, act_dim, per_entry_action_seeds,
                 num_vec_envs_per_entry, dtype=np.float32):
        assert len(per_entry_action_seeds) * num_vec_envs_per_entry == total_envs
        self.shape = (total_envs, act_dim)
        self.dtype = dtype
        self.low = np.full(self.shape, -1.0, dtype=dtype)
        self.high = np.full(self.shape, 1.0, dtype=dtype)
        self._subs = [
            Box(low=-1.0, high=1.0,
                shape=(num_vec_envs_per_entry, act_dim),
                dtype=dtype, seed=int(s))
            for s in per_entry_action_seeds
        ]

    def sample(self):
        return np.concatenate([sub.sample() for sub in self._subs], axis=0)

    def contains(self, x):
        return bool(np.all(x >= -1.0) and np.all(x <= 1.0)
                    and tuple(x.shape) == self.shape)


class RelaxWrapper(Wrapper):
    def __init__(self, env: Env, action_seed: int = 0, *,
                 per_entry_action_seeds: list = None,
                 num_vec_envs_per_entry: int = None):
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

        if per_entry_action_seeds is not None:
            assert isinstance(env, VectorEnv), \
                "per_entry_action_seeds is only supported for VectorEnv"
            assert num_vec_envs_per_entry is not None
            total_envs, _ = env.action_space.shape
            self._action_space = _PerEntrySampledBox(
                total_envs=total_envs,
                act_dim=self.act_dim,
                per_entry_action_seeds=per_entry_action_seeds,
                num_vec_envs_per_entry=num_vec_envs_per_entry,
                dtype=np.float32,
            )
        else:
            self._action_space = Box(
                low=-1,
                high=1,
                shape=env.action_space.shape,
                dtype=np.float32,
                seed=action_seed,
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

def create_vector_env(name: str, num_envs: int, seed: int, action_seed: int = 0,
                      mode: str = "serial", backend: str = "gymnasium",
                      *,
                      per_entry_env_seeds: list = None,
                      per_entry_action_seeds: list = None,
                      **kwargs):
    """Build a VectorEnv of size ``num_envs``.

    When ``per_entry_env_seeds`` is provided (list of length ``N`` with
    ``num_envs == N * num_vec_envs_per_entry``), the returned VectorEnv's
    per-env seeds are chosen to match what ``N`` standalone runs, each with
    its own master seed from the list, would have produced for their own
    length-``num_vec_envs_per_entry`` VectorEnvs. Similarly,
    ``per_entry_action_seeds`` gives each vmap entry its own random-action
    RNG for the warmup sampler. Together these make a vmapped pack behave
    exactly like running each entry standalone -- at the env/action-seed
    sites, not just buffer/network init."""
    dummy_action_dim = int(kwargs.pop("dummy_action_dim", 0) or 0)
    dummy_action_alpha = float(kwargs.pop("dummy_action_alpha", 0.0) or 0.0)

    num_vec_envs_per_entry = None
    seeds_override = None
    if per_entry_env_seeds is not None:
        N = len(per_entry_env_seeds)
        assert num_envs % N == 0, \
            f"num_envs={num_envs} must be divisible by len(per_entry_env_seeds)={N}"
        num_vec_envs_per_entry = num_envs // N
        # Each entry's per-env seeds are derived exactly as a standalone
        # VectorEnv of size num_vec_envs_per_entry would derive them.
        flat = []
        for env_s in per_entry_env_seeds:
            if num_vec_envs_per_entry > 1:
                rng_i = np.random.default_rng(int(env_s))
                flat.extend(rng_i.integers(0, 2**32 - 1, num_vec_envs_per_entry).tolist())
            else:
                flat.append(int(env_s))
        seeds_override = flat

    if per_entry_action_seeds is not None:
        assert per_entry_env_seeds is not None, \
            "per_entry_action_seeds requires per_entry_env_seeds (they come in pairs)"
        assert len(per_entry_action_seeds) == len(per_entry_env_seeds)

    if backend == "mjx":
        from relax.env.mjx_wrapper import BraxVectorEnv
        if per_entry_env_seeds is not None:
            raise NotImplementedError(
                "per_entry_env_seeds is not yet supported for backend='mjx'"
            )
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
    impl_kwargs = dict(kwargs)
    if seeds_override is not None:
        impl_kwargs["seeds_override"] = seeds_override
    env = Impl(name, num_envs, seed, **impl_kwargs)
    if dummy_action_dim and dummy_action_dim > 0:
        env = DummyActionPenaltyVectorWrapper(env, dummy_action_dim=dummy_action_dim, dummy_action_alpha=dummy_action_alpha)
    if per_entry_action_seeds is not None:
        env = RelaxWrapper(env, action_seed,
                           per_entry_action_seeds=per_entry_action_seeds,
                           num_vec_envs_per_entry=num_vec_envs_per_entry)
    else:
        env = RelaxWrapper(env, action_seed)
    return env, env.obs_dim, env.act_dim
