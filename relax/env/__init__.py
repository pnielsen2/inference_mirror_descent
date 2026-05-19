import numpy as np
from gymnasium import Env, Wrapper
from gymnasium.spaces import Box

from relax.env.vector import VectorEnv, FutexProcessVectorEnv

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

    def get_current_obs(self) -> np.ndarray:
        return self.env.get_current_obs().astype(np.float32, copy=False)

    def step(self, action: np.ndarray):
        action = action.astype(self.original_action_dtype)
        if self.needs_rescale:
            action *= self.original_action_half_range
            action += self.original_action_center
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs.astype(np.float32, copy=False), reward, terminated, truncated, info

def create_vector_env(name: str, num_envs: int, seed: int, action_seed: int = 0,
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

    impl_kwargs = dict(kwargs)
    if seeds_override is not None:
        impl_kwargs["seeds_override"] = seeds_override
    env = FutexProcessVectorEnv(name, num_envs, seed, **impl_kwargs)
    if per_entry_action_seeds is not None:
        env = RelaxWrapper(env, action_seed,
                           per_entry_action_seeds=per_entry_action_seeds,
                           num_vec_envs_per_entry=num_vec_envs_per_entry)
    else:
        env = RelaxWrapper(env, action_seed)
    return env, env.obs_dim, env.act_dim
