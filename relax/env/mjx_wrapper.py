"""Gymnasium-compatible wrapper around Brax environments with MJX backend.

This module provides :class:`BraxGymnasiumWrapper` (single-env) and
:class:`BraxVectorEnv` (batched/vmapped) so that the rest of the ``relax``
codebase can use MJX-accelerated environments without changing the training
loop.

The wrapper also exposes :meth:`differentiable_step` which returns pure JAX
arrays and is traceable through ``jax.grad`` / ``jax.jacrev``.

Important: We deliberately do **not** use Brax's ``AutoResetWrapper`` because
it replaces the terminal observation with the reset observation, which is
incompatible with the Gymnasium convention where ``step()`` returns the true
terminal obs and the caller invokes ``reset()`` explicitly.
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box

from relax.env.vector.base import VectorEnv

# Gymnasium env name -> Brax env name
GYMNASIUM_TO_BRAX = {
    "HalfCheetah-v4": "halfcheetah",
    "HalfCheetah-v3": "halfcheetah",
    "HalfCheetah-v2": "halfcheetah",
    "Hopper-v4": "hopper",
    "Hopper-v3": "hopper",
    "Hopper-v2": "hopper",
    "Walker2d-v4": "walker2d",
    "Walker2d-v3": "walker2d",
    "Walker2d-v2": "walker2d",
    "Ant-v4": "ant",
    "Ant-v3": "ant",
    "Ant-v2": "ant",
    "Humanoid-v4": "humanoid",
    "Humanoid-v3": "humanoid",
    "Humanoid-v2": "humanoid",
    "HumanoidStandup-v4": "humanoidstandup",
    "HumanoidStandup-v2": "humanoidstandup",
    "Swimmer-v4": "swimmer",
    "Swimmer-v3": "swimmer",
    "Swimmer-v2": "swimmer",
    "Reacher-v4": "reacher",
    "Reacher-v2": "reacher",
    "Pusher-v4": "pusher",
    "Pusher-v2": "pusher",
    "InvertedPendulum-v4": "inverted_pendulum",
    "InvertedPendulum-v2": "inverted_pendulum",
    "InvertedDoublePendulum-v4": "inverted_double_pendulum",
    "InvertedDoublePendulum-v2": "inverted_double_pendulum",
}


def _resolve_brax_name(gym_name: str) -> str:
    if gym_name in GYMNASIUM_TO_BRAX:
        return GYMNASIUM_TO_BRAX[gym_name]
    # Allow passing brax names directly
    return gym_name


def _create_brax_env(gym_name: str, episode_length: int = 1000, **kwargs):
    """Create a Brax PipelineEnv with MJX backend + EpisodeWrapper only.

    We intentionally skip ``AutoResetWrapper`` so that ``step()`` returns the
    true terminal observation (matching Gymnasium semantics).
    """
    import brax.envs as brax_envs
    from brax.envs.wrappers import training

    brax_name = _resolve_brax_name(gym_name)
    raw_env = brax_envs.get_environment(brax_name, backend="mjx", **kwargs)
    env = training.EpisodeWrapper(raw_env, episode_length=episode_length, action_repeat=1)
    return raw_env, env


class _SpecStub:
    """Minimal stand-in for gymnasium ``EnvSpec`` so that code accessing
    ``env.spec.id`` (e.g. wandb group naming) keeps working.

    Includes ``additional_wrappers`` so that Gymnasium 1.0's
    ``Wrapper.spec`` property can ``deepcopy`` this object without error.
    """

    def __init__(self, env_id: str):
        self.id = env_id
        self.additional_wrappers = ()


class BraxGymnasiumWrapper(Env):
    """Wraps a single Brax MJX environment behind the Gymnasium ``Env`` API.

    Internally keeps a Brax ``State`` and converts between JAX ↔ NumPy at the
    boundary so that the rest of the codebase (which expects NumPy obs/actions)
    works unchanged.

    Parameters
    ----------
    gym_name : str
        Gymnasium-style environment name (e.g. ``"HalfCheetah-v4"``).
    seed : int
        Seed for the initial ``reset`` and for the PRNG used by subsequent
        resets.
    episode_length : int
        Maximum episode length (Brax truncates after this many steps).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        gym_name: str,
        seed: int = 0,
        episode_length: int = 1000,
    ):
        self._gym_name = gym_name
        self._brax_env_raw, self._brax_env = _create_brax_env(
            gym_name, episode_length=episode_length
        )

        # Discover sizes via a dummy reset
        dummy_rng = jax.random.PRNGKey(0)
        dummy_state = jax.jit(self._brax_env.reset)(dummy_rng)
        obs_size = int(dummy_state.obs.shape[-1])
        act_size = int(self._brax_env_raw.action_size)

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(act_size,),
            dtype=np.float32,
            seed=0,
        )

        self.spec = _SpecStub(gym_name)
        self._rng = jax.random.PRNGKey(seed)
        self._state = None  # will be set by reset()

        # JIT-compile step and reset
        self._jit_step = jax.jit(self._brax_env.step)
        self._jit_reset = jax.jit(self._brax_env.reset)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = jax.random.PRNGKey(seed)
        self._rng, reset_rng = jax.random.split(self._rng)
        self._state = self._jit_reset(reset_rng)
        obs = np.asarray(self._state.obs, dtype=np.float32)
        return obs, {}

    def step(self, action):
        action_jax = jnp.asarray(action, dtype=jnp.float32)
        self._state = self._jit_step(self._state, action_jax)

        obs = np.asarray(self._state.obs, dtype=np.float32)
        reward = float(self._state.reward)
        done = bool(self._state.done)

        # EpisodeWrapper sets done=True either when the env itself terminates
        # (e.g. Hopper falling) or when the episode length limit is hit.
        # It stores a ``truncation`` flag in info to distinguish the two.
        truncation = float(self._state.info.get("truncation", 0.0))
        if done and truncation > 0.5:
            terminated = False
            truncated = True
        else:
            terminated = done
            truncated = False

        info = {k: float(v) for k, v in self._state.metrics.items()}
        return obs, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        pass

    @property
    def unwrapped(self):
        return self

    # ------------------------------------------------------------------
    # Differentiable interface (pure JAX, no NumPy conversion)
    # ------------------------------------------------------------------

    @property
    def brax_env(self):
        """Access the underlying Brax environment for direct JAX use."""
        return self._brax_env

    @property
    def brax_env_raw(self):
        """Access the raw (unwrapped) Brax PipelineEnv."""
        return self._brax_env_raw

    @property
    def brax_state(self):
        """Current Brax ``State`` (JAX arrays)."""
        return self._state

    def differentiable_step(self, state, action):
        """JAX-traceable single step.

        Supports **forward-mode** AD (``jax.jvp``, ``jax.jacfwd``).
        Reverse-mode (``jax.grad``, ``jax.jacrev``) is **not** supported
        because MJX's constraint solver uses ``lax.while_loop`` with
        dynamic bounds.

        Parameters
        ----------
        state : brax.envs.base.State
            Current Brax state.
        action : jax.Array
            Action array of shape ``(act_dim,)``.

        Returns
        -------
        next_state : brax.envs.base.State
            Next Brax state (obs, reward, done, etc. are all JAX arrays).
        """
        return self._brax_env.step(state, action)

    def differentiable_reset(self, rng):
        """JAX-traceable reset.

        Parameters
        ----------
        rng : jax.Array
            JAX PRNG key.

        Returns
        -------
        state : brax.envs.base.State
        """
        return self._brax_env.reset(rng)


class BraxVectorEnv(VectorEnv):
    """Batched Brax MJX environment using ``jax.vmap``.

    Much faster than :class:`SerialVectorEnv` since all ``num_envs``
    environments are stepped in a single vectorised JAX call on the GPU.

    Selective auto-reset: when any sub-env is done after ``step()``, the
    next call to ``reset()`` only re-initialises the finished sub-envs
    (matching ``SerialVectorEnv`` behaviour).

    Parameters
    ----------
    gym_name : str
        Gymnasium-style environment name.
    num_envs : int
        Number of parallel environments.
    seed : int
        Base seed; each sub-env gets a unique derived seed.
    episode_length : int
        Maximum episode length.
    """

    def __init__(
        self,
        gym_name: str,
        num_envs: int,
        seed: int,
        episode_length: int = 1000,
    ):
        assert num_envs > 0
        self.num_envs = num_envs
        self._gym_name = gym_name

        self._brax_env_raw, self._brax_env = _create_brax_env(
            gym_name, episode_length=episode_length
        )

        # Discover sizes
        dummy_rng = jax.random.PRNGKey(0)
        dummy_state = jax.jit(self._brax_env.reset)(dummy_rng)
        obs_size = int(dummy_state.obs.shape[-1])
        act_size = int(self._brax_env_raw.action_size)

        self.obs_dim = obs_size
        self.act_dim = act_size

        self.single_observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.single_action_space = Box(
            low=-1.0, high=1.0, shape=(act_size,), dtype=np.float32
        )
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_envs, obs_size),
            dtype=np.float32,
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(num_envs, act_size),
            dtype=np.float32,
        )

        self.spec = _SpecStub(gym_name)

        # Build vmapped step / reset
        self._vmap_reset = jax.jit(jax.vmap(self._brax_env.reset))
        self._vmap_step = jax.jit(jax.vmap(self._brax_env.step))

        # JIT a selective-reset function: given old states, new (reset) states,
        # and a boolean mask, return the merged pytree.
        @jax.jit
        def _selective_reset(old_states, new_states, shall_reset):
            """Replace sub-env states where shall_reset is True."""
            def _where(old_leaf, new_leaf):
                # Broadcast the boolean mask to the leaf shape.
                mask = shall_reset
                while mask.ndim < old_leaf.ndim:
                    mask = mask[..., None]
                return jnp.where(mask, new_leaf, old_leaf)
            return jax.tree.map(_where, old_states, new_states)

        self._selective_reset = _selective_reset

        self._rng = jax.random.PRNGKey(seed)
        self._states = None  # pytree with leading dim num_envs
        self._shall_reset = np.ones((num_envs,), dtype=np.bool_)

    # ------------------------------------------------------------------
    # VectorEnv / Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = jax.random.PRNGKey(seed)
        self._rng, reset_rng = jax.random.split(self._rng)
        rngs = jax.random.split(reset_rng, self.num_envs)

        if self._states is None or np.all(self._shall_reset):
            # First reset or all envs done → full reset
            self._states = self._vmap_reset(rngs)
        else:
            # Selective reset: only re-initialise done sub-envs
            fresh_states = self._vmap_reset(rngs)
            mask_jax = jnp.asarray(self._shall_reset)
            self._states = self._selective_reset(
                self._states, fresh_states, mask_jax
            )

        self._shall_reset[:] = False
        obs = np.asarray(self._states.obs, dtype=np.float32)
        return obs, {}

    def step(self, action: np.ndarray):
        action_jax = jnp.asarray(action, dtype=jnp.float32)
        self._states = self._vmap_step(self._states, action_jax)

        obs = np.asarray(self._states.obs, dtype=np.float32)
        reward = np.asarray(self._states.reward, dtype=np.float64)
        done = np.asarray(self._states.done, dtype=np.bool_)

        terminated = done.copy()
        truncated = np.zeros_like(done)

        # Distinguish termination from truncation using EpisodeWrapper's flag
        if "truncation" in self._states.info:
            trunc_flags = np.asarray(self._states.info["truncation"]) > 0.5
            truncated = trunc_flags
            terminated = done & ~trunc_flags

        self._shall_reset = done
        return obs, reward, terminated, truncated, {}

    def close(self):
        pass

    @property
    def unwrapped(self):
        return self

    # ------------------------------------------------------------------
    # Differentiable interface
    # ------------------------------------------------------------------

    @property
    def brax_env(self):
        return self._brax_env

    @property
    def brax_env_raw(self):
        return self._brax_env_raw

    @property
    def brax_states(self):
        return self._states

    def differentiable_step(self, states, actions):
        """Vmapped JAX-traceable step. Supports forward-mode AD (``jax.jacfwd``)."""
        return jax.vmap(self._brax_env.step)(states, actions)

    def differentiable_reset(self, rngs):
        """Vmapped JAX-traceable reset."""
        return jax.vmap(self._brax_env.reset)(rngs)
