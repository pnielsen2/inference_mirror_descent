import os
from pathlib import Path
import pickle

import numpy as np
import jax, jax.numpy as jnp

from relax.utils.experience import Experience
from relax.utils.persistence import make_persist
from relax.utils.typing_utils import Metric


def _split_info(info):
    """Split a metric dict into (scalar_info, array_info) with one host sync.

    All 0-d jax scalars are stacked on device, transferred in a single D→H copy,
    and unpacked on host as python floats. Arrays are transferred individually.
    """
    scalar_keys = []
    scalar_vals = []
    array_info = {}
    for k, v in info.items():
        if jnp.ndim(v) == 0:
            scalar_keys.append(k)
            scalar_vals.append(v)
        else:
            array_info[k] = np.asarray(v)
    if scalar_keys:
        stacked = np.asarray(jnp.stack(scalar_vals))  # single D→H sync
        scalar_info = {k: float(stacked[i]) for i, k in enumerate(scalar_keys)}
    else:
        scalar_info = {}
    return scalar_info, array_info


def _split_info_vmap(info):
    """Vmapped counterpart of _split_info.

    Under vmap, every metric that was 0-d becomes 1-d with a leading seed axis.
    Returns (scalar_info, array_info) where scalar values are np.ndarray[N]
    (one per seed) and array values are np.ndarray of shape [N, ...].
    """
    scalar_keys = []
    scalar_vals = []
    array_info = {}
    for k, v in info.items():
        if jnp.ndim(v) == 1:
            scalar_keys.append(k)
            scalar_vals.append(v)
        else:
            array_info[k] = np.asarray(v)
    if scalar_keys:
        stacked = np.asarray(jnp.stack(scalar_vals))  # [num_scalars, N]
        scalar_info = {k: np.asarray(stacked[i]) for i, k in enumerate(scalar_keys)}
    else:
        scalar_info = {}
    return scalar_info, array_info


class Algorithm:
    # NOTE: a not elegant blanket implementation of the algorithm interface
    def _implement_common_behavior(self, stateless_update, stateless_get_action, stateless_get_deterministic_action, stateless_get_value=None):
        self._update = jax.jit(stateless_update)
        self._get_action = jax.jit(stateless_get_action)
        self._get_deterministic_action = jax.jit(stateless_get_deterministic_action)
        if stateless_get_value is not None:
            self._get_value = jax.jit(stateless_get_value)
        # Store the un-jitted stateless fns so vmap-wrappers can compose
        # cleanly (jit-of-vmap instead of vmap-of-jit).
        self._stateless_update = stateless_update
        self._stateless_get_action = stateless_get_action
        self._stateless_get_deterministic_action = stateless_get_deterministic_action
        self._stateless_get_value = stateless_get_value
        self._update_vmap = None
        self._get_action_vmap_fn = None
        self._get_deterministic_action_vmap_fn = None

    def _ensure_vmap_compiled(self):
        """Lazily build vmapped+jitted stateless fns. Idempotent."""
        if self._update_vmap is None:
            self._update_vmap = jax.jit(jax.vmap(self._stateless_update))
        if self._get_action_vmap_fn is None:
            self._get_action_vmap_fn = jax.jit(jax.vmap(self._stateless_get_action))
        if self._get_deterministic_action_vmap_fn is None:
            self._get_deterministic_action_vmap_fn = jax.jit(
                jax.vmap(self._stateless_get_deterministic_action)
            )

    def update(self, key: jax.Array, data: Experience) -> Metric:
        self.state, info = self._update(key, self.state, data)
        return _split_info(info)

    def update_vmap(self, key: jax.Array, data: Experience) -> Metric:
        """Vmapped update. key/state/data must have a leading seed axis [N]."""
        self._ensure_vmap_compiled()
        self.state, info = self._update_vmap(key, self.state, data)
        return _split_info_vmap(info)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        action = self._get_action(key, self.get_policy_params(), obs)
        return np.asarray(action)

    def get_deterministic_action(self, obs: np.ndarray) -> np.ndarray:
        action = self._get_deterministic_action(self.get_policy_params(), obs)
        return np.asarray(action)

    def get_value(self, obs: np.ndarray) -> np.ndarray:
        value = self._get_value(self.get_value_params(), obs)
        return np.asarray(value)

    def save(self, path: str) -> None:
        state = jax.device_get(self.state)
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.state = jax.device_put(state)

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params())
        with open(path, "wb") as f:
            pickle.dump(policy, f)
            
    def save_q(self, path: str) -> None:
        policy = jax.device_get(self.get_value_params())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def save_policy_structure(self, root: os.PathLike, dummy_obs: jax.Array) -> None:
        root = Path(root)

        key = jax.random.key(0)
        stochastic = make_persist(self._get_action._fun)(key, self.get_policy_params(), dummy_obs)
        deterministic = make_persist(self._get_deterministic_action._fun)(self.get_policy_params(), dummy_obs)

        stochastic.save(root / "stochastic.pkl")
        stochastic.save_info(root / "stochastic.txt")
        deterministic.save(root / "deterministic.pkl")
        deterministic.save_info(root / "deterministic.txt")

    def save_q_structure(self, root: os.PathLike, dummy_obs: jax.Array, dummy_action: jax.Array) -> None:
        root = Path(root)

        key = jax.random.key(0)
        # stochastic = make_persist(self._get_value._fun)(key, self.get_policy_params(), dummy_obs)
        deterministic = make_persist(self._get_value._fun)(self.get_value_params(), dummy_obs, dummy_action) # []

        # stochastic.save(root / "stochastic.pkl")
        # stochastic.save_info(root / "stochastic.txt")
        deterministic.save(root / "q_func.pkl")
        deterministic.save_info(root / "q_func.txt")

    def get_policy_params(self):
        return self.state.params.policy

    def get_value_params(self):
        return self.state.params.value

    def warmup(self, data: Experience) -> None:
        key = jax.random.key(0)
        obs = data.obs[0]
        policy_params = self.get_policy_params()
        self._update(key, self.state, data)
        self._get_action(key, policy_params, obs)
        self._get_deterministic_action(policy_params, obs)

    def warmup_vmap(self, data: Experience, N: int) -> None:
        """Trigger JIT tracing for the vmapped entry points. ``data`` has a
        leading [N] seed axis. ``self.state`` must already be vmap-stacked."""
        self._ensure_vmap_compiled()
        key = jax.random.split(jax.random.key(0), N)
        obs = data.obs[:, 0]  # [N, obs_dim] — one obs vector per seed
        policy_params = self.get_policy_params()
        self._update_vmap(key, self.state, data)
        self._get_action_vmap_fn(key, policy_params, obs)
        self._get_deterministic_action_vmap_fn(policy_params, obs)

    def compute_validation_metrics(self, key: jax.Array, data: Experience) -> Metric:
        """Compute validation metrics on a batch of data without updating state.

        By default, this reuses the stateless update function stored in
        ``self._update`` to obtain the same metric dictionary as a training
        step, but discards the updated state. Algorithms that require a
        different notion of validation can override this method.
        """

        _, info = self._update(key, self.state, data)
        scalar_info, _ = _split_info(info)
        return scalar_info

    def get_effective_hparams(self) -> dict:
        """Return a dict of effective hyperparameters for logging.

        Subclasses can override this to expose any internal hyperparameters,
        including parameters that are overridden relative to the raw CLI args
        (e.g., clamped values, derived quantities, or algorithm-specific
        interpretations). The default implementation returns an empty dict.
        """

        return {}
