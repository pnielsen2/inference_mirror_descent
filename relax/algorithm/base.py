import pickle

import numpy as np
import jax, jax.numpy as jnp

from relax.utils.experience import Experience
from relax.utils.typing_utils import Metric


def _split_info_vmap(info):
    """Split a vmapped metric dict into (scalar_info, array_info) with one host sync.

    Under vmap, every metric that was 0-d becomes 1-d with a leading seed axis.
    All such 1-d jax scalars are stacked on device, transferred in a single
    D→H copy, and unpacked on host as np.ndarray[N] (one per seed). Higher-rank
    arrays are transferred individually.
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
        # Store the un-jitted stateless fns so vmap-wrappers can compose
        # cleanly (jit-of-vmap instead of vmap-of-jit).
        self._stateless_update = stateless_update
        self._stateless_get_action = stateless_get_action
        self._stateless_get_deterministic_action = stateless_get_deterministic_action
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

    def update_vmap(self, key: jax.Array, data: Experience) -> Metric:
        """Vmapped update. key/state/data must have a leading seed axis [N]."""
        self._ensure_vmap_compiled()
        self.state, info = self._update_vmap(key, self.state, data)
        return _split_info_vmap(info)

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

    def get_policy_params(self):
        return self.state.params.policy

    def get_value_params(self):
        return self.state.params.value

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

    def get_effective_hparams(self) -> dict:
        """Return a dict of effective hyperparameters for logging.

        Subclasses can override this to expose any internal hyperparameters,
        including parameters that are overridden relative to the raw CLI args
        (e.g., clamped values, derived quantities, or algorithm-specific
        interpretations). The default implementation returns an empty dict.
        """

        return {}
