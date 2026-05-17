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



