from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class SASRStandardizer:
    mu_s: jnp.ndarray
    std_s: jnp.ndarray
    mu_a: jnp.ndarray
    std_a: jnp.ndarray
    mu_r: jnp.ndarray
    std_r: jnp.ndarray

    def std_s_vec(self, s: jnp.ndarray) -> jnp.ndarray:
        return (s - self.mu_s) / jnp.maximum(self.std_s, 1e-6)

    def std_a_vec(self, a: jnp.ndarray) -> jnp.ndarray:
        return (a - self.mu_a) / jnp.maximum(self.std_a, 1e-6)

    def std_r_scalar(self, r: jnp.ndarray) -> jnp.ndarray:
        return (r - self.mu_r) / jnp.maximum(self.std_r, 1e-6)

    def unstd_a_vec(self, a_std: jnp.ndarray) -> jnp.ndarray:
        return a_std * jnp.maximum(self.std_a, 1e-6) + self.mu_a


def _sasr_standardizer_flatten(s: SASRStandardizer):
    children = (s.mu_s, s.std_s, s.mu_a, s.std_a, s.mu_r, s.std_r)
    aux_data = None
    return children, aux_data


def _sasr_standardizer_unflatten(aux_data, children):
    mu_s, std_s, mu_a, std_a, mu_r, std_r = children
    return SASRStandardizer(mu_s=mu_s, std_s=std_s, mu_a=mu_a, std_a=std_a, mu_r=mu_r, std_r=std_r)


jax.tree_util.register_pytree_node(
    SASRStandardizer,
    _sasr_standardizer_flatten,
    _sasr_standardizer_unflatten,
)
