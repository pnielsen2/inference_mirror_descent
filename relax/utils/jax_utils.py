import jax, jax.numpy as jnp
import jax.scipy as jsp
import haiku as hk

def mask_average(x: jax.Array, mask: jax.Array) -> jax.Array:
    return jnp.sum(x * mask) / jnp.maximum(jnp.sum(mask), 1)

def fix_repr(cls):
    """Delete haiku's auto-generated __repr__ method, in favor of dataclass's one"""
    del cls.__repr__
    postinit = getattr(cls, "__post_init__")
    def __post_init__(self):
        postinit(self)
        if hk.running_init():
            print(self)
    cls.__post_init__ = __post_init__
    return cls

def is_broadcastable(src, dst):
    try:
        return jnp.broadcast_shapes(src, dst) == dst
    except ValueError:
        return False

def random_key_from_data(data: jax.Array) -> jax.Array:
    # Create a random key deterministically from data, like hashing
    mean = jnp.mean(data)
    std = jnp.std(data)
    seed = (mean * std).view(jnp.uint32)
    key = jax.random.key(seed)
    return key


def latent_to_action_normalcdf(z: jax.Array, eps: float = 1e-6) -> jax.Array:
    u = jnp.float32(0.5) * (jnp.float32(1.0) + jsp.special.erf(z / jnp.sqrt(jnp.float32(2.0))))
    a = jnp.float32(2.0) * u - jnp.float32(1.0)
    if eps is None:
        return a
    eps_f = jnp.float32(eps)
    return jnp.clip(a, -jnp.float32(1.0) + eps_f, jnp.float32(1.0) - eps_f)


def action_to_latent_normalcdf(a: jax.Array, eps: float = 1e-6) -> jax.Array:
    p = (a + jnp.float32(1.0)) * jnp.float32(0.5)
    eps_f = jnp.float32(eps)
    p = jnp.clip(p, eps_f, jnp.float32(1.0) - eps_f)
    return jnp.sqrt(jnp.float32(2.0)) * jsp.special.erfinv(jnp.float32(2.0) * p - jnp.float32(1.0))
