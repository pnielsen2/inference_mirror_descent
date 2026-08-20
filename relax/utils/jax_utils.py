import jax, jax.numpy as jnp
import haiku as hk
import optax

def mask_average(x: jax.Array, mask: jax.Array) -> jax.Array:
    return jnp.sum(x * mask) / jnp.maximum(jnp.sum(mask), 1)


def stack_trees(trees):
    """Stack a list of pytrees of identical structure along a new leading axis."""
    return jax.tree.map(lambda *xs: jnp.stack(xs), *trees)


def unstack_tree(stacked, n: int):
    """Inverse of :func:`stack_trees`: split a stacked pytree back into a list of N."""
    return tuple(jax.tree.map(lambda x: x[i], stacked) for i in range(n))


def is_due(step, delay):
    """``step`` is a multiple of ``delay``, i.e. this update fires.

    ``delay`` may be a traced per-seed scalar (update periods are packed as
    float32 hps so they are vmappable "easy" ablations); it is floored to an
    integer >= 1, so a 0 period degrades to "every step" instead of a
    mod-by-zero. A non-firing step is a pure no-op via the callers' ``cond``.
    """
    return step % jnp.maximum(jnp.asarray(delay).astype(jnp.int32), 1) == 0


def delayed_param_update(optim, params, grads, opt_state, lr, step, delay):
    """Apply ``optim.update`` -> scale by ``-lr`` -> ``optax.apply_updates``,
    but only when ``step % delay == 0``. Otherwise pass ``(params, opt_state)``
    through unchanged. PRNG-free; the caller supplies a per-seed ``lr`` so this
    works under vmap.
    """
    def do_update(po):
        update, new_opt_state = optim.update(grads, po[1], params=po[0])
        update = jax.tree.map(lambda u: -lr * u, update)
        new_params = optax.apply_updates(po[0], update)
        return new_params, new_opt_state
    return jax.lax.cond(
        is_due(step, delay),
        do_update,
        lambda po: po,
        (params, opt_state),
    )


def delayed_target_update(params, target_params, polyak_tau, step, delay):
    """Polyak-averaged target update gated by ``step % delay == 0``."""
    return jax.lax.cond(
        is_due(step, delay),
        lambda tp: optax.incremental_update(params, tp, polyak_tau), # Polyak Update
        lambda tp: tp,
        target_params,
    )

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


