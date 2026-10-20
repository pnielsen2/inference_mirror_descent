"""Distillation buffer: the tilted targets the diffusion policy is regressed onto.

Every update denoises ``K = --num_denoised_actions`` tilted actions per state of
the minibatch, i.e. a *block* of ``n = --batch_size * K`` fresh ``(s', a')``
pairs. By default that block is score-matched once and thrown away. With
``--distillation_buffer_size B`` it is instead ring-written into a buffer of the
last ``B`` blocks (capacity ``n * B``), and the policy is distilled off the whole
buffer, so every pair is regressed on over ``B`` updates rather than one -- more
gradient steps per denoiser pass, which is the expensive half of an update.

``--distillation_steps E`` sets how many passes over the buffer one update
takes. Ordinary distillation cuts each pass into ``--batch_size`` minibatches,
so an update takes ``E * K * B`` optimizer steps. Boltzmann reweighting instead
uses the full weighted buffer as one batch and takes ``E`` optimizer steps.
Single-candidate Boltzmann/best selection stores one action per state.

Staleness is what is being traded away: a target written ``B`` updates ago was
drawn from the policy tilted by an older Q, so it pulls the policy toward where
it was heading then. ``B`` is the horizon over which that is judged acceptable.
"""
import jax, jax.numpy as jnp
import optax


# DPMD updates its noise controller every 250 updates at 5 env steps/update.
DPMD_NOISE_PERIOD = 1250


def dpmd_noise_schedule(act_dim, total_step):
    """Tabulate DPMD's data-independent Gaussian std, including its initial 0.5.

    Its entropy surrogate is d * (log(0.1 * exp(log_alpha)) + log(2*pi*e)/2);
    Adam(0.007) drives it toward -0.9*d, starting at log_alpha=log(5).
    Index 0 precedes update 0; subsequent entries follow controller updates.
    Keeping the clock in env steps preserves the curve when changing UTD.
    """
    optim = optax.adam(0.007)
    initial = jnp.log(jnp.float32(5.0))

    def step(carry, _):
        log_alpha, opt_state = carry
        grad = act_dim * (log_alpha + jnp.log(0.1) + 0.5 * jnp.log(2 * jnp.pi * jnp.e) + 0.9)
        delta, opt_state = optim.update(grad, opt_state)
        log_alpha = optax.apply_updates(log_alpha, delta)
        return (log_alpha, opt_state), 0.1 * jnp.exp(log_alpha)

    _, std = jax.lax.scan(step, (initial, optim.init(initial)), None,
                         length=(total_step + DPMD_NOISE_PERIOD - 1) // DPMD_NOISE_PERIOD + 1)
    return jnp.concatenate((jnp.array([0.5], jnp.float32), std))


def push(buffer, rows, step):
    """Ring-write this update's ``rows`` ``[n, obs_dim + act_dim]`` into ``buffer``.

    Block ``step % B`` is overwritten, so the buffer holds the last ``B`` blocks
    (``B`` is read off the shapes). Written on every update, including those where
    the ``--delay_policy_update`` gate keeps the policy from stepping, so no
    denoised target is ever discarded.
    """
    blocks = buffer.shape[0] // rows.shape[0]
    if blocks == 1:
        return rows  # capacity is one block: the buffer IS this update's targets
    # The first update fills every block with itself, so no gradient step ever
    # sees an unwritten row (a zero action at a zero state) as a target.
    return jnp.where(step == 0, jnp.tile(rows, (blocks, 1)), jax.lax.dynamic_update_slice(
        buffer, rows, (jnp.remainder(step, blocks) * rows.shape[0], 0)))


def epoch_batches(buffer, key, *, batch_size, epochs):
    """``epochs`` reshuffled passes over ``buffer``, cut into ``batch_size`` rows each.

    Returns all ``epochs * len(buffer) // batch_size`` minibatches in order, so the
    caller's optimizer loop stays a flat ``for``. One key per pass via ``fold_in``.
    """
    per_epoch = buffer.shape[0] // batch_size
    if per_epoch == 1:
        return [buffer] * epochs  # the single minibatch of a pass IS the buffer
    out = []
    for e in range(epochs):
        perm = jax.random.permutation(jax.random.fold_in(key, e), buffer.shape[0])
        out += [buffer[perm[m * batch_size:(m + 1) * batch_size]] for m in range(per_epoch)]
    return out
