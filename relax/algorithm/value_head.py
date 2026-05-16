"""V(s) network used by ``critic_normalization='ema'`` / KL-budget mode.

Owns the haiku-transformed value network, the inferred ``obs_dim`` (sniffed
from the Q net's first linear weight), and three small utilities used by
DPMD:

* ``init_params(key)`` / ``init_opt_state(params)`` for vmap-mode setup.
* ``apply(params, obs)`` for the in-jit ``q_mean_from_x`` advantage normalization.
* ``apply_vmap(vparams_N, obs_N)`` for the host-side ``get_action_vmap`` rollout
  block (lazy-jitted on first call).
* ``update_step(state, q_for_v, next_obs, lr_q, optim)`` for the on-policy V TD
  update inside ``stateless_update``.

Behavior is byte-identical to the previous in-DPMD implementation:
``init`` uses ``jax.random.PRNGKey(42)`` exactly; the MLP is three hidden
layers of ``hidden_dim`` with ReLU; loss is ``mean((V - sg(Q_for_V))^2)``
with the same per-seed ``lr_q`` scaling and ``optax.scale_by_adam`` updates.
"""
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import haiku as hk
import optax

from relax.network.blocks import ValueNet


def _sniff_obs_and_hidden(q_params_first, act_dim: int):
    """Infer (obs_dim, hidden_dim) from the first Q-net linear weight.

    Haiku FlatMaps use slash-separated keys like 'q_net/linear',
    'q_net/linear_1', ...; we match exactly the one ending in '/linear' (the
    input layer). Falls back to a MuJoCo-ish (17, 256) when the layout is
    unrecognized so that initialization never crashes.
    """
    first_w = None
    for k, v in q_params_first.items():
        if k.endswith('/linear') and isinstance(v, dict) and 'w' in v:
            w = v['w']
            if hasattr(w, 'shape'):
                first_w = w
                break
    if first_w is not None:
        return first_w.shape[0] - act_dim, first_w.shape[1]
    return 17, 256


@dataclass
class ValueHead:
    obs_dim: int
    hidden_dim: int
    _value_net: hk.Transformed
    _apply_vmap_jit: Optional[object] = None  # set lazily

    @classmethod
    def from_q_params(cls, q_params_first, act_dim: int) -> "ValueHead":
        obs_dim, hidden_dim = _sniff_obs_and_hidden(q_params_first, act_dim)
        value_net = hk.without_apply_rng(
            hk.transform(lambda obs: ValueNet(
                hidden_sizes=(hidden_dim, hidden_dim, hidden_dim),
                activation=jax.nn.relu,
            )(obs))
        )
        return cls(obs_dim=obs_dim, hidden_dim=hidden_dim, _value_net=value_net)

    # --- Initialization -------------------------------------------------
    def init_params(self, key: jax.Array):
        sample_obs = jnp.zeros((1, self.obs_dim))
        return self._value_net.init(key, sample_obs)

    def init_opt_state(self, params, optim: optax.GradientTransformation):
        return optim.init(params)

    # --- In-jit forward (used inside MALA sampler) ----------------------
    def apply(self, params, obs: jax.Array) -> jax.Array:
        return self._value_net.apply(params, obs)

    # --- Host-side vmapped forward (used by get_action_vmap rollout) ----
    def apply_vmap(self, vparams, obs: jax.Array) -> jax.Array:
        if self._apply_vmap_jit is None:
            v_apply = self._value_net.apply

            @jax.jit
            def _vmap_jit(vp, ob):
                def single(vp_i, ob_i):
                    v = v_apply(vp_i, ob_i)
                    if isinstance(v, tuple):
                        v = v[0]
                    return v
                return jax.vmap(single)(vp, ob)
            self._apply_vmap_jit = _vmap_jit
        return self._apply_vmap_jit(vparams, obs)

    # --- In-jit V TD update -------------------------------------------
    def update_step(self, state, q_for_v: jax.Array, next_obs: jax.Array,
                    lr_q: jax.Array, optim: optax.GradientTransformation):
        """One Adam step on V(s') against ``q_for_v`` (with stop-grad).

        Returns ``(value_params_updated, value_opt_state_updated, value_loss)``.
        """
        def value_loss_fn(v_params):
            v_pred = self._value_net.apply(v_params, next_obs)
            return jnp.mean((v_pred - jax.lax.stop_gradient(q_for_v)) ** 2)

        v_loss, v_grads = jax.value_and_grad(value_loss_fn)(state.value_params)
        v_updates, value_opt_state_updated = optim.update(
            v_grads, state.opt_state.value, state.value_params
        )
        v_updates = jax.tree.map(lambda u: -lr_q * u, v_updates)
        value_params_updated = optax.apply_updates(state.value_params, v_updates)
        return value_params_updated, value_opt_state_updated, v_loss
