from typing import NamedTuple, Tuple
 
import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle
 
from relax.algorithm.base import Algorithm
from relax.network.diffv2 import Diffv2Net, Diffv2Params
from relax.utils.experience import Experience
from relax.utils.typing_utils import Metric
from relax.utils.jax_utils import random_key_from_data
 
class Diffv2OptStates(NamedTuple):
    q: tuple  # tuple of N optax.OptState, one per Q network
    policy: optax.OptState
    value: optax.OptState = None  # Optional V(s) network for normalized advantage guidance


class MalaSampleResult(NamedTuple):
    """Full result bundle from one MALA-corrected sampling pass.

    Returned by :func:`stateless_get_action_mala_full`. Call sites destructure
    only the fields they need:

    * Public sampler (``stateless_get_action_env``) returns ``(action, q,
      log_eta_scales)``.
    * The TD update path uses ``action``, ``log_eta_scales``, and the four
      MALA diagnostics for wandb logging.
    """
    action: jax.Array
    q: jax.Array
    log_eta_scales: jax.Array
    mala_acc_rate: jax.Array
    mala_eta_scale: jax.Array
    per_level_acc: jax.Array
    per_level_clip: jax.Array


class Diffv2TrainState(NamedTuple):
    params: Diffv2Params
    opt_state: Diffv2OptStates
    step: int
    log_eta_scales: jax.Array
    tfg_eta: jax.Array
    # Normalized advantage guidance state
    value_params: hk.Params = None  # V(s) network params (optional)
    advantage_second_moment_ema: float = 1.0  # EMA of E[A^2] where A = Q - V
    advantage_third_moment_ema: float = 0.0
    dist_shift_covariance_ema: float = 0.0
    dist_shift_shape_ema: float = -1.0        # EMA of s₂ = (2γc + κ₃) / v^(3/2), dimensionless shape
    dist_shift_coeff_ema: float = -1.0        # EMA of B = 2γc + κ₃, raw η²-coefficient (alt to shape)
    dist_shift_shape3_ema: float = 0.0       # EMA of s₃ = τ / v², third-order dimensionless shape
    # Per-seed vmappable hyperparameters. Scalars in single-seed mode; under
    # VmapOffPolicyTrainer each field becomes a [N]-shaped array so vmap maps
    # one scalar value to each seed.
    gamma: jax.Array = 0.99            # discount factor
    polyak_tau: jax.Array = 0.005       # target net soft-update rate
    lr_q: jax.Array = 1e-4              # Q optimizer LR (applied as -lr*update)
    lr_policy: jax.Array = 1e-4         # policy optimizer LR
    guidance_mult: jax.Array = 1.0      # guidance strength multiplier
    adv_ema_tau: jax.Array = 0.0005     # advantage-moment EMA rate
    shape_ema_tau: jax.Array = 0.0001   # dimensionless shape EMA rate
    kl_budget_val: jax.Array = 1.0      # KL budget δ (host uses for η cap)
    reward_scale: jax.Array = 1.0       # reward scaling for TD / huber δ
    x0_hat_clip_radius: jax.Array = 1.0  # clip radius for x0 prediction
    mala_adapt_rate: jax.Array = 0.05   # MALA step-size adaptation rate
    q_td_huber_width: jax.Array = float("inf")  # Q TD huber loss width (in reward units)
    shape3_ema_tau: jax.Array = 5e-5    # third-order dimensionless shape EMA rate


class DPMD(Algorithm):

    def __init__(
        self,
        agent: Diffv2Net,
        params: Diffv2Params,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        lr_policy: float | None = None,
        lr_q: float | None = None,
        tau: float = 0.005,
        delay_update: int = 2,
        reward_scale: float = 0.2,
        q_critic_agg: str = "min",
        q_bootstrap_agg: str = "min",
        tfg_eta: float = 0.0,
        x0_hat_clip_radius: float = 1.0,
        mala_adapt_rate: float = 0.05,
        mala_guided_predictor: bool = False,
        mala_no_predictor: bool = False,
        ddim_predictor: bool = False,
        q_td_huber_width: float = float("inf"),
        batch_independent_guidance: bool = False,
        guidance_strength_multiplier: float = 1.0,
        # Energy/score scaling for exploration
        energy_multiplier: float = 1.0,
        # Critic normalization for guidance
        critic_normalization: str = "none",
        advantage_ema_tau: float = 0.0005,
        shape_ema_tau: float = 0.0001,  # slower EMA for dimensionless shape s₂ = (2γc+κ₃)/v^(3/2)
        initial_advantage_second_moment_ema: float = 1.0,
        initial_dist_shift_shape_ema: float = -1.0,
        # KL budget and distribution-shift adaptive eta
        kl_budget: float | None = None,
        one_step_dist_shift_eta: bool = False,
    ):
        self.agent = agent

        # --- Algorithm scalars (each assigned exactly once) ---
        self.gamma = gamma
        self.tau = tau
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.lr_policy = float(lr if lr_policy is None else lr_policy)
        self.lr_q = float(lr if lr_q is None else lr_q)
        self.q_td_huber_width = float(q_td_huber_width)
        self.batch_independent_guidance = bool(batch_independent_guidance)
        self.guidance_strength_multiplier = float(guidance_strength_multiplier)
        self.energy_multiplier = float(energy_multiplier)
        self.x0_hat_clip_radius = float(x0_hat_clip_radius)
        self.mala_adapt_rate = float(mala_adapt_rate)
        self.advantage_ema_tau = float(advantage_ema_tau)
        self.shape_ema_tau = float(shape_ema_tau)
        self.shape3_ema_tau = 5e-5
        self.initial_advantage_second_moment_ema = float(initial_advantage_second_moment_ema)
        self.initial_dist_shift_shape_ema = float(initial_dist_shift_shape_ema)
        self.initial_dist_shift_coeff_ema = self.initial_dist_shift_shape_ema * (max(self.initial_advantage_second_moment_ema, 0.0) ** 1.5)
        self.kl_budget = float(kl_budget) if kl_budget is not None else None
        self.q_critic_agg = str(q_critic_agg)
        self.q_bootstrap_agg = str(q_bootstrap_agg)
        self.critic_normalization = str(critic_normalization)
        self.mala_guided_predictor = bool(mala_guided_predictor)
        self.mala_no_predictor = bool(mala_no_predictor)
        self.ddim_predictor = bool(ddim_predictor)
        self.tfg_eta = float(tfg_eta)
        self.one_step_dist_shift_eta = bool(one_step_dist_shift_eta)
        self.on_policy_ema = (self.kl_budget is not None)
        self.policy_loss_key = "losses/Policy_epsilon_MSE"

        self._validate_invariants()

        # --- Optimizers: unscaled Adam; per-seed state.lr_{q,policy} is applied at update time. ---
        self.optim, self.policy_optim = self._setup_optimizers()

        # --- Optional V(s) network for normalized-advantage guidance (critic_normalization='ema'). ---
        value_params_init, value_opt_state_init = self._setup_value_network(params)

        # --- Cached scalars consumed by _build_initial_state and inner closures ---
        # `timesteps` is also captured by closures defined further down in __init__
        # (e.g. stateless_get_action_mala_full's mala_chain), so it must remain a
        # local in __init__'s scope -- not just an attribute on self.
        timesteps = self.agent.num_timesteps
        self._init_log_eta_scale = jnp.float32(0.0)  # eta_scale = exp(0) = 1.0
        self._init_tfg_eta = self.tfg_eta
        self._timesteps = int(timesteps)

        # --- Initial vmap-stackable train state ---
        self.state = self._build_initial_state(params, value_params_init, value_opt_state_init)

        # --- Schedule cache for wandb SNR x-axis ---
        B_sched = self.agent.diffusion.beta_schedule()
        self._alphas_cumprod = np.asarray(B_sched.alphas_cumprod)  # [T]
        self._snr = self._alphas_cumprod / np.maximum(1.0 - self._alphas_cumprod, 1e-8)

        # --- N-ary Q aggregation helpers (used by sampling and update) ---
        def hard_min_q_n(q_means):
            """Element-wise min across N Q means."""
            q = q_means[0]
            for m in q_means[1:]:
                q = jnp.minimum(q, m)
            return q

        def aggregate_q_fn_outer(q_means):
            n = len(q_means)
            if self.q_critic_agg == "min":
                return hard_min_q_n(q_means)
            return sum(q_means) * jnp.float32(1.0 / n)

        @jax.jit
        def stateless_update(
            key: jax.Array, state: Diffv2TrainState, data: Experience
        ) -> Tuple[Diffv2OptStates, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q_params = state.params.q          # tuple of N Q params
            target_q_params = state.params.target_q  # tuple of N target Q params
            policy_params = state.params.policy
            q_opt_states = state.opt_state.q   # tuple of N opt states
            policy_opt_state = state.opt_state.policy
            step = state.step
            log_eta_scales = state.log_eta_scales
            num_q = len(q_params)
            # Split enough keys: base keys + per-Q keys for langevin noise.
            # key_splits[2,5,6,7] are reserved (held to preserve PRNG layout
            # against earlier alpha-tuning / randomize-Q / shuffle paths).
            key_splits = jax.random.split(key, 8 + num_q)
            next_eval_key = key_splits[0]
            new_eval_key = key_splits[1]
            diffusion_time_key = key_splits[3]
            diffusion_noise_key = key_splits[4]
            q_langevin_keys = key_splits[8:]  # num_q keys

            # --- N-ary Q helpers ---
            def reduce_min_q(means_list):
                """Element-wise min across N Q means."""
                q = means_list[0]
                for m in means_list[1:]:
                    q = jnp.minimum(q, m)
                return q

            def aggregate_q_n(q_means):
                """min or mean aggregation across N Q means (matches --q_critic_agg)."""
                if q_critic_agg == "min":
                    return reduce_min_q(q_means)
                return sum(q_means) * jnp.float32(1.0 / num_q)

            reward *= state.reward_scale

            # Sample a single next-action and evaluate target-Q on it (td_actions=1).
            mala_result = stateless_get_action_mala_full(
                next_eval_key, state, next_obs, hard_min_q_n,
            )
            next_action = mala_result.action
            log_eta_scales = mala_result.log_eta_scales
            q_target_per_q = [self.agent.q(tqp, next_obs, next_action) for tqp in target_q_params]

            not_done = (1 - done)
            # Clipped double Q-learning: all Qs bootstrap from min(target_Q_1..N).
            q_target_min_for_backup = reduce_min_q(q_target_per_q)
            shared_backup = reward + not_done * state.gamma * q_target_min_for_backup
            q_backup_per_q = [shared_backup] * num_q
            delta = state.q_td_huber_width * state.reward_scale
            use_huber = jnp.isfinite(delta)
            delta_safe = jnp.where(use_huber, delta, jnp.float32(1.0))

            def huber_loss(e):
                abs_e = jnp.abs(e)
                quad = jnp.minimum(abs_e, delta_safe)
                lin = abs_e - quad
                return jnp.float32(0.5) * quad * quad + delta_safe * lin

            def compute_td_loss(td_err):
                per_elem = jnp.where(use_huber, huber_loss(td_err), td_err * td_err)
                return jnp.mean(per_elem)

            # Per-Q TD loss + Adam step. vmapped across the N Q networks so XLA
            # emits a single batched matmul per layer (and a single batched Adam
            # update) instead of N independent unrolled ops.
            def single_q_train_step(qp, opt_s, backup_qi):
                def q_loss_fn(p):
                    q_pred_mean = self.agent.q(p, obs, action)
                    td_err = q_pred_mean - backup_qi
                    return compute_td_loss(td_err), q_pred_mean

                (qi_loss, qi_pred), qi_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(qp)
                update, new_opt = self.optim.update(qi_grads, opt_s, params=qp)
                update = jax.tree.map(lambda u: -state.lr_q * u, update)
                new_qp = optax.apply_updates(qp, update)
                return new_qp, new_opt, qi_loss, qi_pred

            stacked_q_params = jax.tree.map(lambda *ps: jnp.stack(ps), *q_params)
            stacked_q_opt = jax.tree.map(lambda *ss: jnp.stack(ss), *q_opt_states)
            stacked_backup = jnp.stack(q_backup_per_q)

            stacked_new_qp, stacked_new_opt, all_q_losses, _all_q_preds = jax.vmap(
                single_q_train_step
            )(stacked_q_params, stacked_q_opt, stacked_backup)

            q_params = tuple(jax.tree.map(lambda x: x[i], stacked_new_qp) for i in range(num_q))
            q_opt_states = tuple(jax.tree.map(lambda x: x[i], stacked_new_opt) for i in range(num_q))

            def policy_loss_fn(policy_params) -> jax.Array:
                # Constant-weight diffusion score-matching loss against
                # next_action (target action sampled above). The Q-based
                # reweighting path has been removed; this is bit-equivalent
                # to upstream `inference_mirror_descent`'s eps_mse path:
                #   weighted_p_loss(diffusion_noise_key, jnp.ones_like(weights),
                #                   denoiser, t, target_action, reduction="mean")
                # (see inference_mirror_descent/relax/algorithm/dpmd.py:1793).
                # The simplified `p_loss` uses optax.squared_error (matching
                # weighted_p_loss), NOT optax.l2_loss which would be a 0.5x
                # multiplier and silently produce a non-bit-identical run.
                def denoiser(t, x):
                    return self.agent.policy(policy_params, next_obs, x, t)

                t = jax.random.randint(
                    diffusion_time_key,
                    (obs.shape[0],),
                    0,
                    self.agent.num_timesteps,
                )
                return self.agent.diffusion.p_loss(
                    diffusion_noise_key,
                    denoiser,
                    t,
                    jax.lax.stop_gradient(next_action),
                )

            total_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_params)

            # update networks
            def param_update(optim, params, grads, opt_state, lr_value=None):
                update, new_opt_state = optim.update(grads, opt_state, params=params)
                if lr_value is not None:
                    update = jax.tree.map(lambda u: -lr_value * u, update)
                new_params = optax.apply_updates(params, update)
                return new_params, new_opt_state

            def delay_param_update(optim, params, grads, opt_state, lr_value=None):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda po: param_update(optim, po[0], grads, po[1], lr_value),
                    lambda po: po,
                    (params, opt_state)
                )

            def delay_target_update(params, target_params, tau):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda target_params: optax.incremental_update(params, target_params, tau),
                    lambda target_params: target_params,
                    target_params
                )

            policy_lr = state.lr_policy
            policy_params, policy_opt_state = delay_param_update(self.policy_optim, policy_params, policy_grads, policy_opt_state, lr_value=policy_lr)

            new_target_q_list = []
            for qi in range(num_q):
                new_target_q_list.append(delay_target_update(q_params[qi], target_q_params[qi], state.polyak_tau))
            target_q_params = tuple(new_target_q_list)

            # Normalized advantage guidance: train V(s) and update A² EMA
            value_params_updated = state.value_params
            value_opt_state_updated = state.opt_state.value
            new_adv_second_moment_ema = state.advantage_second_moment_ema
            value_loss_log = jnp.float32(0.0)
            adv_second_moment_log = jnp.float32(0.0)
            
            if self.critic_normalization == "ema" and state.value_params is not None:
                # KL-budget / on-policy-EMA mode: train V(s') against
                # on-policy Q(s', a') targets. q_target_per_q is already
                # computed at (next_obs, next_action) with a' ~ π_current.
                # The legacy off-policy V path (critic_normalization='ema'
                # without kl_budget) was removed; an assertion in __init__
                # enforces that combo.
                q_for_v = aggregate_q_n(q_target_per_q)

                def value_loss_fn(v_params):
                    v_pred = self._value_net.apply(v_params, next_obs)
                    return jnp.mean((v_pred - jax.lax.stop_gradient(q_for_v)) ** 2)

                v_loss, v_grads = jax.value_and_grad(value_loss_fn)(state.value_params)
                v_updates, value_opt_state_updated = self.optim.update(v_grads, state.opt_state.value, state.value_params)
                v_updates = jax.tree.map(lambda u: -state.lr_q * u, v_updates)
                value_params_updated = optax.apply_updates(state.value_params, v_updates)
                value_loss_log = v_loss

                # On-policy mode: EMA is updated outside jit in sample();
                # pass through here.
                adv_second_moment_log = state.advantage_second_moment_ema

            state = state._replace(
                params=Diffv2Params(q_params, target_q_params, policy_params),
                opt_state=Diffv2OptStates(q=q_opt_states, policy=policy_opt_state, value=value_opt_state_updated),
                step=step + 1,
                log_eta_scales=log_eta_scales,
                value_params=value_params_updated,
                advantage_second_moment_ema=new_adv_second_moment_ema,
            )

            # --- Losses ---
            # Q_MSE: average loss across ensemble members
            q_mse = jnp.mean(all_q_losses)

            info = {
                self.policy_loss_key: total_loss,
                "losses/Q_MSE": q_mse,
            }

            # V_MSE: only when V network exists
            if self.critic_normalization != "none" and state.value_params is not None:
                info["losses/V_MSE"] = value_loss_log

            # --- MALA per-level arrays (logged as wandb.Table line plots) ---
            # Only include when MALA sampling actually runs (otherwise arrays are NaN)
            if self.agent.energy_mode and self.agent.mala_steps > 0:
                info["MALA/acceptance_rate"] = mala_result.per_level_acc
                info["MALA/clip_frac"] = mala_result.per_level_clip
                info["MALA/eta_scale"] = jnp.exp(log_eta_scales)

            # --- Q section ---
            if self.critic_normalization == "ema":
                info["Critic/inv_sqrt(E(Var(Q))_ema)"] = jnp.float32(1.0) / jnp.sqrt(jnp.maximum(new_adv_second_moment_ema, jnp.float32(1e-6)))
            return state, info

        def stateless_get_action_mala_full(
            key: jax.Array,
            state: Diffv2TrainState,
            obs: jax.Array,
            aggregate_q_fn,
        ) -> MalaSampleResult:
            policy_params = state.params.policy
            q_params_tuple = state.params.q
            log_eta_scales_in = state.log_eta_scales
            tfg_eta_current = state.tfg_eta
            value_params = state.value_params
            adv_second_moment_ema = state.advantage_second_moment_ema
            x0_hat_clip_radius_hp = state.x0_hat_clip_radius
            mala_adapt_rate_hp = state.mala_adapt_rate
            guidance_mult_hp = state.guidance_mult

            single = obs.ndim == 1
            if single:
                obs_batch = obs[None, :]
            else:
                obs_batch = obs

            shape = (*obs_batch.shape[:-1], self.agent.act_dim)

            def energy_model(t, x):
                E = self.agent.energy_fn(policy_params, obs_batch, x, t)
                # Scale base energy by energy_multiplier (tempers the base distribution)
                return self.energy_multiplier * E

            def mala_chain(single_key: jax.Array, log_eta_scales_init: jax.Array):
                key_x, loop_key = jax.random.split(single_key)
                aggregate_q_local = aggregate_q_fn
                B = self.agent.diffusion.beta_schedule()

                def q_min_from_x0(x0_in):
                    q_means = [self.agent.q(qp, obs_batch, x0_in) for qp in q_params_tuple]
                    return aggregate_q_local(q_means)

                def sample_x0_for_mala_energy(sample_key, x0_hat, t_idx):
                    return x0_hat, jnp.clip(x0_hat, -x0_hat_clip_radius_hp, x0_hat_clip_radius_hp)

                def q_mean_from_x(x_in, t_idx):
                    noise_pred = self.agent.policy(policy_params, obs_batch, x_in, t_idx)
                    # Scale base score by energy_multiplier (tempers the base distribution)
                    # Guidance component is NOT scaled - only the base policy score
                    noise_pred_scaled = self.energy_multiplier * noise_pred
                    x0_hat = (
                        x_in * B.sqrt_recip_alphas_cumprod[t_idx]
                        - noise_pred_scaled * B.sqrt_recipm1_alphas_cumprod[t_idx]
                    )
                    x0_q = jnp.clip(x0_hat, -x0_hat_clip_radius_hp, x0_hat_clip_radius_hp)
                    q = q_min_from_x0(x0_q)

                    agg_fn = jnp.sum if self.batch_independent_guidance else jnp.mean
                    mult = guidance_mult_hp

                    # Critic normalization: use (Q - V) / std instead of Q
                    if self.critic_normalization == "ema" and value_params is not None:
                        v = self._value_net.apply(value_params, obs_batch)
                        advantage = q - v
                        adv_std = jnp.sqrt(jnp.maximum(adv_second_moment_ema, jnp.float32(1e-6)))
                        return mult * agg_fn(advantage / adv_std)

                    return mult * agg_fn(q)

                _tfg_eta_schedule_ones = jnp.ones((timesteps,), dtype=jnp.float32)

                def lambda_for_step(t_idx: jax.Array, tfg_eta_current: jax.Array) -> jax.Array:
                    t_next = jnp.maximum(t_idx - 1, 0)
                    return tfg_eta_current * _tfg_eta_schedule_ones[t_next]

                def grad_guidance(x_in, t_idx):
                    def guided(_):
                        return jax.grad(lambda xx: q_mean_from_x(xx, t_idx))(x_in)

                    def unguided(_):
                        return jnp.zeros_like(x_in)

                    lam = lambda_for_step(t_idx, tfg_eta_current)
                    return jax.lax.cond(lam > 0.0, guided, unguided, operand=None)

                def energy_total(t, x, sample_key):
                    def only_model(x_in):
                        return energy_model(t, x_in), jnp.float32(0.0)

                    def with_q(x_in):
                        E_mod = energy_model(t, x_in)
                        noise_pred = self.agent.policy(policy_params, obs_batch, x_in, t)
                        x0_hat = (
                            x_in * B.sqrt_recip_alphas_cumprod[t]
                            - noise_pred * B.sqrt_recipm1_alphas_cumprod[t]
                        )
                        x0_preclip, x0_q = sample_x0_for_mala_energy(sample_key, x0_hat, t)
                        clip_frac = jnp.mean((jnp.abs(x0_preclip) > x0_hat_clip_radius_hp).astype(jnp.float32))
                        q_min = q_min_from_x0(x0_q)
                        lambda_t = lambda_for_step(t, tfg_eta_current)
                        return E_mod - lambda_t * q_min, clip_frac

                    lambda_t = lambda_for_step(t, tfg_eta_current)
                    return jax.lax.cond(lambda_t > 0.0, with_q, only_model, x)

                def predictor_step(t_idx: jax.Array, x_in: jax.Array, k_in: jax.Array):
                    # DDIM-style deterministic predictor (no sampled noise).
                    # The PRNG split is retained -- and the resulting `z_key`
                    # intentionally unused -- to keep `k_out` bit-identical
                    # with the legacy stochastic DDPM predictor path.
                    noise_pred = self.agent.policy(policy_params, obs_batch, x_in, t_idx)
                    noise_pred_scaled = self.energy_multiplier * noise_pred  # tempers base score; guidance is NOT scaled
                    k_out, _z_key = jax.random.split(k_in)
                    if self.mala_guided_predictor:
                        grad_q = grad_guidance(x_in, t_idx)
                        sigma_t = B.sqrt_one_minus_alphas_cumprod[t_idx]
                        lambda_t = lambda_for_step(t_idx, tfg_eta_current)
                        eps_pred = noise_pred_scaled - lambda_t * sigma_t * grad_q
                    else:
                        eps_pred = noise_pred_scaled
                    model_mean, _ = self.agent.diffusion.p_mean_variance(t_idx, x_in, eps_pred)
                    return model_mean, k_out

                x0 = jax.random.normal(key_x, shape)

                eta_base_min = jnp.maximum(jnp.min(B.betas), jnp.float32(1e-8))
                eta_base_max = jnp.maximum(jnp.max(B.betas), jnp.float32(1e-8))
                log_eta_min = jnp.log(jnp.float32(1e-8) / eta_base_max)
                # Fixed step-size cap of 0.5 (per-level recurrence cap removed).
                log_eta_max = jnp.log(jnp.float32(0.5) / eta_base_min)

                def level_body(i, carry):
                    x_curr, k, log_eta_scales, acc_sum, acc_count, per_level_acc, per_level_clip_frac = carry
                    t = timesteps - 1 - i

                    def do_mala_level(carry_in, t_corr: jax.Array):
                        x_in, k_in, log_eta_scales_in, acc_sum_in, acc_count_in = carry_in
                        eta_base_t = jnp.maximum(B.betas[t_corr], jnp.float32(1e-8))
                        log_eta_scale0 = log_eta_scales_in[t_corr]
                        eta_upper = jnp.float32(0.5)

                        def mala_body(_, state):
                            x_step, k_step, log_eta_scale, acc_sum_step, acc_count_step, clip_sum_step = state

                            E_x, vjp_x, clip_x = jax.vjp(lambda xx: energy_total(t_corr, xx, k_step), x_step, has_aux=True)
                            grad_E_x = vjp_x(jnp.ones_like(E_x))[0]

                            k_step, noise_key, u_key = jax.random.split(k_step, 3)
                            eta_k = jnp.clip(
                                jnp.exp(log_eta_scale) * eta_base_t,
                                jnp.float32(1e-8),
                                eta_upper,
                            )
                            z = jax.random.normal(noise_key, x_step.shape)
                            sd = jnp.sqrt(jnp.float32(2.0) * eta_k)
                            x_prop = x_step - eta_k * grad_E_x + sd * z

                            E_x_prop, vjp_x_prop, _clip_prop = jax.vjp(lambda xx: energy_total(t_corr, xx, k_step), x_prop, has_aux=True)
                            grad_E_x_prop = vjp_x_prop(jnp.ones_like(E_x_prop))[0]

                            mean_f = x_step - eta_k * grad_E_x
                            mean_r = x_prop - eta_k * grad_E_x_prop

                            def log_gauss(xv, meanv):
                                diff = xv - meanv
                                return -jnp.sum(diff * diff, axis=-1) / (jnp.float32(4.0) * eta_k)

                            log_q_prop_given_x = log_gauss(x_prop, mean_f)
                            log_q_x_given_prop = log_gauss(x_step, mean_r)

                            log_alpha = (-E_x_prop + E_x) + (log_q_x_given_prop - log_q_prop_given_x)
                            u = jax.random.uniform(u_key, E_x.shape)
                            accept = jnp.log(u) < jnp.minimum(jnp.float32(0.0), log_alpha)

                            x_new = jnp.where(accept[..., None], x_prop, x_step)

                            acc_rate = jnp.mean(accept.astype(jnp.float32).reshape(-1))
                            target = jnp.float32(0.574)
                            adapt_rate = mala_adapt_rate_hp
                            log_eta_scale = log_eta_scale + adapt_rate * (acc_rate - target)
                            log_eta_scale = jnp.clip(log_eta_scale, log_eta_min, log_eta_max)

                            return (
                                x_new,
                                k_step,
                                log_eta_scale,
                                acc_sum_step + acc_rate,
                                acc_count_step + jnp.float32(1.0),
                                clip_sum_step + clip_x,
                            )

                        x_out, k_out, log_eta_scale_final, acc_sum_level, acc_count_level, clip_sum_level = jax.lax.fori_loop(
                            0,
                            self.agent.mala_steps,
                            mala_body,
                            (x_in, k_in, log_eta_scale0, jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0)),
                        )

                        log_eta_scales_in = log_eta_scales_in.at[t_corr].set(log_eta_scale_final)
                        return (
                            x_out,
                            k_out,
                            log_eta_scales_in,
                            acc_sum_in + acc_sum_level,
                            acc_count_in + acc_count_level,
                            clip_sum_level,
                        )

                    if self.mala_no_predictor:
                        acc_sum_before = acc_sum
                        acc_count_before = acc_count
                        x_curr, k, log_eta_scales, acc_sum, acc_count, clip_sum_level = do_mala_level(
                            (x_curr, k, log_eta_scales, acc_sum, acc_count), t
                        )
                        level_acc_sum = acc_sum - acc_sum_before
                        level_acc_count = acc_count - acc_count_before
                        level_acc = level_acc_sum / jnp.maximum(level_acc_count, jnp.float32(1.0))
                        per_level_acc = per_level_acc.at[t].set(level_acc)
                        mala_steps_f = jnp.float32(self.agent.mala_steps)
                        per_level_clip_frac = per_level_clip_frac.at[t].set(clip_sum_level / jnp.maximum(mala_steps_f, jnp.float32(1.0)))
                        return x_curr, k, log_eta_scales, acc_sum, acc_count, per_level_acc, per_level_clip_frac

                    acc_sum_before = acc_sum
                    acc_count_before = acc_count
                    x_curr, k, log_eta_scales, acc_sum, acc_count, clip_sum_level = do_mala_level(
                        (x_curr, k, log_eta_scales, acc_sum, acc_count), t
                    )
                    level_acc_sum = acc_sum - acc_sum_before
                    level_acc_count = acc_count - acc_count_before
                    level_acc = level_acc_sum / jnp.maximum(level_acc_count, jnp.float32(1.0))
                    per_level_acc = per_level_acc.at[t].set(level_acc)
                    mala_steps_f = jnp.float32(self.agent.mala_steps)
                    per_level_clip_frac = per_level_clip_frac.at[t].set(clip_sum_level / jnp.maximum(mala_steps_f, jnp.float32(1.0)))
                    x_next, k = predictor_step(t, x_curr, k)
                    return x_next, k, log_eta_scales, acc_sum, acc_count, per_level_acc, per_level_clip_frac

                init_per_level_acc = jnp.zeros((timesteps,), dtype=jnp.float32)
                init_per_level_clip_frac = jnp.zeros((timesteps,), dtype=jnp.float32)
                x_final, _, log_eta_scales_out, acc_sum_final, acc_count_final, per_level_acc_out, per_level_clip_frac_out = jax.lax.fori_loop(
                    0,
                    timesteps,
                    level_body,
                    (x0, loop_key, log_eta_scales_init, jnp.float32(0.0), jnp.float32(0.0), init_per_level_acc, init_per_level_clip_frac),
                )

                mala_acc_rate = acc_sum_final / jnp.maximum(acc_count_final, jnp.float32(1.0))
                mala_eta_scale = jnp.float32(jnp.nan)

                act_final = jnp.clip(x_final, -1.0, 1.0)
                q_means_f = [self.agent.q(qp, obs_batch, act_final) for qp in q_params_tuple]
                q = aggregate_q_fn(q_means_f)
                return act_final, q, log_eta_scales_out, mala_acc_rate, mala_eta_scale, per_level_acc_out, per_level_clip_frac_out

            # 3-way split kept verbatim to preserve PRNG layout from a deleted
            # multi-particle / particle-select sampling path; the two unused
            # keys must continue to be split off here for byte-exact PRNG match.
            key_sample, _key_select, _noise_key = jax.random.split(key, 3)
            act, q, log_eta_scales_out, mala_acc_rate, mala_eta_scale, pl_acc, pl_clip = mala_chain(
                key_sample, log_eta_scales_in
            )
            if single:
                return MalaSampleResult(
                    action=act[0], q=q[0], log_eta_scales=log_eta_scales_out,
                    mala_acc_rate=mala_acc_rate, mala_eta_scale=mala_eta_scale,
                    per_level_acc=pl_acc, per_level_clip=pl_clip,
                )
            return MalaSampleResult(
                action=act, q=q, log_eta_scales=log_eta_scales_out,
                mala_acc_rate=mala_acc_rate, mala_eta_scale=mala_eta_scale,
                per_level_acc=pl_acc, per_level_clip=pl_clip,
            )

        def stateless_get_action_env(
            key: jax.Array,
            state: Diffv2TrainState,
            obs: jax.Array,
        ):
            r = stateless_get_action_mala_full(key, state, obs, aggregate_q_fn_outer)
            return r.action, r.q, r.log_eta_scales

        def stateless_get_deterministic_action_env(
            state: Diffv2TrainState,
            obs: jax.Array,
        ):
            key = random_key_from_data(obs)
            return stateless_get_action_env(key, state, obs)

        self._implement_common_behavior(stateless_update, stateless_get_action_env, stateless_get_deterministic_action_env)

    def get_policy_params(self):
        # All stateless_* sampler/update functions take the full Diffv2TrainState;
        # JAX prunes unused leaves at trace time.
        return self.state

    def get_policy_params_to_save(self):
        return self.state

    def get_rollout_params(self):
        return self.state

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.state = jax.device_put(state)

    def get_current_tfg_eta(self) -> float:
        return float(self.state.tfg_eta)

    def set_tfg_eta(self, new_tfg_eta: float) -> None:
        self.state = self.state._replace(tfg_eta=jnp.float32(new_tfg_eta))

    def _validate_invariants(self):
        """Enforce simplify_walkthrough preconditions: only branches actually
        exercised by the three protected launch commands are supported."""
        if self.q_critic_agg not in {"min", "mean"}:
            raise ValueError("q_critic_agg must be 'min' or 'mean' in simplify_walkthrough.")
        if self.q_bootstrap_agg != "min":
            raise ValueError(
                "q_bootstrap_agg must be 'min' in simplify_walkthrough "
                "(the 'mean' TD-bootstrap branch has been removed)."
            )
        if self.critic_normalization not in {"none", "ema"}:
            raise ValueError("critic_normalization must be 'none' or 'ema' in simplify_walkthrough.")
        if self.critic_normalization == "ema" and self.kl_budget is None:
            raise ValueError(
                "simplify_walkthrough requires --kl_budget whenever "
                "--critic_normalization=ema; the legacy off-policy V branch "
                "(V trained on Q(s, a_replay)) has been removed."
            )
        if not (self.agent.energy_mode and self.agent.mala_steps > 0):
            raise ValueError(
                "simplify_walkthrough requires energy_mode=True and mala_steps > 0; "
                "the non-MALA / non-energy sampling branches have been removed."
            )

    def _setup_optimizers(self):
        """Unscaled Adam optimizers for Q and policy; per-seed lr_{q,policy}
        is applied at update time from state, not baked into the chain."""
        return optax.scale_by_adam(), optax.scale_by_adam()

    def _setup_value_network(self, params):
        """Construct the V(s) network used by critic_normalization='ema'
        (KL-budget / on-policy-EMA mode). Returns (value_params, value_opt_state)
        or (None, None) when V is not used."""
        if self.critic_normalization != "ema":
            self._value_net = None
            self._obs_dim = None
            return None, None

        from relax.network.blocks import ValueNet
        # Infer obs_dim from the Q net's first Linear weight: it has shape
        # (obs_dim + act_dim, hidden). Haiku FlatMaps use slash-separated
        # keys like 'q_net/linear', 'q_net/linear_1', ...; match exactly
        # the one ending in '/linear' (the input layer).
        first_w = None
        for k, v in params.q[0].items():
            if k.endswith('/linear') and isinstance(v, dict) and 'w' in v:
                w = v['w']
                if hasattr(w, 'shape'):
                    first_w = w
                    break
        if first_w is not None:
            obs_dim_inferred = first_w.shape[0] - self.agent.act_dim
            hidden_dim = first_w.shape[1]
        else:
            obs_dim_inferred, hidden_dim = 17, 256  # MuJoCo-ish fallback

        value_net = hk.without_apply_rng(
            hk.transform(lambda obs: ValueNet(
                hidden_sizes=(hidden_dim, hidden_dim, hidden_dim),
                activation=jax.nn.relu,
            )(obs))
        )
        sample_obs = jnp.zeros((1, obs_dim_inferred))
        value_params_init = value_net.init(jax.random.PRNGKey(42), sample_obs)
        value_opt_state_init = self.optim.init(value_params_init)
        self._value_net = value_net
        self._obs_dim = obs_dim_inferred
        return value_params_init, value_opt_state_init

    def _build_initial_state(self, params, value_params_init, value_opt_state_init):
        """Construct a fresh Diffv2TrainState from a given set of network params.

        Factored out of __init__ so vmap-mode setup can call it N times with
        different init seeds and stack the results along a leading seed axis.
        """
        return Diffv2TrainState(
            params=params,
            opt_state=Diffv2OptStates(
                q=tuple(self.optim.init(qp) for qp in params.q),
                policy=self.policy_optim.init(params.policy),
                value=value_opt_state_init,
            ),
            step=jnp.int32(0),
            log_eta_scales=jnp.full((self._timesteps,), self._init_log_eta_scale, dtype=jnp.float32),
            tfg_eta=jnp.float32(self._init_tfg_eta),
            value_params=value_params_init,
            advantage_second_moment_ema=jnp.float32(self.initial_advantage_second_moment_ema),
            advantage_third_moment_ema=jnp.float32(0.0),
            dist_shift_covariance_ema=jnp.float32(0.0),
            dist_shift_shape_ema=jnp.float32(self.initial_dist_shift_shape_ema),
            dist_shift_coeff_ema=jnp.float32(self.initial_dist_shift_coeff_ema),
            dist_shift_shape3_ema=jnp.float32(0.0),
            gamma=jnp.float32(self.gamma),
            polyak_tau=jnp.float32(self.tau),
            lr_q=jnp.float32(self.lr_q),
            lr_policy=jnp.float32(self.lr_policy),
            guidance_mult=jnp.float32(self.guidance_strength_multiplier),
            adv_ema_tau=jnp.float32(self.advantage_ema_tau),
            shape_ema_tau=jnp.float32(self.shape_ema_tau),
            kl_budget_val=jnp.float32(self.kl_budget if self.kl_budget is not None else 1.0),
            reward_scale=jnp.float32(self.reward_scale),
            x0_hat_clip_radius=jnp.float32(self.x0_hat_clip_radius),
            mala_adapt_rate=jnp.float32(self.mala_adapt_rate),
            q_td_huber_width=jnp.float32(self.q_td_huber_width),
            shape3_ema_tau=jnp.float32(self.shape3_ema_tau),
        )

    def make_vmapped_state(self, params_list, value_init_keys=None):
        """Build a vmapped train-state by stacking N independent initial states.

        params_list: list of N Diffv2Params (one per seed).
        value_init_keys: optional list of N jax.random keys for value-net init.
                        Ignored when no V-network is used.
        Returns a Diffv2TrainState with a leading [N] axis on every leaf.
        """
        N = len(params_list)
        if self._value_net is not None:
            if value_init_keys is None:
                # Derive deterministic per-seed keys from the standard init seed.
                value_init_keys = [jax.random.PRNGKey(42) for _ in range(N)]
            sample_obs = jnp.zeros((1, self._obs_dim))
            vparams_list = [self._value_net.init(k, sample_obs) for k in value_init_keys]
            vopt_list = [self.optim.init(vp) for vp in vparams_list]
        else:
            vparams_list = [None] * N
            vopt_list = [None] * N
        states = [
            self._build_initial_state(p, vp, vo)
            for p, vp, vo in zip(params_list, vparams_list, vopt_list)
        ]
        return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *states)

    def get_action_vmap(self, key: jax.Array, obs: np.ndarray):
        """Vmapped counterpart of get_action. 

        obs: numpy array of shape [N, num_envs, obs_dim].
        Returns (action [N, num_envs, act_dim], q_per_env [N, num_envs],
        v_per_env [N, num_envs]).
        """
        self._ensure_vmap_compiled()
        params = self.get_rollout_params()
        out = self._get_action_vmap_fn(key, params, obs)
        if not (isinstance(out, tuple) and len(out) == 3):
            raise RuntimeError(
                f"Vmap get_action: expected (action, q, log_eta_scales); got {type(out)}"
            )
        action, q, log_eta_scales = out
        # log_eta_scales: shape [N, timesteps] — matches stacked state layout.
        self.state = self.state._replace(log_eta_scales=log_eta_scales)
        action_np = np.asarray(action)
        q_per_env = np.asarray(q)  # [N, num_envs]

        if not self.on_policy_ema:
            return action_np, q_per_env, None

        if getattr(self, "_value_net_apply_vmap", None) is None:
            v_net_apply = self._value_net.apply
            @jax.jit
            def _v_apply_vmap(vp, ob):
                def single(vp_i, ob_i):
                    v = v_net_apply(vp_i, ob_i)
                    if isinstance(v, tuple):
                        v = v[0]
                    return v
                return jax.vmap(single)(vp, ob)
            self._value_net_apply_vmap = _v_apply_vmap
        v_params = self.state.value_params
        obs_j = jnp.asarray(obs)
        v = self._value_net_apply_vmap(v_params, obs_j)
        v_per_env = np.asarray(v)  # [N, num_envs]
        return action_np, q_per_env, v_per_env

    def get_effective_hparams(self) -> dict:
        return {
            "lr_policy_effective": float(self.lr_policy),
            "lr_q_effective": float(self.lr_q),
        }

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def _compute_q_ensemble_var(
        self,
        action: np.ndarray,
        obs: np.ndarray,
        *,
        q_params=None,
        tfg_eta=None,
    ) -> float:
        """Compute mean variance of Q across ensemble members for the given (obs, action)."""
        obs_j = jnp.asarray(obs)
        act_j = jnp.asarray(action)
        if obs_j.ndim == 1:
            obs_j = obs_j[None, :]
            act_j = act_j[None, :]
        q_params_tuple = tuple(self.state.params.q if q_params is None else q_params)
        if len(q_params_tuple) < 2:
            return 0.0
        if getattr(self, "_q_ensemble_var_jit", None) is None:
            q_fn = self.agent.q

            @jax.jit
            def _fn(q_params_tuple, s, a):
                means = [q_fn(qp, s, a) for qp in q_params_tuple]
                stacked = jnp.stack(means, axis=0)
                return jnp.mean(jnp.var(stacked, axis=0))

            self._q_ensemble_var_jit = _fn
        return float(self._q_ensemble_var_jit(q_params_tuple, obs_j, act_j))
