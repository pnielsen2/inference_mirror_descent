from typing import NamedTuple, Optional, Tuple, Union

import numpy as np
import jax
import jax.numpy as jnp
import optax
import haiku as hk

from relax.algorithm.base import Algorithm
from relax.network.model_based import ModelBasedNet, ModelBasedParams
from relax.utils.experience import Experience, SequenceExperience
from relax.utils.typing_utils import Metric


class MbPcOptStates(NamedTuple):
    policy: optax.OptState
    dynamics: optax.OptState
    reward: optax.OptState
    value: optax.OptState
    # Optional: seq_policy optimizer state (when joint_seq=True)
    seq_policy: Optional[optax.OptState] = None


class MbPcHyperparams(NamedTuple):
    log_lr_scale_policy: jax.Array
    log_lr_scale_dynamics: jax.Array
    log_lr_scale_reward: jax.Array
    log_lr_scale_value: jax.Array


class MbPcTrainState(NamedTuple):
    params: ModelBasedParams
    opt_state: MbPcOptStates
    step: jax.Array
    hyper: MbPcHyperparams
    hyper_opt_state: optax.OptState


class DPMDMBPC(Algorithm):

    def __init__(
        self,
        agent: ModelBasedNet,
        params: ModelBasedParams,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        tau: float = 0.005,
        reward_scale: float = 0.2,
        use_value: bool = False,
        sprime_num_particles: int = 16,
        sprime_refresh_steps: int = 3,
        sprime_refresh_type: str = "recurrence",
        sprime_cs: float = 0.08,
        bprop_refresh_steps: int = 0,
        use_crn: bool = True,
        tfg_lambda: float = 0.0,
        action_steps_per_level: int = 0,
        action_recur_steps: int = 0,
        H_plan: int = 1,
        joint_seq: bool = False,
        open_loop: bool = False,
        supervised_steps: int = 1,
        lr_policy=None,
        lr_dyn=None,
        lr_reward=None,
        lr_value=None,
        value_mc_samples: int = 8,
        value_td_mode: str = "replay",
        value_td_num_actions: int = 1,
        deterministic_dyn: bool = False,
        entropic_sprime_agg: bool = False,
        ucb_sprime_coeff: float = 0.0,
        use_hypergrad: bool = False,
        hypergrad_lr: float = 1e-3,
        hypergrad_period: int = 100,
        hypergrad_accum_steps: int = 1,
    ):
        self.agent = agent
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.reward_scale = float(reward_scale)
        self.use_value = bool(use_value)

        # Planner hyperparameters (not yet used in sampling; wired for future s'-aware planner)
        self.sprime_num_particles = 1 if deterministic_dyn else int(sprime_num_particles)
        self.sprime_refresh_steps = int(sprime_refresh_steps)
        self.sprime_refresh_type = str(sprime_refresh_type)
        self.sprime_cs = float(sprime_cs)
        self.bprop_refresh_steps = int(bprop_refresh_steps)
        self.use_crn = bool(use_crn)
        self.tfg_lambda = float(tfg_lambda)
        self.action_steps_per_level = int(action_steps_per_level)
        self.action_recur_steps = int(action_recur_steps)
        self.H_plan = int(H_plan)
        self.H_train = int(H_plan)  # Unified: H_train = H_plan
        self.joint_seq = bool(joint_seq) and H_plan > 1  # Only use joint_seq when H > 1
        self.open_loop = bool(open_loop) and H_plan > 1  # Only use open_loop when H > 1
        
        # Open-loop execution state: cached action sequence and current index
        self._cached_actions = None  # Will be [H, act_dim] when planning
        self._action_index = 0  # Current position in cached sequence
        self.supervised_steps = int(supervised_steps)
        self.value_mc_samples = int(value_mc_samples)
        self.value_td_mode = str(value_td_mode)
        self.value_td_num_actions = int(value_td_num_actions)
        self.inference_value_td = self.value_td_mode == "inference"
        self.unguided_value_td = self.value_td_mode == "unguided"
        self.deterministic_dyn = bool(deterministic_dyn)
        self.entropic_sprime_agg = bool(entropic_sprime_agg)
        # Optional UCB-style aggregation over planner-sampled next states:
        # mean + k * std instead of plain mean when summarizing R+gamma*V
        # across the s' population, where k = ucb_sprime_coeff. When k != 0,
        # this takes precedence over the entropic_sprime_agg setting.
        self.ucb_sprime_coeff = float(ucb_sprime_coeff)
        self.use_hypergrad = bool(use_hypergrad)
        self.hypergrad_lr = float(hypergrad_lr)
        self.hypergrad_period = int(hypergrad_period)
        # Number of stochastic validation evaluations to average per
        # hypergradient update. When > 1, we reuse the same train batch and
        # parameters but evaluate the validation objective under multiple
        # independent RNG keys to reduce hypergradient variance.
        self.hypergrad_accum_steps = int(hypergrad_accum_steps)

        if self.action_steps_per_level > 0 and self.action_recur_steps > 0:
            raise ValueError("For dpmd_mb_pc, action_steps_per_level (MALA) and action_recur_steps (TFG recurrence) cannot both be > 0.")

        # Stochastic H-step planning is now supported; constraint removed.

        policy_lr = lr if lr_policy is None else float(lr_policy)
        dyn_lr = lr if lr_dyn is None else float(lr_dyn)
        reward_lr = lr if lr_reward is None else float(lr_reward)
        value_lr = lr if lr_value is None else float(lr_value)

        # Store base learning rates so that we can later report effective
        # learning rates after hypergradient scaling.
        self.policy_lr = policy_lr
        self.dyn_lr = dyn_lr
        self.reward_lr = reward_lr
        self.value_lr = value_lr

        self.policy_optim = optax.adam(self.policy_lr)
        self.dyn_optim = optax.adam(self.dyn_lr)
        self.reward_optim = optax.adam(self.reward_lr)
        self.value_optim = optax.adam(self.value_lr)
        self.hyper_optim = optax.adam(self.hypergrad_lr)
        # seq_policy shares learning rate with policy
        self.seq_policy_optim = optax.adam(self.policy_lr) if self.joint_seq else None

        # Initialize seq_policy optimizer state if using joint sequence mode
        seq_policy_opt_state = None
        if self.joint_seq and params.seq_policy is not None:
            seq_policy_opt_state = self.seq_policy_optim.init(params.seq_policy)

        opt_state = MbPcOptStates(
            policy=self.policy_optim.init(params.policy),
            dynamics=self.dyn_optim.init(params.dynamics),
            reward=self.reward_optim.init(params.reward),
            value=self.value_optim.init(params.value),
            seq_policy=seq_policy_opt_state,
        )

        hyper = MbPcHyperparams(
            log_lr_scale_policy=jnp.float32(0.0),
            log_lr_scale_dynamics=jnp.float32(0.0),
            log_lr_scale_reward=jnp.float32(0.0),
            log_lr_scale_value=jnp.float32(0.0),
        )

        hyper_opt_state = self.hyper_optim.init(hyper)

        self.state = MbPcTrainState(
            params=params,
            opt_state=opt_state,
            step=jnp.int32(0),
            hyper=hyper,
            hyper_opt_state=hyper_opt_state,
        )

        def _effective_lr_scales(h: MbPcHyperparams):
            scale_policy = jnp.exp(h.log_lr_scale_policy)
            scale_dyn = jnp.exp(h.log_lr_scale_dynamics)
            scale_reward = jnp.exp(h.log_lr_scale_reward)
            scale_value = jnp.exp(h.log_lr_scale_value)
            return scale_policy, scale_dyn, scale_reward, scale_value

        def _predict_next_obs_deterministic(
            dyn_p: hk.Params,
            s: jax.Array,
            a: jax.Array,
        ) -> jax.Array:
            batch = s.shape[0]
            s_t = jnp.zeros_like(s)
            t_zeros = jnp.zeros((batch,), dtype=jnp.int32)
            return self.agent.dynamics(dyn_p, s, a, s_t, t_zeros)

        def _ddim_sample_next_obs(
            key_dyn: jax.Array,
            dyn_p: hk.Params,
            s: jax.Array,
            a: jax.Array,
        ) -> jax.Array:
            """DDIM-style sample from the diffusion dynamics model.

            Given state s and action a, sample next state s' by running the
            full diffusion denoising process through all timesteps.
            s and a should be [1, dim] shaped for single-sample case.
            """
            B_dyn = self.agent.dyn_diffusion.beta_schedule()
            T_dyn = self.agent.dyn_diffusion.num_timesteps
            obs_dim = self.agent.obs_dim

            # Initialize at noise level T
            x = 0.5 * jax.random.normal(key_dyn, (1, obs_dim), dtype=jnp.float32)

            def body_fn(x_t: jax.Array, t_idx: jax.Array):
                t_batch = jnp.full((1,), t_idx, dtype=jnp.int32)
                noise_pred = self.agent.dynamics(dyn_p, s, a, x_t, t_batch)
                model_mean, model_log_variance = self.agent.dyn_diffusion.p_mean_variance(
                    t_idx, x_t, noise_pred
                )
                # DDIM-style deterministic update
                x_next = model_mean
                return x_next, None

            t_seq = jnp.arange(T_dyn)[::-1]
            x_final, _ = jax.lax.scan(body_fn, x, t_seq)

            s_next = jnp.where(jnp.isfinite(x_final), x_final, jnp.zeros_like(x_final))
            return s_next

        def core_supervised_update(
            key: jax.Array,
            params: ModelBasedParams,
            opt_state: MbPcOptStates,
            hyper: MbPcHyperparams,
            obs: jax.Array,
            action: jax.Array,
            next_obs: jax.Array,
            reward_scaled: jax.Array,
            actions_seq: jax.Array = None,
        ) -> Tuple[ModelBasedParams, MbPcOptStates, jax.Array, jax.Array, jax.Array]:
            """Core supervised update for dynamics, reward, and policy.

            Args:
                actions_seq: Optional [B, H, act_dim] array of H consecutive actions
                    for H-step policy training. When provided (H_train > 1), the policy
                    is trained to denoise all H actions conditioned on `obs` with
                    horizon index embeddings.
            """
            (
                policy_params,
                target_policy_params,
                dyn_params,
                reward_params,
                value_params,
                target_value_params,
                log_alpha,
                seq_policy_params,
            ) = params

            (
                policy_opt_state,
                dyn_opt_state,
                reward_opt_state,
                value_opt_state,
                seq_policy_opt_state,
            ) = opt_state

            model_key, policy_key = jax.random.split(key, 2)

            lr_scale_policy, lr_scale_dyn, lr_scale_reward, _ = _effective_lr_scales(hyper)

            def dynamics_loss_fn(dyn_p: hk.Params) -> jax.Array:
                if self.deterministic_dyn:
                    s_pred = _predict_next_obs_deterministic(dyn_p, obs, action)
                    return jnp.mean((s_pred - next_obs) ** 2)
                else:
                    def dyn_model(t, x_t):
                        return self.agent.dynamics(dyn_p, obs, action, x_t, t)

                    t_key, noise_key = jax.random.split(model_key)
                    t = jax.random.randint(
                        t_key,
                        (obs.shape[0],),
                        0,
                        self.agent.num_timesteps,
                    )
                    return self.agent.dyn_diffusion.p_loss(
                        noise_key,
                        dyn_model,
                        t,
                        next_obs,
                    )

            dyn_loss, dyn_grads = jax.value_and_grad(dynamics_loss_fn)(dyn_params)

            def reward_loss_fn(rew_p: hk.Params) -> jax.Array:
                r_pred = self.agent.reward(rew_p, obs, action, next_obs)
                return jnp.mean((r_pred - reward_scaled) ** 2)

            rew_loss, rew_grads = jax.value_and_grad(reward_loss_fn)(reward_params)

            # Define loss functions for per-step policy and joint sequence policy
            def policy_loss_fn(policy_p: hk.Params) -> jax.Array:
                """Per-step policy loss (factorized or single-step)."""
                if actions_seq is not None and not self.joint_seq:
                    # Factorized mode: denoise each action separately with horizon index
                    B = obs.shape[0]
                    H = self.H_train

                    def loss_for_horizon_step(h: int, key_h: jax.Array) -> jax.Array:
                        """Compute diffusion loss for horizon step h."""
                        t_key, noise_key = jax.random.split(key_h)
                        t = jax.random.randint(
                            t_key,
                            (B,),
                            0,
                            self.agent.num_timesteps,
                        )
                        action_h = actions_seq[:, h, :]
                        h_idx = jnp.full((B,), h, dtype=jnp.float32)

                        def denoiser_h(t_in, x):
                            return self.agent.policy(policy_p, obs, x, t_in, h_idx)

                        loss_h = self.agent.diffusion.p_loss(
                            noise_key,
                            denoiser_h,
                            t,
                            jax.lax.stop_gradient(action_h),
                        )
                        return loss_h

                    keys_h = jax.random.split(policy_key, H)
                    losses = jax.vmap(loss_for_horizon_step)(jnp.arange(H), keys_h)
                    return jnp.mean(losses)
                else:
                    # Single-step policy training or warmup with Experience data
                    h_zeros = jnp.zeros((obs.shape[0],), dtype=jnp.float32)

                    def denoiser(t, x):
                        return self.agent.policy(policy_p, obs, x, t, h_zeros)

                    t_key, noise_key = jax.random.split(policy_key)
                    t = jax.random.randint(
                        t_key,
                        (obs.shape[0],),
                        0,
                        self.agent.num_timesteps,
                    )
                    loss = self.agent.diffusion.p_loss(
                        noise_key,
                        denoiser,
                        t,
                        jax.lax.stop_gradient(action),
                    )
                    return loss

            def seq_policy_loss_fn(seq_policy_p: hk.Params) -> jax.Array:
                """Joint sequence policy loss."""
                B = obs.shape[0]

                def seq_denoiser(t_in, x_seq):
                    return self.agent.seq_policy(seq_policy_p, obs, x_seq, t_in)

                t_key, noise_key = jax.random.split(policy_key)
                t = jax.random.randint(
                    t_key,
                    (B,),
                    0,
                    self.agent.num_timesteps,
                )
                loss = self.agent.diffusion.p_loss(
                    noise_key,
                    seq_denoiser,
                    t,
                    jax.lax.stop_gradient(actions_seq),
                )
                return loss

            # Compute gradients based on mode
            if self.joint_seq and actions_seq is not None and self.agent.seq_policy is not None:
                # Joint sequence mode: train seq_policy, skip per-step policy training
                seq_pol_loss, seq_pol_grads = jax.value_and_grad(seq_policy_loss_fn)(seq_policy_params)
                pol_loss = seq_pol_loss
                pol_grads = jax.tree_map(jnp.zeros_like, policy_params)  # No update to per-step policy
            else:
                # Per-step policy mode (factorized H-step or single-step)
                pol_loss, pol_grads = jax.value_and_grad(policy_loss_fn)(policy_params)
                seq_pol_grads = None

            def param_update(optim, params_in, grads_in, opt_state_in, scale: jax.Array):
                updates, new_opt_state = optim.update(grads_in, opt_state_in)
                updates = jax.tree_map(lambda u: scale * u, updates)
                new_params = optax.apply_updates(params_in, updates)
                return new_params, new_opt_state

            dyn_params, dyn_opt_state = param_update(
                self.dyn_optim, dyn_params, dyn_grads, dyn_opt_state, lr_scale_dyn
            )
            reward_params, reward_opt_state = param_update(
                self.reward_optim, reward_params, rew_grads, reward_opt_state, lr_scale_reward
            )
            policy_params, policy_opt_state = param_update(
                self.policy_optim, policy_params, pol_grads, policy_opt_state, lr_scale_policy
            )

            # Update seq_policy if in joint sequence mode
            if self.joint_seq and seq_pol_grads is not None and self.seq_policy_optim is not None:
                seq_policy_params, seq_policy_opt_state = param_update(
                    self.seq_policy_optim, seq_policy_params, seq_pol_grads, seq_policy_opt_state, lr_scale_policy
                )

            new_params = ModelBasedParams(
                policy=policy_params,
                target_policy=target_policy_params,
                dynamics=dyn_params,
                reward=reward_params,
                value=value_params,
                target_value=target_value_params,
                log_alpha=log_alpha,
                seq_policy=seq_policy_params,
            )

            new_opt_state = MbPcOptStates(
                policy=policy_opt_state,
                dynamics=dyn_opt_state,
                reward=reward_opt_state,
                value=value_opt_state,
                seq_policy=seq_policy_opt_state,
            )

            return new_params, new_opt_state, dyn_loss, rew_loss, pol_loss

        @jax.jit
        def stateless_update(
            key: jax.Array,
            state: MbPcTrainState,
            data: Union[Experience, SequenceExperience],
        ) -> Tuple[MbPcTrainState, Metric]:
            params = state.params
            opt_state = state.opt_state
            step = state.step
            hyper = state.hyper

            # Effective learning-rate scales under the current hyperparameters.
            # The policy/dynamics/reward scales are consumed inside
            # core_supervised_update; the value-network scale is used below
            # when applying value parameter updates.
            lr_scale_policy, lr_scale_dyn, lr_scale_reward, lr_scale_value = _effective_lr_scales(hyper)

            # Handle both Experience and SequenceExperience data types
            # Check for SequenceExperience by looking for 'actions' attribute
            is_sequence = hasattr(data, 'actions')
            if is_sequence:
                # SequenceExperience: extract first transition for dyn/reward,
                # full sequence for policy
                obs = data.obs  # [B, obs_dim]
                actions_seq = data.actions  # [B, H, act_dim]
                action = actions_seq[:, 0, :]  # First action for dynamics/reward
                reward = data.rewards[:, 0]  # First reward
                next_obs = data.next_obs_seq[:, 0, :]  # Next obs after first action
                done = data.dones[:, 0]  # Done flag for first transition
            else:
                # Standard Experience
                obs, action, reward, next_obs, done = (
                    data.obs,
                    data.action,
                    data.reward,
                    data.next_obs,
                    data.done,
                )
                actions_seq = None

            reward_scaled = reward * self.reward_scale

            # Split off a key for any model-based Monte Carlo estimate of V(next_state).
            value_key, key = jax.random.split(key)

            (
                policy_params,
                target_policy_params,
                dyn_params,
                reward_params,
                value_params,
                target_value_params,
                log_alpha,
                seq_policy_params,
            ) = params

            (
                policy_opt_state,
                dyn_opt_state,
                reward_opt_state,
                value_opt_state,
                seq_policy_opt_state,
            ) = opt_state

            if self.use_value:
                def inference_v_next(
                    params_planner: ModelBasedParams,
                    value_targ_p: hk.Params,
                    key_iv: jax.Array,
                    s: jax.Array,
                ) -> jax.Array:
                    """Estimate E[V(s') | s] using the full planner-based inference sampler.

                    For each state s_i in the batch, sample value_td_num_actions actions and, for each
                    action, a full s' population of size sprime_num_particles from the planner, then
                    average V_targ over all resulting states.
                    """
                    batch_size = s.shape[0]
                    num_actions = max(self.value_td_num_actions, 1)

                    def per_state(key_s: jax.Array, s_i: jax.Array) -> jax.Array:
                        keys_a = jax.random.split(key_s, num_actions)

                        def per_action(k_a: jax.Array) -> jax.Array:
                            _, sps_clean = planner_rollout_single(k_a, params_planner, s_i)
                            return sps_clean  # (sprime_num_particles, obs_dim)

                        # (num_actions, sprime_num_particles, obs_dim)
                        sps_all = jax.vmap(per_action)(keys_a)
                        # For TD, treat planner-sampled next states as a fixed distribution and
                        # do not backpropagate gradients through the planner trajectory.
                        sps_all = jax.lax.stop_gradient(sps_all)
                        sps_flat = sps_all.reshape(-1, self.agent.obs_dim)
                        v_vals = self.agent.value(value_targ_p, sps_flat)
                        if self.ucb_sprime_coeff != 0.0:
                            mean_v = jnp.mean(v_vals)
                            std_v = jnp.std(v_vals)
                            return mean_v + self.ucb_sprime_coeff * std_v
                        else:
                            return jnp.mean(v_vals)

                    keys_s = jax.random.split(key_iv, batch_size)
                    return jax.vmap(per_state)(keys_s, s)

                def _ddim_sample_action(
                    key_da: jax.Array,
                    policy_p: hk.Params,
                    s_i: jax.Array,
                ) -> jax.Array:
                    """Deterministic DDIM-style sample from the base diffusion policy at state s_i.

                    This mirrors GaussianDiffusion.p_sample but drops the stochastic noise term,
                    yielding a single unguided action sample from the base policy.
                    """

                    # Initialize latent action at the terminal diffusion level.
                    x = 0.5 * jax.random.normal(
                        key_da,
                        (1, self.agent.act_dim),
                        dtype=jnp.float32,
                    )

                    def model_fn(t_idx: jax.Array, x_t: jax.Array) -> jax.Array:
                        # Always pass h=0 for single-step sampling
                        h_zeros = jnp.zeros((1,), dtype=jnp.float32)
                        return self.agent.policy(
                            policy_p,
                            s_i[None, :],
                            x_t,
                            t_idx,
                            h_zeros,
                        )

                    def body_fn(x_t: jax.Array, t_idx: jax.Array):
                        noise_pred = model_fn(t_idx, x_t)
                        model_mean, model_log_variance = self.agent.diffusion.p_mean_variance(
                            t_idx,
                            x_t,
                            noise_pred,
                        )
                        # DDIM-style deterministic update: drop the stochastic term.
                        x_next = model_mean
                        return x_next, None

                    t_seq = jnp.arange(self.agent.num_timesteps)[::-1]
                    x_final, _ = jax.lax.scan(body_fn, x, t_seq)

                    act = jnp.clip(x_final[0], -1.0, 1.0)
                    act = jnp.where(jnp.isfinite(act), act, jnp.zeros_like(act))
                    return act

                def unguided_v_next(
                    policy_p: hk.Params,
                    dyn_p: hk.Params,
                    value_targ_p: hk.Params,
                    key_uv: jax.Array,
                    s: jax.Array,
                ) -> jax.Array:
                    """Estimate E_base[V(s') | s] using unguided DDIM samples from the base policy.

                    For each state s_i in the batch, sample value_td_num_actions actions from the
                    base diffusion policy (without planner guidance or MALA), propagate each
                    action through the learned dynamics to obtain s', and average V_targ(s').
                    """

                    batch_size = s.shape[0]
                    num_actions = max(self.value_td_num_actions, 1)

                    def per_state(key_s: jax.Array, s_i: jax.Array) -> jax.Array:
                        keys_a = jax.random.split(key_s, num_actions)

                        def per_action(k_a: jax.Array) -> jax.Array:
                            a_i = _ddim_sample_action(k_a, policy_p, s_i)

                            if self.deterministic_dyn:
                                s_next = _predict_next_obs_deterministic(
                                    dyn_p,
                                    s_i[None, :],
                                    a_i[None, :],
                                )[0]
                            else:
                                def dyn_model(t_idx: jax.Array, x_t: jax.Array) -> jax.Array:
                                    return self.agent.dynamics(
                                        dyn_p,
                                        s_i[None, :],
                                        a_i[None, :],
                                        x_t,
                                        t_idx,
                                    )

                                s_next = self.agent.dyn_diffusion.p_sample(
                                    k_a,
                                    dyn_model,
                                    (1, self.agent.obs_dim),
                                )[0]

                            v_val = self.agent.value(value_targ_p, s_next[None, :])[0]
                            return v_val

                        v_vals = jax.vmap(per_action)(keys_a)
                        if self.ucb_sprime_coeff != 0.0:
                            mean_v = jnp.mean(v_vals)
                            std_v = jnp.std(v_vals)
                            return mean_v + self.ucb_sprime_coeff * std_v
                        else:
                            return jnp.mean(v_vals)

                    keys_s = jax.random.split(key_uv, batch_size)
                    return jax.vmap(per_state)(keys_s, s)

                def value_loss_fn(val_p: hk.Params) -> jax.Array:
                    v = self.agent.value(val_p, obs)
                    if self.inference_value_td:
                        # Planner-based TD: use the inference sampler to define the next-state
                        # distribution and average V_targ over all planner-sampled states.
                        params_planner = ModelBasedParams(
                            policy=policy_params,
                            target_policy=target_policy_params,
                            dynamics=dyn_params,
                            reward=reward_params,
                            value=val_p,
                            target_value=target_value_params,
                            log_alpha=log_alpha,
                        )
                        v_next = inference_v_next(
                            params_planner,
                            target_value_params,
                            value_key,
                            obs,
                        )
                    elif self.unguided_value_td:
                        # Unguided TD: sample actions from the base diffusion policy (DDIM)
                        # and propagate through the dynamics model to obtain s'.
                        v_next = unguided_v_next(
                            policy_params,
                            dyn_params,
                            target_value_params,
                            value_key,
                            obs,
                        )
                    else:
                        # Replay TD: bootstrap purely from buffer next_obs.
                        v_next = self.agent.value(target_value_params, next_obs)
                    v_target = reward_scaled + (1.0 - done) * self.gamma * v_next
                    return jnp.mean((v - jax.lax.stop_gradient(v_target)) ** 2)

                val_loss, val_grads = jax.value_and_grad(value_loss_fn)(value_params)
            else:
                val_loss = jnp.array(0.0, dtype=jnp.float32)
                val_grads = jax.tree_map(jnp.zeros_like, value_params)

            def param_update(optim, params_in, grads_in, opt_state_in):
                updates, new_opt_state = optim.update(grads_in, opt_state_in)
                updates = jax.tree_map(lambda u: lr_scale_value * u, updates)
                new_params = optax.apply_updates(params_in, updates)
                return new_params, new_opt_state

            value_params, value_opt_state = param_update(
                self.value_optim, value_params, val_grads, value_opt_state
            )

            target_value_params = optax.incremental_update(
                value_params,
                target_value_params,
                self.tau,
            )

            params_with_value = ModelBasedParams(
                policy=policy_params,
                target_policy=target_policy_params,
                dynamics=dyn_params,
                reward=reward_params,
                value=value_params,
                target_value=target_value_params,
                log_alpha=log_alpha,
                seq_policy=seq_policy_params,
            )

            opt_state_full = MbPcOptStates(
                policy=policy_opt_state,
                dynamics=dyn_opt_state,
                reward=reward_opt_state,
                value=value_opt_state,
                seq_policy=seq_policy_opt_state,
            )

            key, sup_key = jax.random.split(key)

            new_params, new_opt_state, dyn_loss, rew_loss, pol_loss = core_supervised_update(
                sup_key,
                params_with_value,
                opt_state_full,
                hyper,
                obs,
                action,
                next_obs,
                reward_scaled,
                actions_seq,
            )

            new_state = MbPcTrainState(
                params=new_params,
                opt_state=new_opt_state,
                step=step + 1,
                hyper=hyper,
                hyper_opt_state=state.hyper_opt_state,
            )

            logs: Metric = {
                "dyn_loss": dyn_loss,
                "reward_loss": rew_loss,
                "value_loss": val_loss,
                "policy_loss_bc": pol_loss,
            }

            return new_state, logs

        @jax.jit
        def stateless_supervised_update(
            key: jax.Array,
            state: MbPcTrainState,
            data: Union[Experience, SequenceExperience],
        ) -> Tuple[MbPcTrainState, Metric]:
            params = state.params
            opt_state = state.opt_state
            step = state.step
            hyper = state.hyper

            # Handle both Experience and SequenceExperience data types
            # Check for SequenceExperience by looking for 'actions' attribute
            is_sequence = hasattr(data, 'actions')
            if is_sequence:
                # SequenceExperience: extract first transition for dyn/reward,
                # full sequence for policy
                obs = data.obs  # [B, obs_dim]
                actions_seq = data.actions  # [B, H, act_dim]
                action = actions_seq[:, 0, :]  # First action for dynamics/reward
                reward = data.rewards[:, 0]  # First reward
                next_obs = data.next_obs_seq[:, 0, :]  # Next obs after first action
            else:
                # Standard Experience
                obs, action, reward, next_obs, _ = (
                    data.obs,
                    data.action,
                    data.reward,
                    data.next_obs,
                    data.done,
                )
                actions_seq = None

            reward_scaled = reward * self.reward_scale

            new_params, new_opt_state, dyn_loss, rew_loss, pol_loss = core_supervised_update(
                key,
                params,
                opt_state,
                hyper,
                obs,
                action,
                next_obs,
                reward_scaled,
                actions_seq,
            )

            val_loss = jnp.array(0.0, dtype=jnp.float32)

            new_state = MbPcTrainState(
                params=new_params,
                opt_state=new_opt_state,
                step=step,
                hyper=hyper,
                hyper_opt_state=state.hyper_opt_state,
            )

            logs: Metric = {
                "dyn_loss": dyn_loss,
                "reward_loss": rew_loss,
                "value_loss": val_loss,
                "policy_loss_bc": pol_loss,
            }

            return new_state, logs

        def planner_rollout_single(
            key: jax.Array,
            params_planner: ModelBasedParams,
            obs_i: jax.Array,
        ) -> Tuple[jax.Array, jax.Array]:
            (
                policy_params,
                target_policy_params,
                dyn_params,
                reward_params,
                value_params,
                target_value_params,
                log_alpha,
                _seq_policy_params,
            ) = params_planner

            # Single-observation reverse diffusion loop with a same-noise-level s' population
            # that is refreshed at each level using the transition diffusion.
            T = self.agent.num_timesteps
            x_key, noise_key, s_key = jax.random.split(key, 3)
            x = 0.5 * jax.random.normal(x_key, (1, self.agent.act_dim), dtype=jnp.float32)
            noise = jax.random.normal(noise_key, (T, 1, self.agent.act_dim), dtype=jnp.float32)

            # Initialize s'_T at maximum noise, analogous to a_T.
            sps = jax.random.normal(
                s_key,
                (self.sprime_num_particles, self.agent.obs_dim),
                dtype=jnp.float32,
            )

            B_act = self.agent.diffusion.beta_schedule()
            B_dyn = self.agent.dyn_diffusion.beta_schedule()

            def refresh_sps(
                key_s: jax.Array,
                sps_t: jax.Array,
                a_clean_t: jax.Array,
                t: jax.Array,
            ):
                L = self.sprime_refresh_steps
                if L <= 0:
                    return key_s, sps_t, sps_t

                B = self.bprop_refresh_steps
                if B < 0:
                    B = 0
                if B > L:
                    B = L

                def step(carry, _):
                    key_i, x_dyn, x0_hat_prev = carry
                    key_i, key_noise = jax.random.split(key_i)
                    batch = x_dyn.shape[0]
                    s_rep = jnp.repeat(obs_i[None, :], batch, axis=0)
                    a_rep = jnp.repeat(a_clean_t, batch, axis=0)
                    t_rep = jnp.full((batch,), t)
                    dyn_eps = self.agent.dynamics(
                        dyn_params,
                        s_rep,
                        a_rep,
                        x_dyn,
                        t_rep,
                    )
                    x0_hat_s = (
                        x_dyn * B_dyn.sqrt_recip_alphas_cumprod[t]
                        - dyn_eps * B_dyn.sqrt_recipm1_alphas_cumprod[t]
                    )
                    alpha_bar_t = B_dyn.alphas_cumprod[t]
                    sqrt_alpha_t = B_dyn.sqrt_alphas_cumprod[t]
                    sqrt_oma_t = B_dyn.sqrt_one_minus_alphas_cumprod[t]

                    if self.sprime_refresh_type == "ula":
                        # ULA-style refresh at fixed noise level t using an approximate
                        # state score from the dynamics diffusion model.
                        denom = jnp.maximum(
                            jnp.float32(1.0) - alpha_bar_t,
                            jnp.float32(1e-6),
                        )
                        # Approximate score ∇_{x_t} log q(x_t | s,a) via Tweedie x0_hat.
                        sc = (sqrt_alpha_t * x0_hat_s - x_dyn) / denom
                        eta_s = jnp.float32(self.sprime_cs) * denom
                        noise_dyn = jax.random.normal(
                            key_noise,
                            x_dyn.shape,
                            dtype=jnp.float32,
                        )
                        sd = jnp.sqrt(
                            jnp.float32(2.0)
                            * jnp.maximum(eta_s, jnp.float32(1e-12))
                        )
                        x_dyn_next = x_dyn + eta_s * sc + sd * noise_dyn
                    else:
                        # Default: Tweedie-style re-noising recurrence at level t.
                        noise_dyn = jax.random.normal(
                            key_noise,
                            x_dyn.shape,
                            dtype=jnp.float32,
                        )
                        x_dyn_next = (
                            sqrt_alpha_t * x0_hat_s
                            + sqrt_oma_t * noise_dyn
                        )
                    return (key_i, x_dyn_next, x0_hat_s), None

                # No backprop through refresh: compute s' with a full L-step recurrence
                # and stop gradients on the clean estimate.
                if B == 0:
                    init_carry = (key_s, sps_t, sps_t)
                    (key_out, sps_out, x0_hat_last), _ = jax.lax.scan(
                        step,
                        init_carry,
                        None,
                        length=L,
                    )
                    x0_hat_last = jax.lax.stop_gradient(x0_hat_last)
                    return key_out, sps_out, x0_hat_last

                # Split into (L - B) no-grad refresh steps and B grad-carrying steps.
                num_init = L - B
                if num_init > 0:
                    init_carry = (key_s, sps_t, sps_t)
                    (key_mid, sps_mid, _), _ = jax.lax.scan(
                        step,
                        init_carry,
                        None,
                        length=num_init,
                    )
                    sps_mid = jax.lax.stop_gradient(sps_mid)
                else:
                    key_mid = key_s
                    sps_mid = sps_t

                init_carry = (key_mid, sps_mid, sps_mid)
                (key_out, sps_out, x0_hat_last), _ = jax.lax.scan(
                    step,
                    init_carry,
                    None,
                    length=B,
                )
                return key_out, sps_out, x0_hat_last

            def body_fn(carry, inp):
                x_t, sps_t, key_s = carry
                t, z = inp

                # Reward objective that threads the s' population and RNG key and returns
                # both the scalar expected reward and auxiliaries needed for sampling.
                def reward_objective(
                    x_local: jax.Array,
                    key_s_local: jax.Array,
                    sps_t_local: jax.Array,
                ):
                    # Always pass h=0 for single-step planner
                    h_zeros = jnp.zeros((1,), dtype=jnp.float32)
                    noise_pred_local = self.agent.policy(
                        policy_params,
                        obs_i[None, :],
                        x_local,
                        t,
                        h_zeros,
                    )
                    a_clean_local = (
                        x_local * B_act.sqrt_recip_alphas_cumprod[t]
                        - noise_pred_local * B_act.sqrt_recipm1_alphas_cumprod[t]
                    )
                    a_clean_local = jnp.clip(a_clean_local, -1.0, 1.0)
                    a_clean_local = jnp.where(
                        jnp.isfinite(a_clean_local),
                        a_clean_local,
                        jnp.zeros_like(a_clean_local),
                    )
                    if self.deterministic_dyn:
                        s_next = _predict_next_obs_deterministic(
                            dyn_params,
                            obs_i[None, :],
                            a_clean_local,
                        )
                        sps_clean = jnp.repeat(
                            s_next,
                            self.sprime_num_particles,
                            axis=0,
                        )
                        key_s_new = key_s_local
                        sps_new = sps_clean
                    else:
                        key_s_new, sps_new, sps_clean = refresh_sps(
                            key_s_local,
                            sps_t_local,
                            a_clean_local,
                            t,
                        )
                    s_rep = jnp.repeat(
                        obs_i[None, :],
                        self.sprime_num_particles,
                        axis=0,
                    )
                    a_rep = jnp.repeat(
                        a_clean_local,
                        self.sprime_num_particles,
                        axis=0,
                    )
                    r_vals = self.agent.reward(
                        reward_params,
                        s_rep,
                        a_rep,
                        sps_clean,
                    )
                    if self.use_value:
                        v_vals = self.agent.value(
                            value_params,
                            sps_clean,
                        )
                        q_vals = r_vals + self.gamma * v_vals
                    else:
                        q_vals = r_vals

                    # Aggregate R+gamma*V across the s' population. When
                    # ucb_sprime_coeff != 0, use a simple UCB-style statistic
                    # mean + k * std. Otherwise, optionally use the entropic
                    # aggregator when configured, falling back to a plain
                    # mean.
                    if self.ucb_sprime_coeff != 0.0:
                        mean_q = jnp.mean(q_vals)
                        std_q = jnp.std(q_vals)
                        mean_r = mean_q + self.ucb_sprime_coeff * std_q
                    elif self.entropic_sprime_agg and self.tfg_lambda != 0.0:
                        z = self.tfg_lambda * q_vals
                        z_max = jnp.max(z)
                        log_mean_exp = z_max + jnp.log(
                            jnp.mean(jnp.exp(z - z_max))
                        )
                        mean_r = log_mean_exp / self.tfg_lambda
                    else:
                        mean_r = jnp.mean(q_vals)
                    aux = (noise_pred_local, key_s_new, sps_new, sps_clean)
                    return mean_r, aux

                # Full Metropolis-adjusted Langevin (MALA) corrector in x_t using an energy
                # that combines the model-based energy (when available) with the
                # long-horizon-scaled expected reward.
                def mala_corrector(x_init, key_s_init, sps_init):
                    def energy_fn(
                        x_cur: jax.Array,
                        key_s_cur: jax.Array,
                        sps_cur: jax.Array,
                    ):
                        mean_r, (_, key_s_new, sps_new, _) = reward_objective(
                            x_cur,
                            key_s_cur,
                            sps_cur,
                        )
                        if getattr(self.agent, "energy_mode", False) and self.agent.energy_fn is not None:
                            # Model-based energy from the energy-parameterized policy.
                            # Always pass h=0 for single-step planner
                            h_zeros_e = jnp.zeros((1,), dtype=jnp.float32)
                            E_model = self.agent.energy_fn(
                                policy_params,
                                obs_i[None, :],
                                x_cur,
                                t,
                                h_zeros_e,
                            )
                            # Reduce to scalar in case of batch shape.
                            E_model = jnp.mean(E_model)
                        else:
                            E_model = jnp.array(0.0, dtype=jnp.float32)

                        # Total energy: base model energy minus long-horizon-scaled reward.
                        # Include tfg_lambda so that MALA and diffusion guidance share the same tilt.
                        E = E_model - self.tfg_lambda * mean_r
                        aux = (key_s_new, sps_new)
                        return E, aux

                    def log_gauss(xv: jax.Array, meanv: jax.Array, eta: jax.Array) -> jax.Array:
                        diff = xv - meanv
                        return -jnp.sum(diff * diff, axis=-1) / (
                            jnp.float32(4.0) * eta
                        )

                    # Base step size for this diffusion level, analogous to eta_base_t in dpmd.
                    eta_base_t = jnp.maximum(
                        B_act.betas[t],
                        jnp.float32(1e-8),
                    )

                    # Per-level adaptive log step-scale (reset each diffusion level).
                    log_eta_scale0 = jnp.float32(0.0)
                    grad_E0 = jnp.zeros_like(x_init)

                    def corr_body(i, state):
                        x_cur, key_s_cur, sps_cur, log_eta_scale, grad_E_cur = state

                        # Current energy and gradient
                        (E_x, (key_s_mid, sps_mid)), grad_E_x = jax.value_and_grad(
                            energy_fn,
                            argnums=0,
                            has_aux=True,
                        )(x_cur, key_s_cur, sps_cur)

                        # Adaptive step size: eta_k = exp(log_eta_scale) * eta_base_t.
                        eta_k = jnp.clip(
                            jnp.exp(log_eta_scale) * eta_base_t,
                            jnp.float32(1e-8),
                            jnp.float32(0.5),
                        )

                        key_s_mid, noise_key, u_key = jax.random.split(
                            key_s_mid,
                            3,
                        )
                        z = jax.random.normal(
                            noise_key,
                            x_cur.shape,
                            dtype=jnp.float32,
                        )
                        sd = jnp.sqrt(jnp.float32(2.0) * eta_k)
                        x_prop = x_cur - eta_k * grad_E_x + sd * z

                        # Proposal energy and gradient
                        (E_x_prop, (key_s_prop, sps_prop)), grad_E_x_prop = (
                            jax.value_and_grad(
                                energy_fn,
                                argnums=0,
                                has_aux=True,
                            )(x_prop, key_s_mid, sps_mid)
                        )

                        mean_f = x_cur - eta_k * grad_E_x
                        mean_r = x_prop - eta_k * grad_E_x_prop

                        log_q_prop_given_x = log_gauss(x_prop, mean_f, eta_k)
                        log_q_x_given_prop = log_gauss(x_cur, mean_r, eta_k)

                        log_alpha = (
                            -E_x_prop
                            + E_x
                            + (log_q_x_given_prop - log_q_prop_given_x)
                        )

                        u = jax.random.uniform(
                            u_key,
                            E_x.shape,
                            dtype=jnp.float32,
                        )
                        accept = jnp.log(u) < jnp.minimum(
                            jnp.float32(0.0),
                            log_alpha,
                        )

                        def choose(new, old):
                            # Ensure scalar predicate for lax.cond; accept may be shape () or (1,).
                            pred = jnp.squeeze(accept)
                            return jax.lax.cond(
                                pred,
                                lambda _: new,
                                lambda _: old,
                                operand=None,
                            )

                        x_next = choose(x_prop, x_cur)
                        key_s_next = choose(key_s_prop, key_s_mid)
                        sps_next = choose(sps_prop, sps_mid)
                        grad_E_next = choose(grad_E_x_prop, grad_E_x)
                        acc_rate = jnp.mean(accept.astype(jnp.float32))
                        target = jnp.float32(0.574)
                        adapt_rate = jnp.float32(0.05)
                        log_eta_scale = log_eta_scale + adapt_rate * (acc_rate - target)

                        return (x_next, key_s_next, sps_next, log_eta_scale, grad_E_next)

                    x_final, key_s_final, sps_final, _, grad_E_final = jax.lax.fori_loop(
                        0,
                        self.action_steps_per_level,
                        corr_body,
                        (x_init, key_s_init, sps_init, log_eta_scale0, grad_E0),
                    )
                    return x_final, key_s_final, sps_final, grad_E_final

                if self.action_steps_per_level > 0:
                    x_used, key_s_used, sps_used, grad_E_used = mala_corrector(
                        x_t,
                        key_s,
                        sps_t,
                    )

                    if self.tfg_lambda != 0.0:
                        if getattr(self.agent, "energy_mode", False) and self.agent.energy_fn is not None:
                            def energy_model_single(x_in: jax.Array) -> jax.Array:
                                # Always pass h=0 for single-step planner
                                h_zeros_e = jnp.zeros((1,), dtype=jnp.float32)
                                E_m = self.agent.energy_fn(
                                    policy_params,
                                    obs_i[None, :],
                                    x_in,
                                    t,
                                    h_zeros_e,
                                )
                                return jnp.mean(E_m)

                            grad_E_model = jax.grad(energy_model_single)(x_used)
                        else:
                            grad_E_model = jnp.zeros_like(x_used)

                        grad_x = (grad_E_model - grad_E_used) / self.tfg_lambda
                    else:
                        grad_x = jnp.zeros_like(x_used)

                    _, (noise_pred, key_s_new, sps_t_new, sps_clean) = reward_objective(
                        x_used,
                        key_s_used,
                        sps_used,
                    )
                elif self.action_recur_steps > 0:
                    def recur_body(carry_recur, _):
                        x_cur, key_s_cur, sps_cur = carry_recur
                        key_s_cur, key_down, key_up = jax.random.split(key_s_cur, 3)

                        grad_x_recur, (noise_pred_recur, key_s_mid, sps_mid, _sps_clean_mid) = jax.grad(
                            reward_objective,
                            argnums=0,
                            has_aux=True,
                        )(x_cur, key_s_cur, sps_cur)

                        sigma_t = B_act.sqrt_one_minus_alphas_cumprod[t]
                        eps_guided = noise_pred_recur - self.tfg_lambda * sigma_t * grad_x_recur
                        model_mean_recur, model_log_variance_recur = self.agent.diffusion.p_mean_variance(
                            t,
                            x_cur,
                            eps_guided,
                        )

                        noise_down = jax.random.normal(key_down, x_cur.shape, dtype=jnp.float32)
                        x_tm1 = model_mean_recur + (t > 0) * jnp.exp(0.5 * model_log_variance_recur) * noise_down

                        alpha_t = B_act.alphas[t]
                        sqrt_alpha_t = jnp.sqrt(alpha_t)
                        sqrt_oma_t = jnp.sqrt(1.0 - alpha_t)
                        noise_up = jax.random.normal(key_up, x_cur.shape, dtype=jnp.float32)
                        x_back = sqrt_alpha_t * x_tm1 + sqrt_oma_t * noise_up

                        return (x_back, key_s_mid, sps_mid), None

                    (x_used, key_s_used, sps_used), _ = jax.lax.scan(
                        recur_body,
                        (x_t, key_s, sps_t),
                        None,
                        length=self.action_recur_steps,
                    )

                    grad_x, (noise_pred, key_s_new, sps_t_new, sps_clean) = jax.grad(
                        reward_objective,
                        argnums=0,
                        has_aux=True,
                    )(x_used, key_s_used, sps_used)
                else:
                    x_used, key_s_used, sps_used = x_t, key_s, sps_t
                    grad_x, (noise_pred, key_s_new, sps_t_new, sps_clean) = jax.grad(
                        reward_objective,
                        argnums=0,
                        has_aux=True,
                    )(x_used, key_s_used, sps_used)

                sigma_t = B_act.sqrt_one_minus_alphas_cumprod[t]
                noise_pred_guided = noise_pred - self.tfg_lambda * sigma_t * grad_x
                model_mean, model_log_variance = self.agent.diffusion.p_mean_variance(
                    t,
                    x_used,
                    noise_pred_guided,
                )
                x_next = model_mean + (t > 0) * jnp.exp(0.5 * model_log_variance) * z
                return (x_next, sps_t_new, key_s_new), sps_clean

            t_seq = jnp.arange(self.agent.num_timesteps)[::-1]
            (x_final, _, _), sps_clean_seq = jax.lax.scan(
                body_fn,
                (x, sps, s_key),
                (t_seq, noise),
            )
            sps_clean_final = sps_clean_seq[-1]
            act = jnp.clip(x_final[0], -1.0, 1.0)
            act = jnp.where(jnp.isfinite(act), act, jnp.zeros_like(act))
            return act, sps_clean_final

        def planner_rollout_horizon(
            key: jax.Array,
            params_planner: ModelBasedParams,
            obs_i: jax.Array,
        ) -> Tuple[jax.Array, jax.Array]:
            """Multi-step horizon planner for H_plan > 1.

            Implements the "mental model" horizon planner: the policy outputs an
            H-step action sequence conditioned only on the current state obs_i,
            and the dynamics model is used to roll out the resulting H-step state
            trajectory inside the objective. For deterministic dynamics, this uses
            single-step predictions. For stochastic (diffusion) dynamics, this
            samples from the diffusion dynamics model using DDIM at each horizon
            step. The state trajectory is *not* part of the Markov state of the
            MALA/diffusion process; it is recomputed from obs_i and the current
            Tweedie action estimates whenever needed.
            """
            (
                policy_params,
                _target_policy_params,
                dyn_params,
                reward_params,
                value_params,
                _target_value_params,
                _log_alpha,
                seq_policy_params,
            ) = params_planner

            H = self.H_plan
            T = self.agent.num_timesteps
            act_dim = self.agent.act_dim

            x_key, noise_key, loop_key = jax.random.split(key, 3)

            # Initialize H actions at noise level T.
            x = 0.5 * jax.random.normal(x_key, (H, act_dim), dtype=jnp.float32)
            noise = jax.random.normal(noise_key, (T, H, act_dim), dtype=jnp.float32)

            B_act = self.agent.diffusion.beta_schedule()

            # Precompute discount factors: gamma^0, gamma^1, ..., gamma^{H-1}, gamma^H
            gamma_powers = self.gamma ** jnp.arange(H + 1, dtype=jnp.float32)

            # =================================================================
            # Helper functions for horizon planning.
            # These take obs_0 as an explicit argument to avoid closure capture
            # issues with JAX's nested autodiff tracing.
            # =================================================================

            def _normalize_obs0(obs_0: jax.Array) -> jax.Array:
                """Normalize obs_0 to a 1D [obs_dim] vector.

                JAX's nested grad/scan transformations may introduce extra
                diffusion-step or batch dimensions (e.g., [T, obs_dim] or
                [B, T, obs_dim]) on a conceptually constant initial state.
                This helper collapses all leading dimensions and returns the
                first element as a 1D vector.
                """
                obs_0 = jax.lax.stop_gradient(obs_0)
                if obs_0.ndim == 1:
                    return obs_0
                # Last axis is obs_dim; collapse everything else.
                flat = obs_0.reshape(-1, obs_0.shape[-1])
                return flat[0]

            def get_tweedie_actions(x_t: jax.Array, t_idx: jax.Array, obs_0: jax.Array):
                """Get Tweedie action estimates for all H actions.
                
                Args:
                    x_t: Noisy actions [H, act_dim]
                    t_idx: Diffusion timestep (scalar)
                    obs_0: Initial observation (possibly with extra dims)
                """
                obs_vec = _normalize_obs0(obs_0)  # [obs_dim]

                if self.joint_seq and self.agent.seq_policy is not None:
                    t_batch = jnp.atleast_1d(t_idx)
                    noise_pred = self.agent.seq_policy(
                        seq_policy_params,
                        obs_vec[None, :],          # [1, obs_dim]
                        x_t[None, :, :],           # [1, H, act_dim]
                        t_batch,                   # [1]
                    )[0]
                    
                    a_clean = (
                        x_t * B_act.sqrt_recip_alphas_cumprod[t_idx]
                        - noise_pred * B_act.sqrt_recipm1_alphas_cumprod[t_idx]
                    )
                    a_clean = jnp.clip(a_clean, -1.0, 1.0)
                    a_clean = jnp.where(jnp.isfinite(a_clean), a_clean, jnp.zeros_like(a_clean))
                    return a_clean, noise_pred
                else:
                    s_cond = jnp.repeat(obs_vec[None, :], H, axis=0)  # [H, obs_dim]
                    h_indices = jnp.arange(H, dtype=jnp.float32)

                    def single_tweedie(s_h, x_h, h_idx):
                        h_batch = jnp.array([h_idx])
                        noise_pred = self.agent.policy(
                            policy_params, s_h[None, :], x_h[None, :], t_idx, h_batch,
                        )
                        a_clean = (
                            x_h * B_act.sqrt_recip_alphas_cumprod[t_idx]
                            - noise_pred[0] * B_act.sqrt_recipm1_alphas_cumprod[t_idx]
                        )
                        a_clean = jnp.clip(a_clean, -1.0, 1.0)
                        a_clean = jnp.where(jnp.isfinite(a_clean), a_clean, jnp.zeros_like(a_clean))
                        return a_clean, noise_pred[0]

                    return jax.vmap(single_tweedie)(s_cond, x_t, h_indices)

            def rollout_states(a_clean: jax.Array, obs_0: jax.Array, key_rollout: jax.Array = None) -> jax.Array:
                """Roll out H-step state trajectory.
                
                Args:
                    a_clean: Clean actions [H, act_dim]
                    obs_0: Initial observation (possibly with extra dims)
                    key_rollout: Random key for stochastic dynamics (None for deterministic)
                    
                Returns:
                    s_traj: [H+1, obs_dim] containing [s_0, s_1, ..., s_H]
                """
                obs_vec = _normalize_obs0(obs_0)  # [obs_dim]

                if self.deterministic_dyn:
                    def step(s_cur: jax.Array, a_h: jax.Array) -> Tuple[jax.Array, jax.Array]:
                        # s_cur is [obs_dim]; lift to batch-1 for dynamics call.
                        s_next = _predict_next_obs_deterministic(
                            dyn_params, s_cur[None, :], a_h[None, :],
                        )[0]
                        return s_next, s_next

                    # Scan over H actions; carry is [obs_dim].
                    _, s_seq = jax.lax.scan(step, obs_vec, a_clean)
                else:
                    # Stochastic dynamics: use DDIM sampling for each step
                    # Split key for each horizon step
                    keys_h = jax.random.split(key_rollout, H)
                    
                    def step_stochastic(carry, inputs):
                        s_cur, _ = carry
                        a_h, k_h = inputs
                        # s_cur is [obs_dim]; lift to batch-1 for dynamics call.
                        s_next = _ddim_sample_next_obs(
                            k_h, dyn_params, s_cur[None, :], a_h[None, :],
                        )[0]
                        return (s_next, None), s_next

                    # Scan over H actions with keys; carry is (s_cur, None).
                    _, s_seq = jax.lax.scan(step_stochastic, (obs_vec, None), (a_clean, keys_h))

                return jnp.concatenate([obs_vec[None, :], s_seq], axis=0)

            def compute_horizon_objective(x_t: jax.Array, t_idx: jax.Array, obs_0: jax.Array, key_dyn: jax.Array = None):
                """Compute H-step objective for guidance.
                
                Args:
                    x_t: Noisy actions [H, act_dim]
                    t_idx: Diffusion timestep (scalar)
                    obs_0: Initial observation [obs_dim] - passed explicitly
                    key_dyn: Random key for stochastic dynamics (None for deterministic)
                """
                a_clean, noise_preds = get_tweedie_actions(x_t, t_idx, obs_0)
                s_traj = rollout_states(a_clean, obs_0, key_dyn)

                def single_reward(s_h, a_h, sp_h):
                    return self.agent.reward(
                        reward_params, s_h[None, :], a_h[None, :], sp_h[None, :],
                    )[0]

                r_vals = jax.vmap(single_reward)(s_traj[:H], a_clean, s_traj[1:])
                total_obj = jnp.sum(gamma_powers[:H] * r_vals)

                if self.use_value:
                    v_terminal = self.agent.value(value_params, s_traj[-1:])[0]
                    total_obj = total_obj + gamma_powers[H] * v_terminal

                return total_obj, (a_clean, s_traj, noise_preds)

            def horizon_energy(x_t: jax.Array, t_idx: jax.Array, obs_0: jax.Array, key_dyn: jax.Array = None):
                """Energy function for MALA on H-action horizon.
                
                Args:
                    x_t: Noisy actions [H, act_dim]
                    t_idx: Diffusion timestep (scalar)
                    obs_0: Initial observation (possibly with extra dims)
                    key_dyn: Random key for stochastic dynamics (None for deterministic)
                """
                obs_vec = _normalize_obs0(obs_0)  # [obs_dim]
                
                obj, aux = compute_horizon_objective(x_t, t_idx, obs_vec, key_dyn)

                if getattr(self.agent, "energy_mode", False):
                    if self.joint_seq and self.agent.seq_energy_fn is not None:
                        t_batch = jnp.atleast_1d(t_idx)
                        E_model = self.agent.seq_energy_fn(
                            seq_policy_params, obs_vec[None, :], x_t[None, :, :], t_batch,
                        )[0]
                    elif self.agent.energy_fn is not None:
                        h_indices_e = jnp.arange(H, dtype=jnp.float32)

                        def single_energy(x_h, h_idx):
                            h_batch = jnp.array([h_idx])
                            E_m = self.agent.energy_fn(
                                policy_params, obs_vec[None, :], x_h[None, :], t_idx, h_batch,
                            )
                            return E_m[0]

                        E_model = jnp.sum(jax.vmap(single_energy)(x_t, h_indices_e))
                    else:
                        E_model = jnp.array(0.0, dtype=jnp.float32)
                else:
                    E_model = jnp.array(0.0, dtype=jnp.float32)

                E = E_model - self.tfg_lambda * obj
                return E, aux

            def mala_corrector_horizon(
                x_init: jax.Array,
                key_init: jax.Array,
                t_idx: jax.Array,
                obs_0: jax.Array,
            ) -> Tuple[jax.Array, jax.Array]:
                """MALA corrector for H-action horizon.
                
                Args:
                    x_init: Initial action sequence [H, act_dim]
                    key_init: Random key
                    t_idx: Diffusion timestep (scalar)
                    obs_0: Initial observation [obs_dim] - passed explicitly
                """
                eta_base_t = jnp.maximum(B_act.betas[t_idx], jnp.float32(1e-8))

                def corr_body(i, state):
                    x_cur, key_cur, log_eta_scale = state

                    # Split key for dynamics sampling (for stochastic dynamics)
                    key_cur, key_dyn = jax.random.split(key_cur)
                    
                    # Current energy and gradient - pass obs_0 and key explicitly
                    (E_x, _), grad_E_x = jax.value_and_grad(
                        lambda x: horizon_energy(x, t_idx, obs_0, key_dyn),
                        has_aux=True,
                    )(x_cur)

                    eta_k = jnp.clip(
                        jnp.exp(log_eta_scale) * eta_base_t,
                        jnp.float32(1e-8),
                        jnp.float32(0.5),
                    )

                    key_cur, noise_key, u_key, key_dyn_prop = jax.random.split(key_cur, 4)
                    z = jax.random.normal(noise_key, x_cur.shape, dtype=jnp.float32)
                    sd = jnp.sqrt(jnp.float32(2.0) * eta_k)
                    x_prop = x_cur - eta_k * grad_E_x + sd * z

                    # Proposal energy and gradient
                    (E_x_prop, _), grad_E_x_prop = jax.value_and_grad(
                        lambda x: horizon_energy(x, t_idx, obs_0, key_dyn_prop),
                        has_aux=True,
                    )(x_prop)

                    # MH acceptance ratio
                    mean_f = x_cur - eta_k * grad_E_x
                    mean_r = x_prop - eta_k * grad_E_x_prop

                    def log_gauss(xv, meanv, eta):
                        diff = xv - meanv
                        return -jnp.sum(diff * diff) / (jnp.float32(4.0) * eta)

                    log_q_prop_given_x = log_gauss(x_prop, mean_f, eta_k)
                    log_q_x_given_prop = log_gauss(x_cur, mean_r, eta_k)

                    log_alpha = -E_x_prop + E_x + log_q_x_given_prop - log_q_prop_given_x

                    u = jax.random.uniform(u_key, (), dtype=jnp.float32)
                    accept = jnp.log(u) < jnp.minimum(jnp.float32(0.0), log_alpha)

                    x_next = jax.lax.cond(accept, lambda: x_prop, lambda: x_cur)

                    # Adapt step size
                    acc_rate = accept.astype(jnp.float32)
                    target = jnp.float32(0.574)
                    adapt_rate = jnp.float32(0.05)
                    log_eta_scale_next = log_eta_scale + adapt_rate * (acc_rate - target)

                    return (x_next, key_cur, log_eta_scale_next)

                init_state = (x_init, key_init, jnp.float32(0.0))
                x_final, key_final, _ = jax.lax.fori_loop(
                    0, self.action_steps_per_level, corr_body, init_state,
                )
                return x_final, key_final

            def body_fn_horizon(carry, inp):
                x_t, key_t = carry
                t, z = inp

                if self.action_steps_per_level > 0:
                    x_used, key_new = mala_corrector_horizon(x_t, key_t, t, obs_i)
                else:
                    x_used, key_new = x_t, key_t

                # Split key for dynamics sampling (for stochastic dynamics)
                key_new, key_dyn, key_dyn_grad = jax.random.split(key_new, 3)

                # Compute objective - pass obs_i and dynamics key explicitly
                _, (a_clean, s_traj, noise_preds) = compute_horizon_objective(
                    x_used, t, obs_i, key_dyn,
                )

                # Guidance gradient if tfg_lambda > 0
                if self.tfg_lambda != 0.0:
                    grad_obj = jax.grad(
                        lambda x: compute_horizon_objective(x, t, obs_i, key_dyn_grad)[0],
                    )(x_used)
                    sigma_t = B_act.sqrt_one_minus_alphas_cumprod[t]
                    noise_preds_guided = noise_preds - self.tfg_lambda * sigma_t * grad_obj
                else:
                    noise_preds_guided = noise_preds

                # Diffusion step for each action independently
                def single_diffusion_step(x_h, noise_pred_h, z_h):
                    model_mean, model_log_variance = self.agent.diffusion.p_mean_variance(
                        t,
                        x_h[None, :],
                        noise_pred_h[None, :],
                    )
                    model_mean = jnp.squeeze(model_mean, axis=0)
                    model_log_variance = jnp.squeeze(model_log_variance)
                    x_next = model_mean + (t > 0) * jnp.exp(0.5 * model_log_variance) * z_h
                    return x_next

                x_next = jax.vmap(single_diffusion_step)(
                    x_used,
                    noise_preds_guided,
                    z,
                )

                return (x_next, key_new), s_traj

            t_seq = jnp.arange(T)[::-1]
            (x_final, _), s_traj_seq = jax.lax.scan(
                body_fn_horizon,
                (x, loop_key),
                (t_seq, noise),
            )

            # Return all H actions and the predicted state trajectory
            s_traj_final = s_traj_seq[-1]
            acts = jnp.clip(x_final, -1.0, 1.0)  # [H, act_dim]
            acts = jnp.where(jnp.isfinite(acts), acts, jnp.zeros_like(acts))
            s_next_pred = s_traj_final[1]  # Predicted next state from action 0

            return acts, s_next_pred

        def stateless_get_action(
            key: jax.Array,
            params: ModelBasedParams,
            obs: jax.Array,
        ) -> jax.Array:

            def sample_single(k: jax.Array, obs_i: jax.Array) -> jax.Array:
                if self.H_plan > 1:
                    # Use horizon planner for multi-step action planning when H_plan > 1
                    # Works with both deterministic and stochastic dynamics
                    acts, _ = planner_rollout_horizon(k, params, obs_i)
                    return acts[0]  # Return first action for closed-loop execution
                else:
                    # Use standard single-action planner (H_plan == 1)
                    act, _ = planner_rollout_single(k, params, obs_i)
                    return act

            if obs.ndim == 1:
                return sample_single(key, obs)
            else:
                B = obs.shape[0]
                keys = jax.random.split(key, B)
                return jax.vmap(sample_single)(keys, obs)

        def stateless_get_deterministic_action(
            params: ModelBasedParams,
            obs: jax.Array,
        ) -> jax.Array:
            key = jax.random.key(0)
            act = stateless_get_action(key, params, obs)
            if act.ndim == 2 and act.shape[0] == 1:
                act = act[0]
            return act

        self._implement_common_behavior(
            stateless_update,
            stateless_get_action,
            stateless_get_deterministic_action,
        )
        self._update_supervised = jax.jit(stateless_supervised_update)

        # Function to get all H actions for open-loop execution.
        # For a single observation obs with shape [obs_dim], this returns
        # [H, act_dim]. For a batch of observations with shape [B, obs_dim],
        # this returns [H, B, act_dim].
        def stateless_get_action_sequence(
            key: jax.Array,
            params: ModelBasedParams,
            obs: jax.Array,
        ) -> jax.Array:
            """Get full H-action sequence for open-loop execution."""

            def sample_single_seq(k: jax.Array, obs_i: jax.Array) -> jax.Array:
                if self.H_plan > 1:
                    # Works with both deterministic and stochastic dynamics
                    acts_i, _ = planner_rollout_horizon(k, params, obs_i)
                    return acts_i  # [H, act_dim]
                else:
                    act_i, _ = planner_rollout_single(k, params, obs_i)
                    return act_i[None, :]  # [1, act_dim]

            # Single-env case: obs is [obs_dim]
            if obs.ndim == 1:
                return sample_single_seq(key, obs)

            # Vectorized env: obs is [B, obs_dim]
            B = obs.shape[0]
            keys = jax.random.split(key, B)
            # vmap over batch: [B, H, act_dim]
            acts_bh = jax.vmap(sample_single_seq)(keys, obs)
            # Reorder to [H, B, act_dim] so that indexing on the leading
            # horizon dimension works uniformly for both cases.
            return jnp.swapaxes(acts_bh, 0, 1)

        self._get_action_sequence = jax.jit(stateless_get_action_sequence)

        def _value_replay_td_loss(
            params_val: ModelBasedParams,
            data_val: Experience,
        ) -> jax.Array:
            """Simple replay-based TD loss for the value network on validation data.

            This ignores model-based TD modes and uses a 1-step bootstrap from
            target_value on next_obs. It serves as the value-component of the
            overall hypergradient validation objective.
            """

            obs_v, reward_v, next_obs_v, done_v = (
                data_val.obs,
                data_val.reward,
                data_val.next_obs,
                data_val.done,
            )
            reward_scaled_v = reward_v * self.reward_scale

            (
                _policy_p,
                _targ_policy_p,
                _dyn_p,
                _reward_p,
                value_p,
                target_value_p,
                _log_alpha_p,
            ) = params_val

            v = self.agent.value(value_p, obs_v)
            v_next = self.agent.value(target_value_p, next_obs_v)
            v_target = reward_scaled_v + (1.0 - done_v) * self.gamma * v_next
            return jnp.mean((v - jax.lax.stop_gradient(v_target)) ** 2)

        def _hyper_validation_loss(
            params_val: ModelBasedParams,
            data_val: Experience,
            key: jax.Array,
        ) -> jax.Array:
            """Validation loss used for hypergradients over all four heads.

            This aggregates dynamics, reward, policy, and replay-style value TD
            losses on the validation batch so that each log learning-rate
            scale can receive a meaningful hypergradient signal.
            """

            obs_v, action_v, reward_v, next_obs_v, done_v = (
                data_val.obs,
                data_val.action,
                data_val.reward,
                data_val.next_obs,
                data_val.done,
            )
            reward_scaled_v = reward_v * self.reward_scale

            (
                policy_p,
                target_policy_p,
                dyn_p,
                reward_p,
                _value_p,
                _target_value_p,
                _log_alpha_p,
            ) = params_val

            # Dynamics validation loss
            def dynamics_loss_fn_v(dyn_p_in: hk.Params) -> jax.Array:
                if self.deterministic_dyn:
                    s_pred = _predict_next_obs_deterministic(dyn_p_in, obs_v, action_v)
                    return jnp.mean((s_pred - next_obs_v) ** 2)
                else:
                    def dyn_model(t, x_t):
                        return self.agent.dynamics(dyn_p_in, obs_v, action_v, x_t, t)

                    t_key, noise_key = jax.random.split(jax.random.fold_in(key, 0))
                    t = jax.random.randint(
                        t_key,
                        (obs_v.shape[0],),
                        0,
                        self.agent.num_timesteps,
                    )
                    return self.agent.dyn_diffusion.p_loss(
                        noise_key,
                        dyn_model,
                        t,
                        next_obs_v,
                    )

            dyn_loss_v = dynamics_loss_fn_v(dyn_p)

            # Reward validation loss
            def reward_loss_fn_v(rew_p_in: hk.Params) -> jax.Array:
                r_pred = self.agent.reward(rew_p_in, obs_v, action_v, next_obs_v)
                return jnp.mean((r_pred - reward_scaled_v) ** 2)

            rew_loss_v = reward_loss_fn_v(reward_p)

            # Policy validation loss (behavior-cloning style diffusion loss)
            def policy_loss_fn_v(policy_p_in: hk.Params) -> jax.Array:
                # Always pass h=0 for single-step validation
                h_zeros = jnp.zeros((obs_v.shape[0],), dtype=jnp.float32)

                def denoiser(t, x):
                    return self.agent.policy(policy_p_in, obs_v, x, t, h_zeros)

                t_key, noise_key = jax.random.split(jax.random.fold_in(key, 1))
                t = jax.random.randint(
                    t_key,
                    (obs_v.shape[0],),
                    0,
                    self.agent.num_timesteps,
                )
                loss = self.agent.diffusion.p_loss(
                    noise_key,
                    denoiser,
                    t,
                    jax.lax.stop_gradient(action_v),
                )
                return loss

            pol_loss_v = policy_loss_fn_v(policy_p)

            # Value replay TD loss component
            val_td_loss_v = _value_replay_td_loss(params_val, data_val)

            # Aggregate all four components. For now we use a simple sum; if
            # needed this can be extended to allow per-head weights.
            return dyn_loss_v + rew_loss_v + pol_loss_v + val_td_loss_v

        def _hyper_step(
            key: jax.Array,
            state: MbPcTrainState,
            train_data: Experience,
            val_data: Experience,
        ) -> Tuple[MbPcTrainState, Metric, Metric]:
            """One-step train+validation hypergradient update.

            This performs a normal training update using the current
            hyperparameters (log learning-rate scales), then differentiates a
            replay-style validation TD loss through that training step with
            respect to the log scales, and finally applies a gradient step in
            hyperparameter space.
            """

            accum_steps = max(1, int(self.hypergrad_accum_steps))

            def loss_wrt_h(h: MbPcHyperparams):
                # Split the incoming key into a training and validation
                # component so that we can evaluate the validation objective
                # under multiple independent RNG seeds when desired.
                train_key, val_key = jax.random.split(key)
                # Build a temporary state with updated hyperparameters.
                tmp_state = MbPcTrainState(
                    params=state.params,
                    opt_state=state.opt_state,
                    step=state.step,
                    hyper=h,
                    hyper_opt_state=state.hyper_opt_state,
                )

                # 1) Training step with current hyperparameters.
                new_state, train_logs = stateless_update(train_key, tmp_state, train_data)

                # 2) Validation loss evaluated at the updated parameters.
                # When accum_steps > 1, average the validation objective over
                # multiple independent RNG keys to reduce hypergradient
                # variance while keeping the number of train updates per
                # hyper step fixed at 1.
                if accum_steps == 1:
                    val_loss = _hyper_validation_loss(new_state.params, val_data, val_key)
                else:
                    val_keys = jax.random.split(val_key, accum_steps)

                    def single_val_loss(kv: jax.Array) -> jax.Array:
                        return _hyper_validation_loss(new_state.params, val_data, kv)

                    val_losses = jax.vmap(single_val_loss)(val_keys)
                    val_loss = jnp.mean(val_losses)
                return val_loss, (new_state, train_logs)

            (val_loss, (new_state, train_logs)), hyper_grads = jax.value_and_grad(
                loss_wrt_h,
                argnums=0,
                has_aux=True,
            )(state.hyper)

            # Gradient step in log-learning-rate space.
            hyper_updates, new_hyper_opt_state = self.hyper_optim.update(
                hyper_grads,
                state.hyper_opt_state,
                params=state.hyper,
            )
            new_hyper = optax.apply_updates(state.hyper, hyper_updates)

            final_state = MbPcTrainState(
                params=new_state.params,
                opt_state=new_state.opt_state,
                step=new_state.step,
                hyper=new_hyper,
                hyper_opt_state=new_hyper_opt_state,
            )

            # For now, treat the validation loss as a single scalar metric.
            val_logs: Metric = {"value_loss_replay": val_loss}
            return final_state, train_logs, val_logs

        # Store a partially-applied hyper step for use when hypergrad is enabled.
        self._hyper_step = jax.jit(_hyper_step)

    def get_effective_hparams(self) -> dict:
        """Expose effective hyperparameters for logging.

        In particular, sprime_num_particles may be overridden internally when
        deterministic dynamics are enabled, so we report its effective value
        here for wandb / config logging.
        """

        return {
            "sprime_num_particles": int(self.sprime_num_particles),
        }

    def get_policy_params(self):
        return self.state.params

    def update_supervised(self, key: jax.Array, data: Experience) -> Metric:
        self.state, info = self._update_supervised(key, self.state, data)
        return {k: float(v) for k, v in info.items() if not k.startswith('hist')}, {k: v for k, v in info.items() if k.startswith('hist')}

    def hyper_update(
        self,
        key: jax.Array,
        train_data: Experience,
        val_data: Experience,
    ) -> Tuple[Metric, Metric, Metric]:
        """Public entry point for a hypergradient-aware update.

        This is only used when the trainer enables hypergradient updates.
        It performs a single train+validation hyper step on the internal
        MbPcTrainState and returns (info, dist_info, val_info) dictionaries
        following the OffPolicyTrainer.update convention.
        """

        if not self.use_hypergrad:
            # Fallback: behave like a standard update followed by validation
            # metrics. This should not normally be hit when the trainer gate
            # is configured correctly, but keeps behavior well-defined.
            info, dist_info_only = self.update(key, train_data)
            val_key = jax.random.fold_in(key, 1)
            val_info_only = self.compute_validation_metrics(val_key, val_data)
            return info, dist_info_only, val_info_only

        # Perform a single hypergradient step using the Adam meta-optimizer
        # configured with hypergrad_lr.
        new_state, train_logs, val_logs = self._hyper_step(
            key,
            self.state,
            train_data,
            val_data,
        )
        self.state = new_state

        info = {k: float(v) for k, v in train_logs.items() if not k.startswith("hist")}
        dist_info: Metric = {}
        val_info = {k: float(v) for k, v in val_logs.items() if not k.startswith("hist")}
        return info, dist_info, val_info

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        """Get action with optional open-loop execution.
        
        In open-loop mode (when H_plan > 1 and open_loop=True), executes the full
        H-action plan sequentially without replanning until the sequence is exhausted.
        """
        if not self.open_loop:
            # Standard closed-loop: replan every step
            action = self._get_action(key, self.get_policy_params(), obs)
            return np.asarray(action)
        
        # Open-loop mode: use cached action sequence
        if self._cached_actions is None or self._action_index >= len(self._cached_actions):
            # Need to replan: get full H-action sequence
            acts = self._get_action_sequence(key, self.get_policy_params(), obs)
            self._cached_actions = np.asarray(acts)
            self._action_index = 0
        
        # Return next action from cached sequence
        action = self._cached_actions[self._action_index]
        self._action_index += 1
        return action

    def reset_open_loop(self):
        """Reset open-loop execution state. Call this at episode boundaries."""
        self._cached_actions = None
        self._action_index = 0
