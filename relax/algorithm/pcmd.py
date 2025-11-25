from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp
import optax
import haiku as hk
import numpy as np
import pickle

from relax.algorithm.base import Algorithm
from relax.network.pcmd import PcNet, PcParams
from relax.pcmd.standardizer import SASRStandardizer
from relax.pcmd.levels import PcLevelsConfig, build_levels_jax
from relax.pcmd.sampler_fac_rr import jax_run_pc_mala_fac_rr_seq
from relax.utils.diffusion import GaussianDiffusion
from relax.utils.experience import Experience
from relax.utils.typing_utils import Metric


class PcOptStates(NamedTuple):
    policy: optax.OptState
    dynamics: optax.OptState
    reward: optax.OptState
    value: optax.OptState


class PcTrainState(NamedTuple):
    params: PcParams
    opt_state: PcOptStates
    step: jax.Array
    standardizer: SASRStandardizer


class PCMD(Algorithm):

    def __init__(
        self,
        net: PcNet,
        params: PcParams,
        *,
        gamma: float = 0.99,
        lr_policy: float = 1e-3,
        lr_dyn: float = 1e-3,
        lr_reward: float = 1e-3,
        lr_value: float = 1e-3,
        reward_scale: float = 0.2,
        td_lambda: float = 0.95,
        ema_tau: float = 0.005,
        H_train: int = 1,
        H_plan: int = 1,
        num_timesteps: int = 20,
        beta_schedule_scale: float = 0.8,
        beta_schedule_type: str = "linear",
        levels_cfg: PcLevelsConfig = PcLevelsConfig(),
        points_per_seed: int = 20,
        refresh_L: int = 3,
        action_steps_per_level: int = 1,
        use_ula_refresh: bool = True,
        cs: float = 0.08,
        cs_accept: float = 0.1,
        bprop_refresh: bool = False,
        accept_sprime: str = "none",
        adapt_s_accept: bool = False,
        s_accept_target: float = 0.5,
        s_accept_lr: float = 0.05,
        use_crn: bool = True,
        level_offset: int = 1,
    ):
        self.net = net
        self.gamma = float(gamma)
        self.reward_scale = float(reward_scale)
        self.td_lambda = float(td_lambda)
        self.ema_tau = float(ema_tau)
        self.H_train = int(H_train)
        self.H_plan = int(H_plan)

        # Gaussian diffusion used for VP corruption / score targets
        self.diffusion = GaussianDiffusion(
            num_timesteps=int(num_timesteps),
            beta_schedule_scale=float(beta_schedule_scale),
            beta_schedule_type=str(beta_schedule_type),
        )

        # Build PC levels using the same cumulative alpha schedule as the GaussianDiffusion.
        Bcoef_levels = self.diffusion.beta_schedule()
        alpha_bar_np = np.array(Bcoef_levels.alphas_cumprod)
        K_levels = int(alpha_bar_np.shape[0])
        lam = float(levels_cfg.lambda_scale)
        name = str(levels_cfg.beta_schedule)
        if name == "linear":
            beta_t_np = np.linspace(0.0, lam, K_levels, dtype=np.float64)
        elif name == "late_constant":
            frac = 0.7
            cut = int(np.floor(frac * (K_levels - 1)))
            beta_t_np = np.zeros(K_levels, dtype=np.float64)
            beta_t_np[cut:] = lam
        elif name == "late_ramp":
            frac = 0.7
            cut = int(np.floor(frac * (K_levels - 1)))
            ramp_len = max(K_levels - cut, 1)
            beta_t_np = np.zeros(K_levels, dtype=np.float64)
            beta_t_np[cut:] = np.linspace(0.0, lam, ramp_len, dtype=np.float64)
        elif name == "sigma_power":
            p = float(levels_cfg.sigma_p)
            sigma2_np = np.maximum(1.0 - alpha_bar_np, 0.0)
            beta_t_np = lam * sigma2_np ** p
        else:
            beta_t_np = np.linspace(0.0, lam, K_levels, dtype=np.float64)
        eta_a_np = np.maximum(1.0 - alpha_bar_np, 1e-6)
        self.levels_jax = {
            "K": K_levels,
            "alpha_bar": jnp.asarray(alpha_bar_np, dtype=jnp.float32),
            "beta_t": jnp.asarray(beta_t_np, dtype=jnp.float32),
            "eta_a": jnp.asarray(eta_a_np, dtype=jnp.float32),
        }

        self.points_per_seed = int(points_per_seed)
        self.refresh_L = int(refresh_L)
        self.action_steps_per_level = int(action_steps_per_level)
        self.use_ula_refresh = bool(use_ula_refresh)
        self.cs = float(cs)
        self.cs_accept = float(cs_accept)
        self.bprop_refresh = bool(bprop_refresh)
        self.accept_sprime = str(accept_sprime)
        self.adapt_s_accept = bool(adapt_s_accept)
        self.s_accept_target = float(s_accept_target)
        self.s_accept_lr = float(s_accept_lr)
        self.use_crn = bool(use_crn)
        self.level_offset = int(level_offset)

        self.policy_optim = optax.adam(lr_policy)
        self.dyn_optim = optax.adam(lr_dyn)
        self.reward_optim = optax.adam(lr_reward)
        self.value_optim = optax.adam(lr_value)

        opt_state = PcOptStates(
            policy=self.policy_optim.init(params.policy),
            dynamics=self.dyn_optim.init(params.dynamics),
            reward=self.reward_optim.init(params.reward),
            value=self.value_optim.init(params.value),
        )

        mu_s = jnp.zeros((self.net.obs_dim,), dtype=jnp.float32)
        std_s = jnp.ones_like(mu_s)
        mu_a = jnp.zeros((self.net.act_dim,), dtype=jnp.float32)
        std_a = jnp.ones_like(mu_a)
        mu_r = jnp.zeros((), dtype=jnp.float32)
        std_r = jnp.ones_like(mu_r)
        std = SASRStandardizer(mu_s=mu_s, std_s=std_s, mu_a=mu_a, std_a=std_a, mu_r=mu_r, std_r=std_r)

        self.state = PcTrainState(
            params=params,
            opt_state=opt_state,
            step=jnp.int32(0),
            standardizer=std,
        )

        @jax.jit
        def stateless_update(
            key: jax.Array,
            state: PcTrainState,
            data: Experience,
        ) -> Tuple[PcTrainState, Metric]:
            params = state.params
            opt_state = state.opt_state
            step = state.step
            st = state.standardizer

            obs, action, reward, done, next_obs = (
                data.obs,
                data.action,
                data.reward,
                data.done,
                data.next_obs,
            )

            # Match DPMD’s convention of scaling rewards to keep value targets well-conditioned.
            reward = reward * self.reward_scale

            B = reward.shape[0]
            H = 1

            s0 = obs
            a_seq = action[:, None, :]
            s_seq = next_obs[:, None, :]
            r_seq = reward[:, None]
            d_seq = done.astype(jnp.float32)[:, None]
            # For fairness with other algorithms, do not maintain a separate
            # running standardizer; use raw observations/actions/rewards.
            st_new = st
            s0_std = s0
            a_seq_std = a_seq
            s_seq_std = s_seq
            r_seq_std = r_seq

            key, kt, k_a, k_s, k_r, k_v = jax.random.split(key, 6)
            # Discrete VP timesteps shared with other diffusion-based methods
            t_idx = jax.random.randint(kt, shape=(B,), minval=0, maxval=self.diffusion.num_timesteps)
            # Precompute diffusion coefficients once per update
            Bcoef = self.diffusion.beta_schedule()

            def flatten_time(x):
                return x.reshape((B * H,) + x.shape[2:])

            a0_flat = flatten_time(a_seq_std)
            s1_flat = flatten_time(s_seq_std)
            r_flat = flatten_time(r_seq_std)[..., 0]
            d_flat = flatten_time(d_seq)[..., 0]

            s0_rep = jnp.repeat(s0_std, repeats=H, axis=0)
            t_rep = jnp.repeat(t_idx, repeats=H, axis=0)

            def policy_loss_fn(policy_params: hk.Params) -> jax.Array:
                key_local = k_a
                key_local, k1 = jax.random.split(key_local)
                a0 = a_seq_std[:, 0, :]
                noise = jax.random.normal(k1, a0.shape, dtype=a0.dtype)
                # q(a_t | a0, t_idx) with per-sample timesteps
                a_t = jax.vmap(self.diffusion.q_sample)(t_idx, a0, noise)
                alpha_bar = Bcoef.alphas_cumprod[t_idx]
                sqrt_alpha = Bcoef.sqrt_alphas_cumprod[t_idx]
                denom = jnp.maximum(1.0 - alpha_bar, 1e-6)
                # Target score for VP: (a_t - sqrt(alpha_bar)*a0) / (1-alpha_bar)
                s_star = (a_t - sqrt_alpha[:, None] * a0) / denom[:, None]

                # Use the same scalar alpha_bar that defines corruption as the
                # time-conditioning input to the networks.
                t_f = alpha_bar.astype(jnp.float32)

                def single_score(s0_i, t_i, a_i):
                    def energy_single(a_vec):
                        e = self.net.policy(
                            policy_params,
                            s0_i[None, :],
                            a_vec[None, :],
                            jnp.asarray([t_i], jnp.float32),
                        )
                        return e[0]

                    return jax.grad(energy_single)(a_i)

                grad_e = jax.vmap(single_score)(s0_std, t_f, a_t)
                # Interpret gradient as epsilon prediction via VP relationship.
                # For x_t = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * eps,
                # the ideal gradient under our DSM (grad_e = -s_star) satisfies
                #   grad_e = - eps / sqrt(1-alpha_bar).
                # Thus eps_pred = -sqrt(1-alpha_bar) * grad_e should match eps.
                std = Bcoef.sqrt_one_minus_alphas_cumprod[t_idx]
                eps_pred = -std[:, None] * grad_e
                # Per-dimension MSE between predicted epsilon and true noise
                loss = jnp.mean(jnp.sum((eps_pred - noise) ** 2, axis=-1) / self.net.act_dim)
                return loss

            def dyn_loss_fn(dyn_params: hk.Params) -> jax.Array:
                key_local = k_s
                key_local, k1, k2 = jax.random.split(key_local, 3)
                noise_a = jax.random.normal(k1, a0_flat.shape, dtype=a0_flat.dtype)
                noise_s = jax.random.normal(k2, s1_flat.shape, dtype=s1_flat.dtype)
                a_t = jax.vmap(self.diffusion.q_sample)(t_rep, a0_flat, noise_a)
                s_t = jax.vmap(self.diffusion.q_sample)(t_rep, s1_flat, noise_s)
                alpha_bar_rep = Bcoef.alphas_cumprod[t_rep]
                sqrt_alpha_rep = Bcoef.sqrt_alphas_cumprod[t_rep]
                denom_rep = jnp.maximum(1.0 - alpha_bar_rep, 1e-6)
                s_star = (s_t - sqrt_alpha_rep[:, None] * s1_flat) / denom_rep[:, None]

                # As above, use alpha_bar as the network time-conditioning scalar.
                t_rep_f = alpha_bar_rep.astype(jnp.float32)

                def single_state_score(s0_i, a_i, s_i, t_i):
                    def energy_single(st_vec):
                        e = self.net.dynamics(
                            dyn_params,
                            s0_i[None, :],
                            a_i[None, :],
                            st_vec[None, :],
                            jnp.asarray([t_i], jnp.float32),
                        )
                        return e[0]

                    return jax.grad(energy_single)(s_i)

                grad_e = jax.vmap(single_state_score)(s0_rep, a_t, s_t, t_rep_f)
                # As above, interpret state-gradient as epsilon prediction for s_t.
                std_rep = Bcoef.sqrt_one_minus_alphas_cumprod[t_rep]
                eps_pred_s = -std_rep[:, None] * grad_e
                loss = jnp.mean(jnp.sum((eps_pred_s - noise_s) ** 2, axis=-1) / self.net.obs_dim)
                return loss

            def reward_loss_fn(rew_params: hk.Params) -> jax.Array:
                key_local = k_r
                key_local, k1, k2 = jax.random.split(key_local, 3)
                noise_a = jax.random.normal(k1, a0_flat.shape, dtype=a0_flat.dtype)
                noise_s = jax.random.normal(k2, s1_flat.shape, dtype=s1_flat.dtype)
                a_t = jax.vmap(self.diffusion.q_sample)(t_rep, a0_flat, noise_a)
                s_t = jax.vmap(self.diffusion.q_sample)(t_rep, s1_flat, noise_s)
                alpha_bar_rep = Bcoef.alphas_cumprod[t_rep]
                r_pred = self.net.reward(
                    rew_params,
                    s0_rep,
                    a_t,
                    s_t,
                    alpha_bar_rep.astype(jnp.float32),
                )
                return jnp.mean((r_pred - r_flat) ** 2)

            def value_loss_fn(val_params: hk.Params) -> jax.Array:
                key_local = k_v
                key_local, k1 = jax.random.split(key_local)
                noise_s = jax.random.normal(k1, s1_flat.shape, dtype=s1_flat.dtype)
                s_t = jax.vmap(self.diffusion.q_sample)(t_rep, s1_flat, noise_s)
                alpha_bar_rep = Bcoef.alphas_cumprod[t_rep]
                # Use a "clean" time code for the target network (analogous to t=0
                # in the original PCMD), and alpha_bar for the noisy inputs.
                v_next = self.net.value(
                    params.value_targ,
                    s1_flat,
                    jnp.ones_like(t_rep, dtype=jnp.float32),
                )
                lambda_td = self.td_lambda
                bootstrap_coef = 1.0 - lambda_td
                y = r_flat + self.gamma * bootstrap_coef * (1.0 - d_flat) * jax.lax.stop_gradient(v_next)
                v_pred = self.net.value(val_params, s_t, alpha_bar_rep.astype(jnp.float32))
                return jnp.mean((v_pred - y) ** 2)

            l_pol, g_pol = jax.value_and_grad(policy_loss_fn)(params.policy)
            l_dyn, g_dyn = jax.value_and_grad(dyn_loss_fn)(params.dynamics)
            l_rew, g_rew = jax.value_and_grad(reward_loss_fn)(params.reward)
            l_val, g_val = jax.value_and_grad(value_loss_fn)(params.value)

            updates_pol, new_op_pol = self.policy_optim.update(g_pol, opt_state.policy, params.policy)
            updates_dyn, new_op_dyn = self.dyn_optim.update(g_dyn, opt_state.dynamics, params.dynamics)
            updates_rew, new_op_rew = self.reward_optim.update(g_rew, opt_state.reward, params.reward)
            updates_val, new_op_val = self.value_optim.update(g_val, opt_state.value, params.value)

            new_policy = optax.apply_updates(params.policy, updates_pol)
            new_dyn = optax.apply_updates(params.dynamics, updates_dyn)
            new_rew = optax.apply_updates(params.reward, updates_rew)
            new_val = optax.apply_updates(params.value, updates_val)

            new_val_targ = optax.incremental_update(new_val, params.value_targ, step_size=self.ema_tau)

            new_params = PcParams(
                policy=new_policy,
                dynamics=new_dyn,
                reward=new_rew,
                value=new_val,
                value_targ=new_val_targ,
            )

            new_opt_state = PcOptStates(
                policy=new_op_pol,
                dynamics=new_op_dyn,
                reward=new_op_rew,
                value=new_op_val,
            )

            new_state = PcTrainState(
                params=new_params,
                opt_state=new_opt_state,
                step=step + 1,
                standardizer=st_new,
            )

            logs: Metric = {
                "loss/policy": l_pol,
                "loss/dynamics": l_dyn,
                "loss/reward": l_rew,
                "loss/value": l_val,
            }
            return new_state, logs

        def stateless_get_action(
            key: jax.Array,
            state: PcTrainState,
            obs: jax.Array,
        ) -> jax.Array:
            st = state.standardizer

            def single_action(key_single: jax.Array, obs_single: jax.Array) -> jax.Array:
                s0_std = (obs_single - st.mu_s) / jnp.maximum(st.std_s, 1e-6)
                a_init_std = jnp.zeros((self.H_plan, self.net.act_dim), dtype=jnp.float32)

                def pol_energy(s0, a, t):
                    return self.net.policy(state.params.policy, s0[None, :], a[None, :], jnp.asarray([t], jnp.float32))[0]

                def dyn_energy(s0, a, s, t):
                    return self.net.dynamics(state.params.dynamics, s0[None, :], a[None, :], s[None, :], jnp.asarray([t], jnp.float32))[0]

                def rew_fn(s0, a, s, t):
                    return self.net.reward(state.params.reward, s0[None, :], a[None, :], s[None, :], jnp.asarray([t], jnp.float32))[0]

                def val_fn(s, t):
                    return self.net.value(state.params.value, s[None, :], jnp.asarray([t], jnp.float32))[0]

                a_seq_std, *_ = jax_run_pc_mala_fac_rr_seq(
                    key_single,
                    self.levels_jax,
                    self.H_plan,
                    self.points_per_seed,
                    self.refresh_L,
                    self.action_steps_per_level,
                    self.use_ula_refresh,
                    self.cs,
                    self.cs_accept,
                    self.bprop_refresh,
                    self.accept_sprime,
                    self.adapt_s_accept,
                    self.s_accept_target,
                    self.s_accept_lr,
                    self.use_crn,
                    self.level_offset,
                    self.gamma,
                    s0_std,
                    a_init_std,
                    pol_energy,
                    dyn_energy,
                    rew_fn,
                    val_fn,
                )

                a0_std = a_seq_std[0]
                act = a0_std * jnp.maximum(st.std_a, 1e-6) + st.mu_a
                return act

            if obs.ndim == 1:
                return single_action(key, obs)
            else:
                B = obs.shape[0]
                keys = jax.random.split(key, B)
                return jax.vmap(single_action)(keys, obs)

        def stateless_get_deterministic_action(
            state: PcTrainState,
            obs: jax.Array,
        ) -> jax.Array:
            key = jax.random.key(0)
            return stateless_get_action(key, state, obs)

        self._implement_common_behavior(
            stateless_update,
            stateless_get_action,
            stateless_get_deterministic_action,
        )

    def get_policy_params(self):
        return self.state

    def get_value_params(self):
        return ()

    def save_policy(self, path: str) -> None:
        state = jax.device_get(self.state)
        with open(path, "wb") as f:
            pickle.dump(state, f)
