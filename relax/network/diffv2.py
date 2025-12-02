from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple, Union, Optional

import jax, jax.numpy as jnp
import haiku as hk
import math

from relax.network.blocks import Activation, DistributionalQNet2, DACERPolicyNet, QNet, EnergyPolicyNet
from relax.network.common import WithSquashedGaussianPolicy
from relax.utils.diffusion import GaussianDiffusion
from relax.utils.jax_utils import random_key_from_data

class Diffv2Params(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    target_poicy: hk.Params
    log_alpha: jax.Array


@dataclass
class Diffv2Net:
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    num_timesteps: int
    act_dim: int
    num_particles: int
    target_entropy: float
    noise_scale: float
    beta_schedule_scale: float
    beta_schedule_type: str = 'linear'
    energy_mode: bool = False
    energy_fn: Optional[Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]] = None
    mala_steps: int = 1

    @property
    def diffusion(self) -> GaussianDiffusion:
        return GaussianDiffusion(self.num_timesteps, 
                                 self.beta_schedule_scale,
                                 self.beta_schedule_type)

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, log_alpha, q1_params, q2_params = policy_params

        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)

        def energy_model_fn(t, x):
            return self.energy_fn(policy_params, obs, x, t)

        def sample(key: jax.Array) -> Union[jax.Array, jax.Array]:
            if self.energy_mode:
                act = self.mala_sample(key, model_fn, energy_model_fn, (*obs.shape[:-1], self.act_dim))
            else:
                act = self.diffusion.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
            q1 = self.q(q1_params, obs, act)
            q2 = self.q(q2_params, obs, act)
            q = jnp.minimum(q1, q2)
            return act.clip(-1, 1), q

        key, noise_key = jax.random.split(key)
        if self.num_particles == 1:
            act, _ = sample(key)
        else:
            keys = jax.random.split(key, self.num_particles)
            acts, qs = jax.vmap(sample)(keys)
            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
        act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale
        return act

    def mala_sample(
        self,
        key: jax.Array,
        model_fn: Callable,
        energy_fn: Callable,
        shape: Sequence[int]
    ) -> jax.Array:
        batch_shape = shape[:-1]
        act_dim = shape[-1]

        # Start from random noise
        key, x_key = jax.random.split(key)
        x = jax.random.normal(x_key, shape)

        B = self.diffusion.beta_schedule()

        def loop_body(i, args):
            x_curr, k = args
            t = self.num_timesteps - 1 - i

            # Base step size for this diffusion level, analogous to eta_base_k in pc_mala
            eta_base_t = jnp.maximum(B.betas[t], jnp.float32(1e-8))

            # Per-level adaptive log step-scale (reset each diffusion step, as in pc_mala)
            log_eta_scale0 = jnp.float32(0.0)

            def mala_body(_, state):
                x_in, k_in, log_eta_scale = state

                # Energy and its gradient
                E_x = energy_fn(t, x_in)
                grad_E_x = model_fn(t, x_in)

                # Adaptive step size: eta_k = exp(log_eta_scale) * eta_base_t
                eta_k = jnp.clip(
                    jnp.exp(log_eta_scale) * eta_base_t,
                    jnp.float32(1e-8),
                    jnp.float32(0.5),
                )

                # Proposal
                k_in, noise_key, u_key = jax.random.split(k_in, 3)
                z = jax.random.normal(noise_key, x_in.shape)

                sd = jnp.sqrt(jnp.float32(2.0) * eta_k)
                x_prop = x_in - eta_k * grad_E_x + sd * z

                # Acceptance ratio
                E_x_prop = energy_fn(t, x_prop)
                grad_E_x_prop = model_fn(t, x_prop)

                # log q(x|x') and log q(x'|x) for MALA Gaussian proposals
                mean_f = x_in - eta_k * grad_E_x
                mean_r = x_prop - eta_k * grad_E_x_prop

                def log_gauss(xv, meanv):
                    diff = xv - meanv
                    # Sum over action dimensions, keep batch dimension
                    return -jnp.sum(diff * diff, axis=-1) / (jnp.float32(4.0) * eta_k)

                log_q_prop_given_x = log_gauss(x_prop, mean_f)
                log_q_x_given_prop = log_gauss(x_in, mean_r)

                # Target: p(x) ∝ exp(-E(x))
                log_alpha = (-E_x_prop + E_x) + (log_q_x_given_prop - log_q_prop_given_x)

                u = jax.random.uniform(u_key, E_x.shape)
                accept = jnp.log(u) < jnp.minimum(jnp.float32(0.0), log_alpha)

                x_new = jnp.where(accept[..., None], x_prop, x_in)

                # Adapt log_eta_scale towards target acceptance ~0.574
                acc_rate = jnp.mean(accept.astype(jnp.float32))
                target = jnp.float32(0.574)
                adapt_rate = jnp.float32(0.05)
                log_eta_scale = log_eta_scale + adapt_rate * (acc_rate - target)

                return x_new, k_in, log_eta_scale

            # Run a small number of MALA corrector steps at level t
            x_curr, k, _ = jax.lax.fori_loop(
                0,
                self.mala_steps,
                mala_body,
                (x_curr, k, log_eta_scale0),
            )

            # DDPM-style predictor step t -> t-1 using the (possibly corrected) state
            noise_pred = model_fn(t, x_curr)
            model_mean, model_log_variance = self.diffusion.p_mean_variance(t, x_curr, noise_pred)

            k, z_key = jax.random.split(k)
            z = jax.random.normal(z_key, x_curr.shape)

            x_next = model_mean + (t > 0) * jnp.exp(0.5 * model_log_variance) * z

            return x_next, k

        x_final, _ = jax.lax.fori_loop(0, self.num_timesteps, loop_body, (x, key))
        return x_final

    def get_action_guided(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, log_alpha, q1_params, q2_params = policy_params

        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)

        def guided_sample(single_key: jax.Array) -> Tuple[jax.Array, jax.Array]:
            x_key, noise_key = jax.random.split(single_key)
            shape = (*obs.shape[:-1], self.act_dim)
            x = 0.5 * jax.random.normal(x_key, shape)
            noise = jax.random.normal(noise_key, (self.num_timesteps, *shape))

            t_seq = jnp.arange(self.num_timesteps)[::-1]
            i_seq = jnp.arange(self.num_timesteps)
            guide_idx = self.num_timesteps // 2

            def body_fn(carry, inputs):
                x_curr, x_guide = carry
                i, t, eps_t = inputs
                noise_pred = model_fn(t, x_curr)
                model_mean, model_log_variance = self.diffusion.p_mean_variance(t, x_curr, noise_pred)
                x_next = model_mean + (t > 0) * jnp.exp(0.5 * model_log_variance) * eps_t
                x_guide = jnp.where(i == guide_idx, x_curr, x_guide)
                return (x_next, x_guide), None

            (x_final, x_guide), _ = jax.lax.scan(
                body_fn,
                (x, jnp.zeros_like(x)),
                (i_seq, t_seq, noise),
            )

            act_final = x_final.clip(-1, 1)
            q1 = self.q(q1_params, obs, x_guide)
            q2 = self.q(q2_params, obs, x_guide)
            q = jnp.minimum(q1, q2)
            return act_final, q

        key, noise_key = jax.random.split(key)
        if self.num_particles == 1:
            act, _ = guided_sample(key)
        else:
            keys = jax.random.split(key, self.num_particles)
            acts, qs = jax.vmap(guided_sample)(keys)
            q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
            act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
        act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.noise_scale
        return act

    def get_batch_actions(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array, q_func: Callable) -> jax.Array:
        batch_flatten_obs = obs.repeat(self.num_particles, axis=0)
        batch_flatten_actions = self.get_action(key, policy_params, batch_flatten_obs)
        batch_q = q_func(batch_flatten_obs, batch_flatten_actions).reshape(-1, self.num_particles)
        max_q_idx = batch_q.argmax(axis=1)
        batch_action = batch_flatten_actions.reshape(obs.shape[0], -1, self.act_dim) # ?
        slice = lambda x, y: x[y]
        # action: batch_size, repeat_size, idx: batch_size
        best_action = jax.vmap(slice, (0, 0))(batch_action, max_q_idx)
        return best_action

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        key = random_key_from_data(obs)
        # Allow an optional fifth element (e.g., transition_params) in policy_params
        # and ignore it for deterministic action computation.
        if isinstance(policy_params, (tuple, list)) and len(policy_params) == 5:
            policy_params, log_alpha, q1_params, q2_params, _ = policy_params
        else:
            policy_params, log_alpha, q1_params, q2_params = policy_params
        log_alpha = -jnp.inf
        policy_params = (policy_params, log_alpha, q1_params, q2_params)
        return self.get_action(key, policy_params, obs)

    def q_evaluate(
        self, key: jax.Array, q_params: hk.Params, obs: jax.Array, act: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        q_mean, q_std = self.q(q_params, obs, act)
        z = jax.random.normal(key, q_mean.shape)
        z = jnp.clip(z, -3.0, 3.0)  # NOTE: Why not truncated normal?
        q_value = q_mean + q_std * z
        return q_mean, q_std, q_value

def create_diffv2_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    diffusion_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
    num_particles: int = 4,
    noise_scale: float = 0.05,
    target_entropy_scale: float = 0.9,
    beta_schedule_scale: float = 0.3,
    beta_schedule_type: str = "linear",
    energy_param: bool = False,
    mala_steps: int = 1,
    ) -> Tuple[Diffv2Net, Diffv2Params]:
    # q = hk.without_apply_rng(hk.transform(lambda obs, act: DistributionalQNet2(hidden_sizes, activation)(obs, act)))
    q = hk.without_apply_rng(hk.transform(lambda obs, act: QNet(hidden_sizes, activation)(obs, act)))
    
    if energy_param:
        # Energy parameterization: Network outputs scalar E(x).
        # Policy (score/noise) is grad(E(x)).
        policy_net = hk.without_apply_rng(hk.transform(lambda obs, act, t: EnergyPolicyNet(diffusion_hidden_sizes, activation)(obs, act, t)))

        # We define the external policy interface (which returns noise/score) as grad(E) w.r.t action
        def policy_score_fn(obs, act, t):
            # Note: EnergyPolicyNet outputs scalar.
            # We want grad w.r.t 'act' (arg 1)
            return jax.grad(lambda a: policy_net.apply(None, obs, a, t).sum())(act)

        # We still need to init parameters using the base network
        policy = policy_net

        # For Diffv2Net.policy, we pass the function that computes the gradient
        # Diffv2Net expects: policy(params, obs, act, t)
        policy_apply = lambda params, obs, act, t: jax.grad(lambda a: policy_net.apply(params, obs, a, t).sum())(act)
        energy_apply = policy_net.apply
        
    else:
        policy = hk.without_apply_rng(hk.transform(lambda obs, act, t: DACERPolicyNet(diffusion_hidden_sizes, activation)(obs, act, t)))
        policy_apply = policy.apply
        energy_apply = None

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, policy_key = jax.random.split(key, 3)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs, act, 0)
        target_policy_params = policy_params
        log_alpha = jnp.array(math.log(5), dtype=jnp.float32) # math.log(3) or math.log(5) choose one
        return Diffv2Params(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = Diffv2Net(
        q=q.apply,
        policy=policy_apply,
        num_timesteps=num_timesteps,
        act_dim=act_dim,
        target_entropy=-act_dim * target_entropy_scale,
        num_particles=num_particles,
        noise_scale=noise_scale,
        beta_schedule_scale=beta_schedule_scale,
        beta_schedule_type=beta_schedule_type,
        energy_mode=energy_param,
        energy_fn=energy_apply,
        mala_steps=mala_steps,
    )
    return net, params
