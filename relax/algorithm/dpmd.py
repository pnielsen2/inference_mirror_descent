from typing import NamedTuple, Tuple
import functools

import jax, jax.numpy as jnp
import numpy as np
import optax
import haiku as hk
import pickle

from relax.algorithm.base import Algorithm
from relax.network.dacer import DACERNet, DACERParams
from relax.network.diffv2 import Diffv2Net, Diffv2Params
from relax.utils.experience import Experience
from relax.utils.typing_utils import Metric
from relax.utils.jax_utils import random_key_from_data, latent_to_action_normalcdf
from relax.utils.dist_aggregation import (
    mixture_mean_var,
    aggregate_q_distributions,
    gaussian_cross_entropy,
)


@jax.custom_vjp
def mean_gaussian_cross_entropy_natgrad(
    pred_mean: jax.Array,
    pred_var: jax.Array,
    target_mean: jax.Array,
    target_var: jax.Array,
) -> jax.Array:
    return jnp.mean(gaussian_cross_entropy(pred_mean, pred_var, target_mean, target_var))


def _mean_gaussian_cross_entropy_natgrad_fwd(
    pred_mean: jax.Array,
    pred_var: jax.Array,
    target_mean: jax.Array,
    target_var: jax.Array,
):
    ce = gaussian_cross_entropy(pred_mean, pred_var, target_mean, target_var)
    return jnp.mean(ce), (pred_mean, pred_var, target_mean, target_var)


def _mean_gaussian_cross_entropy_natgrad_bwd(res, g: jax.Array):
    pred_mean, pred_var, target_mean, target_var = res
    eps = jnp.float32(1e-6)

    v = jnp.maximum(pred_var, eps)
    vt = jnp.maximum(target_var, jnp.float32(0.0))
    delta = pred_mean - target_mean

    denom = jnp.maximum(jnp.float32(delta.size), jnp.float32(1.0))

    # Full natural gradient in (mu, v=sigma^2) coordinates:
    # F^{-1} = diag(v, 2 v^2). Apply to Euclidean gradients of mean cross-entropy.
    grad_mu_nat = delta / denom
    grad_v_nat = (v - (vt + delta * delta)) / denom

    grad_mu_nat = g * grad_mu_nat
    grad_v_nat = g * grad_v_nat

    return (
        grad_mu_nat,
        grad_v_nat,
        jnp.zeros_like(target_mean),
        jnp.zeros_like(target_var),
    )


mean_gaussian_cross_entropy_natgrad.defvjp(
    _mean_gaussian_cross_entropy_natgrad_fwd,
    _mean_gaussian_cross_entropy_natgrad_bwd,
)


# =============================================================================
# DSAC-T Loss Functions (Distributional SAC with Three Refinements)
# =============================================================================
# 
# DSAC-T introduces three refinements to stabilize distributional critic training:
#
# 1. Expected Value Substitution: Use the expected Q-value (mean) instead of 
#    sampled values for the mean-related gradient term. This reduces gradient
#    variance by treating the target as a point mass.
#
# 2. Adaptive Clipping (Eq. 23): Clip the TD error in the variance-related gradient by
#    b = xi * b_ema, where b_ema is the EMA of batch-mean sigma, and xi is typically 3 
#    (three-sigma rule). This prevents gradient explosion when TD errors are large.
#
# 3. Omega Scaling (Eq. 24-27): The loss is conceptually scaled by omega = sigma^2,
#    which CANCELS the per-sample 1/sigma^2 in the mean gradient. After normalizing
#    by (omega_ema + eps), the gradients become:
#    - Mean: delta / (omega_ema + eps)  [NO per-sample sigma^2!]
#    - Var: 0.5 * (1 - (target_var + delta^2) / sigma^2) / (omega_ema + eps)
#
# Reference: "Distributional Soft Actor-Critic with Three Refinements" (DSAC-T)
# https://arxiv.org/abs/2310.05858

def dsac_t_loss(
    pred_mean: jax.Array,
    pred_var: jax.Array,
    target_mean: jax.Array,
    target_var: jax.Array,
    *,
    expected_value_sub: bool = False,
    adaptive_clip_xi: float = 0.0,
    omega_scaling: bool = False,
    omega_ema: float = 1.0,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """
    Compute DSAC-T loss with all three refinements.
    
    Args:
        pred_mean: Predicted Q mean [batch]
        pred_var: Predicted Q variance [batch]
        target_mean: Target Q mean (r + gamma * Q') [batch]
        target_var: Target Q variance (gamma^2 * var(Q')) [batch]
        expected_value_sub: If True, use target_var=0 for mean gradient (Refinement 1)
        adaptive_clip_xi: Clipping factor xi; if > 0, clip TD by xi*sigma (Refinement 2)
        omega_scaling: If True, scale loss by omega/omega_ema (Refinement 3)
        omega_ema: Moving average of batch-mean variance for omega scaling
    
    Returns:
        (loss, omega, b): loss scalar, current omega for EMA update, current b for EMA update
    """
    eps = jnp.float32(1e-6)
    pred_var = jnp.maximum(pred_var, eps)
    pred_std = jnp.sqrt(pred_var)
    
    # TD error (delta in the paper)
    td_error = target_mean - pred_mean
    
    # Effective target variance for loss computation
    # Refinement 1: Expected value substitution sets target_var=0
    effective_target_var = jax.lax.cond(
        jnp.bool_(expected_value_sub),
        lambda: jnp.zeros_like(target_var),
        lambda: jnp.maximum(target_var, jnp.float32(0.0)),
    )
    
    # Refinement 2: Adaptive clipping of TD error for variance gradient
    # b = xi * sigma; clip td_error to [-b, b] for variance term
    use_adaptive_clip = adaptive_clip_xi > 0.0
    b = adaptive_clip_xi * pred_std
    td_error_clipped = jax.lax.cond(
        jnp.bool_(use_adaptive_clip),
        lambda: jnp.clip(td_error, -b, b),
        lambda: td_error,
    )
    
    # Gaussian cross-entropy loss
    # CE = 0.5 * log(2*pi*var) + (target_var + td_error^2) / (2*var)
    # For DSAC-T, we use td_error_clipped^2 in the variance-related term
    log_term = jnp.float32(0.5) * jnp.log(jnp.float32(2.0) * jnp.pi * pred_var)
    
    # Use clipped TD error for the squared term (affects variance gradient)
    # But mean gradient still uses original TD error through the cross-entropy formula
    mse_term = (effective_target_var + td_error_clipped ** 2) / (jnp.float32(2.0) * pred_var)
    
    per_sample_loss = log_term + mse_term
    
    # Refinement 3: Omega scaling
    # omega = mean(pred_var), scale loss by omega / (omega_ema + eps_omega)
    omega = jnp.mean(pred_var)
    eps_omega = jnp.float32(1e-6)
    
    if omega_scaling:
        scale_factor = omega / (omega_ema + eps_omega)
        per_sample_loss = per_sample_loss * scale_factor
    
    loss = jnp.mean(per_sample_loss)
    
    # Return b as mean for EMA tracking (used for logging/adaptive clipping)
    b_mean = jnp.mean(pred_std) if use_adaptive_clip else jnp.float32(1.0)
    
    return loss, omega, b_mean


@jax.custom_vjp
def dsac_t_loss_with_custom_grad(
    pred_mean: jax.Array,
    pred_var: jax.Array,
    target_mean: jax.Array,
    target_var: jax.Array,
    adaptive_clip_xi: jax.Array,
    b_ema: jax.Array,
    omega_ema: jax.Array,
    use_omega_scaling: jax.Array,
) -> jax.Array:
    """
    DSAC-T loss with custom gradients implementing all three refinements.
    
    This uses custom VJP to properly implement:
    1. Expected value substitution: mean gradient uses expected TD error (automatic)
    2. Adaptive clipping: clip TD error by ξ * b_ema for variance gradient (Eq. 23)
    3. Omega scaling: changes gradient formula per Eq. 27
    
    Args:
        adaptive_clip_xi: Scalar, clipping factor ξ (0 = no clipping, typically 3.0)
        b_ema: Scalar, EMA of batch-mean σ (used for clipping bound b = ξ * b_ema)
        omega_ema: Scalar, EMA of batch-mean σ² (used for gradient normalization)
        use_omega_scaling: Scalar bool, whether to use omega scaling
    
    Reference: DSAC-T paper Eq. 23-27
    """
    eps = jnp.float32(1e-6)
    v = jnp.maximum(pred_var, eps)
    vt = jnp.maximum(target_var, jnp.float32(0.0))
    
    ce = (
        jnp.float32(0.5) * jnp.log(jnp.float32(2.0) * jnp.pi * v)
        + (vt + (target_mean - pred_mean) ** 2) / (jnp.float32(2.0) * v)
    )
    return jnp.mean(ce)


def _dsac_t_loss_fwd(
    pred_mean: jax.Array,
    pred_var: jax.Array,
    target_mean: jax.Array,
    target_var: jax.Array,
    adaptive_clip_xi: jax.Array,
    b_ema: jax.Array,
    omega_ema: jax.Array,
    use_omega_scaling: jax.Array,
):
    eps = jnp.float32(1e-6)
    v = jnp.maximum(pred_var, eps)
    vt = jnp.maximum(target_var, jnp.float32(0.0))
    delta = pred_mean - target_mean
    
    ce = (
        jnp.float32(0.5) * jnp.log(jnp.float32(2.0) * jnp.pi * v)
        + (vt + delta ** 2) / (jnp.float32(2.0) * v)
    )
    return jnp.mean(ce), (pred_mean, pred_var, target_mean, target_var, adaptive_clip_xi, b_ema, omega_ema, use_omega_scaling)


def _dsac_t_loss_bwd(res, g: jax.Array):
    """
    Custom backward pass implementing DSAC-T gradient refinements.
    
    From DSAC-T paper Section 4.3:
    
    The scaled objective is L_scaled = ω * L where ω = σ²(s,a).
    This causes the per-sample σ² to CANCEL in the mean gradient.
    After normalizing by (ω̄ + ε), the gradients become:
    
    - Mean gradient (Eq. 27): δ / (ω̄ + ε)  [NO per-sample σ² in denominator!]
    - Var gradient (Eq. 27): 0.5 * (1 - (σ_t² + δ_clipped²)/σ²) / (ω̄ + ε)
    
    Refinements:
    1. Expected value substitution: Mean gradient uses δ (TD error with expected target)
    2. Adaptive clipping (Eq. 23): b = ξ * σ̄ where σ̄ is b_ema (EMA of batch-mean σ)
    3. Omega scaling (Eq. 24-27): Gradient formula changes as described above
    """
    pred_mean, pred_var, target_mean, target_var, adaptive_clip_xi, b_ema, omega_ema, use_omega_scaling = res
    eps = jnp.float32(1e-6)
    eps_omega = jnp.float32(1e-6)
    
    v = jnp.maximum(pred_var, eps)
    vt = jnp.maximum(target_var, jnp.float32(0.0))
    delta = pred_mean - target_mean  # TD error
    
    n = jnp.maximum(jnp.float32(delta.size), jnp.float32(1.0))
    
    # Refinement 2: Adaptive clipping for variance gradient (Eq. 23)
    # Clipping bound b = ξ * σ̄ where σ̄ is b_ema (EMA of batch-mean σ)
    # NOT per-sample σ! The paper uses moving average for stability.
    use_clip = adaptive_clip_xi > jnp.float32(0.0)
    b = adaptive_clip_xi * b_ema  # Use b_ema, not per-sample sigma!
    delta_clipped = jnp.where(use_clip, jnp.clip(delta, -b, b), delta)
    
    # Compute gradients based on whether omega scaling is enabled
    # 
    # With omega scaling (Eq. 27):
    #   The loss is L_scaled = ω * CE where ω = σ².
    #   Taking gradient and normalizing by (ω̄ + ε):
    #   - Mean grad: σ² * (δ/σ²) / (ω̄ + ε) = δ / (ω̄ + ε)  [σ² cancels!]
    #   - Var grad: σ² * 0.5*(1/σ² - (σ_t²+δ²)/σ⁴) / (ω̄ + ε)
    #             = 0.5 * (1 - (σ_t²+δ²)/σ²) / (ω̄ + ε)
    #
    # Without omega scaling (standard CE gradient):
    #   - Mean grad: δ / σ²
    #   - Var grad: 0.5 * (1/σ² - (σ_t²+δ²)/σ⁴)
    
    omega_denom = omega_ema + eps_omega
    
    # Mean gradient
    grad_mu_omega = delta / omega_denom / n  # Omega scaling: δ / (ω̄ + ε)
    grad_mu_std = delta / v / n              # Standard: δ / σ²
    grad_mu = jnp.where(use_omega_scaling, grad_mu_omega, grad_mu_std)
    
    # Variance gradient (use clipped delta)
    # With omega scaling: 0.5 * (1 - (σ_t² + δ²)/σ²) / (ω̄ + ε)
    grad_v_omega = jnp.float32(0.5) * (jnp.float32(1.0) - (vt + delta_clipped ** 2) / v) / omega_denom / n
    # Standard: 0.5 * (1/σ² - (σ_t² + δ²)/σ⁴)
    grad_v_std = jnp.float32(0.5) * (jnp.reciprocal(v) - (vt + delta_clipped ** 2) / (v ** 2)) / n
    grad_v = jnp.where(use_omega_scaling, grad_v_omega, grad_v_std)
    
    # Apply upstream gradient
    grad_mu = g * grad_mu
    grad_v = g * grad_v
    
    return (
        grad_mu,
        grad_v,
        jnp.zeros_like(target_mean),
        jnp.zeros_like(target_var),
        jnp.zeros_like(adaptive_clip_xi),
        jnp.zeros_like(b_ema),
        jnp.zeros_like(omega_ema),
        jnp.zeros_like(use_omega_scaling),
    )


dsac_t_loss_with_custom_grad.defvjp(_dsac_t_loss_fwd, _dsac_t_loss_bwd)


class Diffv2OptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState
    value: optax.OptState = None  # Optional V(s) network for normalized advantage guidance


class Diffv2TrainState(NamedTuple):
    params: Diffv2Params
    opt_state: Diffv2OptStates
    step: int
    entropy: float
    running_mean: float
    running_std: float
    log_eta_scales: jax.Array
    tfg_lambda: jax.Array
    # DSAC-T state: moving averages for omega scaling and adaptive clipping
    dsac_omega_ema: float = 1.0  # EMA of batch-mean predicted variance
    dsac_b_ema: float = 1.0      # EMA of batch-mean predicted std (for adaptive clipping)
    # Normalized advantage guidance state
    value_params: hk.Params = None  # V(s) network params (optional)
    advantage_second_moment_ema: float = 1.0  # EMA of E[A^2] where A = Q - V

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
        alpha_lr: float = 3e-2,
        lr_schedule_end: float = 5e-5,
        tau: float = 0.005,
        delay_alpha_update: int = 250,
        delay_update: int = 2,
        reward_scale: float = 0.2,
        td_actions: int = 1,
        num_samples: int = 200,
        use_ema: bool = True,
        use_reweighting: bool = True,
        use_reward_critic: bool = False,
        pure_bc_training: bool = False,
        off_policy_td: bool = False,
        no_entropy_tuning: bool = False,
        q_critic_agg: str = "min",
        entropic_risk_beta: float = 1.0,
        fix_q_norm_bug: bool = False,
        tfg_lambda: float = 0.0,
        tfg_lambda_schedule: str = "constant",
        tfg_recur_steps: int = 0,
        particle_selection_lambda: float = np.inf,
        x0_hat_clip_radius: float = 1.0,
        supervised_steps: int = 1,
        single_q_network: bool = False,
        lr_schedule_steps: int = int(5e4),
        lr_schedule_begin: int = int(2.5e4),
        mala_per_level_eta: bool = False,
        mala_adapt_rate: float = 0.05,
        mala_init_eta_scale: float = 1.0,
        mala_guided_predictor: bool = False,
        mala_predictor_first: bool = False,
        ddim_predictor: bool = False,
        latent_action_space: bool = False,
        q_td_huber_width: float = float("inf"),
        decorrelated_q_batches: bool = False,
        q_bootstrap_agg: str = "min",
        langevin_q_noise: bool = False,
        critic_grad_modifier: str = "none",
        natural_gradient_critic: bool = False,
        # DSAC-T refinements
        dsac_expected_value_sub: bool = False,
        dsac_adaptive_clip_xi: float = 0.0,
        dsac_omega_scaling: bool = False,
        dsac_omega_tau: float = 0.005,
        # Energy/score scaling for exploration
        energy_multiplier: float = 1.0,
        # Critic normalization for guidance
        critic_normalization: str = "none",  # "none", "ema", "distributional"
        advantage_ema_tau: float = 0.005,
        # Policy loss type
        policy_loss_type: str = "eps_mse",  # "eps_mse" or "ula_kl"
    ):
        self.agent = agent
        timesteps = self.agent.num_timesteps
        self.gamma = gamma
        self.tau = tau
        self.delay_alpha_update = delay_alpha_update
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.td_actions = int(td_actions)
        self.num_samples = num_samples

        lr_policy_eff = float(lr if lr_policy is None else lr_policy)
        lr_q_eff = float(lr if lr_q is None else lr_q)

        self.lr_policy = lr_policy_eff
        self.lr_q = lr_q_eff

        self.optim = optax.adam(lr_q_eff)
        self.latent_action_space = bool(latent_action_space)
        self.q_td_huber_width = float(q_td_huber_width)
        self.decorrelated_q_batches = bool(decorrelated_q_batches)
        self.q_bootstrap_agg = str(q_bootstrap_agg)
        self.langevin_q_noise = bool(langevin_q_noise)

        critic_grad_modifier = str(critic_grad_modifier)
        if bool(natural_gradient_critic) and critic_grad_modifier == "none":
            critic_grad_modifier = "natgrad"
        if critic_grad_modifier not in {"none", "variance_scaled", "natgrad"}:
            raise ValueError(f"Unknown critic_grad_modifier: {critic_grad_modifier}")
        self.critic_grad_modifier = critic_grad_modifier

        init_eta_scale = jnp.maximum(jnp.float32(mala_init_eta_scale), jnp.float32(1e-8))
        init_log_eta_scale = jnp.log(init_eta_scale)
        if lr_schedule_steps > 0:
            lr_schedule = optax.schedules.linear_schedule(
                init_value=lr_policy_eff,
                end_value=lr_schedule_end,
                transition_steps=int(lr_schedule_steps),
                transition_begin=int(lr_schedule_begin),
            )
            self.policy_optim = optax.adam(learning_rate=lr_schedule)
        else:
            # Constant LR (no annealing)
            self.policy_optim = optax.adam(learning_rate=lr_policy_eff)
        self.alpha_optim = optax.adam(alpha_lr)
        self.entropy = 0.0

        # Initialize V network for critic normalization (uses same lr as Q)
        value_params_init = None
        value_opt_state_init = None
        critic_normalization = str(critic_normalization)
        if critic_normalization in ("ema", "distributional"):
            # V(s) network: same architecture as Q but takes only obs
            # Infer obs_dim and hidden_sizes from Q network params
            from relax.network.blocks import ValueNet, DistributionalValueNet
            # Q network input is (obs, act), so first layer input_dim = obs_dim + act_dim
            q1_first_layer_key = next(k for k in params.q1.keys() if 'linear' in k.lower() or 'w' in str(params.q1[k]))
            first_layer = params.q1[q1_first_layer_key] if isinstance(params.q1[q1_first_layer_key], dict) else params.q1
            # Find the first weight matrix to get input dim
            def find_first_weight(d, depth=0):
                if depth > 5:
                    return None
                if isinstance(d, dict):
                    for k, v in d.items():
                        if 'w' in k.lower():
                            return v
                        result = find_first_weight(v, depth + 1)
                        if result is not None:
                            return result
                return None
            first_w = find_first_weight(params.q1)
            if first_w is not None:
                input_dim = first_w.shape[0]
                obs_dim_inferred = input_dim - agent.act_dim
                hidden_dim = first_w.shape[1]
            else:
                # Fallback: use reasonable defaults
                obs_dim_inferred = 17  # Common MuJoCo obs dim
                hidden_dim = 256
            
            if critic_normalization == "ema":
                # Standard value network outputting single scalar
                value_net = hk.without_apply_rng(
                    hk.transform(lambda obs: ValueNet(
                        hidden_sizes=(hidden_dim, hidden_dim, hidden_dim),
                        activation=jax.nn.relu,
                    )(obs))
                )
            else:  # distributional
                # Distributional value network outputting (mean, log_var)
                value_net = hk.without_apply_rng(
                    hk.transform(lambda obs: DistributionalValueNet(
                        hidden_sizes=(hidden_dim, hidden_dim, hidden_dim),
                        activation=jax.nn.relu,
                    )(obs))
                )
            sample_obs = jnp.zeros((1, obs_dim_inferred))
            value_params_init = value_net.init(jax.random.PRNGKey(42), sample_obs)
            value_opt_state_init = self.optim.init(value_params_init)
            self._value_net = value_net
            self._obs_dim = obs_dim_inferred
        else:
            self._value_net = None
            self._obs_dim = None

        self.state = Diffv2TrainState(
            params=params,
            opt_state=Diffv2OptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                # policy=self.optim.init(params.policy),
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
                value=value_opt_state_init,
            ),
            step=jnp.int32(0),
            entropy=jnp.float32(0.0),
            running_mean=jnp.float32(0.0),
            running_std=jnp.float32(1.0),
            log_eta_scales=jnp.full((timesteps,), init_log_eta_scale, dtype=jnp.float32),
            tfg_lambda=jnp.float32(tfg_lambda),
            value_params=value_params_init,
            advantage_second_moment_ema=jnp.float32(1.0),
        )
        self.use_ema = use_ema
        self.use_reweighting = use_reweighting
        self.use_reward_critic = use_reward_critic
        self.pure_bc_training = pure_bc_training
        self.off_policy_td = off_policy_td
        self.no_entropy_tuning = no_entropy_tuning
        self.q_critic_agg = q_critic_agg
        self.entropic_risk_beta = float(entropic_risk_beta)
        self.fix_q_norm_bug = fix_q_norm_bug
        self.tfg_lambda = float(tfg_lambda)
        self.tfg_lambda_schedule = tfg_lambda_schedule
        self.tfg_recur_steps = tfg_recur_steps
        self.particle_selection_lambda = particle_selection_lambda
        self.x0_hat_clip_radius = float(x0_hat_clip_radius)
        self.supervised_steps = int(supervised_steps)
        self.single_q_network = single_q_network
        self.mala_per_level_eta = bool(mala_per_level_eta)
        self.mala_adapt_rate = float(mala_adapt_rate)
        self.mala_init_eta_scale = float(mala_init_eta_scale)
        self.mala_guided_predictor = bool(mala_guided_predictor)
        self.mala_predictor_first = bool(mala_predictor_first)
        self.ddim_predictor = bool(ddim_predictor)

        # DSAC-T refinements
        self.dsac_expected_value_sub = bool(dsac_expected_value_sub)
        self.dsac_adaptive_clip_xi = float(dsac_adaptive_clip_xi)
        self.dsac_omega_scaling = bool(dsac_omega_scaling)
        self.dsac_omega_tau = float(dsac_omega_tau)

        # Energy/score scaling: multiplies base energy/score (not guidance) to temper the base distribution
        self.energy_multiplier = float(energy_multiplier)

        # Critic normalization: use (Q - V) / std instead of Q for guidance
        # "none": raw Q, "ema": global EMA of A², "distributional": per-state V variance
        self.critic_normalization = str(critic_normalization)
        self.advantage_ema_tau = float(advantage_ema_tau)

        # Policy loss type: "eps_mse" (standard) or "ula_kl" (ULA-step KL weighting)
        self.policy_loss_type = str(policy_loss_type)

        if self.tfg_lambda_schedule == "linear":
            idx = jnp.arange(timesteps, dtype=jnp.float32)
            denom = jnp.maximum(timesteps - 1, 1)
            t_levels = 1.0 - idx / denom

            def lambda_for_step(t_idx: jax.Array, tfg_lambda_current: jax.Array) -> jax.Array:
                t_next = jnp.maximum(t_idx - 1, 0)
                return tfg_lambda_current * t_levels[t_next]

        elif self.tfg_lambda_schedule == "snr":
            # lambda_t = lambda * alpha_bar_t  =  lambda * SNR/(1+SNR)
            # Gives c_t = lambda*(1 - alpha_bar_t), bounded in [0, lambda].
            # Hessian-term loss contribution is ~constant across noise levels.
            B_init = self.agent.diffusion.beta_schedule()
            snr_scales = B_init.alphas_cumprod  # [T], ~1 at t=0 (clean), ~0 at t=T-1 (noisy)

            def lambda_for_step(t_idx: jax.Array, tfg_lambda_current: jax.Array) -> jax.Array:
                t_next = jnp.maximum(t_idx - 1, 0)
                return tfg_lambda_current * snr_scales[t_next]

        else:
            ones = jnp.ones((timesteps,), dtype=jnp.float32)

            def lambda_for_step(t_idx: jax.Array, tfg_lambda_current: jax.Array) -> jax.Array:
                t_next = jnp.maximum(t_idx - 1, 0)
                return tfg_lambda_current * ones[t_next]

        # Aggregation of twin Qs for *signals* (reweighting, tilt, logging, etc.).
        # For distributional critics, Q returns (mean, var). We aggregate means for signals.
        # TD targets use q_bootstrap_agg for distributional aggregation.

        def hard_min_q_mean(q1_mean: jax.Array, q2_mean: jax.Array) -> jax.Array:
            """Min aggregation on means only (for action selection)."""
            return jnp.minimum(q1_mean, q2_mean)

        def precision_weighted_q_mean(
            q1_mean: jax.Array,
            q1_var: jax.Array,
            q2_mean: jax.Array,
            q2_var: jax.Array,
        ) -> jax.Array:
            eps = jnp.float32(1e-6)
            w1 = jnp.reciprocal(jax.lax.stop_gradient(q1_var) + eps)
            w2 = jnp.reciprocal(jax.lax.stop_gradient(q2_var) + eps)
            denom = w1 + w2
            return (w1 * q1_mean + w2 * q2_mean) / denom

        if q_critic_agg == "min":
            def aggregate_q_mean(q1_mean: jax.Array, q2_mean: jax.Array) -> jax.Array:
                return hard_min_q_mean(q1_mean, q2_mean)
        elif q_critic_agg == "mean":
            def aggregate_q_mean(q1_mean: jax.Array, q2_mean: jax.Array) -> jax.Array:
                return 0.5 * (q1_mean + q2_mean)
        elif q_critic_agg == "max":
            def aggregate_q_mean(q1_mean: jax.Array, q2_mean: jax.Array) -> jax.Array:
                return jnp.maximum(q1_mean, q2_mean)
        elif q_critic_agg == "random":
            # For random aggregation we pick either q1 or q2 uniformly at random.
            # The actual sampling of which critic to use happens inside action sampling
            # (per particle) and inside update steps (per update), so this is only a
            # safe fallback if used directly.
            def aggregate_q_mean(q1_mean: jax.Array, q2_mean: jax.Array) -> jax.Array:
                return hard_min_q_mean(q1_mean, q2_mean)
        elif q_critic_agg == "entropic":
            # Entropic risk: (1/β)*log(mean(exp(β*Q))) = (1/β)*(logsumexp(β*Q1, β*Q2) - log(2))
            # β>0: risk-seeking/optimistic (β→∞ gives max).
            # β→0: risk-neutral (mean).
            # β<0: risk-averse/pessimistic (β→-∞ gives min).
            _ent_beta = jnp.float32(self.entropic_risk_beta)
            _use_mean_fallback = jnp.abs(_ent_beta) < 1e-6
            def aggregate_q_mean(q1_mean: jax.Array, q2_mean: jax.Array) -> jax.Array:
                mean_val = jnp.float32(0.5) * (q1_mean + q2_mean)
                stacked = jnp.stack([_ent_beta * q1_mean, _ent_beta * q2_mean], axis=0)
                ent_val = (jnp.float32(1.0) / _ent_beta) * (jax.scipy.special.logsumexp(stacked, axis=0) - jnp.log(2.0))
                return jnp.where(_use_mean_fallback, mean_val, ent_val)
        elif q_critic_agg == "precision":
            # Precision-weighted mean: weight each critic by inverse predicted variance.
            # Only meaningful for distributional critics; for point critics (var=0), this
            # reduces to (near) mean aggregation.
            def aggregate_q_mean(q1_mean: jax.Array, q2_mean: jax.Array) -> jax.Array:
                return hard_min_q_mean(q1_mean, q2_mean)
        else:
            raise ValueError(
                f"Invalid q_critic_agg: {q_critic_agg}. Expected 'min', 'mean', 'max', 'random', 'entropic', or 'precision'."
            )
        
        def aggregate_q(q1_mean: jax.Array, q1_var: jax.Array, q2_mean: jax.Array, q2_var: jax.Array) -> jax.Array:
            """Aggregate Q values for action selection and signals."""
            if q_critic_agg == "precision":
                return precision_weighted_q_mean(q1_mean, q1_var, q2_mean, q2_var)
            return aggregate_q_mean(q1_mean, q2_mean)

        def hard_min_q(
            q1_mean: jax.Array,
            q2_or_q1_var: jax.Array,
            q2_mean: jax.Array | None = None,
            _q2_var: jax.Array | None = None,
        ) -> jax.Array:
            """Min aggregation that is compatible with both (q1_mean, q2_mean) and (q1_mean, q1_var, q2_mean, q2_var)."""
            q2_mean_eff = q2_or_q1_var if q2_mean is None else q2_mean
            return jnp.minimum(q1_mean, q2_mean_eff)

        if np.isinf(particle_selection_lambda):
            def select_action_from_particles(acts: jax.Array, qs: jax.Array, key: jax.Array) -> jax.Array:
                q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
                act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
                return act
        else:
            lambda_sel = jnp.array(particle_selection_lambda, dtype=jnp.float32)

            def select_action_from_particles(acts: jax.Array, qs: jax.Array, key: jax.Array) -> jax.Array:
                logits = lambda_sel * qs
                logits = logits - jnp.max(logits, axis=0, keepdims=True)
                idx = jax.random.categorical(key, logits, axis=0)
                idx = idx[None, :]
                act = jnp.take_along_axis(acts, idx[..., None], axis=0).squeeze(axis=0)
                return act

        def sample_with_particles(
            key: jax.Array,
            log_alpha: jax.Array,
            single: bool,
            single_sampler,
            log_eta_scales_in: jax.Array,
        ) -> jax.Array:
            key_sample, key_select, noise_key = jax.random.split(key, 3)
            if self.agent.num_particles == 1:
                act, _, log_eta_scales_out = single_sampler(key_sample, log_eta_scales_in)
            else:
                keys = jax.random.split(key_sample, self.agent.num_particles)
                acts, qs, log_eta_scales_outs = jax.vmap(lambda k: single_sampler(k, log_eta_scales_in))(keys)
                act = select_action_from_particles(acts, qs, key_select)
                log_eta_scales_out = jnp.mean(log_eta_scales_outs, axis=0)

            if not self.no_entropy_tuning:
                act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.agent.noise_scale

            if single:
                return act[0], log_eta_scales_out
            else:
                return act, log_eta_scales_out

        def sample_action_with_agg(
            key: jax.Array,
            params,
            obs: jax.Array,
            aggregate_q_fn,
            randomize_q: bool = False,
        ) -> jax.Array:
            policy_params, log_alpha, q1_params, q2_params, log_eta_scales_in, tfg_lambda_current, value_params, adv_second_moment_ema = params
            if self.agent.energy_mode and self.agent.mala_steps > 0:
                return stateless_get_action_mala(key, params, obs, aggregate_q_fn, randomize_q=randomize_q)
            elif self.tfg_recur_steps > 0:
                act, log_eta_scales_out = stateless_get_action_tfg_recur(key, params, obs, aggregate_q_fn, randomize_q=randomize_q)
                return act, log_eta_scales_out
            else:
                def _do_tfg(_):
                    return stateless_get_action_tfg_recur(key, params, obs, aggregate_q_fn, randomize_q=randomize_q)

                def _do_base(_):
                    return stateless_get_action_base(key, params, obs, aggregate_q_fn, randomize_q=randomize_q)

                return jax.lax.cond(
                    tfg_lambda_current != jnp.float32(0.0),
                    _do_tfg,
                    _do_base,
                    operand=None,
                )

        @jax.jit
        def stateless_update(
            key: jax.Array, state: Diffv2TrainState, data: Experience
        ) -> Tuple[Diffv2OptStates, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            next_action_buffer = data.next_action  # SARSA-style: action taken at next_obs (may be None)
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state, _value_opt_state = state.opt_state
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std
            log_eta_scales = state.log_eta_scales
            tfg_lambda_current = state.tfg_lambda
            dsac_omega_ema = state.dsac_omega_ema
            dsac_b_ema = state.dsac_b_ema
            next_eval_key, new_eval_key, new_q1_eval_key, new_q2_eval_key, log_alpha_key, diffusion_time_key, diffusion_noise_key, q_agg_key, q2_shuffle_key, q1_langevin_key, q2_langevin_key = jax.random.split(
                key, 11)

            if self.q_critic_agg == "random":
                critic_idx = jax.random.randint(q_agg_key, (), 0, 2)

                def aggregate_q_signal(q1_mean: jax.Array, q1_var: jax.Array, q2_mean: jax.Array, q2_var: jax.Array) -> jax.Array:
                    return jax.lax.select(critic_idx == 0, q1_mean, q2_mean)

            else:

                def aggregate_q_signal(q1_mean: jax.Array, q1_var: jax.Array, q2_mean: jax.Array, q2_var: jax.Array) -> jax.Array:
                    return aggregate_q(q1_mean, q1_var, q2_mean, q2_var)

            reward *= self.reward_scale

            def get_min_q(s, a):
                """Get aggregated Q value (mean only) for signals like reweighting."""
                q1_mean, q1_var = self.agent.q(q1_params, s, a)
                q2_mean, q2_var = self.agent.q(q2_params, s, a)
                q_mean = aggregate_q_signal(q1_mean, q1_var, q2_mean, q2_var)
                if self.use_reward_critic:
                    q_mean = q_mean * (jnp.float32(1.0) / jnp.float32(1.0 - self.gamma))
                return q_mean

            def get_min_target_q(s, a):
                """Get min target Q mean for action selection."""
                q1_mean, q1_var = self.agent.q(target_q1_params, s, a)
                q2_mean, q2_var = self.agent.q(target_q2_params, s, a)
                q_mean = jnp.minimum(q1_mean, q2_mean)
                return q_mean
            
            def get_target_q_dist(s, a):
                """Get both Q distributions from target networks."""
                q1_mean, q1_var = self.agent.q(target_q1_params, s, a)
                q2_mean, q2_var = self.agent.q(target_q2_params, s, a)
                return q1_mean, q1_var, q2_mean, q2_var

            reward_loss = jnp.float32(0.0)

            mala_acc_rate = jnp.float32(jnp.nan)
            mala_eta_scale = jnp.float32(jnp.nan)

            mala_eta_scale_in = jnp.float32(jnp.nan)
            mala_eta_scale_out = jnp.float32(jnp.nan)

            q_backup_mean = jnp.float32(jnp.nan)
            q_backup_std = jnp.float32(jnp.nan)
            q_backup_min = jnp.float32(jnp.nan)
            q_backup_max = jnp.float32(jnp.nan)
            q_target_mean = jnp.float32(jnp.nan)
            q_target_std = jnp.float32(jnp.nan)
            reward_mean = jnp.float32(jnp.nan)
            done_frac = jnp.float32(jnp.nan)

            q1_pred_mean = jnp.float32(jnp.nan)
            q1_pred_std = jnp.float32(jnp.nan)
            q2_pred_mean = jnp.float32(jnp.nan)
            q2_pred_std = jnp.float32(jnp.nan)
            td_error1_std = jnp.float32(jnp.nan)
            td_error2_std = jnp.float32(jnp.nan)

            mean_abs_action = jnp.float32(jnp.nan)
            std_action = jnp.float32(jnp.nan)
            clip_frac = jnp.float32(jnp.nan)

            if not self.use_reward_critic:
                if self.off_policy_td:
                    # SARSA-style off-policy: use the action actually taken at next_obs
                    # (stored as next_action in buffer via track_next_action)
                    next_action = next_action_buffer
                    q1_target_mean, q1_target_var = self.agent.q(target_q1_params, next_obs, next_action)
                    q2_target_mean, q2_target_var = self.agent.q(target_q2_params, next_obs, next_action)
                    q_target_q1 = q1_target_mean
                    q_target_q2 = q2_target_mean
                    q_target_q1_var = q1_target_var
                    q_target_q2_var = q2_target_var
                    q_target = jnp.minimum(q_target_q1, q_target_q2)
                    q_target_min = q_target
                else:
                    # On-policy: sample fresh actions from current policy
                    td_params = (policy_params, log_alpha, q1_params, q2_params, log_eta_scales, tfg_lambda_current, state.value_params, state.advantage_second_moment_ema)
                    use_multi_td = jnp.bool_(self.td_actions > 1)

                    if (not self.mala_per_level_eta) and self.agent.energy_mode and self.agent.mala_steps > 0:
                        mala_eta_scale_in = jnp.exp(log_eta_scales[0])

                    def sample_next_action_and_stats(k: jax.Array, log_eta_scales_in: jax.Array):
                        act, log_eta_scales_out, mala_acc_rate_out, mala_eta_scale_out = sample_action_with_agg_metrics(
                            k,
                            (policy_params, log_alpha, q1_params, q2_params, log_eta_scales_in, tfg_lambda_current, state.value_params, state.advantage_second_moment_ema),
                            next_obs,
                            hard_min_q,
                        )
                        return act, log_eta_scales_out, mala_acc_rate_out, mala_eta_scale_out

                    def sample_single_td():
                        return sample_next_action_and_stats(next_eval_key, log_eta_scales)

                    def sample_multi_td():
                        batch = next_obs.shape[0]
                        next_obs_rep = jnp.tile(next_obs, (int(self.td_actions), 1))
                        act_sel_flat, acts_particles, probs_particles, log_eta_scales_out, mala_acc_rate_out, mala_eta_scale_out = sample_action_with_particle_details_metrics(
                            next_eval_key,
                            (policy_params, log_alpha, q1_params, q2_params, log_eta_scales, tfg_lambda_current, state.value_params, state.advantage_second_moment_ema),
                            next_obs_rep,
                            hard_min_q,
                        )
                        acts_sel = act_sel_flat.reshape((int(self.td_actions), batch, self.agent.act_dim))
                        # Get Q distributions for each particle
                        q1_means, q1_vars = jax.vmap(lambda a: self.agent.q(target_q1_params, next_obs_rep, a), in_axes=0)(acts_particles)
                        q2_means, q2_vars = jax.vmap(lambda a: self.agent.q(target_q2_params, next_obs_rep, a), in_axes=0)(acts_particles)
                        
                        # Aggregate across particles using mixture distribution
                        q1_mix_mean, q1_mix_var = mixture_mean_var(q1_means, q1_vars, probs_particles)
                        q2_mix_mean, q2_mix_var = mixture_mean_var(q2_means, q2_vars, probs_particles)
                        
                        # Reshape and mean across td_actions
                        q1_t = q1_mix_mean.reshape((int(self.td_actions), batch)).mean(axis=0)
                        q2_t = q2_mix_mean.reshape((int(self.td_actions), batch)).mean(axis=0)
                        q1_v = q1_mix_var.reshape((int(self.td_actions), batch)).mean(axis=0)
                        q2_v = q2_mix_var.reshape((int(self.td_actions), batch)).mean(axis=0)
                        
                        q_particles = jnp.minimum(q1_means, q2_means)
                        q_exp = jnp.sum(probs_particles * q_particles, axis=0)
                        q_t = q_exp.reshape((int(self.td_actions), batch)).mean(axis=0)
                        return acts_sel[0], log_eta_scales_out, mala_acc_rate_out, mala_eta_scale_out, q_t, q1_t, q2_t, q1_v, q2_v

                    def pick_td(_):
                        a, acts_particles, probs_particles, log_eta_scales_out, mala_acc_rate_out, mala_eta_scale_out = sample_action_with_particle_details_metrics(
                            next_eval_key,
                            (policy_params, log_alpha, q1_params, q2_params, log_eta_scales, tfg_lambda_current, state.value_params, state.advantage_second_moment_ema),
                            next_obs,
                            hard_min_q,
                        )
                        # Get Q distributions for each particle
                        q1_means, q1_vars = jax.vmap(lambda actp: self.agent.q(target_q1_params, next_obs, actp), in_axes=0)(acts_particles)
                        q2_means, q2_vars = jax.vmap(lambda actp: self.agent.q(target_q2_params, next_obs, actp), in_axes=0)(acts_particles)
                        
                        # Aggregate across particles using mixture distribution
                        q1_mix_mean, q1_mix_var = mixture_mean_var(q1_means, q1_vars, probs_particles)
                        q2_mix_mean, q2_mix_var = mixture_mean_var(q2_means, q2_vars, probs_particles)
                        
                        # For the min Q value (means only)
                        q_particles = jnp.minimum(q1_means, q2_means)
                        q_t = jnp.sum(probs_particles * q_particles, axis=0)
                        return a, log_eta_scales_out, mala_acc_rate_out, mala_eta_scale_out, q_t, q1_mix_mean, q2_mix_mean, q1_mix_var, q2_mix_var

                    def pick_multi(_):
                        return sample_multi_td()

                    next_action, log_eta_scales, mala_acc_rate, mala_eta_scale, q_target, q_target_q1, q_target_q2, q_target_q1_var, q_target_q2_var = jax.lax.cond(
                        use_multi_td,
                        pick_multi,
                        pick_td,
                        operand=None,
                    )
                    q_target_min = q_target

                # Compute TD backups based on q_bootstrap_agg mode
                # For distributional critics, we also compute target variances
                # Target distribution: r + gamma * Q_target (variance scales by gamma^2)
                gamma_sq = jnp.float32(self.gamma ** 2)
                not_done = (1 - done)
                
                if self.q_bootstrap_agg == "independent":
                    # Each Q bootstraps from its own target network
                    q_backup_q1 = reward + not_done * self.gamma * q_target_q1
                    q_backup_q2 = reward + not_done * self.gamma * q_target_q2
                    # Variance: Var[r + gamma*Q] = gamma^2 * Var[Q] (reward is deterministic)
                    q_backup_q1_var = not_done * gamma_sq * q_target_q1_var
                    q_backup_q2_var = not_done * gamma_sq * q_target_q2_var
                elif self.q_bootstrap_agg == "mean":
                    # Both Qs bootstrap from the mean of target networks
                    # Mean of independent normals: (X1+X2)/2 ~ N((mu1+mu2)/2, (var1+var2)/4)
                    q_target_avg = jnp.float32(0.5) * (q_target_q1 + q_target_q2)
                    q_target_avg_var = jnp.float32(0.25) * (q_target_q1_var + q_target_q2_var)
                    q_backup_q1 = reward + not_done * self.gamma * q_target_avg
                    q_backup_q2 = q_backup_q1
                    q_backup_q1_var = not_done * gamma_sq * q_target_avg_var
                    q_backup_q2_var = q_backup_q1_var
                elif self.q_bootstrap_agg == "mixture":
                    # Use mixture distribution aggregation
                    q_mix_mean, q_mix_var = aggregate_q_distributions(
                        q_target_q1, q_target_q1_var, q_target_q2, q_target_q2_var, "mixture"
                    )
                    q_backup_q1 = reward + not_done * self.gamma * q_mix_mean
                    q_backup_q2 = q_backup_q1
                    q_backup_q1_var = not_done * gamma_sq * q_mix_var
                    q_backup_q2_var = q_backup_q1_var
                elif self.q_bootstrap_agg == "pick_min":
                    # DSAC-T style: choose parameters from the critic with the lower mean
                    q_pick_mean, q_pick_var = aggregate_q_distributions(
                        q_target_q1, q_target_q1_var, q_target_q2, q_target_q2_var, "pick_min"
                    )
                    q_backup_q1 = reward + not_done * self.gamma * q_pick_mean
                    q_backup_q2 = q_backup_q1
                    q_backup_q1_var = not_done * gamma_sq * q_pick_var
                    q_backup_q2_var = q_backup_q1_var
                else:
                    # Default "min": both Qs bootstrap from pessimistic min target
                    # For distributional min, use aggregate_q_distributions
                    q_min_mean, q_min_var = aggregate_q_distributions(
                        q_target_q1, q_target_q1_var, q_target_q2, q_target_q2_var, "min"
                    )
                    q_backup_q1 = reward + not_done * self.gamma * q_min_mean
                    q_backup_q2 = q_backup_q1
                    q_backup_q1_var = not_done * gamma_sq * q_min_var
                    q_backup_q2_var = q_backup_q1_var
                q_backup = q_backup_q1  # For logging (uses Q1's backup as representative)

                q_backup_mean = jnp.mean(q_backup)
                q_backup_std = jnp.std(q_backup)
                q_backup_min = jnp.min(q_backup)
                q_backup_max = jnp.max(q_backup)
                q_target_mean = jnp.mean(q_target)
                q_target_std = jnp.std(q_target)
                reward_mean = jnp.mean(reward)
                done_frac = jnp.mean(done)

                # Huber loss helper (shared by Q1 and Q2)
                delta = jnp.float32(self.q_td_huber_width) * jnp.float32(self.reward_scale)
                use_huber = jnp.isfinite(delta)

                def huber_loss(e):
                    abs_e = jnp.abs(e)
                    quad = jnp.minimum(abs_e, delta)
                    lin = abs_e - quad
                    return jnp.float32(0.5) * quad * quad + delta * lin

                def compute_td_loss(td_err):
                    per_elem = jax.lax.cond(
                        use_huber,
                        lambda ee: huber_loss(ee),
                        lambda ee: ee * ee,
                        td_err,
                    )
                    return jnp.mean(per_elem)
                
                # DSAC-T refinements: compute omega scale factor for loss scaling
                use_dsac_t = (self.dsac_expected_value_sub or 
                              self.dsac_adaptive_clip_xi > 0.0 or 
                              self.dsac_omega_scaling)
                
                # Compute CURRENT omega from Q predictions BEFORE update (for omega scaling and EMA)
                # omega = batch-mean of predicted variance (Eq. 24 in DSAC-T paper)
                # b = batch-mean of predicted std (for adaptive clipping)
                if self.agent.distributional_critic and (self.dsac_omega_scaling or self.dsac_adaptive_clip_xi > 0.0):
                    _, q1_var_current = self.agent.q(q1_params, obs, action)
                    _, q2_var_current = self.agent.q(q2_params, obs, action)
                    dsac_current_omega = jnp.float32(0.5) * (jnp.mean(q1_var_current) + jnp.mean(q2_var_current))
                    dsac_current_b = jnp.sqrt(dsac_current_omega)
                else:
                    dsac_current_omega = jnp.float32(1.0)
                    dsac_current_b = jnp.float32(1.0)
                
                # DSAC-T parameters for custom gradient
                dsac_clip_xi = jnp.float32(self.dsac_adaptive_clip_xi)
                dsac_use_omega_scaling = jnp.bool_(self.dsac_omega_scaling)
                
                def compute_distributional_td_loss(pred_mean, pred_var, target_mean, target_var):
                    """Compute TD loss for distributional critic using cross-entropy."""
                    # DSAC-T: Use custom gradient with all three refinements
                    # Pass b_ema (EMA of σ) for adaptive clipping and omega_ema for gradient scaling
                    if use_dsac_t:
                        return dsac_t_loss_with_custom_grad(
                            pred_mean, pred_var, target_mean, target_var,
                            dsac_clip_xi, dsac_b_ema, dsac_omega_ema, dsac_use_omega_scaling
                        )
                    # Standard gradient modifiers
                    if self.critic_grad_modifier == "natgrad":
                        return mean_gaussian_cross_entropy_natgrad(pred_mean, pred_var, target_mean, target_var)
                    if self.critic_grad_modifier == "variance_scaled":
                        ce = gaussian_cross_entropy(pred_mean, pred_var, target_mean, target_var)
                        return jnp.mean(jax.lax.stop_gradient(pred_var) * ce)
                    return jnp.mean(gaussian_cross_entropy(pred_mean, pred_var, target_mean, target_var))

                def compute_distributional_td_ce(pred_mean, pred_var, target_mean, target_var):
                    """Compute raw cross-entropy (for logging/comparability)."""
                    return jnp.mean(gaussian_cross_entropy(pred_mean, pred_var, target_mean, target_var))

                # Decorrelated Q batches: split batch so Q1 and Q2 see different transitions
                if self.decorrelated_q_batches:
                    batch_size = obs.shape[0]
                    half = batch_size // 2
                    # Shuffle first, then split to ensure both halves are random
                    perm = jax.random.permutation(q2_shuffle_key, batch_size)
                    idx_q1 = perm[:half]
                    idx_q2 = perm[half:]
                    
                    obs_q1, action_q1 = obs[idx_q1], action[idx_q1]
                    obs_q2, action_q2 = obs[idx_q2], action[idx_q2]
                    q_backup_q1_batch = q_backup_q1[idx_q1]
                    q_backup_q2_batch = q_backup_q2[idx_q2]
                    q_backup_q1_var_batch = q_backup_q1_var[idx_q1]
                    q_backup_q2_var_batch = q_backup_q2_var[idx_q2]

                    def q1_loss_fn(q_params: hk.Params) -> jax.Array:
                        q_mean, q_var = self.agent.q(q_params, obs_q1, action_q1)
                        if self.agent.distributional_critic:
                            q_loss = compute_distributional_td_loss(q_mean, q_var, q_backup_q1_batch, q_backup_q1_var_batch)
                        else:
                            td_err = q_mean - q_backup_q1_batch
                            q_loss = compute_td_loss(td_err)
                        return q_loss, q_mean

                    def q2_loss_fn(q_params: hk.Params) -> jax.Array:
                        q_mean, q_var = self.agent.q(q_params, obs_q2, action_q2)
                        if self.agent.distributional_critic:
                            q_loss = compute_distributional_td_loss(q_mean, q_var, q_backup_q2_batch, q_backup_q2_var_batch)
                        else:
                            td_err = q_mean - q_backup_q2_batch
                            q_loss = compute_td_loss(td_err)
                        return q_loss, q_mean
                else:
                    def q1_loss_fn(q_params: hk.Params) -> jax.Array:
                        q_mean, q_var = self.agent.q(q_params, obs, action)
                        if self.agent.distributional_critic:
                            q_loss = compute_distributional_td_loss(q_mean, q_var, q_backup_q1, q_backup_q1_var)
                        else:
                            td_err = q_mean - q_backup_q1
                            q_loss = compute_td_loss(td_err)
                        return q_loss, q_mean
                    
                    def q2_loss_fn(q_params: hk.Params) -> jax.Array:
                        q_mean, q_var = self.agent.q(q_params, obs, action)
                        if self.agent.distributional_critic:
                            q_loss = compute_distributional_td_loss(q_mean, q_var, q_backup_q2, q_backup_q2_var)
                        else:
                            td_err = q_mean - q_backup_q2
                            q_loss = compute_td_loss(td_err)
                        return q_loss, q_mean

                (q1_loss, q1), q1_grads = jax.value_and_grad(q1_loss_fn, has_aux=True)(q1_params)
                # SGLD: Add Langevin noise to gradients if enabled
                if self.langevin_q_noise:
                    batch_size = obs.shape[0]
                    langevin_scale = jnp.sqrt(2.0 * self.lr_q / batch_size)
                    # Generate independent noise for each parameter leaf
                    leaves, treedef = jax.tree_util.tree_flatten(q1_grads)
                    q1_noise_keys = jax.random.split(q1_langevin_key, len(leaves))
                    noisy_leaves = [g + langevin_scale * jax.random.normal(k, g.shape) for g, k in zip(leaves, q1_noise_keys)]
                    q1_grads = jax.tree_util.tree_unflatten(treedef, noisy_leaves)
                q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
                q1_params = optax.apply_updates(q1_params, q1_update)
                
                if self.single_q_network:
                    # Use the same params for both Q networks
                    q2_params = q1_params
                    q2_loss = q1_loss
                    q2 = q1
                else:
                    (q2_loss, q2), q2_grads = jax.value_and_grad(q2_loss_fn, has_aux=True)(q2_params)
                    # SGLD: Add Langevin noise to gradients if enabled
                    if self.langevin_q_noise:
                        leaves2, treedef2 = jax.tree_util.tree_flatten(q2_grads)
                        q2_noise_keys = jax.random.split(q2_langevin_key, len(leaves2))
                        noisy_leaves2 = [g + langevin_scale * jax.random.normal(k, g.shape) for g, k in zip(leaves2, q2_noise_keys)]
                        q2_grads = jax.tree_util.tree_unflatten(treedef2, noisy_leaves2)
                    q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
                    q2_params = optax.apply_updates(q2_params, q2_update)

                q1_pred_mean = jnp.mean(q1)
                q1_pred_std = jnp.std(q1)
                q2_pred_mean = jnp.mean(q2)
                q2_pred_std = jnp.std(q2)
                if self.decorrelated_q_batches:
                    td_error1_std = jnp.std(q1 - q_backup_q1_batch)
                    td_error2_std = jnp.std(q2 - q_backup_q2_batch)
                else:
                    td_error1_std = jnp.std(q1 - q_backup_q1)
                    td_error2_std = jnp.std(q2 - q_backup_q2)
            else:
                if self.pure_bc_training:
                    next_action = action
                else:
                    td_params = (policy_params, log_alpha, q1_params, q2_params, log_eta_scales, tfg_lambda_current, state.value_params, state.advantage_second_moment_ema)

                    if (not self.mala_per_level_eta) and self.agent.energy_mode and self.agent.mala_steps > 0:
                        mala_eta_scale_in = jnp.exp(log_eta_scales[0])

                    next_action, log_eta_scales, mala_acc_rate, mala_eta_scale = sample_action_with_agg_metrics(
                        next_eval_key,
                        td_params,
                        next_obs,
                        hard_min_q,
                    )

                    if (not self.mala_per_level_eta) and self.agent.energy_mode and self.agent.mala_steps > 0:
                        mala_eta_scale_out = jnp.exp(log_eta_scales[0])

                if self.latent_action_space:
                    next_action_env = latent_to_action_normalcdf(next_action)
                else:
                    next_action_env = next_action
                mean_abs_action = jnp.mean(jnp.abs(next_action_env))
                std_action = jnp.std(next_action_env)
                clip_frac = jnp.mean((jnp.abs(next_action_env) > jnp.float32(0.99)).astype(jnp.float32))

                def reward_loss_fn(q_params: hk.Params) -> jax.Array:
                    r_pred_mean, r_pred_var = self.agent.q(q_params, obs, action)
                    r_loss = jnp.mean((r_pred_mean - reward) ** 2)
                    return r_loss, r_pred_mean

                (r1_loss, q1), q1_grads = jax.value_and_grad(reward_loss_fn, has_aux=True)(q1_params)
                q1_update, q1_opt_state = self.optim.update(q1_grads, q1_opt_state)
                q1_params = optax.apply_updates(q1_params, q1_update)
                
                if self.single_q_network:
                    # Use the same params for both Q networks
                    q2_params = q1_params
                    r2_loss = r1_loss
                    q2 = q1
                else:
                    (r2_loss, q2), q2_grads = jax.value_and_grad(reward_loss_fn, has_aux=True)(q2_params)
                    q2_update, q2_opt_state = self.optim.update(q2_grads, q2_opt_state)
                    q2_params = optax.apply_updates(q2_params, q2_update)
                reward_loss = 0.5 * (r1_loss + r2_loss)

            def policy_loss_fn(policy_params) -> jax.Array:
                if self.pure_bc_training:
                    cond_obs = obs
                    target_action = action
                    q_min = get_min_q(cond_obs, target_action)
                    q_mean, q_std = q_min.mean(), q_min.std()
                    scaled_q = jnp.zeros_like(q_min)
                    base_q_weights = jnp.ones_like(q_min)
                    loss_weights = jnp.ones_like(q_min)
                else:
                    cond_obs = next_obs
                    target_action = next_action
                    q_min = get_min_q(cond_obs, target_action)
                    q_mean, q_std = q_min.mean(), q_min.std()
                    if self.fix_q_norm_bug:
                        norm_q = (q_min - running_mean) / running_std
                    else:
                        norm_q = q_min - running_mean / running_std
                    scaled_q = norm_q.clip(-3.0, 3.0) / jnp.exp(log_alpha)
                    base_q_weights = jnp.exp(scaled_q)
                    if self.use_reweighting:
                        loss_weights = base_q_weights
                    else:
                        loss_weights = jnp.ones_like(base_q_weights)

                def denoiser(t, x):
                    return self.agent.policy(policy_params, cond_obs, x, t)

                if self.policy_loss_type in ("ula_kl", "guided_ula_kl", "e2e_guided_ula_kl"):
                    # --- All ULA-based losses use importance-sampled noise levels ---
                    B = self.agent.diffusion.beta_schedule()
                    target_action_sg = jax.lax.stop_gradient(target_action)

                    # Per-level ULA step sizes (stop-gradiented)
                    eta_scales_sg = jax.lax.stop_gradient(jnp.exp(log_eta_scales))
                    eta_all = eta_scales_sg * B.betas  # eta_t for each level [T]

                    # IS weights: sample t proportional to the t-dependent loss weight
                    if self.policy_loss_type == "ula_kl":
                        # epsilon-space weight: eta_t / (1 - alpha_bar_t)
                        one_minus_abar_all = jnp.maximum(
                            jnp.float32(1.0) - B.alphas_cumprod, jnp.float32(1e-8)
                        )
                        is_weights = eta_all / one_minus_abar_all  # [T]
                    else:
                        # score-space weight: eta_t  (the 1/4 is a constant)
                        is_weights = eta_all  # [T]

                    log_is_w = jnp.log(jnp.maximum(is_weights, jnp.float32(1e-30)))
                    t = jax.random.categorical(
                        diffusion_time_key, log_is_w, shape=(obs.shape[0],)
                    )

                    # After IS, per-sample w_t is replaced by constant mean(w_t)
                    mean_is_w = jax.lax.stop_gradient(is_weights.mean())

                    # Common forward pass
                    noise = jax.random.normal(diffusion_noise_key, target_action_sg.shape)
                    x_noisy = jax.vmap(self.agent.diffusion.q_sample)(t, target_action_sg, noise)
                    noise_pred = denoiser(t, x_noisy)

                    if self.policy_loss_type == "ula_kl":
                        # --- IS-corrected ULA-KL: mean_w * ||eps_theta - eps||^2 ---
                        per_sample_loss = mean_is_w * jnp.sum(
                            (noise_pred - noise) ** 2, axis=-1
                        )
                        loss = (loss_weights.reshape(-1) * per_sample_loss).mean()

                    elif self.policy_loss_type == "guided_ula_kl":
                        # --- IS-corrected Guided ULA-KL (Option A, Hessian-weighted) ---
                        # Original: (eta_t/4)*||M_t delta_s||^2  →  IS: (mean_eta/4)*||M_t delta_s||^2
                        cond_obs_sg = jax.lax.stop_gradient(cond_obs)
                        sqrt_1m_abar = B.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
                        s_theta = -noise_pred / sqrt_1m_abar
                        s_cond = -noise / sqrt_1m_abar
                        delta_s = s_theta - s_cond

                        alpha_bar_t = B.alphas_cumprod[t]
                        lambda_t = lambda_for_step(t, tfg_lambda_current)
                        c_t = lambda_t * (1.0 - alpha_bar_t) / alpha_bar_t

                        # Scalar Q for a single (action, obs) pair
                        def q_scalar_single(a, o):
                            a_b, o_b = a[None], o[None]
                            q1m, q1v = self.agent.q(q1_params, o_b, a_b)
                            q2m, q2v = self.agent.q(q2_params, o_b, a_b)
                            return aggregate_q_signal(q1m, q1v, q2m, q2v)[0]

                        # Hessian of Q at clean action (stop-gradiented)
                        H_Q = jax.lax.stop_gradient(
                            jax.vmap(jax.hessian(q_scalar_single))(
                                target_action_sg, cond_obs_sg
                            )
                        )  # [batch, act_dim, act_dim]

                        # M_t = I + c_t * H_Q
                        act_dim = target_action.shape[-1]
                        M_t = jnp.eye(act_dim)[None] + c_t[:, None, None] * H_Q

                        M_delta_s = jnp.einsum('bij,bj->bi', M_t, delta_s)
                        per_sample_loss = (mean_is_w / 4.0) * jnp.sum(M_delta_s ** 2, axis=-1)
                        loss = (loss_weights.reshape(-1) * per_sample_loss).mean()

                    else:  # e2e_guided_ula_kl
                        # --- IS-corrected end-to-end guided drift MSE ---
                        # Original: (eta_t/4)*||g_theta - g_target||^2  →  IS: (mean_eta/4)*...
                        cond_obs_sg = jax.lax.stop_gradient(cond_obs)
                        sqrt_1m_abar = B.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
                        sqrt_abar = B.sqrt_alphas_cumprod[t].reshape(-1, 1)
                        s_theta = -noise_pred / sqrt_1m_abar
                        s_cond = -noise / sqrt_1m_abar

                        # Model's Tweedie estimate (depends on policy_params)
                        x0_hat_theta = (x_noisy - sqrt_1m_abar * noise_pred) / sqrt_abar
                        x0_hat_clipped = jnp.clip(
                            x0_hat_theta,
                            -self.x0_hat_clip_radius,
                            self.x0_hat_clip_radius,
                        )

                        # Scalar Q for a single (action, obs) pair
                        def q_scalar_single(a, o):
                            a_b, o_b = a[None], o[None]
                            q1m, q1v = self.agent.q(q1_params, o_b, a_b)
                            q2m, q2v = self.agent.q(q2_params, o_b, a_b)
                            return aggregate_q_signal(q1m, q1v, q2m, q2v)[0]

                        # Q gradient at model's Tweedie (depends on policy_params)
                        q_grad_model = jax.vmap(jax.grad(q_scalar_single))(
                            x0_hat_clipped, cond_obs
                        )

                        # Q gradient at clean action (fixed target)
                        q_grad_target = jax.lax.stop_gradient(
                            jax.vmap(jax.grad(q_scalar_single))(
                                target_action_sg, cond_obs_sg
                            )
                        )

                        # Guidance scale: lambda / sqrt(alpha_bar)
                        guidance_scale = self.tfg_lambda / sqrt_abar

                        g_theta = s_theta + guidance_scale * q_grad_model
                        g_target = jax.lax.stop_gradient(
                            s_cond + guidance_scale * q_grad_target
                        )

                        delta_g = g_theta - g_target
                        per_sample_loss = (mean_is_w / 4.0) * jnp.sum(delta_g ** 2, axis=-1)
                        loss = (loss_weights.reshape(-1) * per_sample_loss).mean()

                else:
                    # eps_mse: uniform sampling, standard weighted MSE
                    t = jax.random.randint(
                        diffusion_time_key,
                        (obs.shape[0],),
                        0,
                        self.agent.num_timesteps,
                    )
                    loss = self.agent.diffusion.weighted_p_loss(
                        diffusion_noise_key,
                        loss_weights,
                        denoiser,
                        t,
                        jax.lax.stop_gradient(target_action),
                    )

                return loss, (base_q_weights, scaled_q, q_mean, q_std)

            (total_loss, (q_weights, scaled_q, q_mean, q_std)), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)

            # update alpha
            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                approx_entropy = 0.5 * self.agent.act_dim * jnp.log( 2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha)) ** 2)
                log_alpha_loss = -1 * log_alpha * (-1 * jax.lax.stop_gradient(approx_entropy) + self.agent.target_entropy)
                return log_alpha_loss

            # update networks
            def param_update(optim, params, grads, opt_state):
                update, new_opt_state = optim.update(grads, opt_state)
                new_params = optax.apply_updates(params, update)
                return new_params, new_opt_state

            def delay_param_update(optim, params, grads, opt_state):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda params, opt_state: param_update(optim, params, grads, opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            def delay_alpha_param_update(optim, params, opt_state):
                return jax.lax.cond(
                    step % self.delay_alpha_update == 0,
                    lambda params, opt_state: param_update(optim, params, jax.grad(log_alpha_loss_fn)(params), opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            def delay_target_update(params, target_params, tau):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda target_params: optax.incremental_update(params, target_params, tau),
                    lambda target_params: target_params,
                    target_params
                )

            policy_params, policy_opt_state = delay_param_update(self.policy_optim, policy_params, policy_grads, policy_opt_state)
            if not self.no_entropy_tuning:
                log_alpha, log_alpha_opt_state = delay_alpha_param_update(self.alpha_optim, log_alpha, log_alpha_opt_state)

            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            if self.single_q_network:
                target_q2_params = target_q1_params
            else:
                target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)
            target_policy_params = delay_target_update(policy_params, target_policy_params, self.tau)

            new_running_mean = running_mean + 0.001 * (q_mean - running_mean)
            new_running_std = running_std + 0.001 * (q_std - running_std)

            # DSAC-T EMA updates using pre-computed omega and b (from before gradient update)
            # Per DSAC-T paper Eq. 25-26: Polyak update for omega and b EMAs
            if self.agent.distributional_critic and (self.dsac_omega_scaling or self.dsac_adaptive_clip_xi > 0.0):
                dsac_tau = jnp.float32(self.dsac_omega_tau)
                new_dsac_omega_ema = dsac_omega_ema + dsac_tau * (dsac_current_omega - dsac_omega_ema)
                new_dsac_b_ema = dsac_b_ema + dsac_tau * (dsac_current_b - dsac_b_ema)
            else:
                new_dsac_omega_ema = dsac_omega_ema
                new_dsac_b_ema = dsac_b_ema

            # Normalized advantage guidance: train V(s) and update A² EMA
            value_params_updated = state.value_params
            value_opt_state_updated = state.opt_state.value
            new_adv_second_moment_ema = state.advantage_second_moment_ema
            value_loss_log = jnp.float32(0.0)
            adv_second_moment_log = jnp.float32(0.0)
            
            if self.critic_normalization != "none" and state.value_params is not None:
                # Get Q values for current (obs, action) pairs using the aggregation
                q1_for_v, _ = self.agent.q(q1_params, obs, action)
                q2_for_v, _ = self.agent.q(q2_params, obs, action)
                q_for_v = aggregate_q_signal(q1_for_v, jnp.zeros_like(q1_for_v), q2_for_v, jnp.zeros_like(q2_for_v))
                if self.use_reward_critic:
                    q_for_v = q_for_v * (jnp.float32(1.0) / jnp.float32(1.0 - self.gamma))
                
                if self.critic_normalization == "ema":
                    # V(s) predicts E[Q(s, a)] under tilted policy - train with MSE
                    def value_loss_fn(v_params):
                        v_pred = self._value_net.apply(v_params, obs)
                        return jnp.mean((v_pred - jax.lax.stop_gradient(q_for_v)) ** 2)
                    
                    v_loss, v_grads = jax.value_and_grad(value_loss_fn)(state.value_params)
                    v_updates, value_opt_state_updated = self.optim.update(v_grads, state.opt_state.value, state.value_params)
                    value_params_updated = optax.apply_updates(state.value_params, v_updates)
                    value_loss_log = v_loss
                    
                    # Update A² EMA: EMA of (Q - V)²
                    v_current = self._value_net.apply(state.value_params, obs)
                    advantage_squared = (q_for_v - v_current) ** 2
                    adv_second_moment_batch = jnp.mean(advantage_squared)
                    adv_ema_tau = jnp.float32(self.advantage_ema_tau)
                    new_adv_second_moment_ema = state.advantage_second_moment_ema + adv_ema_tau * (adv_second_moment_batch - state.advantage_second_moment_ema)
                    adv_second_moment_log = adv_second_moment_batch
                else:  # distributional
                    # Distributional V(s) outputs (mean, log_var) - train with NLL
                    def value_loss_fn(v_params):
                        v_mean, v_log_var = self._value_net.apply(v_params, obs)
                        v_var = jnp.exp(v_log_var)
                        # NLL of Gaussian: 0.5 * (log(var) + (q - mean)^2 / var)
                        nll = 0.5 * (v_log_var + (jax.lax.stop_gradient(q_for_v) - v_mean) ** 2 / jnp.maximum(v_var, 1e-6))
                        return jnp.mean(nll)
                    
                    v_loss, v_grads = jax.value_and_grad(value_loss_fn)(state.value_params)
                    v_updates, value_opt_state_updated = self.optim.update(v_grads, state.opt_state.value, state.value_params)
                    value_params_updated = optax.apply_updates(state.value_params, v_updates)
                    value_loss_log = v_loss
                    
                    # Log the learned variance (no EMA needed for distributional)
                    v_mean_current, v_log_var_current = self._value_net.apply(state.value_params, obs)
                    adv_second_moment_log = jnp.mean(jnp.exp(v_log_var_current))

            state = Diffv2TrainState(
                params=Diffv2Params(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha),
                opt_state=Diffv2OptStates(q1=q1_opt_state, q2=q2_opt_state, policy=policy_opt_state, log_alpha=log_alpha_opt_state, value=value_opt_state_updated),
                step=step + 1,
                entropy=jnp.float32(0.0),
                running_mean=new_running_mean,
                running_std=new_running_std,
                log_eta_scales=log_eta_scales,
                tfg_lambda=tfg_lambda_current,
                dsac_omega_ema=new_dsac_omega_ema,
                dsac_b_ema=new_dsac_b_ema,
                value_params=value_params_updated,
                advantage_second_moment_ema=new_adv_second_moment_ema,
            )

            if self.agent.distributional_critic:
                q1_pred_mean_log, q1_pred_var_log = self.agent.q(q1_params, obs, action)
                q2_pred_mean_log, q2_pred_var_log = self.agent.q(q2_params, obs, action)
                q1_ce_loss = compute_distributional_td_ce(q1_pred_mean_log, q1_pred_var_log, q_backup_q1, q_backup_q1_var)
                q2_ce_loss = compute_distributional_td_ce(q2_pred_mean_log, q2_pred_var_log, q_backup_q2, q_backup_q2_var)
            else:
                q1_ce_loss = q1_loss
                q2_ce_loss = q2_loss

            info = {
                "q1_loss": q1_ce_loss if not self.use_reward_critic else 0.0,
                "q1_mean": jnp.mean(q1) if not self.use_reward_critic else 0.0,
                "q1_max": jnp.max(q1) if not self.use_reward_critic else 0.0,
                "q1_min": jnp.min(q1) if not self.use_reward_critic else 0.0,
                "q2_loss": q2_ce_loss if not self.use_reward_critic else 0.0,
                "policy_loss": total_loss,
                "alpha": jnp.exp(log_alpha),
                "MALA/acceptance_rate": mala_acc_rate,
                "td/q1_ce_loss": q1_ce_loss,
                "td/q2_ce_loss": q2_ce_loss,
                "td/q_backup_mean": q_backup_mean,
                "td/q_backup_std": q_backup_std,
                "td/q_backup_min": q_backup_min,
                "td/q_backup_max": q_backup_max,
                "td/q_target_mean": q_target_mean,
                "td/q_target_std": q_target_std,
                "td/reward_mean": reward_mean,
                "td/done_frac": done_frac,
                "td/q1_pred_mean": q1_pred_mean,
                "td/q1_pred_std": q1_pred_std,
                "td/q2_pred_mean": q2_pred_mean,
                "td/q2_pred_std": q2_pred_std,
                "td/td_error1_std": td_error1_std,
                "td/td_error2_std": td_error2_std,
                "act/mean_abs_action": mean_abs_action,
                "act/std_action": std_action,
                "act/clip_frac": clip_frac,
                "q_weights_std": jnp.std(q_weights),
                "q_weights_mean": jnp.mean(q_weights),
                "q_weights_min": jnp.min(q_weights),
                "q_weights_max": jnp.max(q_weights),
                "scale_q_mean": jnp.mean(scaled_q),
                "scale_q_std": jnp.std(scaled_q),
                "running_q_mean": new_running_mean,
                "running_q_std": new_running_std,
                "entropy_approx": 0.5 * self.agent.act_dim * jnp.log( 2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha)) ** 2),
            }

            if not self.mala_per_level_eta:
                info["MALA/eta_scale_in"] = mala_eta_scale_in
                info["MALA/eta_scale_out"] = mala_eta_scale_out

            if not self.mala_per_level_eta:
                info["MALA/eta_scale"] = mala_eta_scale
            else:
                info["hist/MALA/eta_scale"] = jnp.exp(log_eta_scales)
            if self.use_reward_critic:
                info["reward_loss"] = reward_loss
            # Critic normalization logging
            if self.critic_normalization == "ema":
                info["norm_adv/value_mse"] = value_loss_log
                info["norm_adv/adv_second_moment"] = adv_second_moment_log
                info["norm_adv/adv_second_moment_ema"] = new_adv_second_moment_ema
                info["norm_adv/adv_std_ema"] = jnp.sqrt(jnp.maximum(new_adv_second_moment_ema, 1e-6))
            elif self.critic_normalization == "distributional":
                info["norm_adv/value_nll"] = value_loss_log
                info["norm_adv/mean_var"] = adv_second_moment_log
                info["norm_adv/mean_std"] = jnp.sqrt(jnp.maximum(adv_second_moment_log, 1e-6))
            # DSAC-T logging
            if self.agent.distributional_critic and (self.dsac_omega_scaling or self.dsac_adaptive_clip_xi > 0.0):
                info["dsac_t/omega_current"] = dsac_current_omega
                info["dsac_t/omega_ema"] = new_dsac_omega_ema
                info["dsac_t/b_current"] = dsac_current_b
                info["dsac_t/b_ema"] = new_dsac_b_ema
                info["dsac_t/clip_bound"] = dsac_clip_xi * new_dsac_b_ema  # b = ξ * σ̄
            return state, info

        @jax.jit
        def stateless_supervised_update(
            key: jax.Array,
            state: Diffv2TrainState,
            data: Experience,
        ) -> Tuple[Diffv2TrainState, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state, _value_opt_state = state.opt_state
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std

            next_eval_key, diffusion_time_key, diffusion_noise_key, q_agg_key = jax.random.split(key, 4)

            if self.q_critic_agg == "random":
                critic_idx = jax.random.randint(q_agg_key, (), 0, 2)

                def aggregate_q_signal(
                    q1_mean: jax.Array,
                    q1_var: jax.Array,
                    q2_mean: jax.Array,
                    q2_var: jax.Array,
                ) -> jax.Array:
                    return jax.lax.select(critic_idx == 0, q1_mean, q2_mean)

            else:

                def aggregate_q_signal(
                    q1_mean: jax.Array,
                    q1_var: jax.Array,
                    q2_mean: jax.Array,
                    q2_var: jax.Array,
                ) -> jax.Array:
                    return aggregate_q(q1_mean, q1_var, q2_mean, q2_var)

            reward = reward * self.reward_scale

            def get_min_q(s, a):
                q1_mean, q1_var = self.agent.q(q1_params, s, a)
                q2_mean, q2_var = self.agent.q(q2_params, s, a)
                q = aggregate_q_signal(q1_mean, q1_var, q2_mean, q2_var)
                if self.use_reward_critic:
                    q = q * (jnp.float32(1.0) / jnp.float32(1.0 - self.gamma))
                return q

            if self.pure_bc_training:
                cond_obs = obs
                target_action = action
                q_min = get_min_q(cond_obs, target_action)
                q_mean, q_std = q_min.mean(), q_min.std()
                scaled_q = jnp.zeros_like(q_min)
                base_q_weights = jnp.ones_like(q_min)
                loss_weights = jnp.ones_like(q_min)
            else:
                td_params = (policy_params, log_alpha, q1_params, q2_params, log_eta_scales, tfg_lambda_current, state.value_params, state.advantage_second_moment_ema)
                next_action, _ = sample_action_with_agg(next_eval_key, td_params, next_obs, hard_min_q)
                cond_obs = next_obs
                target_action = next_action
                q_min = get_min_q(cond_obs, target_action)
                q_mean, q_std = q_min.mean(), q_min.std()
                if self.fix_q_norm_bug:
                    norm_q = (q_min - running_mean) / running_std
                else:
                    norm_q = q_min - running_mean / running_std
                scaled_q = norm_q.clip(-3.0, 3.0) / jnp.exp(log_alpha)
                base_q_weights = jnp.exp(scaled_q)
                if self.use_reweighting:
                    loss_weights = base_q_weights
                else:
                    loss_weights = jnp.ones_like(base_q_weights)

            def policy_loss_fn(policy_p) -> jax.Array:
                def denoiser(t, x):
                    return self.agent.policy(policy_p, cond_obs, x, t)

                t = jax.random.randint(
                    diffusion_time_key,
                    (obs.shape[0],),
                    0,
                    self.agent.num_timesteps,
                )
                loss = self.agent.diffusion.weighted_p_loss(
                    diffusion_noise_key,
                    loss_weights,
                    denoiser,
                    t,
                    jax.lax.stop_gradient(target_action),
                )

                return loss, (base_q_weights, scaled_q, q_mean, q_std)

            (total_loss, (q_weights, scaled_q, q_mean, q_std)), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)

            def param_update(optim, params, grads, opt_state):
                update, new_opt_state = optim.update(grads, opt_state)
                new_params = optax.apply_updates(params, update)
                return new_params, new_opt_state

            def delay_param_update(optim, params, grads, opt_state):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda params, opt_state: param_update(optim, params, grads, opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            policy_params, policy_opt_state = delay_param_update(self.policy_optim, policy_params, policy_grads, policy_opt_state)

            state = Diffv2TrainState(
                params=Diffv2Params(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, target_policy_params, log_alpha),
                opt_state=Diffv2OptStates(q1=q1_opt_state, q2=q2_opt_state, policy=policy_opt_state, log_alpha=log_alpha_opt_state, value=state.opt_state.value),
                step=step,
                entropy=jnp.float32(0.0),
                running_mean=running_mean,
                running_std=running_std,
                log_eta_scales=log_eta_scales,
                tfg_lambda=tfg_lambda_current,
                value_params=state.value_params,
                advantage_second_moment_ema=state.advantage_second_moment_ema,
            )

            info = {
                "q1_loss": jnp.float32(0.0),
                "q1_mean": jnp.float32(0.0),
                "q1_max": jnp.float32(0.0),
                "q1_min": jnp.float32(0.0),
                "q2_loss": jnp.float32(0.0),
                "policy_loss": total_loss,
                "alpha": jnp.exp(log_alpha),
                "q_weights_std": jnp.std(q_weights),
                "q_weights_mean": jnp.mean(q_weights),
                "q_weights_min": jnp.min(q_weights),
                "q_weights_max": jnp.max(q_weights),
                "scale_q_mean": jnp.mean(scaled_q),
                "scale_q_std": jnp.std(scaled_q),
                "running_q_mean": running_mean,
                "running_q_std": running_std,
                "entropy_approx": 0.5 * self.agent.act_dim * jnp.log( 2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha)) ** 2),
            }

            if self.use_reward_critic:
                info["reward_loss"] = jnp.float32(0.0)

            return state, info

        def stateless_get_action_tfg_recur(
            key: jax.Array,
            params,
            obs: jax.Array,
            aggregate_q_fn,
            randomize_q: bool = False,
        ) -> jax.Array:
            policy_params, log_alpha, q1_params, q2_params, log_eta_scales_in, tfg_lambda_current, value_params, adv_second_moment_ema = params

            single = obs.ndim == 1
            if single:
                obs_batch = obs[None, :]
            else:
                obs_batch = obs

            shape = (*obs_batch.shape[:-1], self.agent.act_dim)
            B = self.agent.diffusion.beta_schedule()
            timesteps = self.agent.num_timesteps

            def model_fn(t, x):
                # Scale base score by energy_multiplier (tempers the base distribution)
                return self.energy_multiplier * self.agent.policy(policy_params, obs_batch, x, t)

            def tfg_sample(single_key: jax.Array):
                if randomize_q and self.q_critic_agg == "random":
                    x_key, loop_key, q_agg_key = jax.random.split(single_key, 3)
                    critic_idx = jax.random.randint(q_agg_key, (), 0, 2)

                    def aggregate_q_local(
                        q1_mean: jax.Array,
                        q1_var: jax.Array,
                        q2_mean: jax.Array,
                        q2_var: jax.Array,
                    ) -> jax.Array:
                        return jax.lax.select(critic_idx == 0, q1_mean, q2_mean)

                else:
                    x_key, loop_key = jax.random.split(single_key)
                    aggregate_q_local = aggregate_q_fn

                def q_mean_from_x(x_in, t_idx):
                    noise_pred = model_fn(t_idx, x_in)
                    x0_hat = (
                        x_in * B.sqrt_recip_alphas_cumprod[t_idx]
                        - noise_pred * B.sqrt_recipm1_alphas_cumprod[t_idx]
                    )
                    x0_hat = jnp.clip(x0_hat, -self.x0_hat_clip_radius, self.x0_hat_clip_radius)
                    q1_mean, q1_var = self.agent.q(q1_params, obs_batch, x0_hat)
                    q2_mean, q2_var = self.agent.q(q2_params, obs_batch, x0_hat)
                    q = aggregate_q_local(q1_mean, q1_var, q2_mean, q2_var)
                    if self.use_reward_critic:
                        q = q * (jnp.float32(1.0) / jnp.float32(1.0 - self.gamma))
                    
                    # Critic normalization: use (Q - V) / std instead of Q
                    if self.critic_normalization == "ema" and value_params is not None:
                        v = self._value_net.apply(value_params, obs_batch)
                        advantage = q - v
                        adv_std = jnp.sqrt(jnp.maximum(adv_second_moment_ema, jnp.float32(1e-6)))
                        return jnp.mean(advantage / adv_std)
                    elif self.critic_normalization == "distributional" and value_params is not None:
                        v_mean, v_log_var = self._value_net.apply(value_params, obs_batch)
                        advantage = q - v_mean
                        adv_std = jnp.sqrt(jnp.maximum(jnp.exp(v_log_var), jnp.float32(1e-6)))
                        return jnp.mean(advantage / adv_std)
                    
                    return jnp.mean(q)

                def grad_guidance(x_in, t_idx):
                    # Use Q-gradient guidance only if lambda_for_step(t_idx) > 0; otherwise no guidance
                    def guided(x):
                        return jax.grad(lambda xx: q_mean_from_x(xx, t_idx))(x)

                    def unguided(x):
                        return jnp.zeros_like(x)

                    lam = lambda_for_step(t_idx, tfg_lambda_current)
                    return jax.lax.cond(lam > 0.0, guided, unguided, x_in)

                x = 0.5 * jax.random.normal(x_key, shape)
                t_seq = jnp.arange(timesteps)[::-1]

                def body_fn(carry, t):
                    x_t, key_t = carry

                    def recur_step(carry_recur, _):
                        x_cur, key_cur = carry_recur
                        key_cur, key_down, key_up = jax.random.split(key_cur, 3)
                        noise_down = jax.random.normal(key_down, x_cur.shape)
                        noise_up = jax.random.normal(key_up, x_cur.shape)

                        noise_pred = model_fn(t, x_cur)
                        grad_q = grad_guidance(x_cur, t)
                        sigma_t = B.sqrt_one_minus_alphas_cumprod[t]
                        lambda_t = lambda_for_step(t, tfg_lambda_current)
                        eps_guided = noise_pred - lambda_t * sigma_t * grad_q
                        model_mean, model_log_variance = self.agent.diffusion.p_mean_variance(
                            t, x_cur, eps_guided
                        )
                        x_tm1 = model_mean + (t > 0) * jnp.exp(0.5 * model_log_variance) * noise_down

                        alpha_t = B.alphas[t]
                        sqrt_alpha_t = jnp.sqrt(alpha_t)
                        sqrt_one_minus_alpha_t = jnp.sqrt(1.0 - alpha_t)
                        x_back = sqrt_alpha_t * x_tm1 + sqrt_one_minus_alpha_t * noise_up

                        return (x_back, key_cur), None

                    if self.tfg_recur_steps > 0:
                        (x_t, key_t), _ = jax.lax.scan(
                            recur_step,
                            (x_t, key_t),
                            jnp.arange(self.tfg_recur_steps),
                        )

                    key_t, key_down = jax.random.split(key_t)
                    noise_down = jax.random.normal(key_down, x_t.shape)
                    noise_pred = model_fn(t, x_t)
                    grad_q = grad_guidance(x_t, t)
                    sigma_t = B.sqrt_one_minus_alphas_cumprod[t]
                    lambda_t = lambda_for_step(t, tfg_lambda_current)
                    eps_guided = noise_pred - lambda_t * sigma_t * grad_q
                    model_mean, model_log_variance = self.agent.diffusion.p_mean_variance(
                        t, x_t, eps_guided
                    )
                    x_next = model_mean + (t > 0) * jnp.exp(0.5 * model_log_variance) * noise_down
                    return (x_next, key_t), None

                (final_carry, _) = jax.lax.scan(
                    body_fn,
                    (x, loop_key),
                    t_seq,
                )
                x_final, _ = final_carry
                act_final = x_final if self.latent_action_space else jnp.clip(x_final, -1.0, 1.0)
                q1_mean, q1_var = self.agent.q(q1_params, obs_batch, act_final)
                q2_mean, q2_var = self.agent.q(q2_params, obs_batch, act_final)
                q = aggregate_q_local(q1_mean, q1_var, q2_mean, q2_var)
                if self.use_reward_critic:
                    q = q * (jnp.float32(1.0) / jnp.float32(1.0 - self.gamma))
                return act_final, q

            def single_sampler(single_key: jax.Array):
                return tfg_sample(single_key)

            def single_sampler_with_eta(single_key: jax.Array, log_eta_scales_local: jax.Array):
                act, q = tfg_sample(single_key)
                return act, q, log_eta_scales_local

            return sample_with_particles(key, log_alpha, single, single_sampler_with_eta, log_eta_scales_in)

        def stateless_get_action_mala_full(
            key: jax.Array,
            params,
            obs: jax.Array,
            aggregate_q_fn,
            randomize_q: bool = False,
            return_details: bool = False,
        ):
            policy_params, log_alpha, q1_params, q2_params, log_eta_scales_in, tfg_lambda_current, value_params, adv_second_moment_ema = params

            single = obs.ndim == 1
            if single:
                obs_batch = obs[None, :]
            else:
                obs_batch = obs

            shape = (*obs_batch.shape[:-1], self.agent.act_dim)
            B = self.agent.diffusion.beta_schedule()
            timesteps = self.agent.num_timesteps

            def energy_model(t, x):
                E = self.agent.energy_fn(policy_params, obs_batch, x, t)
                # Scale base energy by energy_multiplier (tempers the base distribution)
                return self.energy_multiplier * E

            def mala_chain(single_key: jax.Array, log_eta_scales_init: jax.Array):
                if randomize_q and self.q_critic_agg == "random":
                    key_x, loop_key, q_agg_key = jax.random.split(single_key, 3)
                    critic_idx = jax.random.randint(q_agg_key, (), 0, 2)

                    def aggregate_q_local(
                        q1_mean: jax.Array,
                        q1_var: jax.Array,
                        q2_mean: jax.Array,
                        q2_var: jax.Array,
                    ) -> jax.Array:
                        return jax.lax.select(critic_idx == 0, q1_mean, q2_mean)

                else:
                    key_x, loop_key = jax.random.split(single_key)
                    aggregate_q_local = aggregate_q_fn

                def q_min_from_x0(x0_in):
                    q1_mean, q1_var = self.agent.q(q1_params, obs_batch, x0_in)
                    q2_mean, q2_var = self.agent.q(q2_params, obs_batch, x0_in)
                    return aggregate_q_local(q1_mean, q1_var, q2_mean, q2_var)

                def q_mean_from_x(x_in, t_idx):
                    noise_pred = self.agent.policy(policy_params, obs_batch, x_in, t_idx)
                    x0_hat = (
                        x_in * B.sqrt_recip_alphas_cumprod[t_idx]
                        - noise_pred * B.sqrt_recipm1_alphas_cumprod[t_idx]
                    )
                    x0_hat = jnp.clip(x0_hat, -self.x0_hat_clip_radius, self.x0_hat_clip_radius)
                    q = q_min_from_x0(x0_hat)
                    if self.use_reward_critic:
                        q = q * (jnp.float32(1.0) / jnp.float32(1.0 - self.gamma))
                    
                    # Critic normalization: use (Q - V) / std instead of Q
                    if self.critic_normalization == "ema" and value_params is not None:
                        v = self._value_net.apply(value_params, obs_batch)
                        advantage = q - v
                        adv_std = jnp.sqrt(jnp.maximum(adv_second_moment_ema, jnp.float32(1e-6)))
                        return jnp.mean(advantage / adv_std)
                    elif self.critic_normalization == "distributional" and value_params is not None:
                        v_mean, v_log_var = self._value_net.apply(value_params, obs_batch)
                        advantage = q - v_mean
                        adv_std = jnp.sqrt(jnp.maximum(jnp.exp(v_log_var), jnp.float32(1e-6)))
                        return jnp.mean(advantage / adv_std)
                    
                    return jnp.mean(q)

                def grad_guidance(x_in, t_idx):
                    def guided(x):
                        return jax.grad(lambda xx: q_mean_from_x(xx, t_idx))(x)

                    def unguided(x):
                        return jnp.zeros_like(x)

                    lam = lambda_for_step(t_idx, tfg_lambda_current)
                    return jax.lax.cond(lam > 0.0, guided, unguided, x_in)

                def energy_total(t, x):
                    def only_model(x_in):
                        return energy_model(t, x_in)

                    def with_q(x_in):
                        E_mod = energy_model(t, x_in)
                        noise_pred = self.agent.policy(policy_params, obs_batch, x_in, t)
                        x0_hat = (
                            x_in * B.sqrt_recip_alphas_cumprod[t]
                            - noise_pred * B.sqrt_recipm1_alphas_cumprod[t]
                        )
                        x0_hat_clipped = jnp.clip(x0_hat, -self.x0_hat_clip_radius, self.x0_hat_clip_radius)
                        q_min = q_min_from_x0(x0_hat_clipped)
                        lambda_t = lambda_for_step(t, tfg_lambda_current)
                        return E_mod - lambda_t * q_min

                    lambda_t = lambda_for_step(t, tfg_lambda_current)
                    return jax.lax.cond(lambda_t > 0.0, with_q, only_model, x)

                def predictor_step(t_idx: jax.Array, x_in: jax.Array, k_in: jax.Array):
                    noise_pred = self.agent.policy(policy_params, obs_batch, x_in, t_idx)
                    # Scale base score by energy_multiplier (tempers the base distribution)
                    # Guidance component is NOT scaled - only the base policy score
                    noise_pred_scaled = self.energy_multiplier * noise_pred
                    if self.mala_guided_predictor:
                        grad_q = grad_guidance(x_in, t_idx)
                        sigma_t = B.sqrt_one_minus_alphas_cumprod[t_idx]
                        lambda_t = lambda_for_step(t_idx, tfg_lambda_current)
                        eps_pred = noise_pred_scaled - lambda_t * sigma_t * grad_q
                    else:
                        eps_pred = noise_pred_scaled
                    model_mean, model_log_variance = self.agent.diffusion.p_mean_variance(
                        t_idx, x_in, eps_pred
                    )
                    k_out, z_key = jax.random.split(k_in)
                    if self.ddim_predictor:
                        # DDIM-style deterministic update (no noise)
                        x_out = model_mean
                    else:
                        # DDPM-style stochastic update
                        z = jax.random.normal(z_key, x_in.shape)
                        x_out = model_mean + (t_idx > 0) * jnp.exp(0.5 * model_log_variance) * z
                    return x_out, k_out

                x0 = jax.random.normal(key_x, shape)

                eta_base_min = jnp.maximum(jnp.min(B.betas), jnp.float32(1e-8))
                eta_base_max = jnp.maximum(jnp.max(B.betas), jnp.float32(1e-8))
                log_eta_min = jnp.log(jnp.float32(1e-8) / eta_base_max)
                log_eta_max = jnp.log(jnp.float32(0.5) / eta_base_min)

                if self.mala_per_level_eta:

                    def level_body(i, carry):
                        x_curr, k, log_eta_scales, acc_sum, acc_count = carry
                        t = timesteps - 1 - i

                        def do_mala_level(carry_in, t_corr: jax.Array):
                            x_in, k_in, log_eta_scales_in, acc_sum_in, acc_count_in = carry_in
                            eta_base_t = jnp.maximum(B.betas[t_corr], jnp.float32(1e-8))
                            log_eta_scale0 = log_eta_scales_in[t_corr]

                            def mala_body(_, state):
                                x_step, k_step, log_eta_scale, acc_sum_step, acc_count_step = state

                                E_x, vjp_x = jax.vjp(lambda xx: energy_total(t_corr, xx), x_step)
                                grad_E_x = vjp_x(jnp.ones_like(E_x))[0]

                                k_step, noise_key, u_key = jax.random.split(k_step, 3)
                                eta_k = jnp.clip(
                                    jnp.exp(log_eta_scale) * eta_base_t,
                                    jnp.float32(1e-8),
                                    jnp.float32(0.5),
                                )
                                z = jax.random.normal(noise_key, x_step.shape)
                                sd = jnp.sqrt(jnp.float32(2.0) * eta_k)
                                x_prop = x_step - eta_k * grad_E_x + sd * z

                                E_x_prop, vjp_x_prop = jax.vjp(lambda xx: energy_total(t_corr, xx), x_prop)
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
                                adapt_rate = jnp.float32(self.mala_adapt_rate)
                                log_eta_scale = log_eta_scale + adapt_rate * (acc_rate - target)
                                log_eta_scale = jnp.clip(log_eta_scale, log_eta_min, log_eta_max)

                                return (
                                    x_new,
                                    k_step,
                                    log_eta_scale,
                                    acc_sum_step + acc_rate,
                                    acc_count_step + jnp.float32(1.0),
                                )

                            x_out, k_out, log_eta_scale_final, acc_sum_level, acc_count_level = jax.lax.fori_loop(
                                0,
                                self.agent.mala_steps,
                                mala_body,
                                (x_in, k_in, log_eta_scale0, jnp.float32(0.0), jnp.float32(0.0)),
                            )

                            log_eta_scales_in = log_eta_scales_in.at[t_corr].set(log_eta_scale_final)
                            return (
                                x_out,
                                k_out,
                                log_eta_scales_in,
                                acc_sum_in + acc_sum_level,
                                acc_count_in + acc_count_level,
                            )

                        if self.mala_predictor_first:
                            x_pred, k = predictor_step(t, x_curr, k)
                            t_corr = t - 1

                            def _with_mala(carry_in):
                                return do_mala_level(carry_in, t_corr)

                            def _no_mala(carry_in):
                                return carry_in

                            x_curr, k, log_eta_scales, acc_sum, acc_count = jax.lax.cond(
                                t > 0,
                                _with_mala,
                                _no_mala,
                                (x_pred, k, log_eta_scales, acc_sum, acc_count),
                            )
                            return x_curr, k, log_eta_scales, acc_sum, acc_count

                        x_curr, k, log_eta_scales, acc_sum, acc_count = do_mala_level(
                            (x_curr, k, log_eta_scales, acc_sum, acc_count), t
                        )
                        x_next, k = predictor_step(t, x_curr, k)
                        return x_next, k, log_eta_scales, acc_sum, acc_count

                    x_final, _, log_eta_scales_out, acc_sum_final, acc_count_final = jax.lax.fori_loop(
                        0,
                        timesteps,
                        level_body,
                        (x0, loop_key, log_eta_scales_init, jnp.float32(0.0), jnp.float32(0.0)),
                    )

                    mala_acc_rate = acc_sum_final / jnp.maximum(acc_count_final, jnp.float32(1.0))
                    mala_eta_scale = jnp.float32(jnp.nan)

                else:

                    log_eta_scale_shared0 = log_eta_scales_init[0]

                    def level_body(i, carry):
                        x_curr, k, log_eta_scale_shared, acc_sum, acc_count = carry
                        t = timesteps - 1 - i

                        def do_mala_shared(carry_in, t_corr: jax.Array):
                            x_in, k_in, log_eta_scale_in, acc_sum_in, acc_count_in = carry_in
                            eta_base_t = jnp.maximum(B.betas[t_corr], jnp.float32(1e-8))

                            def mala_body(_, state):
                                x_step, k_step, log_eta_scale, acc_sum_step, acc_count_step = state

                                E_x, vjp_x = jax.vjp(lambda xx: energy_total(t_corr, xx), x_step)
                                grad_E_x = vjp_x(jnp.ones_like(E_x))[0]

                                k_step, noise_key, u_key = jax.random.split(k_step, 3)
                                eta_k = jnp.clip(
                                    jnp.exp(log_eta_scale) * eta_base_t,
                                    jnp.float32(1e-8),
                                    jnp.float32(0.5),
                                )
                                z = jax.random.normal(noise_key, x_step.shape)
                                sd = jnp.sqrt(jnp.float32(2.0) * eta_k)
                                x_prop = x_step - eta_k * grad_E_x + sd * z

                                E_x_prop, vjp_x_prop = jax.vjp(lambda xx: energy_total(t_corr, xx), x_prop)
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
                                adapt_rate = jnp.float32(self.mala_adapt_rate)
                                log_eta_scale = log_eta_scale + adapt_rate * (acc_rate - target)
                                log_eta_scale = jnp.clip(log_eta_scale, log_eta_min, log_eta_max)

                                return (
                                    x_new,
                                    k_step,
                                    log_eta_scale,
                                    acc_sum_step + acc_rate,
                                    acc_count_step + jnp.float32(1.0),
                                )

                            x_out, k_out, log_eta_scale_out, acc_sum_level, acc_count_level = jax.lax.fori_loop(
                                0,
                                self.agent.mala_steps,
                                mala_body,
                                (x_in, k_in, log_eta_scale_in, jnp.float32(0.0), jnp.float32(0.0)),
                            )

                            return (
                                x_out,
                                k_out,
                                log_eta_scale_out,
                                acc_sum_in + acc_sum_level,
                                acc_count_in + acc_count_level,
                            )

                        if self.mala_predictor_first:
                            x_pred, k = predictor_step(t, x_curr, k)
                            t_corr = t - 1

                            def _with_mala(carry_in):
                                return do_mala_shared(carry_in, t_corr)

                            def _no_mala(carry_in):
                                return carry_in

                            x_curr, k, log_eta_scale_shared, acc_sum, acc_count = jax.lax.cond(
                                t > 0,
                                _with_mala,
                                _no_mala,
                                (x_pred, k, log_eta_scale_shared, acc_sum, acc_count),
                            )
                            return x_curr, k, log_eta_scale_shared, acc_sum, acc_count

                        x_curr, k, log_eta_scale_shared, acc_sum, acc_count = do_mala_shared(
                            (x_curr, k, log_eta_scale_shared, acc_sum, acc_count), t
                        )
                        x_next, k = predictor_step(t, x_curr, k)
                        return x_next, k, log_eta_scale_shared, acc_sum, acc_count

                    x_final, _, log_eta_scale_shared_final, acc_sum_final, acc_count_final = jax.lax.fori_loop(
                        0,
                        timesteps,
                        level_body,
                        (x0, loop_key, log_eta_scale_shared0, jnp.float32(0.0), jnp.float32(0.0)),
                    )

                    log_eta_scales_out = jnp.full((timesteps,), log_eta_scale_shared_final, dtype=jnp.float32)
                    mala_acc_rate = acc_sum_final / jnp.maximum(acc_count_final, jnp.float32(1.0))
                    mala_eta_scale = jnp.exp(log_eta_scale_shared_final)

                act_final = jnp.clip(x_final, -1.0, 1.0)
                act_final = x_final if self.latent_action_space else act_final
                q1_mean, q1_var = self.agent.q(q1_params, obs_batch, act_final)
                q2_mean, q2_var = self.agent.q(q2_params, obs_batch, act_final)
                q = aggregate_q_fn(q1_mean, q1_var, q2_mean, q2_var)
                if self.use_reward_critic:
                    q = q * (jnp.float32(1.0) / jnp.float32(1.0 - self.gamma))
                return act_final, q, log_eta_scales_out, mala_acc_rate, mala_eta_scale

            def single_sampler(single_key: jax.Array, log_eta_scales_local: jax.Array):
                return mala_chain(single_key, log_eta_scales_local)

            def sample_with_particles_metrics(
                key: jax.Array,
                log_alpha: jax.Array,
                single: bool,
                single_sampler,
                log_eta_scales_in: jax.Array,
            ):
                key_sample, key_select, noise_key = jax.random.split(key, 3)
                if self.agent.num_particles == 1:
                    act, _, log_eta_scales_out, mala_acc_rate, mala_eta_scale = single_sampler(key_sample, log_eta_scales_in)
                else:
                    keys = jax.random.split(key_sample, self.agent.num_particles)
                    acts, qs, log_eta_scales_outs, mala_acc_rates, mala_eta_scales = jax.vmap(
                        lambda k: single_sampler(k, log_eta_scales_in)
                    )(keys)
                    act = select_action_from_particles(acts, qs, key_select)
                    log_eta_scales_out = jnp.mean(log_eta_scales_outs, axis=0)
                    mala_acc_rate = jnp.mean(mala_acc_rates, axis=0)
                    mala_eta_scale = jnp.mean(mala_eta_scales, axis=0)

                if not self.no_entropy_tuning:
                    act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.agent.noise_scale

                if single:
                    return act[0], log_eta_scales_out, mala_acc_rate, mala_eta_scale
                else:
                    return act, log_eta_scales_out, mala_acc_rate, mala_eta_scale

            def sample_with_particles_metrics_details(
                key: jax.Array,
                log_alpha: jax.Array,
                single: bool,
                single_sampler,
                log_eta_scales_in: jax.Array,
            ):
                key_sample, key_select, noise_key = jax.random.split(key, 3)

                if self.agent.num_particles == 1:
                    act, q, log_eta_scales_out, mala_acc_rate, mala_eta_scale = single_sampler(key_sample, log_eta_scales_in)
                    acts = act[None, ...]
                    qs = q[None, ...]
                    probs = jnp.ones_like(qs)
                else:
                    keys = jax.random.split(key_sample, self.agent.num_particles)
                    acts, qs, log_eta_scales_outs, mala_acc_rates, mala_eta_scales = jax.vmap(
                        lambda k: single_sampler(k, log_eta_scales_in)
                    )(keys)
                    log_eta_scales_out = jnp.mean(log_eta_scales_outs, axis=0)
                    mala_acc_rate = jnp.mean(mala_acc_rates, axis=0)
                    mala_eta_scale = jnp.mean(mala_eta_scales, axis=0)

                    if np.isinf(particle_selection_lambda):
                        idx = jnp.argmax(qs, axis=0)
                        probs = jax.nn.one_hot(idx, self.agent.num_particles).T
                    else:
                        logits = lambda_sel * qs
                        logits = logits - jnp.max(logits, axis=0, keepdims=True)
                        probs = jax.nn.softmax(logits, axis=0)

                    act = select_action_from_particles(acts, qs, key_select)

                if not self.no_entropy_tuning:
                    act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.agent.noise_scale

                if single:
                    return act[0], acts, probs, log_eta_scales_out, mala_acc_rate, mala_eta_scale
                else:
                    return act, acts, probs, log_eta_scales_out, mala_acc_rate, mala_eta_scale

            if return_details:
                return sample_with_particles_metrics_details(key, log_alpha, single, single_sampler, log_eta_scales_in)
            return sample_with_particles_metrics(key, log_alpha, single, single_sampler, log_eta_scales_in)

        def stateless_get_action_mala_particles_full(
            key: jax.Array,
            params,
            obs: jax.Array,
            aggregate_q_fn,
            randomize_q: bool = False,
        ):
            return stateless_get_action_mala_full(key, params, obs, aggregate_q_fn, randomize_q=randomize_q, return_details=True)

        def stateless_get_action_mala(
            key: jax.Array,
            params,
            obs: jax.Array,
            aggregate_q_fn,
            randomize_q: bool = False,
        ) -> jax.Array:
            act, log_eta_scales_out, _, _ = stateless_get_action_mala_full(
                key,
                params,
                obs,
                aggregate_q_fn,
                randomize_q=randomize_q,
            )
            return act, log_eta_scales_out

        def sample_action_with_agg_metrics(
            key: jax.Array,
            params,
            obs: jax.Array,
            aggregate_q_fn,
            randomize_q: bool = False,
        ):
            policy_params, log_alpha, q1_params, q2_params, log_eta_scales_in, _tfg_lambda_current, _value_params, _adv_second_moment_ema = params
            if self.agent.energy_mode and self.agent.mala_steps > 0:
                act, log_eta_scales_out, mala_acc_rate, mala_eta_scale = stateless_get_action_mala_full(
                    key,
                    params,
                    obs,
                    aggregate_q_fn,
                    randomize_q=randomize_q,
                )
                return act, log_eta_scales_out, mala_acc_rate, mala_eta_scale
            else:
                act, log_eta_scales_out = sample_action_with_agg(key, params, obs, aggregate_q_fn, randomize_q=randomize_q)
                return act, log_eta_scales_out, jnp.float32(jnp.nan), jnp.float32(jnp.nan)

        def sample_action_with_particle_details_metrics(
            key: jax.Array,
            params,
            obs: jax.Array,
            aggregate_q_fn,
            randomize_q: bool = False,
        ):
            if self.agent.energy_mode and self.agent.mala_steps > 0:
                return stateless_get_action_mala_particles_full(key, params, obs, aggregate_q_fn, randomize_q=randomize_q)
            act, log_eta_scales_out, mala_acc_rate, mala_eta_scale = sample_action_with_agg_metrics(
                key,
                params,
                obs,
                aggregate_q_fn,
                randomize_q=randomize_q,
            )
            single = obs.ndim == 1
            if single:
                acts = act[None, None, :]
                probs = jnp.ones((1, 1), dtype=jnp.float32)
                return act, acts, probs, log_eta_scales_out, mala_acc_rate, mala_eta_scale
            acts = act[None, ...]
            probs = jnp.ones((1, act.shape[0]), dtype=jnp.float32)
            return act, acts, probs, log_eta_scales_out, mala_acc_rate, mala_eta_scale

        def stateless_get_action_base(
            key: jax.Array,
            params,
            obs: jax.Array,
            aggregate_q_fn,
            randomize_q: bool = False,
        ) -> jax.Array:
            policy_params, log_alpha, q1_params, q2_params, log_eta_scales_in, _tfg_lambda_current, _value_params, _adv_second_moment_ema = params

            single = obs.ndim == 1
            if single:
                obs_batch = obs[None, :]
            else:
                obs_batch = obs

            shape = (*obs_batch.shape[:-1], self.agent.act_dim)

            def model_fn(t, x):
                # Scale base score by energy_multiplier (tempers the base distribution)
                return self.energy_multiplier * self.agent.policy(policy_params, obs_batch, x, t)

            def energy_model_fn(t, x):
                # Scale base energy by energy_multiplier (tempers the base distribution)
                return self.energy_multiplier * self.agent.energy_fn(policy_params, obs_batch, x, t)

            def base_sample(single_key: jax.Array):
                if randomize_q and self.q_critic_agg == "random":
                    sample_key, q_agg_key = jax.random.split(single_key)
                    critic_idx = jax.random.randint(q_agg_key, (), 0, 2)

                    def aggregate_q_local(
                        q1_mean: jax.Array,
                        q1_var: jax.Array,
                        q2_mean: jax.Array,
                        q2_var: jax.Array,
                    ) -> jax.Array:
                        return jax.lax.select(critic_idx == 0, q1_mean, q2_mean)

                else:
                    sample_key = single_key
                    aggregate_q_local = aggregate_q_fn
                if self.agent.energy_mode:
                    act = self.agent.mala_sample(sample_key, model_fn, energy_model_fn, shape)
                else:
                    act = self.agent.diffusion.p_sample(
                        sample_key,
                        model_fn,
                        shape,
                        deterministic=self.ddim_predictor,
                    )
                act = act if self.latent_action_space else jnp.clip(act, -1.0, 1.0)
                q1_mean, q1_var = self.agent.q(q1_params, obs_batch, act)
                q2_mean, q2_var = self.agent.q(q2_params, obs_batch, act)
                q = aggregate_q_local(q1_mean, q1_var, q2_mean, q2_var)
                if self.use_reward_critic:
                    q = q * (jnp.float32(1.0) / jnp.float32(1.0 - self.gamma))
                return act, q

            def single_sampler(single_key: jax.Array, log_eta_scales_local: jax.Array):
                act, q = base_sample(single_key)
                return act, q, log_eta_scales_local

            return sample_with_particles(key, log_alpha, single, single_sampler, log_eta_scales_in)

        def stateless_get_action_env(
            key: jax.Array,
            params,
            obs: jax.Array,
        ) -> jax.Array:
            return sample_action_with_agg(key, params, obs, aggregate_q, randomize_q=(self.q_critic_agg == "random"))

        def stateless_get_deterministic_action_env(
            params,
            obs: jax.Array,
        ) -> jax.Array:
            key = random_key_from_data(obs)
            return stateless_get_action_env(key, params, obs)

        self._implement_common_behavior(stateless_update, stateless_get_action_env, stateless_get_deterministic_action_env)
        self._update_supervised = jax.jit(stateless_supervised_update)

        # KL(π_tilt || π_0) estimation for mirror descent monitoring
        # KL = E_{a~π_tilt}[η Q(s,a)] - log E_{a~π_0}[e^{η Q(s,a)}]
        @functools.partial(jax.jit, static_argnums=(3,))
        def stateless_estimate_md_kl(
            key: jax.Array,
            params,
            obs: jax.Array,
            n_samples: int = 16,
        ) -> jax.Array:
            """Estimate KL(π_tilt || π_0) for a batch of observations."""
            policy_params, log_alpha, q1_params, q2_params, log_eta_scales_in, tfg_lambda_current, value_params, adv_second_moment_ema = params

            single = obs.ndim == 1
            if single:
                obs_batch = obs[None, :]
            else:
                obs_batch = obs

            shape = (*obs_batch.shape[:-1], self.agent.act_dim)
            batch_size = obs_batch.shape[0]

            def model_fn(t, x):
                return self.energy_multiplier * self.agent.policy(policy_params, obs_batch, x, t)

            def energy_model_fn(t, x):
                return self.energy_multiplier * self.agent.energy_fn(policy_params, obs_batch, x, t)

            # Sample from base policy π_0 (unguided) and evaluate Q
            def sample_base_with_q(sample_key: jax.Array):
                if self.agent.energy_mode:
                    act = self.agent.mala_sample(sample_key, model_fn, energy_model_fn, shape)
                else:
                    act = self.agent.diffusion.p_sample(
                        sample_key,
                        model_fn,
                        shape,
                        deterministic=self.ddim_predictor,
                    )
                act = act if self.latent_action_space else jnp.clip(act, -1.0, 1.0)
                q1_mean, _ = self.agent.q(q1_params, obs_batch, act)
                q2_mean, _ = self.agent.q(q2_params, obs_batch, act)
                q = aggregate_q_mean(q1_mean, q2_mean)
                if self.use_reward_critic:
                    q = q * (jnp.float32(1.0) / jnp.float32(1.0 - self.gamma))
                return q  # [batch_size]

            # Sample from tilted policy π_tilt (guided) and evaluate Q
            def sample_tilted_with_q(sample_key: jax.Array):
                # Use the guided sampling path
                act, _ = sample_action_with_agg(
                    sample_key,
                    params,
                    obs_batch,
                    aggregate_q,
                    randomize_q=False,
                )
                # Evaluate Q on the tilted sample
                q1_mean, _ = self.agent.q(q1_params, obs_batch, act)
                q2_mean, _ = self.agent.q(q2_params, obs_batch, act)
                q = aggregate_q_mean(q1_mean, q2_mean)
                if self.use_reward_critic:
                    q = q * (jnp.float32(1.0) / jnp.float32(1.0 - self.gamma))
                return q  # [batch_size]

            key_base, key_tilted = jax.random.split(key)

            # Sample n_samples from each distribution
            base_keys = jax.random.split(key_base, n_samples)
            tilted_keys = jax.random.split(key_tilted, n_samples)

            # Q values from base policy samples: [n_samples, batch_size]
            q_base = jax.vmap(sample_base_with_q)(base_keys)
            # Q values from tilted policy samples: [n_samples, batch_size]
            q_tilted = jax.vmap(sample_tilted_with_q)(tilted_keys)

            # First term: E_{a~π_tilt}[η Q(s,a)] = mean over samples of (η * Q)
            # Shape: [batch_size]
            term1 = tfg_lambda_current * jnp.mean(q_tilted, axis=0)

            # Second term: log E_{a~π_0}[e^{η Q(s,a)}] = logsumexp(η Q) - log(n_samples)
            # Shape: [batch_size]
            term2 = jax.scipy.special.logsumexp(tfg_lambda_current * q_base, axis=0) - jnp.log(jnp.float32(n_samples))

            # KL = term1 - term2, averaged over batch
            kl_per_obs = term1 - term2
            kl_mean = jnp.mean(kl_per_obs)

            return kl_mean

        self._estimate_md_kl = stateless_estimate_md_kl

    def get_policy_params(self):
        return (
            self.state.params.policy,
            self.state.params.log_alpha,
            self.state.params.q1,
            self.state.params.q2,
            self.state.log_eta_scales,
            self.state.tfg_lambda,
            self.state.value_params,
            self.state.advantage_second_moment_ema,
        )

    def get_policy_params_to_save(self):
        return (
            self.state.params.target_poicy,
            self.state.params.log_alpha,
            self.state.params.q1,
            self.state.params.q2,
            self.state.log_eta_scales,
            self.state.tfg_lambda,
            self.state.value_params,
            self.state.advantage_second_moment_ema,
        )

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        if not hasattr(state, "log_eta_scales"):
            timesteps = self.agent.num_timesteps
            init_eta_scale = jnp.maximum(jnp.float32(self.mala_init_eta_scale), jnp.float32(1e-8))
            init_log_eta_scale = jnp.log(init_eta_scale)
            state = Diffv2TrainState(
                params=state.params,
                opt_state=state.opt_state,
                step=state.step,
                entropy=state.entropy,
                running_mean=state.running_mean,
                running_std=state.running_std,
                log_eta_scales=jnp.full((timesteps,), init_log_eta_scale, dtype=jnp.float32),
                tfg_lambda=jnp.float32(self.tfg_lambda),
                value_params=getattr(state, 'value_params', None),
                advantage_second_moment_ema=getattr(state, 'advantage_second_moment_ema', jnp.float32(1.0)),
            )
        elif not hasattr(state, "tfg_lambda"):
            state = Diffv2TrainState(
                params=state.params,
                opt_state=state.opt_state,
                step=state.step,
                entropy=state.entropy,
                running_mean=state.running_mean,
                running_std=state.running_std,
                log_eta_scales=state.log_eta_scales,
                tfg_lambda=jnp.float32(self.tfg_lambda),
                value_params=getattr(state, 'value_params', None),
                advantage_second_moment_ema=getattr(state, 'advantage_second_moment_ema', jnp.float32(1.0)),
            )
        elif (not self.mala_per_level_eta) and hasattr(state, "log_eta_scales"):
            # Enforce shared eta-scale invariant when running in shared mode.
            timesteps = self.agent.num_timesteps
            shared = jnp.asarray(state.log_eta_scales)[0]
            state = Diffv2TrainState(
                params=state.params,
                opt_state=state.opt_state,
                step=state.step,
                entropy=state.entropy,
                running_mean=state.running_mean,
                running_std=state.running_std,
                log_eta_scales=jnp.full((timesteps,), shared, dtype=jnp.float32),
                tfg_lambda=state.tfg_lambda if hasattr(state, "tfg_lambda") else jnp.float32(self.tfg_lambda),
                value_params=getattr(state, 'value_params', None),
                advantage_second_moment_ema=getattr(state, 'advantage_second_moment_ema', jnp.float32(1.0)),
            )
        self.state = jax.device_put(state)

    def get_current_tfg_lambda(self) -> float:
        return float(self.state.tfg_lambda)

    def estimate_md_kl(self, key: jax.Array, obs: np.ndarray, n_samples: int = 16) -> float:
        """Estimate KL(π_tilt || π_0) for given observations.
        
        Uses the formula: KL = E_{a~π_tilt}[η Q(s,a)] - log E_{a~π_0}[e^{η Q(s,a)}]
        Returns 0.0 if tfg_lambda is 0 (no tilting).
        """
        if float(self.state.tfg_lambda) == 0.0:
            return 0.0
        kl = self._estimate_md_kl(key, self.get_policy_params(), jnp.asarray(obs), n_samples)
        return float(kl)

    def set_tfg_lambda(self, new_tfg_lambda: float) -> None:
        self.state = self.state._replace(tfg_lambda=jnp.float32(new_tfg_lambda))

    def get_effective_hparams(self) -> dict:
        return {
            "lr_policy_effective": float(self.lr_policy),
            "lr_q_effective": float(self.lr_q),
            "td_actions": int(self.td_actions),
        }

    def save_policy(self, path: str) -> None:
        policy = jax.device_get(self.get_policy_params_to_save())
        with open(path, "wb") as f:
            pickle.dump(policy, f)

    def get_action(self, key: jax.Array, obs: np.ndarray) -> np.ndarray:
        out = self._get_action(key, self.get_policy_params_to_save(), obs)
        if isinstance(out, tuple) and len(out) == 2:
            action, log_eta_scales = out
            self.state = self.state._replace(log_eta_scales=log_eta_scales)
        else:
            action = out
        return np.asarray(action)

    def update_supervised(self, key: jax.Array, data: Experience) -> Metric:
        self.state, info = self._update_supervised(key, self.state, data)
        return {k: float(v) for k, v in info.items() if not k.startswith('hist')}, {k: v for k, v in info.items() if k.startswith('hist')}