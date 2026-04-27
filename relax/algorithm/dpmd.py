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
    q: tuple  # tuple of N optax.OptState, one per Q network
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
    tfg_eta: jax.Array
    # DSAC-T state: moving averages for omega scaling and adaptive clipping
    dsac_omega_ema: float = 1.0  # EMA of batch-mean predicted variance
    dsac_b_ema: float = 1.0      # EMA of batch-mean predicted std (for adaptive clipping)
    # Normalized advantage guidance state
    value_params: hk.Params = None  # V(s) network params (optional)
    target_value_params: hk.Params = None  # Polyak-averaged V target (for td_value_training)
    advantage_second_moment_ema: float = 1.0  # EMA of E[A^2] where A = Q - V
    advantage_third_moment_ema: float = 0.0   # EMA of E[A^3] (skewness), for dist_shift_eta
    dist_shift_covariance_ema: float = 0.0    # EMA of E[D_psi * A] (covariance), for dist_shift_eta
    dist_shift_shape_ema: float = -1.0        # EMA of s₂ = (2γc + κ₃) / v^(3/2), dimensionless shape
    dist_shift_coeff_ema: float = -1.0        # EMA of B = 2γc + κ₃, raw η²-coefficient (alt to shape)
    dist_shift_shape3_ema: float = 0.0       # EMA of s₃ = τ / v², third-order dimensionless shape
    frozen_q: tuple = None  # tuple of N frozen Q params (or None) for soft-PI
    train_policy: float = 1.0  # 0.0 during Q-only phase of soft-PI
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
    # Per-slot mean-vs-min Q-aggregation selector used only when the algo is
    # instantiated with q_critic_agg == "vmap_mean_min". 0.0 => mean, 1.0 =>
    # min, interpolated linearly in between. Threaded through both the
    # training-time Q signal (stateless_update's aggregate_q_n) and the
    # sampling-time aggregator (via the params tuple into sample_action_with_agg),
    # so the vmap path mirrors the non-vmap 'mean'/'min' paths exactly.
    q_critic_agg_idx: jax.Array = 0.0

class DPMD(Algorithm):

    @staticmethod
    def _make_lr_schedule(schedule_type: str, init_lr: float, end_lr: float,
                          steps: int, begin: int = 0):
        """Create an optax LR schedule. Supports 'linear', 'cosine', and 'log_linear'."""
        if schedule_type == "log_linear":
            # Geometric decay: lr(t) = init_lr^(1 - progress) * end_lr^(progress)
            # Same as linear interpolation in log-space.
            _init = float(init_lr)
            _end = float(end_lr)
            _steps = int(steps)
            _begin = int(begin)
            def _log_linear_schedule(count):
                t = jnp.clip((count - _begin) / max(_steps, 1), 0.0, 1.0)
                if _init > 0 and _end > 0:
                    return _init ** (1.0 - t) * _end ** t
                # Fallback to linear if either endpoint is 0
                return (1.0 - t) * _init + t * _end
            return _log_linear_schedule
        if schedule_type == "cosine":
            return optax.schedules.cosine_decay_schedule(
                init_value=init_lr,
                decay_steps=steps,
                alpha=end_lr / max(init_lr, 1e-30),
            )
        # Default: linear
        return optax.schedules.linear_schedule(
            init_value=init_lr,
            end_value=end_lr,
            transition_steps=steps,
            transition_begin=begin,
        )

    @staticmethod
    def _lr_at_step(schedule_type: str, init_lr: jax.Array, end_lr: float,
                    steps: int, begin: int, count: jax.Array) -> jax.Array:
        init_lr = jnp.asarray(init_lr, dtype=jnp.float32)
        end_lr = jnp.asarray(end_lr, dtype=jnp.float32)
        count = jnp.asarray(count, dtype=jnp.float32)
        if schedule_type == "constant" or steps <= 0:
            return init_lr
        if schedule_type == "cosine":
            t = jnp.clip(count / max(int(steps), 1), 0.0, 1.0)
            decay = jnp.float32(0.5) * (jnp.float32(1.0) + jnp.cos(jnp.pi * t))
            return end_lr + (init_lr - end_lr) * decay
        t = jnp.clip((count - int(begin)) / max(int(steps), 1), 0.0, 1.0)
        if schedule_type == "log_linear":
            log_lr = jnp.exp(
                (jnp.float32(1.0) - t) * jnp.log(jnp.maximum(init_lr, jnp.float32(1e-30)))
                + t * jnp.log(jnp.maximum(end_lr, jnp.float32(1e-30)))
            )
            linear_lr = (jnp.float32(1.0) - t) * init_lr + t * end_lr
            return jnp.where((init_lr > 0) & (end_lr > 0), log_lr, linear_lr)
        return (jnp.float32(1.0) - t) * init_lr + t * end_lr

    def _policy_lr_at_step(self, init_lr: jax.Array, step: jax.Array) -> jax.Array:
        return self._lr_at_step(
            self._policy_lr_schedule_type,
            init_lr,
            self._policy_lr_schedule_end,
            self._policy_lr_schedule_steps,
            self._policy_lr_schedule_begin,
            step,
        )

    @staticmethod
    def _optimizer_count(opt_state) -> jax.Array:
        if hasattr(opt_state, "count"):
            return opt_state.count
        if isinstance(opt_state, tuple) and len(opt_state) > 0 and hasattr(opt_state[0], "count"):
            return opt_state[0].count
        raise TypeError(f"Unsupported optimizer state type for count extraction: {type(opt_state)}")

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
        q_critic_agg_idx: float = 0.0,
        entropic_risk_beta: float = 1.0,
        fix_q_norm_bug: bool = False,
        tfg_eta: float = 0.0,
        tfg_eta_schedule: str = "constant",
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
        mala_recurrence_cap: bool = False,
        mala_guided_predictor: bool = False,
        mala_predictor_first: bool = False,
        ddim_predictor: bool = False,
        latent_action_space: bool = False,
        q_td_huber_width: float = float("inf"),
        decorrelated_q_batches: bool = False,
        q_bootstrap_agg: str = "min",
        langevin_q_noise: bool = False,
        td_use_target_policy: bool = False,
        batch_independent_guidance: bool = False,
        guidance_strength_multiplier: float = 1.0,
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
        advantage_ema_tau: float = 0.0005,
        shape_ema_tau: float = 0.0001,  # slower EMA for dimensionless shape s₂ = (2γc+κ₃)/v^(3/2)
        initial_advantage_second_moment_ema: float = 1.0,
        initial_dist_shift_shape_ema: float = -1.0,
        shape3_ema_tau: float = 0.00005,  # even slower EMA for s₃ = τ/v² (third-order shape)
        # Policy loss type
        policy_loss_type: str = "eps_mse",  # "eps_mse" or "ula_kl"
        policy_loss_reduction: str = "mean",  # "mean" or "sum" (sum over action dims, mean over batch)
        # Soft policy iteration mode
        soft_pi_mode: bool = False,
        # Optimizer choice and LR schedule
        optimizer_type: str = "adam",  # "adam" or "adamw"
        weight_decay: float = 1e-4,
        lr_policy_schedule_type: str = "linear",  # "constant", "linear", "cosine", or "log_linear"
        lr_q_schedule_type: str = "constant",  # "constant", "linear", "cosine", or "log_linear"
        lr_q_schedule_end: float | None = None,  # end LR for Q schedule (None → use lr_schedule_end)
        # KL budget and distribution-shift adaptive eta
        kl_budget: float | None = None,
        dist_shift_eta: bool = False,
        one_step_dist_shift_eta: bool = False,
        two_step_dist_shift_eta: bool = False,
        direct_eta_coeff_ema: bool = False,
        # Entropic-risk / exponential-moment critic parameterization:
        #   "none" = standard Q critic
        #   "u"    = raw output is u = η·Q_η, trained with pseudo-loss
        #   "w"    = raw output is W, trained with direct MSE Bellman regression
        entropic_critic_param: str = "none",
        td_value_training: bool = False,
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

        # Build optimizer factory based on optimizer_type
        optimizer_type = str(optimizer_type).lower()
        weight_decay = float(weight_decay)
        def _make_optimizer(lr_or_schedule):
            if optimizer_type == "adamw":
                return optax.adamw(learning_rate=lr_or_schedule, weight_decay=weight_decay)
            return optax.adam(learning_rate=lr_or_schedule)
        def _make_optimizer_unscaled():
            # Returns an optimizer whose update is NOT multiplied by -lr.
            # Caller must apply `-state.lr_X * update` manually before
            # optax.apply_updates. Lets the LR be a per-seed runtime value
            # (vmappable) instead of baked into the chain at construction.
            if optimizer_type == "adamw":
                return optax.chain(
                    optax.scale_by_adam(),
                    optax.add_decayed_weights(weight_decay),
                )
            return optax.scale_by_adam()
        self._make_optimizer = _make_optimizer

        # Q optimizer (with optional schedule)
        lr_q_schedule_type = str(lr_q_schedule_type).lower()
        lr_q_schedule_end_eff = float(lr_q_schedule_end if lr_q_schedule_end is not None else lr_schedule_end)
        if lr_q_schedule_type != "constant" and lr_schedule_steps > 0:
            q_lr_sched = self._make_lr_schedule(
                lr_q_schedule_type, lr_q_eff, lr_q_schedule_end_eff,
                int(lr_schedule_steps), int(lr_schedule_begin),
            )
            self.optim = _make_optimizer(q_lr_sched)
            self._q_lr_runtime = False  # LR baked into the schedule chain
        else:
            # Constant LR path: use unscaled optimizer and apply lr at update time.
            # This makes state.lr_q a per-seed runtime value under vmap.
            self.optim = _make_optimizer_unscaled()
            self._q_lr_runtime = True
        self.latent_action_space = bool(latent_action_space)
        self.q_td_huber_width = float(q_td_huber_width)
        self.decorrelated_q_batches = bool(decorrelated_q_batches)
        self.q_bootstrap_agg = str(q_bootstrap_agg)
        self.langevin_q_noise = bool(langevin_q_noise)
        self.td_use_target_policy = bool(td_use_target_policy)
        self.batch_independent_guidance = bool(batch_independent_guidance)
        self.guidance_strength_multiplier = float(guidance_strength_multiplier)

        critic_grad_modifier = str(critic_grad_modifier)
        if bool(natural_gradient_critic) and critic_grad_modifier == "none":
            critic_grad_modifier = "natgrad"
        if critic_grad_modifier not in {"none", "variance_scaled", "natgrad"}:
            raise ValueError(f"Unknown critic_grad_modifier: {critic_grad_modifier}")
        self.critic_grad_modifier = critic_grad_modifier

        init_eta_scale = jnp.maximum(jnp.float32(mala_init_eta_scale), jnp.float32(1e-8))
        init_log_eta_scale = jnp.log(init_eta_scale)
        # Policy optimizer (with optional schedule)
        lr_policy_schedule_type = str(lr_policy_schedule_type).lower()
        if lr_policy_schedule_type != "constant" and lr_schedule_steps > 0:
            self._policy_lr_schedule_type = lr_policy_schedule_type
        else:
            self._policy_lr_schedule_type = "constant"
        self._policy_lr_schedule_end = float(lr_schedule_end)
        self._policy_lr_schedule_steps = int(lr_schedule_steps)
        self._policy_lr_schedule_begin = int(lr_schedule_begin)
        self.policy_optim = _make_optimizer_unscaled()
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
            # Find the first Linear layer weight in Q network params.
            # Haiku FlatMaps use slash-separated keys like
            # 'q_net/linear', 'q_net/linear_1', etc.  We want exactly
            # the one ending in '/linear' (not '/linear_1' etc.) which
            # is the input layer with shape (obs+act, hidden).
            first_w = None
            for k, v in params.q[0].items():
                if k.endswith('/linear') and isinstance(v, dict) and 'w' in v:
                    w = v['w']
                    if hasattr(w, 'shape'):
                        first_w = w
                        break
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

        self.soft_pi_mode = bool(soft_pi_mode)
        # Cached for rebuilding states (single-seed init + vmap init).
        self._init_log_eta_scale = init_log_eta_scale
        self._init_tfg_eta = float(tfg_eta)
        self._timesteps = int(timesteps)
        self._td_value_training = bool(td_value_training)

        # Per-seed hp attrs needed by _build_initial_state; the canonical
        # assignments below re-set the same values with no behavior change.
        self.advantage_ema_tau = float(advantage_ema_tau)
        self.shape_ema_tau = float(shape_ema_tau)
        self.initial_advantage_second_moment_ema = float(initial_advantage_second_moment_ema)
        self.initial_dist_shift_shape_ema = float(initial_dist_shift_shape_ema)
        self.initial_dist_shift_coeff_ema = self.initial_dist_shift_shape_ema * (max(self.initial_advantage_second_moment_ema, 0.0) ** 1.5)
        self.kl_budget = float(kl_budget) if kl_budget is not None else None
        self.x0_hat_clip_radius = float(x0_hat_clip_radius)
        self.mala_adapt_rate = float(mala_adapt_rate)
        self.shape3_ema_tau = float(shape3_ema_tau)
        self.q_critic_agg_idx = float(q_critic_agg_idx)

        self.state = self._build_initial_state(params, value_params_init, value_opt_state_init)
        self.use_ema = use_ema
        self.use_reweighting = use_reweighting
        self.use_reward_critic = use_reward_critic
        self.pure_bc_training = pure_bc_training
        self.off_policy_td = off_policy_td
        self.no_entropy_tuning = no_entropy_tuning
        self.q_critic_agg = q_critic_agg
        self.entropic_risk_beta = float(entropic_risk_beta)
        self.fix_q_norm_bug = fix_q_norm_bug
        self.tfg_eta = float(tfg_eta)
        self.tfg_eta_schedule = tfg_eta_schedule
        self.tfg_recur_steps = tfg_recur_steps
        self.particle_selection_lambda = particle_selection_lambda
        self.x0_hat_clip_radius = float(x0_hat_clip_radius)
        self.supervised_steps = int(supervised_steps)
        self.single_q_network = single_q_network
        self.mala_per_level_eta = bool(mala_per_level_eta)
        self.mala_adapt_rate = float(mala_adapt_rate)
        self.mala_init_eta_scale = float(mala_init_eta_scale)
        self.mala_recurrence_cap = bool(mala_recurrence_cap)
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
        self.shape_ema_tau = float(shape_ema_tau)

        # KL budget mode: on-policy EMA, V trained on (s', a')
        self.kl_budget = float(kl_budget) if kl_budget is not None else None
        self.dist_shift_eta = bool(dist_shift_eta)
        self.one_step_dist_shift_eta = bool(one_step_dist_shift_eta)
        self.two_step_dist_shift_eta = bool(two_step_dist_shift_eta)
        self.direct_eta_coeff_ema = bool(direct_eta_coeff_ema)
        self.shape3_ema_tau = float(shape3_ema_tau)
        self.td_value_training = bool(td_value_training)
        self.on_policy_ema = (self.kl_budget is not None)

        # Entropic-risk / exponential-moment critic.
        # "u" mode: raw output is u = η·Q_η; W = exp(u).
        # "w" mode: raw output is W directly; Q_η = log(max(W,ε))/η.
        # η is identified with reward_scale.
        # Discounted entropic-risk with a stationary critic is approximate
        # (not exact for γ < 1) in both modes.
        self.entropic_critic_param = str(entropic_critic_param).lower()
        if self.entropic_critic_param not in ("none", "u", "w"):
            raise ValueError(f"entropic_critic_param must be 'none', 'u', or 'w', got '{self.entropic_critic_param}'")
        self.use_entropic_q = (self.entropic_critic_param != "none")
        self._entropic_u_mode = (self.entropic_critic_param == "u")
        self._entropic_w_mode = (self.entropic_critic_param == "w")
        if self.use_entropic_q:
            if self.agent.distributional_critic:
                raise ValueError("--entropic_critic_param is incompatible with --distributional_critic")
            # Small positive ε for clipping W in target-side Bellman recursion (w-mode)
            self._entropic_w_eps = 1e-6

        # Policy loss type: "eps_mse" (standard) or "ula_kl" (ULA-step KL weighting)
        self.policy_loss_type = str(policy_loss_type)
        self.policy_loss_reduction = str(policy_loss_reduction)  # "mean" or "sum"
        _policy_loss_key_map = {
            "eps_mse": "losses/Policy_epsilon_MSE",
            "ula_kl": "losses/Policy_ULA_KL",
            "guided_ula_kl": "losses/Policy_guided_ULA_KL",
            "e2e_guided_ula_kl": "losses/Policy_e2e_guided_ULA_KL",
        }
        self.policy_loss_key = _policy_loss_key_map.get(self.policy_loss_type, f"losses/Policy_{self.policy_loss_type}")

        # Store schedule arrays for logging (SNR x-axis in line plots)
        B_sched = self.agent.diffusion.beta_schedule()
        self._alphas_cumprod = np.asarray(B_sched.alphas_cumprod)  # [T]
        self._snr = self._alphas_cumprod / np.maximum(1.0 - self._alphas_cumprod, 1e-8)

        if self.tfg_eta_schedule == "linear":
            idx = jnp.arange(timesteps, dtype=jnp.float32)
            denom = jnp.maximum(timesteps - 1, 1)
            t_levels = 1.0 - idx / denom

            def lambda_for_step(t_idx: jax.Array, tfg_eta_current: jax.Array) -> jax.Array:
                t_next = jnp.maximum(t_idx - 1, 0)
                return tfg_eta_current * t_levels[t_next]

        elif self.tfg_eta_schedule == "snr":
            # lambda_t = lambda * alpha_bar_t  =  lambda * SNR/(1+SNR)
            # Gives c_t = lambda*(1 - alpha_bar_t), bounded in [0, lambda].
            # Hessian-term loss contribution is ~constant across noise levels.
            B_init = self.agent.diffusion.beta_schedule()
            snr_scales = B_init.alphas_cumprod  # [T], ~1 at t=0 (clean), ~0 at t=T-1 (noisy)

            def lambda_for_step(t_idx: jax.Array, tfg_eta_current: jax.Array) -> jax.Array:
                t_next = jnp.maximum(t_idx - 1, 0)
                return tfg_eta_current * snr_scales[t_next]

        else:
            ones = jnp.ones((timesteps,), dtype=jnp.float32)

            def lambda_for_step(t_idx: jax.Array, tfg_eta_current: jax.Array) -> jax.Array:
                t_next = jnp.maximum(t_idx - 1, 0)
                return tfg_eta_current * ones[t_next]

        # Aggregation of twin Qs for *signals* (reweighting, tilt, logging, etc.).
        # For distributional critics, Q returns (mean, var). We aggregate means for signals.
        # TD targets use q_bootstrap_agg for distributional aggregation.

        # --- Entropic-Q readout for downstream consumers (signals, guidance).
        # u-mode: raw output is u = η·Q_η  →  Q_η = u / η
        # w-mode: raw output is W           →  Q_η = log(max(W, ε)) / η
        # η is the *dynamic* tfg_eta (= state.tfg_eta), set via _entropic_eta_ref
        # at the start of each traced function.  The TD backup path uses raw
        # network output and does NOT go through this.
        #
        # Mutable closure ref: set to tfg_eta_current at each JIT entry point
        # so that _u_to_q picks up the traced value.
        _entropic_eta_ref = [jnp.float32(self.reward_scale)]  # default; overwritten per-call
        if self._entropic_u_mode:
            def _u_to_q(val):
                return val / _entropic_eta_ref[0]
        elif self._entropic_w_mode:
            _w_eps = jnp.float32(self._entropic_w_eps)
            def _u_to_q(val):
                return jnp.log(jnp.maximum(val, _w_eps)) / _entropic_eta_ref[0]
        else:
            def _u_to_q(val):
                return val

        # --- N-ary Q aggregation helpers (used by sampling and update) ---
        def hard_min_q_n(q_means, q_vars=None):
            """Element-wise min across N Q means."""
            q = q_means[0]
            for m in q_means[1:]:
                q = jnp.minimum(q, m)
            return _u_to_q(q)

        def make_aggregate_q_fn(q_critic_agg_idx_hp=None):
            """Build the N-ary aggregation used by sampling functions.

            Returns Q-like values (u/η when entropic_q is on).

            When q_critic_agg == "vmap_mean_min", the returned function closes
            over ``q_critic_agg_idx_hp`` (a JAX scalar) and blends mean/min
            per slot: (1-idx)*mean + idx*min. All other Python-level branches
            are unchanged. Callers that have the per-slot value in scope
            (either via state.q_critic_agg_idx or the params tuple) pass it
            here so the sampling path matches the training path exactly.
            """
            def aggregate_q_fn(q_means, q_vars):
                n = len(q_means)
                if q_critic_agg == "vmap_mean_min":
                    mean_val = sum(q_means) * jnp.float32(1.0 / n)
                    min_q = q_means[0]
                    for m in q_means[1:]:
                        min_q = jnp.minimum(min_q, m)
                    if q_critic_agg_idx_hp is None:
                        # No per-slot idx in scope: fall back to 'min' so the
                        # mode is still a defined function. In practice every
                        # call site that uses vmap_mean_min threads the idx.
                        return _u_to_q(min_q)
                    idx = jnp.clip(jnp.float32(q_critic_agg_idx_hp), 0.0, 1.0)
                    return _u_to_q((1.0 - idx) * mean_val + idx * min_q)
                if q_critic_agg == "precision":
                    eps = jnp.float32(1e-6)
                    precisions = [jnp.reciprocal(jax.lax.stop_gradient(v) + eps) for v in q_vars]
                    total_prec = sum(precisions)
                    return _u_to_q(sum(p * m for p, m in zip(precisions, q_means)) / total_prec)
                elif q_critic_agg == "min":
                    return hard_min_q_n(q_means)  # already applies _u_to_q
                elif q_critic_agg == "mean":
                    return _u_to_q(sum(q_means) * jnp.float32(1.0 / n))
                elif q_critic_agg == "max":
                    q = q_means[0]
                    for m in q_means[1:]:
                        q = jnp.maximum(q, m)
                    return _u_to_q(q)
                elif q_critic_agg == "entropic":
                    _ent_beta = jnp.float32(self.entropic_risk_beta)
                    _use_mean_fallback = jnp.abs(_ent_beta) < 1e-6
                    mean_val = sum(q_means) * jnp.float32(1.0 / n)
                    stacked = jnp.stack([_ent_beta * m for m in q_means], axis=0)
                    ent_val = (jnp.float32(1.0) / _ent_beta) * (jax.scipy.special.logsumexp(stacked, axis=0) - jnp.log(jnp.float32(n)))
                    return _u_to_q(jnp.where(_use_mean_fallback, mean_val, ent_val))
                else:
                    # Default fallback (random, unknown): use min
                    return hard_min_q_n(q_means)  # already applies _u_to_q
            return aggregate_q_fn

        # Default aggregator without a bound idx. Used only by call sites
        # that do not need per-slot behavior (e.g. the outer stateless_get_action
        # entry point, which then passes params into sample_action_with_agg
        # where the per-slot rebind happens). Any site that needs the per-slot
        # variant under vmap_mean_min must build a fresh aggregator with
        # make_aggregate_q_fn(idx_hp).
        aggregate_q_fn_outer = make_aggregate_q_fn(None)

        if np.isinf(particle_selection_lambda):
            def select_action_from_particles(acts: jax.Array, qs: jax.Array, key: jax.Array):
                q_best_ind = jnp.argmax(qs, axis=0, keepdims=True)
                act = jnp.take_along_axis(acts, q_best_ind[..., None], axis=0).squeeze(axis=0)
                q_sel = jnp.take_along_axis(qs, q_best_ind, axis=0).squeeze(axis=0)
                return act, q_sel
        else:
            lambda_sel = jnp.array(particle_selection_lambda, dtype=jnp.float32)

            def select_action_from_particles(acts: jax.Array, qs: jax.Array, key: jax.Array):
                logits = lambda_sel * qs
                logits = logits - jnp.max(logits, axis=0, keepdims=True)
                idx = jax.random.categorical(key, logits, axis=0)
                idx = idx[None, :]
                act = jnp.take_along_axis(acts, idx[..., None], axis=0).squeeze(axis=0)
                q_sel = jnp.take_along_axis(qs, idx, axis=0).squeeze(axis=0)
                return act, q_sel

        def sample_with_particles(
            key: jax.Array,
            log_alpha: jax.Array,
            single: bool,
            single_sampler,
            log_eta_scales_in: jax.Array,
        ):
            key_sample, key_select, noise_key = jax.random.split(key, 3)
            if self.agent.num_particles == 1:
                act, q, log_eta_scales_out = single_sampler(key_sample, log_eta_scales_in)
            else:
                keys = jax.random.split(key_sample, self.agent.num_particles)
                acts, qs, log_eta_scales_outs = jax.vmap(lambda k: single_sampler(k, log_eta_scales_in))(keys)
                act, q = select_action_from_particles(acts, qs, key_select)
                log_eta_scales_out = jnp.mean(log_eta_scales_outs, axis=0)

            if not self.no_entropy_tuning:
                act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.agent.noise_scale

            if single:
                return act[0], q[0], log_eta_scales_out
            else:
                return act, q, log_eta_scales_out

        def sample_action_with_agg(
            key: jax.Array,
            params,
            obs: jax.Array,
            aggregate_q_fn,
            randomize_q: bool = False,
        ) -> jax.Array:
            policy_params, log_alpha, q_params_tuple, log_eta_scales_in, tfg_eta_current, value_params, adv_second_moment_ema, target_value_params, reward_scale_hp, x0_hat_clip_radius_hp, mala_adapt_rate_hp, guidance_mult_hp, gamma_hp, q_critic_agg_idx_hp = params
            _entropic_eta_ref[0] = tfg_eta_current
            # Rebind aggregate_q_fn so the sampling path respects the per-slot
            # q_critic_agg_idx when the algo is in vmap_mean_min mode -- but
            # ONLY for the policy-side tilting aggregator. Callers that pin
            # aggregation to hard_min_q_n (e.g. stateless_update's TD-target
            # action sampling at line 2043, and sample_action_with_agg_metrics
            # which forwards the caller's hard_min_q_n) are intentionally
            # asking for min because the TD target always uses min regardless
            # of q_critic_agg. Detect that by identity: only rebind if the
            # caller handed us the no-idx-bound aggregate_q_fn_outer (the
            # neutral tilting aggregator).
            if q_critic_agg == "vmap_mean_min" and aggregate_q_fn is aggregate_q_fn_outer:
                aggregate_q_fn = make_aggregate_q_fn(q_critic_agg_idx_hp)
            if self.agent.energy_mode and self.agent.mala_steps > 0:
                return stateless_get_action_mala(key, params, obs, aggregate_q_fn, randomize_q=randomize_q)
            elif self.tfg_recur_steps > 0:
                return stateless_get_action_tfg_recur(key, params, obs, aggregate_q_fn, randomize_q=randomize_q)
            else:
                def _do_tfg(_):
                    return stateless_get_action_tfg_recur(key, params, obs, aggregate_q_fn, randomize_q=randomize_q)

                def _do_base(_):
                    return stateless_get_action_base(key, params, obs, aggregate_q_fn, randomize_q=randomize_q)

                return jax.lax.cond(
                    tfg_eta_current != jnp.float32(0.0),
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
            q_params = state.params.q          # tuple of N Q params
            target_q_params = state.params.target_q  # tuple of N target Q params
            policy_params = state.params.policy
            target_policy_params = state.params.target_policy
            td_policy_params = target_policy_params if self.td_use_target_policy else policy_params
            log_alpha = state.params.log_alpha
            q_opt_states = state.opt_state.q   # tuple of N opt states
            policy_opt_state = state.opt_state.policy
            log_alpha_opt_state = state.opt_state.log_alpha
            _value_opt_state = state.opt_state.value
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std
            log_eta_scales = state.log_eta_scales
            tfg_eta_current = state.tfg_eta
            _entropic_eta_ref[0] = tfg_eta_current
            dsac_omega_ema = state.dsac_omega_ema
            dsac_b_ema = state.dsac_b_ema
            num_q = len(q_params)
            # In soft-PI mode, use frozen Q for guidance (action sampling);
            # live Q is only used for TD learning and evaluation.
            if self.soft_pi_mode:
                guidance_q = state.frozen_q
            else:
                guidance_q = q_params
            # Split enough keys: base keys + per-Q keys for langevin noise
            key_splits = jax.random.split(key, 8 + num_q)
            next_eval_key = key_splits[0]
            new_eval_key = key_splits[1]
            log_alpha_key = key_splits[2]
            diffusion_time_key = key_splits[3]
            diffusion_noise_key = key_splits[4]
            q_agg_key = key_splits[5]
            q_shuffle_key = key_splits[6]
            # key_splits[7] reserved
            q_langevin_keys = key_splits[8:]  # num_q keys

            # --- N-ary Q helpers ---
            def eval_all_q(qp_tuple, s, a):
                """Evaluate all N Q networks. Returns (means_list, vars_list)."""
                means, vrs = [], []
                for qp in qp_tuple:
                    m, v = self.agent.q(qp, s, a)
                    means.append(m)
                    vrs.append(v)
                return means, vrs

            def reduce_min_q(means_list):
                """Element-wise min across N Q means."""
                q = means_list[0]
                for m in means_list[1:]:
                    q = jnp.minimum(q, m)
                return q

            if self.q_critic_agg == "random":
                critic_idx = jax.random.randint(q_agg_key, (), 0, num_q)

                def aggregate_q_n(q_means, q_vars):
                    stacked = jnp.stack(q_means, axis=0)
                    return _u_to_q(stacked[critic_idx])

            else:

                def aggregate_q_n(q_means, q_vars):
                    if q_critic_agg == "vmap_mean_min":
                        # Per-slot selector: compute both reductions, blend
                        # with state.q_critic_agg_idx (0 = mean, 1 = min).
                        # Under vmap the selector is a [N]-shaped array, so
                        # the blend is per-slot. Broadcast rhs-scalar over
                        # the (batch, ...) shape of the Q reductions.
                        mean_val = sum(q_means) * jnp.float32(1.0 / num_q)
                        min_val = reduce_min_q(q_means)
                        idx = jnp.clip(jnp.float32(state.q_critic_agg_idx), 0.0, 1.0)
                        return _u_to_q((1.0 - idx) * mean_val + idx * min_val)
                    if q_critic_agg == "min":
                        return _u_to_q(reduce_min_q(q_means))
                    elif q_critic_agg == "mean":
                        return _u_to_q(sum(q_means) * jnp.float32(1.0 / num_q))
                    elif q_critic_agg == "max":
                        q = q_means[0]
                        for m in q_means[1:]:
                            q = jnp.maximum(q, m)
                        return _u_to_q(q)
                    elif q_critic_agg == "entropic":
                        _ent_beta = jnp.float32(self.entropic_risk_beta)
                        _use_mean_fallback = jnp.abs(_ent_beta) < 1e-6
                        mean_val = sum(q_means) * jnp.float32(1.0 / num_q)
                        stacked = jnp.stack([_ent_beta * m for m in q_means], axis=0)
                        ent_val = (jnp.float32(1.0) / _ent_beta) * (jax.scipy.special.logsumexp(stacked, axis=0) - jnp.log(jnp.float32(num_q)))
                        return _u_to_q(jnp.where(_use_mean_fallback, mean_val, ent_val))
                    elif q_critic_agg == "precision":
                        # Precision-weighted mean
                        eps = jnp.float32(1e-6)
                        precisions = [jnp.float32(1.0) / jnp.maximum(v, eps) for v in q_vars]
                        total_prec = sum(precisions)
                        return _u_to_q(sum(p * m for p, m in zip(precisions, q_means)) / total_prec)
                    else:
                        return _u_to_q(reduce_min_q(q_means))

            reward *= state.reward_scale

            def get_min_q(s, a):
                """Get aggregated Q value (mean only) for signals like reweighting.

                Returns Q_η = u/η when entropic_q is on (via aggregate_q_n
                which calls _u_to_q internally).
                """
                q_means, q_vars = eval_all_q(q_params, s, a)
                q_mean = aggregate_q_n(q_means, q_vars)
                if self.use_reward_critic:
                    q_mean = q_mean * (jnp.float32(1.0) / jnp.float32(1.0 - state.gamma))
                return q_mean

            def get_min_target_q(s, a):
                """Get min target Q mean for action selection.

                Returns Q_η = u/η when entropic_q is on.
                """
                q_means, _q_vars = eval_all_q(target_q_params, s, a)
                return _u_to_q(reduce_min_q(q_means))

            def get_target_q_dist(s, a):
                """Get all N Q distributions from target networks."""
                return eval_all_q(target_q_params, s, a)

            reward_loss = jnp.float32(0.0)

            mala_acc_rate = jnp.float32(jnp.nan)
            mala_eta_scale = jnp.float32(jnp.nan)
            mala_pl_acc = jnp.full((timesteps,), jnp.nan, dtype=jnp.float32)
            mala_pl_clip = jnp.full((timesteps,), jnp.nan, dtype=jnp.float32)

            mala_eta_scale_in = jnp.float32(jnp.nan)
            mala_eta_scale_out = jnp.float32(jnp.nan)

            q_backup_mean = jnp.float32(jnp.nan)
            q_backup_std = jnp.float32(jnp.nan)
            q_backup_min = jnp.float32(jnp.nan)
            q_backup_max = jnp.float32(jnp.nan)
            q_target_mean = jnp.float32(jnp.nan)
            q_target_std = jnp.float32(jnp.nan)
            q_target_var_mean = jnp.float32(jnp.nan)
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

            _use_entropic_q = self.use_entropic_q  # True for either u or w mode
            _entropic_u = self._entropic_u_mode
            _entropic_w = self._entropic_w_mode

            q_backup_for_v = None
            q_target_for_v = None
            next_obs_for_v = None

            if not self.use_reward_critic:
                if self.off_policy_td:
                    # SARSA-style off-policy: use the action actually taken at next_obs
                    # (stored as next_action in buffer via track_next_action)
                    next_action = next_action_buffer
                    tq_means, tq_vars = eval_all_q(target_q_params, next_obs, next_action)
                    q_target = reduce_min_q(tq_means)
                    q_target_min = q_target
                    # Per-Q targets for independent bootstrap
                    q_target_per_q = tq_means  # list of N means
                    q_target_var_per_q = tq_vars  # list of N vars
                else:
                    # On-policy: sample fresh actions from current policy
                    td_params = (td_policy_params, log_alpha, guidance_q, log_eta_scales, tfg_eta_current, state.value_params, state.advantage_second_moment_ema, state.target_value_params, state.reward_scale, state.x0_hat_clip_radius, state.mala_adapt_rate, state.guidance_mult, state.gamma, state.q_critic_agg_idx)
                    use_multi_td = jnp.bool_(self.td_actions > 1)

                    if (not self.mala_per_level_eta) and self.agent.energy_mode and self.agent.mala_steps > 0:
                        mala_eta_scale_in = jnp.exp(log_eta_scales[0])

                    def sample_next_action_and_stats(k: jax.Array, log_eta_scales_in: jax.Array):
                        act, log_eta_scales_out, mala_acc_rate_out, mala_eta_scale_out, pl_acc_out, pl_clip_out = sample_action_with_agg_metrics(
                            k,
                            (td_policy_params, log_alpha, guidance_q, log_eta_scales_in, tfg_eta_current, state.value_params, state.advantage_second_moment_ema, state.target_value_params, state.reward_scale, state.x0_hat_clip_radius, state.mala_adapt_rate, state.guidance_mult, state.gamma, state.q_critic_agg_idx),
                            next_obs,
                            hard_min_q_n,
                        )
                        return act, log_eta_scales_out, mala_acc_rate_out, mala_eta_scale_out, pl_acc_out, pl_clip_out

                    def sample_single_td():
                        return sample_next_action_and_stats(next_eval_key, log_eta_scales)

                    def sample_multi_td():
                        batch = next_obs.shape[0]
                        next_obs_rep = jnp.tile(next_obs, (int(self.td_actions), 1))
                        act_sel_flat, acts_particles, probs_particles, log_eta_scales_out, mala_acc_rate_out, mala_eta_scale_out, pl_acc_out, pl_clip_out = sample_action_with_particle_details_metrics(
                            next_eval_key,
                            (td_policy_params, log_alpha, guidance_q, log_eta_scales, tfg_eta_current, state.value_params, state.advantage_second_moment_ema, state.target_value_params, state.reward_scale, state.x0_hat_clip_radius, state.mala_adapt_rate, state.guidance_mult, state.gamma, state.q_critic_agg_idx),
                            next_obs_rep,
                            hard_min_q_n,
                        )
                        acts_sel = act_sel_flat.reshape((int(self.td_actions), batch, self.agent.act_dim))
                        # Get Q distributions for each target Q network per particle
                        per_q_means_list = []
                        per_q_vars_list = []
                        for tqp in target_q_params:
                            qm, qv = jax.vmap(lambda a: self.agent.q(tqp, next_obs_rep, a), in_axes=0)(acts_particles)
                            per_q_means_list.append(qm)
                            per_q_vars_list.append(qv)

                        # Per-Q mixture aggregation across particles
                        # u-mode: logsumexp (entropic aggregation in log-W space)
                        # w-mode / standard: weighted arithmetic mean
                        q_target_per_q_list = []
                        q_target_var_per_q_list = []
                        if _entropic_u:
                            for qm in per_q_means_list:
                                agg = _entropic_particle_agg(qm, probs_particles)
                                q_target_per_q_list.append(agg.reshape((int(self.td_actions), batch)).mean(axis=0))
                                q_target_var_per_q_list.append(jnp.zeros_like(q_target_per_q_list[-1]))
                        else:
                            for qm, qv in zip(per_q_means_list, per_q_vars_list):
                                mix_m, mix_v = mixture_mean_var(qm, qv, probs_particles)
                                q_target_per_q_list.append(mix_m.reshape((int(self.td_actions), batch)).mean(axis=0))
                                q_target_var_per_q_list.append(mix_v.reshape((int(self.td_actions), batch)).mean(axis=0))

                        # Min-Q across critics for overall target
                        min_q_particles = per_q_means_list[0]
                        for mqm in per_q_means_list[1:]:
                            min_q_particles = jnp.minimum(min_q_particles, mqm)
                        if _entropic_u:
                            q_t = _entropic_particle_agg(min_q_particles, probs_particles)
                            q_t = q_t.reshape((int(self.td_actions), batch)).mean(axis=0)
                        else:
                            q_exp = jnp.sum(probs_particles * min_q_particles, axis=0)
                            q_t = q_exp.reshape((int(self.td_actions), batch)).mean(axis=0)
                        return acts_sel[0], log_eta_scales_out, mala_acc_rate_out, mala_eta_scale_out, pl_acc_out, pl_clip_out, q_t, q_target_per_q_list, q_target_var_per_q_list

                    _entropic_gamma = jnp.float32(state.gamma)
                    _entropic_inv_gamma = jnp.where(_entropic_gamma > 0, jnp.float32(1.0) / _entropic_gamma, jnp.float32(0.0))

                    def _entropic_particle_agg(u_particles, weights):
                        """Weighted logsumexp: logsumexp_k(γ·u_k + log w_k) / γ.

                        Used only in u-mode.  Replaces the arithmetic weighted
                        mean Σ w_k·u_k.  For a single particle with weight 1
                        this reduces to u, matching the K=1 case.
                        """
                        log_w = jnp.log(weights + 1e-30)
                        return jax.scipy.special.logsumexp(
                            _entropic_gamma * u_particles + log_w, axis=0
                        ) * _entropic_inv_gamma

                    def pick_td(_):
                        a, acts_particles, probs_particles, log_eta_scales_out, mala_acc_rate_out, mala_eta_scale_out, pl_acc_out, pl_clip_out = sample_action_with_particle_details_metrics(
                            next_eval_key,
                            (td_policy_params, log_alpha, guidance_q, log_eta_scales, tfg_eta_current, state.value_params, state.advantage_second_moment_ema, state.target_value_params, state.reward_scale, state.x0_hat_clip_radius, state.mala_adapt_rate, state.guidance_mult, state.gamma, state.q_critic_agg_idx),
                            next_obs,
                            hard_min_q_n,
                        )
                        # Per-Q target distributions from particles
                        per_q_means_list = []
                        per_q_vars_list = []
                        for tqp in target_q_params:
                            qm, qv = jax.vmap(lambda actp: self.agent.q(tqp, next_obs, actp), in_axes=0)(acts_particles)
                            per_q_means_list.append(qm)
                            per_q_vars_list.append(qv)

                        q_target_per_q_list = []
                        q_target_var_per_q_list = []
                        if _entropic_u:
                            # u-mode: logsumexp aggregation across particles
                            for qm in per_q_means_list:
                                q_target_per_q_list.append(_entropic_particle_agg(qm, probs_particles))
                                q_target_var_per_q_list.append(jnp.zeros_like(q_target_per_q_list[-1]))
                        else:
                            for qm, qv in zip(per_q_means_list, per_q_vars_list):
                                mix_m, mix_v = mixture_mean_var(qm, qv, probs_particles)
                                q_target_per_q_list.append(mix_m)
                                q_target_var_per_q_list.append(mix_v)

                        min_q_particles = per_q_means_list[0]
                        for mqm in per_q_means_list[1:]:
                            min_q_particles = jnp.minimum(min_q_particles, mqm)
                        if _entropic_u:
                            q_t = _entropic_particle_agg(min_q_particles, probs_particles)
                        else:
                            q_t = jnp.sum(probs_particles * min_q_particles, axis=0)
                        return a, log_eta_scales_out, mala_acc_rate_out, mala_eta_scale_out, pl_acc_out, pl_clip_out, q_t, q_target_per_q_list, q_target_var_per_q_list

                    def pick_multi(_):
                        return sample_multi_td()

                    next_action, log_eta_scales, mala_acc_rate, mala_eta_scale, mala_pl_acc, mala_pl_clip, q_target, q_target_per_q, q_target_var_per_q = jax.lax.cond(
                        use_multi_td,
                        pick_multi,
                        pick_td,
                        operand=None,
                    )
                    q_target_min = q_target

                # Compute TD backups based on q_bootstrap_agg mode
                # q_target_per_q: list of N target means; q_target_var_per_q: list of N target vars
                gamma_sq = jnp.float32(state.gamma) ** 2
                not_done = (1 - done)

                if self.q_bootstrap_agg == "independent":
                    # Each Q bootstraps from its own target network
                    q_backup_per_q = [reward + not_done * state.gamma * tqm for tqm in q_target_per_q]
                    q_backup_var_per_q = [not_done * gamma_sq * tqv for tqv in q_target_var_per_q]
                elif self.q_bootstrap_agg == "mean":
                    # All Qs bootstrap from the mean of all target networks
                    inv_n = jnp.float32(1.0 / num_q)
                    q_target_avg = sum(q_target_per_q) * inv_n
                    q_target_avg_var = sum(q_target_var_per_q) * (inv_n ** 2)
                    shared_backup = reward + not_done * state.gamma * q_target_avg
                    shared_backup_var = not_done * gamma_sq * q_target_avg_var
                    q_backup_per_q = [shared_backup] * num_q
                    q_backup_var_per_q = [shared_backup_var] * num_q
                else:
                    # Default "min" (and mixture/pick_min fall through here for N>2)
                    q_min_mean = reduce_min_q(q_target_per_q)
                    # For variance, use the variance of the critic with the min mean
                    # (element-wise selection)
                    if num_q == 2 and self.q_bootstrap_agg in ("mixture", "pick_min"):
                        q_agg_mean, q_agg_var = aggregate_q_distributions(
                            q_target_per_q[0], q_target_var_per_q[0],
                            q_target_per_q[1], q_target_var_per_q[1],
                            self.q_bootstrap_agg,
                        )
                    else:
                        q_agg_mean = q_min_mean
                        # Pick variance from whichever Q had the min mean (element-wise)
                        q_agg_var = q_target_var_per_q[0]
                        for i in range(1, num_q):
                            is_min = q_target_per_q[i] <= q_agg_mean + 1e-8
                            q_agg_var = jnp.where(is_min, q_target_var_per_q[i], q_agg_var)
                    shared_backup = reward + not_done * state.gamma * q_agg_mean
                    shared_backup_var = not_done * gamma_sq * q_agg_var
                    q_backup_per_q = [shared_backup] * num_q
                    q_backup_var_per_q = [shared_backup_var] * num_q
                if _entropic_w:
                    # w-mode Bellman target: y = exp(η·r) · clip(W_next, ε)^γ
                    # when done, y = exp(η·r) (continuation term vanishes).
                    # reward was scaled by reward_scale; entropic η is tfg_eta.
                    # So η·r = (tfg_eta / reward_scale) · reward.
                    _w_eps_jnp = jnp.float32(self._entropic_w_eps)
                    _gamma_jnp = jnp.float32(state.gamma)
                    _eta_over_rs = tfg_eta_current / state.reward_scale
                    exp_reward = jnp.exp(reward * _eta_over_rs)
                    def _w_backup(w_next):
                        w_next_clipped = jnp.maximum(w_next, _w_eps_jnp)
                        return exp_reward * jnp.where(done > 0.5, jnp.float32(1.0),
                                                      jnp.power(w_next_clipped, _gamma_jnp))
                    if self.q_bootstrap_agg == "independent":
                        q_backup_per_q = [_w_backup(tqm) for tqm in q_target_per_q]
                    elif self.q_bootstrap_agg == "mean":
                        inv_n = jnp.float32(1.0 / num_q)
                        w_avg = sum(q_target_per_q) * inv_n
                        shared_w_backup = _w_backup(w_avg)
                        q_backup_per_q = [shared_w_backup] * num_q
                    else:
                        w_min = reduce_min_q(q_target_per_q)
                        shared_w_backup = _w_backup(w_min)
                        q_backup_per_q = [shared_w_backup] * num_q
                    q_backup_var_per_q = [jnp.zeros_like(q_backup_per_q[0])] * num_q
                elif _entropic_u:
                    # u-mode: backup is η·r + γ·u_target.  reward = reward_scale·r.
                    # Correct reward contribution: (η / reward_scale) · reward.
                    _eta_over_rs = tfg_eta_current / state.reward_scale
                    _reward_correction = reward * (_eta_over_rs - jnp.float32(1.0))
                    q_backup_per_q = [qb + _reward_correction for qb in q_backup_per_q]

                q_backup = q_backup_per_q[0]  # For logging (uses Q0's backup as representative)
                q_backup_for_v = q_backup
                q_target_for_v = q_target
                next_obs_for_v = next_obs

                q_backup_mean = jnp.mean(q_backup)
                q_backup_std = jnp.std(q_backup)
                q_backup_min = jnp.min(q_backup)
                q_backup_max = jnp.max(q_backup)
                q_target_mean = jnp.mean(q_target)
                q_target_std = jnp.std(q_target)
                if self.agent.distributional_critic:
                    q_target_var_mean = jnp.mean(jnp.stack(q_target_var_per_q))
                reward_mean = jnp.mean(reward)
                done_frac = jnp.mean(done)

                # Huber loss helper (shared by Q1 and Q2). delta is per-seed under
                # vmap; clamp to finite for the huber branch so inf·0 doesn't
                # produce NaN gradients when jax.vmap rewrites lax.cond → where.
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

                def compute_entropic_loss(u_pred, t_target):
                    """Pseudo-loss whose gradient in u matches MSE in W=exp(u) space.

                    L(u, t) = exp(t - u) - (t - u) - 1

                    dL/du = 1 - exp(t - u) = 1 - y_W / W_θ

                    The induced update in W-space is, to first order, the same as
                    ordinary MSE regression on W.
                    """
                    diff = t_target - u_pred  # t - u
                    # Clamp diff to avoid exp overflow.  At diff=30, exp≈1e13 which
                    # is already far into float32 territory; beyond that the loss is
                    # dominated by the linear term anyway.
                    diff_clamped = jnp.clip(diff, -30.0, 30.0)
                    per_elem = jnp.exp(diff_clamped) - diff - jnp.float32(1.0)
                    return jnp.mean(per_elem)
                
                # DSAC-T refinements: compute omega scale factor for loss scaling
                use_dsac_t = (self.dsac_expected_value_sub or 
                              self.dsac_adaptive_clip_xi > 0.0 or 
                              self.dsac_omega_scaling)
                
                # Compute CURRENT omega from Q predictions BEFORE update (for omega scaling and EMA)
                if self.agent.distributional_critic and (self.dsac_omega_scaling or self.dsac_adaptive_clip_xi > 0.0):
                    q_vars_current = [self.agent.q(qp, obs, action)[1] for qp in q_params]
                    dsac_current_omega = sum(jnp.mean(v) for v in q_vars_current) * jnp.float32(1.0 / num_q)
                    dsac_current_b = jnp.sqrt(dsac_current_omega)
                else:
                    dsac_current_omega = jnp.float32(1.0)
                    dsac_current_b = jnp.float32(1.0)
                
                dsac_clip_xi = jnp.float32(self.dsac_adaptive_clip_xi)
                dsac_use_omega_scaling = jnp.bool_(self.dsac_omega_scaling)
                
                def compute_distributional_td_loss(pred_mean, pred_var, target_mean, target_var):
                    """Compute TD loss for distributional critic using cross-entropy."""
                    if use_dsac_t:
                        return dsac_t_loss_with_custom_grad(
                            pred_mean, pred_var, target_mean, target_var,
                            dsac_clip_xi, dsac_b_ema, dsac_omega_ema, dsac_use_omega_scaling
                        )
                    if self.critic_grad_modifier == "natgrad":
                        return mean_gaussian_cross_entropy_natgrad(pred_mean, pred_var, target_mean, target_var)
                    if self.critic_grad_modifier == "variance_scaled":
                        ce = gaussian_cross_entropy(pred_mean, pred_var, target_mean, target_var)
                        return jnp.mean(jax.lax.stop_gradient(pred_var) * ce)
                    return jnp.mean(gaussian_cross_entropy(pred_mean, pred_var, target_mean, target_var))

                def compute_distributional_td_ce(pred_mean, pred_var, target_mean, target_var):
                    """Compute raw cross-entropy (for logging/comparability)."""
                    return jnp.mean(gaussian_cross_entropy(pred_mean, pred_var, target_mean, target_var))

                # --- D_ψ TD targets (dist_shift_eta only) ---
                if self.dist_shift_eta and self.agent.d_psi is not None:
                    q_for_dpsi = aggregate_q_n(q_target_per_q, q_target_var_per_q)
                    v_for_dpsi = self._value_net.apply(state.value_params, next_obs)
                    adv_for_dpsi = q_for_dpsi - v_for_dpsi
                    adv_sq = adv_for_dpsi ** 2
                    d_psi_bootstrap = [self.agent.d_psi(tqp, next_obs, next_action)
                                       for tqp in target_q_params]
                    d_psi_bootstrap_mean = sum(d_psi_bootstrap) * jnp.float32(1.0 / num_q)
                    d_psi_td_target = adv_sq + jnp.float32(state.gamma) * not_done * d_psi_bootstrap_mean
                else:
                    d_psi_td_target = None

                # --- N-ary Q loss computation and gradient update (vmapped for GPU efficiency) ---
                # Prepare per-Q data as stacked arrays with leading dim = num_q
                if self.decorrelated_q_batches:
                    batch_size = obs.shape[0]
                    chunk = batch_size // num_q
                    perm = jax.random.permutation(q_shuffle_key, batch_size)
                    stacked_obs_q = jnp.stack([obs[perm[i*chunk:(i+1)*chunk]] for i in range(num_q)])
                    stacked_act_q = jnp.stack([action[perm[i*chunk:(i+1)*chunk]] for i in range(num_q)])
                    stacked_backup_q = jnp.stack([q_backup_per_q[i][perm[i*chunk:(i+1)*chunk]] for i in range(num_q)])
                    stacked_backup_var_q = jnp.stack([q_backup_var_per_q[i][perm[i*chunk:(i+1)*chunk]] for i in range(num_q)])
                    if d_psi_td_target is not None:
                        stacked_dpsi_target = jnp.stack([d_psi_td_target[perm[i*chunk:(i+1)*chunk]] for i in range(num_q)])
                    else:
                        stacked_dpsi_target = jnp.zeros_like(stacked_backup_q)
                else:
                    stacked_obs_q = jnp.broadcast_to(obs[None], (num_q,) + obs.shape)
                    stacked_act_q = jnp.broadcast_to(action[None], (num_q,) + action.shape)
                    stacked_backup_q = jnp.stack(q_backup_per_q)
                    stacked_backup_var_q = jnp.stack(q_backup_var_per_q)
                    if d_psi_td_target is not None:
                        stacked_dpsi_target = jnp.broadcast_to(d_psi_td_target[None], (num_q,) + d_psi_td_target.shape)
                    else:
                        stacked_dpsi_target = jnp.zeros_like(stacked_backup_q)

                # Stack Q params and opt states into batched pytrees (each leaf: (num_q, ...))
                stacked_q_params = jax.tree.map(lambda *ps: jnp.stack(ps), *q_params)
                stacked_q_opt = jax.tree.map(lambda *ss: jnp.stack(ss), *q_opt_states)

                def single_q_train_step(qp, opt_s, obs_qi, act_qi, backup_qi, backup_var_qi, lang_key, dpsi_target_qi):
                    """Loss, grad, and optimizer update for one Q network."""
                    def q_loss_fn(p):
                        if self.dist_shift_eta and self.agent.q_and_d_psi is not None:
                            q_mean, q_var, d_psi_pred = self.agent.q_and_d_psi(p, obs_qi, act_qi)
                        else:
                            q_mean, q_var = self.agent.q(p, obs_qi, act_qi)
                        if _entropic_u:
                            # u-mode pseudo-loss: network output u ≈ η·Q_η.
                            # backup_qi is the log-target t = η·r + (1-d)·γ·u_target.
                            loss = compute_entropic_loss(q_mean, backup_qi)
                        elif _entropic_w:
                            # w-mode MSE: network output is W directly.
                            # backup_qi is y = exp(η·r) · clip(W_next, ε)^γ.
                            # Raw prediction is unconstrained; only target-side
                            # continuation output is clipped (in _w_backup above).
                            loss = jnp.mean((q_mean - backup_qi) ** 2)
                        elif self.agent.distributional_critic:
                            loss = compute_distributional_td_loss(q_mean, q_var, backup_qi, backup_var_qi)
                        else:
                            td_err = q_mean - backup_qi
                            loss = compute_td_loss(td_err)
                        if self.dist_shift_eta and self.agent.q_and_d_psi is not None:
                            d_psi_loss = jnp.mean((d_psi_pred - jax.lax.stop_gradient(dpsi_target_qi)) ** 2)
                            loss = loss + d_psi_loss
                        return loss, q_mean
                    (qi_loss, qi_pred), qi_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(qp)
                    if self.langevin_q_noise:
                        batch_size_eff = obs_qi.shape[0]
                        langevin_scale = jnp.sqrt(2.0 * state.lr_q / batch_size_eff)
                        flat_grads, treedef = jax.tree_util.tree_flatten(qi_grads)
                        noise_keys = jax.random.split(lang_key, len(flat_grads))
                        noisy_flat = [g + langevin_scale * jax.random.normal(k, g.shape)
                                      for g, k in zip(flat_grads, noise_keys)]
                        qi_grads = jax.tree_util.tree_unflatten(treedef, noisy_flat)
                    update, new_opt = self.optim.update(qi_grads, opt_s, params=qp)
                    if self._q_lr_runtime:
                        update = jax.tree.map(lambda u: -state.lr_q * u, update)
                    new_qp = optax.apply_updates(qp, update)
                    return new_qp, new_opt, qi_loss, qi_pred

                if self.single_q_network:
                    # Single Q network: train once, replicate to all slots
                    new_qp0, new_opt0, q0_loss, q0_pred = single_q_train_step(
                        q_params[0], q_opt_states[0],
                        stacked_obs_q[0], stacked_act_q[0],
                        stacked_backup_q[0], stacked_backup_var_q[0],
                        q_langevin_keys[0],
                        stacked_dpsi_target[0],
                    )
                    q_params = tuple([new_qp0] * num_q)
                    q_opt_states = tuple([new_opt0] * num_q)
                    all_q_losses = jnp.full((num_q,), q0_loss)
                    all_q_preds = jnp.broadcast_to(q0_pred[None], (num_q,) + q0_pred.shape)
                else:
                    # Parallel training: vmap across all N Q networks
                    stacked_new_qp, stacked_new_opt, all_q_losses, all_q_preds = jax.vmap(
                        single_q_train_step
                    )(
                        stacked_q_params, stacked_q_opt,
                        stacked_obs_q, stacked_act_q,
                        stacked_backup_q, stacked_backup_var_q,
                        q_langevin_keys,
                        stacked_dpsi_target,
                    )
                    # Unstack back to tuples for downstream code
                    q_params = tuple(jax.tree.map(lambda x: x[i], stacked_new_qp) for i in range(num_q))
                    q_opt_states = tuple(jax.tree.map(lambda x: x[i], stacked_new_opt) for i in range(num_q))

                # Logging: use first two Qs for backward-compatible metric names
                q1_loss = all_q_losses[0]
                q2_loss = all_q_losses[min(1, num_q - 1)]
                q1 = all_q_preds[0]
                q2 = all_q_preds[min(1, num_q - 1)]
                q1_pred_mean = jnp.mean(q1)
                q1_pred_std = jnp.std(q1)
                q2_pred_mean = jnp.mean(q2)
                q2_pred_std = jnp.std(q2)
                td_error1_std = jnp.std(q1 - stacked_backup_q[0])
                td_error2_std = jnp.std(q2 - stacked_backup_q[min(1, num_q - 1)])
            else:
                if self.pure_bc_training:
                    next_action = action
                else:
                    td_params = (td_policy_params, log_alpha, guidance_q, log_eta_scales, tfg_eta_current, state.value_params, state.advantage_second_moment_ema, state.target_value_params, state.reward_scale, state.x0_hat_clip_radius, state.mala_adapt_rate, state.guidance_mult, state.gamma, state.q_critic_agg_idx)

                    if (not self.mala_per_level_eta) and self.agent.energy_mode and self.agent.mala_steps > 0:
                        mala_eta_scale_in = jnp.exp(log_eta_scales[0])

                    next_action, log_eta_scales, mala_acc_rate, mala_eta_scale, mala_pl_acc, mala_pl_clip = sample_action_with_agg_metrics(
                        next_eval_key,
                        td_params,
                        next_obs,
                        hard_min_q_n,
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

                def reward_loss_fn(qp: hk.Params) -> jax.Array:
                    r_pred_mean, r_pred_var = self.agent.q(qp, obs, action)
                    r_loss = jnp.mean((r_pred_mean - reward) ** 2)
                    return r_loss, r_pred_mean

                r_losses = []
                new_q_params_list = list(q_params)
                new_q_opt_states_list = list(q_opt_states)
                for qi in range(num_q):
                    if self.single_q_network and qi > 0:
                        new_q_params_list[qi] = new_q_params_list[0]
                        r_losses.append(r_losses[0])
                    else:
                        (ri_loss, _qi_pred), qi_grads = jax.value_and_grad(reward_loss_fn, has_aux=True)(new_q_params_list[qi])
                        qi_update, qi_new_opt = self.optim.update(qi_grads, new_q_opt_states_list[qi], params=new_q_params_list[qi])
                        if self._q_lr_runtime:
                            qi_update = jax.tree.map(lambda u: -state.lr_q * u, qi_update)
                        new_q_params_list[qi] = optax.apply_updates(new_q_params_list[qi], qi_update)
                        new_q_opt_states_list[qi] = qi_new_opt
                        r_losses.append(ri_loss)
                q_params = tuple(new_q_params_list)
                q_opt_states = tuple(new_q_opt_states_list)
                all_q_losses = jnp.stack(r_losses)
                reward_loss = jnp.mean(all_q_losses)
                q1 = self.agent.q(q_params[0], obs, action)[0]
                q2 = self.agent.q(q_params[min(1, num_q-1)], obs, action)[0] if num_q > 1 else q1

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

                    _sum_reduction = (self.policy_loss_reduction == "sum")
                    if self.policy_loss_type == "ula_kl":
                        # --- IS-corrected ULA-KL: mean_w * ||eps_theta - eps||^2 ---
                        sq_err = (noise_pred - noise) ** 2
                        if _sum_reduction:
                            per_sample_loss = mean_is_w * jnp.sum(sq_err, axis=-1)
                        else:
                            per_sample_loss = mean_is_w * jnp.mean(sq_err, axis=-1)
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
                        lambda_t = lambda_for_step(t, tfg_eta_current)
                        c_t = lambda_t * (1.0 - alpha_bar_t) / alpha_bar_t

                        # Scalar Q for a single (action, obs) pair
                        def q_scalar_single(a, o):
                            a_b, o_b = a[None], o[None]
                            qms, qvs = eval_all_q(q_params, o_b, a_b)
                            return aggregate_q_n(qms, qvs)[0]

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
                        _reduce_dims = jnp.sum if _sum_reduction else jnp.mean
                        per_sample_loss = (mean_is_w / 4.0) * _reduce_dims(M_delta_s ** 2, axis=-1)
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
                            -state.x0_hat_clip_radius,
                            state.x0_hat_clip_radius,
                        )

                        # Scalar Q for a single (action, obs) pair
                        def q_scalar_single(a, o):
                            a_b, o_b = a[None], o[None]
                            qms, qvs = eval_all_q(q_params, o_b, a_b)
                            return aggregate_q_n(qms, qvs)[0]

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
                        guidance_scale = tfg_eta_current / sqrt_abar

                        g_theta = s_theta + guidance_scale * q_grad_model
                        g_target = jax.lax.stop_gradient(
                            s_cond + guidance_scale * q_grad_target
                        )

                        delta_g = g_theta - g_target
                        _reduce_dims = jnp.sum if _sum_reduction else jnp.mean
                        per_sample_loss = (mean_is_w / 4.0) * _reduce_dims(delta_g ** 2, axis=-1)
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
                        reduction=self.policy_loss_reduction,
                    )

                return loss, (base_q_weights, scaled_q, q_mean, q_std)

            (total_loss, (q_weights, scaled_q, q_mean, q_std)), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)

            # update alpha
            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                approx_entropy = 0.5 * self.agent.act_dim * jnp.log( 2 * jnp.pi * jnp.exp(1) * (0.1 * jnp.exp(log_alpha)) ** 2)
                log_alpha_loss = -1 * log_alpha * (-1 * jax.lax.stop_gradient(approx_entropy) + self.agent.target_entropy)
                return log_alpha_loss

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

            def delay_alpha_param_update(optim, params, opt_state):
                return jax.lax.cond(
                    step % self.delay_alpha_update == 0,
                    lambda po: param_update(optim, po[0], jax.grad(log_alpha_loss_fn)(po[0]), po[1]),
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

            policy_lr = self._policy_lr_at_step(state.lr_policy, self._optimizer_count(policy_opt_state))
            policy_params, policy_opt_state = delay_param_update(self.policy_optim, policy_params, policy_grads, policy_opt_state, lr_value=policy_lr)
            if not self.no_entropy_tuning:
                log_alpha, log_alpha_opt_state = delay_alpha_param_update(self.alpha_optim, log_alpha, log_alpha_opt_state)

            new_target_q_list = []
            for qi in range(num_q):
                if self.single_q_network and qi > 0:
                    new_target_q_list.append(new_target_q_list[0])
                else:
                    new_target_q_list.append(delay_target_update(q_params[qi], target_q_params[qi], state.polyak_tau))
            target_q_params = tuple(new_target_q_list)
            target_policy_params = delay_target_update(policy_params, target_policy_params, state.polyak_tau)

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
                if self.td_value_training and self.critic_normalization == "ema" and q_target_for_v is not None:
                    v_target = jax.lax.stop_gradient(q_target_for_v)
                    delta_v = state.q_td_huber_width * state.reward_scale
                    use_huber_v = jnp.isfinite(delta_v)
                    delta_v_safe = jnp.where(use_huber_v, delta_v, jnp.float32(1.0))

                    def value_loss_fn(v_params):
                        v_pred = self._value_net.apply(v_params, next_obs_for_v)
                        err = v_pred - v_target
                        abs_err = jnp.abs(err)
                        quad = jnp.minimum(abs_err, delta_v_safe)
                        huber = jnp.float32(0.5) * quad * quad + delta_v_safe * (abs_err - quad)
                        per_elem = jnp.where(use_huber_v, huber, err * err)
                        return jnp.mean(per_elem)

                    v_loss, v_grads = jax.value_and_grad(value_loss_fn)(state.value_params)
                    v_updates, value_opt_state_updated = self.optim.update(v_grads, state.opt_state.value, state.value_params)
                    if self._q_lr_runtime:
                        v_updates = jax.tree.map(lambda u: -state.lr_q * u, v_updates)
                    value_params_updated = optax.apply_updates(state.value_params, v_updates)
                    value_loss_log = v_loss
                    adv_second_moment_log = state.advantage_second_moment_ema

                elif self.on_policy_ema and self.critic_normalization == "ema" and not self.use_reward_critic:
                    # KL-budget mode: train V(s') with on-policy targets Q(s', a').
                    # q_target_per_q/q_target_var_per_q are already computed at
                    # (next_obs, next_action) where a' ~ π_current.  Reuse them.
                    q_for_v = aggregate_q_n(q_target_per_q, q_target_var_per_q)

                    def value_loss_fn(v_params):
                        v_pred = self._value_net.apply(v_params, next_obs)
                        return jnp.mean((v_pred - jax.lax.stop_gradient(q_for_v)) ** 2)

                    v_loss, v_grads = jax.value_and_grad(value_loss_fn)(state.value_params)
                    v_updates, value_opt_state_updated = self.optim.update(v_grads, state.opt_state.value, state.value_params)
                    if self._q_lr_runtime:
                        v_updates = jax.tree.map(lambda u: -state.lr_q * u, v_updates)
                    value_params_updated = optax.apply_updates(state.value_params, v_updates)
                    value_loss_log = v_loss

                    # On-policy mode: EMA is updated outside jit in sample(); pass through here
                    adv_second_moment_log = state.advantage_second_moment_ema
                else:
                    # Legacy mode: train V(s) with off-policy Q(s, a_replay) targets
                    q_means_for_v, q_vars_for_v = eval_all_q(q_params, obs, action)
                    q_for_v = aggregate_q_n(q_means_for_v, q_vars_for_v)
                    if self.use_reward_critic:
                        q_for_v = q_for_v * (jnp.float32(1.0) / jnp.float32(1.0 - state.gamma))

                    if self.critic_normalization == "ema":
                        def value_loss_fn(v_params):
                            v_pred = self._value_net.apply(v_params, obs)
                            return jnp.mean((v_pred - jax.lax.stop_gradient(q_for_v)) ** 2)

                        v_loss, v_grads = jax.value_and_grad(value_loss_fn)(state.value_params)
                        v_updates, value_opt_state_updated = self.optim.update(v_grads, state.opt_state.value, state.value_params)
                        if self._q_lr_runtime:
                            v_updates = jax.tree.map(lambda u: -state.lr_q * u, v_updates)
                        value_params_updated = optax.apply_updates(state.value_params, v_updates)
                        value_loss_log = v_loss

                        # Update A² EMA: EMA of (Q - V)²
                        v_current = self._value_net.apply(state.value_params, obs)
                        advantage_squared = (q_for_v - v_current) ** 2
                        adv_second_moment_batch = jnp.mean(advantage_squared)
                        adv_ema_tau = state.adv_ema_tau
                        new_adv_second_moment_ema = state.advantage_second_moment_ema + adv_ema_tau * (adv_second_moment_batch - state.advantage_second_moment_ema)
                        adv_second_moment_log = adv_second_moment_batch
                    else:  # distributional
                        def value_loss_fn(v_params):
                            v_mean, v_log_var = self._value_net.apply(v_params, obs)
                            v_var = jnp.exp(v_log_var)
                            nll = 0.5 * (v_log_var + (jax.lax.stop_gradient(q_for_v) - v_mean) ** 2 / jnp.maximum(v_var, 1e-6))
                            return jnp.mean(nll)

                        v_loss, v_grads = jax.value_and_grad(value_loss_fn)(state.value_params)
                        v_updates, value_opt_state_updated = self.optim.update(v_grads, state.opt_state.value, state.value_params)
                        if self._q_lr_runtime:
                            v_updates = jax.tree.map(lambda u: -state.lr_q * u, v_updates)
                        value_params_updated = optax.apply_updates(state.value_params, v_updates)
                        value_loss_log = v_loss

                        v_mean_current, v_log_var_current = self._value_net.apply(state.value_params, obs)
                        adv_second_moment_log = jnp.mean(jnp.exp(v_log_var_current))

            target_value_params_updated = state.target_value_params
            if self.td_value_training and state.target_value_params is not None:
                target_value_params_updated = delay_target_update(value_params_updated, state.target_value_params, state.polyak_tau)

            state = Diffv2TrainState(
                params=Diffv2Params(q_params, target_q_params, policy_params, target_policy_params, log_alpha),
                opt_state=Diffv2OptStates(q=q_opt_states, policy=policy_opt_state, log_alpha=log_alpha_opt_state, value=value_opt_state_updated),
                step=step + 1,
                entropy=jnp.float32(0.0),
                running_mean=new_running_mean,
                running_std=new_running_std,
                log_eta_scales=log_eta_scales,
                tfg_eta=tfg_eta_current,
                dsac_omega_ema=new_dsac_omega_ema,
                dsac_b_ema=new_dsac_b_ema,
                value_params=value_params_updated,
                target_value_params=target_value_params_updated,
                advantage_second_moment_ema=new_adv_second_moment_ema,
                advantage_third_moment_ema=state.advantage_third_moment_ema,
                dist_shift_covariance_ema=state.dist_shift_covariance_ema,
                dist_shift_shape_ema=state.dist_shift_shape_ema,
                dist_shift_coeff_ema=state.dist_shift_coeff_ema,
                dist_shift_shape3_ema=state.dist_shift_shape3_ema,
                frozen_q=state.frozen_q,
                train_policy=state.train_policy,
                gamma=state.gamma,
                polyak_tau=state.polyak_tau,
                lr_q=state.lr_q,
                lr_policy=state.lr_policy,
                guidance_mult=state.guidance_mult,
                adv_ema_tau=state.adv_ema_tau,
                shape_ema_tau=state.shape_ema_tau,
                kl_budget_val=state.kl_budget_val,
                reward_scale=state.reward_scale,
                x0_hat_clip_radius=state.x0_hat_clip_radius,
                mala_adapt_rate=state.mala_adapt_rate,
                q_td_huber_width=state.q_td_huber_width,
                shape3_ema_tau=state.shape3_ema_tau,
                q_critic_agg_idx=state.q_critic_agg_idx,
            )

            # --- Losses ---
            # Q_MSE: average loss across ensemble members
            q_mse = jnp.mean(all_q_losses) if not self.use_reward_critic else jnp.float32(0.0)

            policy_eps_mse_per_dim = total_loss / jnp.float32(self.agent.act_dim) if self.policy_loss_reduction == "sum" else total_loss

            info = {
                self.policy_loss_key: policy_eps_mse_per_dim,
                "losses/Q_MSE": q_mse,
            }

            # V_MSE: only when V network exists
            if self.critic_normalization != "none" and state.value_params is not None:
                info["losses/V_MSE"] = value_loss_log

            # Normalized distributional critic CE: exp(mean_CE - H_standard)
            # H_standard = 0.5*(1 + log(2*pi)) is the entropy of a 1-D standard Gaussian
            if self.agent.distributional_critic and not self.use_reward_critic:
                q1_pred_mean_log, q1_pred_var_log = self.agent.q(q_params[0], obs, action)
                q2_pred_mean_log, q2_pred_var_log = self.agent.q(q_params[min(1, num_q-1)], obs, action)
                q1_ce_loss = compute_distributional_td_ce(q1_pred_mean_log, q1_pred_var_log, q_backup_per_q[0], q_backup_var_per_q[0])
                q2_ce_loss = compute_distributional_td_ce(q2_pred_mean_log, q2_pred_var_log, q_backup_per_q[min(1, num_q-1)], q_backup_var_per_q[min(1, num_q-1)])
                h_std = jnp.float32(0.5 * (1.0 + jnp.log(2.0 * jnp.pi)))
                avg_ce = (q1_ce_loss + q2_ce_loss) * jnp.float32(0.5)
                info["losses/Q_Exp(XEnt)_normalized"] = jnp.exp(avg_ce - h_std)

            # --- MALA per-level arrays (logged as wandb.Table line plots) ---
            # Only include when MALA sampling actually runs (otherwise arrays are NaN)
            if self.agent.energy_mode and self.agent.mala_steps > 0 and not self.pure_bc_training:
                info["MALA/acceptance_rate"] = mala_pl_acc
                info["MALA/clip_frac"] = mala_pl_clip
                if self.mala_per_level_eta:
                    info["MALA/eta_scale"] = jnp.exp(log_eta_scales)
                else:
                    # Broadcast scalar eta_scale to per-level array for uniform table format
                    info["MALA/eta_scale"] = jnp.broadcast_to(mala_eta_scale, mala_pl_acc.shape)

            # --- Q section ---
            if self.critic_normalization == "ema":
                _q_ema_key = "Critic/inv_sqrt(E(Var(Q_η))_ema)" if _use_entropic_q else "Critic/inv_sqrt(E(Var(Q))_ema)"
                info[_q_ema_key] = jnp.float32(1.0) / jnp.sqrt(jnp.maximum(new_adv_second_moment_ema, jnp.float32(1e-6)))
            if self.agent.distributional_critic:
                info["Critic/target_variance"] = q_target_var_mean
            return state, info

        @jax.jit
        def stateless_supervised_update(
            key: jax.Array,
            state: Diffv2TrainState,
            data: Experience,
        ) -> Tuple[Diffv2TrainState, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q_params_s = state.params.q
            target_q_params_s = state.params.target_q
            policy_params = state.params.policy
            target_policy_params = state.params.target_policy
            td_policy_params = target_policy_params if self.td_use_target_policy else policy_params
            log_alpha = state.params.log_alpha
            q_opt_states_s = state.opt_state.q
            policy_opt_state = state.opt_state.policy
            log_alpha_opt_state = state.opt_state.log_alpha
            step = state.step
            running_mean = state.running_mean
            running_std = state.running_std
            log_eta_scales = state.log_eta_scales
            tfg_eta_current = state.tfg_eta
            _entropic_eta_ref[0] = tfg_eta_current
            num_q_s = len(q_params_s)

            next_eval_key, diffusion_time_key, diffusion_noise_key, q_agg_key = jax.random.split(key, 4)

            # Build a per-slot aggregator when in vmap_mean_min mode so the
            # policy-side Q signal (reweighting / tilting) blends mean and min
            # using state.q_critic_agg_idx for this vmap slot.
            get_min_q_agg = (
                make_aggregate_q_fn(state.q_critic_agg_idx)
                if q_critic_agg == "vmap_mean_min"
                else aggregate_q_fn_outer
            )
            def get_min_q(s, a):
                q_means, q_vars = [], []
                for qp in q_params_s:
                    m, v = self.agent.q(qp, s, a)
                    q_means.append(m)
                    q_vars.append(v)
                q = get_min_q_agg(q_means, q_vars)
                if self.use_reward_critic:
                    q = q * (jnp.float32(1.0) / jnp.float32(1.0 - state.gamma))
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
                td_params = (td_policy_params, log_alpha, q_params_s, log_eta_scales, tfg_eta_current, state.value_params, state.advantage_second_moment_ema, state.target_value_params, state.reward_scale, state.x0_hat_clip_radius, state.mala_adapt_rate, state.guidance_mult, state.gamma, state.q_critic_agg_idx)
                next_action, _q, _ = sample_action_with_agg(next_eval_key, td_params, next_obs, hard_min_q_n)
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
                    reduction=self.policy_loss_reduction,
                )

                return loss, (base_q_weights, scaled_q, q_mean, q_std)

            (total_loss, (q_weights, scaled_q, q_mean, q_std)), policy_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(policy_params)

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

            policy_lr = self._policy_lr_at_step(state.lr_policy, self._optimizer_count(policy_opt_state))
            policy_params, policy_opt_state = delay_param_update(self.policy_optim, policy_params, policy_grads, policy_opt_state, lr_value=policy_lr)

            state = Diffv2TrainState(
                params=Diffv2Params(q_params_s, target_q_params_s, policy_params, target_policy_params, log_alpha),
                opt_state=Diffv2OptStates(q=q_opt_states_s, policy=policy_opt_state, log_alpha=log_alpha_opt_state, value=state.opt_state.value),
                step=step,
                entropy=jnp.float32(0.0),
                running_mean=running_mean,
                running_std=running_std,
                log_eta_scales=log_eta_scales,
                tfg_eta=tfg_eta_current,
                value_params=state.value_params,
                target_value_params=state.target_value_params,
                advantage_second_moment_ema=state.advantage_second_moment_ema,
                advantage_third_moment_ema=state.advantage_third_moment_ema,
                dist_shift_covariance_ema=state.dist_shift_covariance_ema,
                dist_shift_shape_ema=state.dist_shift_shape_ema,
                dist_shift_coeff_ema=state.dist_shift_coeff_ema,
                dist_shift_shape3_ema=state.dist_shift_shape3_ema,
                frozen_q=state.frozen_q,
                train_policy=state.train_policy,
                gamma=state.gamma,
                polyak_tau=state.polyak_tau,
                lr_q=state.lr_q,
                lr_policy=state.lr_policy,
                guidance_mult=state.guidance_mult,
                adv_ema_tau=state.adv_ema_tau,
                shape_ema_tau=state.shape_ema_tau,
                kl_budget_val=state.kl_budget_val,
                reward_scale=state.reward_scale,
                x0_hat_clip_radius=state.x0_hat_clip_radius,
                mala_adapt_rate=state.mala_adapt_rate,
                q_td_huber_width=state.q_td_huber_width,
                shape3_ema_tau=state.shape3_ema_tau,
                q_critic_agg_idx=state.q_critic_agg_idx,
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
            policy_params, log_alpha, q_params_tuple, log_eta_scales_in, tfg_eta_current, value_params, adv_second_moment_ema, target_value_params, reward_scale_hp, x0_hat_clip_radius_hp, mala_adapt_rate_hp, guidance_mult_hp, gamma_hp, q_critic_agg_idx_hp = params

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
                    num_q_local = len(q_params_tuple)
                    critic_idx = jax.random.randint(q_agg_key, (), 0, num_q_local)

                    def aggregate_q_local(q_means, q_vars):
                        stacked = jnp.stack(q_means, axis=0)
                        return stacked[critic_idx]

                else:
                    x_key, loop_key = jax.random.split(single_key)
                    aggregate_q_local = aggregate_q_fn

                def q_mean_from_x(x_in, t_idx):
                    noise_pred = model_fn(t_idx, x_in)
                    x0_hat = (
                        x_in * B.sqrt_recip_alphas_cumprod[t_idx]
                        - noise_pred * B.sqrt_recipm1_alphas_cumprod[t_idx]
                    )
                    x0_hat = jnp.clip(x0_hat, -x0_hat_clip_radius_hp, x0_hat_clip_radius_hp)
                    q_means, q_vars = [], []
                    for qp in q_params_tuple:
                        m, v = self.agent.q(qp, obs_batch, x0_hat)
                        q_means.append(m)
                        q_vars.append(v)
                    q = aggregate_q_local(q_means, q_vars)
                    if self.use_reward_critic:
                        q = q * (jnp.float32(1.0) / jnp.float32(1.0 - gamma_hp))

                    agg_fn = jnp.sum if self.batch_independent_guidance else jnp.mean
                    mult = guidance_mult_hp

                    # Critic normalization: use (Q - V) / std instead of Q
                    if self.critic_normalization == "ema" and value_params is not None:
                        v_params_for_adv = target_value_params if self.td_value_training and target_value_params is not None else value_params
                        v = self._value_net.apply(v_params_for_adv, obs_batch)
                        advantage = q - v
                        adv_std = jnp.sqrt(jnp.maximum(adv_second_moment_ema, jnp.float32(1e-6)))
                        return mult * agg_fn(advantage / adv_std)
                    elif self.critic_normalization == "distributional" and value_params is not None:
                        v_mean, v_log_var = self._value_net.apply(value_params, obs_batch)
                        advantage = q - v_mean
                        adv_std = jnp.sqrt(jnp.maximum(jnp.exp(v_log_var), jnp.float32(1e-6)))
                        return mult * agg_fn(advantage / adv_std)

                    return mult * agg_fn(q)

                def grad_guidance(x_in, t_idx):
                    # Use Q-gradient guidance only if lambda_for_step(t_idx) > 0; otherwise no guidance
                    def guided(x):
                        return jax.grad(lambda xx: q_mean_from_x(xx, t_idx))(x)

                    def unguided(x):
                        return jnp.zeros_like(x)

                    lam = lambda_for_step(t_idx, tfg_eta_current)
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
                        lambda_t = lambda_for_step(t, tfg_eta_current)
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
                    lambda_t = lambda_for_step(t, tfg_eta_current)
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
                q_means_f, q_vars_f = [], []
                for qp in q_params_tuple:
                    m, v = self.agent.q(qp, obs_batch, act_final)
                    q_means_f.append(m)
                    q_vars_f.append(v)
                q = aggregate_q_local(q_means_f, q_vars_f)
                if self.use_reward_critic:
                    q = q * (jnp.float32(1.0) / jnp.float32(1.0 - gamma_hp))
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
            policy_params, log_alpha, q_params_tuple, log_eta_scales_in, tfg_eta_current, value_params, adv_second_moment_ema, target_value_params, reward_scale_hp, x0_hat_clip_radius_hp, mala_adapt_rate_hp, guidance_mult_hp, gamma_hp, q_critic_agg_idx_hp = params

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
                    num_q_local = len(q_params_tuple)
                    critic_idx = jax.random.randint(q_agg_key, (), 0, num_q_local)

                    def aggregate_q_local(q_means, q_vars):
                        stacked = jnp.stack(q_means, axis=0)
                        return stacked[critic_idx]

                else:
                    key_x, loop_key = jax.random.split(single_key)
                    aggregate_q_local = aggregate_q_fn

                def q_min_from_x0(x0_in):
                    q_means, q_vars = [], []
                    for qp in q_params_tuple:
                        m, v = self.agent.q(qp, obs_batch, x0_in)
                        q_means.append(m)
                        q_vars.append(v)
                    return aggregate_q_local(q_means, q_vars)

                def q_mean_from_x(x_in, t_idx):
                    noise_pred = self.agent.policy(policy_params, obs_batch, x_in, t_idx)
                    x0_hat = (
                        x_in * B.sqrt_recip_alphas_cumprod[t_idx]
                        - noise_pred * B.sqrt_recipm1_alphas_cumprod[t_idx]
                    )
                    x0_hat = jnp.clip(x0_hat, -x0_hat_clip_radius_hp, x0_hat_clip_radius_hp)
                    q = q_min_from_x0(x0_hat)
                    if self.use_reward_critic:
                        q = q * (jnp.float32(1.0) / jnp.float32(1.0 - gamma_hp))

                    agg_fn = jnp.sum if self.batch_independent_guidance else jnp.mean
                    mult = guidance_mult_hp

                    # Critic normalization: use (Q - V) / std instead of Q
                    if self.critic_normalization == "ema" and value_params is not None:
                        v_params_for_adv = target_value_params if self.td_value_training and target_value_params is not None else value_params
                        v = self._value_net.apply(v_params_for_adv, obs_batch)
                        advantage = q - v
                        adv_std = jnp.sqrt(jnp.maximum(adv_second_moment_ema, jnp.float32(1e-6)))
                        return mult * agg_fn(advantage / adv_std)
                    elif self.critic_normalization == "distributional" and value_params is not None:
                        v_mean, v_log_var = self._value_net.apply(value_params, obs_batch)
                        advantage = q - v_mean
                        adv_std = jnp.sqrt(jnp.maximum(jnp.exp(v_log_var), jnp.float32(1e-6)))
                        return mult * agg_fn(advantage / adv_std)

                    return mult * agg_fn(q)

                def grad_guidance(x_in, t_idx):
                    def guided(x):
                        return jax.grad(lambda xx: q_mean_from_x(xx, t_idx))(x)

                    def unguided(x):
                        return jnp.zeros_like(x)

                    lam = lambda_for_step(t_idx, tfg_eta_current)
                    return jax.lax.cond(lam > 0.0, guided, unguided, x_in)

                def energy_total(t, x):
                    def only_model(x_in):
                        return energy_model(t, x_in), jnp.float32(0.0)

                    def with_q(x_in):
                        E_mod = energy_model(t, x_in)
                        noise_pred = self.agent.policy(policy_params, obs_batch, x_in, t)
                        x0_hat = (
                            x_in * B.sqrt_recip_alphas_cumprod[t]
                            - noise_pred * B.sqrt_recipm1_alphas_cumprod[t]
                        )
                        clip_frac = jnp.mean((jnp.abs(x0_hat) > x0_hat_clip_radius_hp).astype(jnp.float32))
                        x0_hat_clipped = jnp.clip(x0_hat, -x0_hat_clip_radius_hp, x0_hat_clip_radius_hp)
                        q_min = q_min_from_x0(x0_hat_clipped)
                        lambda_t = lambda_for_step(t, tfg_eta_current)
                        return E_mod - lambda_t * q_min, clip_frac

                    lambda_t = lambda_for_step(t, tfg_eta_current)
                    return jax.lax.cond(lambda_t > 0.0, with_q, only_model, x)

                def predictor_step(t_idx: jax.Array, x_in: jax.Array, k_in: jax.Array):
                    noise_pred = self.agent.policy(policy_params, obs_batch, x_in, t_idx)
                    # Scale base score by energy_multiplier (tempers the base distribution)
                    # Guidance component is NOT scaled - only the base policy score
                    noise_pred_scaled = self.energy_multiplier * noise_pred
                    if self.mala_guided_predictor:
                        grad_q = grad_guidance(x_in, t_idx)
                        sigma_t = B.sqrt_one_minus_alphas_cumprod[t_idx]
                        lambda_t = lambda_for_step(t_idx, tfg_eta_current)
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
                if self.mala_recurrence_cap:
                    # Per-level cap: sigma_t = sqrt(1 - alpha_bar_t), matching recurrence drift
                    eta_cap_per_level = B.sqrt_one_minus_alphas_cumprod
                    log_eta_max = jnp.log(jnp.max(eta_cap_per_level) / eta_base_min)
                else:
                    eta_cap_per_level = None
                    log_eta_max = jnp.log(jnp.float32(0.5) / eta_base_min)

                if self.mala_per_level_eta:

                    def level_body(i, carry):
                        x_curr, k, log_eta_scales, acc_sum, acc_count, per_level_acc, per_level_clip_frac = carry
                        t = timesteps - 1 - i

                        def do_mala_level(carry_in, t_corr: jax.Array):
                            x_in, k_in, log_eta_scales_in, acc_sum_in, acc_count_in = carry_in
                            eta_base_t = jnp.maximum(B.betas[t_corr], jnp.float32(1e-8))
                            log_eta_scale0 = log_eta_scales_in[t_corr]
                            eta_upper = eta_cap_per_level[t_corr] if eta_cap_per_level is not None else jnp.float32(0.5)

                            def mala_body(_, state):
                                x_step, k_step, log_eta_scale, acc_sum_step, acc_count_step, clip_sum_step = state

                                E_x, vjp_x, clip_x = jax.vjp(lambda xx: energy_total(t_corr, xx), x_step, has_aux=True)
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

                                E_x_prop, vjp_x_prop, _clip_prop = jax.vjp(lambda xx: energy_total(t_corr, xx), x_prop, has_aux=True)
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

                        if self.mala_predictor_first:
                            x_pred, k = predictor_step(t, x_curr, k)
                            t_corr = t - 1

                            def _with_mala(carry_in):
                                return do_mala_level(carry_in, t_corr)

                            def _no_mala(carry_in):
                                return (*carry_in, jnp.float32(0.0))

                            acc_sum_before = acc_sum
                            acc_count_before = acc_count
                            x_curr, k, log_eta_scales, acc_sum, acc_count, clip_sum_level = jax.lax.cond(
                                t > 0,
                                _with_mala,
                                _no_mala,
                                (x_pred, k, log_eta_scales, acc_sum, acc_count),
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

                else:

                    log_eta_scale_shared0 = log_eta_scales_init[0]

                    def level_body(i, carry):
                        x_curr, k, log_eta_scale_shared, acc_sum, acc_count, per_level_acc, per_level_clip_frac = carry
                        t = timesteps - 1 - i

                        def do_mala_shared(carry_in, t_corr: jax.Array):
                            x_in, k_in, log_eta_scale_in, acc_sum_in, acc_count_in = carry_in
                            eta_base_t = jnp.maximum(B.betas[t_corr], jnp.float32(1e-8))
                            eta_upper = eta_cap_per_level[t_corr] if eta_cap_per_level is not None else jnp.float32(0.5)

                            def mala_body(_, state):
                                x_step, k_step, log_eta_scale, acc_sum_step, acc_count_step, clip_sum_step = state

                                E_x, vjp_x, clip_x = jax.vjp(lambda xx: energy_total(t_corr, xx), x_step, has_aux=True)
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

                                E_x_prop, vjp_x_prop, _clip_prop = jax.vjp(lambda xx: energy_total(t_corr, xx), x_prop, has_aux=True)
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

                            x_out, k_out, log_eta_scale_out, acc_sum_level, acc_count_level, clip_sum_level = jax.lax.fori_loop(
                                0,
                                self.agent.mala_steps,
                                mala_body,
                                (x_in, k_in, log_eta_scale_in, jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0)),
                            )

                            return (
                                x_out,
                                k_out,
                                log_eta_scale_out,
                                acc_sum_in + acc_sum_level,
                                acc_count_in + acc_count_level,
                                clip_sum_level,
                            )

                        if self.mala_predictor_first:
                            x_pred, k = predictor_step(t, x_curr, k)
                            t_corr = t - 1

                            def _with_mala(carry_in):
                                return do_mala_shared(carry_in, t_corr)

                            def _no_mala(carry_in):
                                return (*carry_in, jnp.float32(0.0))

                            acc_sum_before = acc_sum
                            acc_count_before = acc_count
                            x_curr, k, log_eta_scale_shared, acc_sum, acc_count, clip_sum_level = jax.lax.cond(
                                t > 0,
                                _with_mala,
                                _no_mala,
                                (x_pred, k, log_eta_scale_shared, acc_sum, acc_count),
                            )
                            level_acc_sum = acc_sum - acc_sum_before
                            level_acc_count = acc_count - acc_count_before
                            level_acc = level_acc_sum / jnp.maximum(level_acc_count, jnp.float32(1.0))
                            per_level_acc = per_level_acc.at[t].set(level_acc)
                            mala_steps_f = jnp.float32(self.agent.mala_steps)
                            per_level_clip_frac = per_level_clip_frac.at[t].set(clip_sum_level / jnp.maximum(mala_steps_f, jnp.float32(1.0)))
                            return x_curr, k, log_eta_scale_shared, acc_sum, acc_count, per_level_acc, per_level_clip_frac

                        acc_sum_before = acc_sum
                        acc_count_before = acc_count
                        x_curr, k, log_eta_scale_shared, acc_sum, acc_count, clip_sum_level = do_mala_shared(
                            (x_curr, k, log_eta_scale_shared, acc_sum, acc_count), t
                        )
                        level_acc_sum = acc_sum - acc_sum_before
                        level_acc_count = acc_count - acc_count_before
                        level_acc = level_acc_sum / jnp.maximum(level_acc_count, jnp.float32(1.0))
                        per_level_acc = per_level_acc.at[t].set(level_acc)
                        mala_steps_f = jnp.float32(self.agent.mala_steps)
                        per_level_clip_frac = per_level_clip_frac.at[t].set(clip_sum_level / jnp.maximum(mala_steps_f, jnp.float32(1.0)))
                        x_next, k = predictor_step(t, x_curr, k)
                        return x_next, k, log_eta_scale_shared, acc_sum, acc_count, per_level_acc, per_level_clip_frac

                    init_per_level_acc = jnp.zeros((timesteps,), dtype=jnp.float32)
                    init_per_level_clip_frac = jnp.zeros((timesteps,), dtype=jnp.float32)
                    x_final, _, log_eta_scale_shared_final, acc_sum_final, acc_count_final, per_level_acc_out, per_level_clip_frac_out = jax.lax.fori_loop(
                        0,
                        timesteps,
                        level_body,
                        (x0, loop_key, log_eta_scale_shared0, jnp.float32(0.0), jnp.float32(0.0), init_per_level_acc, init_per_level_clip_frac),
                    )

                    log_eta_scales_out = jnp.full((timesteps,), log_eta_scale_shared_final, dtype=jnp.float32)
                    mala_acc_rate = acc_sum_final / jnp.maximum(acc_count_final, jnp.float32(1.0))
                    mala_eta_scale = jnp.exp(log_eta_scale_shared_final)

                act_final = jnp.clip(x_final, -1.0, 1.0)
                act_final = x_final if self.latent_action_space else act_final
                q_means_f, q_vars_f = [], []
                for qp in q_params_tuple:
                    m, v = self.agent.q(qp, obs_batch, act_final)
                    q_means_f.append(m)
                    q_vars_f.append(v)
                q = aggregate_q_fn(q_means_f, q_vars_f)
                if self.use_reward_critic:
                    q = q * (jnp.float32(1.0) / jnp.float32(1.0 - gamma_hp))
                return act_final, q, log_eta_scales_out, mala_acc_rate, mala_eta_scale, per_level_acc_out, per_level_clip_frac_out

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
                    act, q, log_eta_scales_out, mala_acc_rate, mala_eta_scale, pl_acc, pl_clip = single_sampler(key_sample, log_eta_scales_in)
                else:
                    keys = jax.random.split(key_sample, self.agent.num_particles)
                    acts, qs, log_eta_scales_outs, mala_acc_rates, mala_eta_scales, pl_accs, pl_clips = jax.vmap(
                        lambda k: single_sampler(k, log_eta_scales_in)
                    )(keys)
                    act, q = select_action_from_particles(acts, qs, key_select)
                    log_eta_scales_out = jnp.mean(log_eta_scales_outs, axis=0)
                    mala_acc_rate = jnp.mean(mala_acc_rates, axis=0)
                    mala_eta_scale = jnp.mean(mala_eta_scales, axis=0)
                    pl_acc = jnp.mean(pl_accs, axis=0)
                    pl_clip = jnp.mean(pl_clips, axis=0)

                if not self.no_entropy_tuning:
                    act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.agent.noise_scale

                if single:
                    return act[0], q[0], log_eta_scales_out, mala_acc_rate, mala_eta_scale, pl_acc, pl_clip
                else:
                    return act, q, log_eta_scales_out, mala_acc_rate, mala_eta_scale, pl_acc, pl_clip

            def sample_with_particles_metrics_details(
                key: jax.Array,
                log_alpha: jax.Array,
                single: bool,
                single_sampler,
                log_eta_scales_in: jax.Array,
            ):
                key_sample, key_select, noise_key = jax.random.split(key, 3)

                if self.agent.num_particles == 1:
                    act, q, log_eta_scales_out, mala_acc_rate, mala_eta_scale, pl_acc, pl_clip = single_sampler(key_sample, log_eta_scales_in)
                    acts = act[None, ...]
                    qs = q[None, ...]
                    probs = jnp.ones_like(qs)
                else:
                    keys = jax.random.split(key_sample, self.agent.num_particles)
                    acts, qs, log_eta_scales_outs, mala_acc_rates, mala_eta_scales, pl_accs, pl_clips = jax.vmap(
                        lambda k: single_sampler(k, log_eta_scales_in)
                    )(keys)
                    log_eta_scales_out = jnp.mean(log_eta_scales_outs, axis=0)
                    mala_acc_rate = jnp.mean(mala_acc_rates, axis=0)
                    mala_eta_scale = jnp.mean(mala_eta_scales, axis=0)
                    pl_acc = jnp.mean(pl_accs, axis=0)
                    pl_clip = jnp.mean(pl_clips, axis=0)

                    if np.isinf(particle_selection_lambda):
                        idx = jnp.argmax(qs, axis=0)
                        probs = jax.nn.one_hot(idx, self.agent.num_particles).T
                    else:
                        logits = lambda_sel * qs
                        logits = logits - jnp.max(logits, axis=0, keepdims=True)
                        probs = jax.nn.softmax(logits, axis=0)

                    act, _q_sel = select_action_from_particles(acts, qs, key_select)

                if not self.no_entropy_tuning:
                    act = act + jax.random.normal(noise_key, act.shape) * jnp.exp(log_alpha) * self.agent.noise_scale

                if single:
                    return act[0], acts, probs, log_eta_scales_out, mala_acc_rate, mala_eta_scale, pl_acc, pl_clip
                else:
                    return act, acts, probs, log_eta_scales_out, mala_acc_rate, mala_eta_scale, pl_acc, pl_clip

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
        ):
            act, q, log_eta_scales_out, _, _, _, _ = stateless_get_action_mala_full(
                key,
                params,
                obs,
                aggregate_q_fn,
                randomize_q=randomize_q,
            )
            return act, q, log_eta_scales_out

        def sample_action_with_agg_metrics(
            key: jax.Array,
            params,
            obs: jax.Array,
            aggregate_q_fn,
            randomize_q: bool = False,
        ):
            policy_params, log_alpha, q_params_tuple, log_eta_scales_in, _tfg_eta_current, _value_params, _adv_second_moment_ema, _target_value_params, _reward_scale_hp, _x0_hat_clip_radius_hp, _mala_adapt_rate_hp, _guidance_mult_hp, gamma_hp, _q_critic_agg_idx_hp = params
            if self.agent.energy_mode and self.agent.mala_steps > 0:
                act, _q, log_eta_scales_out, mala_acc_rate, mala_eta_scale, pl_acc, pl_clip = stateless_get_action_mala_full(
                    key,
                    params,
                    obs,
                    aggregate_q_fn,
                    randomize_q=randomize_q,
                )
                return act, log_eta_scales_out, mala_acc_rate, mala_eta_scale, pl_acc, pl_clip
            else:
                _ts = self.agent.num_timesteps
                act, _q, log_eta_scales_out = sample_action_with_agg(key, params, obs, aggregate_q_fn, randomize_q=randomize_q)
                return act, log_eta_scales_out, jnp.float32(jnp.nan), jnp.float32(jnp.nan), jnp.full((_ts,), jnp.nan), jnp.full((_ts,), jnp.nan)

        def sample_action_with_particle_details_metrics(
            key: jax.Array,
            params,
            obs: jax.Array,
            aggregate_q_fn,
            randomize_q: bool = False,
        ):
            if self.agent.energy_mode and self.agent.mala_steps > 0:
                return stateless_get_action_mala_particles_full(key, params, obs, aggregate_q_fn, randomize_q=randomize_q)
            act, log_eta_scales_out, mala_acc_rate, mala_eta_scale, pl_acc, pl_clip = sample_action_with_agg_metrics(
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
                return act, acts, probs, log_eta_scales_out, mala_acc_rate, mala_eta_scale, pl_acc, pl_clip
            acts = act[None, ...]
            probs = jnp.ones((1, act.shape[0]), dtype=jnp.float32)
            return act, acts, probs, log_eta_scales_out, mala_acc_rate, mala_eta_scale, pl_acc, pl_clip

        def stateless_get_action_base(
            key: jax.Array,
            params,
            obs: jax.Array,
            aggregate_q_fn,
            randomize_q: bool = False,
        ) -> jax.Array:
            policy_params, log_alpha, q_params_tuple, log_eta_scales_in, _tfg_eta_current, _value_params, _adv_second_moment_ema, _target_value_params, _reward_scale_hp, _x0_hat_clip_radius_hp, _mala_adapt_rate_hp, _guidance_mult_hp, gamma_hp, _q_critic_agg_idx_hp = params

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
                    num_q_local = len(q_params_tuple)
                    critic_idx = jax.random.randint(q_agg_key, (), 0, num_q_local)

                    def aggregate_q_local(q_means, q_vars):
                        stacked = jnp.stack(q_means, axis=0)
                        return stacked[critic_idx]

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
                q_means, q_vars = [], []
                for qp in q_params_tuple:
                    m, v = self.agent.q(qp, obs_batch, act)
                    q_means.append(m)
                    q_vars.append(v)
                q = aggregate_q_local(q_means, q_vars)
                if self.use_reward_critic:
                    q = q * (jnp.float32(1.0) / jnp.float32(1.0 - gamma_hp))
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
            return sample_action_with_agg(key, params, obs, aggregate_q_fn_outer, randomize_q=(self.q_critic_agg == "random"))

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
            policy_params, log_alpha, q_params_tuple, log_eta_scales_in, tfg_eta_current, value_params, adv_second_moment_ema, target_value_params, reward_scale_hp, x0_hat_clip_radius_hp, mala_adapt_rate_hp, guidance_mult_hp, gamma_hp, q_critic_agg_idx_hp = params
            # Per-slot aggregate under vmap_mean_min; otherwise use the
            # string-branching aggregate_q_fn_outer unchanged.
            aggregate_q_fn_here = (
                make_aggregate_q_fn(q_critic_agg_idx_hp)
                if q_critic_agg == "vmap_mean_min"
                else aggregate_q_fn_outer
            )

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

            def eval_q_agg(act):
                q_means, q_vars = [], []
                for qp in q_params_tuple:
                    m, v = self.agent.q(qp, obs_batch, act)
                    q_means.append(m)
                    q_vars.append(v)
                return aggregate_q_fn_here(q_means, q_vars)

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
                q = eval_q_agg(act)
                if self.use_reward_critic:
                    q = q * (jnp.float32(1.0) / jnp.float32(1.0 - gamma_hp))
                return q  # [batch_size]

            # Sample from tilted policy π_tilt (guided) and evaluate Q
            def sample_tilted_with_q(sample_key: jax.Array):
                # Use the guided sampling path
                act, _q, _ = sample_action_with_agg(
                    sample_key,
                    params,
                    obs_batch,
                    aggregate_q_fn_here,
                    randomize_q=False,
                )
                # Evaluate Q on the tilted sample
                q = eval_q_agg(act)
                if self.use_reward_critic:
                    q = q * (jnp.float32(1.0) / jnp.float32(1.0 - gamma_hp))
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
            term1 = tfg_eta_current * jnp.mean(q_tilted, axis=0)

            # Second term: log E_{a~π_0}[e^{η Q(s,a)}] = logsumexp(η Q) - log(n_samples)
            # Shape: [batch_size]
            term2 = jax.scipy.special.logsumexp(tfg_eta_current * q_base, axis=0) - jnp.log(jnp.float32(n_samples))

            # KL = term1 - term2, averaged over batch
            kl_per_obs = term1 - term2
            kl_mean = jnp.mean(kl_per_obs)

            return kl_mean

        self._estimate_md_kl = stateless_estimate_md_kl

    def get_policy_params(self):
        return (
            self.state.params.policy,
            self.state.params.log_alpha,
            self.state.params.q,
            self.state.log_eta_scales,
            self.state.tfg_eta,
            self.state.value_params,
            self.state.advantage_second_moment_ema,
            self.state.target_value_params,
            self.state.reward_scale,
            self.state.x0_hat_clip_radius,
            self.state.mala_adapt_rate,
            self.state.guidance_mult,
            self.state.gamma,
            self.state.q_critic_agg_idx,
        )

    def get_policy_params_to_save(self):
        return (
            self.state.params.target_policy,
            self.state.params.log_alpha,
            self.state.params.q,
            self.state.log_eta_scales,
            self.state.tfg_eta,
            self.state.value_params,
            self.state.advantage_second_moment_ema,
            self.state.target_value_params,
            self.state.reward_scale,
            self.state.x0_hat_clip_radius,
            self.state.mala_adapt_rate,
            self.state.guidance_mult,
            self.state.gamma,
            self.state.q_critic_agg_idx,
        )

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)

        # --- Migrate old q1/q2 params to tuple-based format ---
        loaded_params = state.params
        if hasattr(loaded_params, 'q1') and not hasattr(loaded_params, 'q'):
            # Old format: Diffv2Params(q1, q2, target_q1, target_q2, policy, target_policy, log_alpha)
            loaded_params = Diffv2Params(
                q=(loaded_params.q1, loaded_params.q2),
                target_q=(loaded_params.target_q1, loaded_params.target_q2),
                policy=loaded_params.policy,
                target_policy=loaded_params.target_policy,
                log_alpha=loaded_params.log_alpha,
            )
        loaded_opt = state.opt_state
        if hasattr(loaded_opt, 'q1') and not hasattr(loaded_opt, 'q'):
            loaded_opt = Diffv2OptStates(
                q=(loaded_opt.q1, loaded_opt.q2),
                policy=loaded_opt.policy,
                log_alpha=loaded_opt.log_alpha,
                value=getattr(loaded_opt, 'value', None),
            )
        frozen_q = getattr(state, 'frozen_q', loaded_params.q)
        train_policy = getattr(state, 'train_policy', True)

        if not hasattr(state, "log_eta_scales"):
            timesteps = self.agent.num_timesteps
            init_eta_scale = jnp.maximum(jnp.float32(self.mala_init_eta_scale), jnp.float32(1e-8))
            init_log_eta_scale = jnp.log(init_eta_scale)
            state = Diffv2TrainState(
                params=loaded_params,
                opt_state=loaded_opt,
                step=state.step,
                entropy=state.entropy,
                running_mean=state.running_mean,
                running_std=state.running_std,
                log_eta_scales=jnp.full((timesteps,), init_log_eta_scale, dtype=jnp.float32),
                tfg_eta=jnp.float32(self.tfg_eta),
                value_params=getattr(state, 'value_params', None),
                target_value_params=getattr(state, 'target_value_params', getattr(state, 'value_params', None)),
                advantage_second_moment_ema=getattr(state, 'advantage_second_moment_ema', jnp.float32(1.0)),
                frozen_q=frozen_q,
                train_policy=train_policy,
                gamma=getattr(state, 'gamma', jnp.float32(self.gamma)),
                polyak_tau=getattr(state, 'polyak_tau', jnp.float32(self.tau)),
                lr_q=getattr(state, 'lr_q', jnp.float32(self.lr_q)),
                lr_policy=getattr(state, 'lr_policy', jnp.float32(self.lr_policy)),
                guidance_mult=getattr(state, 'guidance_mult', jnp.float32(self.guidance_strength_multiplier)),
                adv_ema_tau=getattr(state, 'adv_ema_tau', jnp.float32(self.advantage_ema_tau)),
                shape_ema_tau=getattr(state, 'shape_ema_tau', jnp.float32(self.shape_ema_tau)),
                kl_budget_val=getattr(state, 'kl_budget_val', jnp.float32(self.kl_budget if self.kl_budget is not None else 1.0)),
                reward_scale=getattr(state, 'reward_scale', jnp.float32(self.reward_scale)),
                x0_hat_clip_radius=getattr(state, 'x0_hat_clip_radius', jnp.float32(self.x0_hat_clip_radius)),
                mala_adapt_rate=getattr(state, 'mala_adapt_rate', jnp.float32(self.mala_adapt_rate)),
                q_td_huber_width=getattr(state, 'q_td_huber_width', jnp.float32(self.q_td_huber_width)),
                shape3_ema_tau=getattr(state, 'shape3_ema_tau', jnp.float32(self.shape3_ema_tau)),
                q_critic_agg_idx=getattr(state, 'q_critic_agg_idx', jnp.float32(self.q_critic_agg_idx)),
            )
        elif not hasattr(state, "tfg_eta"):
            state = Diffv2TrainState(
                params=loaded_params,
                opt_state=loaded_opt,
                step=state.step,
                entropy=state.entropy,
                running_mean=state.running_mean,
                running_std=state.running_std,
                log_eta_scales=state.log_eta_scales,
                tfg_eta=jnp.float32(self.tfg_eta),
                value_params=getattr(state, 'value_params', None),
                target_value_params=getattr(state, 'target_value_params', getattr(state, 'value_params', None)),
                advantage_second_moment_ema=getattr(state, 'advantage_second_moment_ema', jnp.float32(1.0)),
                frozen_q=frozen_q,
                train_policy=train_policy,
                gamma=getattr(state, 'gamma', jnp.float32(self.gamma)),
                polyak_tau=getattr(state, 'polyak_tau', jnp.float32(self.tau)),
                lr_q=getattr(state, 'lr_q', jnp.float32(self.lr_q)),
                lr_policy=getattr(state, 'lr_policy', jnp.float32(self.lr_policy)),
                guidance_mult=getattr(state, 'guidance_mult', jnp.float32(self.guidance_strength_multiplier)),
                adv_ema_tau=getattr(state, 'adv_ema_tau', jnp.float32(self.advantage_ema_tau)),
                shape_ema_tau=getattr(state, 'shape_ema_tau', jnp.float32(self.shape_ema_tau)),
                kl_budget_val=getattr(state, 'kl_budget_val', jnp.float32(self.kl_budget if self.kl_budget is not None else 1.0)),
                reward_scale=getattr(state, 'reward_scale', jnp.float32(self.reward_scale)),
                x0_hat_clip_radius=getattr(state, 'x0_hat_clip_radius', jnp.float32(self.x0_hat_clip_radius)),
                mala_adapt_rate=getattr(state, 'mala_adapt_rate', jnp.float32(self.mala_adapt_rate)),
                q_td_huber_width=getattr(state, 'q_td_huber_width', jnp.float32(self.q_td_huber_width)),
                shape3_ema_tau=getattr(state, 'shape3_ema_tau', jnp.float32(self.shape3_ema_tau)),
                q_critic_agg_idx=getattr(state, 'q_critic_agg_idx', jnp.float32(self.q_critic_agg_idx)),
            )
        elif (not self.mala_per_level_eta) and hasattr(state, "log_eta_scales"):
            # Enforce shared eta-scale invariant when running in shared mode.
            timesteps = self.agent.num_timesteps
            shared = jnp.asarray(state.log_eta_scales)[0]
            state = Diffv2TrainState(
                params=loaded_params,
                opt_state=loaded_opt,
                step=state.step,
                entropy=state.entropy,
                running_mean=state.running_mean,
                running_std=state.running_std,
                log_eta_scales=jnp.full((timesteps,), shared, dtype=jnp.float32),
                tfg_eta=state.tfg_eta if hasattr(state, "tfg_eta") else jnp.float32(self.tfg_eta),
                value_params=getattr(state, 'value_params', None),
                target_value_params=getattr(state, 'target_value_params', getattr(state, 'value_params', None)),
                advantage_second_moment_ema=getattr(state, 'advantage_second_moment_ema', jnp.float32(self.initial_advantage_second_moment_ema)),
                frozen_q=frozen_q,
                train_policy=train_policy,
                gamma=getattr(state, 'gamma', jnp.float32(self.gamma)),
                polyak_tau=getattr(state, 'polyak_tau', jnp.float32(self.tau)),
                lr_q=getattr(state, 'lr_q', jnp.float32(self.lr_q)),
                lr_policy=getattr(state, 'lr_policy', jnp.float32(self.lr_policy)),
                guidance_mult=getattr(state, 'guidance_mult', jnp.float32(self.guidance_strength_multiplier)),
                adv_ema_tau=getattr(state, 'adv_ema_tau', jnp.float32(self.advantage_ema_tau)),
                shape_ema_tau=getattr(state, 'shape_ema_tau', jnp.float32(self.shape_ema_tau)),
                kl_budget_val=getattr(state, 'kl_budget_val', jnp.float32(self.kl_budget if self.kl_budget is not None else 1.0)),
                reward_scale=getattr(state, 'reward_scale', jnp.float32(self.reward_scale)),
                x0_hat_clip_radius=getattr(state, 'x0_hat_clip_radius', jnp.float32(self.x0_hat_clip_radius)),
                mala_adapt_rate=getattr(state, 'mala_adapt_rate', jnp.float32(self.mala_adapt_rate)),
                q_td_huber_width=getattr(state, 'q_td_huber_width', jnp.float32(self.q_td_huber_width)),
                shape3_ema_tau=getattr(state, 'shape3_ema_tau', jnp.float32(self.shape3_ema_tau)),
                q_critic_agg_idx=getattr(state, 'q_critic_agg_idx', jnp.float32(self.q_critic_agg_idx)),
            )
        else:
            # Already has all fields; just ensure tuple format and new fields
            state = Diffv2TrainState(
                params=loaded_params,
                opt_state=loaded_opt,
                step=state.step,
                entropy=state.entropy,
                running_mean=state.running_mean,
                running_std=state.running_std,
                log_eta_scales=state.log_eta_scales,
                tfg_eta=state.tfg_eta,
                dsac_omega_ema=getattr(state, 'dsac_omega_ema', jnp.float32(1.0)),
                dsac_b_ema=getattr(state, 'dsac_b_ema', jnp.float32(1.0)),
                value_params=getattr(state, 'value_params', None),
                target_value_params=getattr(state, 'target_value_params', getattr(state, 'value_params', None)),
                advantage_second_moment_ema=getattr(state, 'advantage_second_moment_ema', jnp.float32(self.initial_advantage_second_moment_ema)),
                advantage_third_moment_ema=getattr(state, 'advantage_third_moment_ema', jnp.float32(0.0)),
                dist_shift_covariance_ema=getattr(state, 'dist_shift_covariance_ema', jnp.float32(0.0)),
                dist_shift_shape_ema=getattr(state, 'dist_shift_shape_ema', jnp.float32(self.initial_dist_shift_shape_ema)),
                dist_shift_coeff_ema=getattr(state, 'dist_shift_coeff_ema', jnp.float32(self.initial_dist_shift_coeff_ema)),
                dist_shift_shape3_ema=getattr(state, 'dist_shift_shape3_ema', jnp.float32(0.0)),
                frozen_q=frozen_q,
                train_policy=train_policy,
                gamma=getattr(state, 'gamma', jnp.float32(self.gamma)),
                polyak_tau=getattr(state, 'polyak_tau', jnp.float32(self.tau)),
                lr_q=getattr(state, 'lr_q', jnp.float32(self.lr_q)),
                lr_policy=getattr(state, 'lr_policy', jnp.float32(self.lr_policy)),
                guidance_mult=getattr(state, 'guidance_mult', jnp.float32(self.guidance_strength_multiplier)),
                adv_ema_tau=getattr(state, 'adv_ema_tau', jnp.float32(self.advantage_ema_tau)),
                shape_ema_tau=getattr(state, 'shape_ema_tau', jnp.float32(self.shape_ema_tau)),
                kl_budget_val=getattr(state, 'kl_budget_val', jnp.float32(self.kl_budget if self.kl_budget is not None else 1.0)),
                reward_scale=getattr(state, 'reward_scale', jnp.float32(self.reward_scale)),
                x0_hat_clip_radius=getattr(state, 'x0_hat_clip_radius', jnp.float32(self.x0_hat_clip_radius)),
                mala_adapt_rate=getattr(state, 'mala_adapt_rate', jnp.float32(self.mala_adapt_rate)),
                q_td_huber_width=getattr(state, 'q_td_huber_width', jnp.float32(self.q_td_huber_width)),
                shape3_ema_tau=getattr(state, 'shape3_ema_tau', jnp.float32(self.shape3_ema_tau)),
                q_critic_agg_idx=getattr(state, 'q_critic_agg_idx', jnp.float32(self.q_critic_agg_idx)),
            )
        self.state = jax.device_put(state)

    def get_current_tfg_eta(self) -> float:
        return float(self.state.tfg_eta)

    def estimate_md_kl(self, key: jax.Array, obs: np.ndarray, n_samples: int = 16) -> float:
        """Estimate KL(π_tilt || π_0) for given observations.
        
        Uses the formula: KL = E_{a~π_tilt}[η Q(s,a)] - log E_{a~π_0}[e^{η Q(s,a)}]
        Returns 0.0 if tfg_eta is 0 (no tilting).
        """
        if float(self.state.tfg_eta) == 0.0:
            return 0.0
        kl = self._estimate_md_kl(key, self.get_policy_params(), jnp.asarray(obs), n_samples)
        return float(kl)

    def set_tfg_eta(self, new_tfg_eta: float) -> None:
        self.state = self.state._replace(tfg_eta=jnp.float32(new_tfg_eta))

    def set_train_policy(self, train: bool) -> None:
        """Enable/disable policy training (for soft policy iteration)."""
        self.state = self.state._replace(train_policy=train)
        if not train:
            # Freeze current Q params for guidance
            self.state = self.state._replace(frozen_q=self.state.params.q)

    def spi_q_values(self, obs: np.ndarray, action: np.ndarray):
        """Return (q_live, q_frozen) for SPI diagnostics."""
        obs_j, act_j = jnp.asarray(obs), jnp.asarray(action)
        q_live_means = [self.agent.q(qp, obs_j, act_j)[0] for qp in self.state.params.q]
        q_frozen_means = [self.agent.q(qp, obs_j, act_j)[0] for qp in self.state.frozen_q]
        q_live = sum(q_live_means) / len(q_live_means)
        q_frozen = sum(q_frozen_means) / len(q_frozen_means)
        return np.asarray(q_live), np.asarray(q_frozen)

    def _build_initial_state(self, params, value_params_init, value_opt_state_init):
        """Construct a fresh Diffv2TrainState from a given set of network params.

        Factored out of __init__ so vmap-mode setup can call it N times with
        different init seeds and stack the results along a leading seed axis.
        """
        frozen_q_init = tuple(jax.tree.map(lambda x: x, qp) for qp in params.q) if self.soft_pi_mode else None
        return Diffv2TrainState(
            params=params,
            opt_state=Diffv2OptStates(
                q=tuple(self.optim.init(qp) for qp in params.q),
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
                value=value_opt_state_init,
            ),
            step=jnp.int32(0),
            entropy=jnp.float32(0.0),
            running_mean=jnp.float32(0.0),
            running_std=jnp.float32(1.0),
            log_eta_scales=jnp.full((self._timesteps,), self._init_log_eta_scale, dtype=jnp.float32),
            tfg_eta=jnp.float32(self._init_tfg_eta),
            value_params=value_params_init,
            target_value_params=value_params_init if self._td_value_training else None,
            advantage_second_moment_ema=jnp.float32(self.initial_advantage_second_moment_ema),
            advantage_third_moment_ema=jnp.float32(0.0),
            dist_shift_covariance_ema=jnp.float32(0.0),
            dist_shift_shape_ema=jnp.float32(self.initial_dist_shift_shape_ema),
            dist_shift_coeff_ema=jnp.float32(self.initial_dist_shift_coeff_ema),
            dist_shift_shape3_ema=jnp.float32(0.0),
            frozen_q=frozen_q_init,
            train_policy=jnp.float32(1.0),
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
            q_critic_agg_idx=jnp.float32(self.q_critic_agg_idx),
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
        """Vmapped counterpart of get_action. Requires on_policy_ema mode.

        obs: numpy array of shape [N, num_envs, obs_dim].
        Returns (action [N, num_envs, act_dim], q_per_env [N, num_envs],
        v_per_env [N, num_envs]).
        """
        if not self.on_policy_ema:
            raise NotImplementedError(
                "VmapOffPolicyTrainer currently requires on_policy_ema mode (use --kl_budget)."
            )
        self._ensure_vmap_compiled()
        params = self.get_policy_params_to_save()
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
        v_params = (
            self.state.target_value_params
            if self.td_value_training and self.state.target_value_params is not None
            else self.state.value_params
        )
        obs_j = jnp.asarray(obs)
        v = self._value_net_apply_vmap(v_params, obs_j)
        v_per_env = np.asarray(v)  # [N, num_envs]
        return action_np, q_per_env, v_per_env

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

    def get_action(self, key: jax.Array, obs: np.ndarray):
        out = self._get_action(key, self.get_policy_params_to_save(), obs)
        if isinstance(out, tuple) and len(out) == 3:
            action, q, log_eta_scales = out
            self.state = self.state._replace(log_eta_scales=log_eta_scales)
            action_np = np.asarray(action)

            if self.on_policy_ema:
                # Return per-env Q and V arrays for on-policy EMA updates
                q_per_env = np.asarray(q)  # [num_envs]
                obs_j = jnp.asarray(obs)
                if obs_j.ndim == 1:
                    obs_j = obs_j[None, :]
                v_params = self.state.target_value_params if self.td_value_training and self.state.target_value_params is not None else self.state.value_params
                v = self._value_net.apply(v_params, obs_j)
                if isinstance(v, tuple):
                    v = v[0]
                v_per_env = np.asarray(v)  # [num_envs]
                return action_np, q_per_env, v_per_env
            else:
                q_agg = float(jnp.mean(q))
                q_var = self._compute_q_ensemble_var(action_np, obs)
                return action_np, q_agg, q_var
        elif isinstance(out, tuple) and len(out) == 2:
            action, log_eta_scales = out
            self.state = self.state._replace(log_eta_scales=log_eta_scales)
            return np.asarray(action), None, None
        else:
            return np.asarray(out), None, None

    def _compute_q_ensemble_var(self, action: np.ndarray, obs: np.ndarray) -> float:
        """Compute mean variance of Q across ensemble members for the given (obs, action)."""
        obs_j = jnp.asarray(obs)
        act_j = jnp.asarray(action)
        if obs_j.ndim == 1:
            obs_j = obs_j[None, :]
            act_j = act_j[None, :]
        if len(self.state.params.q) < 2:
            return 0.0
        if getattr(self, "_q_ensemble_var_jit", None) is None:
            q_fn = self.agent.q
            u_mode = self._entropic_u_mode
            w_mode = self._entropic_w_mode
            w_eps = getattr(self, "_entropic_w_eps", 1e-8)

            @jax.jit
            def _fn(q_params_tuple, s, a, tfg_eta):
                means = [q_fn(qp, s, a)[0] for qp in q_params_tuple]
                stacked = jnp.stack(means, axis=0)
                if u_mode:
                    stacked = stacked / tfg_eta
                elif w_mode:
                    stacked = jnp.log(jnp.maximum(stacked, w_eps)) / tfg_eta
                return jnp.mean(jnp.var(stacked, axis=0))

            self._q_ensemble_var_jit = _fn
        return float(self._q_ensemble_var_jit(
            tuple(self.state.params.q), obs_j, act_j, self.state.tfg_eta
        ))

    def evaluate_value(self, obs: np.ndarray) -> "float | None":
        """Return mean V(obs) under current value params, or None if no V network."""
        if self._value_net is None or self.state.value_params is None:
            return None
        obs_j = jnp.asarray(obs)
        if obs_j.ndim == 1:
            obs_j = obs_j[None, :]
        v = self._value_net.apply(self.state.value_params, obs_j)
        if isinstance(v, tuple):
            v = v[0]  # distributional V returns (mean, log_var)
        return float(jnp.mean(v))

    def evaluate_d_psi(self, obs: np.ndarray, action: np.ndarray) -> "np.ndarray | None":
        """Return per-env mean D_ψ(obs, action) averaged across Q ensemble, or None."""
        if self.agent.d_psi is None:
            return None
        obs_j = jnp.asarray(obs)
        act_j = jnp.asarray(action)
        if obs_j.ndim == 1:
            obs_j = obs_j[None, :]
            act_j = act_j[None, :]
        d_vals = [self.agent.d_psi(qp, obs_j, act_j) for qp in self.state.params.q]
        d_mean = sum(d_vals) / len(d_vals)
        return np.asarray(d_mean)

    def update_supervised(self, key: jax.Array, data: Experience) -> Metric:
        from relax.algorithm.base import _split_info
        self.state, info = self._update_supervised(key, self.state, data)
        return _split_info(info)