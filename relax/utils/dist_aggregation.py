"""
Distribution aggregation utilities for distributional critics.

These functions aggregate normal distributions across actions (mixture) and 
across Q networks (treating as independent random variables).

Key design: Non-distributional critics work as a special case with variance=0,
which reduces all aggregation operations to the original point-estimate behavior.
"""

from typing import Tuple
import jax
import jax.numpy as jnp


# =============================================================================
# Mixture aggregation (across actions for a single Q)
# =============================================================================

def mixture_mean_var(means: jax.Array, variances: jax.Array, weights: jax.Array = None) -> Tuple[jax.Array, jax.Array]:
    """
    Compute mean and variance of a mixture of normal distributions.
    
    Given N normal distributions N(mu_i, sigma_i^2) with weights w_i,
    the mixture has:
        mean = sum(w_i * mu_i)
        var = sum(w_i * sigma_i^2) + sum(w_i * mu_i^2) - mean^2
            = sum(w_i * (sigma_i^2 + mu_i^2)) - mean^2
    
    Args:
        means: Shape [N, ...] - means of each component
        variances: Shape [N, ...] - variances of each component
        weights: Shape [N, ...] - weights (will be normalized). If None, uniform weights.
    
    Returns:
        (mixture_mean, mixture_variance) each with shape [...]
    """
    if weights is None:
        weights = jnp.ones_like(means) / means.shape[0]
    else:
        # Normalize weights
        weights = weights / jnp.sum(weights, axis=0, keepdims=True)
    
    # Mixture mean
    mixture_mean = jnp.sum(weights * means, axis=0)
    
    # Mixture variance: E[X^2] - E[X]^2
    # E[X^2] = sum(w_i * (var_i + mu_i^2))
    second_moment = jnp.sum(weights * (variances + means ** 2), axis=0)
    mixture_var = second_moment - mixture_mean ** 2
    
    # Ensure non-negative variance (numerical stability)
    mixture_var = jnp.maximum(mixture_var, jnp.float32(0.0))
    
    return mixture_mean, mixture_var


# =============================================================================
# Aggregation across Q networks (treating as independent random variables)
# =============================================================================

def aggregate_independent_min(mean1: jax.Array, var1: jax.Array, 
                              mean2: jax.Array, var2: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Aggregate two independent normal distributions using min.
    
    For point estimates (var=0), this is just min(mean1, mean2).
    For distributions, we approximate min(X1, X2) using the Clark approximation:
    If X1 ~ N(mu1, sigma1^2) and X2 ~ N(mu2, sigma2^2) are independent, then
    min(X1, X2) ≈ N(mu_min, sigma_min^2) where:
        theta = sqrt(sigma1^2 + sigma2^2)
        alpha = (mu1 - mu2) / theta
        mu_min = mu1 * Phi(-alpha) + mu2 * Phi(alpha) - theta * phi(alpha)
        sigma_min^2 ≈ (sigma1^2 + mu1^2) * Phi(-alpha) + (sigma2^2 + mu2^2) * Phi(alpha)
                      - 2 * mu_min^2 - theta * (mu1 + mu2) * phi(alpha)
    
    For var1=var2=0, reduces to min(mean1, mean2).
    """
    # Handle point estimate case (both variances are zero or very small)
    eps = jnp.float32(1e-8)
    total_var = var1 + var2
    is_point_estimate = total_var < eps
    
    # Point estimate result
    point_mean = jnp.minimum(mean1, mean2)
    point_var = jnp.zeros_like(mean1)
    
    # Distributional case: Clark approximation
    theta = jnp.sqrt(total_var + eps)  # Add eps for numerical stability
    alpha = (mean1 - mean2) / theta
    
    # Standard normal CDF and PDF
    phi_alpha = jnp.exp(-0.5 * alpha ** 2) / jnp.sqrt(2.0 * jnp.pi)
    Phi_alpha = jax.scipy.stats.norm.cdf(alpha)
    Phi_neg_alpha = 1.0 - Phi_alpha
    
    # Mean of min
    dist_mean = mean1 * Phi_neg_alpha + mean2 * Phi_alpha - theta * phi_alpha
    
    # Variance of min (simplified approximation)
    # Using: Var[min] ≈ E[X1^2]Phi(-α) + E[X2^2]Phi(α) - (E[X1] + E[X2])θφ(α) - E[min]^2
    second_moment = (var1 + mean1**2) * Phi_neg_alpha + (var2 + mean2**2) * Phi_alpha
    second_moment = second_moment - theta * (mean1 + mean2) * phi_alpha
    dist_var = jnp.maximum(second_moment - dist_mean**2, jnp.float32(0.0))
    
    # Select based on whether it's a point estimate
    out_mean = jnp.where(is_point_estimate, point_mean, dist_mean)
    out_var = jnp.where(is_point_estimate, point_var, dist_var)
    
    return out_mean, out_var


def aggregate_independent_max(mean1: jax.Array, var1: jax.Array,
                              mean2: jax.Array, var2: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Aggregate two independent normal distributions using max.
    
    Uses the identity: max(X1, X2) = -min(-X1, -X2)
    For point estimates (var=0), this is just max(mean1, mean2).
    """
    # max(X1, X2) = -min(-X1, -X2)
    neg_mean, neg_var = aggregate_independent_min(-mean1, var1, -mean2, var2)
    return -neg_mean, neg_var


def aggregate_independent_mean(mean1: jax.Array, var1: jax.Array,
                               mean2: jax.Array, var2: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Aggregate two independent normal distributions using mean.
    
    If X1 ~ N(mu1, sigma1^2) and X2 ~ N(mu2, sigma2^2), then
    (X1 + X2)/2 ~ N((mu1+mu2)/2, (sigma1^2+sigma2^2)/4)
    
    For point estimates (var=0), this is just (mean1+mean2)/2.
    """
    out_mean = 0.5 * (mean1 + mean2)
    out_var = 0.25 * (var1 + var2)
    return out_mean, out_var


def aggregate_independent_independent(mean1: jax.Array, var1: jax.Array,
                                      mean2: jax.Array, var2: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Keep Q1 and Q2 independent (no aggregation). Returns separate distributions for each.
    
    Returns: (mean1, var1, mean2, var2)
    """
    return mean1, var1, mean2, var2


def aggregate_independent_mixture(mean1: jax.Array, var1: jax.Array,
                                  mean2: jax.Array, var2: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Aggregate two distributions as a 50/50 mixture.
    
    Unlike 'mean' which averages the random variables, 'mixture' creates a 
    mixture distribution where we pick Q1 or Q2 with probability 0.5 each.
    
    Mixture mean = 0.5 * (mu1 + mu2)  [same as mean]
    Mixture var = 0.5 * (sigma1^2 + mu1^2) + 0.5 * (sigma2^2 + mu2^2) - mixture_mean^2
                = 0.5 * (sigma1^2 + sigma2^2) + 0.5 * (mu1^2 + mu2^2) - 0.25 * (mu1 + mu2)^2
                = 0.5 * (sigma1^2 + sigma2^2) + 0.25 * (mu1 - mu2)^2
    
    For point estimates, this gives variance = 0.25 * (mu1 - mu2)^2
    """
    out_mean = 0.5 * (mean1 + mean2)
    # Mixture variance = average of component variances + variance of component means
    out_var = 0.5 * (var1 + var2) + 0.25 * (mean1 - mean2) ** 2
    return out_mean, out_var


def aggregate_pick_min(mean1: jax.Array, var1: jax.Array,
                       mean2: jax.Array, var2: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Pick parameters of the critic with the lower mean (DSAC-T style).

    Unlike `aggregate_independent_min`, this does NOT form an approximate
    distribution for min(X1, X2). It simply selects either (mean1,var1) or
    (mean2,var2) based on which mean is smaller.
    """
    pick_1 = mean1 <= mean2
    out_mean = jnp.where(pick_1, mean1, mean2)
    out_var = jnp.where(pick_1, var1, var2)
    return out_mean, out_var


# =============================================================================
# Unified aggregation interface
# =============================================================================

def aggregate_q_distributions(
    mean1: jax.Array, var1: jax.Array,
    mean2: jax.Array, var2: jax.Array,
    mode: str
) -> Tuple[jax.Array, jax.Array]:
    """
    Aggregate two Q distributions based on the specified mode.
    
    Args:
        mean1, var1: First Q distribution parameters
        mean2, var2: Second Q distribution parameters  
        mode: One of "min", "max", "mean", "mixture", "pick_min"
    
    Returns:
        (aggregated_mean, aggregated_variance)
    """
    if mode == "min":
        return aggregate_independent_min(mean1, var1, mean2, var2)
    elif mode == "max":
        return aggregate_independent_max(mean1, var1, mean2, var2)
    elif mode == "mean":
        return aggregate_independent_mean(mean1, var1, mean2, var2)
    elif mode == "mixture":
        return aggregate_independent_mixture(mean1, var1, mean2, var2)
    elif mode == "pick_min":
        return aggregate_pick_min(mean1, var1, mean2, var2)
    else:
        raise ValueError(f"Unknown aggregation mode: {mode}")


def get_mean_only(mean: jax.Array, var: jax.Array) -> jax.Array:
    """Extract just the mean (for action selection where we ignore variance)."""
    return mean


# =============================================================================
# KL divergence for TD learning
# =============================================================================

def gaussian_kl_divergence(
    pred_mean: jax.Array, pred_var: jax.Array,
    target_mean: jax.Array, target_var: jax.Array
) -> jax.Array:
    """
    KL divergence D_KL(target || pred) between two Gaussians.
    
    D_KL(N(mu_t, sigma_t^2) || N(mu_p, sigma_p^2)) = 
        log(sigma_p/sigma_t) + (sigma_t^2 + (mu_t - mu_p)^2) / (2*sigma_p^2) - 0.5
    
    For numerical stability, we work with variances and add small epsilon.
    When target variance is 0 (point estimate target), this reduces to 
    MSE-like behavior.
    """
    eps = jnp.float32(1e-6)
    
    # Ensure variances are positive
    pred_var = jnp.maximum(pred_var, eps)
    target_var = jnp.maximum(target_var, eps)
    
    pred_std = jnp.sqrt(pred_var)
    target_std = jnp.sqrt(target_var)
    
    kl = (
        jnp.log(pred_std / target_std)
        + (target_var + (target_mean - pred_mean) ** 2) / (2.0 * pred_var)
        - 0.5
    )
    
    return kl


def gaussian_nll(
    pred_mean: jax.Array, pred_var: jax.Array,
    target_mean: jax.Array
) -> jax.Array:
    """
    Negative log-likelihood of target under predicted Gaussian.
    
    -log N(target | pred_mean, pred_var) = 0.5 * log(2*pi*pred_var) + (target - pred_mean)^2 / (2*pred_var)
    
    This is equivalent to Gaussian KL when target has no variance (point estimate target).
    """
    eps = jnp.float32(1e-6)
    pred_var = jnp.maximum(pred_var, eps)
    
    nll = 0.5 * jnp.log(2.0 * jnp.pi * pred_var) + (target_mean - pred_mean) ** 2 / (2.0 * pred_var)
    return nll


def gaussian_cross_entropy(
    pred_mean: jax.Array, pred_var: jax.Array,
    target_mean: jax.Array, target_var: jax.Array
) -> jax.Array:
    """
    Cross-entropy H(target, pred) = -E_{x~target}[log pred(x)] between two Gaussians.
    
    H(N(mu_t, sigma_t^2), N(mu_p, sigma_p^2)) = 
        0.5 * log(2*pi*sigma_p^2) + (sigma_t^2 + (mu_t - mu_p)^2) / (2*sigma_p^2)
    
    This equals KL(target || pred) + H(target), so gradients are identical to KL.
    When target_var = 0, this reduces exactly to gaussian_nll, making it directly
    comparable to NLL-based runs.
    """
    eps = jnp.float32(1e-6)
    pred_var = jnp.maximum(pred_var, eps)
    target_var = jnp.maximum(target_var, jnp.float32(0.0))
    
    ce = (
        0.5 * jnp.log(2.0 * jnp.pi * pred_var)
        + (target_var + (target_mean - pred_mean) ** 2) / (2.0 * pred_var)
    )
    return ce
