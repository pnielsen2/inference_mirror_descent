from typing import Optional
from dataclasses import dataclass

import numpy as np
import jax, jax.numpy as jnp

@dataclass(frozen=True)
class BetaScheduleCoefficients:
    betas: jax.Array
    alphas: jax.Array
    alphas_cumprod: jax.Array
    alphas_cumprod_prev: jax.Array
    sqrt_alphas_cumprod: jax.Array
    sqrt_one_minus_alphas_cumprod: jax.Array
    log_one_minus_alphas_cumprod: jax.Array
    sqrt_recip_alphas_cumprod: jax.Array
    sqrt_recipm1_alphas_cumprod: jax.Array
    posterior_variance: jax.Array
    posterior_log_variance_clipped: jax.Array
    posterior_mean_coef1: jax.Array
    posterior_mean_coef2: jax.Array

    @staticmethod
    def from_beta(betas: np.ndarray):
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = np.log(1. - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_log_variance_clipped = np.log(np.maximum(posterior_variance, 1e-20))
        posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)

        return BetaScheduleCoefficients(
            *jax.device_put((
                betas, alphas, alphas_cumprod, alphas_cumprod_prev,
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, log_one_minus_alphas_cumprod,
                sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod,
                posterior_variance, posterior_log_variance_clipped, posterior_mean_coef1, posterior_mean_coef2
            ))
        )

    @staticmethod
    def cosine_beta_schedule(timesteps: int):
        s = 0.008
        t = np.arange(0, timesteps + 1) / timesteps
        alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        alphas_cumprod /= alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = np.clip(betas, 0, 0.999)
        return betas
    
    @staticmethod
    def linear_beta_schedule(timesteps: int, beta_start=1e-4, beta_end=0.999):
        return np.linspace(beta_start, beta_end, timesteps, dtype=np.float64)

    @staticmethod
    def constant_kl_beta_schedule(timesteps: int, snr_max=1000.0):
        """Constant-KL schedule: noise levels equally spaced in log(1 + SNR).

        Derived from requiring constant mutual-information loss per step:
            I(x_0; x_{k-1}) - I(x_0; x_k) = const  for all k
        where I(x_0; x_t) = (d/2) log(1 + SNR(t)).

        This gives:
            alpha_bar_k = 1 - R^{-(T-k)/T},  R = 1 + SNR_max
        with k=0 cleanest (alpha_bar ~ 1) and k=T-1 noisiest (alpha_bar > 0).
        Pure noise (alpha_bar = 0) is an implicit endpoint outside the schedule;
        the reverse process starts from N(0,I) and the model at t=T-1 provides
        the first denoising step.
        """
        R = 1.0 + snr_max
        T = timesteps
        k = np.arange(T, dtype=np.float64)
        exponent = (T - k) / T
        alphas_cumprod = 1.0 - R ** (-exponent)
        # Derive betas: beta_k = 1 - alpha_bar_k / alpha_bar_{k-1}
        alphas_cumprod_with_1 = np.concatenate([[1.0], alphas_cumprod])
        betas = 1.0 - alphas_cumprod_with_1[1:] / alphas_cumprod_with_1[:-1]
        betas = np.clip(betas, 1e-8, 0.999)
        return betas

def build_beta_schedule(
    num_timesteps: int,
    beta_schedule_type: str,
    snr_max: float,
) -> BetaScheduleCoefficients:
    """Eager builder for the DDPM noise schedule.

    Returns a :class:`BetaScheduleCoefficients` holding all 13 precomputed
    arrays. Called once at ``ActorCritic.create`` time so downstream code
    can just attribute-access ``model.schedule.sqrt_alphas_cumprod`` etc.
    instead of calling a method that re-derives them.
    """
    target_abar_0 = snr_max / (1.0 + snr_max)

    if beta_schedule_type == 'constant_kl':
        betas = BetaScheduleCoefficients.constant_kl_beta_schedule(
            num_timesteps, snr_max=snr_max)
    elif beta_schedule_type == 'cosine':
        raw_betas = BetaScheduleCoefficients.cosine_beta_schedule(num_timesteps)
        scale = (1.0 - target_abar_0) / raw_betas[0]
        betas = np.clip(scale * raw_betas, 0, 0.999)
    elif beta_schedule_type == 'linear':
        raw_betas = BetaScheduleCoefficients.linear_beta_schedule(num_timesteps)
        scale = (1.0 - target_abar_0) / raw_betas[0]
        betas = np.clip(scale * raw_betas, 0, 0.999)
    else:
        raise ValueError(f"Unknown beta_schedule_type: {beta_schedule_type}")
    return BetaScheduleCoefficients.from_beta(betas)


def p_mean_variance(
    schedule: BetaScheduleCoefficients,
    t,
    x: jax.Array,
    noise_pred: jax.Array,
    x_recon_clip_radius: Optional[float] = None,
):
    """DDPM reverse-process posterior mean / log-variance.

    ``t`` is the diffusion timestep index (scalar). Returns
    ``(model_mean, model_log_variance)`` where ``model_log_variance`` is
    just ``schedule.posterior_log_variance_clipped[t]``.
    """
    x_recon = x * schedule.sqrt_recip_alphas_cumprod[t] - noise_pred * schedule.sqrt_recipm1_alphas_cumprod[t]
    if x_recon_clip_radius is not None:
        r = jnp.float32(x_recon_clip_radius)
        x_recon = jnp.clip(x_recon, -r, r)
    model_mean = x_recon * schedule.posterior_mean_coef1[t] + x * schedule.posterior_mean_coef2[t]
    model_log_variance = schedule.posterior_log_variance_clipped[t]
    return model_mean, model_log_variance
