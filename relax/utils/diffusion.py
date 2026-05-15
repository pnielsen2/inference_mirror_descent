from typing import Protocol, Optional
from dataclasses import dataclass

import numpy as np
import jax, jax.numpy as jnp
import optax

class DiffusionModel(Protocol):
    def __call__(self, t: jax.Array, x: jax.Array) -> jax.Array:
        ...

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

@dataclass(frozen=True)
class GaussianDiffusion:
    num_timesteps: int
    beta_schedule_scale: float = 0.3
    beta_schedule_type: str = 'linear'
    x_recon_clip_radius: Optional[float] = 1.0
    snr_max: float = 124.0

    def beta_schedule(self):
        with jax.ensure_compile_time_eval():
            target_abar_0 = self.snr_max / (1.0 + self.snr_max)

            if self.beta_schedule_type == 'constant_kl':
                betas = BetaScheduleCoefficients.constant_kl_beta_schedule(
                    self.num_timesteps, snr_max=self.snr_max)
            elif self.beta_schedule_type == 'cosine':
                raw_betas = BetaScheduleCoefficients.cosine_beta_schedule(self.num_timesteps)
                scale = (1.0 - target_abar_0) / raw_betas[0]
                betas = np.clip(scale * raw_betas, 0, 0.999)
            elif self.beta_schedule_type == 'linear':
                raw_betas = BetaScheduleCoefficients.linear_beta_schedule(self.num_timesteps)
                scale = (1.0 - target_abar_0) / raw_betas[0]
                betas = np.clip(scale * raw_betas, 0, 0.999)
            return BetaScheduleCoefficients.from_beta(betas)

    def p_mean_variance(self, t: int, x: jax.Array, noise_pred: jax.Array):
        B = self.beta_schedule()
        x_recon = x * B.sqrt_recip_alphas_cumprod[t] - noise_pred * B.sqrt_recipm1_alphas_cumprod[t]
        if self.x_recon_clip_radius is not None:
            r = jnp.float32(self.x_recon_clip_radius)
            x_recon = jnp.clip(x_recon, -r, r)
        model_mean = x_recon * B.posterior_mean_coef1[t] + x * B.posterior_mean_coef2[t]
        model_log_variance = B.posterior_log_variance_clipped[t]
        return model_mean, model_log_variance

    def q_sample(self, t: int, x_start: jax.Array, noise: jax.Array):
        B = self.beta_schedule()
        return B.sqrt_alphas_cumprod[t] * x_start + B.sqrt_one_minus_alphas_cumprod[t] * noise

    def p_loss(self, key: jax.Array, model: DiffusionModel, t: jax.Array, x_start: jax.Array):
        """Standard diffusion denoising loss: mean( (eps_pred - eps)**2 ).

        Equivalent to upstream `inference_mirror_descent`'s
        `weighted_p_loss(weights=ones, ..., reduction="mean")` after XLA
        elides the multiply-by-ones. Note that upstream uses
        `optax.squared_error = (x-y)**2`, NOT `optax.l2_loss` which is
        `0.5 * (x-y)**2`; the factor of 2 matters for bit-exact
        reproducibility against the reference run.
        """
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]

        noise = jax.random.normal(key, x_start.shape)
        x_noisy = jax.vmap(self.q_sample)(t, x_start, noise)
        noise_pred = model(t, x_noisy)
        loss = optax.squared_error(noise_pred, noise)
        return loss.mean()
