"""Type definitions for the MGMD algorithm.

Split out from ``mgmd.py`` so a walkthrough reader opening that file lands
directly on ``class MGMD`` without paging past ~100 lines of NamedTuples
and dataclass declarations. The types here are:

* :class:`Diffv2OptStates`        — per-network optax states.
* :class:`MalaSampleResult`       — output bundle of one MALA sampler pass.
* :class:`HParams`                — per-seed vmappable hyperparameter block,
                                    lives at ``Diffv2TrainState.hp``.
* :class:`Diffv2TrainState`       — full MGMD train state (a single pytree).
* :class:`MGMDConfig`             — frozen container of every scalar
                                    hyperparameter, built once in
                                    ``scripts/train_mujoco.py``.

``MGMD._build_initial_state`` materialises ``HParams`` from a ``MGMDConfig``
with an explicit field-by-field ``HParams(gamma=cfg.gamma, ...)`` literal,
so the cfg→state name mapping (e.g. ``cfg.advantage_ema_tau`` → ``hp.adv_ema_tau``) lives
at the call site rather than in a separate translation table.
"""
from dataclasses import dataclass
from typing import NamedTuple, Optional

import jax
import optax
import haiku as hk

from relax.network.actor_critic import ActorCriticParams



class Diffv2OptStates(NamedTuple):
    q: tuple  # tuple of N optax.OptState, one per Q network
    policy: optax.OptState
    value: optax.OptState = None  # Optional V(s) network for normalized advantage guidance


class MalaSampleResult(NamedTuple):
    """Result bundle from one MALA-corrected sampling pass.

    Returned by :func:`stateless_get_action_mala_full`. Public sampler
    (``stateless_get_action_env``) returns ``(action, q, log_eta_scales)``;
    the TD update path additionally consumes the per-level diagnostics for
    wandb logging.
    """
    action: jax.Array
    q: jax.Array
    log_eta_scales: jax.Array
    per_level_acc: jax.Array
    per_level_clip: jax.Array
    # [T]: score-optimal cost of the interval between knots j and j+1, harvested
    # from the MH drifts. Present only under ``schedule_cost``. The last entry
    # scores the N(0, I) initialization rather than an interval; the schedule
    # update drops it.
    schedule_cost: Optional[jax.Array] = None
    # [T+1, K, batch, act_dim]: one sample of each level's law, x_levels[j] ~
    # rho_lambda_j, with j = T the N(0, I) the chain starts from (lambda = -inf).
    # Present only under ``collect_levels``; nothing in training needs it, it is
    # what lets scripts/test_adaptive_schedule.py re-derive the cost from scratch.
    x_levels: Optional[jax.Array] = None


class HParams(NamedTuple):
    """Per-seed vmappable hyperparameter scalars.

    Scalars in single-seed mode; under :class:`VmapOffPolicyTrainer` each
    field becomes a ``[N]``-shaped array so vmap maps one scalar value to
    each seed. Lives as ``Diffv2TrainState.hp`` so the algorithmic state
    fields above stay visually separate from the frozen-per-seed hp block.
    """
    gamma: jax.Array = 0.99            # discount factor
    q_polyak_tau: jax.Array = 0.005     # target-Q soft-update rate
    policy_polyak_tau: jax.Array = 0.005  # target-policy soft-update rate
    # Update periods, held as float32 (like every other hp) so they are
    # vmappable per-seed ablations; ``is_due`` floors them to int >= 1.
    delay_target_q_update: jax.Array = 2.0
    delay_policy_update: jax.Array = 2.0
    delay_target_policy_update: jax.Array = 2.0
    lr_q: jax.Array = 1e-4              # Q optimizer LR (applied as -lr*update)
    lr_policy: jax.Array = 1e-4         # policy optimizer LR
    guidance_mult: jax.Array = 1.0      # guidance strength multiplier
    guidance_mult_increasing: jax.Array = 0.0  # 1.0 => use normalized alpha_t schedule from TFG
    adv_ema_tau: jax.Array = 0.0005     # advantage-moment EMA rate
    shape_ema_tau: jax.Array = 0.0001   # dimensionless shape EMA rate
    adv_norm_ema_rate: jax.Array = 0.001  # EMA rate for --ema_advantage_normalization running mean/std
    s_hat_ema_rate: jax.Array = 0.001   # EMA rate for the --estimate_s_hat log-space accumulators
    kl_budget_val: jax.Array = 1.0      # KL budget δ (host uses for β cap)
    reward_scale: jax.Array = 1.0       # reward scaling for TD / huber δ
    x0_hat_clip_radius: jax.Array = 1.0  # clip radius for x0 prediction
    mala_adapt_rate: jax.Array = 0.05   # MALA step-size adaptation rate
    q_td_huber_width: jax.Array = float("inf")  # Q TD huber loss width (in reward units)
    alpha: jax.Array = 1.0              # composite MD energy scale α: π_new ∝ π_old^α · exp(β·Q)
    T: jax.Array = 0.0                  # composite MD entropy temperature T
    eta: jax.Array = 0.0                # composite MD step size η
    s_hat: jax.Array = 1.0              # assumed clean-action scale; shifts the noise schedule by -2 log s_hat in log-SNR
    # --beta_schedule_type adaptive: fraction of the way the knots move toward the
    # equal-cost layout each update. Lives here rather than on the frozen config
    # so it is a per-seed (vmap-packable) ablation axis like lr_q. The levels
    # themselves are a [T] array, so they sit on the state, not here.
    noise_schedule_gamma: jax.Array = 1e-3
    # Updates to wait before the knots start moving. The cost is only meaningful
    # once the denoiser is Tweedie-consistent; before that d x0hat/dx scales like
    # 1/sqrt(abar) instead of sqrt(abar) and the cost explodes at the noisy end.
    noise_schedule_warmup: jax.Array = 1e5


class Diffv2TrainState(NamedTuple):
    params: ActorCriticParams
    opt_state: Diffv2OptStates
    step: int
    log_eta_scales: jax.Array
    beta: jax.Array                              # composite MD guidance strength β; scales Q in E_total = α·E_θ - β·Q
    # Normalized advantage guidance state
    value_params: hk.Params = None             # V(s) network params (optional)
    advantage_second_moment_ema: float = 1.0   # M = EMA(E[A²]); used by ema_eta.py to compute β = sqrt(2δ/M)
    advantage_third_moment_ema: float = 0.0
    dist_shift_covariance_ema: float = 0.0
    dist_shift_shape_ema: float = -1.0        # EMA of s₂ = (2γc + κ₃) / v^(3/2), dimensionless shape
    q_running_mean: float = 0.0              # EMA(batch mean of online agg-Q at next-actions); --ema_advantage_normalization
    # Guidance-Q divisor. --ema_advantage_normalization puts the EMA of the POOLED
    # batch std here (the diffusion_policy_online_rl port); --ema_within_advantage_
    # normalization puts the contraharmonic mean of the WITHIN-state sd here instead.
    q_running_std: float = 1.0
    # relax/algorithm/normalizers.py accumulators over the K denoised actions per
    # state: linear-space moments of the within-state sd of Q (-> advantage scale)
    # and log-space moments of the per-state action sd (-> s_hat and tau^2).
    sigma_q_within_ema: float = 0.0
    sigma_q_within_sq_ema: float = 0.0
    log_s_ema: float = 0.0
    log_s_sq_ema: float = 0.0
    policy_loss: jax.Array = 0.0             # last computed policy loss; held constant on non-update steps
    # [T] free log-SNR knots, cleanest first, under --beta_schedule_type adaptive;
    # None for every fixed schedule family. Entries 0 and T-1 are the pinned ends
    # (--noise_schedule_log_snr_{max,min}), which the equal-cost update returns
    # unchanged, so the range stays fixed without being stored separately.
    log_snr_levels: Optional[jax.Array] = None
    # [batch*K*--distillation_buffer_size, obs_dim + act_dim] ring of the tilted
    # (next_obs | action) targets the policy is distilled onto; see
    # relax/algorithm/distillation.py. None only for states built outside MGMD.
    distill_buffer: Optional[jax.Array] = None
    hp: HParams = HParams()


@dataclass(frozen=True)
class MGMDConfig:
    """All scalar hyperparameters of the MGMD algorithm, frozen at construction.

    Constructed once in ``scripts/train_mujoco.py`` from CLI args and handed
    to :class:`MGMD`. Field names match argparse attribute names so the
    walkthrough has a single source of truth for what each knob controls.
    """
    gamma: float = 0.99
    q_polyak_tau: float = 0.005
    policy_polyak_tau: float = 0.005
    lr_policy: float = 1e-4
    lr_q: float = 1e-4
    delay_target_q_update: int = 2
    delay_policy_update: int = 2
    delay_target_policy_update: int = 2
    use_target_policy_training: bool = False
    use_target_q_sampling_training: bool = False
    reward_scale: float = 0.2
    q_agg_sample: str = "min"
    beta: float = 0.0
    x0_hat_clip_radius: float = 1.0
    mala_adapt_rate: float = 0.05
    denoising_predictor: str = "DDPM_mean"
    q_td_huber_width: float = float("inf")
    batch_independent_guidance: bool = False
    guidance_strength_multiplier: float = 1.0
    guidance_strength_schedule: str = "constant"
    alpha: float = 1.0
    T: float = 0.0
    eta: float = 0.0
    s_hat: float = 1.0
    advantage_normalization: bool = False
    advantage_ema_tau: float = 0.0005
    shape_ema_tau: float = 0.0001
    initial_advantage_second_moment_ema: float = 1.0
    initial_dist_shift_shape_ema: float = -1.0
    kl_budget: Optional[float] = None
    one_step_dist_shift_beta: bool = False
    guidance_gradient_space: str = "xt"
    critic_update_steps: int = 1
    policy_update_steps: int = 1
    num_denoised_actions: int = 1
    # The distillation buffer's capacity is batch_size * num_denoised_actions *
    # distillation_buffer_size, so the trainer's minibatch size has to be known
    # here to size it; it is the same --batch_size the trainer is handed.
    batch_size: int = 256
    distillation_buffer_size: int = 1
    distillation_steps: int = 1
    batch_advantage_normalization: bool = False
    q_loss_normalization: bool = False
    ema_advantage_normalization: bool = False
    ema_within_advantage_normalization: bool = False
    advantage_norm_ema_rate: float = 0.001
    estimate_s_hat: bool = False
    s_hat_ema_rate: float = 0.001
    lr_anneal: bool = False
    # Defaults reproduce diffusion_policy_online_rl's default LR-vs-ENV-step curve:
    # its policy schedule is linear(begin=2.5e4, steps=5e4) in POLICY-OPTIM steps
    # with lr 3e-4 -> 3e-5 (factor 0.1); at its defaults there are 10 env steps per
    # policy-optim step (num_vec_envs=5 * delay_update=2), so begin/steps map to
    # 2.5e5 / 5e5 env steps. Annealing here is a pure function of env steps, so this
    # curve is independent of this codebase's own num_vec_envs / update_per_iteration.
    lr_anneal_end_factor: float = 0.1
    lr_anneal_transition_begin: int = 250000
    lr_anneal_transition_steps: int = 500000
    orthogonal_init: bool = False
    latent_action: bool = False
    guidance_snr_anneal: str = "none"
    noise_schedule_gamma: float = 1e-3
    noise_schedule_warmup: float = 1e5
    noise_schedule_log_snr_max: float = 15.0
    noise_schedule_log_snr_min: float = -15.0

    @classmethod
    def from_args(cls, args) -> "MGMDConfig":
        """Build a frozen MGMDConfig from the argparse ``Namespace`` produced
        by :mod:`relax.cli.train_args`. Handles the ``--lr`` → ``--lr_q`` /
        ``--lr_policy`` fallback so the single source of truth for the
        config-from-CLI mapping lives here, not in ``train_mujoco.py``.
        """
        lr_policy = args.lr if args.lr_policy is None else args.lr_policy
        lr_q = args.lr if args.lr_q is None else args.lr_q
        guidance_strength_multiplier = args.guidance_strength_multiplier
        guidance_strength_schedule = "constant"
        if isinstance(guidance_strength_multiplier, str):
            guidance_strength_schedule = guidance_strength_multiplier
            guidance_strength_multiplier = 1.0
        return cls(
            gamma=args.gamma,
            q_polyak_tau=args.q_polyak_tau,
            policy_polyak_tau=args.policy_polyak_tau,
            lr_policy=float(lr_policy),
            lr_q=float(lr_q),
            delay_target_q_update=args.delay_target_q_update,
            delay_policy_update=args.delay_policy_update,
            delay_target_policy_update=args.delay_target_policy_update,
            use_target_policy_training=args.use_target_policy_training,
            use_target_q_sampling_training=args.use_target_q_sampling_training,
            critic_update_steps=args.critic_update_steps,
            policy_update_steps=args.policy_update_steps,
            reward_scale=args.reward_scale,
            q_agg_sample=args.q_agg_sample,
            beta=args.beta,
            x0_hat_clip_radius=args.x0_hat_clip_radius,
            mala_adapt_rate=args.mala_adapt_rate,
            denoising_predictor=args.denoising_predictor,
            q_td_huber_width=args.q_td_huber_width,
            batch_independent_guidance=args.batch_independent_guidance,
            guidance_strength_multiplier=float(guidance_strength_multiplier),
            guidance_strength_schedule=guidance_strength_schedule,
            alpha=args.alpha,
            T=args.T,
            eta=args.eta,
            s_hat=args.s_hat,
            advantage_normalization=args.advantage_normalization,
            advantage_ema_tau=args.advantage_ema_tau,
            shape_ema_tau=args.shape_ema_tau,
            initial_advantage_second_moment_ema=args.initial_advantage_second_moment_ema,
            initial_dist_shift_shape_ema=args.initial_dist_shift_shape_ema,
            kl_budget=args.kl_budget,
            one_step_dist_shift_beta=args.one_step_dist_shift_beta,
            guidance_gradient_space=args.guidance_gradient_space,
            num_denoised_actions=args.num_denoised_actions,
            batch_size=args.batch_size,
            distillation_buffer_size=args.distillation_buffer_size,
            distillation_steps=args.distillation_steps,
            batch_advantage_normalization=args.batch_advantage_normalization,
            q_loss_normalization=args.q_loss_normalization,
            ema_advantage_normalization=args.ema_advantage_normalization,
            ema_within_advantage_normalization=args.ema_within_advantage_normalization,
            advantage_norm_ema_rate=args.advantage_norm_ema_rate,
            estimate_s_hat=args.estimate_s_hat,
            s_hat_ema_rate=args.s_hat_ema_rate,
            lr_anneal=args.lr_anneal,
            lr_anneal_end_factor=args.lr_anneal_end_factor,
            lr_anneal_transition_begin=args.lr_anneal_transition_begin,
            lr_anneal_transition_steps=args.lr_anneal_transition_steps,
            orthogonal_init=args.orthogonal_init,
            latent_action=args.latent_action,
            guidance_snr_anneal=args.guidance_snr_anneal,
            noise_schedule_gamma=args.noise_schedule_gamma,
            noise_schedule_warmup=args.noise_schedule_warmup,
            noise_schedule_log_snr_max=args.noise_schedule_log_snr_max,
            noise_schedule_log_snr_min=args.noise_schedule_log_snr_min,
        )
