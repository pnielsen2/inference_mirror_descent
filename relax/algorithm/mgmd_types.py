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


class HParams(NamedTuple):
    """Per-seed vmappable hyperparameter scalars.

    Scalars in single-seed mode; under :class:`VmapOffPolicyTrainer` each
    field becomes a ``[N]``-shaped array so vmap maps one scalar value to
    each seed. Lives as ``Diffv2TrainState.hp`` so the algorithmic state
    fields above stay visually separate from the frozen-per-seed hp block.
    """
    gamma: jax.Array = 0.99            # discount factor
    polyak_tau: jax.Array = 0.005       # target net soft-update rate
    lr_q: jax.Array = 1e-4              # Q optimizer LR (applied as -lr*update)
    lr_policy: jax.Array = 1e-4         # policy optimizer LR
    guidance_mult: jax.Array = 1.0      # guidance strength multiplier
    adv_ema_tau: jax.Array = 0.0005     # advantage-moment EMA rate
    shape_ema_tau: jax.Array = 0.0001   # dimensionless shape EMA rate
    kl_budget_val: jax.Array = 1.0      # KL budget δ (host uses for β cap)
    reward_scale: jax.Array = 1.0       # reward scaling for TD / huber δ
    x0_hat_clip_radius: jax.Array = 1.0  # clip radius for x0 prediction
    mala_adapt_rate: jax.Array = 0.05   # MALA step-size adaptation rate
    q_td_huber_width: jax.Array = float("inf")  # Q TD huber loss width (in reward units)
    alpha: jax.Array = 1.0              # composite MD energy scale α: π_new ∝ π_old^α · exp(β·Q)
    T: jax.Array = 0.0                  # composite MD entropy temperature T
    eta: jax.Array = 0.0                # composite MD step size η


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
    policy_loss: jax.Array = 0.0             # last computed policy loss; held constant on non-update steps
    hp: HParams = HParams()


@dataclass(frozen=True)
class MGMDConfig:
    """All scalar hyperparameters of the MGMD algorithm, frozen at construction.

    Constructed once in ``scripts/train_mujoco.py`` from CLI args and handed
    to :class:`MGMD`. Field names match argparse attribute names so the
    walkthrough has a single source of truth for what each knob controls.
    """
    gamma: float = 0.99
    polyak_tau: float = 0.005
    lr_policy: float = 1e-4
    lr_q: float = 1e-4
    delay_update: int = 2
    reward_scale: float = 0.2
    q_agg_sample: str = "min"
    beta: float = 0.0
    x0_hat_clip_radius: float = 1.0
    mala_adapt_rate: float = 0.05
    denoising_predictor: str = "DDPM_mean"
    q_td_huber_width: float = float("inf")
    batch_independent_guidance: bool = False
    guidance_strength_multiplier: float = 1.0
    alpha: float = 1.0
    T: float = 0.0
    eta: float = 0.0
    advantage_normalization: bool = False
    advantage_ema_tau: float = 0.0005
    shape_ema_tau: float = 0.0001
    initial_advantage_second_moment_ema: float = 1.0
    initial_dist_shift_shape_ema: float = -1.0
    kl_budget: Optional[float] = None
    one_step_dist_shift_beta: bool = False

    @classmethod
    def from_args(cls, args) -> "MGMDConfig":
        """Build a frozen MGMDConfig from the argparse ``Namespace`` produced
        by :mod:`relax.cli.train_args`. Handles the ``--lr`` → ``--lr_q`` /
        ``--lr_policy`` fallback so the single source of truth for the
        config-from-CLI mapping lives here, not in ``train_mujoco.py``.
        """
        lr_policy = args.lr if args.lr_policy is None else args.lr_policy
        lr_q = args.lr if args.lr_q is None else args.lr_q
        return cls(
            gamma=args.gamma,
            polyak_tau=args.polyak_tau,
            lr_policy=float(lr_policy),
            lr_q=float(lr_q),
            delay_update=args.delay_update,
            reward_scale=args.reward_scale,
            q_agg_sample=args.q_agg_sample,
            beta=args.beta,
            x0_hat_clip_radius=args.x0_hat_clip_radius,
            mala_adapt_rate=args.mala_adapt_rate,
            denoising_predictor=args.denoising_predictor,
            q_td_huber_width=args.q_td_huber_width,
            batch_independent_guidance=args.batch_independent_guidance,
            guidance_strength_multiplier=args.guidance_strength_multiplier,
            alpha=args.alpha,
            T=args.T,
            eta=args.eta,
            advantage_normalization=args.advantage_normalization,
            advantage_ema_tau=args.advantage_ema_tau,
            shape_ema_tau=args.shape_ema_tau,
            initial_advantage_second_moment_ema=args.initial_advantage_second_moment_ema,
            initial_dist_shift_shape_ema=args.initial_dist_shift_shape_ema,
            kl_budget=args.kl_budget,
            one_step_dist_shift_beta=args.one_step_dist_shift_beta,
        )
