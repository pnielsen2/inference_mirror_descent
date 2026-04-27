import argparse
import json
import os.path
from pathlib import Path
import time
from functools import partial

import jax, jax.numpy as jnp

from relax.algorithm.sac import SAC
from relax.algorithm.dsact import DSACT
from relax.algorithm.dacer import DACER
from relax.algorithm.dacer_doubleq import DACERDoubleQ
from relax.algorithm.qsm import QSM
from relax.algorithm.dipo import DIPO
from relax.algorithm.qvpo import QVPO
from relax.algorithm.sdac import SDAC
from relax.algorithm.dpmd import DPMD
from relax.algorithm.dpmd_bc import DPMDBC
from relax.algorithm.idem import IDEM
from relax.algorithm.pcmd import PCMD
from relax.buffer import TreeBuffer
from relax.network.sac import create_sac_net
from relax.network.dsact import create_dsact_net
from relax.network.dacer import create_dacer_net
from relax.network.dacer_doubleq import create_dacer_doubleq_net
from relax.network.qsm import create_qsm_net
from relax.network.dipo import create_dipo_net
from relax.network.diffv2 import create_diffv2_net
from relax.network.model_based import create_model_based_net
from relax.network.qvpo import create_qvpo_net
from relax.network.pcmd import create_pcmd_net
from relax.pcmd.levels import PcLevelsConfig
from relax.trainer.off_policy import OffPolicyTrainer
from relax.env import create_env, create_vector_env
from relax.utils.experience import Experience, ObsActionPair
from relax.utils.fs import PROJECT_ROOT
from relax.utils.random_utils import seeding

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="sdac")
    parser.add_argument("--env", type=str, default="HalfCheetah-v3")
    parser.add_argument("--backend", type=str, default="gymnasium", choices=["gymnasium", "mjx"], help="Physics backend: 'gymnasium' (standard MuJoCo C) or 'mjx' (JAX-based MuJoCo via Brax, differentiable).")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument(
        "--dummy_action_dim",
        type=int,
        default=0,
        help=(
            "If > 0, expand the environment action dimension to dummy_action_dim by adding dummy action coordinates "
            "that do not affect dynamics. The reward is modified by subtracting dummy_action_alpha * ||a_dummy||^2. "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--dummy_action_alpha",
        type=float,
        default=0.0,
        help="Penalty coefficient alpha for the dummy action L2 penalty.",
    )
    parser.add_argument("--num_vec_envs", type=int, default=5)
    parser.add_argument("--parallel_seeds", type=int, default=1, help="If > 1, train N independent seeds in parallel on a single device via jax.vmap. Requires --alg dpmd with --kl_budget or --kl_budget_per_dim. Env layout uses a single VectorEnv of size parallel_seeds * num_vec_envs.")
    parser.add_argument("--hp_pack", type=str, default=None, help="Path to a JSON file with per-seed hyperparameter overrides. Each key is an argparse attribute name of this script (e.g. 'tau', 'advantage_ema_tau', 'guidance_strength_multiplier', 'kl_budget', 'shape_ema_tau', 'seed') mapped to a list of length parallel_seeds. Applied after vmap state construction; internally translated to Diffv2TrainState field names via _CLI_TO_FIELD.")
    parser.add_argument("--hp_pack_inline", type=str, default=None, help="Inline JSON equivalent of --hp_pack. Same schema (key -> list of length parallel_seeds). Preferred over --hp_pack: keeps the pack fully self-contained in the CLI string (useful when the slurm command is surfaced by wandb) instead of embedding a path to a scratch file. When both are set, --hp_pack_inline wins.")
    parser.add_argument("--sweep_id", type=int, default=None, help="Launcher-assigned integer identifying this sweep. When set, every wandb run from this invocation is placed in wandb group 'sweep_<sweep_id>', and each per-vmap-slot run's config includes a 'config_tag' field built from sweep_id + the per-slot hyperparameters (excluding seed/env) so a single tag value filters wandb to all runs across envs/seeds that share this hp configuration.")
    parser.add_argument("--config_tag_keys", type=str, default=None, help="Comma-separated list of argparse attribute names whose values should be included in the per-slot config_tag. Typically set automatically by scripts/launch.py to the union of all --ablate hard+easy flags (minus env and seed). Values come from the hp_pack (per-slot) when the key is a pack key, else from this script's CLI args (shared across all vmap slots within the job).")
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)
    parser.add_argument("--start_step", type=int, default=int(3e4)) # other envs 3e4
    parser.add_argument("--total_step", type=int, default=int(1e6))
    parser.add_argument("--update_per_iteration", type=int, default=1)
    parser.add_argument("--supervised_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_policy", type=float, default=None)
    parser.add_argument("--lr_q", type=float, default=None)
    parser.add_argument("--lr_dyn", type=float, default=None)
    parser.add_argument("--lr_reward", type=float, default=None)
    parser.add_argument("--lr_value", type=float, default=None)
    parser.add_argument("--lr_schedule_end", type=float, default=3e-5)
    parser.add_argument("--alpha_lr", type=float, default=7e-3)
    parser.add_argument("--delay_alpha_update", type=int, default=250)
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for the Q critic. Default 0.99.")
    parser.add_argument("--tau", type=float, default=0.005, help="Polyak averaging coefficient for target network updates. Default 0.005.")
    parser.add_argument("--delay_update", type=int, default=2, help="Update policy and target networks every delay_update steps. Default 2.")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num_particles", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--cluster", default=False, action="store_true")
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--timing_log_every", type=int, default=0)

    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size for training updates.")
    parser.add_argument("--val_batch_size", type=int, default=None, help="Mini-batch size for validation updates. If None, defaults to batch_size.")
    parser.add_argument("--beta_schedule_scale", type=float, default=0.8)
    parser.add_argument("--beta_schedule_type", type=str, default='linear', help="Noise schedule type. 'linear': linear beta schedule. 'cosine': cosine schedule (Nichol & Dhariwal). 'constant_kl': constant mutual-information-loss per step, spacing noise levels uniformly in log(1+SNR).")
    parser.add_argument("--snr_max", type=float, default=124.0, help="Maximum SNR (at cleanest noise level). Controls alpha_bar_0 = snr_max/(1+snr_max). Default 124.0 matches the cosine schedule with s=0.008 offset at T=20. All schedule types use this to set the same clean endpoint, so you can switch between cosine/constant_kl/linear while keeping the noise range comparable.")
    parser.add_argument("--reward_scale", type=float, default=0.2, help="Scale factor applied to rewards before Q/value learning. Default 0.2 matches original DPMD. Set to 1.0 for clarity when using inference-time guidance (adjust tfg_eta/particle_selection_lambda accordingly).")
    parser.add_argument("--tfg_eta", type=float, default=0.0, help="Guidance strength lambda for dpmd training-free Q-guidance. If 0, no Q-guidance is applied.")
    parser.add_argument("--tfg_eta_end", type=float, default=None, help="Final value of tfg_eta at end of training. If set, tfg_eta interpolates log-linearly from tfg_eta to tfg_eta_end over training. Default None uses constant tfg_eta.")
    parser.add_argument("--log_md_kl_every", type=int, default=int(1e9), help="Log KL(π_tilt || π_0) every N steps. Default 1e9 (effectively disabled). Requires sampling from base policy, so adds overhead when enabled.")
    parser.add_argument("--critic_normalization", type=str, default="none", choices=["none", "ema", "distributional"], help="Normalization mode for Q guidance. 'none': use raw Q. 'ema': train V(s) to predict E[Q], normalize (Q-V) by sqrt(EMA[A^2]). 'distributional': train distributional V(s) outputting (mean, var), normalize by per-state std.")
    parser.add_argument("--kl_budget", type=float, default=None, help="Total KL divergence budget δ for guidance. Per-dimension budget is δ / act_dim. Sets η = sqrt(2δ), enables V network and on-policy advantage EMA. Replaces --critic_normalization ema --tfg_eta. Default None (disabled).")
    parser.add_argument("--kl_budget_per_dim", type=float, default=None, help="Per-dimension KL divergence budget δ_d for guidance. Total budget δ = δ_d * act_dim. Sets η = sqrt(2δ), enables V network and on-policy advantage EMA. Replaces --critic_normalization ema --tfg_eta. Default None (disabled).")
    parser.add_argument("--dist_shift_eta", action="store_true", default=False, help="Adaptive η from distribution-shift second-order expansion. Trains D_ψ head on Q networks, maintains on-policy EMAs for m2, m3, covariance. Computes η* = -m2^(3/2)/(2γc + m3), capped by KL budget. Requires --kl_budget or --kl_budget_per_dim (defaults to --kl_budget_per_dim=5.33 if neither set).")
    parser.add_argument("--one_step_dist_shift_eta", action="store_true", default=False, help="Adaptive η from second-order expansion using one-step Monte Carlo covariance estimate. No D_ψ head; estimates c from consecutive (A_t, A_{t+1}) pairs. Same η* formula as --dist_shift_eta but simpler. Requires --kl_budget or --kl_budget_per_dim (defaults to --kl_budget_per_dim=5.33 if neither set).")
    parser.add_argument("--two_step_dist_shift_eta", action="store_true", default=False, help="Adaptive η from third-order expansion using two-step MC estimates. Cubic surrogate for expected improvement; solves 1+2s₂x+3s₃x²=0 for optimal x=η√v. Falls back to quadratic (one-step) then KL ceiling. Requires --kl_budget or --kl_budget_per_dim (defaults to --kl_budget_per_dim=5.33 if neither set).")
    parser.add_argument("--direct_eta_coeff_ema", action="store_true", default=False, help="Alternative EMA parameterization for adaptive η: track B = EMA(2γc + κ₃) (the η²-coefficient) directly, instead of the dimensionless shape S = EMA((2γc+κ₃)/v^(3/2)). Quadratic step becomes η* = -V/B. Only affects the one-step quadratic path; ignored when --two_step_dist_shift_eta is on.")
    parser.add_argument("--equal_episode_weighting", action="store_true", default=False, help="Weight episodes equally in TD learning (sample episode then transition within) and update on-policy EMAs with episode-averaged quantities at episode boundaries instead of per-step.")
    parser.add_argument("--advantage_ema_tau", type=float, default=0.0005, help="Per-step EMA rate for advantage second/third moments. With --equal_episode_weighting, converted to per-episode rate via 1-(1-tau)^1000.")
    parser.add_argument("--shape_ema_tau", type=float, default=0.0001, help="Per-step EMA rate for dimensionless shape s2. With --equal_episode_weighting, converted to per-episode rate via 1-(1-tau)^1000.")
    parser.add_argument("--initial_advantage_second_moment_ema", type=float, default=1.0, help="Initial value for the advantage second moment EMA E[A^2].")
    parser.add_argument("--initial_dist_shift_shape_ema", type=float, default=-1.0, help="Initial value for the dimensionless distribution-shift shape EMA s2 = (2γc + κ₃) / v^(3/2).")
    parser.add_argument("--shape3_ema_tau", type=float, default=0.00005, help="Per-step EMA rate for third-order shape s3. With --equal_episode_weighting, converted to per-episode rate via 1-(1-tau)^1000.")
    parser.add_argument("--td_actions", type=int, default=1, help="Number of denoised next-actions to sample for TD target computation (mean Q over td_actions). Default 1.")
    parser.add_argument("--td_value_training", action="store_true", default=False, help="Train V(s) with the same TD backup target as Q instead of soft regression on E[Q]. Adds a target V network with Polyak averaging (same tau as Q). Advantages use V_target: A = Q - V_target.")
    parser.add_argument("--tfg_patience", type=float, default=float("inf"), help="If finite, reduce tfg_eta when the logged episode_return has failed to improve for tfg_patience consecutive logged points at the current tfg_eta. Default inf disables the scheduler.")
    parser.add_argument("--tfg_reduction_factor", type=float, default=1.0, help="Multiplicative factor applied to tfg_eta when plateau patience is exceeded. Default 1.0 is a no-op.")
    parser.add_argument("--x0_hat_clip_radius", type=float, default=float("inf"), help="Clipping radius r for Tweedie clean-action estimates x0_hat used inside guidance/Q evaluation. x0_hat is clipped to [-r, r] before being passed into Q / model-based objectives. Default inf (no clip); in non-latent mode the network-side denoising clip is separately hardcoded to 1.0 to match normalized action bounds.")
    parser.add_argument("--tfg_eta_schedule", type=str, default="constant", help="Schedule for guidance lambda vs noise level. 'constant': lambda_t=lambda. 'linear': linearly from lambda at t=0 to 0 at t=T. 'snr': lambda_t=lambda*alpha_bar_t (SNR-proportional), gives bounded c_t=lambda*(1-alpha_bar_t) and ~constant Hessian-term loss.")
    parser.add_argument("--fix_q_norm_bug", action="store_true", default=False, help="If set, use the corrected normalization (q_min - running_mean) / (running_std + eps) instead of the original buggy form.")
    parser.add_argument("--q_critic_agg", type=str, default="min", help="Aggregation for twin Qs when used as a signal (tilting, reweighting): 'min', 'mean', 'max', 'random', 'entropic', 'precision', or 'vmap_mean_min'. 'entropic' uses log(mean(exp(Q))), a soft-max. 'precision' uses inverse-variance-weighted mean (for distributional critics). TD targets always use 'min'. 'vmap_mean_min' makes both the training-time Q signal and the sampling-time aggregator a per-slot blend (1-idx)*mean + idx*min selected by --q_critic_agg_idx / hp_pack 'q_critic_agg_idx'; idx=0 exactly matches 'mean' and idx=1 exactly matches 'min'.")
    parser.add_argument("--q_critic_agg_idx", type=float, default=0.0, help="Per-slot vmappable selector active only when --q_critic_agg vmap_mean_min. 0.0 => Q-aggregation is mean everywhere it matters (training signal + sampling-time tilting + env-rollout action selection); 1.0 => min. Linear blend in between. Supplied per vmap slot via hp_pack['q_critic_agg_idx'].")
    parser.add_argument("--entropic_risk_beta", type=float, default=1.0, help="Temperature for entropic risk aggregation (only used with --q_critic_agg entropic). Aggregation is (1/beta)*log(mean(exp(beta*Q))). beta>0: risk-seeking/optimistic (beta->inf gives max). beta->0: risk-neutral (mean). beta<0: risk-averse/pessimistic (beta->-inf gives min). Default 1.0.")
    parser.add_argument("--particle_selection_lambda", type=float, default=float("inf"), help="Temperature for selecting an action among multiple particles using exp(particle_selection_lambda * Q(a)). Default inf reproduces argmax over Q.")
    parser.add_argument("--dpmd_recurrence_steps", type=int, default=0, help="Number of recurrence-style TFG inner steps per diffusion level for dpmd (0 disables recurrence).")
    parser.add_argument("--dpmd_constant_weight", action="store_true", default=False, help="If set for dpmd, disable Q-based reweighting in the diffusion score-matching loss and use constant weights.")
    parser.add_argument("--single_q_network", action="store_true", default=False, help="If set, train a single Q network instead of twin Q networks. The same Q is used for both Q1 and Q2.")
    parser.add_argument("--num_q_networks", type=int, default=2, help="Number of Q critic networks to train (default 2, i.e. twin Q).")
    parser.add_argument("--dpmd_use_reward_critic", action="store_true", default=False, help="If set for dpmd, replace the Q critic with a 1-step reward network trained from the replay buffer and use it (scaled by 1/(1-gamma)) for tilting/guidance.")
    parser.add_argument("--dpmd_pure_bc_training", action="store_true", default=False, help="If set for dpmd, train the diffusion policy purely by behavior cloning from replay actions (no critic-based tilting at training time), while still using the critic for inference-time guidance.")
    parser.add_argument("--dpmd_off_policy_td", action="store_true", default=False, help="If set for dpmd, use off-policy (buffer) actions for the critic TD target instead of on-policy (sampled from current policy) actions.")
    parser.add_argument("--dpmd_no_entropy_tuning", action="store_true", default=False, help="If set for dpmd, disable action noise and alpha/entropy tuning (makes DPMD more similar to dpmd_mb_pc).")
    parser.add_argument("--lr_annealing", action="store_true", default=False, help="(Deprecated) Shorthand for --lr_schedule_type linear. Kept for backward compat.")
    parser.add_argument("--lr_schedule_type", type=str, default="constant", choices=["constant", "linear", "cosine", "log_linear"], help="Policy LR schedule type. 'constant': no annealing. 'linear': linear decay. 'cosine': cosine decay. 'log_linear': geometric decay (linear in log-space). All decay to lr_schedule_end. Default constant.")
    parser.add_argument("--q_lr_schedule_type", type=str, default="constant", choices=["constant", "linear", "cosine", "log_linear"], help="Q LR schedule type. Same options as --lr_schedule_type. Default constant.")
    parser.add_argument("--q_lr_schedule_end", type=float, default=None, help="End LR for Q schedule. Default None uses --lr_schedule_end.")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"], help="Optimizer for policy and Q networks. Default adam.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW optimizer. Only used when --optimizer adamw. Default 1e-4.")
    parser.add_argument("--dpmd_bc_noisy_q_guided", action="store_true", default=False, help="For dpmd_bc: train Q on noisy forward-diffused actions and use guided sampling that evaluates Q on an intermediate noisy diffusion state.")
    parser.add_argument("--dpmd_bc_tfg_recurrence", action="store_true", default=False, help="For dpmd_bc: apply a recurrence-style training-free guidance step at inference time using Q as f(x_0)=exp(Q(x_0)).")
    parser.add_argument("--dpmd_bc_recurrence_steps", type=int, default=3, help="For dpmd_bc: number of recurrence steps per noise level when using tfg_recurrence.")
    parser.add_argument("--energy_param", action="store_true", default=False, help="Parameterize the diffusion model as the gradient of an energy function.")
    parser.add_argument("--mala_steps", type=int, default=0, help="Number of MALA correction steps per diffusion step. If > 0, automatically enables energy_param.")
    parser.add_argument("--gaussian_prior_baseline", action="store_true", default=False, help="Add Gaussian prior baseline so untrained model samples from N(0,I). For energy mode: E(x)=0.5||x||^2+E_net(x). For eps mode: eps(x,t)=sigma_t*x+eps_net(x,t).")
    parser.add_argument("--zero_init_q", action="store_true", default=False, help="Zero-initialize the last layer of the Q network so Q(s,a)=0 at init. Prevents early Hessian amplification in guided_ula_kl loss.")
    parser.add_argument("--mala_per_level_eta", action="store_true", default=False, help="If set, learn a separate MALA eta-scale for each diffusion noise level. Default behavior (flag off) uses a single shared eta-scale across all noise levels.")
    parser.add_argument("--mala_adapt_rate", type=float, default=0.05, help="Robbins-Monro adaptation rate for MALA log_eta_scale updates.")
    parser.add_argument("--mala_init_eta_scale", type=float, default=1.0, help="Initial MALA eta_scale multiplier (eta_k = eta_scale * beta_t, then clipped).")
    parser.add_argument("--mala_recurrence_cap", action="store_true", default=False, help="If set, cap MALA step size at sigma_t = sqrt(1 - alpha_bar_t) per noise level instead of the fixed 0.5 cap. This matches the drift of the Tweedie denoise-renoise recurrence, which is exact MCMC for unguided diffusion. Prevents pathological step size growth when tfg_eta is annealed to zero.")
    parser.add_argument("--mala_guided_predictor", action="store_true", default=False, help="If set, apply Q-guidance (TFG-style eps guidance) in the DDPM predictor step after each MALA correction step.")
    parser.add_argument("--mala_predictor_first", action="store_true", default=False, help="If set, use predictor->corrector ordering for MALA sampling: apply DDPM predictor step t->t-1 first, then run MALA corrector targeting energy_total at level t-1.")
    parser.add_argument("--ddim_predictor", action="store_true", default=False, help="If set, use deterministic DDIM-style predictor (no noise) instead of stochastic DDPM. Recommended for MALA sampling since the stochastic noise is redundant with MALA corrections.")
    parser.add_argument("--q_td_huber_width", type=float, default=float("inf"), help="Huber width (delta) for critic TD error in DPMD. Default inf recovers the current MSE TD loss. Effective width is scaled by reward_scale internally.")
    parser.add_argument("--decorrelated_q_batches", action="store_true", default=False, help="If set, Q1 and Q2 see shuffled versions of the same batch, decorrelating their training data to maintain disagreement at large batch sizes.")
    parser.add_argument("--q_bootstrap_agg", type=str, default="min", choices=["min", "independent", "mean", "mixture", "pick_min"], help="Aggregation mode for Q TD targets. 'min' (default): both Qs bootstrap from min(Q1_target, Q2_target). 'independent': Q1 bootstraps from Q1_target, Q2 from Q2_target. 'mean': both Qs bootstrap from mean(Q1_target, Q2_target). 'mixture': both Qs bootstrap from mixture distribution of Q1 and Q2. 'pick_min': pick (mean,var) from whichever target critic has lower mean (DSAC-T style).")
    parser.add_argument("--langevin_q_noise", action="store_true", default=False, help="If set, add SGLD noise to Q gradients: grad += sqrt(2*lr_q/batch_size)*noise. This approximates posterior sampling over Q-functions (LSAC-style).")
    parser.add_argument("--td_use_target_policy", action="store_true", default=False, help="If set, sample next-actions for TD backups from the target policy (Polyak-updated with --tau), matching the target-Q treatment. Default: online policy.")
    parser.add_argument("--batch_independent_guidance", action="store_true", default=False, help="If set, use jnp.sum instead of jnp.mean inside the guided predictor's q_mean_from_x, so the per-sample Q gradient is independent of batch size (fixes the 1/B attenuation).")
    parser.add_argument("--guidance_strength_multiplier", type=float, default=1.0, help="Constant multiplier applied to the guided-predictor Q scalar before jax.grad. Composes with --batch_independent_guidance.")
    parser.add_argument("--distributional_critic", action="store_true", default=False, help="If set, use a distributional Q critic that outputs (mean, variance) and is trained with cross-entropy loss instead of MSE.")
    parser.add_argument("--entropic_critic_param", type=str, default="none", choices=["none", "u", "w"], help="Entropic/exponential-moment critic parameterization. 'none' (default): standard Q critic. 'u': network outputs u=eta*Q_eta, trained with pseudo-loss exp(t-u)-(t-u)-1; Bellman target t=eta*r+(1-d)*gamma*u_target. 'w': network outputs W directly, trained with MSE; Bellman target y=exp(eta*r)*clip(w_next,eps)^gamma. Both modes approximate discounted entropic-risk RL (not exact for gamma<1).")
    parser.add_argument("--critic_grad_modifier", type=str, default="none", choices=["none", "variance_scaled", "natgrad"], help="Distributional critic gradient modifier. 'none' (default): standard CE gradients. 'variance_scaled': multiply CE by stop_gradient(var) (legacy/halfway). 'natgrad': full natural gradient in (mean,var) coordinates. Only effective with --distributional_critic.")
    parser.add_argument("--natural_gradient_critic", action="store_true", default=False, help="Deprecated alias for --critic_grad_modifier natgrad. If set and --critic_grad_modifier is left at default, enables natgrad. Only effective with --distributional_critic.")
    # DSAC-T refinements (Distributional SAC with Three Refinements, arxiv.org/abs/2310.05858)
    parser.add_argument("--dsac_expected_value_sub", action="store_true", default=False, help="DSAC-T Refinement 1: Expected value substitution. Use target_var=0 for the mean-related gradient, reducing variance. Only effective with --distributional_critic.")
    parser.add_argument("--dsac_adaptive_clip_xi", type=float, default=0.0, help="DSAC-T Refinement 2: Adaptive clipping factor xi. If > 0, clip TD error by xi*sigma in variance gradient (typically xi=3 for three-sigma rule). Set to 0 to disable. Only effective with --distributional_critic.")
    parser.add_argument("--dsac_omega_scaling", action="store_true", default=False, help="DSAC-T Refinement 3: Omega scaling. Scale loss by omega/(omega_ema+eps) where omega=mean(pred_var). Normalizes gradients across reward scales. Only effective with --distributional_critic.")
    parser.add_argument("--dsac_omega_tau", type=float, default=0.005, help="Polyak averaging rate for DSAC-T omega and b EMA updates. Only used when --dsac_omega_scaling or --dsac_adaptive_clip_xi > 0.")
    parser.add_argument("--latent_action_space", action="store_true", default=False, help="If set, treat policy/Q/diffusion actions as unconstrained latent variables z. Only when interacting with the environment (and for logging) we squash via a = 2*NormalCDF(z)-1. Replay buffer stores latents, so all models train on latent actions.")
    parser.add_argument("--latent_action_eps", type=float, default=1e-6, help="Epsilon for clamping probabilities when converting between env actions and latent actions.")
    parser.add_argument("--model_q_mc_samples", type=int, default=8)
    parser.add_argument("--energy_multiplier", type=float, default=1.0, help="Multiplier for base energy/score during sampling. Values < 1 temper (flatten) the base distribution, increasing entropy. Guidance signal is NOT scaled. Default 1.0 (no tempering).")
    parser.add_argument("--policy_loss_type", type=str, default="eps_mse", choices=["eps_mse", "ula_kl", "guided_ula_kl", "e2e_guided_ula_kl"], help="Policy diffusion loss type. 'eps_mse': standard MSE on epsilon. 'ula_kl': ULA-step KL (eta_t/(1-abar) weighted epsilon MSE). 'guided_ula_kl': Hessian-weighted score MSE via M_t = I + c_t*H_Q where c_t = lambda*(1-abar)/abar. 'e2e_guided_ula_kl': end-to-end guided drift MSE (Q gradient flows through Tweedie estimate). All guided losses treat Q params and MALA step size as fixed.")
    parser.add_argument("--policy_loss_reduction", type=str, default="mean", choices=["mean", "sum"], help="How to reduce the policy loss across action dimensions. 'mean': average over both batch and action dims (default). 'sum': sum over action dims, then mean over batch. 'sum' keeps loss scale proportional to act_dim and matches the VLB decomposition.")

    # Soft policy iteration
    parser.add_argument("--soft_pi_mode", action="store_true", default=False, help="Enable soft policy iteration: outer loop over policy improvement steps, each running iterations_per_pi_step iterations with frozen Q for guidance.")
    parser.add_argument("--iterations_per_pi_step", type=int, default=100000, help="Number of iterations (env steps) per policy improvement step in soft-PI mode.")
    parser.add_argument("--num_pi_steps", type=int, default=10, help="Number of policy improvement steps in soft-PI mode.")

    # Validation / hold-out configuration
    parser.add_argument("--use_validation", action="store_true", default=False,
                        help="If set, enable a held-out validation split for logging and (optionally) hypergradient updates.")
    parser.add_argument("--validation_size", type=int, default=5000,
                        help="(Legacy) Number of most recent transitions to reserve for validation when no dedicated validation buffer is used.")
    parser.add_argument("--validation_ratio", type=float, default=0.0,
                        help=(
                            "Fraction of experience reserved for a dedicated validation buffer. "
                            "When > 0 and use_validation is enabled, an extra vector env is created "
                            "for validation and a separate replay buffer of size buffer_size * validation_ratio "
                            "is used for validation batches, leaving the main buffer purely for training."
                        ))

    # Hypergradient configuration (dpmd_mb_pc only for now)
    parser.add_argument("--use_hypergrad", action="store_true", default=False,
                        help="If set (for dpmd_mb_pc), enable hypergradient-based tuning of learning rate scales using validation loss.")
    parser.add_argument("--hypergrad_lr", type=float, default=1e-3,
                        help="Meta learning rate for hypergradient updates of log learning-rate scales.")
    parser.add_argument("--hypergrad_period", type=int, default=100,
                        help="Apply a hypergradient update every hypergrad_period parameter updates when use_hypergrad is enabled.")
    parser.add_argument("--hypergrad_accum_steps", type=int, default=1,
                        help=(
                            "Number of stochastic validation evaluations to average per hypergradient update. "
                            "When > 1, each hypergrad step uses a single train batch but averages the validation "
                            "objective over hypergrad_accum_steps independent RNG keys to reduce hypergradient variance."
                        ))

    # PCMD sampler hyperparameters
    parser.add_argument("--pcmd_points_per_seed", type=int, default=20)
    parser.add_argument("--pcmd_refresh_L", type=int, default=3)
    parser.add_argument("--pcmd_action_steps_per_level", type=int, default=1)
    parser.add_argument("--pcmd_cs", type=float, default=0.08)
    parser.add_argument("--pcmd_cs_accept", type=float, default=0.1)
    parser.add_argument("--pcmd_accept_sprime", type=str, default="none")
    parser.add_argument("--pcmd_s_accept_target", type=float, default=0.5)
    parser.add_argument("--pcmd_s_accept_lr", type=float, default=0.05)
    parser.add_argument("--pcmd_level_offset", type=int, default=1)
    parser.add_argument("--pcmd_H_plan", type=int, default=1)

    parser.add_argument("--pcmd_use_ula_refresh", dest="pcmd_use_ula_refresh", action="store_true")
    parser.add_argument("--pcmd_no_ula_refresh", dest="pcmd_use_ula_refresh", action="store_false")
    parser.set_defaults(pcmd_use_ula_refresh=True)

    parser.add_argument("--pcmd_use_crn", dest="pcmd_use_crn", action="store_true")
    parser.add_argument("--pcmd_no_use_crn", dest="pcmd_use_crn", action="store_false")
    parser.set_defaults(pcmd_use_crn=True)

    parser.add_argument("--pcmd_bprop_refresh", action="store_true", default=False,
                        help="Enable backpropagation through refresh in the PCMD sampler.")

    parser.add_argument("--pcmd_adapt_s_accept", dest="pcmd_adapt_s_accept", action="store_true")
    parser.add_argument("--pcmd_no_adapt_s_accept", dest="pcmd_adapt_s_accept", action="store_false")
    parser.set_defaults(pcmd_adapt_s_accept=False)

    # DPMD_MB_PC planner hyperparameters
    parser.add_argument("--sprime_num_particles", type=int, default=16)
    parser.add_argument("--sprime_refresh_steps", type=int, default=3)
    parser.add_argument(
        "--sprime_refresh_type",
        type=str,
        default="recurrence",
        choices=["recurrence", "ula"],
    )
    parser.add_argument("--sprime_cs", type=float, default=0.08)
    parser.add_argument("--bprop_refresh_steps", type=int, default=0)
    parser.add_argument(
        "--pc_deterministic_dyn",
        action="store_true",
        default=False,
        help=(
            "If set, train the dpmd_mb_pc dynamics model as a simple deterministic next-state "
            "regressor using MSE on next_obs instead of diffusion p_loss, and use deterministic "
            "next-state predictions for model-based value TD and planner rewards."
        ),
    )
    parser.add_argument(
        "--pc_entropic_sprime_agg",
        action="store_true",
        default=False,
        help=(
            "For dpmd_mb_pc, use entropic risk aggregation over s' particles when computing the "
            "R+gamma*V signal for guidance, with tfg_eta as the entropic temperature."
        ),
    )

    parser.add_argument(
        "--pc_ucb_sprime_coeff",
        type=float,
        default=0.0,
        help=(
            "For dpmd_mb_pc, aggregate planner-sampled R+gamma*V (and planner-based "
            "inference V when value_td_mode='inference') using a UCB-style statistic "
            "mean + k * std with coefficient k = pc_ucb_sprime_coeff. A value of 0 "
            "reduces to plain mean aggregation."
        ),
    )

    parser.add_argument(
        "--pc_action_recur_steps",
        type=int,
        default=0,
        help=(
            "Number of TFG recurrence-style inner steps per diffusion level for dpmd_mb_pc "
            "action sampling. When > 0, use recurrence-based guidance instead of MALA."
        ),
    )

    parser.add_argument(
        "--pc_H_plan",
        type=int,
        default=1,
        help=(
            "Planning and training horizon for dpmd_mb_pc. When > 1 (requires --pc_deterministic_dyn), "
            "denoise H actions in parallel during planning and train the policy on H-step sequences "
            "from the replay buffer with horizon index embeddings. The guidance objective is the "
            "H-step unrolled sum of discounted rewards plus terminal value. H=1 recovers standard "
            "single-action behavior."
        ),
    )
    parser.add_argument(
        "--pc_joint_seq",
        action="store_true",
        default=False,
        help=(
            "When enabled with --pc_H_plan > 1, use a joint sequence denoiser that models "
            "correlations across the H-step action sequence. The policy network takes the "
            "full action sequence as input and outputs noise for all H actions jointly. "
            "Default (off) uses a factorized prior where each action is denoised separately "
            "conditioned on its horizon index."
        ),
    )
    parser.add_argument(
        "--open_loop",
        action="store_true",
        default=False,
        help=(
            "When enabled with --pc_H_plan > 1, execute the full H-action plan sequentially "
            "without replanning until the sequence is exhausted. Default (off) replans at "
            "every step (closed-loop MPC)."
        ),
    )

    parser.add_argument("--pc_use_crn", dest="pc_use_crn", action="store_true")
    parser.add_argument("--pc_no_use_crn", dest="pc_use_crn", action="store_false")
    parser.set_defaults(pc_use_crn=True)

    parser.add_argument("--pc_use_value", action="store_true", default=False)
    parser.add_argument(
        "--value_td_mode",
        type=str,
        default="replay",
        choices=["replay", "unguided", "inference"],
        help=(
            "Value TD operator for dpmd_mb_pc when pc_use_value is enabled: "
            "'replay' (v_next = V(next_obs)), 'unguided' (v_next = E_base[V(s')], "
            "using unguided DDIM sampling from the base diffusion policy without planner guidance), "
            "or 'inference' (v_next = E_planner[V(s')], using the full planner-based inference sampler)."
        ),
    )
    parser.add_argument(
        "--value_td_num_actions",
        type=int,
        default=1,
        help=(
            "Number of planner actions sampled per state when value_td_mode='inference' for "
            "dpmd_mb_pc value TD. Each action produces a population of sprime_num_particles "
            "next states; all states across actions are used in the TD target."
        ),
    )
 
    args = parser.parse_args()

    # Automatically enable energy parameterization if MALA steps are requested
    if args.mala_steps > 0:
        args.energy_param = True

    # For DPMD, disallow using both MALA and recurrence at the same time.
    if args.alg == 'dpmd' and args.mala_steps > 0 and args.dpmd_recurrence_steps > 0:
        raise ValueError("For dpmd, --mala_steps and --dpmd_recurrence_steps cannot both be > 0.")

    # --dist_shift_eta / --one_step_dist_shift_eta implies a KL budget (default 5.33 per dim)
    if args.dist_shift_eta and args.kl_budget is None and args.kl_budget_per_dim is None:
        args.kl_budget_per_dim = 5.33
    if args.one_step_dist_shift_eta and args.kl_budget is None and args.kl_budget_per_dim is None:
        args.kl_budget_per_dim = 5.33
    if args.two_step_dist_shift_eta and args.kl_budget is None and args.kl_budget_per_dim is None:
        args.kl_budget_per_dim = 5.33

    # Hypergradient-based tuning currently relies on a held-out validation
    # split. Require validation to be enabled when hypergrad is requested.
    if args.use_hypergrad and not args.use_validation:
        raise ValueError("--use_hypergrad requires --use_validation to be enabled, since hypergradients are computed from validation loss.")

    # Configure LR schedules for diffusion-policy algorithms (dpmd, dpmd_mb).
    # --lr_annealing is a deprecated shorthand for --lr_schedule_type linear.
    lr_schedule_type = args.lr_schedule_type
    if args.lr_annealing and lr_schedule_type == "constant":
        lr_schedule_type = "linear"
    # Enable schedule steps if EITHER policy or Q schedule is non-constant.
    any_schedule = (lr_schedule_type != "constant") or (args.q_lr_schedule_type != "constant")
    if any_schedule:
        dpmd_lr_schedule_steps = int(args.total_step)
        dpmd_lr_schedule_begin = 0
    else:
        dpmd_lr_schedule_steps = 0
        dpmd_lr_schedule_begin = 0

    if args.debug:
        from jax import config
        config.update("jax_disable_jit", True)

    def _derive_seeds(master: int):
        """Map a master int seed to the tuple (env_seed, env_action_seed,
        eval_env_seed, buffer_seed, init_network_seed, train_seed) that a
        standalone run with --seed=master would use."""
        rng, _ = seeding(int(master))
        return tuple(int(x) for x in rng.integers(0, 2**32 - 1, 6))

    master_seed = args.seed
    env_seed, env_action_seed, eval_env_seed, buffer_seed, init_network_seed, train_seed = _derive_seeds(master_seed)
    init_network_key = jax.random.key(init_network_seed)
    train_key = jax.random.key(train_seed)
    del init_network_seed, train_seed

    N_seeds = int(args.parallel_seeds)
    if N_seeds > 1:
        if args.alg != "dpmd":
            raise ValueError("--parallel_seeds > 1 currently only supports --alg dpmd.")
        if args.num_vec_envs <= 0:
            raise ValueError("--parallel_seeds > 1 requires --num_vec_envs > 0.")

    # Load the hp_pack once from whichever source is set; inline wins if both.
    # Empty dict if neither is set (equivalent to no pack). Used below for
    # per-entry master seeds AND the later _replace overrides on the vmap state.
    def _load_hp_pack(a):
        if a.hp_pack_inline is not None:
            return json.loads(a.hp_pack_inline)
        if a.hp_pack is not None:
            with open(a.hp_pack) as f:
                return json.load(f)
        return None
    _hp_loaded = _load_hp_pack(args)

    # Optional per-vmap-entry master seeds from hp_pack["seed"]. When present,
    # every seed site (buffer RNG, network init, env RNG, action-sample RNG) is
    # derived from its entry's master, so the pack behaves exactly like running
    # each --seed S_i standalone on its own GPU.
    per_entry_masters = None
    if N_seeds > 1 and _hp_loaded is not None:
        _hp_peek = _hp_loaded
        if "seed" in _hp_peek:
            per_entry_masters = [int(s) for s in _hp_peek["seed"]]
            if len(per_entry_masters) != N_seeds:
                raise ValueError(
                    f"--hp_pack 'seed' has length {len(per_entry_masters)}; "
                    f"expected {N_seeds} (= --parallel_seeds)"
                )
    if per_entry_masters is not None:
        _per_entry_derived = [_derive_seeds(m) for m in per_entry_masters]
        per_entry_env_seeds = [t[0] for t in _per_entry_derived]
        per_entry_action_seeds = [t[1] for t in _per_entry_derived]
        per_entry_eval_env_seeds = [t[2] for t in _per_entry_derived]
        per_entry_buffer_seeds = [t[3] for t in _per_entry_derived]
        per_entry_init_network_seeds = [t[4] for t in _per_entry_derived]
        per_entry_train_seeds = [t[5] for t in _per_entry_derived]
    else:
        per_entry_env_seeds = None
        per_entry_action_seeds = None
        per_entry_eval_env_seeds = None
        per_entry_buffer_seeds = None
        per_entry_init_network_seeds = None
        per_entry_train_seeds = None

    total_envs = args.num_vec_envs * max(1, N_seeds)

    if args.num_vec_envs > 0:
        env, obs_dim, act_dim = create_vector_env(
            args.env,
            total_envs,
            env_seed,
            env_action_seed,
            mode="futex",
            backend=args.backend,
            dummy_action_dim=args.dummy_action_dim,
            dummy_action_alpha=args.dummy_action_alpha,
            per_entry_env_seeds=per_entry_env_seeds,
            per_entry_action_seeds=per_entry_action_seeds,
        )
    else:
        env, obs_dim, act_dim = create_env(
            args.env,
            env_seed,
            env_action_seed,
            dummy_action_dim=args.dummy_action_dim,
            dummy_action_alpha=args.dummy_action_alpha,
            backend=args.backend,
        )
    eval_env = None

    # Resolve KL budget: --kl_budget sets the total directly;
    # --kl_budget_per_dim sets it as per_dim * act_dim.
    if args.kl_budget is not None and args.kl_budget_per_dim is not None:
        parser.error("--kl_budget and --kl_budget_per_dim are mutually exclusive")
    if args.kl_budget_per_dim is not None:
        args.kl_budget = args.kl_budget_per_dim * act_dim
    if args.kl_budget is not None:
        args.critic_normalization = "ema"
        args.tfg_eta = float((2.0 * args.kl_budget) ** 0.5)

    hidden_sizes = [args.hidden_dim] * args.hidden_num
    diffusion_hidden_sizes = [args.diffusion_hidden_dim] * args.hidden_num

    # Use SARSA-style buffer (with next_action) when dpmd_off_policy_td is enabled
    include_next_action = getattr(args, 'dpmd_off_policy_td', False)
    if N_seeds > 1 and per_entry_buffer_seeds is not None:
        # Per-entry master seeds override: buffer s uses the standalone
        # buffer_seed that would come from hp_pack["seed"][s].
        buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=args.buffer_size,
                                            seed=per_entry_buffer_seeds[0],
                                            include_next_action=include_next_action)
        buffers_list = [buffer] + [
            TreeBuffer.from_experience(
                obs_dim, act_dim, size=args.buffer_size,
                seed=per_entry_buffer_seeds[s + 1], include_next_action=include_next_action,
            )
            for s in range(N_seeds - 1)
        ]
    else:
        buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=args.buffer_size,
                                            seed=buffer_seed,
                                            include_next_action=include_next_action)
        if N_seeds > 1:
            buffers_list = [buffer] + [
                TreeBuffer.from_experience(
                    obs_dim, act_dim, size=args.buffer_size,
                    seed=buffer_seed + s + 1, include_next_action=include_next_action,
                )
                for s in range(N_seeds - 1)
            ]
        else:
            buffers_list = None

    # Optional dedicated validation environment and buffer. When enabled via
    # use_validation and validation_ratio>0, we create an extra vector env for
    # validation and a separate replay buffer whose capacity is scaled by
    # validation_ratio relative to the training buffer.
    val_env = None
    val_buffer = None
    if args.use_validation and args.validation_ratio > 0.0 and args.num_vec_envs > 0:
        # Keep num_vec_envs as the number of training envs and allocate an
        # additional set of validation envs according to the requested ratio.
        train_envs = args.num_vec_envs
        r = float(args.validation_ratio)
        # Solve approximately for val_envs such that
        #   r ≈ val_envs / (train_envs + val_envs)
        # and ensure at least one validation env.
        if r >= 1.0:
            val_envs = train_envs
        else:
            val_envs = max(1, int(round(train_envs * r / max(1e-8, 1.0 - r))))

        if per_entry_env_seeds is not None:
            _val_per_entry_env_seeds = [s + 1 for s in per_entry_env_seeds]
            _val_per_entry_action_seeds = [s + 1 for s in per_entry_action_seeds]
        else:
            _val_per_entry_env_seeds = None
            _val_per_entry_action_seeds = None
        val_env, _val_obs_dim, _val_act_dim = create_vector_env(
            args.env,
            val_envs,
            env_seed + 1,
            env_action_seed + 1,
            mode="futex",
            backend=args.backend,
            per_entry_env_seeds=_val_per_entry_env_seeds,
            per_entry_action_seeds=_val_per_entry_action_seeds,
        )

        val_buffer_size = max(1, int(args.buffer_size * r))
        val_buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=val_buffer_size, seed=buffer_seed + 1)

    gelu = partial(jax.nn.gelu, approximate=False)
    
    print(f"Algorithm: {args.alg}")

    if args.alg == 'sdac':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_diffv2_net(
            init_network_key,
            obs_dim,
            act_dim,
            hidden_sizes,
            diffusion_hidden_sizes,
            mish,
            num_timesteps=args.diffusion_steps,
            num_particles=args.num_particles,
            noise_scale=args.noise_scale,
            beta_schedule_scale=args.beta_schedule_scale,
            beta_schedule_type=args.beta_schedule_type,
            energy_param=args.energy_param,
            mala_steps=args.mala_steps,
            gaussian_prior_baseline=args.gaussian_prior_baseline,
            snr_max=args.snr_max,
        )
        algorithm = SDAC(agent, params, lr=args.lr, alpha_lr=args.alpha_lr, delay_alpha_update=args.delay_alpha_update, lr_schedule_end=args.lr_schedule_end)
    
    elif args.alg == 'dpmd':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))

        def _make_diffv2(net_key):
            return create_diffv2_net(
                net_key,
                obs_dim,
                act_dim,
                hidden_sizes,
                diffusion_hidden_sizes,
                mish,
                num_timesteps=args.diffusion_steps,
                num_particles=args.num_particles,
                noise_scale=args.noise_scale,
                beta_schedule_scale=args.beta_schedule_scale,
                beta_schedule_type=args.beta_schedule_type,
                energy_param=args.energy_param,
                mala_steps=args.mala_steps,
                single_q_network=args.single_q_network,
                num_q_networks=args.num_q_networks,
                x_recon_clip_radius=args.x0_hat_clip_radius if args.latent_action_space else 1.0,
                gaussian_prior_baseline=args.gaussian_prior_baseline,
                distributional_critic=args.distributional_critic,
                snr_max=args.snr_max,
                zero_init_q=args.zero_init_q,
                dist_shift_eta=args.dist_shift_eta,
            )

        if N_seeds > 1:
            if per_entry_init_network_seeds is not None:
                seed_init_keys = jnp.stack([
                    jax.random.key(s) for s in per_entry_init_network_seeds
                ])
            else:
                seed_init_keys = jax.random.split(init_network_key, N_seeds)
            _pairs = [_make_diffv2(k) for k in seed_init_keys]
            agent = _pairs[0][0]
            dpmd_params_list = [p for (_a, p) in _pairs]
            params = dpmd_params_list[0]
        else:
            agent, params = _make_diffv2(init_network_key)
            dpmd_params_list = None

        algorithm = DPMD(
            agent,
            params,
            gamma=args.gamma,
            lr=args.lr,
            lr_policy=args.lr_policy,
            lr_q=args.lr_q,
            alpha_lr=args.alpha_lr,
            delay_alpha_update=args.delay_alpha_update,
            tau=args.tau,
            delay_update=args.delay_update,
            lr_schedule_end=args.lr_schedule_end,
            reward_scale=args.reward_scale,
            td_actions=args.td_actions,
            use_reweighting=not args.dpmd_constant_weight,
            use_reward_critic=args.dpmd_use_reward_critic,
            pure_bc_training=args.dpmd_pure_bc_training,
            off_policy_td=args.dpmd_off_policy_td,
            no_entropy_tuning=args.dpmd_no_entropy_tuning,
            q_critic_agg=args.q_critic_agg,
            q_critic_agg_idx=args.q_critic_agg_idx,
            fix_q_norm_bug=args.fix_q_norm_bug,
            tfg_eta=args.tfg_eta,
            tfg_eta_schedule=args.tfg_eta_schedule,
            tfg_recur_steps=args.dpmd_recurrence_steps,
            particle_selection_lambda=args.particle_selection_lambda,
            x0_hat_clip_radius=args.x0_hat_clip_radius,
            supervised_steps=args.supervised_steps,
            single_q_network=args.single_q_network,
            lr_schedule_steps=dpmd_lr_schedule_steps,
            lr_schedule_begin=dpmd_lr_schedule_begin,
            mala_per_level_eta=args.mala_per_level_eta,
            mala_adapt_rate=args.mala_adapt_rate,
            mala_init_eta_scale=args.mala_init_eta_scale,
            mala_recurrence_cap=args.mala_recurrence_cap,
            mala_guided_predictor=args.mala_guided_predictor,
            mala_predictor_first=args.mala_predictor_first,
            ddim_predictor=args.ddim_predictor,
            latent_action_space=args.latent_action_space,
            q_td_huber_width=args.q_td_huber_width,
            decorrelated_q_batches=args.decorrelated_q_batches,
            q_bootstrap_agg=args.q_bootstrap_agg,
            entropic_risk_beta=args.entropic_risk_beta,
            langevin_q_noise=args.langevin_q_noise,
            td_use_target_policy=args.td_use_target_policy,
            batch_independent_guidance=args.batch_independent_guidance,
            guidance_strength_multiplier=args.guidance_strength_multiplier,
            critic_grad_modifier=args.critic_grad_modifier,
            natural_gradient_critic=args.natural_gradient_critic,
            dsac_expected_value_sub=args.dsac_expected_value_sub,
            dsac_adaptive_clip_xi=args.dsac_adaptive_clip_xi,
            dsac_omega_scaling=args.dsac_omega_scaling,
            dsac_omega_tau=args.dsac_omega_tau,
            energy_multiplier=args.energy_multiplier,
            critic_normalization=args.critic_normalization,
            policy_loss_type=args.policy_loss_type,
            policy_loss_reduction=args.policy_loss_reduction,
            soft_pi_mode=getattr(args, 'soft_pi_mode', False),
            optimizer_type=args.optimizer,
            weight_decay=args.weight_decay,
            lr_policy_schedule_type=lr_schedule_type,
            lr_q_schedule_type=args.q_lr_schedule_type,
            lr_q_schedule_end=args.q_lr_schedule_end,
            kl_budget=args.kl_budget,
            dist_shift_eta=args.dist_shift_eta,
            one_step_dist_shift_eta=args.one_step_dist_shift_eta,
            two_step_dist_shift_eta=args.two_step_dist_shift_eta,
            direct_eta_coeff_ema=args.direct_eta_coeff_ema,
            entropic_critic_param=args.entropic_critic_param,
            advantage_ema_tau=args.advantage_ema_tau,
            shape_ema_tau=args.shape_ema_tau,
            initial_advantage_second_moment_ema=args.initial_advantage_second_moment_ema,
            initial_dist_shift_shape_ema=args.initial_dist_shift_shape_ema,
            td_value_training=args.td_value_training,
        )

        if N_seeds > 1:
            algorithm.state = algorithm.make_vmapped_state(dpmd_params_list)
            if _hp_loaded is not None:
                _hp = _hp_loaded
                # Pack keys are argparse attribute names (what users see on the
                # CLI). Translate the four keys whose Diffv2TrainState field
                # names differ before calling state._replace.
                _CLI_TO_FIELD = {
                    "tau": "polyak_tau",
                    "advantage_ema_tau": "adv_ema_tau",
                    "guidance_strength_multiplier": "guidance_mult",
                    "kl_budget": "kl_budget_val",
                    "initial_advantage_second_moment_ema": "advantage_second_moment_ema",
                    "initial_dist_shift_shape_ema": "dist_shift_shape_ema",
                }
                _allowed = {"lr_q", "lr_policy", "gamma", "tau", "advantage_ema_tau",
                            "guidance_strength_multiplier", "shape_ema_tau", "kl_budget",
                            "initial_advantage_second_moment_ema", "initial_dist_shift_shape_ema",
                            "reward_scale", "x0_hat_clip_radius", "mala_adapt_rate",
                            "q_td_huber_width", "shape3_ema_tau", "q_critic_agg_idx",
                            "seed"}
                _overrides = {}
                for k, v in _hp.items():
                    if k not in _allowed:
                        raise ValueError(f"--hp_pack key '{k}' is not a per-seed vmappable hp. Allowed: {sorted(_allowed)}")
                    if k == "seed":
                        # Already consumed above to drive per-entry buffer and
                        # network-init seeds; not a TrainState field.
                        continue
                    arr = jnp.asarray(v, dtype=jnp.float32)
                    if arr.shape != (N_seeds,):
                        raise ValueError(f"--hp_pack '{k}' has shape {arr.shape}; expected ({N_seeds},)")
                    _overrides[_CLI_TO_FIELD.get(k, k)] = arr
                if (
                    "advantage_second_moment_ema" in _overrides
                    or "dist_shift_shape_ema" in _overrides
                ):
                    adv2 = _overrides.get(
                        "advantage_second_moment_ema",
                        jnp.asarray(algorithm.state.advantage_second_moment_ema, dtype=jnp.float32),
                    )
                    shape = _overrides.get(
                        "dist_shift_shape_ema",
                        jnp.asarray(algorithm.state.dist_shift_shape_ema, dtype=jnp.float32),
                    )
                    _overrides["dist_shift_coeff_ema"] = shape * jnp.power(
                        jnp.maximum(adv2, jnp.float32(0.0)),
                        jnp.float32(1.5),
                    )
                if _overrides:
                    algorithm.state = algorithm.state._replace(**_overrides)
                    if "kl_budget_val" in _overrides:
                        kl_budget_v = jnp.asarray(algorithm.state.kl_budget_val, dtype=jnp.float32)
                        algorithm.state = algorithm.state._replace(
                            tfg_eta=jnp.sqrt(jnp.maximum(jnp.float32(0.0), jnp.float32(2.0) * kl_budget_v))
                        )
                    print(f"[hp_pack] applied per-seed overrides: {list(_overrides.keys())}")
                if per_entry_masters is not None:
                    print(f"[hp_pack] applied per-entry master seeds "
                          f"(buffers + init networks + train keys): {per_entry_masters}")

    elif args.alg == 'dpmd_bc':
        # DPMD variant with diffusion policy trained by plain behavior cloning
        # (unweighted diffusion loss) and Q tilt only at inference time.
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_diffv2_net(
            init_network_key,
            obs_dim,
            act_dim,
            hidden_sizes,
            diffusion_hidden_sizes,
            mish,
            num_timesteps=args.diffusion_steps,
            num_particles=args.num_particles,
            noise_scale=args.noise_scale,
            beta_schedule_scale=args.beta_schedule_scale,
            beta_schedule_type=args.beta_schedule_type,
            energy_param=args.energy_param,
            mala_steps=args.mala_steps,
            gaussian_prior_baseline=args.gaussian_prior_baseline,
            snr_max=args.snr_max,
        )
        algorithm = DPMDBC(
            agent,
            params,
            lr=args.lr,
            alpha_lr=args.alpha_lr,
            delay_alpha_update=args.delay_alpha_update,
            lr_schedule_end=args.lr_schedule_end,
            reward_scale=args.reward_scale,
            train_q_on_noisy_actions=args.dpmd_bc_noisy_q_guided,
            guided_sampling=args.dpmd_bc_noisy_q_guided,
            tfg_recurrence=args.dpmd_bc_tfg_recurrence,
            tfg_recur_steps=args.dpmd_bc_recurrence_steps,
        )

    elif args.alg == 'dpmd_mb':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        from relax.algorithm.dpmd_mb import DPMDMB
        agent, params = create_model_based_net(
            init_network_key,
            obs_dim,
            act_dim,
            hidden_sizes,
            diffusion_hidden_sizes,
            activation=mish,
            num_timesteps=args.diffusion_steps,
            num_particles=args.num_particles,
            noise_scale=args.noise_scale,
            beta_schedule_scale=args.beta_schedule_scale,
            beta_schedule_type=args.beta_schedule_type,
            energy_param=args.energy_param,
            snr_max=args.snr_max,
        )
        algorithm = DPMDMB(
            agent,
            params,
            lr=args.lr,
            alpha_lr=args.alpha_lr,
            delay_alpha_update=args.delay_alpha_update,
            lr_schedule_end=args.lr_schedule_end,
            reward_scale=args.reward_scale,
            num_mc_samples=args.model_q_mc_samples,
            lr_policy=args.lr_policy,
            lr_dyn=args.lr_dyn,
            lr_reward=args.lr_reward,
            lr_value=args.lr_value,
            lr_schedule_steps=dpmd_lr_schedule_steps,
            lr_schedule_begin=dpmd_lr_schedule_begin,
        )

    elif args.alg == 'dpmd_mb_pc':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        from relax.algorithm.dpmd_mb_pc import DPMDMBPC
        agent, params = create_model_based_net(
            init_network_key,
            obs_dim,
            act_dim,
            hidden_sizes,
            diffusion_hidden_sizes,
            activation=mish,
            num_timesteps=args.diffusion_steps,
            num_particles=args.num_particles,
            noise_scale=args.noise_scale,
            beta_schedule_scale=args.beta_schedule_scale,
            beta_schedule_type=args.beta_schedule_type,
            energy_param=args.energy_param,
            H_train=args.pc_H_plan,
            joint_seq=args.pc_joint_seq,
        )
        algorithm = DPMDMBPC(
            agent,
            params,
            lr=args.lr,
            reward_scale=args.reward_scale,
            use_value=args.pc_use_value,
            sprime_num_particles=args.sprime_num_particles,
            sprime_refresh_steps=args.sprime_refresh_steps,
            sprime_refresh_type=args.sprime_refresh_type,
            sprime_cs=args.sprime_cs,
            bprop_refresh_steps=args.bprop_refresh_steps,
            action_steps_per_level=args.mala_steps,
            action_recur_steps=args.pc_action_recur_steps,
            H_plan=args.pc_H_plan,
            joint_seq=args.pc_joint_seq,
            open_loop=args.open_loop,
            use_crn=args.pc_use_crn,
            tfg_eta=args.tfg_eta,
            x0_hat_clip_radius=args.x0_hat_clip_radius,
            supervised_steps=args.supervised_steps,
            lr_policy=args.lr_policy,
            lr_dyn=args.lr_dyn,
            lr_reward=args.lr_reward,
            lr_value=args.lr_value,
            value_mc_samples=args.model_q_mc_samples,
            value_td_mode=args.value_td_mode,
            value_td_num_actions=args.value_td_num_actions,
            deterministic_dyn=args.pc_deterministic_dyn,
            entropic_sprime_agg=args.pc_entropic_sprime_agg,
            ucb_sprime_coeff=args.pc_ucb_sprime_coeff,
            use_hypergrad=args.use_hypergrad,
            hypergrad_lr=args.hypergrad_lr,
            hypergrad_period=args.hypergrad_period,
            hypergrad_accum_steps=args.hypergrad_accum_steps,
        )

    elif args.alg == 'idem':
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_diffv2_net(
            init_network_key,
            obs_dim,
            act_dim,
            hidden_sizes,
            diffusion_hidden_sizes,
            mish,
            num_timesteps=args.diffusion_steps,
            num_particles=args.num_particles,
            noise_scale=args.noise_scale,
            beta_schedule_scale=args.beta_schedule_scale,
            beta_schedule_type=args.beta_schedule_type,
            energy_param=args.energy_param,
            mala_steps=args.mala_steps,
            gaussian_prior_baseline=args.gaussian_prior_baseline,
            snr_max=args.snr_max,
        )
        algorithm = IDEM(agent, params, lr=args.lr, alpha_lr=args.alpha_lr, delay_alpha_update=args.delay_alpha_update, lr_schedule_end=args.lr_schedule_end)
    elif args.alg == "qsm":
        agent, params = create_qsm_net(init_network_key, obs_dim, act_dim, hidden_sizes, num_timesteps=args.diffusion_steps, num_particles=args.num_particles)
        algorithm = QSM(agent, params, lr=args.lr, lr_schedule_end=args.lr_schedule_end)
    elif args.alg == "sac":
        agent, params = create_sac_net(init_network_key, obs_dim, act_dim, hidden_sizes, gelu)
        algorithm = SAC(agent, params, lr=args.lr)
    elif args.alg == "dsact":
        agent, params = create_dsact_net(init_network_key, obs_dim, act_dim, hidden_sizes, gelu)
        algorithm = DSACT(agent, params, lr=args.lr)
    elif args.alg == "dacer":
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_dacer_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish, 
                                         num_timesteps=args.diffusion_steps)
        algorithm = DACER(agent, params, lr=args.lr, lr_schedule_end=args.lr_schedule_end)
    elif args.alg == "dacer_doubleq":
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_dacer_doubleq_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish, num_timesteps=args.diffusion_steps)
        algorithm = DACERDoubleQ(agent, params, lr=args.lr)
    elif args.alg == "dipo":
        diffusion_buffer = TreeBuffer.from_example(
            ObsActionPair.create_example(obs_dim, act_dim),
            args.total_step,
            int(master_rng.integers(0, 2**32 - 1)),
            remove_batch_dim=False
        )
        TreeBuffer.connect(buffer, diffusion_buffer, lambda exp: ObsActionPair(exp.obs, exp.action))

        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))

        agent, params = create_dipo_net(init_network_key, obs_dim, act_dim, hidden_sizes, num_timesteps=args.diffusion_steps)
        algorithm = DIPO(agent, params, diffusion_buffer, lr=args.lr, action_gradient_steps=30, policy_target_delay=2, action_grad_norm=0.16)
    elif args.alg == "qvpo":
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))
        agent, params = create_qvpo_net(init_network_key, obs_dim, act_dim, hidden_sizes, diffusion_hidden_sizes, mish,
                                          num_timesteps=args.diffusion_steps,
                                          num_particles=args.num_particles,
                                          noise_scale=args.noise_scale)
        algorithm = QVPO(agent, params, lr=args.lr, alpha_lr=args.alpha_lr, delay_alpha_update=args.delay_alpha_update)
    elif args.alg == "pcmd":
        def mish(x: jax.Array):
            return x * jnp.tanh(jax.nn.softplus(x))

        # Use the same hidden sizes and activation style as other diffusion-based methods
        pc_net, pc_params = create_pcmd_net(
            init_network_key,
            obs_dim,
            act_dim,
            hidden_sizes,
            activation=mish,
        )

        # Map existing diffusion_steps / beta_schedule_scale to PC-MD level config
        levels_cfg = PcLevelsConfig(
            K=int(args.diffusion_steps),
            alpha_bar_min=0.05,
            alpha_bar_max=1.0,
            beta_schedule="linear",
            lambda_scale=float(args.beta_schedule_scale),
        )

        algorithm = PCMD(
            pc_net,
            pc_params,
            gamma=args.gamma,
            lr_policy=args.lr_policy if args.lr_policy is not None else args.lr,
            lr_dyn=args.lr_dyn if args.lr_dyn is not None else args.lr,
            lr_reward=args.lr_reward if args.lr_reward is not None else args.lr,
            lr_value=args.lr_value if args.lr_value is not None else args.lr,
            ema_tau=0.005,
            H_train=1,
            H_plan=args.pcmd_H_plan,
            num_timesteps=args.diffusion_steps,
            beta_schedule_scale=args.beta_schedule_scale,
            beta_schedule_type=args.beta_schedule_type,
            levels_cfg=levels_cfg,
            points_per_seed=args.pcmd_points_per_seed,
            refresh_L=args.pcmd_refresh_L,
            action_steps_per_level=args.pcmd_action_steps_per_level,
            use_ula_refresh=args.pcmd_use_ula_refresh,
            cs=args.pcmd_cs,
            cs_accept=args.pcmd_cs_accept,
            bprop_refresh=args.pcmd_bprop_refresh,
            accept_sprime=args.pcmd_accept_sprime,
            adapt_s_accept=args.pcmd_adapt_s_accept,
            s_accept_target=args.pcmd_s_accept_target,
            s_accept_lr=args.pcmd_s_accept_lr,
            use_crn=args.pcmd_use_crn,
            level_offset=args.pcmd_level_offset,
        )
    else:
        raise ValueError(f"Invalid algorithm {args.alg}!")

    if args.cluster:
        PROJECT_ROOT = Path('/n/netscratch/nali_lab_seas/Lab/haitongma/sdac_logs')
    
    exp_dir = PROJECT_ROOT / "logs" / args.env / (args.alg + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + f'_s{args.seed}_{args.suffix}')

    # PCMD, DPMD, DPMD_BC, DPMD_MB, and DPMD_MB_PC do not expose a standalone value
    # interface used by save_q_structure; avoid saving value structure for them to
    # prevent setup crashes.
    save_value = args.alg not in ("pcmd", "dpmd", "dpmd_bc", "dpmd_mb", "dpmd_mb_pc")

    # Collect all CLI arguments into a dict for logging and configuration.
    # Allow the algorithm to override or augment these with its own effective
    # hyperparameters (e.g., internally clamped / derived values).
    args_dict = dict(vars(args))
    if not args.use_validation:
        # When validation is disabled, treat the effective validation_size as 0
        # in the logged configuration, so wandb reflects the actual behavior.
        args_dict["validation_size"] = 0
    if hasattr(algorithm, "get_effective_hparams"):
        args_dict.update(algorithm.get_effective_hparams())

    if N_seeds > 1 and per_entry_train_seeds is not None:
        train_key_vmap = jnp.stack([jax.random.key(s) for s in per_entry_train_seeds])
    else:
        train_key_vmap = train_key

    if N_seeds > 1:
        from relax.trainer.vmap_off_policy import VmapOffPolicyTrainer
        trainer = VmapOffPolicyTrainer(
            env=env,
            algorithm=algorithm,
            buffers=buffers_list,
            log_path=exp_dir,
            parallel_seeds=N_seeds,
            per_seed_envs=args.num_vec_envs,
            batch_size=args.batch_size,
            start_step=args.start_step,
            total_step=args.total_step,
            update_per_iteration=args.update_per_iteration,
            update_log_n_env_steps=5 if args.debug else 5000,
            hparams=args_dict,
            hp_pack_path=args.hp_pack,
            hp_pack_dict=_hp_loaded,
            sweep_id=args.sweep_id,
            config_tag_keys=args.config_tag_keys,
        )
        trainer.setup(Experience.create_example(obs_dim, act_dim, trainer.batch_size, include_next_action=include_next_action))
        trainer.run(train_key_vmap)
    else:
        trainer = OffPolicyTrainer(
            env=env,
            algorithm=algorithm,
            buffer=buffer,
            log_path=exp_dir,
            batch_size=args.batch_size,
            val_batch_size=args.val_batch_size,
            start_step=args.start_step,
            total_step=args.total_step,
            sample_per_iteration=1,
            update_per_iteration=args.update_per_iteration,
            evaluate_env=eval_env,
            save_policy_every=int(args.total_step / 20),
            save_value=save_value,
            update_log_n_env_steps=5 if args.debug else 5000,
            debug=args.debug,
            timing_log_every=args.timing_log_every,
            warmup_with="random",
            hparams=args_dict,
            use_validation=args.use_validation,
            validation_size=args.validation_size,
            use_hypergrad=args.use_hypergrad,
            hypergrad_period=args.hypergrad_period,
            val_env=val_env,
            val_buffer=val_buffer,
            validation_ratio=args.validation_ratio,
            track_next_action=include_next_action,  # Enable SARSA-style buffer for off-policy TD
            latent_action_space=args.latent_action_space,
            latent_action_eps=args.latent_action_eps,
            tfg_patience=args.tfg_patience,
            tfg_reduction_factor=args.tfg_reduction_factor,
            tfg_eta_start=args.tfg_eta,
            tfg_eta_end=args.tfg_eta_end,
            log_md_kl_every=args.log_md_kl_every,
            soft_pi_mode=getattr(args, 'soft_pi_mode', False),
            iterations_per_pi_step=getattr(args, 'iterations_per_pi_step', 100000),
            num_pi_steps=getattr(args, 'num_pi_steps', 10),
            equal_episode_weighting=args.equal_episode_weighting,
        )

        trainer.setup(Experience.create_example(obs_dim, act_dim, trainer.batch_size, include_next_action=include_next_action))
        trainer.run(train_key)
