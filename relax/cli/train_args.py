"""Argparse / post-parse validation for ``scripts/train_mujoco.py``.

Kept in its own module so the entry-point reads top-to-bottom and the CLI
surface is the only thing one needs to scan to understand what flags the
three protected launch commands set.
"""
import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # ----- env / run control -------------------------------------------------
    parser.add_argument("--alg", type=str, default="dpmd", choices=["dpmd"])
    parser.add_argument("--env", type=str, default="HalfCheetah-v3")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--num_vec_envs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--start_step", type=int, default=int(3e4)) # other envs 3e4
    parser.add_argument("--total_step", type=int, default=int(1e6))
    parser.add_argument("--cluster", default=False, action="store_true")
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--timing_log_every", type=int, default=0)

    # ----- vmap / sweep plumbing --------------------------------------------
    parser.add_argument("--parallel_seeds", type=int, default=1, help="If > 1, train N independent DPMD seeds in parallel on a single device via jax.vmap. Env layout uses a single VectorEnv of size parallel_seeds * num_vec_envs. Current packed support covers the KL-budget/on-policy-EMA path and fixed-tfg_eta mode.")
    parser.add_argument("--hp_pack_inline", type=str, default=None, help="Inline JSON with per-seed hyperparameter overrides. Each key is an argparse attribute name of this script (e.g. 'tau', 'tfg_eta', 'advantage_ema_tau', 'guidance_strength_multiplier', 'kl_budget', 'shape_ema_tau', 'seed') mapped to a list of length parallel_seeds. Applied after vmap state construction; internally translated to Diffv2TrainState field names via _CLI_TO_FIELD.")
    parser.add_argument("--sweep_id", type=int, default=None, help="Launcher-assigned integer identifying this sweep. When set, every wandb run from this invocation is placed in wandb group 'sweep_<sweep_id>', and each per-vmap-slot run's config includes a 'config_tag' field built from sweep_id + the per-slot hyperparameters (excluding seed/env) so a single tag value filters wandb to all runs across envs/seeds that share this hp configuration.")
    parser.add_argument("--config_tag_keys", type=str, default=None, help="Comma-separated list of argparse attribute names whose values should be included in the per-slot config_tag. Typically set automatically by scripts/launch.py to the union of all --ablate hard+easy flags (minus env and seed). Values come from the hp_pack (per-slot) when the key is a pack key, else from this script's CLI args (shared across all vmap slots within the job).")

    # ----- networks ---------------------------------------------------------
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)
    parser.add_argument("--num_q_networks", type=int, default=2, help="Number of Q critic networks to train (default 2, i.e. twin Q).")
    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size for training updates.")

    # ----- diffusion schedule ----------------------------------------------
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--beta_schedule_scale", type=float, default=0.8)
    parser.add_argument("--beta_schedule_type", type=str, default='linear', help="Noise schedule type. 'linear': linear beta schedule. 'cosine': cosine schedule (Nichol & Dhariwal). 'constant_kl': constant mutual-information-loss per step, spacing noise levels uniformly in log(1+SNR).")
    parser.add_argument("--snr_max", type=float, default=124.0, help="Maximum SNR (at cleanest noise level). Controls alpha_bar_0 = snr_max/(1+snr_max). Default 124.0 matches the cosine schedule with s=0.008 offset at T=20. All schedule types use this to set the same clean endpoint, so you can switch between cosine/constant_kl/linear while keeping the noise range comparable.")

    # ----- optimization -----------------------------------------------------
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_policy", type=float, default=None)
    parser.add_argument("--lr_q", type=float, default=None)
    parser.add_argument("--update_per_iteration", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for the Q critic. Default 0.99.")
    parser.add_argument("--tau", type=float, default=0.005, help="Polyak averaging coefficient for target network updates. Default 0.005.")
    parser.add_argument("--delay_update", type=int, default=2, help="Update policy and target networks every delay_update steps. Default 2.")
    parser.add_argument("--reward_scale", type=float, default=0.2, help="Scale factor applied to rewards before Q/value learning. Default 0.2 matches original DPMD. Set to 1.0 for clarity when using inference-time guidance (adjust tfg_eta accordingly).")

    # ----- Q learning -------------------------------------------------------
    parser.add_argument("--q_critic_agg", type=str, default="min", choices=["min", "mean"], help="Aggregation for the Q signal used in tilting and reweighting. The TD-bootstrap path is hardcoded to 'min' (clipped double-Q).")
    parser.add_argument("--q_td_huber_width", type=float, default=float("inf"), help="Huber width (delta) for critic TD error in DPMD. Default inf recovers the current MSE TD loss. Effective width is scaled by reward_scale internally.")

    # ----- guidance + KL budget --------------------------------------------
    parser.add_argument("--tfg_eta", type=float, default=0.0, help="Guidance strength lambda for dpmd training-free Q-guidance. If 0, no Q-guidance is applied.")
    parser.add_argument("--kl_budget", type=float, default=None, help="Total KL divergence budget δ for guidance. Per-dimension budget is δ / act_dim. Sets η = sqrt(2δ), enables V network and on-policy advantage EMA. Default None (disabled).")
    parser.add_argument("--kl_budget_per_dim", type=float, default=None, help="Per-dimension KL divergence budget δ_d for guidance. Total budget δ = δ_d * act_dim. Sets η = sqrt(2δ), enables V network and on-policy advantage EMA. Default None (disabled).")
    parser.add_argument("--one_step_dist_shift_eta", action="store_true", default=False, help="Adaptive η from second-order expansion using one-step Monte Carlo covariance estimate. No D_ψ head; estimates c from consecutive (A_t, A_{t+1}) pairs. Requires --kl_budget or --kl_budget_per_dim (defaults to --kl_budget_per_dim=5.33 if neither set).")
    parser.add_argument("--advantage_ema_tau", type=float, default=0.0005, help="Per-step EMA rate for advantage second/third moments.")
    parser.add_argument("--shape_ema_tau", type=float, default=0.0001, help="Per-step EMA rate for dimensionless shape s2.")
    parser.add_argument("--initial_advantage_second_moment_ema", type=float, default=1.0, help="Initial value for the advantage second moment EMA E[A^2].")
    parser.add_argument("--initial_dist_shift_shape_ema", type=float, default=-1.0, help="Initial value for the dimensionless distribution-shift shape EMA s2 = (2γc + κ₃) / v^(3/2).")
    parser.add_argument("--x0_hat_clip_radius", type=float, default=float("inf"), help="Clipping radius r for Tweedie clean-action estimates x0_hat used inside guidance/Q evaluation. x0_hat is clipped to [-r, r] before being passed into Q / model-based objectives. Default inf (no clip); in non-latent mode the network-side denoising clip is separately hardcoded to 1.0 to match normalized action bounds.")
    parser.add_argument("--batch_independent_guidance", action="store_true", default=False, help="If set, use jnp.sum instead of jnp.mean inside the guided predictor's q_mean_from_x, so the per-sample Q gradient is independent of batch size (fixes the 1/B attenuation).")
    parser.add_argument("--guidance_strength_multiplier", type=float, default=1.0, help="Constant multiplier applied to the guided-predictor Q scalar before jax.grad. Composes with --batch_independent_guidance.")
    parser.add_argument("--energy_multiplier", type=float, default=1.0, help="Multiplier for base energy/score during sampling. Values < 1 temper (flatten) the base distribution, increasing entropy. Guidance signal is NOT scaled. Default 1.0 (no tempering).")

    # ----- MALA -------------------------------------------------------------
    parser.add_argument("--mala_steps", type=int, default=0, help="Number of MALA correction steps per diffusion step.")
    parser.add_argument("--mala_adapt_rate", type=float, default=0.05, help="Robbins-Monro adaptation rate for MALA log_eta_scale updates.")
    parser.add_argument("--mala_guided_predictor", action="store_true", default=False, help="If set, apply Q-guidance (TFG-style eps guidance) in the DDPM predictor step after each MALA correction step.")
    parser.add_argument("--mala_no_predictor", action="store_true", default=False, help="If set, remove predictor transitions entirely during MALA sampling so each lower-noise level initializes directly from the previous level's post-MALA state.")

    # ----- preserved-but-no-op (validate_args enforces these settings) ------
    # These flags appear in the protected launch commands but no longer
    # select an alternative code path; ``validate_args`` rejects any other
    # value so each is effectively a constraint, not a knob.
    parser.add_argument("--num_particles", type=int, default=1, help="Backward-compatibility flag. simplify_walkthrough supports only single-particle behavior, so this must remain 1.")
    parser.add_argument("--dpmd_constant_weight", action="store_true", default=False, help="If set for dpmd, disable Q-based reweighting in the diffusion score-matching loss and use constant weights.")
    parser.add_argument("--dpmd_no_entropy_tuning", action="store_true", default=False, help="If set for dpmd, disable action noise and alpha/entropy tuning.")
    parser.add_argument("--mala_per_level_eta", action="store_true", default=False, help="If set, learn a separate MALA eta-scale for each diffusion noise level. Default behavior (flag off) uses a single shared eta-scale across all noise levels.")
    parser.add_argument("--ddim_predictor", action="store_true", default=False, help="If set, use deterministic DDIM-style predictor (no noise) instead of stochastic DDPM. Recommended for MALA sampling since the stochastic noise is redundant with MALA corrections.")
    return parser


def validate_args(args, parser: argparse.ArgumentParser) -> None:
    """Post-parse validation. Calls ``parser.error`` on bad combinations.

    simplify_walkthrough only retains the branches actually exercised by the
    three protected launch commands; the flags below are kept in argparse so
    those commands still parse, but their non-default settings are no-ops.
    """
    if (args.num_particles != 1
            or not args.dpmd_no_entropy_tuning
            or not args.dpmd_constant_weight
            or not args.mala_per_level_eta
            or not args.ddim_predictor):
        parser.error(
            "simplify_walkthrough requires: --num_particles 1, "
            "--dpmd_no_entropy_tuning, --dpmd_constant_weight, "
            "--mala_per_level_eta, --ddim_predictor "
            "(the legacy alternatives have been removed)."
        )

    if args.kl_budget is not None and args.kl_budget_per_dim is not None:
        parser.error("--kl_budget and --kl_budget_per_dim are mutually exclusive")

    # --one_step_dist_shift_eta implies a KL budget (default 5.33 per dim)
    if args.one_step_dist_shift_eta and args.kl_budget is None and args.kl_budget_per_dim is None:
        args.kl_budget_per_dim = 5.33

    if args.parallel_seeds <= 0:
        parser.error("--parallel_seeds must be >= 1.")
    if args.num_vec_envs <= 0:
        parser.error("--num_vec_envs must be > 0 in simplify_walkthrough (vectorized/vmapped path only).")
    if args.mala_steps <= 0:
        parser.error(
            "simplify_walkthrough requires --mala_steps > 0; the non-MALA "
            "sampling branches have been removed."
        )
