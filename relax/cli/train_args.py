"""Argparse / post-parse validation for ``scripts/train_mujoco.py``.

Kept in its own module so the entry-point reads top-to-bottom and the CLI
surface is the only thing one needs to scan to understand what flags the
three protected launch commands set.
"""
import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # ----- env / run control -------------------------------------------------
    parser.add_argument("--alg", type=str, default="mgmd", choices=["mgmd"])
    parser.add_argument("--env", type=str, default="HalfCheetah-v3")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--num_vec_envs", type=int, default=5)
    parser.add_argument("--flatten_UTD", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--start_step", type=int, default=int(3e4)) # other envs 3e4
    parser.add_argument("--total_step", type=int, default=int(1e6))
    parser.add_argument("--debug", action='store_true', default=False)

    # ----- vmap / sweep plumbing --------------------------------------------
    parser.add_argument("--parallel_runs", type=int, default=1, help="If > 1, train N independent MGMD runs in parallel on a single device via jax.vmap. Env layout uses a single VectorEnv of size parallel_runs * num_vec_envs. Current packed support covers the KL-budget/on-policy-EMA path and fixed-beta mode.")
    parser.add_argument("--hp_pack_inline", type=str, default=None, help="Inline JSON with per-run hyperparameter overrides. Each key is an argparse attribute name of this script (e.g. 'polyak_tau', 'beta', 'advantage_ema_tau', 'guidance_strength_multiplier', 'kl_budget', 'shape_ema_tau', 'seed') mapped to a list of length parallel_runs. Applied after vmap state construction; internally translated to Diffv2TrainState field names via _CLI_TO_FIELD.")
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
    parser.add_argument("--beta_schedule_type", type=str, default='linear', help="Noise schedule type. 'linear': linear beta schedule. 'cosine': cosine schedule (Nichol & Dhariwal). 'constant_kl': constant mutual-information-loss per step, spacing noise levels uniformly in log(1+SNR).")
    parser.add_argument("--snr_max", type=float, default=124.0, help="Maximum SNR (at cleanest noise level). Controls alpha_bar_0 = snr_max/(1+snr_max). Default 124.0 matches the cosine schedule with s=0.008 offset at T=20. All schedule types use this to set the same clean endpoint, so you can switch between cosine/constant_kl/linear while keeping the noise range comparable.")

    # ----- optimization -----------------------------------------------------
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_policy", type=float, default=None)
    parser.add_argument("--lr_q", type=float, default=None)
    parser.add_argument("--update_per_iteration", type=int, default=1)
    parser.add_argument("--critic_update_steps", type=int, default=1, help="Number of Q/V optimizer steps inside each stateless update. All steps reuse the same sampled minibatch and fixed TD/value targets. Default 1.")
    parser.add_argument("--policy_update_steps", type=int, default=1, help="Number of diffusion-policy optimizer steps when the delay_update gate fires. Steps reuse the same tilted-action batch but resample diffusion timestep/noise. Default 1.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for the Q critic. Default 0.99.")
    parser.add_argument("--polyak_tau", type=float, default=0.005, help="Polyak averaging coefficient for target network updates. Default 0.005.")
    parser.add_argument("--delay_update", type=int, default=2, help="Update policy and target networks every delay_update steps. Default 2.")
    parser.add_argument("--reward_scale", type=float, default=1, help="Scale factor applied to rewards before Q/value learning. Default 0.2 matches original MGMD. Set to 1.0 for clarity when using inference-time guidance (adjust --beta accordingly).")

    # ----- Q learning -------------------------------------------------------
    parser.add_argument("--q_agg_sample", type=str, default="min", choices=["min", "mean"], help="Aggregation for Q used in sampling, both for rollout and for the TD next-action sample. The TD-backup target itself is hardcoded to 'min' (clipped double-Q).")
    parser.add_argument("--q_td_huber_width", type=float, default=float("inf"), help="Huber width (delta) for critic TD error in MGMD. Default inf recovers the current MSE TD loss. Effective width is scaled by reward_scale internally.")

    # ----- guidance + KL budget --------------------------------------------
    parser.add_argument("--alpha", type=float, default=None, help="Composite mirror descent retained-policy exponent α in π_new ∝ π_old^α·exp(β·Q). Exactly two of --alpha, --beta, --T, --eta must be specified.")
    parser.add_argument("--beta", type=float, default=None, help="Composite mirror descent Q coefficient β in π_new ∝ π_old^α·exp(β·Q). Exactly two of --alpha, --beta, --T, --eta must be specified.")
    parser.add_argument("--T", type=float, default=None, help="Composite mirror descent entropy temperature T. Exactly two of --alpha, --beta, --T, --eta must be specified.")
    parser.add_argument("--eta", type=float, default=None, help="Composite mirror descent step size η. Exactly two of --alpha, --beta, --T, --eta must be specified.")
    parser.add_argument("--kl_budget", type=float, default=None, help="Total KL divergence budget δ for guidance. Per-dimension budget is δ / act_dim. Initializes β as sqrt(2δ / M_0) with M_0 = --initial_advantage_second_moment_ema, then adapts β online from the advantage-moment EMA. Enables V network and on-policy advantage EMA. Default None (disabled).")
    parser.add_argument("--kl_budget_per_dim", type=float, default=None, help="Per-dimension KL divergence budget δ_d for guidance. Total budget δ = δ_d * act_dim. Initializes β as sqrt(2δ / M_0) with M_0 = --initial_advantage_second_moment_ema, then adapts β online from the advantage-moment EMA. Enables V network and on-policy advantage EMA. Default None (disabled).")
    parser.add_argument("--one_step_dist_shift_beta", action="store_true", default=False, help="Adaptive β from second-order expansion using one-step Monte Carlo covariance estimate. No D_ψ head; estimates c from consecutive (A_t, A_{t+1}) pairs. Requires --kl_budget or --kl_budget_per_dim (defaults to --kl_budget_per_dim=5.33 if neither set).")
    parser.add_argument("--advantage_normalization", action="store_true", default=False)
    parser.add_argument("--advantage_ema_tau", type=float, default=0.0005, help="Per-step EMA rate for advantage second/third moments.")
    parser.add_argument("--shape_ema_tau", type=float, default=0.0001, help="Per-step EMA rate for dimensionless shape s2.")
    parser.add_argument("--initial_advantage_second_moment_ema", type=float, default=1.0, help="Initial value for the advantage second moment EMA E[A^2].")
    parser.add_argument("--initial_dist_shift_shape_ema", type=float, default=-1.0, help="Initial value for the dimensionless distribution-shift shape EMA s2 = (2γc + κ₃) / v^(3/2).")
    parser.add_argument("--x0_hat_clip_radius", type=float, default=float("inf"), help="Clipping radius r for Tweedie clean-action estimates x0_hat used inside guidance/Q evaluation. x0_hat is clipped to [-r, r] before being passed into Q / model-based objectives. Default inf (no clip); in non-latent mode the network-side denoising clip is separately hardcoded to 1.0 to match normalized action bounds.")
    parser.add_argument("--batch_independent_guidance", action="store_true", default=False, help="If set, use jnp.sum instead of jnp.mean inside the guided predictor's q_mean_from_x, so the per-sample Q gradient is independent of batch size (fixes the 1/B attenuation).")
    parser.add_argument("--guidance_strength_multiplier", type=float, default=1.0, help="Constant multiplier applied to the guided-predictor Q scalar before jax.grad. Composes with --batch_independent_guidance.")
    parser.add_argument("--policy_parameterization", type=str, default="E", choices=["E", "f"], help="Parameterization of the energy network scalar output. 'E' (default): network outputs E_theta; eps_pred = sqrt(1-alpha_bar_t)*grad E. 'f': network outputs f = sqrt(1-alpha_bar_t)*E_theta; eps_pred = grad f, energy_fn = f/sqrt(1-alpha_bar_t).")
    parser.add_argument("--policy_final_layer", type=str, default="default", choices=["default", "ff", "L2", "IP"], help="Final-layer head of the scalar policy network. 'default': replace DACERPolicyNet final layer with Linear(1). 'ff': keep full DACERPolicyNet (act_dim output) and tack on an extra learned Linear(1). 'L2': full backbone then E = -0.5*||v||^2 (no extra params). 'IP': full backbone then E = v·a (inner product with action, no extra params).")
    parser.add_argument("--guidance_gradient_space", type=str, default="xt", choices=["xt", "x0hat", "x0hatclipped"], help="Whether to take the Q gradient with respect to 'xt' or the predicted clean action 'x0hat' or its clipped version 'x0hatclipped'.")

    # ----- MALA -------------------------------------------------------------
    parser.add_argument("--mala_steps", type=int, default=0, help="Number of MALA correction steps per diffusion step.")
    parser.add_argument("--mala_adapt_rate", type=float, default=0.05, help="Robbins-Monro adaptation rate for MALA log_eta_scale updates.")
    parser.add_argument("--denoising_predictor", type=str, default="DDPM_mean", choices=["Identity", "DDPM_mean", "DDIM"], help="Predictor transition after each MALA correction level. Identity skips denoising; DDPM_mean uses the guided DDPM posterior mean; DDIM uses the guided deterministic DDIM update.")

    return parser


def validate_args(args, parser: argparse.ArgumentParser) -> None:
    """Post-parse validation. Calls ``parser.error`` on bad combinations."""
    if args.kl_budget is not None and args.kl_budget_per_dim is not None:
        parser.error("--kl_budget and --kl_budget_per_dim are mutually exclusive")

    cmd_params = {"--alpha": args.alpha, "--beta": args.beta, "--T": args.T, "--eta": args.eta}
    specified_cmd_params = [name for name, value in cmd_params.items() if value is not None]
    if args.hp_pack_inline is None and len(specified_cmd_params) != 2:
        parser.error(
            "Exactly two of --alpha, --beta, --T, --eta must be specified "
            f"(got {specified_cmd_params or 'none'})."
        )
    if args.hp_pack_inline is not None and len(specified_cmd_params) > 2:
        parser.error(
            "At most two of --alpha, --beta, --T, --eta may be specified on the base CLI "
            "when --hp_pack_inline is used."
        )

    if args.advantage_normalization and (
        args.kl_budget is not None
        or args.kl_budget_per_dim is not None
        or args.one_step_dist_shift_beta
    ):
        parser.error(
            "--advantage_normalization is mutually exclusive with --kl_budget, "
            "--kl_budget_per_dim, and --one_step_dist_shift_beta"
        )

    if args.flatten_UTD:
        args.num_vec_envs = 1
        args.update_per_iteration = 1

    # --one_step_dist_shift_beta implies a KL budget (default 5.33 per dim)
    if args.one_step_dist_shift_beta and args.kl_budget is None and args.kl_budget_per_dim is None:
        args.kl_budget_per_dim = 5.33

    if args.parallel_runs <= 0:
        parser.error("--parallel_runs must be >= 1.")
    if args.num_vec_envs <= 0:
        parser.error("--num_vec_envs must be > 0.")
    if args.update_per_iteration <= 0:
        parser.error("--update_per_iteration must be > 0.")
    if args.critic_update_steps <= 0:
        parser.error("--critic_update_steps must be > 0.")
    if args.policy_update_steps <= 0:
        parser.error("--policy_update_steps must be > 0.")
    if args.delay_update <= 0:
        parser.error("--delay_update must be > 0.")
    if args.mala_steps <= 0:
        parser.error("--mala_steps must be > 0; the non-MALA sampling branches have been removed.")
