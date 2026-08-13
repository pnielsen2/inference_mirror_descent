"""Argparse / post-parse validation for ``scripts/train_mujoco.py``.

Kept in its own module so the entry-point reads top-to-bottom and the CLI
surface is the only thing one needs to scan to understand what flags the
three protected launch commands set.
"""
import argparse


def _parse_guidance_strength_multiplier(value: str):
    try:
        return float(value)
    except ValueError:
        normalized = value.lower()
        if normalized == "increasing":
            return normalized
        raise argparse.ArgumentTypeError(
            "--guidance_strength_multiplier must be a float or 'increasing'"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # ----- env / run control -------------------------------------------------
    parser.add_argument("--alg", type=str, default="mgmd", choices=["mgmd"])
    parser.add_argument("--mgmd_variant", type=str, default="mgmd", choices=["mgmd", "rsm", "soft_resample"], help="'mgmd' keeps the current MALA-guided sampler + sampler-distillation policy update. 'rsm' uses an unguided DDPM diffusion sampler for rollout and TD next-action sampling, then updates the policy with RSM-weighted diffusion loss. 'soft_resample' samples N unguided diffusion candidates and resamples from the per-state Boltzmann weights exp(beta * processed_Q).")
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

    # ----- diagnostic snapshots --------------------------------------------
    parser.add_argument("--save_diagnostic_snapshots", action="store_true", default=False, help="Save host-side diagnostic snapshots at selected env steps. Snapshots include the vmapped algorithm state, the next rollout observations, and a fixed replay minibatch for later sampler-distribution analysis.")
    parser.add_argument("--diagnostic_snapshot_steps", type=int, nargs="*", default=[], help="Env-step targets at which to save diagnostic snapshots. The trainer saves the first time the per-run env step reaches or crosses each target.")
    parser.add_argument("--diagnostic_snapshot_batch_size", type=int, default=256, help="Number of replay transitions per run to save in each diagnostic snapshot for TD next-action / distillation diagnostics.")
    parser.add_argument("--diagnostic_snapshot_buffer_fraction", type=float, default=0.0, help="Optional replay-buffer subset fraction to save per run in each diagnostic snapshot. 0 disables subset saving; 0.1 saves a uniform 10%% subset of each run's current valid buffer without advancing buffer RNGs.")
    parser.add_argument("--diagnostic_snapshot_dir", type=str, default=None, help="Directory where diagnostic snapshots are written. Required when --save_diagnostic_snapshots is set.")

    # ----- separate evaluation ---------------------------------------------
    parser.add_argument("--eval_every", type=int, default=0, help="Run separate evaluation every this many training env steps. Default 0 disables evaluation.")
    parser.add_argument("--eval_n_episodes", type=int, default=10, help="Number of complete episodes per seed/run for each separate evaluation.")
    parser.add_argument("--eval_best_of_n_actions", type=int, default=1, help="Best-of-N candidate count used only by separate evaluation. The selected action is executed without post-selection exploration noise.")
    parser.add_argument("--eval_seed", type=int, default=None, help="Base seed for separate evaluation envs. Default derives one from --seed.")

    # ----- networks ---------------------------------------------------------
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)
    parser.add_argument("--num_q_networks", type=int, default=2, help="Number of Q critic networks to train (default 2, i.e. twin Q).")
    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size for training updates.")
    parser.add_argument("--orthogonal_init", action="store_true", default=False, help="Initialize ALL network weight matrices (Q critics, diffusion/energy policy net, and the optional KL-budget V-network) with random orthogonal matrices (haiku Orthogonal, scale 1.0). Biases keep their default zero init. Default off = haiku's TruncatedNormal(1/sqrt(fan_in)).")

    # ----- diffusion schedule ----------------------------------------------
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--beta_schedule_type", type=str, default='linear', help="Noise schedule type. 'linear': linear beta schedule. 'cosine': cosine schedule (Nichol & Dhariwal). 'constant_kl': constant mutual-information-loss per step, spacing noise levels uniformly in log(1+SNR).")
    parser.add_argument("--snr_max", type=float, default=124.0, help="Maximum SNR (at cleanest noise level). Controls alpha_bar_0 = snr_max/(1+snr_max). Default 124.0 matches the cosine schedule with s=0.008 offset at T=20. All schedule types use this to set the same clean endpoint, so you can switch between cosine/constant_kl/linear while keeping the noise range comparable.")

    # ----- optimization -----------------------------------------------------
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_policy", type=float, default=None)
    parser.add_argument("--lr_q", type=float, default=None)
    parser.add_argument("--lr_anneal", action="store_true", default=False, help="Anneal BOTH the Q and policy learning rates by the same linear factor as a function of the env-step count (ported from diffusion_policy_online_rl). Factor is 1.0 up to --lr_anneal_transition_begin env steps, then decreases linearly to --lr_anneal_end_factor over --lr_anneal_transition_steps env steps, then held. The optional V-network LR (KL-budget mode) is not annealed. The defaults reproduce diffusion_policy_online_rl's default LR-vs-env-step curve exactly (see below).")
    parser.add_argument("--lr_anneal_end_factor", type=float, default=0.1, help="Final LR multiplier for --lr_anneal. Default 0.1 reproduces diffusion_policy_online_rl's default policy-LR decay (train_mujoco.py: lr=3e-4 -> lr_schedule_end=3e-5).")
    parser.add_argument("--lr_anneal_transition_begin", type=int, default=250000, help="Env step at which LR annealing begins. Default 250000 reproduces diffusion_policy_online_rl's transition_begin=2.5e4 POLICY-OPTIM steps, converted to env steps via its default 10 env-steps-per-policy-update (num_vec_envs=5 * delay_update=2). Units are ENV steps here.")
    parser.add_argument("--lr_anneal_transition_steps", type=int, default=500000, help="Number of ENV steps over which the LR anneals from 1x down to --lr_anneal_end_factor. Default 500000 = diffusion_policy_online_rl's transition_steps=5e4 policy-optim steps * 10 env-steps-per-policy-update.")
    parser.add_argument("--update_per_iteration", type=int, default=1)
    parser.add_argument("--fused_denoising", action="store_true", default=False, help="Proposed Algorithm 1 Modification: fuse the stepping + training denoising passes into one. The current transition (s'_B = s) is injected as the last --num_vec_envs rows of the training minibatch; the update denoises the whole minibatch, and the denoised next-action for those rows is reused to step the env (dropping the separate get_action denoising pass). --update_per_iteration is respected (the fused update is #1; the rest are training-only). Requires --batch_size > --num_vec_envs.")
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
    parser.add_argument("--guidance_strength_multiplier", type=_parse_guidance_strength_multiplier, default=1.0, help="Multiplier applied to the guided-predictor Q scalar before jax.grad. Accepts a float for constant scaling or 'increasing' for the normalized α_t schedule from TFG (2409.15761). Composes with --batch_independent_guidance.")
    parser.add_argument("--policy_parameterization", type=str, default="E", choices=["E", "f"], help="Parameterization of the energy network scalar output. 'E' (default): network outputs E_theta; eps_pred = sqrt(1-alpha_bar_t)*grad E. 'f': network outputs f = sqrt(1-alpha_bar_t)*E_theta; eps_pred = grad f, energy_fn = f/sqrt(1-alpha_bar_t).")
    parser.add_argument("--policy_final_layer", type=str, default="default", choices=["default", "ff", "L2", "IP"], help="Final-layer head of the scalar policy network. 'default': replace DACERPolicyNet final layer with Linear(1). 'ff': keep full DACERPolicyNet (act_dim output) and tack on an extra learned Linear(1). 'L2': full backbone then E = -0.5*||v||^2 (no extra params). 'IP': full backbone then E = v·a (inner product with action, no extra params).")
    parser.add_argument("--guidance_gradient_space", type=str, default="xt", choices=["xt", "x0hat", "x0hatclipped"], help="Whether to take the Q gradient with respect to 'xt' or the predicted clean action 'x0hat' or its clipped version 'x0hatclipped'.")

    # ----- multi-action denoising + V-free advantage normalization ----------
    parser.add_argument("--num_denoised_actions", type=int, default=1, help="Training-time number K of denoised next-actions per sampled replay state. The TD backup averages clipped-double-Q over these K actions, and the diffusion policy regresses toward all K. K>=2 is required for --batch_advantage_normalization. This is separate from rollout-time --best_of_n_actions. Changes tensor shapes, so it is a 'hard' (non-vmap-packable) sweep axis in launch.py. Default 1.")
    parser.add_argument("--soft_resample_actions", type=int, default=1, help="Candidate count N for --mgmd_variant soft_resample. The sampler draws N unguided diffusion actions per state, forms per-state Boltzmann weights from exp(beta * processed_Q), resamples one action for rollout/TD backup, and trains the diffusion policy against all N candidates with those normalized weights. This is a hard sweep axis because it changes JAX tensor shapes.")
    parser.add_argument("--soft_resample_ess_dump_interval", type=int, default=0, help="Env-step interval for logging full soft-resample ESS/pmax histograms. Scalar ESS/pmax summaries are still logged at the normal update logging cadence. Default 0 disables full histogram dumps.")
    parser.add_argument("--best_of_n_actions", type=int, default=1, help="Rollout-time best-of-N candidate count. N=1 preserves the current uniform single-sample rollout exactly. N>1 denoises N candidate actions at the current env state, picks the highest final online --q_agg_sample Q candidate, then adds DPMD-style learned Gaussian execution noise with std exp(log_best_of_n_noise_scale). Separate from training-time --num_denoised_actions.")
    parser.add_argument("--best_of_n_td_action_sampling", action="store_true", default=False, help="Also use DPMD-style best-of-N for TD next-action sampling. When set, the training TD sampler denoises --best_of_n_td_actions candidates at s', selects the highest final online --q_agg_sample Q candidate, adds the same learned Gaussian best-of-N noise, and returns that single action for both the TD backup and policy distillation. Requires --num_denoised_actions 1.")
    parser.add_argument("--best_of_n_td_actions", type=int, default=None, help="TD next-action best-of-N candidate count used only with --best_of_n_td_action_sampling. Defaults to --best_of_n_actions for backward compatibility, so old commands keep their previous rollout+TD behavior. This is a hard sweep axis because it changes JAX sampler shapes.")
    parser.add_argument("--best_of_n_noise_scale_init", type=float, default=0.5, help="Initial std for learned post-best-of-N rollout Gaussian noise. Default 0.5 matches DPMD's exp(log(5))*noise_scale with noise_scale=0.1.")
    parser.add_argument("--best_of_n_noise_lr", type=float, default=7e-3, help="Adam learning rate for the DPMD-style best-of-N noise scheduler. Used when rollout or TD best-of-N uses N > 1.")
    parser.add_argument("--delay_best_of_n_noise_update", type=int, default=250, help="Update the best-of-N noise scheduler every this many MGMD update steps. Used when rollout or TD best-of-N uses N > 1.")
    parser.add_argument("--best_of_n_noise_target_entropy_scale", type=float, default=0.9, help="Target entropy coefficient c in H_target = -c * act_dim for the learned best-of-N rollout noise scheduler. Default 0.9 matches DPMD.")
    parser.add_argument("--batch_advantage_normalization", action="store_true", default=False, help="V-free guidance normalization. At each MALA/denoising step, rescale Q by 1/sqrt(mean_s Var_K(Q)): the per-state sample variance (ddof=1) of Q over the K denoised Tweedie estimates, averaged over the batch, square-rooted, and stop-gradient'd. Requires --num_denoised_actions >= 2. Composes additively with --beta / other normalizations.")
    parser.add_argument("--q_loss_normalization", action="store_true", default=False, help="V-free guidance normalization. Divide the guidance Q by sqrt(EMA(Q TD loss)), where the EMA (rate --advantage_ema_tau) is tracked in-graph from the critic loss. Needs neither a V network nor multiple actions. Mutually exclusive with --advantage_normalization / --kl_budget(_per_dim) / --one_step_dist_shift_beta.")
    parser.add_argument("--ema_advantage_normalization", action="store_true", default=False, help="V-free guidance normalization ported from diffusion_policy_online_rl. Divide the guidance Q by a slow EMA of the batch std of the online --q_agg_sample-aggregated Q at the sampled next-actions: Q_norm = (Q - mu)/sigma with mu, sigma stop-gradient'd (mu cancels in the guidance gradient, so this is effectively a 1/sigma rescale). EMA rate --advantage_norm_ema_rate. Needs neither a V network nor multiple actions. Mutually exclusive with --advantage_normalization / --kl_budget(_per_dim) / --one_step_dist_shift_beta / --q_loss_normalization.")
    parser.add_argument("--advantage_norm_ema_rate", type=float, default=0.001, help="Per-step EMA rate r for the --ema_advantage_normalization running mean/std: x += r*(batch - x). Default 0.001 (the value hardcoded in diffusion_policy_online_rl).")

    # ----- MALA -------------------------------------------------------------
    parser.add_argument("--mala_steps", type=int, default=0, help="Number of MALA correction steps per diffusion step.")
    parser.add_argument("--mala_adapt_rate", type=float, default=0.05, help="Robbins-Monro adaptation rate for MALA log_eta_scale updates.")
    parser.add_argument("--denoising_predictor", type=str, default="DDPM_mean", choices=["Identity", "DDPM_mean", "DDIM"], help="Predictor transition after each MALA correction level. Identity skips denoising; DDPM_mean uses the guided DDPM posterior mean; DDIM uses the guided deterministic DDIM update.")

    return parser


def validate_args(args, parser: argparse.ArgumentParser) -> None:
    """Post-parse validation. Calls ``parser.error`` on bad combinations."""
    if isinstance(args.guidance_strength_multiplier, str) and args.guidance_strength_multiplier != "increasing":
        parser.error("--guidance_strength_multiplier only supports the string value 'increasing'")

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

    if args.num_denoised_actions < 1:
        parser.error("--num_denoised_actions must be >= 1.")

    if args.soft_resample_actions < 1:
        parser.error("--soft_resample_actions must be >= 1.")

    if args.best_of_n_actions < 1:
        parser.error("--best_of_n_actions must be >= 1.")

    if args.best_of_n_td_actions is None:
        args.best_of_n_td_actions = args.best_of_n_actions

    if args.best_of_n_td_actions < 1:
        parser.error("--best_of_n_td_actions must be >= 1.")

    if args.mgmd_variant in {"rsm", "soft_resample"} and (
        args.best_of_n_actions != 1
        or args.best_of_n_td_action_sampling
        or args.best_of_n_td_actions != 1
    ):
        parser.error(f"--mgmd_variant {args.mgmd_variant} does not support best-of-N options in this implementation.")

    if args.mgmd_variant == "soft_resample" and args.fused_denoising:
        parser.error("--mgmd_variant soft_resample is not supported with --fused_denoising in this first implementation.")

    if args.best_of_n_td_action_sampling and args.num_denoised_actions != 1:
        parser.error("--best_of_n_td_action_sampling currently requires --num_denoised_actions 1.")

    uses_best_of_n_noise = args.best_of_n_actions > 1 or (
        args.best_of_n_td_action_sampling and args.best_of_n_td_actions > 1
    )

    if uses_best_of_n_noise and args.best_of_n_noise_scale_init <= 0:
        parser.error("--best_of_n_noise_scale_init must be > 0 when a best-of-N sampler uses N > 1.")

    if uses_best_of_n_noise and args.best_of_n_noise_lr <= 0:
        parser.error("--best_of_n_noise_lr must be > 0 when a best-of-N sampler uses N > 1.")

    if args.delay_best_of_n_noise_update <= 0:
        parser.error("--delay_best_of_n_noise_update must be > 0.")

    if uses_best_of_n_noise and args.best_of_n_noise_target_entropy_scale <= 0:
        parser.error("--best_of_n_noise_target_entropy_scale must be > 0 when a best-of-N sampler uses N > 1.")

    if uses_best_of_n_noise and args.fused_denoising:
        parser.error("best-of-N samplers with N > 1 are not supported with --fused_denoising in this first implementation.")

    batch_adv_sample_count = args.soft_resample_actions if args.mgmd_variant == "soft_resample" else args.num_denoised_actions
    if args.batch_advantage_normalization and batch_adv_sample_count < 2:
        parser.error(
            "--batch_advantage_normalization needs the per-state Q variance over "
            "the denoised actions, so it requires at least two candidates."
        )

    # --q_loss_normalization drives the same beta/sqrt(E[A^2]-slot) rescale as the
    # V-based paths, so at most one of these guidance-normalization sources may run.
    if args.q_loss_normalization and (
        args.advantage_normalization
        or args.kl_budget is not None
        or args.kl_budget_per_dim is not None
        or args.one_step_dist_shift_beta
    ):
        parser.error(
            "--q_loss_normalization is mutually exclusive with "
            "--advantage_normalization, --kl_budget, --kl_budget_per_dim, and "
            "--one_step_dist_shift_beta"
        )

    # --ema_advantage_normalization rescales the guidance Q by an EMA(std) and is
    # its own guidance-normalization source, so it cannot combine with the other
    # sources that also drive the beta/Q rescale.
    if args.ema_advantage_normalization and (
        args.advantage_normalization
        or args.kl_budget is not None
        or args.kl_budget_per_dim is not None
        or args.one_step_dist_shift_beta
        or args.q_loss_normalization
    ):
        parser.error(
            "--ema_advantage_normalization is mutually exclusive with "
            "--advantage_normalization, --kl_budget, --kl_budget_per_dim, "
            "--one_step_dist_shift_beta, and --q_loss_normalization"
        )

    if args.flatten_UTD:
        args.num_vec_envs = 1
        args.update_per_iteration = 1

    # Fused denoising injects --num_vec_envs current transitions as the last rows
    # of the minibatch, so there must be room for at least one randomly sampled row.
    if args.fused_denoising and args.batch_size <= args.num_vec_envs:
        parser.error(
            "--fused_denoising requires --batch_size > --num_vec_envs "
            f"(got batch_size={args.batch_size}, num_vec_envs={args.num_vec_envs})."
        )

    # --one_step_dist_shift_beta implies a KL budget (default 5.33 per dim)
    if args.one_step_dist_shift_beta and args.kl_budget is None and args.kl_budget_per_dim is None:
        args.kl_budget_per_dim = 5.33

    if args.parallel_runs <= 0:
        parser.error("--parallel_runs must be >= 1.")
    if args.num_vec_envs <= 0:
        parser.error("--num_vec_envs must be > 0.")
    if args.diagnostic_snapshot_batch_size <= 0:
        parser.error("--diagnostic_snapshot_batch_size must be > 0.")
    if args.diagnostic_snapshot_buffer_fraction < 0.0 or args.diagnostic_snapshot_buffer_fraction > 1.0:
        parser.error("--diagnostic_snapshot_buffer_fraction must be between 0 and 1.")
    if any(step <= 0 for step in args.diagnostic_snapshot_steps):
        parser.error("--diagnostic_snapshot_steps must contain positive env-step integers.")
    if args.save_diagnostic_snapshots:
        if not args.diagnostic_snapshot_steps:
            parser.error("--save_diagnostic_snapshots requires --diagnostic_snapshot_steps.")
        if args.diagnostic_snapshot_dir is None:
            parser.error("--save_diagnostic_snapshots requires --diagnostic_snapshot_dir.")
    if args.eval_every < 0:
        parser.error("--eval_every must be >= 0.")
    if args.eval_n_episodes <= 0:
        parser.error("--eval_n_episodes must be > 0.")
    if args.eval_best_of_n_actions <= 0:
        parser.error("--eval_best_of_n_actions must be > 0.")
    if args.update_per_iteration <= 0:
        parser.error("--update_per_iteration must be > 0.")
    if args.critic_update_steps <= 0:
        parser.error("--critic_update_steps must be > 0.")
    if args.policy_update_steps <= 0:
        parser.error("--policy_update_steps must be > 0.")
    if args.delay_update <= 0:
        parser.error("--delay_update must be > 0.")
    if args.mgmd_variant == "mgmd" and args.mala_steps <= 0:
        parser.error("--mala_steps must be > 0; the non-MALA sampling branches have been removed.")
