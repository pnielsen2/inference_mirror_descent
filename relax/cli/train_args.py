"""Argparse / post-parse validation for ``scripts/train_mujoco.py``.

Kept in its own module so the entry-point reads top-to-bottom and the CLI
surface is the only thing one needs to scan to understand what flags the
three protected launch commands set.
"""
import argparse
import json


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
    parser.add_argument("--hp_pack_inline", type=str, default=None, help="Inline JSON with per-run hyperparameter overrides. Each key is an argparse attribute name of this script (e.g. 'q_polyak_tau', 'beta', 'advantage_ema_tau', 'guidance_strength_multiplier', 'kl_budget', 'shape_ema_tau', 'seed') mapped to a list of length parallel_runs. Applied after vmap state construction; internally translated to Diffv2TrainState field names via _CLI_TO_FIELD.")
    parser.add_argument("--sweep_id", type=int, default=None, help="Launcher-assigned integer identifying this sweep. When set, every wandb run from this invocation is placed in wandb group 'sweep_<sweep_id>', and each per-vmap-slot run's config includes a 'config_tag' field built from sweep_id + the per-slot hyperparameters (excluding seed/env) so a single tag value filters wandb to all runs across envs/seeds that share this hp configuration.")
    parser.add_argument("--config_tag_keys", type=str, default=None, help="Comma-separated list of argparse attribute names whose values should be included in the per-slot config_tag. Typically set automatically by scripts/launch.py to the union of all --ablate hard+easy flags (minus env and seed). Values come from the hp_pack (per-slot) when the key is a pack key, else from this script's CLI args (shared across all vmap slots within the job).")

    # ----- diagnostic snapshots --------------------------------------------
    parser.add_argument("--save_diagnostic_snapshots", action="store_true", default=False, help="Save host-side diagnostic snapshots at selected env steps. Snapshots include the vmapped algorithm state, the next rollout observations, and a fixed replay minibatch for later sampler-distribution analysis.")
    parser.add_argument("--diagnostic_snapshot_steps", type=int, nargs="*", default=[], help="Env-step targets at which to save diagnostic snapshots. The trainer saves the first time the per-run env step reaches or crosses each target.")
    parser.add_argument("--diagnostic_snapshot_batch_size", type=int, default=256, help="Number of replay transitions per run to save in each diagnostic snapshot for TD next-action / distillation diagnostics.")
    parser.add_argument("--diagnostic_snapshot_buffer_fraction", type=float, default=0.0, help="Optional replay-buffer subset fraction to save per run in each diagnostic snapshot. 0 disables subset saving; 0.1 saves a uniform 10%% subset of each run's current valid buffer without advancing buffer RNGs.")
    parser.add_argument("--diagnostic_snapshot_dir", type=str, default=None, help="Directory where diagnostic snapshots are written. Required when --save_diagnostic_snapshots is set.")

    # ----- separate evaluation ---------------------------------------------
    parser.add_argument("--eval_every", type=int, default=1000000, help="Run separate evaluation episodes every this many training env steps (0 disables). These are extra episodes in their own envs, recorded to nothing: the training trajectory is bit-identical with evaluation on or off, only wall-clock changes. Logged per run as eval/episode_return_{mean,std,min,max} and eval/episode_length_mean, and mirrored to eval_episode_returns.csv next to episode_returns.csv. Cost per evaluation is ~one episode length of denoising passes (all episodes run in parallel). The default equals the default --total_step, so out of the box this is a single end-of-run evaluation score and nothing during training; pass a fraction of --total_step (e.g. 50000 for 20 points) to get an evaluation curve. See relax/trainer/evaluation.py.")
    parser.add_argument("--eval_n_episodes", type=int, default=10, help="Complete episodes per run per evaluation. Also the number of eval envs allocated per run, since the episodes run in parallel (one env each), so this multiplies the eval env count and its memory, not the number of sequential denoising passes. Every run and every sweep config evaluates from the same --eval_seed-derived initial states, so this is the sample size of a paired comparison.")
    parser.add_argument("--eval_best_of_n_actions", type=int, default=32, help="Candidate count N for evaluation-time best-of-N action selection: N iid actions are denoised per state and the one with the highest --q_agg_sample-aggregated Q is executed. The default 32 therefore scores the best-of-32 deployment policy, whose quality depends on critic accuracy as well as on the policy, at 32x the eval denoising FLOPs of a single sample (but the same number of sequential passes, since the candidates are batched). Set 1 to score the policy exactly as it is rolled out, with no critic involvement. Independent of --num_denoised_actions, which is a training knob.")
    parser.add_argument("--eval_seed", type=int, default=0, help="Seed for the eval envs and the eval action noise. Deliberately independent of --seed: holding it fixed means every config, every seed and every eval point along a curve is evaluated from the same initial states with the same denoising noise, so differences between eval points are differences in the policy rather than in the draw. Change it only to check that a result is not an artifact of one set of eval initial states.")

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
    parser.add_argument("--beta_schedule_type", type=str, default='linear', help="Noise schedule type. 'linear': linear beta schedule. 'cosine': cosine schedule (Nichol & Dhariwal). 'constant_kl': constant mutual-information-loss per step, spacing noise levels uniformly in log(1+SNR). 'adaptive': no family at all -- the T log-SNR levels are free knots, pinned at --noise_schedule_log_snr_{max,min} and initialised to cosine's shape cut to that range, and each batch moves them a fraction --noise_schedule_gamma toward the layout that spends equal cost per step. Cost is the score-optimal one of Williams et al., 'Score-Optimal Diffusion Schedules' (NeurIPS 2024): sigma^2 times the Fisher divergence between adjacent levels, read straight off the MALA drifts, so it is free (see relax/algorithm/noise_schedule.py). The energy net then conditions on lambda rather than on the level index, and distillation draws its noise levels continuously (uniform in the knots' own time coordinate, lambda interpolated between them) so the policy stays in distribution as the schedule moves. Requires --denoising_predictor Identity (or Identity_then_DDPM_mean, whose only non-identity step is below every scored interval) and --guidance_gradient_space xt; incompatible with --s_hat / --estimate_s_hat, since the knots already say where the ladder sits.")
    parser.add_argument("--snr_max", type=float, default=124.0, help="Maximum SNR (at the cleanest noise level), i.e. alpha_bar_0 = snr_max/(1+snr_max). ONLY USED BY 'constant_kl', which is defined by it. 'cosine' and 'linear' are used natively, so their clean endpoint tracks --diffusion_steps (cosine SNR_0 = 124/401/1155 at T=20/40/80); 124.0 is exactly native cosine at T=20.")
    parser.add_argument("--noise_cond_theta", type=int, default=1000, help="Frequency base of the policy net's sinusoidal noise-level embedding, which fixes the input RANGE it resolves: the dim/2 frequencies run from 1 down to theta^-(7/8) rad per unit (dim=16), so the top channel always gives ~1 rad per unit and the bottom sweeps range*theta^-(7/8) radians end to end. Channels sweeping far under a radian are near-constants, i.e. wasted. Default 1000 is shaped for a log-SNR range of [-50, 50] (6 of 8 channels informative, 1 monotone). 10000 is the inherited DDPM/Transformer value, built for a range ~20x wider, under which 5 of 8 are flat over that range -- pass it to reproduce pre-existing runs exactly. This applies to EVERY --beta_schedule_type, so hold it fixed (or ablate it explicitly) when comparing schedules. It does not change how far apart ADJACENT levels sit in embedding space; only rescaling the conditioning input itself would.")
    parser.add_argument("--noise_schedule_log_snr_max", type=float, default=15.0, help="Log-SNR of the CLEANEST knot under --beta_schedule_type adaptive, pinned there for the whole run (the equal-cost update returns both endpoints unchanged, so this is a standing constraint and not just an initial value). Sets how clean the returned sample can get: sigma = sqrt(sigmoid(-lambda)), so the default 15 resolves action detail down to 5.5e-4, ~50x finer than native cosine at T=80 (0.029). Do not raise it much: the cost weight is sigma^2, which vanishes here, so the equal-cost rule assigns the clean tail almost no cost and knots migrate away from it, while the MALA step size there scales with 1-abar and gets too small to move -- at 20 the resulting beta rounds to 1 in float32. Every configuration in the paper's released code sits between 10 and 16. Ignored by every other schedule family.")
    parser.add_argument("--noise_schedule_log_snr_min", type=float, default=-15.0, help="Log-SNR of the NOISIEST knot under --beta_schedule_type adaptive, pinned there for the whole run. The chain is initialised from N(0, I) and this knot is treated as BEING that reference, so it must be noisy enough for the two to agree: at the default -15, abar = 3e-7. Raising it introduces an unmodelled initialisation mismatch (the one term the cost deliberately drops); lowering it only wastes knots in a region the cost rule will evacuate anyway. The paper's image runs sit at -20 to -23. Ignored by every other schedule family.")
    parser.add_argument("--noise_schedule_gamma", type=float, default=1e-3, help="Fraction of the way the adaptive knots move toward the equal-cost layout each update: levels <- gamma*optimal + (1-gamma)*levels (Algorithm 2 of Williams et al.). Acts as an EMA over the per-batch cost estimates, so 1/gamma is its time constant in UPDATES. Calibrated against the paper's released code rather than its stated gamma: there, gamma is applied per schedule *resample*, which only fires once every level has been visited n_l_min times (n_l_min=24 with T=1000 levels at batch 384 for their CIFAR run, i.e. roughly every 63 batches), so their 0.01 is ~1.6e-4 per batch. Our MALA sampler sweeps every level on every update with batch_size samples per level, so a per-update gamma here is ~60x more schedule motion per batch than their image runs at the same number. Default 1e-3 sits near their per-batch rate while still adapting within a few thousand steps. Their 1D runs, which do resample nearly every batch, used 0.01-0.1.")
    parser.add_argument("--noise_schedule_warmup", type=float, default=1e5, help="Updates to run on the INITIAL (cosine-truncated) ladder before --beta_schedule_type adaptive starts moving its knots. Not optional in spirit: the score-optimal cost differences the two adjacent levels' scores, and for an untrained denoiser d x0hat/dx scales like 1/sqrt(abar) rather than the correct sqrt(abar) -- a factor 1.6e7 at lambda=-15 -- so the cost grows monotonically toward the noisy end and the equal-cost rule collapses every interior knot onto --noise_schedule_log_snr_min. That collapse then starves distillation of every level except the noisiest, so the denoiser never becomes consistent and never recovers. Williams et al. burn in the score for 160k iterations on a fixed cosine schedule before adapting for 50k (Appendix C.3), i.e. 76 percent of training. Default 1e5 is 10 percent of a 1e6-step run and ~100x the collapse timescale (1/gamma). Watch Schedule/x0hat_clip_{noisiest,cleanest}: adapting is safe once the noisiest is at or below the cleanest. The cost is still logged during warmup, so you can see it settle. Per-seed vmappable, so --ablate on it is free.")
    parser.add_argument("--guidance_snr_anneal", type=str, default="none", choices=["none", "sqrt_abar", "abar"], help="Damp the Q-guidance strength at low SNR in the MALA TARGET (and, to keep the proposal consistent, in the guided predictor): beta_eff(t) = beta * f(alpha_bar_t). The guided term is Q at the Tweedie estimate x0_hat = (x - omac grad E)/sqrt(abar), so its gradient carries 1/sqrt(abar); an exact score cancels that (the true posterior mean moves with x only as (sqrt(abar)/omac) Cov[x_0|x], which vanishes as the posterior widens to the prior), but any score error is amplified by it -- 1808x at lambda = -15. Left undamped this makes the noisy end of the tilted path astronomically expensive and collapses an adaptive schedule onto lambda_min. 'abar' restores the exact asymptotics: 1 at the clean end, where the Tweedie approximation is good and the sampling target must be left untouched, and decaying like sqrt(abar) at the noisy end so rho_lambda -> N(0, I). (For a Gaussian prior of action scale s the exact form is sigmoid(lambda + 2 log s); abar is the s = 1 case.) 'sqrt_abar' only holds the guidance gradient constant in lambda rather than decaying. 'none' (default) is the historical behaviour. NOTE this changes the density MALA samples for EVERY --beta_schedule_type, so runs with it on are not comparable to runs without it.")
    parser.add_argument("--s_hat", type=float, default=1.0, help="Assumed standard deviation of the clean actions. Shifts whatever --beta_schedule_type is chosen by lambda -> lambda - 2 log(s_hat) in log-SNR, i.e. abar -> abar/(abar + s_hat^2 (1-abar)); equivalent to standardizing the actions by s_hat before diffusing. Schedules are tuned for unit variance, so the default 1.0 is the (no-op) status quo; s_hat<1 stops the schedule wasting steps on the noise-dominated end when the policy is narrower than that. Cost of a wrong value is cosh(log(s/s_hat)), so it is flat: a 2x error costs 1.25x. Not applicable to --beta_schedule_type adaptive, which places its own levels (the knots already fix where the ladder sits) and rejects the combination.")

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
    parser.add_argument("--policy_update_steps", type=int, default=1, help="Number of diffusion-policy optimizer steps when the --delay_policy_update gate fires. Steps reuse the same tilted-action batch but resample diffusion timestep/noise. Multiplies --distillation_steps, which is the same knob one level up (a pass over the distillation buffer). Default 1.")
    parser.add_argument("--distillation_buffer_size", type=int, default=1, help="Number of past updates' tilted-action blocks the distillation buffer holds: capacity = --batch_size * --num_denoised_actions * this. Default 1 is the pre-buffer behaviour, where an update's denoised targets are score-matched once and thrown away. Larger values keep each target for that many updates, so the policy is regressed onto it that many times per denoiser pass (the expensive half of an update), at the cost of staleness -- a target written n updates ago was drawn from the policy tilted by an older Q. Changes tensor shapes, so it is a 'hard' (non-vmap-packable) sweep axis in launch.py. See relax/algorithm/distillation.py.")
    parser.add_argument("--distillation_steps", type=int, default=1, help="Number of reshuffled passes over the distillation buffer per policy update. Each pass cuts the buffer into --batch_size minibatches and takes one score-matching optimizer step per minibatch, so an update takes --distillation_steps * --num_denoised_actions * --distillation_buffer_size steps (times --policy_update_steps). Note the minibatch is --batch_size regardless, so --num_denoised_actions K >= 2 now takes K steps of that size per pass rather than one step on all K*batch targets. Default 1.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for the Q critic. Default 0.99.")
    parser.add_argument("--q_polyak_tau", type=float, default=0.005, help="Polyak averaging coefficient for the target-Q soft update. Default 0.005.")
    parser.add_argument("--policy_polyak_tau", type=float, default=0.005, help="Polyak averaging coefficient for the target-policy soft update (only used with --use_target_policy_training). Default 0.005.")
    parser.add_argument("--delay_target_q_update", type=int, default=2, help="Polyak-update the target Q every this many update steps. Default 2.")
    parser.add_argument("--delay_policy_update", type=int, default=2, help="Take a diffusion-policy optimizer step every this many update steps. Default 2.")
    parser.add_argument("--delay_target_policy_update", type=int, default=2, help="Polyak-update the target policy every this many update steps (only used with --use_target_policy_training). Default 2.")
    parser.add_argument("--use_target_policy_training", action="store_true", default=False, help="Materialize a Polyak-averaged target policy (rate --policy_polyak_tau, period --delay_target_policy_update) and use it INSTEAD of the online policy to denoise the tilted next-actions during TRAINING. Rollout action sampling is unaffected and always uses the online policy. Without this flag no target policy is allocated at all. Incompatible with --fused_denoising, which reuses the training-branch denoised action to step the env.")
    parser.add_argument("--use_target_q_sampling_training", action="store_true", default=False, help="Use the target Q ensemble instead of the online one for the guidance signal when denoising the tilted next-actions during TRAINING. Does not change the TD backup, which already evaluates the target Q at the sampled actions, nor rollout sampling, which keeps using the online Q.")
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
    parser.add_argument("--latent_action", action="store_true", default=False, help="Diffuse actions in an unbounded latent space and map them to the env only at the step boundary via the Gaussian-CDF squash g(a)=erf(a/sqrt(2))=2*Phi(a)-1. The replay buffer stores the latent a (critic/guidance stay in latent space, so no change-of-variables is needed); warmup draws latents whose squash reproduces the usual uniform[-1,1] actions; and the [-1,1] clips (sampler endpoint + network denoising x_recon_clip_radius) are disabled.")
    parser.add_argument("--batch_independent_guidance", action="store_true", default=False, help="If set, use jnp.sum instead of jnp.mean inside the guided predictor's q_mean_from_x, so the per-sample Q gradient is independent of batch size (fixes the 1/B attenuation).")
    parser.add_argument("--guidance_strength_multiplier", type=_parse_guidance_strength_multiplier, default=1.0, help="Multiplier applied to the guided-predictor Q scalar before jax.grad. Accepts a float for constant scaling or 'increasing' for the normalized α_t schedule from TFG (2409.15761). Composes with --batch_independent_guidance.")
    parser.add_argument("--policy_parameterization", type=str, default="E", choices=["E", "f"], help="Parameterization of the energy network scalar output. 'E' (default): network outputs E_theta; eps_pred = sqrt(1-alpha_bar_t)*grad E. 'f': network outputs f = sqrt(1-alpha_bar_t)*E_theta; eps_pred = grad f, energy_fn = f/sqrt(1-alpha_bar_t).")
    parser.add_argument("--policy_final_layer", type=str, default="default", choices=["default", "ff", "L2", "IP"], help="Final-layer head of the scalar policy network. 'default': replace DACERPolicyNet final layer with Linear(1). 'ff': keep full DACERPolicyNet (act_dim output) and tack on an extra learned Linear(1). 'L2': full backbone then E = -0.5*||v||^2 (no extra params). 'IP': full backbone then E = v·a (inner product with action, no extra params).")
    parser.add_argument("--guidance_gradient_space", type=str, default="xt", choices=["xt", "x0hat", "x0hatclipped"], help="Whether to take the Q gradient with respect to 'xt' or the predicted clean action 'x0hat' or its clipped version 'x0hatclipped'.")

    # ----- multi-action denoising + V-free advantage normalization ----------
    parser.add_argument("--num_denoised_actions", type=int, default=1, help="Number K of actions denoised per state in one sampler pass (formerly 'num_particles'). All K share the state and are iid draws. Rollout uses one (index 0, a uniform draw). The TD backup averages the clipped-double-Q over the K actions, and the diffusion policy regresses toward all K. K>=2 is required for --batch_advantage_normalization. Changes tensor shapes, so it is a 'hard' (non-vmap-packable) sweep axis in launch.py. Default 1.")
    parser.add_argument("--batch_advantage_normalization", action="store_true", default=False, help="V-free guidance normalization. At each MALA/denoising step, rescale Q by 1/sqrt(mean_s Var_K(Q)): the per-state sample variance (ddof=1) of Q over the K denoised Tweedie estimates, averaged over the batch, square-rooted, and stop-gradient'd. Requires --num_denoised_actions >= 2. Composes additively with --beta / other normalizations.")
    parser.add_argument("--q_loss_normalization", action="store_true", default=False, help="V-free guidance normalization. Divide the guidance Q by sqrt(EMA(Q TD loss)), where the EMA (rate --advantage_ema_tau) is tracked in-graph from the critic loss. Needs neither a V network nor multiple actions. Mutually exclusive with --advantage_normalization / --kl_budget(_per_dim) / --one_step_dist_shift_beta.")
    parser.add_argument("--ema_advantage_normalization", action="store_true", default=False, help="V-free guidance normalization ported from diffusion_policy_online_rl. Divide the guidance Q by a slow EMA of the batch std of the online --q_agg_sample-aggregated Q at the sampled next-actions: Q_norm = (Q - mu)/sigma with mu, sigma stop-gradient'd (mu cancels in the guidance gradient, so this is effectively a 1/sigma rescale). EMA rate --advantage_norm_ema_rate. Needs neither a V network nor multiple actions. Mutually exclusive with --advantage_normalization / --kl_budget(_per_dim) / --one_step_dist_shift_beta / --q_loss_normalization.")
    parser.add_argument("--advantage_norm_ema_rate", type=float, default=0.001, help="Per-step EMA rate r for the --ema_advantage_normalization running mean/std: x += r*(batch - x). Default 0.001 (the value hardcoded in diffusion_policy_online_rl).")
    parser.add_argument("--ema_within_advantage_normalization", action="store_true", default=False, help="V-free guidance normalization, KL-optimal variant. Divide the guidance Q by an EMA-tracked estimate of the contraharmonic mean E[sigma^2]/E[sigma] of the WITHIN-state sd sigma_Q(o) = sd_K(Q) at fixed state. Only within-state Q variation tilts the policy (adding any c(o) to Q leaves the tilt at o unchanged), so unlike --ema_advantage_normalization -- which EMAs the POOLED batch std and is therefore dominated by the across-state spread of Q -- this measures the quantity the tilt actually depends on. Dividing by sigma_Q(o) is what equalizes the per-state KL spend at beta^2/2; the KL cost of a wrong divisor is (beta^2/2)(sigma_Q/sigma_hat - 1)^2, asymmetric (over-tilting diverges, under-tilting saturates), and its population minimizer is the contraharmonic mean -- larger than the sqrt(E[sigma^2]) of --batch_advantage_normalization by exactly sqrt(1+CV^2). First moment debiased by c4(K-1). EMA rate --advantage_norm_ema_rate. Requires --num_denoised_actions >= 2. See notes/snr_schedule_shift.")
    parser.add_argument("--estimate_s_hat", action="store_true", default=False, help="Estimate --s_hat online instead of holding it fixed. EMAs the log of the per-state action sd over the K denoised actions (pooled over the act_dim coordinates, so the estimator carries d(K-1) degrees of freedom and K=2 already suffices), subtracts the exact additive bias (psi(nu/2) - log(nu/2))/2, and exponentiates -- i.e. the geometric mean, which is what the log-symmetric cosh cost of a wrong s_hat asks for. Takes over from --s_hat (and from any --hp_pack override of it) after the first update; --s_hat still sets the value used until then. Clipped to [0.02, 4.0] as a runaway guard. EMA rate --s_hat_ema_rate. Requires --num_denoised_actions >= 2.")
    parser.add_argument("--s_hat_ema_rate", type=float, default=0.001, help="Per-step EMA rate r for the --estimate_s_hat log-space accumulators: x += r*(batch - x), seeded with the first batch so no initial value is needed. Default 0.001.")

    # ----- MALA -------------------------------------------------------------
    parser.add_argument("--mala_steps", type=int, default=0, help="Number of MALA correction steps per diffusion step.")
    parser.add_argument("--mala_adapt_rate", type=float, default=0.05, help="Robbins-Monro adaptation rate for MALA log_eta_scale updates.")
    parser.add_argument("--denoising_predictor", type=str, default="DDPM_mean", choices=["Identity", "Identity_then_DDPM_mean", "DDPM_mean", "DDIM", "DDIM_unguided"], help="Predictor transition after each MALA correction level. Identity skips denoising; DDPM_mean uses the guided DDPM posterior mean; DDIM uses the guided deterministic DDIM update; DDIM_unguided uses the deterministic DDIM update with the *unguided* eps reused from the last MALA gradient (no extra network evals, no x0 clip, ignores --guidance_strength_multiplier; requires --mala_steps >= 1). Identity_then_DDPM_mean is Identity for every transition BETWEEN noise levels and DDPM_mean for the last one (level 0 -> clean), where the posterior mean is exactly the guided Tweedie estimate: the chain -- MALA targets, acceptance rates, schedule cost -- is bit-identical to Identity's, and only the action read off the cleanest level changes, from that level's (still noisy) sample to its clean prediction. Costs one extra guided predictor pass per sampler call, not T.")

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

    _needs_two_actions = [
        name for name, on in (
            ("--batch_advantage_normalization", args.batch_advantage_normalization),
            ("--ema_within_advantage_normalization", args.ema_within_advantage_normalization),
            ("--estimate_s_hat", args.estimate_s_hat),
        ) if on
    ]
    if _needs_two_actions and args.num_denoised_actions < 2:
        parser.error(
            f"{', '.join(_needs_two_actions)}: the per-state sample variance over the "
            "denoised actions is needed, so --num_denoised_actions >= 2 is required."
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
    # Both EMA modes write the same guidance-Q divisor slot, so they also exclude
    # each other; they differ only in which variance of Q they measure.
    if (args.ema_advantage_normalization or args.ema_within_advantage_normalization) and (
        args.advantage_normalization
        or args.kl_budget is not None
        or args.kl_budget_per_dim is not None
        or args.one_step_dist_shift_beta
        or args.q_loss_normalization
        or (args.ema_advantage_normalization and args.ema_within_advantage_normalization)
    ):
        parser.error(
            "--ema_advantage_normalization and --ema_within_advantage_normalization are "
            "mutually exclusive with each other and with --advantage_normalization, "
            "--kl_budget, --kl_budget_per_dim, --one_step_dist_shift_beta, and "
            "--q_loss_normalization"
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
        parser.error("--eval_every must be >= 0 (0 disables separate evaluation).")
    if args.eval_n_episodes <= 0 or args.eval_best_of_n_actions <= 0:
        parser.error("--eval_n_episodes and --eval_best_of_n_actions must be > 0.")
    if args.update_per_iteration <= 0:
        parser.error("--update_per_iteration must be > 0.")
    if args.critic_update_steps <= 0:
        parser.error("--critic_update_steps must be > 0.")
    if args.policy_update_steps <= 0:
        parser.error("--policy_update_steps must be > 0.")
    if args.distillation_buffer_size <= 0 or args.distillation_steps <= 0:
        parser.error("--distillation_buffer_size and --distillation_steps must be > 0.")
    for _delay_flag in ("delay_target_q_update", "delay_policy_update", "delay_target_policy_update"):
        if getattr(args, _delay_flag) <= 0:
            parser.error(f"--{_delay_flag} must be > 0.")

    # Fused denoising steps the env with the action denoised in the training
    # branch, which under a target policy would make rollouts target-policy
    # actions -- the one thing --use_target_policy_training must not change.
    if args.use_target_policy_training and args.fused_denoising:
        parser.error("--use_target_policy_training is incompatible with --fused_denoising.")
    if args.mala_steps <= 0:
        parser.error("--mala_steps must be > 0; the non-MALA sampling branches have been removed.")

    if args.beta_schedule_type == "adaptive":
        if args.noise_schedule_warmup < 0:
            parser.error("--noise_schedule_warmup is a number of updates and must be >= 0 "
                         f"(got {args.noise_schedule_warmup}).")
        if not (0.0 < args.noise_schedule_gamma <= 1.0):
            parser.error("--noise_schedule_gamma blends the new layout into the old, so it "
                         f"must be in (0, 1] (got {args.noise_schedule_gamma}).")
        if args.noise_schedule_log_snr_max <= args.noise_schedule_log_snr_min:
            parser.error(
                "--noise_schedule_log_snr_max must exceed --noise_schedule_log_snr_min "
                f"(got {args.noise_schedule_log_snr_max} <= {args.noise_schedule_log_snr_min})."
            )
        # The cost compares two levels' scores at the SAME x, which is only what
        # the sampler produces when the predictor moves nothing BETWEEN levels.
        # Identity_then_DDPM_mean qualifies: its one non-identity step is the
        # final read-out to the clean action, below every scored interval.
        if args.denoising_predictor not in ("Identity", "Identity_then_DDPM_mean"):
            parser.error(
                "--beta_schedule_type adaptive is the identity-predictor case of the "
                "score-optimal cost (it scores adjacent levels at the same sample), so it "
                "requires --denoising_predictor Identity or Identity_then_DDPM_mean "
                f"(got {args.denoising_predictor})."
            )
        # The cost is a divergence between the exact level-wise scores; the
        # Jacobian-free drift is an approximation to them.
        if args.guidance_gradient_space != "xt":
            parser.error(
                "--beta_schedule_type adaptive scores the exact level-wise drift, which "
                "only --guidance_gradient_space xt computes "
                f"(got {args.guidance_gradient_space})."
            )
        # The knots already fix each level's log-SNR, so an s_hat shift on top
        # would move the same degree of freedom twice, from two objectives.
        _s_hat_sources = [f"--s_hat {args.s_hat}"] if args.s_hat != 1.0 else []
        if args.estimate_s_hat:
            _s_hat_sources.append("--estimate_s_hat")
        if args.hp_pack_inline is not None and "s_hat" in json.loads(args.hp_pack_inline):
            _s_hat_sources.append("--hp_pack_inline s_hat")
        if _s_hat_sources:
            parser.error(
                f"--beta_schedule_type adaptive is incompatible with {', '.join(_s_hat_sources)}: "
                "its knots already fix where each level sits, so the log-SNR translation "
                "s_hat would apply on top would double-count."
            )
