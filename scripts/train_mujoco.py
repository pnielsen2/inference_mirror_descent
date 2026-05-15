import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import time

import jax, jax.numpy as jnp

from relax.algorithm.dpmd import DPMD
from relax.buffer import TreeBuffer
from relax.network.diffv2 import create_diffv2_net
from relax.env import create_vector_env
from relax.utils.experience import Experience
from relax.utils.fs import PROJECT_ROOT
from relax.utils.random_utils import seeding


def _derive_seeds(master: int):
    """Map a master int seed to the tuple (env_seed, env_action_seed,
    legacy_eval_env_seed, buffer_seed, init_network_seed, train_seed)."""
    rng, _ = seeding(int(master))
    return tuple(int(x) for x in rng.integers(0, 2**32 - 1, 6))


@dataclass
class SeedBundle:
    """All per-seed RNG state needed downstream.

    ``per_entry_env_seeds`` / ``per_entry_action_seeds`` are populated only
    when an hp_pack supplies per-vmap-entry master seeds; in that case
    create_vector_env reproduces the env/action RNGs that N standalone
    --seed S_i runs would have used. Otherwise they are None and the
    standalone env_seed / env_action_seed pair drives the VectorEnv.
    """
    env_seed: int
    env_action_seed: int
    per_entry_env_seeds: Optional[List[int]]
    per_entry_action_seeds: Optional[List[int]]
    buffer_seeds: List[int]      # length N_seeds
    init_keys: jax.Array         # [N_seeds] PRNG keys
    train_keys: jax.Array        # [N_seeds] PRNG keys
    per_entry_masters: Optional[List[int]]   # for logging


def derive_seed_bundle(master_seed: int, N_seeds: int,
                       hp_pack: Optional[dict]) -> SeedBundle:
    """Derive every RNG site (env / action / buffer / network init /
    training key) from ``master_seed``, plus optional per-vmap-entry
    masters from ``hp_pack["seed"]``. When per-entry masters are present,
    every seed site is derived from its entry's master so the pack matches
    standalone --seed S_i runs byte-for-byte.
    """
    env_seed, env_action_seed, _legacy, buffer_seed, init_network_seed, train_seed = _derive_seeds(master_seed)

    per_entry_masters = None
    if hp_pack is not None and "seed" in hp_pack:
        per_entry_masters = [int(s) for s in hp_pack["seed"]]
        if len(per_entry_masters) != N_seeds:
            raise ValueError(
                f"--hp_pack 'seed' has length {len(per_entry_masters)}; "
                f"expected {N_seeds} (= --parallel_seeds)"
            )

    if per_entry_masters is not None:
        derived = [_derive_seeds(m) for m in per_entry_masters]
        per_entry_env_seeds = [t[0] for t in derived]
        per_entry_action_seeds = [t[1] for t in derived]
        buffer_seeds = [t[3] for t in derived]
        init_keys = jnp.stack([jax.random.key(t[4]) for t in derived])
        train_keys = jnp.stack([jax.random.key(t[5]) for t in derived])
    else:
        per_entry_env_seeds = None
        per_entry_action_seeds = None
        # Match the previous per-buffer derivation: buffer 0 uses
        # buffer_seed, buffer s uses buffer_seed + s for s >= 1.
        buffer_seeds = [buffer_seed + i for i in range(N_seeds)]
        init_keys = jax.random.split(jax.random.key(init_network_seed), N_seeds)
        train_keys = jax.random.split(jax.random.key(train_seed), N_seeds)

    return SeedBundle(
        env_seed=env_seed,
        env_action_seed=env_action_seed,
        per_entry_env_seeds=per_entry_env_seeds,
        per_entry_action_seeds=per_entry_action_seeds,
        buffer_seeds=buffer_seeds,
        init_keys=init_keys,
        train_keys=train_keys,
        per_entry_masters=per_entry_masters,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="dpmd", choices=["dpmd"])
    parser.add_argument("--env", type=str, default="HalfCheetah-v3")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--num_vec_envs", type=int, default=5)
    parser.add_argument("--parallel_seeds", type=int, default=1, help="If > 1, train N independent DPMD seeds in parallel on a single device via jax.vmap. Env layout uses a single VectorEnv of size parallel_seeds * num_vec_envs. Current packed support covers the KL-budget/on-policy-EMA path and fixed-tfg_eta mode.")
    parser.add_argument("--hp_pack_inline", type=str, default=None, help="Inline JSON with per-seed hyperparameter overrides. Each key is an argparse attribute name of this script (e.g. 'tau', 'tfg_eta', 'advantage_ema_tau', 'guidance_strength_multiplier', 'kl_budget', 'shape_ema_tau', 'seed') mapped to a list of length parallel_seeds. Applied after vmap state construction; internally translated to Diffv2TrainState field names via _CLI_TO_FIELD.")
    parser.add_argument("--sweep_id", type=int, default=None, help="Launcher-assigned integer identifying this sweep. When set, every wandb run from this invocation is placed in wandb group 'sweep_<sweep_id>', and each per-vmap-slot run's config includes a 'config_tag' field built from sweep_id + the per-slot hyperparameters (excluding seed/env) so a single tag value filters wandb to all runs across envs/seeds that share this hp configuration.")
    parser.add_argument("--config_tag_keys", type=str, default=None, help="Comma-separated list of argparse attribute names whose values should be included in the per-slot config_tag. Typically set automatically by scripts/launch.py to the union of all --ablate hard+easy flags (minus env and seed). Values come from the hp_pack (per-slot) when the key is a pack key, else from this script's CLI args (shared across all vmap slots within the job).")
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)
    parser.add_argument("--start_step", type=int, default=int(3e4)) # other envs 3e4
    parser.add_argument("--total_step", type=int, default=int(1e6))
    parser.add_argument("--update_per_iteration", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_policy", type=float, default=None)
    parser.add_argument("--lr_q", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for the Q critic. Default 0.99.")
    parser.add_argument("--tau", type=float, default=0.005, help="Polyak averaging coefficient for target network updates. Default 0.005.")
    parser.add_argument("--delay_update", type=int, default=2, help="Update policy and target networks every delay_update steps. Default 2.")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num_particles", type=int, default=1, help="Backward-compatibility flag. simplify_walkthrough supports only single-particle behavior, so this must remain 1.")
    parser.add_argument("--cluster", default=False, action="store_true")
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--timing_log_every", type=int, default=0)

    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    parser.add_argument("--batch_size", type=int, default=256, help="Mini-batch size for training updates.")
    parser.add_argument("--beta_schedule_scale", type=float, default=0.8)
    parser.add_argument("--beta_schedule_type", type=str, default='linear', help="Noise schedule type. 'linear': linear beta schedule. 'cosine': cosine schedule (Nichol & Dhariwal). 'constant_kl': constant mutual-information-loss per step, spacing noise levels uniformly in log(1+SNR).")
    parser.add_argument("--snr_max", type=float, default=124.0, help="Maximum SNR (at cleanest noise level). Controls alpha_bar_0 = snr_max/(1+snr_max). Default 124.0 matches the cosine schedule with s=0.008 offset at T=20. All schedule types use this to set the same clean endpoint, so you can switch between cosine/constant_kl/linear while keeping the noise range comparable.")
    parser.add_argument("--reward_scale", type=float, default=0.2, help="Scale factor applied to rewards before Q/value learning. Default 0.2 matches original DPMD. Set to 1.0 for clarity when using inference-time guidance (adjust tfg_eta accordingly).")
    parser.add_argument("--tfg_eta", type=float, default=0.0, help="Guidance strength lambda for dpmd training-free Q-guidance. If 0, no Q-guidance is applied.")
    parser.add_argument("--critic_normalization", type=str, default="none", choices=["none", "ema"], help="Normalization mode for Q guidance. 'none': use raw Q. 'ema': train V(s) to predict E[Q], normalize (Q-V) by sqrt(EMA[A^2]).")
    parser.add_argument("--kl_budget", type=float, default=None, help="Total KL divergence budget δ for guidance. Per-dimension budget is δ / act_dim. Sets η = sqrt(2δ), enables V network and on-policy advantage EMA. Replaces --critic_normalization ema --tfg_eta. Default None (disabled).")
    parser.add_argument("--kl_budget_per_dim", type=float, default=None, help="Per-dimension KL divergence budget δ_d for guidance. Total budget δ = δ_d * act_dim. Sets η = sqrt(2δ), enables V network and on-policy advantage EMA. Replaces --critic_normalization ema --tfg_eta. Default None (disabled).")
    parser.add_argument("--one_step_dist_shift_eta", action="store_true", default=False, help="Adaptive η from second-order expansion using one-step Monte Carlo covariance estimate. No D_ψ head; estimates c from consecutive (A_t, A_{t+1}) pairs. Requires --kl_budget or --kl_budget_per_dim (defaults to --kl_budget_per_dim=5.33 if neither set).")
    parser.add_argument("--advantage_ema_tau", type=float, default=0.0005, help="Per-step EMA rate for advantage second/third moments.")
    parser.add_argument("--shape_ema_tau", type=float, default=0.0001, help="Per-step EMA rate for dimensionless shape s2.")
    parser.add_argument("--initial_advantage_second_moment_ema", type=float, default=1.0, help="Initial value for the advantage second moment EMA E[A^2].")
    parser.add_argument("--initial_dist_shift_shape_ema", type=float, default=-1.0, help="Initial value for the dimensionless distribution-shift shape EMA s2 = (2γc + κ₃) / v^(3/2).")
    parser.add_argument("--x0_hat_clip_radius", type=float, default=float("inf"), help="Clipping radius r for Tweedie clean-action estimates x0_hat used inside guidance/Q evaluation. x0_hat is clipped to [-r, r] before being passed into Q / model-based objectives. Default inf (no clip); in non-latent mode the network-side denoising clip is separately hardcoded to 1.0 to match normalized action bounds.")
    parser.add_argument("--q_critic_agg", type=str, default="min", choices=["min", "mean"], help="Aggregation for the Q signal used in tilting and reweighting. TD-target aggregation is controlled separately by --q_bootstrap_agg.")
    parser.add_argument("--q_bootstrap_agg", type=str, default="min", choices=["min", "mean"], help="Aggregation mode for Q TD targets. 'min' (default): both Qs bootstrap from min(Q1_target, Q2_target) (clipped double Q-learning). 'mean': both Qs bootstrap from mean of all target networks.")
    parser.add_argument("--dpmd_constant_weight", action="store_true", default=False, help="If set for dpmd, disable Q-based reweighting in the diffusion score-matching loss and use constant weights.")
    parser.add_argument("--num_q_networks", type=int, default=2, help="Number of Q critic networks to train (default 2, i.e. twin Q).")
    parser.add_argument("--dpmd_no_entropy_tuning", action="store_true", default=False, help="If set for dpmd, disable action noise and alpha/entropy tuning.")
    parser.add_argument("--mala_steps", type=int, default=0, help="Number of MALA correction steps per diffusion step.")
    parser.add_argument("--mala_per_level_eta", action="store_true", default=False, help="If set, learn a separate MALA eta-scale for each diffusion noise level. Default behavior (flag off) uses a single shared eta-scale across all noise levels.")
    parser.add_argument("--mala_adapt_rate", type=float, default=0.05, help="Robbins-Monro adaptation rate for MALA log_eta_scale updates.")
    parser.add_argument("--mala_guided_predictor", action="store_true", default=False, help="If set, apply Q-guidance (TFG-style eps guidance) in the DDPM predictor step after each MALA correction step.")
    parser.add_argument("--mala_no_predictor", action="store_true", default=False, help="If set, remove predictor transitions entirely during MALA sampling so each lower-noise level initializes directly from the previous level's post-MALA state.")
    parser.add_argument("--ddim_predictor", action="store_true", default=False, help="If set, use deterministic DDIM-style predictor (no noise) instead of stochastic DDPM. Recommended for MALA sampling since the stochastic noise is redundant with MALA corrections.")
    parser.add_argument("--q_td_huber_width", type=float, default=float("inf"), help="Huber width (delta) for critic TD error in DPMD. Default inf recovers the current MSE TD loss. Effective width is scaled by reward_scale internally.")
    parser.add_argument("--batch_independent_guidance", action="store_true", default=False, help="If set, use jnp.sum instead of jnp.mean inside the guided predictor's q_mean_from_x, so the per-sample Q gradient is independent of batch size (fixes the 1/B attenuation).")
    parser.add_argument("--guidance_strength_multiplier", type=float, default=1.0, help="Constant multiplier applied to the guided-predictor Q scalar before jax.grad. Composes with --batch_independent_guidance.")
    parser.add_argument("--energy_multiplier", type=float, default=1.0, help="Multiplier for base energy/score during sampling. Values < 1 temper (flatten) the base distribution, increasing entropy. Guidance signal is NOT scaled. Default 1.0 (no tempering).")

    args = parser.parse_args()

    # simplify_walkthrough only retains the branches actually exercised by the
    # three protected launch commands; the flags below are kept in argparse so
    # those commands still parse, but their non-default settings are no-ops.
    if (args.num_particles != 1
            or not args.dpmd_no_entropy_tuning
            or not args.dpmd_constant_weight
            or not args.mala_per_level_eta):
        parser.error(
            "simplify_walkthrough requires: --num_particles 1, "
            "--dpmd_no_entropy_tuning, --dpmd_constant_weight, "
            "--mala_per_level_eta (the legacy alternatives have been removed)."
        )

    # --one_step_dist_shift_eta implies a KL budget (default 5.33 per dim)
    if args.one_step_dist_shift_eta and args.kl_budget is None and args.kl_budget_per_dim is None:
        args.kl_budget_per_dim = 5.33

    if args.debug:
        from jax import config
        config.update("jax_disable_jit", True)

    N_seeds = int(args.parallel_seeds)
    if N_seeds <= 0:
        raise ValueError("--parallel_seeds must be >= 1.")
    if args.num_vec_envs <= 0:
        raise ValueError("--num_vec_envs must be > 0 in simplify_walkthrough (vectorized/vmapped path only).")

    # Load the inline hp_pack JSON, if any. Used below for per-entry master
    # seeds AND the later _replace overrides on the vmap state.
    _hp_loaded = json.loads(args.hp_pack_inline) if args.hp_pack_inline is not None else None

    seeds = derive_seed_bundle(args.seed, N_seeds, _hp_loaded)

    total_envs = args.num_vec_envs * N_seeds

    env, obs_dim, act_dim = create_vector_env(
        args.env,
        total_envs,
        seeds.env_seed,
        seeds.env_action_seed,
        per_entry_env_seeds=seeds.per_entry_env_seeds,
        per_entry_action_seeds=seeds.per_entry_action_seeds,
    )

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

    include_next_action = False
    buffers_list = [
        TreeBuffer.from_experience(
            obs_dim, act_dim, size=args.buffer_size,
            seed=seeds.buffer_seeds[i],
            include_next_action=include_next_action,
        )
        for i in range(N_seeds)
    ]
    
    print(f"Algorithm: {args.alg}")

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
            beta_schedule_scale=args.beta_schedule_scale,
            beta_schedule_type=args.beta_schedule_type,
            mala_steps=args.mala_steps,
            num_q_networks=args.num_q_networks,
            x_recon_clip_radius=1.0,
            snr_max=args.snr_max,
        )

    _pairs = [_make_diffv2(k) for k in seeds.init_keys]
    agent = _pairs[0][0]
    dpmd_params_list = [p for (_a, p) in _pairs]
    params = dpmd_params_list[0]

    algorithm = DPMD(
        agent,
        params,
        gamma=args.gamma,
        lr=args.lr,
        lr_policy=args.lr_policy,
        lr_q=args.lr_q,
        tau=args.tau,
        delay_update=args.delay_update,
        reward_scale=args.reward_scale,
        q_critic_agg=args.q_critic_agg,
        q_bootstrap_agg=args.q_bootstrap_agg,
        tfg_eta=args.tfg_eta,
        x0_hat_clip_radius=args.x0_hat_clip_radius,
        mala_adapt_rate=args.mala_adapt_rate,
        mala_guided_predictor=args.mala_guided_predictor,
        mala_no_predictor=args.mala_no_predictor,
        ddim_predictor=args.ddim_predictor,
        q_td_huber_width=args.q_td_huber_width,
        batch_independent_guidance=args.batch_independent_guidance,
        guidance_strength_multiplier=args.guidance_strength_multiplier,
        energy_multiplier=args.energy_multiplier,
        critic_normalization=args.critic_normalization,
        kl_budget=args.kl_budget,
        one_step_dist_shift_eta=args.one_step_dist_shift_eta,
        advantage_ema_tau=args.advantage_ema_tau,
        shape_ema_tau=args.shape_ema_tau,
        initial_advantage_second_moment_ema=args.initial_advantage_second_moment_ema,
        initial_dist_shift_shape_ema=args.initial_dist_shift_shape_ema,
    )

    algorithm.state = algorithm.make_vmapped_state(dpmd_params_list)
    if _hp_loaded is not None:
        _hp = _hp_loaded
        _CLI_TO_FIELD = {
            "tau": "polyak_tau",
            "advantage_ema_tau": "adv_ema_tau",
            "guidance_strength_multiplier": "guidance_mult",
            "kl_budget": "kl_budget_val",
            "initial_advantage_second_moment_ema": "advantage_second_moment_ema",
            "initial_dist_shift_shape_ema": "dist_shift_shape_ema",
            "tfg_eta": "tfg_eta",
        }
        _allowed = {"lr_q", "lr_policy", "gamma", "tau", "advantage_ema_tau",
                    "guidance_strength_multiplier", "shape_ema_tau", "tfg_eta", "kl_budget",
                    "initial_advantage_second_moment_ema", "initial_dist_shift_shape_ema",
                    "reward_scale", "x0_hat_clip_radius", "mala_adapt_rate",
                    "q_td_huber_width",
                    "seed"}
        _overrides = {}
        for k, v in _hp.items():
            if k not in _allowed:
                raise ValueError(f"--hp_pack key '{k}' is not a per-seed vmappable hp. Allowed: {sorted(_allowed)}")
            if k == "seed":
                continue
            arr = jnp.asarray(v, dtype=jnp.float32)
            if arr.shape != (N_seeds,):
                raise ValueError(f"--hp_pack '{k}' has shape {arr.shape}; expected ({N_seeds},)")
            _overrides[_CLI_TO_FIELD.get(k, k)] = arr
        if _overrides:
            algorithm.state = algorithm.state._replace(**_overrides)
            if "kl_budget_val" in _overrides:
                kl_budget_v = jnp.asarray(algorithm.state.kl_budget_val, dtype=jnp.float32)
                algorithm.state = algorithm.state._replace(
                    tfg_eta=jnp.sqrt(jnp.maximum(jnp.float32(0.0), jnp.float32(2.0) * kl_budget_v))
                )
            print(f"[hp_pack] applied per-seed overrides: {list(_overrides.keys())}")
        if seeds.per_entry_masters is not None:
            print(f"[hp_pack] applied per-entry master seeds "
                  f"(buffers + init networks + train keys): {seeds.per_entry_masters}")

    if args.cluster:
        PROJECT_ROOT = Path('/n/netscratch/nali_lab_seas/Lab/haitongma/sdac_logs')
    
    exp_dir = PROJECT_ROOT / "logs" / args.env / (args.alg + '_' + time.strftime("%Y-%m-%d_%H-%M-%S") + f'_s{args.seed}_{args.suffix}')

    # Collect all CLI arguments into a dict for logging and configuration.
    # Allow the algorithm to override or augment these with its own effective
    # hyperparameters (e.g., internally clamped / derived values).
    args_dict = dict(vars(args))
    if hasattr(algorithm, "get_effective_hparams"):
        args_dict.update(algorithm.get_effective_hparams())

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
        hp_pack_dict=_hp_loaded,
        sweep_id=args.sweep_id,
        config_tag_keys=args.config_tag_keys,
    )
    trainer.setup(Experience.create_example(obs_dim, act_dim, trainer.batch_size, include_next_action=include_next_action))
    trainer.run(seeds.train_keys)
