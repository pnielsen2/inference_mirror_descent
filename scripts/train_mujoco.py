import json
from pathlib import Path
import time

import jax, jax.numpy as jnp

from relax.algorithm.dpmd import DPMD, DPMDConfig
from relax.buffer import TreeBuffer
from relax.network.diffv2 import create_diffv2_net
from relax.env import create_vector_env
from relax.utils.experience import Experience
from relax.utils.fs import PROJECT_ROOT
from relax.utils.seeding import derive_seed_bundle

from _train_args import build_parser, validate_args


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    if args.debug:
        from jax import config
        config.update("jax_disable_jit", True)

    N_seeds = int(args.parallel_seeds)

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

    buffers_list = [
        TreeBuffer.from_experience(
            obs_dim, act_dim, size=args.buffer_size,
            seed=seeds.buffer_seeds[i],
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

    # Resolve the lr -> lr_{q,policy} fallback once, then pack every DPMD
    # hyperparameter into a single frozen ``DPMDConfig``.
    lr_policy = args.lr if args.lr_policy is None else args.lr_policy
    lr_q = args.lr if args.lr_q is None else args.lr_q
    cfg = DPMDConfig(
        gamma=args.gamma,
        tau=args.tau,
        lr_policy=float(lr_policy),
        lr_q=float(lr_q),
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
        advantage_ema_tau=args.advantage_ema_tau,
        shape_ema_tau=args.shape_ema_tau,
        initial_advantage_second_moment_ema=args.initial_advantage_second_moment_ema,
        initial_dist_shift_shape_ema=args.initial_dist_shift_shape_ema,
        kl_budget=args.kl_budget,
        one_step_dist_shift_eta=args.one_step_dist_shift_eta,
    )
    algorithm = DPMD(agent, params, cfg)

    algorithm.state = algorithm.make_vmapped_state(dpmd_params_list)
    if _hp_loaded is not None:
        from relax.algorithm import hp_pack
        algorithm.state = hp_pack.apply(algorithm.state, _hp_loaded, N_seeds)
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
    trainer.setup(Experience.create_example(obs_dim, act_dim, trainer.batch_size))
    trainer.run(seeds.train_keys)
