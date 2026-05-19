import json
import time

from relax.algorithm.dpmd import DPMD
from relax.algorithm.dpmd_types import DPMDConfig
from relax.env import create_vector_env
from relax.utils.experience import Experience
from relax.utils.fs import PROJECT_ROOT
from relax.utils.seeding import derive_seed_bundle

from relax.cli.train_args import build_parser, validate_args
from relax.cli.train_setup import resolve_kl_budget, build_per_seed_state


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)

    if args.debug:
        from jax import config
        config.update("jax_disable_jit", True)

    # Load the inline hp_pack JSON, if any. Used below for per-entry master
    # seeds AND the later _replace overrides on the vmap state.
    _hp_loaded = json.loads(args.hp_pack_inline) if args.hp_pack_inline is not None else None

    seeds = derive_seed_bundle(args.seed, args.parallel_runs, _hp_loaded)

    env, obs_dim, act_dim = create_vector_env(
        args.env,
        args.num_vec_envs * args.parallel_runs,
        seeds.env_seed,
        seeds.env_action_seed,
        per_entry_env_seeds=seeds.per_entry_env_seeds,
        per_entry_action_seeds=seeds.per_entry_action_seeds,
    )

    # Apply --kl_budget / --kl_budget_per_dim promotion to tfg_eta + V-net.
    resolve_kl_budget(args, act_dim)

    print(f"Algorithm: {args.alg}")
    model, dpmd_params_list, buffers_list = build_per_seed_state(args, seeds, obs_dim, act_dim)
    params = dpmd_params_list[0]

    cfg = DPMDConfig.from_args(args)
    algorithm = DPMD(model, params, cfg, obs_dim=obs_dim, hidden_dim=args.hidden_dim)

    algorithm.state = algorithm.make_vmapped_state(dpmd_params_list)
    if _hp_loaded is not None:
        from relax.algorithm import hp_pack
        algorithm.state = hp_pack.apply(algorithm.state, _hp_loaded, args.parallel_runs)
        if seeds.per_entry_masters is not None:
            print(f"[hp_pack] applied per-entry master seeds "
                  f"(buffers + init networks + train keys): {seeds.per_entry_masters}")

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
        parallel_runs=args.parallel_runs,
        per_run_envs=args.num_vec_envs,
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
