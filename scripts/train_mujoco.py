import argparse
import os.path
from pathlib import Path
import time
from functools import partial
import yaml

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
from relax.utils.log_diff import log_git_details

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alg", type=str, default="sdac")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--num_vec_envs", type=int, default=5)
    parser.add_argument("--hidden_num", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=256)
    parser.add_argument("--start_step", type=int, default=int(3e4)) # other envs 3e4
    parser.add_argument("--total_step", type=int, default=int(1e6))
    parser.add_argument("--update_per_iteration", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_schedule_end", type=float, default=3e-5)
    parser.add_argument("--alpha_lr", type=float, default=7e-3)
    parser.add_argument("--delay_alpha_update", type=float, default=250)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num_particles", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--cluster", default=False, action="store_true")
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--beta_schedule_scale", type=float, default=0.8)
    parser.add_argument("--beta_schedule_type", type=str, default='linear')
    parser.add_argument("--tfg_lambda", type=float, default=0.0,
                        help="Guidance strength lambda for dpmd training-free Q-guidance. If 0, no Q-guidance is applied.")
    parser.add_argument("--tfg_lambda_schedule", type=str, default="constant",
                        help="Schedule type for dpmd guidance lambda as a function of noise level t (e.g., 'constant', 'linear').")
    parser.add_argument("--fix_q_norm_bug", action="store_true", default=False,
                        help="If set, use the corrected normalization (q_min - running_mean) / (running_std + eps) instead of the original buggy form.")
    parser.add_argument("--q_critic_agg", type=str, default="min",
                        help="Aggregation for twin Qs when used as a signal (tilting, reweighting): 'min', 'mean', or 'max'. TD targets always use 'min'.")
    parser.add_argument("--particle_selection_lambda", type=float, default=float("inf"),
                        help="Temperature for selecting an action among multiple particles using exp(particle_selection_lambda * Q(a)). Default inf reproduces argmax over Q.")
    parser.add_argument("--dpmd_recurrence_steps", type=int, default=0,
                        help="Number of recurrence-style TFG inner steps per diffusion level for dpmd (0 disables recurrence).")
    parser.add_argument("--dpmd_constant_weight", action="store_true", default=False,
                        help="If set for dpmd, disable Q-based reweighting in the diffusion score-matching loss and use constant weights.")
    parser.add_argument("--dpmd_bc_noisy_q_guided", action="store_true", default=False,
                        help="For dpmd_bc: train Q on noisy forward-diffused actions and use guided sampling that evaluates Q on an intermediate noisy diffusion state.")
    parser.add_argument("--dpmd_bc_tfg_recurrence", action="store_true", default=False,
                        help="For dpmd_bc: apply a recurrence-style training-free guidance step at inference time using Q as f(x_0)=exp(Q(x_0)).")
    parser.add_argument("--dpmd_bc_recurrence_steps", type=int, default=3,
                        help="For dpmd_bc: number of recurrence steps per noise level when using tfg_recurrence.")
    parser.add_argument("--energy_param", action="store_true", default=False,
                        help="Parameterize the diffusion model as the gradient of an energy function.")
    parser.add_argument("--mala_steps", type=int, default=0,
                        help="Number of MALA correction steps per diffusion step. If > 0, automatically enables energy_param.")
    parser.add_argument("--model_q_mc_samples", type=int, default=8)

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

    args = parser.parse_args()

    # Automatically enable energy parameterization if MALA steps are requested
    if args.mala_steps > 0:
        args.energy_param = True

    # For DPMD, disallow using both MALA and recurrence at the same time.
    if args.alg == 'dpmd' and args.mala_steps > 0 and args.dpmd_recurrence_steps > 0:
        raise ValueError("For dpmd, --mala_steps and --dpmd_recurrence_steps cannot both be > 0.")

    if args.debug:
        from jax import config
        config.update("jax_disable_jit", True)

    master_seed = args.seed
    master_rng, _ = seeding(master_seed)
    env_seed, env_action_seed, eval_env_seed, buffer_seed, init_network_seed, train_seed = map(
        int, master_rng.integers(0, 2**32 - 1, 6)
    )
    init_network_key = jax.random.key(init_network_seed)
    train_key = jax.random.key(train_seed)
    del init_network_seed, train_seed

    if args.num_vec_envs > 0:
        env, obs_dim, act_dim = create_vector_env(args.env, args.num_vec_envs, env_seed, env_action_seed, mode="futex")
    else:
        env, obs_dim, act_dim = create_env(args.env, env_seed, env_action_seed)
    eval_env = None

    hidden_sizes = [args.hidden_dim] * args.hidden_num
    diffusion_hidden_sizes = [args.diffusion_hidden_dim] * args.hidden_num

    buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=int(1e6), seed=buffer_seed)

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
        )
        algorithm = SDAC(agent, params, lr=args.lr, alpha_lr=args.alpha_lr, delay_alpha_update=args.delay_alpha_update, lr_schedule_end=args.lr_schedule_end)
    
    elif args.alg == 'dpmd':
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
        )
        algorithm = DPMD(
            agent,
            params,
            lr=args.lr,
            alpha_lr=args.alpha_lr,
            delay_alpha_update=args.delay_alpha_update,
            lr_schedule_end=args.lr_schedule_end,
            use_reweighting=not args.dpmd_constant_weight,
            q_critic_agg=args.q_critic_agg,
            fix_q_norm_bug=args.fix_q_norm_bug,
            tfg_lambda=args.tfg_lambda,
            tfg_lambda_schedule=args.tfg_lambda_schedule,
            tfg_recur_steps=args.dpmd_recurrence_steps,
            particle_selection_lambda=args.particle_selection_lambda,
        )

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
        )
        algorithm = DPMDBC(
            agent,
            params,
            lr=args.lr,
            alpha_lr=args.alpha_lr,
            delay_alpha_update=args.delay_alpha_update,
            lr_schedule_end=args.lr_schedule_end,
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
        )
        algorithm = DPMDMB(
            agent,
            params,
            lr=args.lr,
            alpha_lr=args.alpha_lr,
            delay_alpha_update=args.delay_alpha_update,
            lr_schedule_end=args.lr_schedule_end,
            num_mc_samples=args.model_q_mc_samples,
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
            gamma=0.99,
            lr_policy=args.lr,
            lr_dyn=args.lr,
            lr_reward=args.lr,
            lr_value=args.lr,
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

    # PCMD, DPMD, DPMD_BC, and DPMD_MB do not expose a standalone value interface
    # used by save_q_structure; avoid saving value structure for them to prevent
    # setup crashes.
    save_value = args.alg not in ("pcmd", "dpmd", "dpmd_bc", "dpmd_mb")

    # Collect all CLI arguments into a dict for logging and configuration.
    args_dict = vars(args)

    trainer = OffPolicyTrainer(
        env=env,
        algorithm=algorithm,
        buffer=buffer,
        start_step=args.start_step,
        total_step=args.total_step,
        sample_per_iteration=1,
        update_per_iteration=args.update_per_iteration,
        evaluate_env=eval_env,
        save_policy_every=int(args.total_step / 20),
        save_value=save_value,
        log_path=exp_dir,
        update_log_n_step=1 if args.debug else 1000,
        warmup_with="random",
        hparams=args_dict,
    )

    trainer.setup(Experience.create_example(obs_dim, act_dim, trainer.batch_size))
    log_git_details(log_file=os.path.join(exp_dir, 'dacer.diff'))
    
    # Save the arguments to a YAML file
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file)
    trainer.run(train_key)
