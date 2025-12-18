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
    parser.add_argument("--supervised_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr_policy", type=float, default=None)
    parser.add_argument("--lr_dyn", type=float, default=None)
    parser.add_argument("--lr_reward", type=float, default=None)
    parser.add_argument("--lr_value", type=float, default=None)
    parser.add_argument("--lr_schedule_end", type=float, default=3e-5)
    parser.add_argument("--alpha_lr", type=float, default=7e-3)
    parser.add_argument("--delay_alpha_update", type=float, default=250)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num_particles", type=int, default=32)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--cluster", default=False, action="store_true")
    parser.add_argument("--debug", action='store_true', default=False)

    parser.add_argument("--buffer_size", type=int, default=int(1e6))
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Mini-batch size for training updates.")
    parser.add_argument("--val_batch_size", type=int, default=None,
                        help="Mini-batch size for validation updates. If None, defaults to batch_size.")
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
    parser.add_argument("--single_q_network", action="store_true", default=False,
                        help="If set, train a single Q network instead of twin Q networks. The same Q is used for both Q1 and Q2.")
    parser.add_argument("--dpmd_use_reward_critic", action="store_true", default=False,
                        help="If set for dpmd, replace the Q critic with a 1-step reward network trained from the replay buffer and use it (scaled by 1/(1-gamma)) for tilting/guidance.")
    parser.add_argument("--dpmd_pure_bc_training", action="store_true", default=False,
                        help="If set for dpmd, train the diffusion policy purely by behavior cloning from replay actions (no critic-based tilting at training time), while still using the critic for inference-time guidance.")
    parser.add_argument("--dpmd_off_policy_td", action="store_true", default=False,
                        help="If set for dpmd, use off-policy (buffer) actions for the critic TD target instead of on-policy (sampled from current policy) actions.")
    parser.add_argument("--dpmd_no_entropy_tuning", action="store_true", default=False,
                        help="If set for dpmd, disable action noise and alpha/entropy tuning (makes DPMD more similar to dpmd_mb_pc).")
    parser.add_argument("--dpmd_long_lr_schedule", action="store_true", default=False,
                        help="If set, anneal the diffusion policy LR over the full training horizon instead of the default 50k-step schedule (applies to dpmd and dpmd_mb).")                    
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
            "R+gamma*V signal for guidance, with tfg_lambda as the entropic temperature."
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

    # Hypergradient-based tuning currently relies on a held-out validation
    # split. Require validation to be enabled when hypergrad is requested.
    if args.use_hypergrad and not args.use_validation:
        raise ValueError("--use_hypergrad requires --use_validation to be enabled, since hypergradients are computed from validation loss.")

    # Configure policy LR schedule for diffusion-policy algorithms (dpmd, dpmd_mb).
    if args.dpmd_long_lr_schedule:
        # Anneal over the full training horizon
        dpmd_lr_schedule_steps = int(args.total_step)
        dpmd_lr_schedule_begin = 0
    else:
        # Preserve original behavior (short 50k-step schedule)
        dpmd_lr_schedule_steps = int(5e4)
        dpmd_lr_schedule_begin = int(2.5e4)

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

    # Use SARSA-style buffer (with next_action) when dpmd_off_policy_td is enabled
    include_next_action = getattr(args, 'dpmd_off_policy_td', False)
    buffer = TreeBuffer.from_experience(obs_dim, act_dim, size=args.buffer_size, seed=buffer_seed, include_next_action=include_next_action)

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

        val_env, _val_obs_dim, _val_act_dim = create_vector_env(
            args.env,
            val_envs,
            env_seed + 1,
            env_action_seed + 1,
            mode="futex",
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
            single_q_network=args.single_q_network,
        )
        algorithm = DPMD(
            agent,
            params,
            lr=args.lr,
            alpha_lr=args.alpha_lr,
            delay_alpha_update=args.delay_alpha_update,
            lr_schedule_end=args.lr_schedule_end,
            use_reweighting=not args.dpmd_constant_weight,
            use_reward_critic=args.dpmd_use_reward_critic,
            pure_bc_training=args.dpmd_pure_bc_training,
            off_policy_td=args.dpmd_off_policy_td,
            no_entropy_tuning=args.dpmd_no_entropy_tuning,
            q_critic_agg=args.q_critic_agg,
            fix_q_norm_bug=args.fix_q_norm_bug,
            tfg_lambda=args.tfg_lambda,
            tfg_lambda_schedule=args.tfg_lambda_schedule,
            tfg_recur_steps=args.dpmd_recurrence_steps,
            particle_selection_lambda=args.particle_selection_lambda,
            supervised_steps=args.supervised_steps,
            single_q_network=args.single_q_network,
            lr_schedule_steps=dpmd_lr_schedule_steps,
            lr_schedule_begin=dpmd_lr_schedule_begin,
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
            energy_param=args.energy_param,
        )
        algorithm = DPMDMB(
            agent,
            params,
            lr=args.lr,
            alpha_lr=args.alpha_lr,
            delay_alpha_update=args.delay_alpha_update,
            lr_schedule_end=args.lr_schedule_end,
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
            reward_scale=0.2,
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
            tfg_lambda=args.tfg_lambda,
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
        update_log_n_step=1 if args.debug else 1000,
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
    )

    trainer.setup(Experience.create_example(obs_dim, act_dim, trainer.batch_size, include_next_action=include_next_action))
    log_git_details(log_file=os.path.join(exp_dir, 'dacer.diff'))
    
    # Save the arguments to a YAML file
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(args_dict, yaml_file)
    trainer.run(train_key)
