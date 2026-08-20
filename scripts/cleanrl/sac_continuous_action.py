# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import csv
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Ensure cleanrl_utils is importable from the scripts/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    csv_output: str = None
    """path to write episode CSV (global_step, episodic_return, episodic_length)"""
    eval_every: int = 0
    """run evaluation episodes every this many env steps (0 disables)"""
    eval_episodes: int = 10
    """number of evaluation episodes per evaluation point"""
    eval_best_of_n: List[int] = field(default_factory=lambda: [32])
    """candidate actions sampled per state during evaluation; the candidate with the highest aggregated Q is executed (1 = plain stochastic policy). One eval curve is produced per value, all at the same checkpoints"""
    eval_q_agg: str = "min"
    """how to aggregate the twin critics when scoring eval candidates: 'min' (SAC's own pessimistic value, as used in its actor loss) or 'mean'"""
    eval_csv_output: str = None
    """path to write the eval CSV (env_step, train_step, eval_best_of_n_actions, episode_index, episode_return, episode_length)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def _best_of_n_actions_batched(actor, qf1, qf2, obs_list, n_list, device, q_agg):
    """One best-of-N action per (obs, N) pair, in a single actor+critic call.

    ``obs_list[i]`` is scored with ``n_list[i]`` iid policy samples; the rows for
    all pairs are concatenated so an eval that sweeps N costs one pair of GPU
    calls per env step instead of one pair per N. Returns an array of actions,
    row ``i`` for pair ``i``.
    """
    with torch.no_grad():
        obs_rows = torch.cat(
            [
                torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(1, -1).expand(n, -1)
                for obs, n in zip(obs_list, n_list)
            ]
        )
        candidates, _, _ = actor.get_action(obs_rows)
        q1 = qf1(obs_rows, candidates).view(-1)
        q2 = qf2(obs_rows, candidates).view(-1)
        q = torch.min(q1, q2) if q_agg == "min" else 0.5 * (q1 + q2)
        picks, offset = [], 0
        for n in n_list:
            picks.append(offset + torch.argmax(q[offset : offset + n]))
            offset += n
        # Single device->host transfer for the whole group.
        return candidates[torch.stack(picks)].cpu().numpy()


def run_eval_episodes(actor, qf1, qf2, eval_envs, device, n_episodes, n_values, q_agg, episode_seeds=None):
    """Best-of-N eval episodes for every N in ``n_values`` at one checkpoint.

    ``eval_envs[i]`` is a dedicated env for ``n_values[i]``. For each episode
    index all envs are reset to the same seed and then stepped in lockstep, so
    the N curves are a paired comparison (identical start states, one shared
    policy-noise stream) rather than independent samples. Returns
    ``{n: (returns, lengths)}``.

    Touches neither the replay buffer nor the training envs. Callers snapshot and
    restore the torch RNG state around this so the training run's random stream
    -- and therefore its training curve -- is unaffected by evaluating.
    """
    k = len(n_values)
    out = {n: ([], []) for n in n_values}
    for ep_idx in range(n_episodes):
        seed = None if episode_seeds is None else int(episode_seeds[ep_idx])
        obs = [env.reset(seed=seed)[0] for env in eval_envs]
        ep_return = [0.0] * k
        ep_length = [0] * k
        active = [True] * k
        while any(active):
            idxs = [i for i in range(k) if active[i]]
            actions = _best_of_n_actions_batched(
                actor, qf1, qf2, [obs[i] for i in idxs], [n_values[i] for i in idxs], device, q_agg
            )
            for row, i in enumerate(idxs):
                next_obs, reward, terminated, truncated, _ = eval_envs[i].step(actions[row])
                obs[i] = next_obs
                ep_return[i] += float(reward)
                ep_length[i] += 1
                if bool(terminated) or bool(truncated):
                    active[i] = False
        for i, n in enumerate(n_values):
            out[n][0].append(ep_return[i])
            out[n][1].append(ep_length[i])
    return out


if __name__ == "__main__":

    args = tyro.cli(Args)
    if args.eval_q_agg not in ("min", "mean"):
        raise ValueError(f"--eval-q-agg must be 'min' or 'mean', got {args.eval_q_agg!r}")
    eval_n_values = sorted({int(n) for n in args.eval_best_of_n})
    if any(n < 1 for n in eval_n_values):
        raise ValueError(f"--eval-best-of-n values must be >= 1, got {args.eval_best_of_n}")
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # CSV episode logging
    csv_file = None
    csv_writer = None
    if args.csv_output:
        csv_path = Path(args.csv_output)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["global_step", "episodic_return", "episodic_length"])

    # Best-of-N evaluation. The eval envs are separate instances, explicitly
    # seeded per eval episode, so evaluating consumes nothing the training loop
    # would have drawn: combined with the torch RNG snapshot below, the training
    # curve is identical to the same run with --eval-every 0.
    eval_envs = []
    eval_csv_file = None
    eval_csv_writer = None
    if args.eval_every:
        # One env per N so the N sweep can be stepped in lockstep from a common
        # start state; each is reset with an explicit per-episode seed below.
        eval_envs = [gym.make(args.env_id) for _ in eval_n_values]
        if args.eval_csv_output:
            eval_csv_path = Path(args.eval_csv_output)
            eval_csv_path.parent.mkdir(parents=True, exist_ok=True)
            eval_csv_file = open(eval_csv_path, "w", newline="")
            eval_csv_writer = csv.writer(eval_csv_file)
            eval_csv_writer.writerow([
                "env_step", "train_step", "eval_best_of_n_actions",
                "episode_index", "episode_return", "episode_length",
            ])

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    # gymnasium >= 1.0 SyncVectorEnv uses next-step autoreset: the step right
    # after an episode ends ignores the action and returns (reset_obs, 0.0,
    # False, False). That is not a real transition and must not be stored.
    autoreset = np.zeros(envs.num_envs, dtype=bool)
    total_episode_steps = 0
    global_step = 0
    while total_episode_steps < args.total_timesteps:
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    ep_len = int(info["episode"]["l"])
                    total_episode_steps += ep_len
                    print(f"global_step={total_episode_steps}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], total_episode_steps)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], total_episode_steps)
                    if csv_writer:
                        csv_writer.writerow([total_episode_steps, float(info["episode"]["r"]), ep_len])
                        csv_file.flush()
                    break
        elif "episode" in infos:
            mask = infos.get("_episode", np.zeros(args.num_envs, dtype=bool))
            for i in range(args.num_envs):
                if mask[i]:
                    ep_len = int(infos["episode"]["l"][i])
                    total_episode_steps += ep_len
                    print(f"global_step={total_episode_steps}, episodic_return={infos['episode']['r'][i]}")
                    writer.add_scalar("charts/episodic_return", infos["episode"]["r"][i], total_episode_steps)
                    writer.add_scalar("charts/episodic_length", infos["episode"]["l"][i], total_episode_steps)
                    if csv_writer:
                        csv_writer.writerow([total_episode_steps, float(infos["episode"]["r"][i]), ep_len])
                        csv_file.flush()
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        if "final_observation" in infos:
            for idx, trunc in enumerate(truncations):
                if trunc:
                    real_next_obs[idx] = infos["final_observation"][idx]
        if not autoreset.any():
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        elif envs.num_envs > 1:
            raise NotImplementedError(
                "per-env autoreset masking is not supported for num_envs > 1"
            )
        autoreset = np.logical_or(terminations, truncations)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        global_step += 1

        if args.eval_every and global_step % args.eval_every == 0:
            # Snapshot/restore so the eval action samples do not advance the
            # training stream (torch.distributions has no per-call generator).
            torch_rng_state = torch.get_rng_state()
            # Same start states and same actor-noise stream for every N at this
            # checkpoint, so the N curves differ only through the argmax.
            eval_seed = args.seed + 100_000 + global_step
            episode_seeds = [eval_seed + 1_000_003 * ep for ep in range(args.eval_episodes)]
            eval_start = time.time()
            torch.manual_seed(eval_seed)
            eval_results = run_eval_episodes(
                actor, qf1, qf2, eval_envs, device,
                args.eval_episodes, eval_n_values, args.eval_q_agg,
                episode_seeds=episode_seeds,
            )
            torch.set_rng_state(torch_rng_state)
            for n_candidates in eval_n_values:
                eval_returns, eval_lengths = eval_results[n_candidates]
                mean_return = float(np.mean(eval_returns))
                print(
                    f"eval env_step={global_step} best_of_{n_candidates} "
                    f"({args.eval_q_agg}) episodes={args.eval_episodes} "
                    f"return_mean={mean_return:.1f} return_std={float(np.std(eval_returns)):.1f} "
                    f"length_mean={float(np.mean(eval_lengths)):.0f}",
                    flush=True,
                )
                writer.add_scalar(f"eval/n{n_candidates}/episode_return_mean", mean_return, global_step)
                writer.add_scalar(f"eval/n{n_candidates}/episode_return_std", float(np.std(eval_returns)), global_step)
                writer.add_scalar(f"eval/n{n_candidates}/episode_length_mean", float(np.mean(eval_lengths)), global_step)
                if eval_csv_writer:
                    for ep_idx, (ep_ret, ep_len) in enumerate(zip(eval_returns, eval_lengths)):
                        eval_csv_writer.writerow([
                            global_step, total_episode_steps, n_candidates,
                            ep_idx, ep_ret, ep_len,
                        ])
                    eval_csv_file.flush()
            print(f"eval env_step={global_step} all_N took={time.time() - eval_start:.0f}s", flush=True)

    envs.close()
    for eval_env in eval_envs:
        eval_env.close()
    writer.close()
    if csv_file:
        csv_file.close()
    if eval_csv_file:
        eval_csv_file.close()
