"""Periodic best-of-N evaluation episodes, alongside the training rollout curve.

``episode_return/<env>`` measures the *behaviour* policy: one action sample per
state, executed in the training envs, every episode counted as it happens. This
module adds the other curve -- every ``--eval_every`` env steps, run
``--eval_n_episodes`` complete episodes per run in their own envs, selecting each
action as the highest-Q of ``--eval_best_of_n_actions`` i.i.d. denoised
candidates, and record nothing to the replay buffer.

Three properties are deliberate:

* **Training is untouched.** Evaluation draws its actions from
  :meth:`MGMD.get_eval_action_vmap`, which writes nothing back to the train
  state (in particular no MALA step-size adaptation), and it has its own envs and
  its own PRNG stream. A run's training trajectory is therefore bit-identical
  with evaluation on or off; only wall-clock changes.

* **All episodes run in parallel.** One env per episode (``num_runs *
  n_episodes`` in total, laid out run-major like the trainer's own env), stepped
  in lockstep for one episode each, so an evaluation costs ~one episode length
  of *sequential* denoising passes instead of ``n_episodes`` of them -- ~10x less
  wall clock at the same number of denoised actions, since a single pass at this
  batch size is latency- rather than FLOP-bound. Envs that finish early are
  masked out of the accounting (the vector env auto-resets and keeps stepping
  them, which is why the loop cannot simply stop at the first ``done``).

* **Every evaluation is paired.** The envs are (re)created per evaluation from
  ``--eval_seed`` alone -- never from ``--seed`` -- and the action keys are
  derived from ``(eval_seed, eval_index, iteration)``. So every config in a sweep,
  every seed, and every eval point along a curve starts from the *same* initial
  states and consumes the *same* denoising noise; differences between two eval
  points are differences in the policy, not in the draw. Recreating the envs is
  also what keeps the accounting honest across eval points: envs that terminated
  early would otherwise be left mid-episode, and the next evaluation would score
  a partial trajectory.
"""
import os
import time
from pathlib import Path

import gymnasium
import jax
import numpy as np

from relax.env import create_vector_env
from relax.trainer.wandb_logging import WandbMultiSeedLogger


class Evaluator:
    """Owns the eval envs, schedule, action call, CSV mirror and wandb keys.

    ``maybe_run`` is a no-op when ``every == 0``, so the trainer's call sites need
    no guard. ``to_env_action`` is the trainer's own action map, so the
    ``--latent_action`` squash is applied exactly as in training.
    """

    def __init__(self, *, algorithm, logger: WandbMultiSeedLogger, log_path: Path,
                 env_name: str, num_runs: int, to_env_action, every: int,
                 n_episodes: int, best_of_n: int, seed: int):
        self.algorithm = algorithm
        self.logger = logger
        self.log_path = log_path
        self.env_name = env_name
        self.num_runs = int(num_runs)
        self.to_env_action = to_env_action
        self.every = int(every)
        self.n_episodes = int(n_episodes)
        self.best_of_n = int(best_of_n)
        self.seed = int(seed)
        self._csv_path = log_path / "eval_episode_returns.csv"
        self._keys = jax.random.split(jax.random.key(self.seed), self.num_runs)
        self._next_step = None
        self._n_evals = 0

    # ------------------------------------------------------------------
    def maybe_run(self, step: int) -> None:
        """Evaluate if ``step`` has reached the next multiple of ``every``."""
        if self.every <= 0:
            return
        step = int(step)
        if self._next_step is None:
            # Anchor on the first step seen (post-warmup), so the first
            # evaluation lands on a multiple of --eval_every rather than on
            # whatever --start_step happens to be.
            self._next_step = (step // self.every + 1) * self.every
        if step < self._next_step:
            return
        self._run(step)
        self._next_step = (step // self.every + 1) * self.every

    # ------------------------------------------------------------------
    def _run(self, step: int) -> None:
        t0 = time.monotonic()
        E = self.n_episodes
        env, max_steps = self._make_env()
        # Per (run, episode): return and length of the FIRST episode each env
        # runs; ``active`` drops an env for good once that episode is done.
        returns = np.zeros((self.num_runs, E), np.float64)
        lengths = np.zeros((self.num_runs, E), np.int64)
        active = np.ones((self.num_runs, E), bool)
        keys = jax.vmap(lambda k: jax.random.fold_in(k, self._n_evals))(self._keys)
        try:
            for i in range(max_steps):
                if not active.any():
                    break
                obs = env.get_current_obs().reshape(self.num_runs, E, -1)
                action = self.algorithm.get_eval_action_vmap(
                    jax.vmap(lambda k: jax.random.fold_in(k, i))(keys), obs, self.best_of_n)
                _, reward, terminated, truncated, _ = env.step(
                    self.to_env_action(action.reshape(self.num_runs * E, -1)))
                returns += reward.reshape(self.num_runs, E) * active
                lengths += active
                active &= ~(terminated | truncated).reshape(self.num_runs, E)
        finally:
            env.close()
        self._n_evals += 1
        self._log(step, returns, lengths, time.monotonic() - t0)

    def _make_env(self):
        """A fresh eval VectorEnv: same env stack as training, fixed seeds.

        ``num_workers`` is capped at the job's CPU allotment (the training
        workers are futex-blocked while we evaluate, so peak process concurrency
        is unchanged) and must divide the env count. Seeds are tiled over runs,
        so run ``r``'s episode ``j`` starts from the same state as every other
        run's episode ``j``.
        """
        n = self.num_runs * self.n_episodes
        budget = min(n, len(os.sched_getaffinity(0)))
        workers = max(w for w in range(1, budget + 1) if n % w == 0)
        seeds = np.random.default_rng(self.seed).integers(0, 2**32 - 1, self.n_episodes).tolist()
        env, _, _ = create_vector_env(self.env_name, n, self.seed, num_workers=workers,
                                      seeds_override=seeds * self.num_runs)
        # One episode per env, so the registered time limit bounds the loop.
        # Read it from the registry rather than env.spec: the vector env exposes
        # the *unwrapped* spec, whose max_episode_steps is None even though the
        # workers' gymnasium.make() applies the TimeLimit. The fallback only
        # matters for an env registered without a limit, where it becomes the
        # length every "episode" is truncated to.
        return env, int(gymnasium.spec(self.env_name).max_episode_steps or 1000)

    def _log(self, step: int, returns: np.ndarray, lengths: np.ndarray, elapsed: float) -> None:
        if not self._csv_path.exists():
            self.log_path.mkdir(parents=True, exist_ok=True)
            with open(self._csv_path, "w") as f:
                f.write("seed_index,step,eval_best_of_n_actions,episode_index,"
                        "episode_return,episode_length\n")
        with open(self._csv_path, "a", buffering=1) as f:
            for s in range(self.num_runs):
                for j, (ret, length) in enumerate(zip(returns[s], lengths[s])):
                    f.write(f"{s},{step},{self.best_of_n},{j},{float(ret)},{int(length)}\n")
                for stat, value in (("mean", returns[s].mean()), ("std", returns[s].std()),
                                    ("min", returns[s].min()), ("max", returns[s].max())):
                    self.logger.add_scalar_per_run(
                        s, f"eval/episode_return_{stat}", float(value), step=step)
                self.logger.add_scalar_per_run(
                    s, "eval/episode_length_mean", float(lengths[s].mean()), step=step)
        print(f"[eval] step {step} best_of_{self.best_of_n}: return/run "
              f"{np.round(returns.mean(axis=1), 1)} ({elapsed:.0f}s)", flush=True)
