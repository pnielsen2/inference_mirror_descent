# Efficient Online Reinforcement Learning for Diffusion Policies

This repository contains the current JAX/Haiku training code for our diffusion-policy RL experiments.

## Minimal setup

```bash
python --version  # should report Python 3.10.x or newer
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# CPU-only JAX
pip install jax==0.6.2 jaxlib==0.6.2

# Or, for CUDA 12 GPUs (matches ~/.venvs/general)
# pip install jax==0.6.2 jaxlib==0.6.2 jax-cuda12-plugin==0.6.2 jax-cuda12-pjrt==0.6.2

pip install -r requirements.txt
pip install -e .
```

## MuJoCo runtime notes for `*-v3` environments

The packed `HalfCheetah-v3` / `Ant-v3` / `Walker2d-v3` / `Humanoid-v3` experiments use Gymnasium's `mujoco-py` path with MuJoCo 2.1.

Assumptions:

- `MuJoCo 2.1` is installed at `$HOME/.mujoco/mujoco210`
- `mujoco-py` can find the NVIDIA and GLEW headers/libraries

Export the same runtime variables used by the working SLURM launcher:

```bash
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$HOME/.local/glew/glew-2.1.0/include:${CPATH:-}"
```

`CPATH` is only needed if `mujoco-py` has to build `cymj` on the current machine.

## Optional extras not included in `requirements.txt`

- `seaborn`, `tbparse`, and `tensorboardX` for optional analysis / plotting utilities that are not needed for the packed MGMD launch path

### HumanoidBench

HumanoidBench is optional and is not installed by `requirements.txt`, because its
package pins older Gymnasium and MuJoCo versions than this project. Install its
code into the already-prepared training venv without replacing that stack:

```bash
source ~/.venvs/general/bin/activate
pip install --no-deps "git+https://github.com/carlosferrazza/humanoid-bench.git"
python -c "import humanoid_bench; import gymnasium as gym; gym.make('h1hand-walk-v0', render_mode=None).close()"
```

The launcher uses `~/.venvs/general` for `humanoidbench/*` by default. Point it
at another fully provisioned venv with `HUMANOIDBENCH_VENV=/path/to/venv`.
HumanoidBench environments are exposed as namespaced ids such as
`humanoidbench/h1hand-walk-v0`. Use `scripts/launch.py --benchmark humanoidbench`
to expand the current official main-task suite into the environment sweep axis.

## Sanity checks after install

```bash
python -c "import relax.futex, relax.spinlock, relax.prctl; import scripts.launch, scripts.train_mujoco"
python scripts/train_mujoco.py --help >/dev/null
python scripts/launch.py --help >/dev/null
```

## Example: MGMD packed sweep launch command

The command below reflects the current `scripts/train_mujoco.py` and `scripts/launch.py` CLI. It packs the easy `eta` ablation and seeds up to `12` runs per GPU while varying environment, denoising predictor, and advantage normalization as hard axes.

```bash
python scripts/launch.py \
  --dry-run \
  --cmd "python scripts/train_mujoco.py \
    --alg mgmd \
    --num_vec_envs 5 \
    --mala_steps 2 \
    --beta_schedule_type cosine \
    --buffer_size 400000 \
    --x0_hat_clip_radius 3.0 \
    --mala_adapt_rate 0.2 \
    --update_per_iteration 4 \
    --lr_q 0.00015 \
    --policy_parameterization E \
    --policy_final_layer ff \
    --lr_policy 0.0003 \
    --T 0 \
    --q_agg_sample mean" \
  --seeds-per-config 2 \
  --ablate env Ant-v3 Humanoid-v3 HalfCheetah-v3 Walker2d-v3 Hopper-v3 Swimmer-v3 \
  --ablate denoising_predictor Identity DDIM \
  --ablate eta .03 .1 .3 1 3 10 \
  --ablate advantage_normalization True False \
  --max-runs-per-gpu 12 \
  --time 1-00:00 \
  --mem 64G
```

Remove `--dry-run` to actually submit the SLURM jobs.

`scripts/launch.py` will automatically:

- infer the `general` venv for `*-v3` commands
- inject `--parallel_runs K` for each packed job
- inline the per-slot overrides via `--hp_pack_inline '<json>'`
- auto-pick CPUs as `min(pack_size * num_vec_envs, 24)` unless you override `--cpus`

## Candidate distillation: reweighting, Boltzmann sampling, best of N

`--candidate_sampler ddim_clipped --policy_parameterization eps` draws unguided proposals
with one network forward per DDIM step. `--candidate_selection` controls improvement:

| Value | Policy target and clipped-double-Q TD backup | Training rollout |
|---|---|---|
| `reweight` | All N actions, weighted per state by `softmax(eta * Q)` | One draw from those weights |
| `boltzmann` | One categorical draw from the same weights | That draw |
| `best` | The maximum-Q action; eta is unused | That action |

Each method takes one policy optimizer step at `--distillation_steps 1` and
buffer size 1; reweighting evaluates all targets in one batch. With
`--ema_advantage_normalization`, Q in the weights is divided by the existing
running standard deviation. Remove that flag for literal unnormalized Q.
`Boltzmann/ess_mean` and `Boltzmann/pmax_mean` reveal whether the weights have
collapsed onto the maximum. The original `first` and `--boltzmann_reweight`
settings remain available.

`ddim_clipped` clips the clean-action prediction to `[-1,1]` at every level
and recomputes the implied epsilon. Its coefficient on the current noisy action
is `sigma_prev / sigma_t <= 1`, avoiding the growing intermediate actions that
final-only clipping cannot fix. The original `ddim` remains available.
`--candidate_sampler ddpm` is the stochastic reference: clipped x0 at every
level plus DDPM posterior noise, with no posterior noise on the final step.
These choices use the configured schedule and unit Gaussian initialization;
they isolate the reverse transition rather than bundle DPMD's other settings.
They also support `--eval_sampler` and `--td_next_action`.

Supplying NaN/Inf controls reproduces the MuJoCo exception seen in sweep 264.
The numerical regression reproduces an overflowing unclipped DDIM chain from
finite inputs and checks that clipped DDIM stays bounded at every level.
The vector environment now rejects any remaining NaN/Inf actions before
dispatch, reporting their env indices instead of passing them into MuJoCo.

`--action_noise 1` adds DPMD's Gaussian noise to training actions and
distillation/TD targets, after computing ranks/weights from clean proposals;
`0` disables it. Targets remain unclipped after injection, as in DPMD.
The noise controller is data independent, so its original Adam(0.007) curve is
tabulated once: initial std 0.5, target entropy -0.9 per action dimension,
asymptotic std about 0.0984. Its 250-update period at five env steps/update is
1250 env steps here, independent of this run's update/data ratio.
`Policy/action_noise_std` logs the actual std. Evaluation adds no such noise.

This grid uses N=32 throughout and evaluates every arm with the same best of 32
base DDIM proposals (`--eval_sampler ddim_clipped` avoids nested selection and
uses the same clipped transition for evaluation).
Compare `eval/episode_return_mean` / `eval_episode_returns.csv`; training
returns include exploration noise.
There are 48 runs in 12 GPU jobs: four environments × three methods, with
noise 0/1 × two seeds vmapped within each job. Selection is a hard axis because
the target-buffer shapes differ; noise, eta and learning rates are easy axes.
Start with the existing eta and policy learning rate, without extra sweeps.

H100 calibration with clipped DDIM (Humanoid-v3, four packed runs, 50 steps):
reweighting and best-of-32 each completed 20,001 training steps and two
ten-episode-per-run evaluations with finite losses/actions, taking 598 s and
563 s respectively. This extrapolates to about 8–9 hours of training per 1M
steps. The 22-hour request leaves headroom for logging, checkpoints, longer
episodes and machine load; queue time is separate. These are numerical
stability/throughput probes, not completed 1M-step learning runs.

```bash
IMD_RUN_ROOT=/n/netscratch/kdbrantley_lab/Lab/$USER/imd_clipped_$(date +%Y%m%d_%H%M%S)
python scripts/launch.py \
  --cmd "python scripts/train_mujoco.py \
    --alg mgmd --num_vec_envs 1 --update_per_iteration 1 \
    --candidate_sampler ddim_clipped --policy_parameterization eps \
    --num_denoised_actions 32 --beta_schedule_type cosine --diffusion_steps 50 \
    --lr_q 0.00015 --lr_policy 0.0003 --delay_policy_update 1 \
    --ema_advantage_normalization --lr_anneal --orthogonal_init \
    --fused_denoising --q_agg_sample mean --T 0 --eta 64 --s_hat 1 \
    --distillation_steps 1 --total_step 1000000 \
    --eval_every 10000 --eval_sampler ddim_clipped --eval_best_of_n_actions 32 \
    --save_diagnostic_snapshots --diagnostic_snapshot_dir $IMD_RUN_ROOT/snapshots \
    --diagnostic_snapshot_steps 10000 50000 100000 200000 300000 400000 500000 600000 700000 800000 825000 850000 875000 900000 925000 950000 975000 1000000" \
  --seeds-per-config 2 \
  --ablate buffer_size 250000 \
  --ablate env Walker2d-v3 Humanoid-v3 Ant-v3 HalfCheetah-v3 \
  --ablate candidate_selection reweight boltzmann best \
  --ablate action_noise 0 1 \
  --max-runs-per-gpu 10 --cpus 8 --time 22:00:00 --mem 64G \
  --log-dir "$IMD_RUN_ROOT" \
  --wandb-offline-base /n/holylabs/kdbrantley_lab/Lab/$USER/wandb \
  --wandb-sync-interval 60
```

Add `--dry-run --no-sweep-id` to inspect the packing without submission or a
wandb API lookup. The early diagnostic snapshots let a failed run's sampler
be investigated from saved parameters; `--diagnostic_snapshot_steps` alone
does not enable saving. To use stochastic DDPM, change both sampler flags to
`ddpm`.

## Sharing across the lab (multi-user setup)

The code is user-agnostic: nothing is hardcoded to a specific home or
netscratch path. Personal locations are derived from `$HOME` / `$USER`, and
the wandb namespace is resolved from env vars with a shared default.

### One-time setup per user

```bash
# 1. Independent clone + venv (see "Minimal setup" above); MuJoCo 2.1 at
#    $HOME/.mujoco/mujoco210 and GLEW at $HOME/.local/glew as documented.
# 2. wandb: log in, then point at the shared namespace.
wandb login
export WANDB_ENTITY=pnielsen2-harvard        # shared entity (invite-based for now)
export WANDB_PROJECT=diffusion_online_rl      # optional; this is the default
```

Add the two `export` lines to your `~/.bashrc`. To migrate to a proper wandb
**Team** later, everyone just changes `WANDB_ENTITY` to the team name — no code
change needed. The default in `relax/utils/fs.py` (`pnielsen2-harvard`) is used
only if `WANDB_ENTITY` is unset.

### Where things live

- **wandb runs** → the shared `$WANDB_ENTITY/$WANDB_PROJECT`. Every member's
  runs land in the same project, so `compute_topsis.py`, `plot_*` etc. see
  everyone's runs. Sweep IDs are drawn from that shared project, so two people
  never collide.
- **Offline wandb staging** → `/n/netscratch/kdbrantley_lab/Lab/$USER/wandb`
  (per user). The `Lab/` tree is group-readable for `kdbrantley_lab`, so you
  can read a colleague's runs in place — no copy needed. Note netscratch is
  scratch storage (subject to purge); wandb cloud is the durable source of
  truth. Override the base with `WANDB_OFFLINE_BASE` or `--wandb-offline-base`.

### Accessing each other's runs for plotting

- **From wandb (default):** `python scripts/compute_topsis.py --sweep-id N`
  queries the shared project and sees all members' runs.
- **From local offline dirs (no internet):** point the analysis scripts at the
  other user's netscratch base, e.g.
  `python scripts/build_sweep_local_metrics.py --sweep-id N --wandb-base /n/netscratch/kdbrantley_lab/Lab/<other-user>/wandb`.

## Acknowledgement
We developed this repo based on [Efficient Online Reinforcement Learning for Diffusion Policies] (https://github.com/mahaitongdae/diffusion_policy_online_rl), which was in turn based on [DACER](https://github.com/happy-yan/DACER-Diffusion-with-Online-RL.git). We thank the authors of both repos for providing a high-quality code base.
