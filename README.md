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
