# Efficient Online Reinforcement Learning for Diffusion Policies

This repository contains the current JAX/Haiku training code for our diffusion-policy RL experiments.

## Minimal setup

The commands below match the currently working `~/.venvs/general` stack.

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

- `pandas`, `seaborn`, `tbparse`, `tensorboard`, `tensorboardX`, and `pyyaml` for analysis / plotting utilities that are not needed for the packed MGMD launch path

## Sanity checks after install

```bash
python -c "import relax.futex, relax.spinlock, relax.prctl; import scripts.launch, scripts.train_mujoco"
python scripts/train_mujoco.py --help >/dev/null
python scripts/launch.py --help >/dev/null
```

## Example: `packed_q` launch command

The command below is a validated `scripts/launch.py --dry-run` example for the packed MGMD / Q-guided sweep path. It packs seeds and easy one-at-a-time ablations up to `8` runs per GPU.

```bash
python scripts/launch.py \
  --dry-run \
  --no-sweep-id \
  --job-name packed_q \
  --wandb-offline-base "$PWD/wandb_offline" \
  --cmd "python scripts/train_mujoco.py \
    --alg mgmd \
    --env HalfCheetah-v3 \
    --suffix packed_q \
    --num_vec_envs 5 \
    --mgmd_constant_weight \
    --tfg_eta 8.0 \
    --num_particles 1 \
    --mala_steps 2 \
    --q_critic_agg mean \
    --beta_schedule_type cosine \
    --beta_schedule_scale 1 \
    --mgmd_no_entropy_tuning \
    --buffer_size 400000 \
    --x0_hat_clip_radius 3.0 \
    --mala_adapt_rate 0.2 \
    --mala_per_level_eta \
    --q_td_huber_width 30.0 \
    --update_per_iteration 8 \
    --lr_q 0.00015 \
    --lr_policy 0.0003 \
    --mala_guided_predictor \
    --ddim_predictor \
    --kl_budget 1024 \
    --one_step_dist_shift_eta \
    --polyak_tau 0.005 \
    --advantage_ema_tau 0.0005 \
    --shape_ema_tau 0.0001 \
    --initial_advantage_second_moment_ema 1.0 \
    --gamma 0.99 \
    --batch_independent_guidance \
    --guidance_strength_multiplier 0.2" \
  --seeds 0 1 \
  --ablate env HalfCheetah-v3 Ant-v3 Walker2d-v3 Humanoid-v3 \
  --oat-ablate lr_policy 0.00015 0.0006 \
  --oat-ablate lr_q 0.000075 0.0003 \
  --oat-ablate shape_ema_tau 0.00005 0.0002 \
  --oat-ablate advantage_ema_tau 0.00025 0.001 \
  --oat-ablate kl_budget 512 2048 \
  --oat-ablate polyak_tau 0.0025 0.01 \
  --oat-ablate gamma 0.998 0.999 \
  --oat-ablate initial_advantage_second_moment_ema 10 \
  --max-runs-per-gpu 8 \
  --time 1-06:00 \
  --mem 64G
```

Remove `--dry-run` to actually submit the SLURM jobs.

`scripts/launch.py` will automatically:

- infer the `general` venv for `*-v3` commands
- inject `--parallel_seeds K` for each packed job
- inline the per-slot overrides via `--hp_pack_inline '<json>'`
- auto-pick CPUs as `min(pack_size * num_vec_envs, 24)` unless you override `--cpus`

If you do not want offline WandB staging, replace `--wandb-offline-base ...` with `--no-wandb-offline`.

## Acknowledgement
We developed this repo based on [Efficient Online Reinforcement Learning for Diffusion Policies] (https://github.com/mahaitongdae/diffusion_policy_online_rl), which was in turn based on [DACER](https://github.com/happy-yan/DACER-Diffusion-with-Online-RL.git). We thank the authors of both repos for providing a high-quality code base.
