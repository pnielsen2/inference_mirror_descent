# Efficient Online Reinforcement Learning for Diffusion Policies

This repository contains the current JAX/Haiku training code for our diffusion-policy RL experiments.

## Harvard RC copy-paste setup for the real `*-v3` example

The real packed-seed launch below uses Gymnasium's `mujoco-py` path, so on Harvard RC you need both the Python environment and a user-space install of `MuJoCo 2.1` plus the `GLEW` headers. The block below is the tested end-to-end setup sequence.

```bash
export PYTHON_BIN=/n/sw/Mambaforge-23.11.0-0/bin/python
mkdir -p "$HOME/.mujoco" "$HOME/.local/glew"

if [ ! -f "$HOME/.mujoco/mujoco210/bin/libmujoco210.so" ]; then
  wget -O /tmp/mujoco210-linux-x86_64.tar.gz https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
  tar -xzf /tmp/mujoco210-linux-x86_64.tar.gz -C "$HOME/.mujoco"
fi

if [ ! -f "$HOME/.local/glew/glew-2.1.0/include/GL/glew.h" ]; then
  wget -O /tmp/glew-2.1.0.tgz https://downloads.sourceforge.net/project/glew/glew/2.1.0/glew-2.1.0.tgz
  tar -xzf /tmp/glew-2.1.0.tgz -C "$HOME/.local/glew"
fi

test -f "$HOME/.mujoco/mujoco210/bin/libmujoco210.so"
test -f "$HOME/.local/glew/glew-2.1.0/include/GL/glew.h"

export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$HOME/.local/glew/glew-2.1.0/include:${CPATH:-}"

"$PYTHON_BIN" --version
"$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install "jax[cuda12]==0.6.2"
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` installs a venv-local `patchelf`, which `mujoco-py` needs if it has to build `cymj` on the compute node the first time it imports a `*-v3` environment. The explicit `LD_LIBRARY_PATH` / `CPATH` exports above mirror what `scripts/launch.py` later injects into the generated `sbatch` script.
The example launch below includes `--no-sweep-id` so it works from a fresh Harvard RC account without requiring a prior `wandb login` on the login node. If you want `launch.py` to auto-assign a sweep id and the background sync loop to upload runs to W&B, run `wandb login` first and remove `--no-sweep-id`.

## Optional extras not included in `requirements.txt`

- `brax==0.14.0` if you want `--backend mjx`

## Sanity checks after install

```bash
python -c "import relax.futex, relax.spinlock, relax.prctl; import scripts.launch, scripts.train_mujoco"
python scripts/train_mujoco.py --help >/dev/null
python scripts/launch.py --help >/dev/null
```

## Example: real `*-v3` packed-seed launch

The command below matches the real packed-seed test launch shape: four `*-v3` MuJoCo jobs (`HalfCheetah`, `Ant`, `Walker2d`, `Humanoid`), each packing seeds `0` and `1` together on one GPU.

```bash
python scripts/launch.py \
  --venv "$PWD/.venv" \
  --no-sweep-id \
  --job-name dpmd_H3 \
  --log-dir "$PWD/logs/slurm/packed_seed_real" \
  --wandb-offline-base "$PWD/wandb_offline" \
  --cmd "python scripts/train_mujoco.py \
    --alg dpmd \
    --env HalfCheetah-v3 \
    --dpmd_constant_weight \
    --tfg_eta 8.0 \
    --num_particles 1 \
    --mala_steps 2 \
    --q_critic_agg mean \
    --beta_schedule_type cosine \
    --beta_schedule_scale 1 \
    --dpmd_no_entropy_tuning \
    --buffer_size 400000 \
    --x0_hat_clip_radius 3.0 \
    --mala_adapt_rate 0.2 \
    --mala_per_level_eta \
    --q_td_huber_width 30.0 \
    --update_per_iteration 8 \
    --lr_q 0.00015 \
    --mala_guided_predictor \
    --ddim_predictor \
    --kl_budget 1024 \
    --one_step_dist_shift_eta" \
  --seeds 0 1 \
  --ablate env HalfCheetah-v3 Ant-v3 Walker2d-v3 Humanoid-v3 \
  --max-runs-per-gpu 2 \
  --cpus 10 \
  --time 1-06:00 \
  --mem 64G
```

Once that launch works, scale it up by adding `--ablate` or `--oat-ablate` axes. (Adding these additional ablate axes to manually pack more runs on a single GPU seems to be broken right now)

`scripts/launch.py` will automatically:

- use `--venv PATH` when you want an explicit environment such as `"$PWD/.venv"`; otherwise it infers a default cluster venv from the command's `--env` version
- prepend any CUDA runtime libraries bundled inside that venv to `PATH` / `LD_LIBRARY_PATH`
- verify that JAX actually sees a GPU on the allocated node before starting training
- inject `--parallel_seeds K` for each packed job
- inline the per-slot overrides via `--hp_pack_inline '<json>'`
- auto-pick CPUs as `min(pack_size * num_vec_envs, 24)` unless you override `--cpus`

If you do not want offline WandB staging, replace `--wandb-offline-base ...` with `--no-wandb-offline`.

## Acknowledgement
We developed this repo based on [Efficient Online Reinforcement Learning for Diffusion Policies] (https://github.com/mahaitongdae/diffusion_policy_online_rl), which was in turn based on [DACER](https://github.com/happy-yan/DACER-Diffusion-with-Online-RL.git). We thank the authors of both repos for providing a high-quality code base.
