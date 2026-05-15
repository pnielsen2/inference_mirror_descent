#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -c 24
#SBATCH --mem=64G
#SBATCH -t 2-00:00
#SBATCH -o /n/home09/pnielsen/inference_mirror_descent/logs/slurm/20260515_173427/%j.out
#SBATCH -e /n/home09/pnielsen/inference_mirror_descent/logs/slurm/20260515_173427/%j.err
#SBATCH --job-name=dpmd_HC3_2

set -euo pipefail
# Venv selected by launch.py based on the gymnasium env version in the
# --env flag (see VENV_BY_ENV_VERSION in scripts/launch.py).
export PATH="/n/home09/pnielsen/.venvs/general/bin:$PATH"
export VIRTUAL_ENV="/n/home09/pnielsen/.venvs/general"
# mujoco210 runtime + NVIDIA/libOpenGL/libEGL paths for cymj GPU builder
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/lib64:${LD_LIBRARY_PATH:-}"
# Only needed if cymj has to rebuild on the compute node (normally cached).
export CPATH="$HOME/.local/glew/glew-2.1.0/include:${CPATH:-}"
cd /n/home09/pnielsen/inference_mirror_descent

echo "Job ID: $SLURM_JOB_ID"
echo "Venv: $VIRTUAL_ENV"
echo "Running: python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v3 --num_vec_envs 5 --dpmd_constant_weight --tfg_eta 8.0 --num_particles 1 --mala_steps 2 --q_critic_agg mean --beta_schedule_type cosine --beta_schedule_scale 1 --dpmd_no_entropy_tuning --buffer_size 400000 --x0_hat_clip_radius 3.0 --mala_adapt_rate 0.2 --mala_per_level_eta --q_td_huber_width 30.0 --update_per_iteration 4 --lr_q 0.00015 --lr_policy 0.0003 --mala_guided_predictor --ddim_predictor --kl_budget 1024 --tau 0.005 --advantage_ema_tau 0.001 --shape_ema_tau 0.0002 --initial_advantage_second_moment_ema 1.0 --gamma 0.99 --sweep_id 58 --config_tag_keys kl_budget --seed 0 --parallel_seeds 40 --hp_pack_inline '{"kl_budget":[16.0,16.0,16.0,16.0,16.0,16.0,16.0,16.0,64.0,64.0,64.0,64.0,64.0,64.0,64.0,64.0,256.0,256.0,256.0,256.0,256.0,256.0,256.0,256.0,1024.0,1024.0,1024.0,1024.0,1024.0,1024.0,1024.0,1024.0,4096.0,4096.0,4096.0,4096.0,4096.0,4096.0,4096.0,4096.0],"seed":[0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7]}'"
echo "Started at: $(date)"

export WANDB_MODE=offline
export WANDB_DIR="/n/netscratch/kdbrantley_lab/Lab/pnielsen/wandb/sweep_58/job_${SLURM_JOB_ID}"
mkdir -p "$WANDB_DIR/wandb"
echo "WANDB_MODE=$WANDB_MODE"
echo "WANDB_DIR=$WANDB_DIR"

# wandb sync --sync-all has a quirk: passing an explicit PATH makes it
# treat PATH as one run dir (which ours isn't -- ours is a parent of
# offline-run-* subdirs), giving "Nothing to sync". Only the cwd-relative
# form ("cd <parent> && wandb sync --sync-all") correctly walks offline-run-*
# subdirs. So we cd into WANDB_DIR (whose `./wandb` child holds all this
# job's offline-run dirs) before each sync call.
_wandb_sync_loop() {
  while true; do
    sleep 60
    (cd "$WANDB_DIR" && nice -n 19 wandb sync --sync-all --include-synced) >> "$WANDB_DIR/sync.log" 2>&1 || true
  done
}
_wandb_sync_loop &
_WANDB_SYNC_PID=$!

_wandb_cleanup() {
  kill $_WANDB_SYNC_PID 2>/dev/null || true
  echo "Final wandb sync at $(date)"
  (cd "$WANDB_DIR" && nice -n 19 wandb sync --sync-all --include-synced) >> "$WANDB_DIR/sync.log" 2>&1 || true
}
trap _wandb_cleanup EXIT INT TERM

python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v3 --num_vec_envs 5 --dpmd_constant_weight --tfg_eta 8.0 --num_particles 1 --mala_steps 2 --q_critic_agg mean --beta_schedule_type cosine --beta_schedule_scale 1 --dpmd_no_entropy_tuning --buffer_size 400000 --x0_hat_clip_radius 3.0 --mala_adapt_rate 0.2 --mala_per_level_eta --q_td_huber_width 30.0 --update_per_iteration 4 --lr_q 0.00015 --lr_policy 0.0003 --mala_guided_predictor --ddim_predictor --kl_budget 1024 --tau 0.005 --advantage_ema_tau 0.001 --shape_ema_tau 0.0002 --initial_advantage_second_moment_ema 1.0 --gamma 0.99 --sweep_id 58 --config_tag_keys kl_budget --seed 0 --parallel_seeds 40 --hp_pack_inline '{"kl_budget":[16.0,16.0,16.0,16.0,16.0,16.0,16.0,16.0,64.0,64.0,64.0,64.0,64.0,64.0,64.0,64.0,256.0,256.0,256.0,256.0,256.0,256.0,256.0,256.0,1024.0,1024.0,1024.0,1024.0,1024.0,1024.0,1024.0,1024.0,4096.0,4096.0,4096.0,4096.0,4096.0,4096.0,4096.0,4096.0],"seed":[0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7]}'

echo "Finished at: $(date)"
