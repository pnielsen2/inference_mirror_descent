#!/bin/bash
#SBATCH -J dpmdmbpc_lr_grid
#SBATCH -p seas_gpu
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 0-6:00
#SBATCH --array=0-15

# Optional: activate your environment here
# source ~/.bashrc
# conda activate your_env

if [ -z "${SWEEP_ID}" ]; then
  echo "ERROR: SWEEP_ID is not set. Before submitting, run:"
  echo "  export SWEEP_ID=your_entity/diffusion_online_rl/<sweep_id>"
  exit 1
fi

wandb agent "${SWEEP_ID}"
