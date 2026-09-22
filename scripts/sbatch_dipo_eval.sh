#!/bin/bash
# DIPO's evaluation protocol (deterministic DDPM-mean chain, one action per
# state, no best-of-N) on the action-selection snapshots. One array task per
# shard of the 24 manifest configs; each covers all ten snapshot steps of its
# configs, reusing one compiled sampler. There is a single setting per snapshot,
# so the whole sweep is ~1 GPU-hour.
#
#   sbatch --array=0-5 scripts/sbatch_dipo_eval.sh
#   sbatch --array=0 scripts/sbatch_dipo_eval.sh --steps 1000000 --episodes 10
#SBATCH -p kempner_h100
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -c 20
#SBATCH --mem=32G
#SBATCH -t 0-00:40
#SBATCH -o /n/home09/pnielsen/inference_mirror_descent/logs/slurm/%A_%a.out
#SBATCH -e /n/home09/pnielsen/inference_mirror_descent/logs/slurm/%A_%a.err
#SBATCH --job-name=dipoeval
set -euo pipefail

export PATH="$HOME/.venvs/general/bin:$PATH"
export VIRTUAL_ENV="$HOME/.venvs/general"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$HOME/.local/glew/glew-2.1.0/include:${CPATH:-}"
PROJECT_DIR=/n/home09/pnielsen/inference_mirror_descent
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

SNAPSHOT_ROOT=/n/netscratch/kdbrantley_lab/Lab/pnielsen/diagnostic_snapshots
MANIFEST="$SNAPSHOT_ROOT/action_selection_eval_100ep_eta32_eta64_1m/manifest.txt"
OUTPUT_DIR="$SNAPSHOT_ROOT/action_selection_eval_100ep_dipo_eval"

CONFIG_SHARDS=${CONFIG_SHARDS:-6}
echo "Job ${SLURM_ARRAY_JOB_ID:-none} task ${SLURM_ARRAY_TASK_ID:?set --array} of $CONFIG_SHARDS on $(hostname)"
echo "Started at: $(date)"
nvidia-smi -L

python -m scripts.evaluate_snapshot_dipo_eval \
  --manifest "$MANIFEST" \
  --config-shard "$SLURM_ARRAY_TASK_ID" --num-config-shards "$CONFIG_SHARDS" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

echo "Finished at: $(date)"
