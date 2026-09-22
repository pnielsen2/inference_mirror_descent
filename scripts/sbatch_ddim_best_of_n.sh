#!/bin/bash
# Unguided-DDIM best-of-N sweep over the action-selection snapshots. Task id
# splits into a manifest config (env x diffusion_steps x training eta) and one of
# STEP_SHARDS strided slices of that config's ten snapshot steps, so the tasks
# stay short: this partition's free GPUs are usually held by the backfill
# scheduler for a higher-priority reservation, and only a job that finishes
# before it starts gets to use one. Every step in a task shares the compiled
# samplers, so shard as coarsely as the queue allows.
#
#   sbatch --array=0-95 scripts/sbatch_ddim_best_of_n.sh                # 24 x 4
#   STEP_SHARDS=1 sbatch --array=0-23 -t 0-03:00 scripts/sbatch_ddim_best_of_n.sh
#   STEP_SHARDS=1 sbatch --array=0 scripts/sbatch_ddim_best_of_n.sh --values 1 4 --episodes 4
#
# Extra arguments are forwarded to the evaluator, which skips settings whose
# episodes are already on disk -- so a timed-out task resumes on resubmit.
#SBATCH -p kempner_h100
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -c 20
#SBATCH --mem=32G
#SBATCH -t 0-00:40
#SBATCH -o /n/home09/pnielsen/inference_mirror_descent/logs/slurm/%A_%a.out
#SBATCH -e /n/home09/pnielsen/inference_mirror_descent/logs/slurm/%A_%a.err
#SBATCH --job-name=ddimbestn
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
OUTPUT_DIR="$SNAPSHOT_ROOT/action_selection_eval_100ep_ddim_bestmin"

STEP_SHARDS=${STEP_SHARDS:-4}
TASK=${SLURM_ARRAY_TASK_ID:?set --array}
CONFIG=$((TASK / STEP_SHARDS))
SHARD=$((TASK % STEP_SHARDS))

echo "Job ${SLURM_ARRAY_JOB_ID:-none} task $TASK (config $CONFIG shard $SHARD/$STEP_SHARDS) on $(hostname)"
echo "Started at: $(date)"
nvidia-smi -L

python -m scripts.evaluate_snapshot_ddim_best_of_n \
  --manifest "$MANIFEST" \
  --config-index "$CONFIG" \
  --step-shard "$SHARD" --num-step-shards "$STEP_SHARDS" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

echo "Finished at: $(date)"
