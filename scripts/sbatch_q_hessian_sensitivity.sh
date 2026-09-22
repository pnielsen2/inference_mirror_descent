#!/bin/bash
#SBATCH -p kempner_h100
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -t 0-02:00
#SBATCH -o /n/home09/pnielsen/inference_mirror_descent/logs/slurm/%j.out
#SBATCH -e /n/home09/pnielsen/inference_mirror_descent/logs/slurm/%j.err
#SBATCH --job-name=qhess
set -euo pipefail

export PATH="$HOME/.venvs/general/bin:$PATH"
export VIRTUAL_ENV="$HOME/.venvs/general"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$HOME/.local/glew/glew-2.1.0/include:${CPATH:-}"
PROJECT_DIR=/n/home09/pnielsen/inference_mirror_descent
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Started at: $(date)"
nvidia-smi -L
python -c "import jax; print('jax devices:', jax.devices())"

# Fast fail: one env, 20 env steps, so a broken pipeline surfaces in ~1 min
# instead of after the full sweep of snapshots.
echo "=== smoke test ==="
python -m scripts.snapshot_q_hessian_sensitivity \
  --steps 100000 \
  --prefix mgmd_2026-09-02_21-12-18_s22 \
  --episodes 10 --actions 30 --max-steps 20 \
  --out /tmp/qhess_smoke_${SLURM_JOB_ID:-0}.csv

echo "=== full run ==="
python -m scripts.snapshot_q_hessian_sensitivity \
  --steps 100000 500000 1000000 \
  --q-agg mean \
  --slot 0 \
  --episodes 10 \
  --actions 30 \
  --out analysis_cache/sweep225_q_hessian_sensitivity.csv

echo "Finished at: $(date)"
