#!/usr/bin/env bash
# Bit-exact baseline launch wrapper for sweep58.
#
# Sets the determinism env vars and submits the 6-env x 5-kl_budget x 8-seed
# ablation that defines the bit-exact baseline. Slurm inherits the exported
# env vars into each compute-node job.
#
# Usage:
#   bash tests/bit_exact_baseline/launch_baseline.sh
#
# This will submit 6 jobs (one per env), each packing 40 vmap entries
# (5 kl_budgets x 8 seeds). Each run produces logs/<env>/mgmd_<ts>_s0_/episode_returns.csv
# which can be compared against baselines/<env>/episode_returns.csv via verify.py.

set -euo pipefail

# 1) Activate venv used for the baseline (jax 0.6.2 + cuda 12 + mujoco 3.4.0).
source /n/home09/pnielsen/.venvs/general/bin/activate

# 2) Determinism env vars (inherited by sbatch -> sbatch script -> python).
#    These three together are what makes runs bit-identical across the
#    reference and simplified codebases on H100 GPUs with the same JAX/CUDA
#    versions captured in env_versions.txt.
export XLA_FLAGS="--xla_gpu_deterministic_ops=true ${XLA_FLAGS:-}"
export TF_CUDNN_DETERMINISTIC=1
export JAX_DEFAULT_MATMUL_PRECISION=highest

# 3) Submit from the repo root (whichever branch is currently checked out).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Pick a sweep_id that doesn't collide with prior sweeps. Override via env.
SWEEP_ID="${SWEEP_ID:-58}"

python scripts/launch.py \
  --cmd "python scripts/train_mujoco.py --alg mgmd --num_vec_envs 5 \
    --mgmd_constant_weight --tfg_eta 8.0 --num_particles 1 --mala_steps 2 \
    --q_critic_agg mean --beta_schedule_type cosine --beta_schedule_scale 1 \
    --mgmd_no_entropy_tuning --buffer_size 400000 --x0_hat_clip_radius 3.0 \
    --mala_adapt_rate 0.2 --mala_per_level_eta --q_td_huber_width 30.0 \
    --update_per_iteration 4 --lr_q 0.00015 --lr_policy 0.0003 \
    --mala_guided_predictor --ddim_predictor --kl_budget 1024 --polyak_tau 0.005 \
    --advantage_ema_tau 0.001 --shape_ema_tau 0.0002 \
    --initial_advantage_second_moment_ema 1.0 --gamma 0.99 \
    --sweep_id ${SWEEP_ID} --config_tag_keys kl_budget" \
  --seeds 0 1 2 3 4 5 6 7 \
  --ablate env Ant-v3 Humanoid-v3 HalfCheetah-v3 Walker2d-v3 Hopper-v3 Swimmer-v3 \
  --ablate kl_budget 16 64 256 1024 4096 \
  --max-runs-per-gpu 40 \
  --time 2-00:00 \
  --mem 64G
