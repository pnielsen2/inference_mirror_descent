#!/bin/bash
# One-off validation before the DDIM best-of-N sweep, on the config the
# historical eta=0 curve was measured on (ant, training eta 32, 40 steps, 1M):
#   A. the eta=0 MALA best-of-N is *reproduced* at HEAD, episode for episode,
#      against the stored CSV -- i.e. the legacy-state repair is faithful and
#      the two curves in the figure come from one code state.
#   B. the DDIM runner works end to end and its per-setting wall clock is
#      measured at the real episode count, on the biggest-observation env too.
#SBATCH -p kempner_h100
#SBATCH --account=kempner_kdbrantley_lab
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:1
#SBATCH -c 24
#SBATCH --mem=48G
#SBATCH -t 0-01:30
#SBATCH -o /n/home09/pnielsen/inference_mirror_descent/logs/slurm/%j.out
#SBATCH -e /n/home09/pnielsen/inference_mirror_descent/logs/slurm/%j.err
#SBATCH --job-name=ddimsmoke
set -euo pipefail

export PATH="$HOME/.venvs/general/bin:$PATH"
export VIRTUAL_ENV="$HOME/.venvs/general"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="$HOME/.local/glew/glew-2.1.0/include:${CPATH:-}"
PROJECT_DIR=/n/home09/pnielsen/inference_mirror_descent
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR:${PYTHONPATH:-}"

ROOT=/n/netscratch/kdbrantley_lab/Lab/pnielsen/diagnostic_snapshots
BASELINE="$ROOT/action_selection_eval_100ep_eta32_eta64_1m"
SMOKE="$ROOT/action_selection_ddim_smoke"
mkdir -p "$SMOKE"

echo "Started at: $(date)"; nvidia-smi -L

echo "=== A. reproduce the stored eta=0 best-of-N (N=1, 100 episodes) ==="
rm -f "$SMOKE/repro_ant_e32_d40.csv"
python -m scripts.evaluate_snapshot_action_selection \
  --snapshot "$ROOT/mgmd_2026-08-10_14-59-26_s0__pid1987646/Ant-v3_step_1000000" \
  --output "$SMOKE/repro_ant_e32_d40.csv" \
  --episodes 100 --values 1 --protocol base --q-agg-sample min
python - "$BASELINE/ant_traineta32_diff40_min.csv" "$SMOKE/repro_ant_e32_d40.csv" <<'PY'
import csv, sys
def returns(path):
    rows = [r for r in csv.DictReader(open(path))
            if r["protocol"] == "base_best_of_n" and int(r["value"]) == 1]
    return [float(r["episode_return"]) for r in sorted(rows, key=lambda r: int(r["episode_index"]))]
stored, fresh = returns(sys.argv[1]), returns(sys.argv[2])
assert len(stored) == len(fresh) == 100, (len(stored), len(fresh))
worst = max(abs(a - b) for a, b in zip(stored, fresh))
print(f"stored mean={sum(stored) / 100:.2f}  fresh mean={sum(fresh) / 100:.2f}  "
      f"max per-episode |diff|={worst:.3e}")
print("REPRODUCED" if worst < 1e-6 else "MISMATCH")
PY

echo "=== B. DDIM runner timings at 100 episodes (ant d40, then humanoid d40) ==="
for config in 0 2; do
  python -m scripts.evaluate_snapshot_ddim_best_of_n \
    --manifest "$BASELINE/manifest.txt" --config-index "$config" \
    --output-dir "$SMOKE" --steps 1000000 --values 1 256 1024 --episodes 100
done

echo "Finished at: $(date)"
