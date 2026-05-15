# Bit-Exact Baseline Test (sweep58)

This directory captures a **frozen, byte-level reference** for a specific
DPMD launch, used to verify that future simplifications of this codebase
do not change numerical behavior.

The reference run produced these baselines at git commit
**`d4bde26`** (branch `main`). The simplified branch tip at the time of
recording is **`4b3ad34`** (branch `simplified`); it was verified to
produce byte-identical CSV output for the same launch.

## What is bit-exact?

Under the exact environment described below, two runs of the same launch
command produce the same `episode_returns.csv` byte-for-byte (modulo
length, since one may have been cancelled earlier than the other).
SHA-256 of any prefix of one matches the same-length prefix of the other.

## Determinism requirements (REQUIRED)

The three env vars below MUST be exported in the shell that submits the
sbatch jobs. They are inherited via Slurm into each compute-node process.
**Without these flags, runs are NOT bit-exact** (different runs can
produce identical results in practice on identical hardware, but the
guarantee comes from these flags).

```bash
export XLA_FLAGS="--xla_gpu_deterministic_ops=true ${XLA_FLAGS:-}"
export TF_CUDNN_DETERMINISTIC=1
export JAX_DEFAULT_MATMUL_PRECISION=highest
```

## Software/hardware lockfile

The baselines in this directory were produced on:

- **GPU**: NVIDIA H100 80GB HBM3 (Slurm partition `kempner_h100`,
  reservation `kempner_kdbrantley_lab`).
- **Python**: 3.10.13
- **Venv**: `/n/home09/pnielsen/.venvs/general`
- **Key packages** (see `env_versions.txt` for the full pinned list):
  jax 0.6.2 + jaxlib 0.6.2 + jax-cuda12-pjrt 0.6.2,
  mujoco 3.4.0 + mujoco-py 2.1.2.14, gymnasium 1.0.0,
  flax 0.10.7, optax via flax, numpy 2.2.6.

If any of these change (especially JAX/CUDA/CUDNN/MuJoCo versions or GPU
SKU), bit-exactness is NOT guaranteed and the baselines should be
re-recorded by running `launch_baseline.sh` on the bit-exact commits.

## Launch command

The 6-env x 5-kl_budget x 8-seed ablation, packed 40-per-job:

```bash
bash tests/bit_exact_baseline/launch_baseline.sh
```

This wraps `python scripts/launch.py` with:

- `--ablate env Ant-v3 Humanoid-v3 HalfCheetah-v3 Walker2d-v3 Hopper-v3 Swimmer-v3`
- `--ablate kl_budget 16 64 256 1024 4096`
- `--seeds 0 1 2 3 4 5 6 7`
- `--max-runs-per-gpu 40` (=> 5 kl_budgets x 8 seeds vmap-packed per job)

Each of the 6 submitted jobs runs one env and produces:
```
logs/<env>/dpmd_<timestamp>_s0_/episode_returns.csv
```

## Baselines captured

| Env             | Rows  | Bytes  | SHA-256 prefix |
|-----------------|-------|--------|----------------|
| Ant-v3          | 1299  | 34603  | see csv_hashes.sha256 |
| Humanoid-v3     | 7999  | 205620 | see csv_hashes.sha256 |
| HalfCheetah-v3  | 201   | 5463   | see csv_hashes.sha256 |
| Walker2d-v3     | 3082  | 80311  | see csv_hashes.sha256 |

Hopper-v3 and Swimmer-v3 baselines are absent: their reference-side
jobs were cancelled before producing data. They can be added by running
`launch_baseline.sh` on commit `d4bde26` and copying their
`episode_returns.csv` into `baselines/Hopper-v3/` and `baselines/Swimmer-v3/`.

## How to verify a new simplification

Workflow when you add another simplification on the `simplified` branch:

```bash
cd /n/home09/pnielsen/inference_mirror_descent
git checkout simplified

# ... edit code, commit ...
git commit -am "Simplification step N: <description>"

# Set determinism vars and submit the baseline launch.
bash tests/bit_exact_baseline/launch_baseline.sh

# Wait until each job has produced at least as many rows as the
# baseline (Ant ~1300, Humanoid ~8000, HalfCheetah ~200, Walker2d ~3100).

# Compare:
python tests/bit_exact_baseline/verify.py
# Exit 0 => bit-exact, all envs.
# Exit 1 => some env regressed; the script prints first-diff line.
```

If verify fails, the simplification changed numerical behavior. Decide
whether the change is intentional (then re-record baselines) or a bug
(then revert).

## Files

| File                      | Purpose |
|---------------------------|---------|
| `README.md`               | This file. |
| `env_versions.txt`        | Frozen Python package versions + system info. |
| `launch_baseline.sh`      | Exports determinism vars and submits the 6-env x 5-kl_budget x 8-seed ablation. |
| `csv_hashes.sha256`       | SHA-256 of each baseline CSV. Run `sha256sum -c csv_hashes.sha256` to verify integrity. |
| `verify.py`               | Compares latest run CSV in `logs/<env>/dpmd_*_s0_/` against the baseline. |
| `baselines/<env>/episode_returns.csv` | Frozen reference CSVs. |
| `launch_scripts/`         | The 6 actual sbatch scripts produced by `launch_baseline.sh` on the recording day, kept for forensic reproducibility. |

## Provenance notes

- Reference run timestamps: 2026-05-15 17:40 to 17:46 EDT, GPUs:
  holygpu8a13201, holygpu8a15201, holygpu8a17404, holygpu8a15203.
- Simplified run timestamps: 2026-05-15 18:18 to 18:33 EDT,
  GPUs varied (4 of the 6 envs verified before remaining jobs cancelled).
- Bit-exactness confirmed on Ant-v3, Humanoid-v3, Walker2d-v3 between
  the recorded baselines and the simplified-branch tree.
