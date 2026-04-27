#!/usr/bin/env python3
"""
SLURM job launcher for experiment sweeps.

Usage examples:
    # Single job
    python scripts/launch.py --cmd "python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4"

    # Multiple seeds
    python scripts/launch.py --cmd "python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4" --seeds 0 1 2 3 4

    # Ablation sweep (changes from base command)
    python scripts/launch.py --cmd "python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4 --tfg_eta 16" \\
        --ablate tfg_eta 8 16 32

    # Multiple ablations with seeds
    python scripts/launch.py --cmd "python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v4" \\
        --seeds 100 101 102 \\
        --ablate tfg_eta 8 16 32 \\
        --ablate num_particles 1 64 128

    # One-at-a-time (non-Cartesian) local sweep around the base command.
    # Regular --ablate axes still form their usual Cartesian product
    # (here: env), while each --oat-ablate axis contributes its values one
    # at a time on top of the base config.
    python scripts/launch.py --cmd "python scripts/train_mujoco.py --alg dpmd --env HalfCheetah-v3 --lr_q 0.00015 --tau 0.005" \\
        --seeds 0 1 \\
        --ablate env HalfCheetah-v3 Ant-v3 Walker2d-v3 Humanoid-v3 \\
        --oat-ablate lr_q 0.000075 0.0003 \\
        --oat-ablate tau 0.0025 0.01

    # Dry run (print commands without submitting)
    python scripts/launch.py --cmd "..." --dry-run
"""

import argparse
import json
import subprocess
import os
import re
import shlex
import sys
import itertools
from collections import OrderedDict
from pathlib import Path
from datetime import datetime


# Which venv to use for each gymnasium environment version suffix.
# Keyed by the integer after "-v" in the `--env` flag (e.g. HalfCheetah-v3 → 3).
# All currently supported env versions launch from the `general` venv so
# standalone and packed runs use the same interpreter during this
# investigation.
# v3 environments still rely on mujoco-py / mujoco210 at runtime; the
# required library paths are set in the sbatch template below.
# To add support for a new env version, add an entry here.
VENV_BY_ENV_VERSION = {
    3: Path.home() / ".venvs" / "general",
    4: Path.home() / ".venvs" / "general",
    5: Path.home() / ".venvs" / "general",
}


# An ablation on any of these flags, whose values are all parseable as floats,
# is automatically treated as "easy" -- its values go into a per-seed hp_pack
# JSON and get vmap-packed via --parallel_seeds. Any other flag (non-numeric
# or not in this map) is "hard" and each value becomes a separate SLURM job.
# Seeds (--seeds) are always an implicit easy axis: multiple seeds share a GPU
# via vmap (each vmap entry gets a distinct RNG derived from the pack's anchor
# seed inside train_mujoco.py).
# Keys/values mirror the `_allowed` set in scripts/train_mujoco.py (the hp_pack
# loader).
# Pack keys match the argparse attribute names (i.e. the CLI flag with -- stripped).
# train_mujoco.py translates any names that differ from the Diffv2TrainState field
# via its own _CLI_TO_FIELD map. Keeping pack keys aligned with argparse means the
# per-slot values logged to wandb have the same key names the user sees on the
# command line, so wandb's filter / group / parallel-coordinates UI works without
# a translation step.
FLAG_TO_HP_KEY = {f: f for f in (
    "lr_q",
    "lr_policy",
    "gamma",
    "tau",
    "advantage_ema_tau",
    "shape_ema_tau",
    "initial_advantage_second_moment_ema",
    "initial_dist_shift_shape_ema",
    "shape3_ema_tau",
    "guidance_strength_multiplier",
    "kl_budget",
    "reward_scale",
    "x0_hat_clip_radius",
    "mala_adapt_rate",
    "q_td_huber_width",
    "q_critic_agg_idx",
)}


def _all_floats(values):
    for v in values:
        try:
            float(v)
        except (TypeError, ValueError):
            return False
    return True


def venv_site_packages(venv_path: Path):
    matches = sorted((venv_path / "lib").glob("python*/site-packages"))
    return matches[0] if matches else None


def cuda_runtime_env_prefixes(venv_path: Path):
    site_packages = venv_site_packages(venv_path)
    if site_packages is None:
        return "", ""
    nvidia_root = site_packages / "nvidia"
    path_prefix = ""
    ld_dirs = []
    cuda_nvcc_bin = nvidia_root / "cuda_nvcc" / "bin"
    if cuda_nvcc_bin.is_dir():
        path_prefix = f"{cuda_nvcc_bin}:"
    if nvidia_root.is_dir():
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            if lib_dir.is_dir():
                ld_dirs.append(str(lib_dir))
    ld_prefix = f"{':'.join(ld_dirs)}:" if ld_dirs else ""
    return path_prefix, ld_prefix


def classify_ablation(flag, values):
    """Return 'easy' iff flag is a per-seed vmappable hp and all its values
    parse as floats. Otherwise 'hard'."""
    if flag in FLAG_TO_HP_KEY and _all_floats(values):
        return "easy"
    return "hard"


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH -p {partition}
#SBATCH --account={account}
#SBATCH --gres=gpu:{gpu_type}:{num_gpus}
#SBATCH -c {cpus}
#SBATCH --mem={mem}
#SBATCH -t {time}
#SBATCH -o {log_dir}/%j.out
#SBATCH -e {log_dir}/%j.err
#SBATCH --job-name={job_name}
{requeue_directives}
set -euo pipefail
# Venv selected by launch.py based on the gymnasium env version in the
# --env flag (see VENV_BY_ENV_VERSION in scripts/launch.py).
export PATH="{cuda_bin_prefix}{venv_path}/bin:$PATH"
export VIRTUAL_ENV="{venv_path}"
export JAX_PLATFORMS=cuda
# mujoco210 runtime + NVIDIA/libOpenGL/libEGL paths for cymj GPU builder
export LD_LIBRARY_PATH="{cuda_ld_library_prefix}$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/lib64:${{LD_LIBRARY_PATH:-}}"
# Only needed if cymj has to rebuild on the compute node (normally cached).
export CPATH="$HOME/.local/glew/glew-2.1.0/include:${{CPATH:-}}"
cd {project_dir}

echo "Job ID: $SLURM_JOB_ID"
echo "Venv: $VIRTUAL_ENV"
python - <<'PY'
try:
    import jax
    devices = jax.devices()
except Exception as exc:
    raise SystemExit("launch.py: failed to initialize CUDA JAX backend: %s" % (exc,))
platforms = sorted(set(d.platform for d in devices))
if not any(p in ("gpu", "cuda") for p in platforms):
    raise SystemExit("launch.py: expected a GPU JAX backend, got %s" % (platforms,))
print("JAX devices:", ", ".join("%s:%s" % (d.platform, getattr(d, "device_kind", "unknown")) for d in devices))
PY
echo "Running: {cmd}"
echo "Started at: $(date)"
{wandb_offline_setup}
{cmd}

echo "Finished at: $(date)"
"""


# Inserted into every sbatch script when offline wandb is enabled (the
# default -- see --no-wandb-offline to opt out). Reasons for offline mode:
#   * ~/.local fills the login-node home quota when a 24-slot pack streams
#     directly, and wandb.ai rate-limits filestream registration when 300+
#     concurrent runs across a sweep hit init in the same second.
# The background _wandb_sync_loop uploads partial run data to wandb.ai every
# {sync_interval}s so the user can still monitor in-progress runs from the
# dashboard (live eval curves show up; run stays "running" until the final
# sync). trap EXIT catches normal exit, errors, and SIGTERM (scancel /
# slurm time limit) so the last sync runs whenever the job ends.
# Per-job WANDB_DIR (includes ${{SLURM_JOB_ID}}) prevents concurrent
# sbatch jobs' sync loops from racing on the same offline-run subdirs.
_WANDB_OFFLINE_BLOCK = """
export WANDB_MODE=offline
export WANDB_DIR="{wandb_dir}"
mkdir -p "$WANDB_DIR/wandb"
echo "WANDB_MODE=$WANDB_MODE"
echo "WANDB_DIR=$WANDB_DIR"

# wandb sync --sync-all has a quirk: passing an explicit PATH makes it
# treat PATH as one run dir (which ours isn't -- ours is a parent of
# offline-run-* subdirs), giving "Nothing to sync". Only the cwd-relative
# form ("cd <parent> && wandb sync --sync-all") correctly walks offline-run-*
# subdirs. So we cd into WANDB_DIR (whose `./wandb` child holds all this
# job's offline-run dirs) before each sync call.
_wandb_sync_loop() {{
  while true; do
    sleep {sync_interval}
    (cd "$WANDB_DIR" && nice -n 19 wandb sync --sync-all --include-synced) >> "$WANDB_DIR/sync.log" 2>&1 || true
  done
}}
_wandb_sync_loop &
_WANDB_SYNC_PID=$!

_wandb_cleanup() {{
  kill $_WANDB_SYNC_PID 2>/dev/null || true
  echo "Final wandb sync at $(date)"
  (cd "$WANDB_DIR" && nice -n 19 wandb sync --sync-all --include-synced) >> "$WANDB_DIR/sync.log" 2>&1 || true
}}
trap _wandb_cleanup EXIT INT TERM
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch SLURM jobs for experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required
    parser.add_argument("--cmd", type=str, required=True,
                        help="Base python command to run")
    
    # Sweep options
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="List of seeds to run (adds --seed X to command)")
    parser.add_argument("--ablate", action="append", nargs="+", metavar=("FLAG", "VALUES"),
                        help="Ablation: --ablate flag_name val1 val2 val3. Can be used multiple times.")
    parser.add_argument("--oat-ablate", "--oat_ablate", action="append", nargs="+",
                        metavar=("FLAG", "VALUES"), dest="oat_ablate",
                        help="One-at-a-time ablation: base config plus each listed value for one flag, "
                             "without taking the Cartesian product across oat axes. Can be used multiple times. "
                             "Regular --ablate axes still form their usual Cartesian product.")
    parser.add_argument("--max-runs-per-gpu", "--max_runs_per_gpu", type=int, default=1,
                        dest="max_runs_per_gpu",
                        help="Max runs vmap-packed on a single GPU (default: 1, no packing). "
                             "Seeds and ablations on per-seed-vmappable float hps are packed "
                             "together up to this size; other ablations stay on separate GPUs.")

    # SLURM options
    parser.add_argument("--seas", action="store_true",
                        help="Use SEAS cluster (seas_gpu partition, H200 GPUs) instead of Kempner")
    parser.add_argument("--partition", "-p", type=str, default=None,
                        help="SLURM partition (default: seas_gpu, or kempner_h100 with --kempner)")
    parser.add_argument("--gpu-type", type=str, default=None,
                        help="GPU type (default: nvidia_h200, or nvidia_h100 with --kempner)")
    parser.add_argument("--account", type=str, default=None,
                        help="SLURM account (default: kempner_kdbrantley_lab for --kempner, kdbrantley_lab otherwise)")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs (default: 1)")
    parser.add_argument("--cpus", "-c", type=int, default=None,
                        help="Number of CPUs per job. Default: auto-picked as "
                             "min(pack_size * num_vec_envs, 24) -- one CPU per "
                             "concurrent env subprocess, capped at the typical "
                             "Kempner H100 per-GPU limit.")
    parser.add_argument("--cpus-cap", type=int, default=24,
                        dest="cpus_cap",
                        help="Upper bound for auto-picked --cpus (default: 24, "
                             "the Kempner H100 per-GPU CPU allotment).")
    parser.add_argument("--mem", type=str, default="32G",
                        help="Memory (default: 32G)")
    parser.add_argument("--time", "-t", type=str, default="0-8:00",
                        help="Time limit (default: 0-8:00)")
    parser.add_argument("--requeue", action="store_true",
                        help="Enable --requeue and --signal=SIGTERM@120 for preemptible partitions")

    # Offline wandb + background sync. On by default because (a) login-node
    # home dirs are too small for a 24-slot pack's filestream buffer and
    # (b) offline-then-sync insulates us from wandb.ai rate-limiting at init.
    parser.add_argument("--no-wandb-offline", dest="wandb_offline",
                        action="store_false", default=True,
                        help="Disable offline wandb + background sync. "
                             "Runs stream directly to wandb.ai (the pre-2026 "
                             "behavior). Use only for small local smoke tests.")
    parser.add_argument("--wandb-offline-base", "--wandb_offline_base",
                        type=str, default=None, dest="wandb_offline_base",
                        help="Base dir for offline wandb run dirs. "
                             "Default: /n/netscratch/kdbrantley_lab/Lab/$USER/wandb. "
                             "Each sbatch job gets <base>/sweep_<N>/job_<slurm_id>/, "
                             "so concurrent jobs' sync loops never race on the "
                             "same offline-run subdirs.")
    parser.add_argument("--wandb-sync-interval", "--wandb_sync_interval",
                        type=int, default=60, dest="wandb_sync_interval",
                        help="Seconds between background `wandb sync --sync-all` "
                             "calls during training (default: 60). A final sync "
                             "runs on job exit via trap EXIT.")

    # Utility options
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without submitting")
    parser.add_argument("--venv", type=str, default=None,
                        help="Path to the virtualenv to activate inside the sbatch script. "
                             "Default: infer from the training command's --env version.")
    parser.add_argument("--job-name", type=str, default=None,
                        help="Job name prefix (default: inferred from command)")
    parser.add_argument("--sweep-id", "--sweep_id", type=int, default=None,
                       help="Override the auto-assigned sweep_id. By default we "
                            "query wandb once for the set of existing "
                            "config.sweep_id values in "
                            "pnielsen2-harvard/diffusion_online_rl and pick the "
                            "smallest unused positive integer, then inject "
                            "'--sweep_id <N>' into every slurm job's command. "
                            "Downstream analysis (compute_topsis.py --sweep-id) "
                            "filters on this field to isolate one sweep.")
    parser.add_argument("--no-sweep-id", "--no_sweep_id", action="store_true",
                       help="Do not assign a sweep_id (compute_topsis.py would "
                            "need the legacy name-prefix mode to pick up the "
                            "runs).")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory for logs (default: logs/slurm/<timestamp>)")
    
    args = parser.parse_args()
    if args.seas:
        args.partition = args.partition or "seas_gpu"
        args.gpu_type = args.gpu_type or "nvidia_h200"
        args.account = args.account or "kdbrantley_lab"
    elif args.requeue:
        args.partition = args.partition or "kempner_requeue"
        args.gpu_type = args.gpu_type or "nvidia_h100_80gb_hbm3"
        args.account = args.account or "kempner_kdbrantley_lab"
    else:
        args.partition = args.partition or "kempner_h100"
        args.gpu_type = args.gpu_type or "nvidia_h100_80gb_hbm3"
        args.account = args.account or "kempner_kdbrantley_lab"
    return args


def infer_job_name(cmd: str) -> str:
    """Extract a short job name from the command."""
    # Try to extract --alg and --env
    alg_match = re.search(r'--alg\s+(\S+)', cmd)
    env_match = re.search(r'--env\s+(\S+)', cmd)
    
    parts = []
    if alg_match:
        parts.append(alg_match.group(1))
    if env_match:
        # Shorten env name (e.g., HalfCheetah-v4 -> HC)
        env = env_match.group(1)
        short_env = ''.join(c for c in env if c.isupper() or c.isdigit())
        parts.append(short_env)
    
    return "_".join(parts) if parts else "job"


def _normalize_cmd(cmd: str) -> str:
    """Collapse embedded newlines / runs of whitespace from a shell-friendly
    multi-line --cmd into a single line. Required so flags we append later
    (e.g. --parallel_seeds) don't land on a new bash line."""
    return " ".join(cmd.split())


def extract_int_flag(cmd: str, flag: str, default: int) -> int:
    """Pull an integer flag value from ``cmd`` (e.g. ``--num_vec_envs 5``)."""
    m = re.search(rf'--{re.escape(flag)}\s+(\S+)', cmd)
    if not m:
        return default
    try:
        return int(m.group(1))
    except ValueError:
        return default


def auto_cpus(pack: dict, cap: int) -> int:
    """Pick --cpus so there's one CPU per concurrent env subprocess, capped.

    Each worker in FutexProcessVectorEnv is a single-threaded subprocess
    (OMP_NUM_THREADS=1); workers default to one-per-env. All workers race
    simultaneously on every env.step(), so one CPU per env avoids the
    barrier-tail overhead of oversubscription. We cap at ``cap`` (typically
    the per-GPU CPU allotment on the target cluster).
    """
    # Fallback matches train_mujoco.py's --num_vec_envs default so auto_cpus
    # sizes correctly when the flag is omitted from the base cmd.
    num_vec_envs = extract_int_flag(pack["cmd"], "num_vec_envs", 5)
    total_envs = max(1, pack["pack_size"]) * max(1, num_vec_envs)
    # Always reserve at least 2 cores so the main process + one worker coexist.
    return max(2, min(total_envs, cap))


def _remove_flag(cmd: str, flag: str) -> str:
    """Strip --flag (and its value, if it has one) from cmd.

    Works for both valued flags (--flag X) and argparse store_true / bare flags
    (--flag followed by another --flag or end-of-string). Returns the cmd with
    the flag cleanly excised; safe to call even if the flag isn't present.
    """
    tokens = cmd.split()
    out = []
    i = 0
    target = f"--{flag}"
    while i < len(tokens):
        if tokens[i] == target:
            # Skip --flag. Also skip the next token IFF it's a value
            # (doesn't start with '--').
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                i += 2
            else:
                i += 1
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def modify_cmd_for_flag(cmd: str, flag: str, value: str) -> str:
    """Replace or add a flag in the command.

    Special boolean semantics for argparse store_true flags:
      value == "True"  -> emit --flag alone (no value).
      value == "False" -> omit --flag entirely (so argparse default=False stays).
    Any other value is treated as a regular --flag VALUE pair.

    The flag is always removed from the existing cmd first, so repeated calls
    for the same flag end up idempotent and don't leave stale copies behind.
    """
    cmd = _remove_flag(cmd, flag)
    if value == "False":
        return cmd
    if value == "True":
        return f"{cmd} --{flag}"
    return f"{cmd} --{flag} {value}"


def extract_flag_value(cmd: str, flag: str):
    """Return the current CLI value for ``--flag`` in ``cmd``.

    Returns:
      - string value for valued flags,
      - "True" for bare/store_true flags present in the command,
      - None if the flag is absent.
    """
    tokens = shlex.split(cmd)
    target = f"--{flag}"
    i = 0
    while i < len(tokens):
        if tokens[i] == target:
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                return tokens[i + 1]
            return "True"
        i += 1
    return None


# Ablations the user writes with string values that the sweep machinery
# secretly translates into a per-slot numeric vmap axis. Kept small and
# explicit so the magic is easy to audit. Each entry maps
#   (user_flag, frozenset-of-user-values)
# to
#   (new_flag, value_translator,  extra_base_cmd_tweaks)
# where value_translator is a dict user_value -> float, and
# extra_base_cmd_tweaks is a list of (flag, value) pairs forced into the
# base cmd so train_mujoco.py knows to activate the vmap path.
_PSEUDO_EASY_ABLATIONS = {
    # Mean vs Min Q-critic aggregation. Users write this as
    #   --ablate q_critic_agg mean min
    # but the trainer can only vmap a float, so launch.py translates it to
    #   --ablate q_critic_agg_idx 0.0 1.0
    # and forces --q_critic_agg vmap_mean_min into the base cmd.
    "q_critic_agg": {
        "allowed_values": {"mean", "min"},
        "new_flag": "q_critic_agg_idx",
        "translator": {"mean": "0.0", "min": "1.0"},
        "base_cmd_override": [("q_critic_agg", "vmap_mean_min")],
    },
}


def preprocess_ablations(base_cmd, ablations, source_label="--ablate"):
    """Translate pseudo-easy ablations into their real vmappable form.

    Only fires when every value the user provided for the flag is in the
    allowed set (e.g. all of {mean, min}). Falls through unchanged otherwise,
    so an unrecognized value keeps the original HARD classification and the
    user gets a loud failure from argparse rather than silent mistranslation.
    """
    if not ablations:
        return base_cmd, ablations, []
    new_ablations = []
    notes = []
    for ab in ablations:
        flag = ab[0]
        values = list(ab[1:])
        spec = _PSEUDO_EASY_ABLATIONS.get(flag)
        if spec is not None and set(values).issubset(spec["allowed_values"]):
            base_user_value = extract_flag_value(base_cmd, flag)
            new_flag = spec["new_flag"]
            new_values = [spec["translator"][v] for v in values]
            for bf, bv in spec["base_cmd_override"]:
                base_cmd = modify_cmd_for_flag(base_cmd, bf, bv)
            if (base_user_value in spec["translator"] and
                    extract_flag_value(base_cmd, new_flag) is None):
                base_cmd = modify_cmd_for_flag(
                    base_cmd, new_flag, spec["translator"][base_user_value]
                )
            new_ablations.append([new_flag] + new_values)
            notes.append(
                f"  translated {source_label} {flag} {' '.join(values)}  ->  "
                f"{source_label} {new_flag} {' '.join(new_values)}"
                + ("  (base cmd now forces "
                   + ", ".join(f"--{bf} {bv}" for bf, bv in spec["base_cmd_override"])
                   + (f", --{new_flag} {spec['translator'][base_user_value]}"
                      if base_user_value in spec["translator"] else "")
                   + ")")
            )
        else:
            new_ablations.append(ab)
    return base_cmd, new_ablations, notes


def _normalize_assignment(flag: str, value: str, base_values: dict):
    """Drop overrides that are semantically identical to the base command."""
    base_val = base_values.get(flag)
    if value == base_val:
        return None
    if base_val is None and value == "False":
        return None
    return value


def _classify_ablations(ablations):
    hard_ablations = []
    easy_ablations = []
    if ablations:
        for ab in ablations:
            flag = ab[0]
            values = list(ab[1:])
            if classify_ablation(flag, values) == "easy":
                easy_ablations.append((flag, values))
            else:
                hard_ablations.append((flag, values))
    return hard_ablations, easy_ablations


def _generate_run_specs(base_cmd: str, seeds: list, ablations: list,
                        oat_ablations: list) -> list:
    """Generate concrete run specs before vmap chunking.

    Each spec has fully resolved hard/easy overrides for one logical run
    (before seeds / easy hps are chunked into vmap packs).
    """
    base_values = {}
    all_flags = set()
    for src in (ablations or []), (oat_ablations or []):
        for ab in src:
            all_flags.add(ab[0])
    for flag in all_flags:
        base_values[flag] = extract_flag_value(base_cmd, flag)

    hard_ablations, easy_ablations = _classify_ablations(ablations)
    oat_hard_ablations, oat_easy_ablations = _classify_ablations(oat_ablations)

    regular_hard_flags = [f for f, _ in hard_ablations]
    regular_hard_lists = [vs for _, vs in hard_ablations]
    regular_hard_combos = (list(itertools.product(*regular_hard_lists))
                           if hard_ablations else [()])

    regular_easy_flags = [f for f, _ in easy_ablations]
    regular_easy_lists = [vs for _, vs in easy_ablations]
    regular_easy_combos = (list(itertools.product(*regular_easy_lists))
                           if easy_ablations else [()])

    oat_variants = [{"hard": {}, "easy": {}, "desc_parts": []}]
    for flag, values in oat_hard_ablations:
        for value in values:
            norm = _normalize_assignment(flag, value, base_values)
            if norm is None:
                continue
            oat_variants.append({
                "hard": {flag: value},
                "easy": {},
                "desc_parts": [f"{flag}={value}"],
            })
    for flag, values in oat_easy_ablations:
        for value in values:
            norm = _normalize_assignment(flag, value, base_values)
            if norm is None:
                continue
            oat_variants.append({
                "hard": {},
                "easy": {flag: value},
                "desc_parts": [f"{flag}={value}"],
            })

    # To mix different oat-easy scenarios inside one vmap pack, launch.py needs
    # to know the base value for every oat-easy flag so entries that *aren't*
    # varying that flag can still get an explicit per-slot value in hp_pack.
    missing_explicit_base = [
        flag for flag, _ in oat_easy_ablations
        if base_values.get(flag) is None
    ]
    if missing_explicit_base:
        missing = ", ".join(f"--{flag}" for flag in missing_explicit_base)
        sys.exit(
            "launch.py: one-at-a-time easy ablations require explicit base values in --cmd "
            "so mixed OAT packs can materialize the per-slot hp_pack. Please add these "
            f"flags explicitly to --cmd: {missing}"
        )

    seed_axis = list(seeds) if seeds else [None]
    runs = []
    seen = set()
    for hard_combo in regular_hard_combos:
        hard_base = dict(zip(regular_hard_flags, hard_combo))
        for easy_combo in regular_easy_combos:
            easy_base = dict(zip(regular_easy_flags, easy_combo))
            for oat in oat_variants:
                hard = dict(hard_base)
                easy = dict(easy_base)
                hard.update(oat["hard"])
                easy.update(oat["easy"])

                # Normalize away any explicit duplicates of the base command.
                norm_hard = OrderedDict()
                for flag, value in sorted(hard.items()):
                    norm = _normalize_assignment(flag, value, base_values)
                    if norm is not None:
                        norm_hard[flag] = value
                norm_easy = OrderedDict()
                for flag, value in sorted(easy.items()):
                    norm = _normalize_assignment(flag, value, base_values)
                    if norm is not None:
                        norm_easy[flag] = value

                desc_parts = [f"{f}={v}" for f, v in norm_hard.items()]
                desc_parts.extend(f"{f}={v}" for f, v in norm_easy.items())
                for seed in seed_axis:
                    key = (tuple(norm_hard.items()), tuple(norm_easy.items()), seed)
                    if key in seen:
                        continue
                    seen.add(key)
                    runs.append({
                        "hard": dict(norm_hard),
                        "easy": dict(norm_easy),
                        "seed": seed,
                        "desc_parts": desc_parts + ([f"seed={seed}"] if seed is not None else []),
                    })
    return runs, base_values


def generate_packs(base_cmd: str, seeds: list, ablations: list,
                   oat_ablations: list, max_runs_per_gpu: int) -> list:
    """Generate vmap packs.

    Each pack represents one SLURM job. A pack has either ``pack_size == 1``
    (plain CLI, no vmap) or ``pack_size > 1`` (vmap-packed via
    ``--parallel_seeds`` plus an ``--hp_pack`` JSON for any per-seed hp
    ablations).

    Partitioning:
      - Hard ablations (``classify_ablation == 'hard'``) form the outer
        Cartesian product; each hard combo becomes a disjoint set of SLURM
        jobs.
      - Easy ablations + seeds form the vmap axis: their Cartesian product
        is chunked into groups of size <= max_runs_per_gpu, and each group
        becomes one SLURM job whose K=group_size entries run in parallel on
        the GPU.

    Returns a list of dicts with keys:
      ``cmd``     -- cmd string. For vmap packs the pack JSON is inlined as
                     ``--hp_pack_inline '<json>'`` directly in the cmd; no
                     sidecar file is needed.
      ``hp_pack`` -- dict ``{hp_key: [val_0, ..., val_{K-1}]}`` or ``None``
                     (returned for introspection / logging; not used for I/O).
      ``desc``    -- short human-readable description for logging.
      ``pack_size`` -- K (number of parallel vmap entries).
    """
    runs, base_values = _generate_run_specs(base_cmd, seeds, ablations, oat_ablations)

    grouped = OrderedDict()
    for run in runs:
        key = tuple(sorted(run["hard"].items()))
        grouped.setdefault(key, []).append(run)

    packs = []
    for hard_key, grouped_runs in grouped.items():
        hard_pairs = list(hard_key)
        hard_desc = [f"{f}={v}" for f, v in hard_pairs]

        for i in range(0, len(grouped_runs), max_runs_per_gpu):
            chunk = grouped_runs[i : i + max_runs_per_gpu]
            K = len(chunk)

            if K == 1:
                run = chunk[0]
                cmd = base_cmd
                for flag, val in hard_pairs:
                    cmd = modify_cmd_for_flag(cmd, flag, val)
                for flag, val in run["easy"].items():
                    cmd = modify_cmd_for_flag(cmd, flag, val)
                if run["seed"] is not None:
                    cmd = modify_cmd_for_flag(cmd, "seed", str(run["seed"]))

                desc = ",".join(run["desc_parts"]) if run["desc_parts"] else "base"
                packs.append({"cmd": cmd, "hp_pack": None,
                              "desc": desc, "pack_size": 1})
                continue

            anchor_seed = chunk[0]["seed"]
            easy_flags_in_chunk = []
            for run in chunk:
                for flag in run["easy"]:
                    if flag not in easy_flags_in_chunk:
                        easy_flags_in_chunk.append(flag)

            hp_pack = {}
            for flag in easy_flags_in_chunk:
                hp_key = FLAG_TO_HP_KEY[flag]
                values = []
                base_val = base_values.get(flag)
                for run in chunk:
                    value = run["easy"].get(flag, base_val)
                    if value is None:
                        sys.exit(
                            "launch.py: internal error building one-at-a-time hp_pack; "
                            f"missing base value for --{flag}."
                        )
                    values.append(float(value))
                hp_pack[hp_key] = values

            chunk_seeds = [run["seed"] for run in chunk]
            if any(s is not None for s in chunk_seeds):
                hp_pack["seed"] = [int(s) for s in chunk_seeds]

            cmd = base_cmd
            for flag, val in hard_pairs:
                cmd = modify_cmd_for_flag(cmd, flag, val)
            if anchor_seed is not None:
                cmd = modify_cmd_for_flag(cmd, "seed", str(anchor_seed))
            cmd = modify_cmd_for_flag(cmd, "parallel_seeds", str(K))
            if hp_pack:
                inline = json.dumps(hp_pack, separators=(",", ":"))
                cmd = f"{cmd} --hp_pack_inline {shlex.quote(inline)}"

            vmap_desc = []
            seeds_in_chunk = [run["seed"] for run in chunk if run["seed"] is not None]
            if seeds_in_chunk:
                if len(set(seeds_in_chunk)) == 1:
                    vmap_desc.append(f"seed={seeds_in_chunk[0]}")
                else:
                    vmap_desc.append(f"seed=[{min(seeds_in_chunk)}..{max(seeds_in_chunk)}]")
            for flag in easy_flags_in_chunk:
                uniq = []
                base_val = base_values.get(flag)
                for run in chunk:
                    value = str(run["easy"].get(flag, base_val))
                    if value not in uniq:
                        uniq.append(value)
                if len(uniq) == 1:
                    vmap_desc.append(f"{flag}={uniq[0]}")
                else:
                    vmap_desc.append(f"{flag}=[{','.join(uniq)}]")
            desc = ",".join(hard_desc + vmap_desc + [f"vmap×{K}"])

            packs.append({"cmd": cmd, "hp_pack": hp_pack,
                          "desc": desc, "pack_size": K})

    return packs


def venv_for_command(cmd: str, override=None) -> Path:
    """Pick the venv for a training command based on its ``--env`` flag.

    Parses the trailing ``-v<N>`` version off the gymnasium env name and
    looks it up in ``VENV_BY_ENV_VERSION``. Aborts with a clear error if
    the command has no ``--env``, the name has no version suffix, or the
    version has no registered venv — we'd rather stop the sweep than
    silently ship jobs to the wrong interpreter.
    """
    if override is not None:
        venv_path = Path(override).expanduser().resolve()
        if not (venv_path / "bin" / "python").exists():
            sys.exit(
                f"launch.py: requested --venv {venv_path} but "
                f"{venv_path / 'bin' / 'python'} does not exist."
            )
        return venv_path
    env_match = re.search(r'--env\s+(\S+)', cmd)
    if not env_match:
        sys.exit(f"launch.py: command has no --env flag, can't pick a venv:\n  {cmd}")
    env_name = env_match.group(1)
    ver_match = re.search(r'-v(\d+)$', env_name)
    if not ver_match:
        sys.exit(
            f"launch.py: env '{env_name}' has no -v<N> version suffix; "
            f"don't know which venv to use."
        )
    version = int(ver_match.group(1))
    if version not in VENV_BY_ENV_VERSION:
        known = ", ".join(f"v{v}" for v in sorted(VENV_BY_ENV_VERSION))
        sys.exit(
            f"launch.py: no venv registered for env version v{version} "
            f"(env='{env_name}'). Known versions: {known}. "
            f"Add an entry to VENV_BY_ENV_VERSION in launch.py."
        )
    venv_path = VENV_BY_ENV_VERSION[version].resolve()
    if not (venv_path / "bin" / "python").exists():
        sys.exit(
            f"launch.py: venv for v{version} envs is {venv_path} but "
            f"{venv_path / 'bin' / 'python'} does not exist."
        )
    return venv_path


def next_unused_sweep_id(project="pnielsen2-harvard/diffusion_online_rl"):
    """Return the smallest positive integer N such that no wandb run in the
    project has ``config.sweep_id == N`` yet. Probes ascending sweep_ids via
    ``len(api.runs(filters={config.sweep_id: N}))`` (a single O(1) backend
    count per probe); stops at the first N with zero matches. Transient
    failures fall back to a timestamp-derived id so launch.py never blocks
    on a wandb outage.

    Implementation note: we do NOT iterate runs and read ``r.config.get("sweep_id")``
    -- the Public API's paginated ``Runs`` iterator returns a stripped config
    that lies about field presence (it reports None for every key). Equality
    filters on ``config.sweep_id`` ARE honored by the backend though, so
    len() on a filtered query is the reliable signal.
    """
    try:
        import wandb
        api = wandb.Api(timeout=20)
        n = 1
        # Hard cap keeps a bugged backend from spinning forever. 10000 sweeps
        # covers any realistic project lifetime.
        while n < 10000:
            runs = api.runs(project, filters={"config.sweep_id": n},
                            per_page=1)
            if len(runs) == 0:
                return n
            n += 1
        raise RuntimeError(
            f"next_unused_sweep_id: probed up to {n} without finding a free "
            f"slot; backend may be misbehaving"
        )
    except Exception as e:
        print(f"[launch.py] wandb query for existing sweep ids failed ({e}); "
              f"falling back to timestamp-derived id", flush=True)
        # Minutes-since-epoch keeps the fallback id stable within one launch
        # invocation and low-collision across concurrent invocations.
        return int(datetime.now().timestamp() // 60)


def main():
    args = parse_args()

    # Setup paths
    project_dir = Path(__file__).parent.parent.resolve()

    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = project_dir / "logs" / "slurm" / timestamp

    log_dir.mkdir(parents=True, exist_ok=True)

    # Resolve sweep_id. Explicit --sweep-id > --no-sweep-id > auto-query wandb.
    if args.no_sweep_id:
        sweep_id = None
    elif args.sweep_id is not None:
        sweep_id = int(args.sweep_id)
    else:
        sweep_id = next_unused_sweep_id()
    if sweep_id is not None:
        print(f"[launch.py] sweep_id = {sweep_id}  "
              f"(wandb group 'sweep_{sweep_id}')")

    # Inject sweep_id into the base command so every pack's derived cmd carries
    # it. We treat it like a first-class CLI flag: replace any existing
    # --sweep_id or append it.
    base_cmd = _normalize_cmd(args.cmd)
    if sweep_id is not None:
        base_cmd = modify_cmd_for_flag(base_cmd, "sweep_id", str(sweep_id))

    # Infer job name if not provided
    job_name_base = args.job_name or infer_job_name(args.cmd)

    if args.max_runs_per_gpu < 1:
        sys.exit(f"launch.py: --max-runs-per-gpu must be >= 1, got {args.max_runs_per_gpu}")

    # Translate pseudo-easy ablations (e.g. q_critic_agg mean/min) into their
    # real vmappable form before partitioning. Print a note so users can see
    # how their --ablate was rewritten.
    base_cmd, args_ablate, pseudo_notes = preprocess_ablations(
        base_cmd, args.ablate, source_label="--ablate"
    )
    base_cmd, args_oat_ablate, pseudo_notes_oat = preprocess_ablations(
        base_cmd, args.oat_ablate, source_label="--oat-ablate"
    )
    pseudo_notes = pseudo_notes + pseudo_notes_oat
    if pseudo_notes:
        print("[launch.py] pseudo-easy ablation translations:")
        for n in pseudo_notes:
            print(n)

    # Tell the trainer which argparse attributes to include in each per-slot
    # config_tag. We pick the union of every --ablate axis (hard and easy)
    # except env and seed: env is the analysis pivot dimension (reward column
    # per env) and seed is the replicate dimension (aggregate across seeds),
    # so including either would over-specify the tag and break seed/env
    # grouping. Other hard axes (booleans like
    # td_use_target_policy, decorrelated_q_batches) MUST be in the tag or
    # runs that differ only in them get silently conflated by config_tag
    # grouping. Pack keys get per-slot values in build_config_tag; hard
    # flag values come from the slurm job's argparse attributes.
    TAG_EXCLUDE = {"env", "seed"}
    tag_sources = (args_ablate or []) + (args_oat_ablate or [])
    tag_keys = sorted({ab[0] for ab in tag_sources if ab[0] not in TAG_EXCLUDE})
    if tag_keys:
        base_cmd = modify_cmd_for_flag(
            base_cmd, "config_tag_keys", ",".join(tag_keys)
        )
        print(f"[launch.py] config_tag_keys (included in every run's "
              f"config_tag): {tag_keys}")

    # Generate all packs (each pack = one SLURM job).
    packs = generate_packs(base_cmd, args.seeds, args_ablate, args_oat_ablate,
                           args.max_runs_per_gpu)

    total_runs = sum(p["pack_size"] for p in packs)
    print(f"Generated {len(packs)} job(s) covering {total_runs} run(s) "
          f"(max_runs_per_gpu={args.max_runs_per_gpu})")
    print(f"Log directory: {log_dir}")

    # Resolve the per-job offline wandb dir template. The ${SLURM_JOB_ID}
    # part is a literal bash variable -- resolved on the compute node, not
    # here -- so each concurrent sbatch job gets a disjoint dir and their
    # background sync loops never race on the same offline-run subdirs.
    if args.wandb_offline:
        if args.wandb_offline_base is not None:
            base = Path(args.wandb_offline_base)
        else:
            user = os.environ.get("USER", "pnielsen")
            base = Path(f"/n/netscratch/kdbrantley_lab/Lab/{user}/wandb")
        sweep_subdir = f"sweep_{sweep_id}" if sweep_id is not None else \
                       f"nosweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_dir_template = f"{base}/{sweep_subdir}/job_${{SLURM_JOB_ID}}"
        print(f"[launch.py] offline wandb: WANDB_DIR template = "
              f"{wandb_dir_template}  (sync every {args.wandb_sync_interval}s)")
    else:
        wandb_dir_template = None
        print("[launch.py] offline wandb: DISABLED (runs will stream to "
              "wandb.ai directly)")
    print()

    submitted = 0
    for i, pack in enumerate(packs):
        cmd = pack["cmd"]
        desc = pack["desc"]
        # The pack (if any) is already inlined into cmd as --hp_pack_inline
        # '<JSON>' by generate_packs. No sidecar pack_i.json is written.
        job_name = f"{job_name_base}_{i}" if len(packs) > 1 else job_name_base
        venv_path = venv_for_command(cmd, args.venv)
        cuda_bin_prefix, cuda_ld_library_prefix = cuda_runtime_env_prefixes(venv_path)

        if args.requeue:
            requeue_directives = "#SBATCH --requeue\n#SBATCH --signal=B:SIGTERM@120"
        else:
            requeue_directives = ""

        cpus = args.cpus if args.cpus is not None else auto_cpus(pack, args.cpus_cap)

        if wandb_dir_template is not None:
            wandb_offline_setup = _WANDB_OFFLINE_BLOCK.format(
                wandb_dir=wandb_dir_template,
                sync_interval=args.wandb_sync_interval,
            )
        else:
            wandb_offline_setup = ""

        script = SBATCH_TEMPLATE.format(
            partition=args.partition,
            account=args.account,
            gpu_type=args.gpu_type,
            num_gpus=args.num_gpus,
            cpus=cpus,
            mem=args.mem,
            time=args.time,
            log_dir=log_dir,
            job_name=job_name,
            project_dir=project_dir,
            cmd=cmd,
            requeue_directives=requeue_directives,
            venv_path=venv_path,
            cuda_bin_prefix=cuda_bin_prefix,
            cuda_ld_library_prefix=cuda_ld_library_prefix,
            wandb_offline_setup=wandb_offline_setup,
        )
        
        if args.dry_run:
            print(f"[{i+1}/{len(packs)}] {desc}  "
                  f"[venv={venv_path.name} cpus={cpus}]")
            print(f"  Command: {cmd}")
            print()
        else:
            # Write script to temp file and submit
            script_path = log_dir / f"{job_name}_{i}.sh"
            script_path.write_text(script)

            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                print(f"[{i+1}/{len(packs)}] Submitted job {job_id}: {desc}  [venv={venv_path.name}]")
                submitted += 1
            else:
                print(f"[{i+1}/{len(packs)}] FAILED: {desc}")
                print(f"  Error: {result.stderr}")
    
    if not args.dry_run:
        print()
        print(f"Submitted {submitted}/{len(packs)} jobs")
        print(f"Monitor with: squeue -u $USER")
        print(f"Logs in: {log_dir}")


if __name__ == "__main__":
    main()
