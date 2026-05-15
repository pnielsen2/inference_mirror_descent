"""Multi-seed wandb logging plumbing for VmapOffPolicyTrainer.

Owns one wandb.Run per vmap seed, plus the host-side per-seed pending-scalar
buffers and the update-step array accumulator. The trainer itself is now
RL-focused; it just calls ``logger.add_scalar_per_seed(...)`` /
``logger.flush_all()`` etc. Behavior (PRNG, log keys, log values, log steps,
config_tag strings) is byte-identical to the old in-trainer implementation.
"""
import os
import random
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import wandb


_WANDB_INIT_STAGGER_MIN_SECONDS = float(
    os.environ.get("WANDB_INIT_STAGGER_MIN_SECONDS", "0.0")
)
_WANDB_INIT_STAGGER_MAX_SECONDS = float(
    os.environ.get("WANDB_INIT_STAGGER_MAX_SECONDS", "3.0")
)


def _format_tag_value(v) -> str:
    """Deterministic stringification of a tag value.

    Booleans: True / False (checked before int because bool is an int
    subclass in Python). Ints/floats: '%g'. Everything else: str()."""
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, (int, float)):
        return f"{v:g}"
    return str(v)


def build_config_tag(hp_pack: Optional[dict], seed_index: int,
                     sweep_id: Optional[int],
                     hparams: Optional[dict] = None,
                     tag_keys: Optional[list] = None) -> str:
    """Deterministic wandb-pasteable identifier for (sweep, per-slot config).

    The tag includes one ``key=value`` pair for every key in ``tag_keys`` --
    drawing per-slot values from ``hp_pack`` when the key is a pack key, else
    the shared value from ``hparams`` (typically the CLI argparse attributes
    for this slurm job). This lets the tag capture both easy (vmappable,
    per-slot) and hard (per-job, shared across the vmap) ablation axes. Seed
    and env are conventionally kept out of the tag at the call site so the
    tag groups runs across envs and seed replicas.
    """
    parts = []
    if tag_keys:
        for k in sorted(tag_keys):
            if hp_pack is not None and k in hp_pack:
                v = hp_pack[k][seed_index]
            elif hparams is not None and k in hparams:
                v = hparams[k]
            else:
                continue  # key not known in either source; skip silently
            parts.append(f"{k}={_format_tag_value(v)}")
    body = "_".join(parts) or "single"
    return f"sweep{sweep_id}_{body}" if sweep_id is not None else body


class WandbMultiSeedLogger:
    """One wandb.Run per vmap seed with batched per-seed scalar flushes.

    Each seed's run uses its own sample_step as the wandb x-axis, so we
    buffer pending scalars per seed; switching to a new step for a given
    seed flushes that seed's pending dict to its run.
    """

    def __init__(
        self,
        *,
        N: int,
        env_name: str,
        log_path: Path,
        wandb_names: Optional[List[str]],
        hp_pack: Optional[dict],
        sweep_id: Optional[int],
        hparams: dict,
        config_tag_keys: Optional[List[str]],
    ):
        self.N = int(N)
        self.env_name = env_name
        self.log_path = log_path
        self._wandb_names = wandb_names
        self._hp_pack = hp_pack
        self.sweep_id = sweep_id
        self.hparams = hparams
        self.config_tag_keys = config_tag_keys

        self._runs: List = []
        self._pending: List[dict] = [{} for _ in range(self.N)]
        self._pending_step: List[Optional[int]] = [None] * self.N
        self._array_accum: dict = {}
        # Set by trainer once the algorithm is constructed; controls the
        # x-axis used for per-level array tables (log2 SNR if available).
        self.snr = None

    def init_runs(self):
        base_name = self.log_path.name
        # Keep group = env name (its original semantics). sweep_id is logged
        # as a regular config field so filtering in wandb is
        # config.sweep_id == N.
        group = self.env_name
        for s in range(self.N):
            if s > 0 and _WANDB_INIT_STAGGER_MAX_SECONDS > 0:
                time.sleep(random.uniform(_WANDB_INIT_STAGGER_MIN_SECONDS,
                                          _WANDB_INIT_STAGGER_MAX_SECONDS))
            name = self._wandb_names[s] if self._wandb_names else f"{base_name}-s{s}"
            cfg = dict(self.hparams)
            cfg["seed_index"] = s
            cfg["parallel_seeds"] = self.N
            # Overwrite each hp_pack key's shared-CLI-default scalar with the
            # per-slot value, so wandb's filter / parallel-coordinates UI
            # reflects the actual hyperparameter this vmap slot is running.
            if self._hp_pack is not None:
                for k, values in self._hp_pack.items():
                    cfg[k] = values[s]
            # Drop the pack transport fields from wandb config -- the
            # per-slot key overwrites above already give wandb exactly the
            # same info and this keeps the logged config clean (no ~1KB
            # JSON blob on every run, no scratch-file paths).
            cfg.pop("hp_pack", None)
            cfg.pop("hp_pack_inline", None)
            if self.sweep_id is not None:
                cfg["sweep_id"] = int(self.sweep_id)
            # config_tag: automatic from whatever hps are in the pack
            # (excluding seed). Used as both the wandb-filter paste string
            # and the internal join key in analysis scripts. Unique per
            # (sweep, config).
            cfg["config_tag"] = build_config_tag(
                self._hp_pack, s, self.sweep_id,
                hparams=self.hparams, tag_keys=self.config_tag_keys,
            )
            # Honor WANDB_DIR when set by the sbatch wrapper (offline-mode
            # sweeps write to per-job netscratch dirs that a background
            # sync loop uploads to wandb.ai). Fall back to /tmp for the
            # local / online-mode path so unchanged setups still work.
            run = wandb.init(
                project="diffusion_online_rl",
                name=name,
                dir=os.environ.get("WANDB_DIR", "/tmp"),
                group=group,
                config=cfg,
                reinit="create_new",
                settings=wandb.Settings(console="off"),
            )
            self._runs.append(run)

    def set_snr(self, snr):
        """Provide diffusion SNR per timestep for log2-SNR-keyed array tables."""
        self.snr = snr

    # ------------------------------------------------------------------
    # Scalar logging
    # ------------------------------------------------------------------
    def add_scalar_per_seed(self, seed: int, tag: str, value: float,
                            step: Optional[int] = None):
        if step is not None:
            # Flush if the step changed.
            if self._pending_step[seed] is not None and step != self._pending_step[seed]:
                self.flush_seed(seed)
            self._pending_step[seed] = int(step)
        self._pending[seed][tag] = float(value)

    def _buffer_per_seed(self, seed: int, data: dict, step: int):
        if self._pending_step[seed] is not None and step != self._pending_step[seed]:
            self.flush_seed(seed)
        self._pending[seed].update(data)
        self._pending_step[seed] = int(step)

    def add_arrays_vmap(self, array_info: dict, sample_steps_per_seed: List[int]):
        """Log per-seed array metrics as wandb.Table with one row per level.

        ``array_info`` maps tag -> np.ndarray of shape [N, levels]. When
        ``self.snr`` is provided and matches ``levels``, the table x-axis is
        ``log2_snr``; otherwise it is integer ``level``.
        """
        if not array_info:
            return

        snr = self.snr
        if snr is not None:
            log2_snr = np.log2(np.maximum(snr, 1e-12))

        for s in range(self.N):
            step = int(sample_steps_per_seed[s])
            seed_arrays = {tag: np.asarray(value)[s] for tag, value in array_info.items()}

            for tag, value in seed_arrays.items():
                arr = np.asarray(value)
                if snr is not None and len(arr) == len(snr):
                    table = wandb.Table(
                        columns=["log2_snr", "value"],
                        data=[[float(log2_snr[i]), float(arr[i])] for i in range(len(arr))],
                    )
                else:
                    table = wandb.Table(
                        columns=["level", "value"],
                        data=[[int(i), float(arr[i])] for i in range(len(arr))],
                    )
                self._buffer_per_seed(s, {tag: table}, step)

    # ------------------------------------------------------------------
    # Update-step array accumulator
    # ------------------------------------------------------------------
    def accumulate_arrays(self, array_info: dict):
        for tag, vals in array_info.items():
            self._array_accum.setdefault(tag, []).append(np.asarray(vals))

    def flush_accumulated_arrays(self, sample_steps_per_seed: List[int]):
        """Average accumulated per-update arrays and log them. No-op if empty."""
        if not self._array_accum:
            return
        averaged = {
            tag: sum(v_list) / len(v_list)
            for tag, v_list in self._array_accum.items()
        }
        self._array_accum.clear()
        self.add_arrays_vmap(averaged, sample_steps_per_seed)

    # ------------------------------------------------------------------
    # Flushing
    # ------------------------------------------------------------------
    def flush_seed(self, s: int):
        if self._pending[s]:
            self._runs[s].log(self._pending[s], step=self._pending_step[s])
            self._pending[s] = {}
        self._pending_step[s] = None

    def flush_all(self):
        for s in range(self.N):
            self.flush_seed(s)

    def finish(self):
        for run in self._runs:
            try:
                run.finish()
            except Exception:
                pass
