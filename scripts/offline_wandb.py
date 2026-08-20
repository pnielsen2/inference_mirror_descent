#!/usr/bin/env python3
"""Read metric histories straight out of offline wandb run dirs (no network).

Two entry points used by the denoiser-gap analysis:

``index_runs(sweep_ids, envs=None)``
    Walk ``$WANDB_OFFLINE_BASE/sweep_<N>/job_*/wandb/offline-run-*`` and return
    a DataFrame with one row per offline run: its path plus the config fields
    needed to slice a sweep (env, config_tag, seed_index, denoiser, T, eta,
    guidance space / strength, ...). Only ``files/config.yaml`` is read, so this
    is cheap even for thousands of runs.

``read_history(run_dir, keys=None)``
    Scan the run's ``run-*.wandb`` transaction log and return a long-format
    DataFrame ``(step, key, value)`` of every scalar history point (optionally
    restricted to ``keys``).

``read_level_tables(run_dir, tag)``
    Load the per-diffusion-level ``wandb.Table`` JSONs that the trainer writes
    for ``MALA/acceptance_rate`` / ``MALA/clip_frac`` / ``MALA/eta_scale``,
    returning a DataFrame ``(step, level_or_log2_snr, value)``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import yaml

from relax.utils.fs import WANDB_OFFLINE_BASE

CONFIG_FIELDS = [
    "env",
    "config_tag",
    "seed_index",
    "denoising_predictor",
    "guidance_gradient_space",
    "guidance_strength_multiplier",
    "T",
    "eta",
    "alpha",
    "beta",
    "lr_q",
    "lr_policy",
    "mala_steps",
    "diffusion_steps",
    "x0_hat_clip_radius",
    "q_agg_sample",
    "num_vec_envs",
    "update_per_iteration",
    "total_step",
    "buffer_size",
    "batch_size",
    "reward_scale",
    "ema_advantage_normalization",
    "num_denoised_actions",
    "ema_within_advantage_normalization",
    "estimate_s_hat",
    "lr_anneal",
    "orthogonal_init",
    "fused_denoising",
    "sweep_id",
    "seed",
]

_TABLE_STEP_RE = re.compile(r"_(\d+)_[0-9a-f]{20}\.table\.json$")


def _cfg_value(node):
    if isinstance(node, dict) and "value" in node:
        return node["value"]
    return node


def _read_config(run_dir: Path) -> Optional[dict]:
    p = run_dir / "files" / "config.yaml"
    if not p.exists():
        return None
    try:
        with open(p) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return None
    out = {k: _cfg_value(cfg.get(k)) for k in CONFIG_FIELDS}
    out["run_dir"] = str(run_dir)
    out["run_id"] = run_dir.name.split("-")[-1]
    return out


def index_runs(sweep_ids: Sequence[int], envs: Optional[Iterable[str]] = None,
               base: Path = WANDB_OFFLINE_BASE) -> pd.DataFrame:
    envs = set(envs) if envs is not None else None
    rows = []
    for sw in sweep_ids:
        for job in sorted((base / f"sweep_{sw}").glob("job_*")):
            for rd in sorted(job.glob("wandb/offline-run-*")):
                cfg = _read_config(rd)
                if cfg is None:
                    continue
                if envs is not None and cfg.get("env") not in envs:
                    continue
                cfg["sweep"] = sw
                cfg["job"] = job.name
                rows.append(cfg)
    df = pd.DataFrame(rows)
    for c in ("T", "eta", "alpha", "beta", "lr_q", "total_step"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def read_summary(run_dir) -> dict:
    """Final value of every scalar metric, from ``files/wandb-summary.json``.

    Orders of magnitude cheaper than :func:`read_history` (one small JSON vs a
    full scan of the run's transaction log), at the cost of giving only the last
    logged value rather than the whole trajectory. Non-scalar entries (the
    per-diffusion-level ``wandb.Table`` refs) are dropped.
    """
    p = Path(run_dir) / "files" / "wandb-summary.json"
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            s = json.load(f) or {}
    except Exception:
        return {}
    return {k: v for k, v in s.items() if isinstance(v, (int, float))}


def index_runs_with_summary(sweep_ids: Sequence[int],
                            envs: Optional[Iterable[str]] = None,
                            base: Path = WANDB_OFFLINE_BASE) -> pd.DataFrame:
    """:func:`index_runs` joined with each run's final-metric summary."""
    idx = index_runs(sweep_ids, envs=envs, base=base)
    if idx.empty:
        return idx
    summ = pd.DataFrame([read_summary(rd) for rd in idx["run_dir"]], index=idx.index)
    return pd.concat([idx, summ], axis=1)


def read_history(run_dir, keys: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Long-format (step, key, value) scalar history from the .wandb log."""
    from wandb.proto import wandb_internal_pb2 as pb
    from wandb.sdk.internal import datastore

    run_dir = Path(run_dir)
    wandb_files = sorted(run_dir.glob("run-*.wandb"))
    if not wandb_files:
        return pd.DataFrame(columns=["step", "key", "value"])
    keys = set(keys) if keys is not None else None

    ds = datastore.DataStore()
    ds.open_for_scan(str(wandb_files[0]))
    recs = []
    while True:
        try:
            data = ds.scan_data()
        except Exception:
            break
        if data is None:
            break
        rec = pb.Record()
        try:
            rec.ParseFromString(data)
        except Exception:
            continue
        if rec.WhichOneof("record_type") != "history":
            continue
        step = None
        vals = {}
        for item in rec.history.item:
            k = item.key or ".".join(item.nested_key)
            if k == "_step":
                step = int(json.loads(item.value_json))
                continue
            if k.startswith("_") or "." in k:
                continue
            if keys is not None and k not in keys:
                continue
            try:
                v = json.loads(item.value_json)
            except Exception:
                continue
            if isinstance(v, (int, float)):
                vals[k] = float(v)
        if step is None or not vals:
            continue
        for k, v in vals.items():
            recs.append((step, k, v))
    return pd.DataFrame(recs, columns=["step", "key", "value"])


def read_level_tables(run_dir, tag: str) -> pd.DataFrame:
    """Per-level table history for e.g. ``MALA/acceptance_rate``.

    Returns (step, level, log2_snr, value); ``level`` is the row index and
    ``log2_snr`` the logged x column when present.
    """
    run_dir = Path(run_dir)
    # wandb flattens the tag to <media>/table/<dirs>/<leaf>_<step>_<hash>.table.json
    tag_path = Path(tag)
    tdir = run_dir / "files" / "media" / "table" / tag_path.parent
    rows = []
    for p in sorted(tdir.glob(f"{tag_path.name}_*.table.json")):
        m = _TABLE_STEP_RE.search(p.name)
        if not m:
            continue
        step = int(m.group(1))
        try:
            with open(p) as f:
                tbl = json.load(f)
        except Exception:
            continue
        cols = tbl.get("columns", [])
        for i, row in enumerate(tbl.get("data", [])):
            d = dict(zip(cols, row))
            rows.append((step, i, d.get("log2_snr", np.nan), d.get("value", np.nan)))
    return pd.DataFrame(rows, columns=["step", "level", "log2_snr", "value"]).sort_values(
        ["step", "level"]
    )
