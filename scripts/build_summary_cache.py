#!/usr/bin/env python3
"""Cache config + final-metric summary for every offline run into one parquet.

``wandb-summary.json`` holds the last logged value of every scalar metric, so a
whole sweep's diagnostics can be indexed without touching the (much larger)
``run-*.wandb`` transaction logs. Run this once per new sweep; downstream
analysis then loads a single parquet.

    python -m scripts.build_summary_cache --sweeps 89 90 91 94 104 105 112 114 117 118
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.offline_wandb import index_runs_with_summary

OUT = Path("analysis_cache/run_summaries.parquet")
RENAME = {
    "losses/Policy_epsilon_MSE": "epsMSE",
    "losses/Q_loss": "Qloss",
    "actions/action_var": "act_var",
    "actions/action_mean": "act_mean",
    "actions/final_action_clip_frac": "clipfrac",
    "Critic/adv_norm_running_std": "adv_std",
    "Critic/adv_norm_running_mean": "adv_mean",
    "Critic/average_Q": "Q_avg",
    "Critic/Q_agg_tilt_bias": "Q_tilt_bias",
    "Global_EMAs/beta": "beta_logged",
    "episodes/episode_length": "ep_len",
    "_step": "step",
    "_runtime": "runtime",
}


def build(sweeps, out: Path = OUT) -> pd.DataFrame:
    frames = []
    for sw in sweeps:
        d = index_runs_with_summary([sw])
        if d.empty:
            print(f"  sweep_{sw}: 0 runs", flush=True)
            continue
        print(f"  sweep_{sw}: {len(d)} runs", flush=True)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)

    # Collapse the per-env episode_return/<env> columns into one `ret`.
    retcols = [c for c in df.columns if c.startswith("episode_return/")]
    env_arr = df["env"].to_numpy()
    ret = np.full(len(df), np.nan)
    for c in retcols:
        env_of_col = c.split("/", 1)[1]
        m = (env_arr == env_of_col) & df[c].notna().to_numpy()
        ret[m] = df.loc[m, c].to_numpy()
    df["ret"] = ret
    df = df.drop(columns=retcols)

    df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})
    for c in ("T", "eta", "alpha", "beta", "lr_q", "lr_policy", "total_step",
              "diffusion_steps", "mala_steps", "x0_hat_clip_radius"):
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Composite-MD effective coefficients: pi_new ~ pi_old^alpha exp(beta Q).
    df["alpha_eff"] = 1.0 / (1.0 + df.eta * df["T"])
    df["beta_eff"] = df.eta / (1.0 + df.eta * df["T"])
    df["frac_done"] = df.step / df.total_step

    # guidance_strength_multiplier mixes floats with the literal "increasing",
    # so object columns are stringified for a stable parquet schema.
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}  ({len(df)} runs, {len(df.columns)} cols)")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    build(ap.parse_args().sweeps, ap.parse_args().out)


if __name__ == "__main__":
    main()
