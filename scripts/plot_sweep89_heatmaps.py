#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ENVS = ["Ant-v3", "HalfCheetah-v3", "Hopper-v3", "Humanoid-v3", "Swimmer-v3", "Walker2d-v3"]
ETA_VALUES = [4, 8, 16, 32, 64, 128]
T_VALUES = [0.0, 0.05, 0.1, 0.15, 0.2]
# Strength dimension: the two ablated DDPM_mean values, then Identity (which did
# not ablate guidance_strength_multiplier) treated as a final value at the end.
IDENTITY_KEY = "Identity"
STRENGTH_KEYS = ["0.2", "1", IDENTITY_KEY]
GUIDANCE_GRADIENT_SPACES = ["xt", "x0hatclipped"]
SCRIPT_DIR = Path(__file__).resolve().parent
TOPSIS_OUT_ROOT = SCRIPT_DIR / "topsis_out"


def _add_strength_key(config_index: pd.DataFrame) -> pd.DataFrame:
    out = config_index.copy()
    out["T"] = pd.to_numeric(out["T"], errors="coerce")
    out["eta"] = pd.to_numeric(out["eta"], errors="coerce")
    out["guidance_strength_multiplier"] = pd.to_numeric(out["guidance_strength_multiplier"], errors="coerce")
    is_identity = out["denoising_predictor"].astype(str) == IDENTITY_KEY
    numeric_label = out["guidance_strength_multiplier"].map(
        lambda v: _format_t(v) if pd.notna(v) else ""
    )
    out["strength_key"] = np.where(is_identity, IDENTITY_KEY, numeric_label)
    return out


def _build_matrix(df: pd.DataFrame, value_col: str, guidance_gradient_space: str, strength_key: str) -> np.ndarray:
    sub = df[
        (df["guidance_gradient_space"] == guidance_gradient_space)
        & (df["strength_key"] == strength_key)
    ].copy()
    if sub.empty:
        return np.full((len(T_VALUES), len(ETA_VALUES)), np.nan, dtype=float)
    pivot = sub.pivot_table(index="T", columns="eta", values=value_col, aggfunc="mean")
    pivot = pivot.reindex(index=T_VALUES, columns=ETA_VALUES)
    return pivot.to_numpy(dtype=float)


def _color_bounds(matrices: list[np.ndarray]) -> tuple[float, float]:
    vals = np.concatenate([m[np.isfinite(m)] for m in matrices if np.isfinite(m).any()]) if any(np.isfinite(m).any() for m in matrices) else np.array([])
    if vals.size == 0:
        return 0.0, 1.0
    vmin = float(vals.min())
    vmax = float(vals.max())
    if vmin == vmax:
        pad = 1.0 if vmin == 0 else max(1.0, abs(vmin) * 0.05)
        return vmin - pad, vmax + pad
    return vmin, vmax


def _format_t(v: float) -> str:
    if np.isclose(v, 0.0):
        return "0"
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _draw_panel(ax, matrix: np.ndarray, vmin: float, vmax: float, value_fmt: str, title: str):
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#d9d9d9")
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", aspect="equal")
    ax.set_xticks(range(len(ETA_VALUES)), [str(v) for v in ETA_VALUES])
    ax.set_yticks(range(len(T_VALUES)), [_format_t(v) for v in T_VALUES])
    ax.set_xlabel("eta")
    ax.set_ylabel("T")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(np.arange(-0.5, len(ETA_VALUES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(T_VALUES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text = "NA" if not np.isfinite(val) else format(val, value_fmt)
            color = "black"
            if np.isfinite(val):
                norm = 0.5 if vmax == vmin else (val - vmin) / (vmax - vmin)
                color = "black" if 0.2 <= norm <= 0.8 else "white"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color=color)
    return im


def _strength_title(strength_key: str) -> str:
    if strength_key == IDENTITY_KEY:
        return "denoiser=Identity"
    return f"guidance_strength_multiplier={strength_key}"


def _plot_figure(df: pd.DataFrame, value_col: str, value_fmt: str, suptitle: str, colorbar_label: str, out_path: Path):
    matrices = []
    for guidance_gradient_space in GUIDANCE_GRADIENT_SPACES:
        for strength_key in STRENGTH_KEYS:
            matrices.append(_build_matrix(df, value_col, guidance_gradient_space, strength_key))
    vmin, vmax = _color_bounds(matrices)
    fig, axes = plt.subplots(
        len(GUIDANCE_GRADIENT_SPACES), len(STRENGTH_KEYS), figsize=(20, 10), constrained_layout=True
    )
    last_im = None
    for row_idx, guidance_gradient_space in enumerate(GUIDANCE_GRADIENT_SPACES):
        for col_idx, strength_key in enumerate(STRENGTH_KEYS):
            ax = axes[row_idx, col_idx]
            matrix = _build_matrix(df, value_col, guidance_gradient_space, strength_key)
            title = f"guidance_gradient_space={guidance_gradient_space}, {_strength_title(strength_key)}"
            last_im = _draw_panel(ax, matrix, vmin, vmax, value_fmt, title)
    fig.suptitle(suptitle, fontsize=16)
    cbar = fig.colorbar(last_im, ax=axes, shrink=0.92)
    cbar.set_label(colorbar_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-id", type=int, default=89)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    sweep_dir = TOPSIS_OUT_ROOT / f"sweep_{args.sweep_id}"
    data_dir = sweep_dir / "local"
    out_dir = args.out_dir if args.out_dir is not None else sweep_dir / "heatmaps"

    config_index = pd.read_csv(data_dir / "config_index.csv")
    per_env_metrics = pd.read_csv(data_dir / "per_config_env_metrics.csv")
    overall_scores = pd.read_csv(data_dir / "overall_scores.csv")

    config_index = _add_strength_key(config_index)
    axes_cols = ["config_tag", "T", "eta", "guidance_gradient_space", "strength_key"]

    env_plot_df = per_env_metrics.merge(config_index[axes_cols], on="config_tag", how="inner")
    overall_plot_df = overall_scores.merge(config_index[axes_cols], on="config_tag", how="inner")

    outputs = []
    for env in ENVS:
        env_df = env_plot_df[env_plot_df["env"] == env].copy()
        out_path = out_dir / f"{env.replace('-', '_')}_tail_return_heatmap.png"
        _plot_figure(
            df=env_df,
            value_col="mean",
            value_fmt=".1f",
            suptitle=f"{env}: tail return mean across runs",
            colorbar_label="Average episode return in final 50k env steps",
            out_path=out_path,
        )
        outputs.append(out_path)

    overall_out_path = out_dir / "overall_fraction_of_baseline_heatmap.png"
    _plot_figure(
        df=overall_plot_df,
        value_col="fraction_of_baseline",
        value_fmt=".3f",
        suptitle="Average fraction of baseline",
        colorbar_label="Average fraction of baseline (2^log_score)",
        out_path=overall_out_path,
    )
    outputs.append(overall_out_path)

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
