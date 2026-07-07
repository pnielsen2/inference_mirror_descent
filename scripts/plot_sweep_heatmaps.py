#!/usr/bin/env python3
"""Render tail-return / fraction-of-baseline heatmaps from the local (offline)
sweep metrics produced by build_sweep_local_metrics.py.

Main figures (sweeps 89 + 90 + 91 combined):
  * one PNG per env -- raw tail-return mean, grid = gradient_space (rows) x
    guidance-strength/denoiser (cols), each panel a T x eta heatmap.
  * one overall PNG -- average fraction of baseline (2^log_score).

Advantage-normalization figure (sweep 92): a single PNG holding all 6 envs plus
overall, each a T x eta heatmap of fraction-of-baseline.

Axes (T, eta) are the union across the plotted sweeps so columns line up;
cells / panels with no finished run are left blank (grey). Unfinished configs
are already excluded upstream by the builder.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ENVS = ["Ant-v3", "HalfCheetah-v3", "Hopper-v3", "Humanoid-v3", "Swimmer-v3", "Walker2d-v3"]
SWEEPS_MAIN = [89, 90, 91]
SWEEP_ADVNORM = 92
GUIDANCE_GRADIENT_SPACES = ["xt", "x0hatclipped"]
IDENTITY_KEY = "Identity"
INCREASING_KEY = "increasing"
SCRIPT_DIR = Path(__file__).resolve().parent
TOPSIS_OUT_ROOT = SCRIPT_DIR / "topsis_out"


def _format_t(v: float) -> str:
    if np.isclose(v, 0.0):
        return "0"
    return f"{v:.3f}".rstrip("0").rstrip(".")


def _strength_key(denoiser, gsm_raw) -> str:
    if str(denoiser) == IDENTITY_KEY:
        return IDENTITY_KEY
    s = str(gsm_raw)
    if s == INCREASING_KEY:
        return INCREASING_KEY
    try:
        return _format_t(float(s))
    except (TypeError, ValueError):
        return s


def _strength_sort_key(sk: str):
    # numeric strengths first (ascending), then 'increasing', then 'Identity'.
    if sk == IDENTITY_KEY:
        return (2, 0.0)
    if sk == INCREASING_KEY:
        return (1, 0.0)
    try:
        return (0, float(sk))
    except ValueError:
        return (0, float("inf"))


def _load_sweep(sweep_id: int):
    d = TOPSIS_OUT_ROOT / f"sweep_{sweep_id}" / "local"
    ci = pd.read_csv(d / "config_index.csv")
    pe = pd.read_csv(d / "per_config_env_metrics.csv")
    ov = pd.read_csv(d / "overall_scores.csv")
    ci["strength_key"] = [
        _strength_key(dn, gs)
        for dn, gs in zip(ci["denoising_predictor"], ci["guidance_strength_multiplier"])
    ]
    ci["T"] = pd.to_numeric(ci["T"], errors="coerce")
    ci["eta"] = pd.to_numeric(ci["eta"], errors="coerce")
    return ci, pe, ov


def _load_main():
    cis, pes, ovs = [], [], []
    for sw in SWEEPS_MAIN:
        ci, pe, ov = _load_sweep(sw)
        cis.append(ci)
        pes.append(pe)
        ovs.append(ov)
    ci = pd.concat(cis, ignore_index=True)
    pe = pd.concat(pes, ignore_index=True)
    ov = pd.concat(ovs, ignore_index=True)
    axes_cols = ["config_tag", "T", "eta", "guidance_gradient_space", "strength_key"]
    env_df = pe.merge(ci[axes_cols], on="config_tag", how="inner")
    env_df["fraction_env"] = np.power(2.0, env_df["log_score"])
    overall_df = ov.merge(ci[axes_cols], on="config_tag", how="inner")
    return ci, env_df, overall_df


def _union_axes(env_df: pd.DataFrame, overall_df: pd.DataFrame):
    t_vals = pd.unique(pd.concat([env_df["T"], overall_df["T"]]).dropna())
    e_vals = pd.unique(pd.concat([env_df["eta"], overall_df["eta"]]).dropna())
    return sorted(float(x) for x in t_vals), sorted(float(x) for x in e_vals)


def _strength_keys_present(df: pd.DataFrame):
    keys = [k for k in df["strength_key"].dropna().unique()]
    return sorted(keys, key=_strength_sort_key)


def _build_matrix(df, value_col, ggs, strength_key, t_vals, e_vals):
    sub = df[(df["guidance_gradient_space"] == ggs) & (df["strength_key"] == strength_key)]
    if sub.empty:
        return np.full((len(t_vals), len(e_vals)), np.nan, dtype=float)
    pivot = sub.pivot_table(index="T", columns="eta", values=value_col, aggfunc="mean")
    pivot = pivot.reindex(index=t_vals, columns=e_vals)
    return pivot.to_numpy(dtype=float)


def _color_bounds(matrices):
    finite = [m[np.isfinite(m)] for m in matrices if np.isfinite(m).any()]
    if not finite:
        return 0.0, 1.0
    vals = np.concatenate(finite)
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmin == vmax:
        pad = 1.0 if vmin == 0 else abs(vmin) * 0.05
        return vmin - pad, vmax + pad
    return vmin, vmax


def _draw_panel(ax, matrix, vmin, vmax, value_fmt, title, t_vals, e_vals):
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#d9d9d9")
    im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", aspect="equal")
    ax.set_xticks(range(len(e_vals)), [str(int(v)) if float(v).is_integer() else str(v) for v in e_vals])
    ax.set_yticks(range(len(t_vals)), [_format_t(v) for v in t_vals])
    ax.set_xlabel("eta")
    ax.set_ylabel("T")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(-0.5, len(e_vals), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(t_vals), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if not np.isfinite(val):
                continue
            norm = 0.5 if vmax == vmin else (val - vmin) / (vmax - vmin)
            color = "black" if 0.2 <= norm <= 0.8 else "white"
            ax.text(j, i, format(val, value_fmt), ha="center", va="center", fontsize=7.5, color=color)
    return im


def _strength_title(sk: str) -> str:
    if sk == IDENTITY_KEY:
        return "denoiser=Identity"
    if sk == INCREASING_KEY:
        return "strength=increasing"
    return f"strength={sk}"


def _plot_main_figure(df, value_col, value_fmt, suptitle, colorbar_label, out_path,
                      strength_keys, t_vals, e_vals):
    matrices = [
        _build_matrix(df, value_col, ggs, sk, t_vals, e_vals)
        for ggs in GUIDANCE_GRADIENT_SPACES for sk in strength_keys
    ]
    vmin, vmax = _color_bounds(matrices)
    nrow, ncol = len(GUIDANCE_GRADIENT_SPACES), len(strength_keys)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 4.2 * nrow), constrained_layout=True, squeeze=False)
    last_im = None
    for r, ggs in enumerate(GUIDANCE_GRADIENT_SPACES):
        for c, sk in enumerate(strength_keys):
            matrix = _build_matrix(df, value_col, ggs, sk, t_vals, e_vals)
            title = f"grad={ggs}, {_strength_title(sk)}"
            last_im = _draw_panel(axes[r][c], matrix, vmin, vmax, value_fmt, title, t_vals, e_vals)
    fig.suptitle(suptitle, fontsize=15)
    cbar = fig.colorbar(last_im, ax=axes, shrink=0.9)
    cbar.set_label(colorbar_label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_advnorm_figure(env_df, overall_df, suptitle, out_path, t_vals, e_vals):
    """One figure: overall + 6 envs, each a single T x eta heatmap of
    fraction-of-baseline, on a shared colorbar."""
    panels = [("Overall", overall_df, "fraction_of_baseline")]
    for env in ENVS:
        panels.append((env, env_df[env_df["env"] == env], "fraction_env"))
    mats = []
    for _, sub, col in panels:
        piv = sub.pivot_table(index="T", columns="eta", values=col, aggfunc="mean")
        mats.append(piv.reindex(index=t_vals, columns=e_vals).to_numpy(dtype=float))
    vmin, vmax = _color_bounds(mats)
    fig, axes = plt.subplots(2, 4, figsize=(4.0 * 4, 4.4 * 2), constrained_layout=True)
    axes = axes.ravel()
    last_im = None
    for idx, ((label, _, _), matrix) in enumerate(zip(panels, mats)):
        last_im = _draw_panel(axes[idx], matrix, vmin, vmax, ".2f", label, t_vals, e_vals)
    for idx in range(len(panels), len(axes)):
        axes[idx].axis("off")
    fig.suptitle(suptitle, fontsize=15)
    cbar = fig.colorbar(last_im, ax=axes.tolist(), shrink=0.9)
    cbar.set_label("Fraction of baseline (2^log_score)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=TOPSIS_OUT_ROOT / "combined" / "heatmaps")
    args = parser.parse_args()
    out_dir = args.out_dir

    # ---- Main figures: sweeps 89 + 90 + 91 ----
    _, env_df, overall_df = _load_main()
    t_vals, e_vals = _union_axes(env_df, overall_df)
    strength_keys = _strength_keys_present(pd.concat([env_df, overall_df], ignore_index=True))
    print(f"Main strengths: {strength_keys}")
    print(f"Main T axis: {t_vals}")
    print(f"Main eta axis: {e_vals}")

    outputs = []
    for env in ENVS:
        out_path = out_dir / f"{env.replace('-', '_')}_tail_return_heatmap.png"
        _plot_main_figure(
            env_df[env_df["env"] == env], "mean", ".0f",
            suptitle=f"{env}: tail return mean across runs",
            colorbar_label="Average episode return in final 50k env steps",
            out_path=out_path, strength_keys=strength_keys, t_vals=t_vals, e_vals=e_vals,
        )
        outputs.append(out_path)

    overall_out = out_dir / "overall_fraction_of_baseline_heatmap.png"
    _plot_main_figure(
        overall_df, "fraction_of_baseline", ".3f",
        suptitle="Average fraction of baseline",
        colorbar_label="Average fraction of baseline (2^log_score)",
        out_path=overall_out, strength_keys=strength_keys, t_vals=t_vals, e_vals=e_vals,
    )
    outputs.append(overall_out)

    # ---- Advantage-normalization figure: sweep 92 ----
    ci92, pe92, ov92 = _load_sweep(SWEEP_ADVNORM)
    axes_cols = ["config_tag", "T", "eta", "guidance_gradient_space", "strength_key"]
    env92 = pe92.merge(ci92[axes_cols], on="config_tag", how="inner")
    env92["fraction_env"] = np.power(2.0, env92["log_score"])
    ov92 = ov92.merge(ci92[axes_cols], on="config_tag", how="inner")
    t92 = sorted(float(x) for x in pd.unique(env92["T"].dropna()))
    e92 = sorted(float(x) for x in pd.unique(env92["eta"].dropna()))
    adv_out = out_dir / "advantage_normalization_heatmap.png"
    _plot_advnorm_figure(
        env92, ov92,
        suptitle="Advantage normalization (Identity, grad=xt): fraction of baseline",
        out_path=adv_out, t_vals=t92, e_vals=e92,
    )
    outputs.append(adv_out)

    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
