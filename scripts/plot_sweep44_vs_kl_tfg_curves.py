import argparse
import colorsys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from plot_sweep_6env_training_curves import (
    _config_tag_from_args,
    _locate_local_csv,
    _parse_running_command,
    _read_local_slurm_headers,
)
from plot_sweep_all_runs_training_curves import (
    FIG_DIR,
    aggregate_curves,
    windowed_mean_curve,
)

ENV_ORDER = [
    "Ant-v3",
    "Hopper-v3",
    "Humanoid-v3",
    "Swimmer-v3",
    "Walker2d-v3",
]
BASELINE_COLOR = "#111827"
TURQUOISE_BASE = "#0f766e"
PURPLE_BASE = "#7e22ce"
BASELINE_SWEEP_ID = 44
KL_SWEEP_ID = 45
TFG_SWEEP_ID = 48
BASELINE_CFG_TAG = "sweep44_single"
KL_VALUES = [16, 64, 256, 1024, 4096]
TFG_VALUES = [16, 32, 64, 128, 256]
TAIL_STEPS = 50_000
SCRIPT_DIR = Path(__file__).parent.resolve()
REPO_DIR = SCRIPT_DIR.parent
SLURM_ROOT = REPO_DIR / "logs" / "slurm"
ADAPTIVE_ETA_LABEL = r"Adaptive $\eta$"
KL_BUDGET_LABEL = "KL Budget"
FIXED_ETA_LABEL = r"Fixed $\eta$"
HUMANOID_ENV = "Humanoid-v3"


def series_specs():
    specs = [("baseline", None, BASELINE_SWEEP_ID, BASELINE_CFG_TAG)]
    specs.extend(("kl", value, KL_SWEEP_ID, f"sweep{KL_SWEEP_ID}_kl_budget={value:g}") for value in KL_VALUES)
    specs.extend(("tfg", value, TFG_SWEEP_ID, f"sweep{TFG_SWEEP_ID}_tfg_eta={value:g}") for value in TFG_VALUES)
    return specs


def load_csv_histories(csv_path, env, csv_cache):
    if csv_path in csv_cache:
        return csv_cache[csv_path]
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        csv_cache[csv_path] = None
        return None
    if "seed" not in df.columns or "step" not in df.columns:
        csv_cache[csv_path] = None
        return None
    value_col = f"episode_return/{env}"
    if value_col not in df.columns:
        candidates = [c for c in df.columns if c.startswith("episode_return/")]
        if len(candidates) != 1:
            csv_cache[csv_path] = None
            return None
        value_col = candidates[0]
    seed_series = df["seed"].astype(int)
    hist_by_seed = {}
    for seed_index_val in sorted(seed_series.dropna().astype(int).unique().tolist()):
        hist = (
            df[seed_series == int(seed_index_val)][["step", value_col]]
            .rename(columns={"step": "_step", value_col: "return"})
            .sort_values("_step")
            .reset_index(drop=True)
        )
        if len(hist) == 0:
            continue
        hist_by_seed[int(seed_index_val)] = hist
    csv_cache[csv_path] = hist_by_seed
    return hist_by_seed


def load_local_histories(specs, exclude_envs):
    selected = []
    wanted = {(sweep_id, cfg_tag): (kind, value) for kind, value, sweep_id, cfg_tag in specs}
    per_tag_counts = defaultdict(int)
    csv_cache = {}
    seen = set()

    if not SLURM_ROOT.exists():
        return selected

    for slurm_out in SLURM_ROOT.rglob("*.out"):
        running, started_at = _read_local_slurm_headers(slurm_out)
        if running is None:
            continue
        args = _parse_running_command(running)
        sweep_id = args.get("sweep_id")
        env = args.get("env")
        if env is None or env in exclude_envs or env not in ENV_ORDER:
            continue
        if not any(key_sweep_id == sweep_id for key_sweep_id, _ in wanted):
            continue
        csv_path = _locate_local_csv(args, started_at)
        if csv_path is None:
            continue
        hist_by_seed = load_csv_histories(csv_path, env, csv_cache)
        if not hist_by_seed:
            continue
        for seed_index_val, hist in hist_by_seed.items():
            cfg_tag = _config_tag_from_args(args, seed_index_val)
            spec = wanted.get((sweep_id, cfg_tag))
            if spec is None:
                continue
            kind, value = spec
            dedupe_key = (kind, value, env, csv_path, int(seed_index_val))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            per_tag_counts[cfg_tag] += 1
            run_id = f"local:{csv_path}:{seed_index_val}"
            selected.append((kind, value, env, run_id, hist))

    for _, _, _, cfg_tag in specs:
        print(f"Loaded {per_tag_counts[cfg_tag]} local histories for {cfg_tag}", flush=True)
    return selected


def stretch_hist_steps_to_target(hist, target_max_step):
    if hist is None or len(hist) == 0:
        return hist
    current_max = float(hist["_step"].max())
    if not np.isfinite(current_max) or current_max <= 0 or current_max >= float(target_max_step):
        return hist
    scaled = hist.copy()
    scale = float(target_max_step) / current_max
    scaled["_step"] = np.rint(scaled["_step"].to_numpy(dtype=float) * scale).astype(int)
    return scaled.sort_values("_step").reset_index(drop=True)


def maybe_adjust_hist_for_plot(hist, kind, env, target_max_step, stretch_kl_humanoid):
    if not stretch_kl_humanoid or kind != "kl" or env != HUMANOID_ENV:
        return hist
    return stretch_hist_steps_to_target(hist, target_max_step)


def palette_from_base(base_color, values, min_saturation, max_saturation, max_lightness, min_lightness):
    if not values:
        return {}
    rgb = to_rgb(base_color)
    hue, _, _ = colorsys.rgb_to_hls(*rgb)
    ordered = sorted(values)
    denom = max(len(ordered) - 1, 1)
    palette = {}
    for idx, value in enumerate(ordered):
        frac = float(idx) / float(denom)
        saturation = min_saturation + frac * (max_saturation - min_saturation)
        lightness = max_lightness + frac * (min_lightness - max_lightness)
        palette[value] = colorsys.hls_to_rgb(hue, lightness, saturation)
    return palette


def label_for(kind, value):
    if kind == "baseline":
        return ADAPTIVE_ETA_LABEL
    if kind == "kl":
        return f"{KL_BUDGET_LABEL}={value:g}"
    if kind == "tfg":
        return rf"{FIXED_ETA_LABEL}={value:g}"
    raise ValueError(kind)


def line_order(kl_values, tfg_values):
    ordered = [("baseline", None)]
    ordered.extend(("kl", value) for value in sorted(kl_values))
    ordered.extend(("tfg", value) for value in sorted(tfg_values))
    return ordered


def color_map(kl_values, tfg_values):
    mapping = {("baseline", None): BASELINE_COLOR}
    mapping.update({("kl", value): color for value, color in palette_from_base(PURPLE_BASE, kl_values, 0.18, 0.92, 0.82, 0.42).items()})
    mapping.update({("tfg", value): color for value, color in palette_from_base(TURQUOISE_BASE, tfg_values, 0.2, 0.95, 0.82, 0.36).items()})
    return mapping


def aggregate_series(selected_runs, bins, bin_size, ci_level, max_steps, stretch_kl_humanoid=False):
    histories = defaultdict(list)
    for kind, value, env, run_id, hist in selected_runs:
        if hist is None or len(hist) == 0:
            print(f"  {env:<12s} {label_for(kind, value):<18s} {run_id} NO_HISTORY", flush=True)
            continue
        plot_hist = maybe_adjust_hist_for_plot(hist, kind, env, max_steps, stretch_kl_humanoid)
        if plot_hist is not hist:
            print(
                f"  {env:<12s} {label_for(kind, value):<18s} {run_id} "
                f"PLOT_ONLY_X_STRETCH {int(hist['_step'].max())}->{int(plot_hist['_step'].max())}",
                flush=True,
            )
        histories[(kind, value, env)].append(plot_hist)

    aggregates = {}
    counts_by_series = defaultdict(lambda: defaultdict(int))
    for key, env_hists in histories.items():
        kind, value, env = key
        curves = [windowed_mean_curve(hist, bins, bin_size) for hist in env_hists]
        mean, ci, counts = aggregate_curves(curves, ci_level)
        aggregates[key] = (mean, ci, counts, len(env_hists))
        counts_by_series[(kind, value)][env] = len(env_hists)
    return aggregates, counts_by_series


def tail_mean_return(hist, tail_steps):
    steps = hist["_step"].to_numpy()
    returns = hist["return"].to_numpy()
    if len(steps) == 0:
        return np.nan
    cutoff = steps.max() - tail_steps
    mask = steps >= cutoff
    if not mask.any():
        return np.nan
    return float(np.mean(returns[mask]))


def collect_tail_scores(selected_runs, tail_steps):
    tail_scores = defaultdict(lambda: defaultdict(list))
    for kind, value, env, run_id, hist in selected_runs:
        score = tail_mean_return(hist, tail_steps)
        if np.isnan(score):
            print(f"  {env:<12s} {label_for(kind, value):<14s} {run_id} NO_TAIL_SCORE", flush=True)
            continue
        tail_scores[(kind, value)][env].append(score)
    return tail_scores


def geometric_mean(values):
    arr = np.asarray(values, dtype=float)
    if np.any(arr <= 0):
        raise ValueError(f"geometric mean requires positive values, got {arr}")
    return float(np.exp(np.mean(np.log(arr))))


def bootstrap_bar_stats(tail_scores, ordered_lines, env_order, bootstrap_samples, ci_level, bootstrap_seed):
    rng = np.random.default_rng(bootstrap_seed)
    stats_by_series = {}
    alpha = 1.0 - ci_level
    for kind, value in ordered_lines:
        series_key = (kind, value)
        env_scores = tail_scores.get(series_key, {})
        missing = [env for env in env_order if not env_scores.get(env)]
        if missing:
            print(f"Skipping {label_for(kind, value)} bar; missing envs {missing}", flush=True)
            continue
        env_means = [float(np.mean(env_scores[env])) for env in env_order]
        point = geometric_mean(env_means)
        boot = np.empty(bootstrap_samples, dtype=float)
        for i in range(bootstrap_samples):
            sampled_env_means = []
            for env in env_order:
                scores = np.asarray(env_scores[env], dtype=float)
                sampled = rng.choice(scores, size=len(scores), replace=True)
                sampled_env_means.append(float(np.mean(sampled)))
            boot[i] = geometric_mean(sampled_env_means)
        lo, hi = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
        stats_by_series[series_key] = {
            "point": point,
            "lo": float(lo),
            "hi": float(hi),
            "env_means": {env: float(np.mean(env_scores[env])) for env in env_order},
            "env_counts": {env: len(env_scores[env]) for env in env_order},
        }
    return stats_by_series


def save_curve_grid(aggregates, ordered_lines, colors, bins, max_steps, out_path):
    fig, axes = plt.subplots(1, len(ENV_ORDER), figsize=(4.6 * len(ENV_ORDER), 4.6), squeeze=False, sharex=True)
    axes = axes.flatten()

    for idx, env in enumerate(ENV_ORDER):
        ax = axes[idx]
        for kind, value in ordered_lines:
            key = (kind, value, env)
            if key not in aggregates:
                continue
            mean, ci, counts, n_runs = aggregates[key]
            valid = ~np.isnan(mean)
            if not valid.any():
                continue
            x = bins / 1e6
            linewidth = 2.4 if kind == "baseline" else 1.7
            color = colors[(kind, value)]
            ax.plot(x[valid], mean[valid], color=color, linewidth=linewidth, label=label_for(kind, value))
            band_mask = valid & ~np.isnan(ci) & (counts == n_runs)
            if band_mask.any():
                alpha = 0.12 if kind == "baseline" else 0.055
                ax.fill_between(x[band_mask], mean[band_mask] - ci[band_mask], mean[band_mask] + ci[band_mask], color=color, alpha=alpha)
        ax.set_title(env.replace("-v3", ""), fontsize=11)
        ax.set_xlabel("Env steps (M)", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Episode return", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=9)
        ax.set_xlim(0, max_steps / 1e6)

    handles = [Line2D([0], [0], color=colors[(kind, value)], linewidth=(2.4 if kind == "baseline" else 1.9), label=label_for(kind, value)) for kind, value in ordered_lines]
    legend_cols = 4 if len(handles) > 8 else min(6, max(1, len(handles)))
    fig.legend(handles=handles, loc="lower center", ncol=legend_cols, fontsize=9, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout()
    plt.subplots_adjust(bottom=(0.24 if len(handles) > 8 else 0.19))
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}", flush=True)
    plt.close(fig)


def save_bar_chart(bar_stats, ordered_lines, colors, out_path, ci_level):
    plotted = [series_key for series_key in ordered_lines if series_key in bar_stats]
    labels = [label_for(kind, value) for kind, value in plotted]
    heights = np.asarray([bar_stats[series_key]["point"] for series_key in plotted], dtype=float)
    lowers = np.asarray([bar_stats[series_key]["point"] - bar_stats[series_key]["lo"] for series_key in plotted], dtype=float)
    uppers = np.asarray([bar_stats[series_key]["hi"] - bar_stats[series_key]["point"] for series_key in plotted], dtype=float)
    colors_list = [colors[series_key] for series_key in plotted]

    fig_width = max(12.5, 0.95 * len(plotted))
    fig, ax = plt.subplots(figsize=(fig_width, 5.2))
    x = np.arange(len(plotted), dtype=float)
    ax.bar(x, heights, color=colors_list, width=0.78, alpha=0.95)
    ax.errorbar(x, heights, yerr=np.vstack([lowers, uppers]), fmt="none", ecolor="#111827", elinewidth=1.2, capsize=4, capthick=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Geometric mean of per-env last-50k mean return", fontsize=10)
    ax.set_title(f"End-of-training geometric mean across environments ({int(ci_level * 100)}% bootstrap CI)", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=8.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}", flush=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-size", type=int, default=10_000)
    ap.add_argument("--ci-level", type=float, default=0.90)
    ap.add_argument("--max-steps", type=int, default=1_000_000)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--bar-out", type=Path, default=None)
    ap.add_argument("--tail-steps", type=int, default=TAIL_STEPS)
    ap.add_argument("--bootstrap-samples", type=int, default=1000)
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--stretch-incomplete-kl-humanoid", action="store_true")
    args = ap.parse_args()

    exclude_envs = {"HalfCheetah-v3"}
    bins = np.arange(args.bin_size, args.max_steps + args.bin_size, args.bin_size, dtype=int)

    all_runs = load_local_histories(series_specs(), exclude_envs)
    if not all_runs:
        raise SystemExit("No runs selected for plotting")

    kl_values = sorted({value for kind, value, _, _, _ in all_runs if kind == "kl" and value is not None})
    tfg_values = sorted({value for kind, value, _, _, _ in all_runs if kind == "tfg" and value is not None})
    ordered_lines = line_order(kl_values, tfg_values)
    colors = color_map(kl_values, tfg_values)

    aggregates, counts_by_series = aggregate_series(
        all_runs,
        bins,
        args.bin_size,
        args.ci_level,
        args.max_steps,
        stretch_kl_humanoid=args.stretch_incomplete_kl_humanoid,
    )
    tail_scores = collect_tail_scores(all_runs, args.tail_steps)
    bar_stats = bootstrap_bar_stats(tail_scores, ordered_lines, ENV_ORDER, args.bootstrap_samples, args.ci_level, args.bootstrap_seed)

    for kind, value in ordered_lines:
        label = label_for(kind, value)
        env_counts = counts_by_series.get((kind, value), {})
        print(f"{label}: {dict(env_counts)}", flush=True)
        if (kind, value) in bar_stats:
            stats = bar_stats[(kind, value)]
            print(f"  {label} geometric_mean={stats['point']:.3f} ci90=[{stats['lo']:.3f}, {stats['hi']:.3f}]", flush=True)

    FIG_DIR.mkdir(exist_ok=True)
    out_path = args.out or (FIG_DIR / "training_curves_sweep44_vs_sweep45_kl_vs_sweep48_tfg.png")
    bar_out_path = args.bar_out or out_path.with_name(f"{out_path.stem}_geomean_bar{out_path.suffix}")
    save_curve_grid(aggregates, ordered_lines, colors, bins, args.max_steps, out_path)
    save_bar_chart(bar_stats, ordered_lines, colors, bar_out_path, args.ci_level)


if __name__ == "__main__":
    main()
