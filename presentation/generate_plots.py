#!/usr/bin/env python3
"""Generate all plots and statistics for the weekly presentation.

IMPORTANT: Some log directories contain event files from TWO different runs
(same dir was reused). We handle this by reading individual event files and
using hostname-based config overrides for known collision directories.
"""

import os, yaml, glob, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tbparse import SummaryReader
from scipy import stats
from collections import defaultdict

# ─── Paths ───────────────────────────────────────────────────────────────────
LOG_BASE = '/n/home09/pnielsen/inference_mirror_descent/logs/HalfCheetah-v4'
OUT_DIR  = '/n/home09/pnielsen/inference_mirror_descent/presentation/figures'
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Known collision directories ─────────────────────────────────────────────
# These directories had a tfg_end=4 run FIRST (on holygpu8a16302), then a
# tfg_end=2 run SECOND that overwrote config.yaml. Each dir has two event
# files from different hosts. The config.yaml says tfg_end=2.0 but the
# holygpu8a16302 event file is actually from the tfg_end=4.0 run.
COLLISION_DIRS = {
    'dpmd_2026-03-18_16-15-44_s0_': {'tfg_end4_host': 'holygpu8a16302', 'tfg_end2_host': 'holygpu8a16103'},
    'dpmd_2026-03-18_16-15-44_s1_': {'tfg_end4_host': 'holygpu8a16302', 'tfg_end2_host': 'holygpu8a16103'},
    'dpmd_2026-03-18_16-15-45_s2_': {'tfg_end4_host': 'holygpu8a16302', 'tfg_end2_host': 'holygpu8a16104'},
}

# ─── Style ───────────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 15,
    'legend.fontsize': 10, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
    'font.family': 'serif',
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.framealpha': 0.9, 'legend.edgecolor': '0.8',
})

C = {
    'base':       '#2171b5',
    'pol3e5':     '#238b45',
    'pol1e5':     '#006d2c',
    'pol3e6':     '#74c476',
    'tfg4':       '#fd8d3c',
    'tfg2':       '#e6550d',
    'tfg1':       '#8c6bb1',
    'mala4':      '#d94701',
    'mala8':      '#7a0177',
    'mala4tfg1':  '#fc4e2a',
    'mala8tfg1':  '#bd0026',
    'mala4tfg2':  '#fd8d3c',
    'mala8tfg2':  '#feb24c',
}

# ─── Data helpers ────────────────────────────────────────────────────────────
def get_event_host(event_path):
    """Extract short hostname from event file path."""
    parts = os.path.basename(event_path).split('.')
    # events.out.tfevents.TIMESTAMP.HOSTNAME.rc.fas.harvard.edu
    if len(parts) >= 5:
        return parts[4]  # e.g. 'holygpu8a16302'
    return 'unknown'


def scan_runs():
    """Scan recent run directories. For collision directories, split into
    separate runs per event file with correct config overrides."""
    dirs = sorted(glob.glob(os.path.join(LOG_BASE, 'dpmd_*')))
    runs = []
    for d in dirs:
        bname = os.path.basename(d)
        try:
            dp = bname.split('_')[1].split('-')
            if int(dp[1]) != 3 or int(dp[2]) < 13:
                continue
        except Exception:
            continue
        cfg_path = os.path.join(d, 'config.yaml')
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        evts = sorted(glob.glob(os.path.join(d, 'events.out.tfevents.*')))
        if not evts:
            continue

        # Check if this is a collision directory
        if bname in COLLISION_DIRS:
            info = COLLISION_DIRS[bname]
            for ef in evts:
                evsz = os.path.getsize(ef)
                if evsz < 50 * 1024:
                    continue
                host = get_event_host(ef)
                # Determine correct tfg_end based on hostname
                if info['tfg_end4_host'] in host:
                    tfg_end_override = 4.0
                elif info['tfg_end2_host'] in host:
                    tfg_end_override = 2.0
                else:
                    continue  # unknown host, skip
                runs.append(dict(
                    name=bname, path=d, event_file=ef,
                    seed=cfg.get('seed', -1),
                    mala=cfg.get('mala_steps', 2),
                    tfg_end=tfg_end_override,
                    lr_sched=cfg.get('lr_schedule_type', None),
                    lr_end=cfg.get('lr_schedule_end', None),
                    q_lr_sched=cfg.get('q_lr_schedule_type', None),
                    ddim=cfg.get('ddim_predictor', False),
                    mg=cfg.get('mala_guided_predictor', False),
                    tfg=cfg.get('tfg_eta', 16.0),
                    evsz=evsz,
                ))
        else:
            # Normal directory: pick largest event file
            best_ef = max(evts, key=os.path.getsize)
            evsz = os.path.getsize(best_ef)
            if evsz < 50 * 1024:
                continue
            runs.append(dict(
                name=bname, path=d, event_file=best_ef,
                seed=cfg.get('seed', -1),
                mala=cfg.get('mala_steps', 2),
                tfg_end=cfg.get('tfg_eta_end', None),
                lr_sched=cfg.get('lr_schedule_type', None),
                lr_end=cfg.get('lr_schedule_end', None),
                q_lr_sched=cfg.get('q_lr_schedule_type', None),
                ddim=cfg.get('ddim_predictor', False),
                mg=cfg.get('mala_guided_predictor', False),
                tfg=cfg.get('tfg_eta', 16.0),
                evsz=evsz,
            ))
    return runs


def best_per_seed(runs):
    """For each seed, pick the run with the largest event file."""
    by = defaultdict(list)
    for r in runs:
        by[r['seed']].append(r)
    return {s: max(rs, key=lambda x: x['evsz']) for s, rs in by.items()}


def load_tb(event_file, tags=('sample/episode_return', 'training/policy_loss')):
    """Load TB scalars from a SINGLE event file (not a directory)."""
    reader = SummaryReader(event_file)
    df = reader.scalars
    out = {}
    for tag in tags:
        sub = df[df['tag'] == tag].sort_values('step')
        if len(sub) > 0:
            out[tag] = (sub['step'].values.astype(float),
                        sub['value'].values.astype(float))
    return out


def load_group(runs, tags=('sample/episode_return', 'training/policy_loss')):
    """Load TB data for a group of runs (best per seed)."""
    bps = best_per_seed(runs)
    return {s: load_tb(r['event_file'], tags) for s, r in bps.items()}


def align(seed_data, tag, gs=10000, max_s=1_000_000):
    """Interpolate all seeds to a regular grid."""
    grid = np.arange(gs, max_s + 1, gs)
    rows = []
    for s in sorted(seed_data):
        if tag not in seed_data[s]:
            rows.append(np.full(len(grid), np.nan))
            continue
        st, v = seed_data[s][tag]
        rows.append(np.interp(grid, st, v, left=np.nan, right=np.nan))
    return grid, np.array(rows)


def align_binned(seed_data, tag, bin_width=5000, max_s=1_000_000):
    """Bin raw data into fixed-width windows, averaging all values in each bin."""
    edges = np.arange(bin_width, max_s + 1, bin_width)
    rows = []
    for s in sorted(seed_data):
        if tag not in seed_data[s]:
            rows.append(np.full(len(edges), np.nan))
            continue
        st, v = seed_data[s][tag]
        row = np.full(len(edges), np.nan)
        for i, e in enumerate(edges):
            mask = (st > e - bin_width) & (st <= e)
            if np.any(mask):
                row[i] = np.mean(v[mask])
        rows.append(row)
    return edges, np.array(rows)


def mean_ci(mat, conf=0.95):
    n = np.sum(~np.isnan(mat), axis=0)
    mu = np.nanmean(mat, axis=0)
    se = np.where(n > 1, np.nanstd(mat, axis=0, ddof=1) / np.sqrt(n), np.nan)
    tc = np.array([stats.t.ppf(1 - (1-conf)/2, max(ni-1, 1)) if ni > 1 else np.nan
                   for ni in n])
    return mu, tc * se, n


def plot_ci(ax, grid, mat, label, color, lw=2, alpha=0.2):
    mu, hw, n = mean_ci(mat)
    ok = n >= 2
    ax.plot(grid[ok]/1e6, mu[ok], label=label, color=color, lw=lw)
    ax.fill_between(grid[ok]/1e6, (mu-hw)[ok], (mu+hw)[ok],
                    color=color, alpha=alpha)


# ─── Consistent statistics (used EVERYWHERE) ────────────────────────────────
def seed_avgs(seed_data, tag='sample/episode_return', last=100_000):
    """Compute per-seed average over last `last` env steps."""
    out = []
    for s in sorted(seed_data):
        if tag not in seed_data[s]:
            continue
        st, v = seed_data[s][tag]
        mask = st > (st[-1] - last)
        if np.sum(mask) >= 2:
            out.append(np.mean(v[mask]))
    return np.array(out)


def group_mean_ci(seed_data, tag='sample/episode_return', last=100_000, conf=0.95):
    """Compute mean and 95% CI from per-seed averages (t-distribution).
    This is the SINGLE method used for all bar charts and statistics."""
    avgs = seed_avgs(seed_data, tag, last)
    if len(avgs) < 2:
        return np.mean(avgs) if len(avgs) > 0 else np.nan, np.nan, np.nan
    mu = np.mean(avgs)
    se = np.std(avgs, ddof=1) / np.sqrt(len(avgs))
    tc = stats.t.ppf(1 - (1-conf)/2, len(avgs) - 1)
    hw = tc * se
    return mu, mu - hw, mu + hw


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan, np.nan
    t, p = stats.ttest_ind(b, a, equal_var=False)
    pool = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    d = (np.mean(b) - np.mean(a)) / pool if pool > 0 else 0
    return t, p, d


def req_n(d, alpha=0.05, power=0.8):
    if abs(d) < 0.01:
        return 9999
    za = stats.norm.ppf(1 - alpha/2)
    zb = stats.norm.ppf(power)
    return int(np.ceil(2 * ((za + zb) / d) ** 2))


def bar_with_ci(ax, items, data_dict, ymin=None):
    """Draw a bar chart using per-seed-average t-distribution 95% CI."""
    for i, (label, gname, color) in enumerate(items):
        if gname not in data_dict:
            continue
        mu, lo, hi = group_mean_ci(data_dict[gname])
        if np.isnan(mu):
            continue
        ax.bar(i, mu, color=color, width=0.55, edgecolor='black', lw=0.5)
        if not np.isnan(lo):
            ax.errorbar(i, mu, yerr=[[mu - lo], [hi - mu]],
                        fmt='none', color='black', capsize=6, lw=1.5)
            offset = (hi - lo) * 0.08
            ax.text(i, hi + offset, f'{mu:.0f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
        else:
            ax.text(i, mu + 50, f'{mu:.0f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([it[0] for it in items])
    if ymin is not None:
        ax.set_ylim(bottom=ymin)
    ax.grid(True, alpha=0.3, axis='y')


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
print("Scanning runs …")
ALL = scan_runs()
print(f"  {len(ALL)} valid runs found\n")

# ─── Group definitions ───────────────────────────────────────────────────────
def G(runs, *, mala=2, tfg_end=None, lr='any', q_lr='any',
      ddim=False, mg=True, tfg=16.0):
    out = []
    for r in runs:
        if r['mala'] != mala or r['mg'] != mg or r['ddim'] != ddim:
            continue
        if r['tfg'] != tfg:
            continue
        if tfg_end is None:
            if r['tfg_end'] is not None:
                continue
        else:
            if r['tfg_end'] != tfg_end:
                continue
        if lr != 'any':
            if lr is None:
                if r['lr_sched'] is not None and r['lr_sched'] != 'constant':
                    continue
            else:
                if r['lr_sched'] != lr:
                    continue
        if q_lr != 'any':
            if q_lr is None:
                if r['q_lr_sched'] is not None and r['q_lr_sched'] != 'constant':
                    continue
            else:
                if r['q_lr_sched'] != q_lr:
                    continue
        out.append(r)
    return out

base_runs = [r for r in ALL if r['mala'] == 2 and r['tfg_end'] is None
             and r['lr_sched'] is None and r['q_lr_sched'] is None
             and not r['ddim'] and r['mg'] and r['tfg'] == 16.0]

pol_lr_runs = {}
for r in ALL:
    if (r['mala'] == 2 and r['tfg_end'] is None and r['lr_sched'] == 'log_linear'
        and r['mg'] and not r['ddim'] and r['tfg'] == 16.0):
        pol_lr_runs.setdefault(r['lr_end'], []).append(r)

tfg_end_runs = {}
for te in [1.0, 2.0, 4.0]:
    rs = G(ALL, tfg_end=te, lr='any', q_lr='any')
    if rs:
        tfg_end_runs[te] = rs

mala4_runs = G(ALL, mala=4, lr='any', q_lr='any')
mala8_runs = G(ALL, mala=8, lr='any', q_lr='any')

mala_tfg = {}
for m in [2, 4, 8]:
    for te in [1.0, 2.0]:
        rs = G(ALL, mala=m, tfg_end=te, lr='any', q_lr='any')
        if rs:
            mala_tfg[(m, te)] = rs

# ─── Load data ───────────────────────────────────────────────────────────────
print("Loading TB data …")
D = {}

D['base'] = load_group(base_runs)
print(f"  base: {len(D['base'])} seeds  (expect 10)")

for k, rs in sorted(pol_lr_runs.items(), key=lambda x: str(x[0])):
    name = f"pol_lr_{k}"
    D[name] = load_group(rs)
    print(f"  {name}: {len(D[name])} seeds  (expect 5)")

for te, rs in sorted(tfg_end_runs.items()):
    name = f"tfg_end_{te}"
    D[name] = load_group(rs)
    seeds = sorted(best_per_seed(rs).keys())
    print(f"  {name}: {len(D[name])} seeds {seeds}  (expect 5, [0,1,2,3,4])")

D['mala_4'] = load_group(mala4_runs)
D['mala_8'] = load_group(mala8_runs)
print(f"  mala_4: {len(D['mala_4'])} seeds  (expect 5)")
print(f"  mala_8: {len(D['mala_8'])} seeds  (expect 5)")

for (m, te), rs in sorted(mala_tfg.items()):
    name = f"mala_{m}_tfg_{te}"
    D[name] = load_group(rs)
    print(f"  {name}: {len(D[name])} seeds  (expect 5)")

TAG = 'sample/episode_return'
POL = 'training/policy_loss'

# ─── Verify all data ─────────────────────────────────────────────────────────
print("\n=== DATA VERIFICATION ===")
for gname, gdata in sorted(D.items()):
    avgs = seed_avgs(gdata)
    mu, lo, hi = group_mean_ci(gdata)
    seeds = sorted(gdata.keys())
    per_seed = []
    for s in seeds:
        if TAG in gdata[s]:
            st, v = gdata[s][TAG]
            mask = st > (st[-1] - 100000)
            per_seed.append(f"s{s}={np.mean(v[mask]):.0f}")
    print(f"  {gname:20s}: n={len(avgs)}, mean={mu:.0f}, "
          f"CI=[{lo:.0f},{hi:.0f}]  seeds: {', '.join(per_seed)}")

# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Base training curve (10 seeds)
# ═════════════════════════════════════════════════════════════════════════════
print("\nFig 1: Base training curve")
fig, ax = plt.subplots(figsize=(9, 4.5))
g, m = align(D['base'], TAG)
plot_ci(ax, g, m, 'Base (10 seeds)', C['base'])
ax.set_xlabel('Environment Steps (millions)')
ax.set_ylabel('Episode Return')
ax.set_title('HalfCheetah-v4 — Base Performance')
ax.legend(loc='lower right')
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig1_base_curve.pdf'))
plt.close()
print("  → fig1_base_curve.pdf")

# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Policy LR annealing: training curves (L) + bar chart (R)
# ═════════════════════════════════════════════════════════════════════════════
print("\nFig 2: Policy LR curves + bar chart")
pol_items = [
    ('Base',           'base',       C['base']),
    ('LR → 3e-5',     'pol_lr_3e-05', C['pol3e5']),
    ('LR → 1e-5',     'pol_lr_1e-05', C['pol1e5']),
    ('LR → 3e-6',     'pol_lr_3e-06', C['pol3e6']),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

for label, gname, color in pol_items:
    if gname not in D:
        continue
    g, m = align(D[gname], TAG)
    plot_ci(ax1, g, m, label, color)
ax1.set_xlabel('Environment Steps (millions)')
ax1.set_ylabel('Episode Return')
ax1.set_title('Policy LR Annealing — Training Curves')
ax1.legend(loc='lower right')

bar_with_ci(ax2, pol_items, D, ymin=12000)
ax2.set_ylabel('Mean Episode Return (last 100k steps)')
ax2.set_title('Policy LR Annealing — Last 100k Avg')

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig2_policy_lr.pdf'))
plt.close()
print("  → fig2_policy_lr.pdf")

# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — TFG lambda annealing: episode return + policy loss
# ═════════════════════════════════════════════════════════════════════════════
print("\nFig 3: TFG annealing curves + policy loss")

tfg_plot = [
    ('Base (η=16)',  'base',         C['base']),
    ('η: 16 → 4',   'tfg_end_4.0',  C['tfg4']),
    ('η: 16 → 2',   'tfg_end_2.0',  C['tfg2']),
    ('η: 16 → 1',   'tfg_end_1.0',  C['tfg1']),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

# Episode return
for label, gname, color in tfg_plot:
    if gname not in D:
        continue
    g, m = align(D[gname], TAG)
    plot_ci(ax1, g, m, label, color)
ax1.set_xlabel('Environment Steps (millions)')
ax1.set_ylabel('Episode Return')
ax1.set_title('η Annealing — Episode Return')
ax1.set_ylim(-500, 16000)
ax1.legend(loc='lower right')

# Policy loss — use 5k-step binned averages
for label, gname, color in tfg_plot:
    if gname not in D:
        continue
    g, m = align_binned(D[gname], POL, bin_width=5000)
    if m is not None and not np.all(np.isnan(m)):
        plot_ci(ax2, g, m, label, color)
ax2.set_xlabel('Environment Steps (millions)')
ax2.set_ylabel('Policy Loss')
ax2.set_title('η Annealing — Policy Loss')
ax2.set_yscale('log')
ax2.set_ylim(bottom=0.05)
ax2.legend(loc='upper left')

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig3_tfg_annealing.pdf'))
plt.close()
print("  → fig3_tfg_annealing.pdf")

# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 4 — MALA steps × TFG annealing comparison
# ═════════════════════════════════════════════════════════════════════════════
print("\nFig 4: MALA × TFG comparison")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

for gname, label, color in [
    ('mala_2_tfg_1.0', 'MALA-2, η→1', C['tfg1']),
    ('mala_4_tfg_1.0', 'MALA-4, η→1', C['mala4tfg1']),
    ('mala_8_tfg_1.0', 'MALA-8, η→1', C['mala8tfg1']),
]:
    if gname not in D:
        continue
    g, m = align(D[gname], TAG)
    plot_ci(ax1, g, m, label, color)
ax1.set_xlabel('Environment Steps (millions)')
ax1.set_ylabel('Episode Return')
ax1.set_title('η→1 with Varying MALA Steps')
ax1.legend(loc='upper left')

for gname, label, color in [
    ('mala_2_tfg_2.0', 'MALA-2, η→2', C['tfg2']),
    ('mala_4_tfg_2.0', 'MALA-4, η→2', C['mala4tfg2']),
    ('mala_8_tfg_2.0', 'MALA-8, η→2', C['mala8tfg2']),
]:
    if gname not in D:
        continue
    g, m = align(D[gname], TAG)
    plot_ci(ax2, g, m, label, color)
ax2.set_xlabel('Environment Steps (millions)')
ax2.set_ylabel('Episode Return')
ax2.set_title('η→2 with Varying MALA Steps')
ax2.legend(loc='upper left')

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig4_mala_tfg.pdf'))
plt.close()
print("  → fig4_mala_tfg.pdf")

# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 5 — MALA steps without annealing: curves + bar chart
# ═════════════════════════════════════════════════════════════════════════════
print("\nFig 5: MALA steps (no annealing)")
mala_items = [
    ('MALA-2 (Base)', 'base',   C['base']),
    ('MALA-4',        'mala_4', C['mala4']),
    ('MALA-8',        'mala_8', C['mala8']),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

for label, gname, color in mala_items:
    g, m = align(D[gname], TAG)
    plot_ci(ax1, g, m, label, color)
ax1.set_xlabel('Environment Steps (millions)')
ax1.set_ylabel('Episode Return')
ax1.set_title('MALA Steps — Training Curves')
ax1.legend(loc='lower right')

bar_with_ci(ax2, mala_items, D, ymin=12000)
ax2.set_ylabel('Mean Episode Return (last 100k steps)')
ax2.set_title('MALA Steps — Last 100k Avg')

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig5_mala_no_anneal.pdf'))
plt.close()
print("  → fig5_mala_no_anneal.pdf")

# ═════════════════════════════════════════════════════════════════════════════
#  FIGURE 6 — Statistical comparison (p-values, sample sizes)
#  Uses SAME seed_avgs + t-CI method as bar charts for consistency
# ═════════════════════════════════════════════════════════════════════════════
print("\nFig 6: Statistical analysis")
base_a = seed_avgs(D['base'])
print(f"  Base: mean={np.mean(base_a):.1f}  std={np.std(base_a,ddof=1):.1f}  n={len(base_a)}")

rows = []
test_items = []
for k in sorted(pol_lr_runs.keys(), reverse=True):  # descending by float value
    test_items.append((f"LR→{k}", f"pol_lr_{k}"))
test_items += [('MALA-4', 'mala_4'), ('MALA-8', 'mala_8')]

for label, gname in test_items:
    if gname not in D:
        continue
    a = seed_avgs(D[gname])
    t, p, d = welch(base_a, a)
    nr = req_n(d)
    na = max(0, nr - len(a))
    rows.append((label, gname, a, t, p, d, nr, na))
    print(f"  {label:12s}: mean={np.mean(a):7.1f}  n={len(a)}  "
          f"t={t:+.3f}  p={p:.4f}  d={d:+.3f}  n_req={nr}  n_add={na}")

# Save stats JSON
stats_out = []
for label, gname, a, t, p, d, nr, na in rows:
    stats_out.append(dict(label=label, mean=float(np.mean(a)),
                          std=float(np.std(a, ddof=1)), n=int(len(a)),
                          t=float(t), p=float(p), d=float(d),
                          n_req=int(nr), n_add=int(na)))
with open(os.path.join(OUT_DIR, 'stats.json'), 'w') as f:
    json.dump(dict(base_mean=float(np.mean(base_a)),
                   base_std=float(np.std(base_a, ddof=1)),
                   base_n=int(len(base_a)),
                   comparisons=stats_out), f, indent=2)

# Plot — uses same seed_avgs + t-CI as bar charts
fig, ax = plt.subplots(figsize=(10, 4.5))

base_m = np.mean(base_a)
ax.axhline(y=base_m, color=C['base'], ls='--', lw=2,
           label=f'Base mean = {base_m:.0f}')

for i, (label, gname, a, t, p, d, nr, na) in enumerate(rows):
    col = '#2ca02c' if p < 0.05 else ('#fd8d3c' if p < 0.1 else '#bdbdbd')
    mu = np.mean(a)
    se = np.std(a, ddof=1) / np.sqrt(len(a))
    hw = stats.t.ppf(0.975, len(a)-1) * se
    ax.bar(i, mu, color=col, width=0.55, edgecolor='black', lw=0.5)
    ax.errorbar(i, mu, yerr=hw, fmt='none', color='black', capsize=6, lw=1.5)
    ax.text(i, mu + hw + 60, f'p={p:.3f}', ha='center',
            va='bottom', fontsize=9)

ax.set_xticks(range(len(rows)))
ax.set_xticklabels([r[0] for r in rows])
ax.set_ylabel('Mean Episode Return (last 100k steps)')
ax.set_title('Welch t-test vs Base  (green: p<0.05,  orange: p<0.1,  grey: n.s.)')
ax.set_ylim(bottom=10000)
ax.legend(loc='lower right')

fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'fig6_stats.pdf'))
plt.close()
print("  → fig6_stats.pdf")

print("\n✓ All figures saved to", OUT_DIR)
