#!/usr/bin/env python3
"""Could a normalizer have delivered a per-env eta with ONE global eta?

``scripts/compute_sweep_normalized_scores.py`` produces a row saying "take
``buffer_size=400000 gamma=0.99 rollout_alpha=1`` and let ``eta`` differ per
environment" -- eta 64 for Ant/Hopper/Humanoid, 128 for HalfCheetah/Swimmer/
Walker2d. A per-env hyperparameter is not shippable. An automatic normalizer
that happens to be ~2x larger in the eta-64 envs than in the eta-128 envs is,
because it produces the same *effective* step from a single eta.

The composite-MD energy is ``U = alpha E_theta - beta Q / N``, and these runs
have ``T=0``, hence ``alpha=1`` and ``beta=eta``, so the effective step size on
the raw Q is

    eta_eff = eta / N,        N = the guidance-Q divisor.

Sweep 237 ran ``--ema_advantage_normalization``, i.e. ``N = S`` where ``S`` is
``Critic/adv_norm_running_std`` (an EMA of the pooled batch sd of the tilting
Q). So the step the row actually prescribes is ``Z = eta_pres / S``. Swapping in
divisor ``N'`` and a single global eta ``g`` gives ``g / N'``, so matching the
row per env requires

    g_req(env) = Z(env) * N'(env).

``g_req`` constant across envs <=> that normalizer reproduces the row's per-env
eta from one number. The dispersion ``max/min`` of ``g_req`` is the residual
per-env tuning the normalizer fails to absorb: 2.00 means it absorbs nothing
(the status quo, where ``g_req`` IS 64 vs 128), 1.00 means it absorbs all of it.

FIRST ORDER ONLY. Each ``N'`` is measured on trajectories produced by the
``S`` divisor. A run that had actually divided by ``N'`` would have followed a
different trajectory and grown a different Q scale, so this answers "what
divisor did these runs exhibit", not "what would that run have converged to".
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from offline_wandb import index_runs, read_history  # noqa: E402

WANDB_BASE = Path("/n/holylabs/kdbrantley_lab/Lab/pnielsen/wandb")
ENVS = ["Ant-v3", "HalfCheetah-v3", "Hopper-v3", "Humanoid-v3", "Swimmer-v3",
        "Walker2d-v3"]

# The row under study: base config with eta freed, and the eta it picked per env.
BASE_TAGS = {
    64: "sweep237_T=0_buffer_size=400000_eta=64_gamma=0.99"
        "_mcmc_proposal_type=euler_maruyama_rollout_alpha=1",
    128: "sweep237_T=0_buffer_size=400000_eta=128_gamma=0.99"
         "_mcmc_proposal_type=euler_maruyama_rollout_alpha=1",
}
# Config-identical pool folded into eta=64 by the scoring script, so the same
# 6 seeds back the normalizer estimate as back the score.
POOL_TAGS = {64: "sweep233_mala_target_acceptance_rate=0.7"}
PRESCRIBED = {"Ant-v3": 64, "HalfCheetah-v3": 128, "Hopper-v3": 64,
              "Humanoid-v3": 64, "Swimmer-v3": 128, "Walker2d-v3": 128}

# Divisor S in use, the TD loss that --q_loss_normalization would divide by, and
# two Q-scale references that no flag currently divides by.
KEYS = ["Critic/adv_norm_running_std", "losses/Q_loss", "Critic/average_Q",
        "Critic/E(Var({Q_i})_env)", "Global_EMAs/beta"]
ADV_EMA_TAU = 0.0005      # --advantage_ema_tau, the rate --q_loss_normalization uses
WINDOW = (800_000, 1_000_000)
ETA_GRID = (1, 2, 4, 8)   # candidate etas to restate on the current eta scale


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wandb-base", type=Path, default=WANDB_BASE)
    ap.add_argument("--envs", nargs="+", default=ENVS)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=2000,
                    help="seed-bootstrap replicates for the RMSE interval")
    ap.add_argument("--out-dir", type=Path,
                    default=SCRIPT_DIR / "topsis_out" / "sweep_237+239" / "normalized_eval")
    ap.add_argument("--cache", type=Path, default=None,
                    help="per-run aggregates CSV; reused if present")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache")
    return ap.parse_args(argv)


def ema_last(steps, vals, tau):
    """Final value of the in-training EMA, reconstructed from a subsampled log.

    The EMA folds ``tau`` per *update*, but the log holds one point per ~``d``
    updates, so the per-point rate is ``1 - (1 - tau)^d``: fold once per logged
    point at the compounded rate rather than pretending the log is the update
    stream (which would make the EMA ~d times too fast).
    """
    if len(vals) < 2:
        return np.nan
    d = float(np.median(np.diff(steps)))
    r = 1.0 - (1.0 - tau) ** max(d, 1.0)
    e = vals[0]
    out = np.empty(len(vals))
    for i, v in enumerate(vals):
        e = e + r * (v - e)
        out[i] = e
    return out


def geo(x):
    """Geometric mean over positive entries: these are all multiplicative scales."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.mean(np.log(x)))) if x.size else np.nan


def _one_run(args):
    """Per-run time-aggregates of every key, over full training and the window."""
    run_dir, meta = args
    h = read_history(run_dir, keys=KEYS)
    out = dict(meta)
    if h.empty:
        return out
    for key, g in h.groupby("key"):
        g = g.sort_values("step")
        steps, vals = g["step"].to_numpy(), g["value"].to_numpy(float)
        short = key.split("/")[-1]
        in_win = (steps >= WINDOW[0]) & (steps <= WINDOW[1])
        out[f"{short}|geo_all"] = geo(vals)
        out[f"{short}|geo_win"] = geo(vals[in_win])
        out[f"{short}|mean_win"] = float(np.mean(vals[in_win])) if in_win.any() else np.nan
        if key == "losses/Q_loss":
            # The divisor is sqrt of the EMA, so the EMA is taken on the loss and
            # the sqrt afterwards -- not the other way round.
            e = ema_last(steps, vals, ADV_EMA_TAU)
            out["qloss_ema|geo_all"] = geo(e)
            out["qloss_ema|geo_win"] = geo(np.asarray(e)[in_win])
        out[f"{short}|n"] = int(len(vals))
    return out


def collect(args):
    idx = index_runs([237, 233], envs=args.envs, base=args.wandb_base)
    want = set(BASE_TAGS.values()) | set(POOL_TAGS.values())
    sub = idx[idx["config_tag"].isin(want)].copy()
    # The pool tag is a whole sweep-233 config, whose eta is its own; label every
    # run by the eta it ran at so a slice is (env, eta) regardless of provenance.
    sub["eta_run"] = pd.to_numeric(sub["eta"], errors="coerce")
    jobs = [(r["run_dir"], {"env": r["env"], "eta_run": r["eta_run"],
                            "sweep": r["sweep"], "seed": r["seed"],
                            "run_id": r["run_id"], "config_tag": r["config_tag"]})
            for _, r in sub.iterrows()]
    print(f"reading {len(jobs)} offline runs with {args.workers} workers ...")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_one_run, jobs))
    return pd.DataFrame(rows)


# Candidate divisors. Each maps a per-run aggregate column to the divisor N'
# the flag would apply, given the suffix picking the averaging window.
def candidates(w):
    return [
        ("--ema_advantage_normalization (in use)", lambda d: d[f"adv_norm_running_std|{w}"],
         "EMA of the pooled batch sd of the tilting Q; the divisor these runs ran with"),
        ("no normalizer", lambda d: 1.0,
         "raw Q straight into the tilt"),
        ("--q_loss_normalization", lambda d: np.sqrt(d[f"qloss_ema|{w}"]),
         "sqrt of the EMA of the critic TD loss"),
        ("Q magnitude (no flag)", lambda d: d[f"average_Q|{w}"],
         "reference only: mean Q, which no flag divides by"),
        ("critic-ensemble sd (no flag)", lambda d: np.sqrt(d[f"E(Var({{Q_i}})_env)|{w}"]),
         "reference only: sd ACROSS ensemble members, not across actions"),
    ]


def slices(runs, w):
    """Per-env aggregates at the eta the row prescribes, plus the other eta.

    The normalizer has to be read off the run the row actually selects, so the
    trajectory it was measured on is the one being reproduced.
    """
    out = {}
    for env, eta in PRESCRIBED.items():
        for label, e in (("pres", eta), ("other", 128 if eta == 64 else 64)):
            g = runs[(runs["env"] == env) & (runs["eta_run"] == float(e))]
            out[(env, label)] = {c: geo(g[c]) for c in runs.columns if "|" in c}
            out[(env, label)]["_rows"] = g
    return out


def evaluate(runs, w, which="pres", rng=None, n_boot=0):
    """Required global eta per env for every candidate divisor, + its dispersion."""
    sl = slices(runs, w)
    rows = []
    for name, fn, note in candidates(w):
        g_req, per_env = {}, {}
        for env, eta_pres in PRESCRIBED.items():
            d = sl[(env, which)]
            S = d[f"adv_norm_running_std|{w}"]
            Z = eta_pres / S                       # the effective step the row prescribes
            N = fn(d)
            g_req[env] = Z * N
            per_env[env] = dict(eta_pres=eta_pres, S=S, Z=Z, N=N, g_req=Z * N)
        vals = np.array([g_req[e] for e in PRESCRIBED], float)
        ok = np.isfinite(vals) & (vals > 0)
        if ok.sum() < len(vals):
            rows.append(dict(normalizer=name, note=note, eta_star=np.nan,
                             rmse_oct=np.nan, spread=np.nan, per_env=per_env,
                             degenerate=True))
            continue
        lg = np.log2(vals)
        eta_star = float(2 ** lg.mean())
        rmse = float(np.sqrt(np.mean((lg - lg.mean()) ** 2)))
        # Split the miss in two. The row asks for one thing only: eta-64 envs at
        # half the eta of eta-128 envs. group_ratio is whether the divisor
        # delivers that (1.00 = yes, 0.50 = absorbs none, which is all the status
        # quo can do by construction); within_oct is the extra per-env scatter
        # the divisor introduces on top, which is pure cost.
        def gratio(envs):
            a = geo([g_req[e] for e in envs if PRESCRIBED[e] == 64])
            b = geo([g_req[e] for e in envs if PRESCRIBED[e] == 128])
            return a / b, a, b
        _, g64, g128 = gratio(list(PRESCRIBED))
        grp = np.array([np.log2(g64 if PRESCRIBED[e] == 64 else g128)
                        for e in PRESCRIBED])
        # Three envs per group: one of them can manufacture the entire group
        # signal, so quote the range over dropping each env in turn. A range
        # straddling 0.50 means the divisor absorbs nothing robustly.
        loeo = [gratio([e for e in PRESCRIBED if e != drop])[0]
                for drop in PRESCRIBED]
        rows.append(dict(normalizer=name, note=note, eta_star=eta_star,
                         rmse_oct=rmse, spread=float(vals.max() / vals.min()),
                         group_ratio=float(g64 / g128),
                         group_lo=float(np.min(loeo)), group_hi=float(np.max(loeo)),
                         within_oct=float(np.sqrt(np.mean((lg - grp) ** 2))),
                         per_env=per_env, degenerate=False))
    if n_boot and rng is not None:
        boot_dispersion(runs, w, which, rows, rng, n_boot)
    return rows


def boot_dispersion(runs, w, which, rows, rng, n_boot):
    """Resample seeds within each (env, eta) cell and redo the whole fit."""
    cells = {}
    for env, eta in PRESCRIBED.items():
        e = eta if which == "pres" else (128 if eta == 64 else 64)
        cells[env] = runs[(runs["env"] == env) & (runs["eta_run"] == float(e))]
    for row in rows:
        if row["degenerate"]:
            row["rmse_p05"] = row["rmse_p95"] = np.nan
            continue
        fn = dict((n, f) for n, f, _ in candidates(w))[row["normalizer"]]
        draws = np.empty(n_boot)
        for b in range(n_boot):
            lg = []
            for env, eta_pres in PRESCRIBED.items():
                g = cells[env]
                take = g.iloc[rng.integers(0, len(g), len(g))]
                d = {c: geo(take[c]) for c in g.columns if "|" in c}
                S = d[f"adv_norm_running_std|{w}"]
                lg.append(np.log2((eta_pres / S) * fn(d)))
            lg = np.array(lg, float)
            draws[b] = np.sqrt(np.mean((lg - lg.mean()) ** 2))
        row["rmse_p05"] = float(np.percentile(draws, 5))
        row["rmse_p95"] = float(np.percentile(draws, 95))


def render_eta_grid(res, w):
    """A --q_loss_normalization eta, restated on the current divisor's eta scale.

    Both schemes are just a step ``eta / N``, so an eta of ``eta_q`` under
    ``sqrt(EMA(Q_loss))`` delivers the same step as

        eta_S = eta_q * S / sqrt(EMA(Q_loss))

    would under the divisor in use. That puts the counterfactual in the units the
    row is written in, where 64 and 128 are the numbers that mean something.
    """
    r = next((x for x in res if x["normalizer"] == "--q_loss_normalization"), None)
    if r is None or r["degenerate"]:
        return []
    envs = list(PRESCRIBED)
    pe = r["per_env"]
    # S / sqrt(EMA(Q_loss)); N' IS sqrt(EMA(Q_loss)) for this candidate.
    k = {e: pe[e]["S"] / pe[e]["N"] for e in envs}
    L = ["", "## `--q_loss_normalization` eta, restated as an equivalent current eta", "",
         "Both schemes apply a step `eta / N`, so running `--q_loss_normalization` "
         "at `eta_q` gives the same step on the raw Q as "
         "`eta_S = eta_q * S / sqrt(EMA(Q_loss))` would give under the divisor now "
         "in use. The row wants `eta_S` to come out at the **target** row.",
         "",
         "| env | target `eta_S` | `S` | `sqrt(EMA(Q_loss))` | `S/sqrt(EMA(Q_loss))` | "
         + " | ".join(f"eta_q={g}" for g in ETA_GRID)
         + " | `eta_q` to hit target |",
         "|---|---|---|---|---|" + "|".join(["---"] * len(ETA_GRID)) + "|---|"]
    for e in envs:
        cells = [f"{g * k[e]:.1f}" for g in ETA_GRID]
        L.append(f"| {e.replace('-v3', '')} | **{PRESCRIBED[e]}** | {pe[e]['S']:.2f} | "
                 f"{pe[e]['N']:.4g} | {k[e]:.1f} | " + " | ".join(cells)
                 + f" | **{pe[e]['g_req']:.2f}** |")
    # Same eta for everyone is the whole point of a normalizer, so score each
    # column by how far off the targets it lands.
    L.append("| **RMSE vs target (octaves)** | - | - | - | - | "
             + " | ".join(
                 f"{np.sqrt(np.mean([np.log2(g * k[e] / PRESCRIBED[e]) ** 2 for e in envs])):.2f}"
                 for g in ETA_GRID) + " | - |")
    L += ["",
          "The last column is the `eta_q` each env would need on its own; a "
          "normalizer works only if that column is flat, and it spans "
          f"{min(pe[e]['g_req'] for e in envs):.2f} to "
          f"{max(pe[e]['g_req'] for e in envs):.2f}. No single `eta_q` in the grid "
          "lands near the targets: Hopper and Swimmer pull one way "
          f"({k['Hopper-v3']:.0f} and {k['Swimmer-v3']:.0f} per unit eta_q) while "
          f"Ant and Walker2d pull the other ({k['Ant-v3']:.0f} and "
          f"{k['Walker2d-v3']:.0f}).",
          ""]
    return L


def render(res, runs, w, args):
    """The report: one summary table, then the per-env detail behind it."""
    ref = next(r for r in res if r["normalizer"].startswith("--ema_advantage"))
    L = [
        "# Could a normalizer have replaced the per-env eta of row 2?",
        "",
        "Row 2 of `adaptive_oracle.md`: base config "
        "`buffer_size=400000 gamma=0.99 rollout_alpha=1`, with **eta 64 for "
        "Ant/Hopper/Humanoid and 128 for HalfCheetah/Swimmer/Walker2d** "
        "(oracle 0.9904 vs 0.9693 fixed).",
        "",
        "These runs have `T=0`, so `alpha=1` and `beta=eta`, and the energy is "
        "`U = E_theta - (eta / N) Q`: the effective mirror-descent step on the "
        "raw Q is **`eta / N`**, `N` = guidance-Q divisor. They ran "
        "`--ema_advantage_normalization`, so `N = S =` "
        "`Critic/adv_norm_running_std`. The step the row prescribes is therefore "
        "`Z = eta_pres / S`, and swapping in divisor `N'` with ONE global eta `g` "
        "reproduces it per env iff",
        "",
        "```",
        "g_req(env) = Z(env) * N'(env)   is constant across envs",
        "```",
        "",
        f"Aggregation: geometric mean over logged steps ({'800k-1M window' if w.endswith('win') else 'all of training'}) "
        f"then over seeds ({len(runs)} runs: 6 seeds/env at eta=64 incl. the "
        "sweep-233 pool, 3 at eta=128).",
        "",
        "`RMSE` is the root-mean-square of `log2(g_req / eta*)` -- the per-env "
        "error in eta, **in octaves (doublings)**, if you shipped the single best "
        "`eta*`. **The benchmark to beat is the status quo, whose `g_req` IS "
        f"64/128, giving exactly {ref['rmse_oct']:.3f} octaves.** A normalizer "
        "only earns its place by scoring BELOW that.",
        "",
        "The miss splits in two. The row asks for exactly one thing -- the "
        "eta-64 envs at half the eta of the eta-128 envs -- so `group ratio` = "
        "geomean(`g_req`) over the eta-64 envs divided by that over the eta-128 "
        "envs measures whether the divisor delivers it: **1.00 absorbs the whole "
        "split, 0.50 absorbs none of it** (the status quo, by construction). "
        "`within` is the extra per-env scatter the divisor introduces on top, "
        "which is pure cost.",
        "",
        "With only three envs per group one env can manufacture the whole group "
        "signal, so `group ratio` is also quoted as a **leave-one-env-out range**; "
        "a range straddling 0.50 means nothing robust is absorbed.",
        "",
        "| divisor `N'` | best single eta* | RMSE (octaves) | boot 5-95% | group ratio | LOEO range | within (oct) | spread max/min | beats 64/128? |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(res, key=lambda r: (r["degenerate"], r["rmse_oct"])):
        if r["degenerate"]:
            L.append(f"| {r['normalizer']} | - | degenerate | - | - | - | - | - | no |")
            continue
        ci = (f"{r.get('rmse_p05', float('nan')):.3f}-{r.get('rmse_p95', float('nan')):.3f}"
              if np.isfinite(r.get("rmse_p05", np.nan)) else "-")
        win = "**yes**" if r["rmse_oct"] < ref["rmse_oct"] - 1e-9 else "no"
        same = " (reference)" if r is ref else ""
        L.append(f"| {r['normalizer']}{same} | {r['eta_star']:.4g} | "
                 f"**{r['rmse_oct']:.3f}** | {ci} | {r['group_ratio']:.2f} | "
                 f"{r['group_lo']:.2f}-{r['group_hi']:.2f} | "
                 f"{r['within_oct']:.3f} | {r['spread']:.2f}x | {win} |")
    L += ["", "## Per-env detail", "",
          "`S` is the divisor in use, `Z = eta_pres/S` the prescribed effective "
          "step, `N'` the candidate divisor, `g_req = Z N'` the global eta that "
          "env would need. Read the `g_req` columns for constancy, not size.",
          ""]
    envs = list(PRESCRIBED)
    L.append("| quantity | " + " | ".join(e.replace("-v3", "") for e in envs) + " |")
    L.append("|---|" + "|".join(["---"] * len(envs)) + "|")
    L.append("| **eta the row picks** | "
             + " | ".join(f"**{PRESCRIBED[e]}**" for e in envs) + " |")
    d0 = ref["per_env"]
    L.append("| `S` (divisor in use) | "
             + " | ".join(f"{d0[e]['S']:.2f}" for e in envs) + " |")
    L.append("| `Z` = prescribed step | "
             + " | ".join(f"{d0[e]['Z']:.4g}" for e in envs) + " |")
    for r in res:
        if r["degenerate"]:
            continue
        L.append(f"| `N'`: {r['normalizer']} | "
                 + " | ".join(f"{r['per_env'][e]['N']:.4g}" for e in envs) + " |")
        L.append(f"| -> `g_req` | "
                 + " | ".join(f"**{r['per_env'][e]['g_req']:.4g}**" for e in envs) + " |")
    L += render_eta_grid(res, w)
    L += ["", "## Caveats", "",
          "- **First order.** Every `N'` is measured on trajectories that "
          "divided by `S`. A run that had actually divided by `N'` would have "
          "grown a different Q scale, so this says what these runs exhibited, "
          "not what such a run would converge to.",
          "- **`--batch_advantage_normalization` and "
          "`--ema_within_advantage_normalization` cannot be evaluated here at "
          "all.** Both divide by a WITHIN-state spread over the K denoised "
          "actions, and this row ran `--num_denoised_actions 1`, so that "
          "variance does not exist in these runs (`ddof=1` on one sample) and "
          "nothing proportional to it was logged. "
          "`Critic/E(Var({Q_i})_env)` is the sd across ensemble MEMBERS at one "
          "action, a different quantity, and it underflows to 0 on Swimmer.",
          "- `--kl_budget` sets `beta = sqrt(2 delta / M)`. If `M` is the pooled "
          "`Var(Q)` it is algebraically the same rule as the divisor already in "
          "use, so it inherits the same residual; only a within-state `M` would "
          "differ, and that is the K>=2 case above.",
          "- `--reward_scale` cannot substitute for eta at all here: it scales Q "
          "and hence `S` by the same factor, which cancels exactly in `eta / S`.",
          "- Trap for anyone rerunning this: `config.yaml` writes `beta` BEFORE "
          "the per-slot `--hp_pack_inline` override, so the eta=128 slots record "
          "a stale `beta=64`. The logged `Global_EMAs/beta` is the runtime value "
          "and equals `eta` (128) on every one of those runs, as `T=0` requires. "
          "This analysis keys off `eta`, never the config's `beta`.",
          ""]
    return "\n".join(L)


def main(argv=None):
    args = parse_args(argv)
    cache = args.cache or (args.out_dir / "eta_normalizer_runs.csv")
    if cache.exists() and not args.refresh:
        runs = pd.read_csv(cache)
        print(f"loaded {len(runs)} run aggregates from {cache}")
    else:
        runs = collect(args)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        runs.to_csv(cache, index=False)
        print(f"wrote {cache}")

    reports = []
    for w in ("geo_win", "geo_all"):
        res = evaluate(runs, w, "pres", np.random.default_rng(0), args.n_boot)
        reports.append(render(res, runs, w, args))
        flat = [{k: v for k, v in r.items() if k != "per_env"} for r in res]
        for r in res:
            for env, d in r["per_env"].items():
                flat.append(dict(normalizer=r["normalizer"], env=env, window=w, **d))
        pd.DataFrame(flat).to_csv(
            args.out_dir / f"eta_normalizer_equivalence_{w}.csv", index=False)
    out = ("\n\n---\n\n".join(reports))
    (args.out_dir / "eta_normalizer_equivalence.md").write_text(out)
    print(out)
    print(f"\nwrote {args.out_dir}/eta_normalizer_equivalence.md")


if __name__ == "__main__":
    main()
