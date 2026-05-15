#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import wandb

ENTITY = "pnielsen2-harvard"
PROJECT = "diffusion_online_rl"
_NON_HP_CONFIG_KEYS = frozenset({
    "seed", "seed_index", "sweep_id", "config_tag", "config_tag_keys",
    "parallel_seeds", "hp_pack", "hp_pack_inline", "env",
})


def _normalize_value(v: Any):
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, int):
        return ("int", v)
    if isinstance(v, float):
        if math.isnan(v):
            return ("float", "NaN")
        if math.isinf(v):
            return ("float", "Infinity" if v > 0 else "-Infinity")
        return ("float", format(v, ".17g"))
    if isinstance(v, str):
        return ("str", v)
    if v is None:
        return ("none", None)
    try:
        hash(v)
    except TypeError:
        return None
    return (type(v).__name__, v)


def _display_value(v: Any) -> str:
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        if math.isinf(v):
            return "Infinity" if v > 0 else "-Infinity"
        return format(v, "g")
    if v is None:
        return "None"
    return str(v)


def _fetch_full_runs(api: wandb.Api, sweep_id: int, workers: int):
    stubs = list(api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={"config.sweep_id": int(sweep_id)},
        per_page=500,
    ))
    if not stubs:
        return []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda rid: api.run(f"{ENTITY}/{PROJECT}/{rid}"), [r.id for r in stubs]))


def _discover_group_keys(runs, forced_keys=None):
    if forced_keys:
        return [k for k in forced_keys if k not in _NON_HP_CONFIG_KEYS]
    value_sets = {}
    for r in runs:
        for k, v in r.config.items():
            if k in _NON_HP_CONFIG_KEYS:
                continue
            nv = _normalize_value(v)
            if nv is None:
                continue
            value_sets.setdefault(k, set()).add(nv)
    return sorted(k for k, vs in value_sets.items() if len(vs) > 1)


def _group_signature(run, keys):
    return tuple((k, _normalize_value(run.config.get(k))) for k in keys)


def _group_display(run, keys):
    return {k: run.config.get(k) for k in keys}


def _group_sort_key(group_runs):
    created = [str(getattr(r, "created_at", "") or "") for r in group_runs]
    names = [str(getattr(r, "name", "") or "") for r in group_runs]
    return min(created) if any(created) else min(names)


def _next_unused_sweep_id(api: wandb.Api, reserved: set[int]) -> int:
    n = 1
    while n < 10000:
        if n in reserved:
            n += 1
            continue
        runs = api.runs(f"{ENTITY}/{PROJECT}", filters={"config.sweep_id": n}, per_page=1)
        if len(runs) == 0:
            reserved.add(n)
            return n
        n += 1
    raise RuntimeError("failed to find an unused sweep_id below 10000")


def _audit_sweep(runs, sweep_id: int, forced_keys=None):
    keys = _discover_group_keys(runs, forced_keys=forced_keys)
    grouped = defaultdict(list)
    for r in runs:
        grouped[_group_signature(r, keys)].append(r)
    groups = []
    for sig, grp in grouped.items():
        sample = grp[0]
        groups.append({
            "signature": sig,
            "display": _group_display(sample, keys),
            "runs": grp,
            "count": len(grp),
            "first_key": _group_sort_key(grp),
        })
    groups.sort(key=lambda g: g["first_key"])
    return {
        "sweep_id": int(sweep_id),
        "runs": runs,
        "group_keys": keys,
        "groups": groups,
    }


def _planned_updates(api: wandb.Api, audits):
    reserved = {a["sweep_id"] for a in audits}
    plans = []
    for audit in audits:
        if len(audit["groups"]) <= 1:
            plans.append({"audit": audit, "assignments": []})
            continue
        assignments = []
        for idx, group in enumerate(audit["groups"]):
            target_sweep_id = audit["sweep_id"] if idx == 0 else _next_unused_sweep_id(api, reserved)
            assignments.append({
                "group": group,
                "target_sweep_id": target_sweep_id,
                "target_config_tag": f"sweep{target_sweep_id}_single",
            })
        plans.append({"audit": audit, "assignments": assignments})
    return plans


def _patch_one(api: wandb.Api, run_id: str, target_sweep_id: int, target_config_tag: str, dry_run: bool):
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")
    current_sweep_id = run.config.get("sweep_id")
    current_config_tag = run.config.get("config_tag")
    if current_sweep_id == target_sweep_id and current_config_tag == target_config_tag:
        return {"id": run_id, "status": "already_ok", "target_sweep_id": target_sweep_id}
    if dry_run:
        return {"id": run_id, "status": "would_patch", "target_sweep_id": target_sweep_id}
    run.config["sweep_id"] = int(target_sweep_id)
    run.config["config_tag"] = target_config_tag
    run.update()
    return {"id": run_id, "status": "patched", "target_sweep_id": target_sweep_id}


def _recompute_topsis(python_bin: str, sweep_ids):
    script_dir = Path(__file__).resolve().parent
    compute_script = script_dir / "compute_topsis.py"
    for sweep_id in sweep_ids:
        subprocess.run([python_bin, str(compute_script), "--sweep-id", str(sweep_id)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-ids", "--sweep_ids", type=int, nargs="+", required=True)
    ap.add_argument("--group-by", "--group_by", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--recompute-topsis", "--recompute_topsis", action="store_true")
    ap.add_argument("--python-bin", "--python_bin", default=sys.executable)
    args = ap.parse_args()

    api = wandb.Api(timeout=60)
    audits = []
    for sweep_id in args.sweep_ids:
        runs = _fetch_full_runs(api, sweep_id, args.workers)
        audit = _audit_sweep(runs, sweep_id, forced_keys=args.group_by)
        audits.append(audit)

    plans = _planned_updates(api, audits)

    any_split = False
    affected_sweep_ids = set()
    print()
    for plan in plans:
        audit = plan["audit"]
        print(f"sweep {audit['sweep_id']}: {len(audit['runs'])} runs")
        print(f"  varying keys: {audit['group_keys']}")
        if len(audit["groups"]) <= 1:
            print("  no mixed configs detected")
            print()
            continue
        any_split = True
        for assignment in plan["assignments"]:
            group = assignment["group"]
            affected_sweep_ids.add(assignment["target_sweep_id"])
            print(
                f"  -> sweep {assignment['target_sweep_id']} | runs={group['count']} | "
                f"first={group['first_key']} | config="
                f"{{{', '.join(f'{k}={_display_value(v)}' for k, v in group['display'].items())}}}"
            )
        print()

    if not any_split:
        print("nothing to patch")
        return

    patch_jobs = []
    for plan in plans:
        for assignment in plan["assignments"]:
            for run in assignment["group"]["runs"]:
                patch_jobs.append((run.id, assignment["target_sweep_id"], assignment["target_config_tag"]))

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(_patch_one, api, run_id, sweep_id, config_tag, args.dry_run): (run_id, sweep_id)
            for run_id, sweep_id, config_tag in patch_jobs
        }
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 20 == 0 or done == len(futs):
                print(f"patched {done}/{len(futs)}")

    counts = Counter(r["status"] for r in results)
    print()
    for status, count in sorted(counts.items()):
        print(f"{status}: {count}")

    if args.recompute_topsis and not args.dry_run:
        print()
        print(f"recomputing TOPSIS for sweep ids: {sorted(affected_sweep_ids)}")
        _recompute_topsis(args.python_bin, sorted(affected_sweep_ids))


if __name__ == "__main__":
    main()
