#!/usr/bin/env python3
"""Backfill wandb config for the legacy 144-run launch so it looks exactly
like a sweep launched by the new code.

For each wandb run that matches the legacy launch timestamp prefix, we:
  1. Read its recorded ``hp_pack`` path + ``seed_index`` from wandb config.
  2. Load the local pack JSON (old key names).
  3. Translate pack keys to CLI/argparse names (adv_ema_tau -> advantage_ema_tau,
     polyak_tau -> tau (historical CLI name), guidance_mult -> guidance_strength_multiplier,
     kl_budget_val -> kl_budget).
  4. Write per-slot scalars for every hp under their CLI name, per-slot
     ``seed``, plus ``sweep_id`` and ``config_tag``, into ``run.config``.
  5. ``run.update()`` to persist.

After the backfill, ``compute_topsis.py --sweep-id <SWEEP_ID>`` pulls this
cohort directly from wandb (filters on config.sweep_id) with no pack-file
dependency.

Usage:
    python scripts/backfill_sweep_config.py
    python scripts/backfill_sweep_config.py --sweep-id 1 --dry-run
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import wandb

ENTITY = "pnielsen2-harvard"
PROJECT = "diffusion_online_rl"
# Legacy launch identifiers (same ones compute_topsis.py's legacy mode uses).
NAME_PREFIX = "dpmd_2026-04-21_2"
NAME_MIN = "dpmd_2026-04-21_22-44-40"

_OLD_TO_NEW_PACK_KEY = {
    "polyak_tau": "tau",  # these runs used --tau (old CLI name)
    "adv_ema_tau": "advantage_ema_tau",
    "guidance_mult": "guidance_strength_multiplier",
    "kl_budget_val": "kl_budget",
}

_PACK_CACHE: dict = {}


def load_pack(path: str):
    if path in _PACK_CACHE:
        return _PACK_CACHE[path]
    with open(path) as f:
        _PACK_CACHE[path] = json.load(f)
    return _PACK_CACHE[path]


def build_config_tag(per_slot: dict, sweep_id: int) -> str:
    """Match relax.trainer.vmap_off_policy.build_config_tag exactly so
    backfilled and new-sweep tags are byte-identical."""
    ordered = sorted(k for k in per_slot if k != "seed")
    body = "_".join(f"{k}={per_slot[k]:g}" for k in ordered) or "single"
    return f"sweep{sweep_id}_{body}"


def in_legacy_launch(name: str) -> bool:
    if not name.startswith(NAME_PREFIX):
        return False
    base = name.rsplit("-s", 1)[0]
    return base >= NAME_MIN


def per_slot_update(r, sweep_id: int):
    """Compute the config-update dict for one run. Returns None if the run
    is missing the required breadcrumbs (hp_pack path, seed_index)."""
    seed_idx = r.config.get("seed_index")
    hp_pack_path = r.config.get("hp_pack")
    if seed_idx is None or hp_pack_path is None:
        return None
    try:
        pack = load_pack(hp_pack_path)
    except FileNotFoundError:
        return None
    per_slot = {}
    for old_k, values in pack.items():
        new_k = _OLD_TO_NEW_PACK_KEY.get(old_k, old_k)
        per_slot[new_k] = values[seed_idx]
    update = {}
    for k, v in per_slot.items():
        update[k] = v
    update["sweep_id"] = int(sweep_id)
    update["config_tag"] = build_config_tag(per_slot, sweep_id)
    return update


def patch_run(api, run_summary, sweep_id: int, dry_run: bool):
    """Fetch the run object, compute the update, apply + save."""
    try:
        run = api.run(f"{ENTITY}/{PROJECT}/{run_summary.id}")
    except Exception as e:
        return {"id": run_summary.id, "status": f"fetch_err: {e}"}
    update = per_slot_update(run, sweep_id)
    if update is None:
        return {"id": run.id, "status": "skipped_no_breadcrumbs"}
    # Check if it was already patched; if config_tag is already the right one,
    # leave it alone (idempotent re-runs).
    existing_tag = run.config.get("config_tag")
    if existing_tag == update["config_tag"]:
        return {"id": run.id, "status": "already_patched", "tag": existing_tag}
    if dry_run:
        return {"id": run.id, "status": "would_patch", "tag": update["config_tag"]}
    for k, v in update.items():
        run.config[k] = v
    try:
        run.update()
    except Exception as e:
        return {"id": run.id, "status": f"update_err: {e}"}
    return {"id": run.id, "status": "patched", "tag": update["config_tag"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sweep-id", "--sweep_id", type=int, default=1,
                    help="sweep_id to assign to the legacy cohort. Default 1; "
                         "leave the default for this project's first retrofit. "
                         "Pass a new value if sweep_id=1 is already used.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute + print the per-run update plan; do not write.")
    ap.add_argument("--workers", type=int, default=4,
                    help="ThreadPool size for parallel wandb updates.")
    args = ap.parse_args()

    api = wandb.Api(timeout=60)
    print(f"Querying wandb for legacy runs (name prefix {NAME_PREFIX!r}, "
          f"base >= {NAME_MIN!r}) ...")
    all_runs = list(api.runs(f"{ENTITY}/{PROJECT}", order="-created_at", per_page=500))
    runs = [r for r in all_runs if in_legacy_launch(r.name)]
    print(f"  matched {len(runs)} runs")

    if not runs:
        print("Nothing to do.")
        return

    print(f"{'Dry-run' if args.dry_run else 'Patching'} {len(runs)} runs "
          f"with sweep_id={args.sweep_id} ...")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(patch_run, api, r, args.sweep_id, args.dry_run): r for r in runs}
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(runs)}")

    from collections import Counter
    by_status = Counter(r["status"].split(":", 1)[0] for r in results)
    print("\nSummary:")
    for s, c in sorted(by_status.items()):
        print(f"  {s}: {c}")

    # Show first failure for debugging if any
    for r in results:
        if r["status"].startswith(("fetch_err", "update_err")):
            print(f"\nFirst error: {r}")
            break


if __name__ == "__main__":
    main()
