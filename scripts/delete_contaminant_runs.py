#!/usr/bin/env python3
"""Delete the 216 wandb runs that contaminated sweep_id=1.

The contaminated launch (2026-04-22 19:19) inherited sweep_id=1 because
launch.py's old next_unused_sweep_id queried by 'group' under the
pre-refactor schema, which has no runs → always returned 1. That collided
with the 144 legacy runs we'd already backfilled to sweep_id=1.

Discriminator: the contaminants have config.q_critic_agg_idx set (from the
new pack schema), the legacy backfilled runs don't. This is exact -- no
timestamp / name-prefix heuristics needed.

Dry-run by default. Pass --apply to actually delete.
"""

from __future__ import annotations

import argparse
import wandb

from relax.utils.fs import WANDB_ENTITY as ENTITY, WANDB_PROJECT as PROJECT


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--apply", action="store_true",
                    help="Actually delete. Without this flag we only list.")
    ap.add_argument("--sweep-id", type=int, default=1,
                    help="Which sweep_id bucket to scrub (default 1).")
    args = ap.parse_args()

    api = wandb.Api(timeout=60)
    runs = list(api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={
            "config.sweep_id": args.sweep_id,
            "config.q_critic_agg_idx": {"$exists": True},
        },
        per_page=500,
    ))
    print(f"Matched {len(runs)} runs with sweep_id={args.sweep_id} "
          f"AND config.q_critic_agg_idx present.")

    if not runs:
        print("Nothing to do.")
        return

    # Sanity-check separator against the legacy cohort that should survive.
    legacy = list(api.runs(
        f"{ENTITY}/{PROJECT}",
        filters={
            "config.sweep_id": args.sweep_id,
            "config.q_critic_agg_idx": {"$exists": False},
        },
        per_page=500,
    ))
    print(f"Legacy runs (sweep_id={args.sweep_id}, no q_critic_agg_idx) "
          f"that will be left untouched: {len(legacy)}")

    if not args.apply:
        print("\n[dry-run] would delete:")
        for r in runs[:5]:
            print(f"  {r.id}  {r.name}  state={r.state}  "
                  f"created_at={r.created_at}")
        if len(runs) > 5:
            print(f"  ... and {len(runs) - 5} more.")
        print("\nRe-run with --apply to actually delete.")
        return

    print(f"\nDeleting {len(runs)} runs ...")
    ok = 0
    errs = []
    for i, r in enumerate(runs, 1):
        try:
            r.delete()
            ok += 1
        except Exception as e:
            errs.append((r.id, str(e)))
        if i % 25 == 0:
            print(f"  deleted {i}/{len(runs)}")
    print(f"\nDone. Deleted {ok}/{len(runs)} runs.")
    if errs:
        print(f"Errors on {len(errs)} runs; first 3: {errs[:3]}")


if __name__ == "__main__":
    main()
