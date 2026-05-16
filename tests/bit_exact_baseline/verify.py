"""Verify a new run is bit-exact against the baseline.

Compares every byte of each new run's episode_returns.csv against the
corresponding baseline CSV in tests/bit_exact_baseline/baselines/<env>/.
Comparison is by prefix: the new run may have more rows than the baseline,
but every row the baseline contains must match byte-for-byte.

Usage:
    # Verify the most recent sweep58-style run for every env
    python tests/bit_exact_baseline/verify.py

    # Verify a specific run dir tree (the script will look for matching
    # logs/<env>/dpmd_*_s0_/episode_returns.csv under it)
    python tests/bit_exact_baseline/verify.py --logs-root /path/to/some/logs

    # Verify a single CSV against its baseline
    python tests/bit_exact_baseline/verify.py --csv logs/Ant-v3/dpmd_.../episode_returns.csv

Exit code 0 if all checked envs are bit-exact (or skipped because no new
run was found), 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
BASELINE_DIR = THIS_DIR / "baselines"
REPO_ROOT = THIS_DIR.parent.parent
DEFAULT_LOGS_ROOT = REPO_ROOT / "logs"

ENVS = ["Ant-v3", "Humanoid-v3", "HalfCheetah-v3", "Walker2d-v3"]


def sha256_prefix(path: Path, nbytes: int) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()


def compare(baseline_csv: Path, new_csv: Path) -> tuple[bool, str]:
    """Return (is_bit_exact_prefix, message)."""
    if not baseline_csv.exists():
        return False, f"baseline missing: {baseline_csv}"
    if not new_csv.exists():
        return False, f"new run missing: {new_csv}"

    base_size = baseline_csv.stat().st_size
    new_size = new_csv.stat().st_size
    cmp_size = min(base_size, new_size)
    partial = new_size < base_size

    h_base = sha256_prefix(baseline_csv, cmp_size)
    h_new = sha256_prefix(new_csv, cmp_size)

    if h_base == h_new:
        # Count rows actually matched (lines fully contained in cmp_size).
        with new_csv.open("rb") as f:
            data = f.read(cmp_size)
        # Number of complete lines = number of '\n' bytes; subtract header if present.
        matched_newlines = data.count(b"\n")
        matched_rows = max(matched_newlines - 1, 0)  # minus header
        with baseline_csv.open() as f:
            base_rows = sum(1 for _ in f) - 1
        if partial:
            return True, (
                f"BIT-EXACT SO FAR (matched {cmp_size} bytes / {matched_rows} rows; "
                f"new run has {new_size} bytes vs baseline {base_size} bytes "
                f"/ {base_rows} rows -- still running)"
            )
        return True, (
            f"BIT-EXACT (matched {cmp_size} bytes / {base_rows} rows of baseline; "
            f"new run has {new_size} bytes total)"
        )
    else:
        # Find the first differing line for diagnostics.
        with baseline_csv.open() as fb, new_csv.open() as fn:
            for i, (lb, ln) in enumerate(zip(fb, fn), start=1):
                if lb != ln:
                    return False, (
                        f"FIRST DIFF at line {i}:\n"
                        f"  baseline: {lb.rstrip()}\n"
                        f"  new run : {ln.rstrip()}"
                    )
        return False, "differ but could not locate first-diff line (unexpected)"


def latest_run_csv(logs_root: Path, env: str) -> Path | None:
    """Find the most recent dpmd_*_s0_/episode_returns.csv under logs/<env>/."""
    env_dir = logs_root / env
    if not env_dir.is_dir():
        return None
    runs = sorted(env_dir.glob("dpmd_*_s0_"), key=lambda p: p.name, reverse=True)
    for r in runs:
        csv = r / "episode_returns.csv"
        if csv.exists() and csv.stat().st_size > 0:
            return csv
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS_ROOT,
                   help="Directory whose <env>/dpmd_*_s0_/episode_returns.csv "
                        "files will be checked (default: repo logs/).")
    p.add_argument("--csv", type=Path, default=None,
                   help="Compare a single CSV against its baseline (env inferred "
                        "from the parent dir name).")
    p.add_argument("--envs", nargs="*", default=ENVS,
                   help=f"Restrict comparison to these envs (default: {ENVS}).")
    args = p.parse_args()

    print(f"baseline dir: {BASELINE_DIR}")

    fails = 0
    checked = 0

    if args.csv is not None:
        env = args.csv.parents[1].name  # logs/<env>/dpmd_.../episode_returns.csv
        baseline = BASELINE_DIR / env / "episode_returns.csv"
        ok, msg = compare(baseline, args.csv)
        print(f"\n[{env}] new={args.csv}")
        print(f"  -> {msg}")
        return 0 if ok else 1

    for env in args.envs:
        baseline = BASELINE_DIR / env / "episode_returns.csv"
        new_csv = latest_run_csv(args.logs_root, env)
        print(f"\n[{env}]")
        print(f"  baseline: {baseline}")
        print(f"  new run : {new_csv}")
        if new_csv is None:
            print(f"  -> SKIP (no new run found under {args.logs_root}/{env}/)")
            continue
        checked += 1
        ok, msg = compare(baseline, new_csv)
        print(f"  -> {msg}")
        if not ok:
            fails += 1

    print(f"\nsummary: checked={checked} fails={fails}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
