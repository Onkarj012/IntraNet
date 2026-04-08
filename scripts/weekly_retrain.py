#!/usr/bin/env python3
"""
Weekly Retrain Script — run every weekend to retrain with latest data.

Usage:
    crontab: 0 6 * * 0 cd /path/to/project && python scripts/weekly_retrain.py

    python scripts/weekly_retrain.py --horizon H60
    python scripts/weekly_retrain.py --horizon H60 --skip-data-sync
"""

import argparse
import subprocess
import sys
import shutil
from datetime import datetime
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HORIZONS = ["H15", "H30", "H60"]


def run_step(name: str, cmd: list, **kwargs):
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"CMD: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        print(f"ERROR: {name} failed with exit code {result.returncode}")
        sys.exit(1)
    print(f"OK: {name} completed")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Weekly retrain for IntradayNet LightGBM")
    parser.add_argument("--horizon", default="H60", choices=["H15", "H30", "H60", "all"])
    parser.add_argument("--keep-versions", type=int, default=4)
    parser.add_argument("--skip-data-sync", action="store_true",
                        help="Skip data sync step")
    parser.add_argument("--skip-prebatch", action="store_true",
                        help="Skip prebatching (use existing npz)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers for prebatching")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    horizons = HORIZONS if args.horizon == "all" else [args.horizon]

    print(f"{'='*60}")
    print(f"WEEKLY RETRAIN — {timestamp}")
    print(f"Horizons: {horizons}")
    print(f"{'='*60}")

    output_base = PROJECT_ROOT / "runs" / f"lgbm_v2_{timestamp}"

    # Step 1: Data sync
    if not args.skip_data_sync:
        sync_script = PROJECT_ROOT / "scripts" / "sync_data.py"
        if sync_script.exists():
            run_step("Data Sync", [sys.executable, str(sync_script)])
        else:
            print("INFO: sync_data.py not found — skipping data sync")
    else:
        print("INFO: Skipping data sync (--skip-data-sync)")

    # Step 2: Prebatch
    if not args.skip_prebatch:
        output_dir = PROJECT_ROOT / "prebatched_v2"
        run_step(
            "Prebatching Features",
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "prebatch_lgbm_v2.py"),
                "--output", str(output_dir),
                "--workers", str(args.workers),
            ],
            cwd=PROJECT_ROOT,
        )
    else:
        print("INFO: Skipping prebatch (--skip-prebatch)")

    prebatched_path = PROJECT_ROOT / "prebatched_v2" / "lgbm_dataset.npz"
    if not prebatched_path.exists():
        print(f"ERROR: Prebatched data not found at {prebatched_path}")
        sys.exit(1)

    # Step 3: Adversarial validation (before training)
    print(f"\n{'='*60}")
    print("STEP: Adversarial Validation (pre-train)")
    print(f"{'='*60}")
    adv_result = run_step(
        "Adversarial Validation",
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "adversarial_validation.py"),
            "--train-data", str(prebatched_path),
            "--recent-days", "60",
            "--n-stocks", "100",
        ],
        cwd=PROJECT_ROOT,
    )

    # Steps 4+: Train each horizon
    for h in horizons:
        print(f"\n{'='*60}")
        print(f"TRAINING: {h}")
        print(f"{'='*60}")

        output_dir = output_base / h
        run_step(
            f"Train {h}",
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "train_lgbm_v2.py"),
                "--data", str(prebatched_path),
                "--output", str(output_dir),
                "--horizon", h,
                "--n-folds", "4",
            ],
            cwd=PROJECT_ROOT,
        )

    # Step 5: Update model symlink
    print(f"\n{'='*60}")
    print("STEP: Update Model Symlink")
    print(f"{'='*60}")
    latest_link = PROJECT_ROOT / "runs" / "lgbm_latest"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(output_base.resolve())
    print(f"  runs/lgbm_latest -> {output_base}")

    # Step 6: Cleanup old versions
    print(f"\n{'='*60}")
    print("STEP: Cleanup Old Versions")
    print(f"{'='*60}")
    versions = sorted(
        (PROJECT_ROOT / "runs").glob("lgbm_v2_?????????*"),
        key=lambda p: p.stat().st_mtime,
    )
    if len(versions) > args.keep_versions:
        for old in versions[: -args.keep_versions]:
            print(f"  Removing: {old}")
            shutil.rmtree(old)
    else:
        print(f"  Keeping all {len(versions)} versions (max={args.keep_versions})")

    # Save retrain log
    log = {
        "timestamp": timestamp,
        "horizons": horizons,
        "output_dir": str(output_base),
        "args": vars(args),
    }
    log_path = output_base / "retrain_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'='*60}")
    print("RETRAIN COMPLETE")
    print(f"Output: {output_base}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
