#!/usr/bin/env python3
"""Local helper to stage Modal trace artifacts into modal/artifacts/<job-name>/.

This is intentionally local-only for now; no remote Modal download flow is implemented.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = REPO_ROOT / "modal" / "artifacts" / "trace_job"
DEFAULT_ARTIFACTS_ROOT = REPO_ROOT / "modal" / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_name", help="Destination subdirectory under modal/artifacts/")
    parser.add_argument(
        "--source-dir",
        default=str(DEFAULT_SOURCE_DIR),
        help=f"Source artifact directory (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--artifacts-root",
        default=str(DEFAULT_ARTIFACTS_ROOT),
        help=f"Artifacts root directory (default: {DEFAULT_ARTIFACTS_ROOT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite destination if it already exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    source_dir = Path(args.source_dir).resolve()
    artifacts_root = Path(args.artifacts_root).resolve()
    destination_dir = artifacts_root / args.job_name

    if not source_dir.is_dir():
        print(f"[download_artifacts] ERROR: source directory not found: {source_dir}")
        return 1

    artifacts_root.mkdir(parents=True, exist_ok=True)

    if destination_dir.exists():
        if not args.force:
            print(
                "[download_artifacts] ERROR: destination already exists. "
                f"Use --force to overwrite: {destination_dir}"
            )
            return 1
        shutil.rmtree(destination_dir)

    shutil.copytree(source_dir, destination_dir)

    print(f"[download_artifacts] source={source_dir}")
    print(f"[download_artifacts] destination={destination_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
