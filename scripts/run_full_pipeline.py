#!/usr/bin/env python3
"""Master CLI entry point to verify locked Version 5 results or execute the full
end-to-end audit, feature extraction, benchmark evaluation, and EDA pipeline
when raw datasets (`MangoLeafBD` and `MLDID`) are mounted locally.

Usage:
    python scripts/run_full_pipeline.py --verify-only
    python scripts/run_full_pipeline.py --mango-root "data/MangoLeafBD" --mldid-root "data/MLDID"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mango_edge_cv.audit import build_audited_manifest
from mango_edge_cv.config import MANGO_MAIN_CLASSES, SEED
from mango_edge_cv.eda import run_image_dataset_eda
from scripts.generate_paper_assets import main as generate_assets_main
from scripts.verify_locked_results import run_verification


def main() -> None:
    parser = argparse.ArgumentParser(description="Mango-Disease-Audit-EdgeAI Master Pipeline Runner")
    parser.add_argument("--mango-root", type=Path, default=None, help="Path to raw MangoLeafBD dataset directory.")
    parser.add_argument("--mldid-root", type=Path, default=None, help="Path to raw MLDID dataset directory.")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify locked Version 5 (v5) tables and regenerate LaTeX/figure assets without re-extracting raw images.",
    )
    parser.add_argument(
        "--run-eda",
        action="store_true",
        help="Run comprehensive EDA on raw dataset directories if present.",
    )
    args = parser.parse_args()

    print(f"[INFO] Mango-Disease-Audit-EdgeAI Pipeline (SEED = {SEED})")

    if args.verify_only or (args.mango_root is None and args.mldid_root is None):
        ok = run_verification(REPO_ROOT / "results" / "tables", REPO_ROOT / "results" / "manifests")
        generate_assets_main()
        sys.exit(0 if ok else 1)

    if args.mango_root is not None and args.mango_root.exists():
        print(f"[INFO] Auditing MangoLeafBD at: {args.mango_root}")
        df_audit = build_audited_manifest(
            str(args.mango_root),
            MANGO_MAIN_CLASSES,
            str(REPO_ROOT / "results" / "manifests" / "manifest_MANGO_rebuilt.csv"),
        )
        print(f"[INFO] Audit complete: {len(df_audit)} rows, {df_audit['group_id'].nunique()} unique specimen clusters.")
        if args.run_eda:
            run_image_dataset_eda(
                dataset_name="MangoLeafBD",
                df_records=df_audit,
                class_list=MANGO_MAIN_CLASSES,
                out_dir=str(REPO_ROOT / "results" / "eda"),
            )

    ok = run_verification(REPO_ROOT / "results" / "tables", REPO_ROOT / "results" / "manifests")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
