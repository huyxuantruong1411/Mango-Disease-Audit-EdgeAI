#!/usr/bin/env python3
"""Verify that all locked Version 5 (v5) experimental CSV tables and manifests
match the ground-truth manuscript metrics specified in PLAN.md (SEED = 20261022).

Usage:
    python scripts/verify_locked_results.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable, List, Tuple

import pandas as pd


def _check_val(
    df: pd.DataFrame,
    row_filter: Callable[[pd.DataFrame], pd.Series],
    col: str,
    expected: Any,
    tol: float = 1e-4,
) -> Tuple[bool, str]:
    sub = df[row_filter(df)]
    if sub.empty:
        return False, f"Row not found for column '{col}' (expected {expected})"
    actual = sub.iloc[0][col]
    if isinstance(expected, (int, float)) and not isinstance(actual, str):
        try:
            actual_f = float(actual)
            ok = abs(actual_f - float(expected)) <= tol
            return ok, f"actual={actual_f}, expected={expected}"
        except (ValueError, TypeError):
            pass
    actual_s = str(actual).strip()
    expected_s = str(expected).strip()
    ok = actual_s == expected_s or expected_s in actual_s
    return ok, f"actual='{actual_s}', expected='{expected_s}'"


def run_verification(tables_dir: Path, manifests_dir: Path) -> bool:
    checks: List[Tuple[str, bool, str]] = []

    # 0. Manifest check
    manifest_path = manifests_dir / "manifest_MANGO_audited.csv"
    if manifest_path.exists():
        mdf = pd.read_csv(manifest_path)
        n_rows = len(mdf)
        n_groups = int(mdf["group_id"].nunique())
        n_classes = int(mdf["label_name"].nunique())
        checks.append(
            ("Manifest: 4,000 total images across 8 classes", n_rows == 4000 and n_classes == 8, f"rows={n_rows}, classes={n_classes}")
        )
        checks.append(
            ("Manifest: 2,554 unique specimen clusters", n_groups == 2554, f"unique_clusters={n_groups}, expected=2554")
        )
    else:
        checks.append(("Manifest file exists", False, f"Missing {manifest_path}"))

    # 1. Table 1: Dataset Leakage and Group Audit
    t1_path = tables_dir / "Table1_Dataset_Leakage_and_Group_Audit.csv"
    if t1_path.exists():
        t1 = pd.read_csv(t1_path)
        total_imgs = int(t1["Total Images"].sum())
        total_clusters = int(t1["Unique Specimen Clusters (pHash<=6 | HSV cosine>=0.992)"].sum())
        mean_dup_ratio = float(t1["Near-Duplicate / Aug-Mix Ratio (%)"].mean())
        leak_pct = float(t1["Random-Split Test Leakage (%)"].iloc[0])
        checks.append(("Table 1: Total Images = 4,000", total_imgs == 4000, f"actual={total_imgs}, expected=4000"))
        checks.append(("Table 1: Total Unique Specimen Clusters = 2,554", total_clusters == 2554, f"actual={total_clusters}, expected=2554"))
        checks.append(("Table 1: Mean Near-Duplicate Ratio = 36.15%", abs(mean_dup_ratio - 36.15) <= 0.05, f"actual={mean_dup_ratio:.2f}%, expected=36.15%"))
        checks.append(("Table 1: Random-Split Test Leakage = 39.0%", abs(leak_pct - 39.0) <= 0.05, f"actual={leak_pct:.1f}%, expected=39.0%"))
    else:
        checks.append(("Table 1 exists", False, f"Missing {t1_path}"))

    # 2. Table 2: Background Shortcut Audit
    t2_path = tables_dir / "Table2_Background_Shortcut_Audit.csv"
    if t2_path.exists():
        t2 = pd.read_csv(t2_path)
        for label, substr, col, exp in [
            ("Table 2: View 2 Segmented Organ Only Acc (94.84%)", "2. Segmented Organ Only", "Group-Split Test Acc (%)", 94.84),
            ("Table 2: View 2 Segmented Organ Only F1 (0.9440)", "2. Segmented Organ Only", "Group-Split Macro-F1", 0.9440),
            ("Table 2: View 3 Background Only Acc (94.84%)", "3. Background Only", "Group-Split Test Acc (%)", 94.84),
            ("Table 2: View 3 Background Only F1 (0.9396)", "3. Background Only", "Group-Split Macro-F1", 0.9396),
            ("Table 2: View 4 Four Corner Patches Acc (85.08%)", "4. Four Corner Patches", "Group-Split Test Acc (%)", 85.08),
            ("Table 2: View 5 File Metadata Only Acc (59.14%)", "5. Image File Metadata", "Group-Split Test Acc (%)", 59.14),
        ]:
            ok, msg = _check_val(t2, lambda d, s=substr: d["Input Representation View"].astype(str).str.contains(s, regex=False), col, exp)
            checks.append((label, ok, msg))
    else:
        checks.append(("Table 2 exists", False, f"Missing {t2_path}"))

    # 3. Table 3: Progressive Ablation Study (M1 to M11)
    t3_path = tables_dir / "Table3_Progressive_Ablation_Study.csv"
    if t3_path.exists():
        t3 = pd.read_csv(t3_path)
        for label, substr, col, exp in [
            ("Table 3: M1 HSV Only Acc (90.38%)", "M1.", "Test Acc (%)", 90.38),
            ("Table 3: M2 HSV + Fast Multi-Scale LBP Acc (97.35%)", "M2.", "Test Acc (%)", 97.35),
            ("Table 3: M3 HSV + Vectorized Sobel HOG Acc (83.40%)", "M3.", "Test Acc (%)", 83.40),
            ("Table 3: M4 Baseline HSV+LBP+HOG Acc (86.61%)", "M4.", "Test Acc (%)", 86.61),
            ("Table 3: M9 Full 2,003D Uncompressed Acc (92.05%)", "M9.", "Test Acc (%)", 92.05),
            ("Table 3: M10 Proposed Stage-1 Block-PCA 303D Acc (97.91%)", "M10.", "Test Acc (%)", 97.91),
            ("Table 3: M10 Proposed Stage-1 Block-PCA 303D F1 (0.9789)", "M10.", "Test Macro-F1", 0.9789),
            ("Table 3: M11 Segmented Organ ROI No-BG Acc (96.37%)", "M11.", "Test Acc (%)", 96.37),
            ("Table 3: M11 Segmented Organ ROI No-BG F1 (0.9549)", "M11.", "Test Macro-F1", 0.9549),
        ]:
            ok, msg = _check_val(t3, lambda d, s=substr: d["Configuration"].astype(str).str.contains(s, regex=False), col, exp)
            checks.append((label, ok, msg))
    else:
        checks.append(("Table 3 exists", False, f"Missing {t3_path}"))

    # 4. Table 4: Confidence-Gated Routing & Calibration
    t4_path = tables_dir / "Table4_Confidence_Gated_and_Calibration.csv"
    if t4_path.exists():
        t4 = pd.read_csv(t4_path)
        for label, substr, col, exp in [
            ("Table 4: Stage-1 LR Only Acc (97.91%)", "Stage-1 LR Only", "Test Accuracy (%)", 97.91),
            ("Table 4: Stage-1 LR Only ECE (0.0094)", "Stage-1 LR Only", "Calibration ECE", 0.0094),
            ("Table 4: Val-Selected tau=0.65 Routed (1.53%)", "tau=0.65", "Stage-2 Routed Samples (%)", 1.53),
            ("Table 4: Val-Selected tau=0.65 Acc (98.47%)", "tau=0.65", "Test Accuracy (%)", 98.47),
            ("Table 4: Val-Selected tau=0.65 Macro-F1 (0.9836)", "tau=0.65", "Group-Split Macro-F1", 0.9836),
            ("Table 4: Val-Selected tau=0.65 ECE (0.0104)", "tau=0.65", "Calibration ECE", 0.0104),
            ("Table 4: Val-Selected tau=0.65 AURC (0.00158)", "tau=0.65", "Selective Risk AURC", 0.00158),
            ("Table 4: High-Sensitivity tau=0.85 Routed (4.32%)", "tau=0.85", "Stage-2 Routed Samples (%)", 4.32),
            ("Table 4: High-Sensitivity tau=0.85 Acc (98.74%)", "tau=0.85", "Test Accuracy (%)", 98.74),
            ("Table 4: High-Sensitivity tau=0.85 Macro-F1 (0.9859)", "tau=0.85", "Group-Split Macro-F1", 0.9859),
            ("Table 4: Random-Split Reference Acc (99.00%)", "REFERENCE", "Test Accuracy (%)", 99.00),
            ("Table 4: Random-Split Reference Macro-F1 (0.9900)", "REFERENCE", "Group-Split Macro-F1", 0.9900),
        ]:
            ok, msg = _check_val(t4, lambda d, s=substr: d["Routing Mode"].astype(str).str.contains(s, regex=False), col, exp)
            checks.append((label, ok, msg))
    else:
        checks.append(("Table 4 exists", False, f"Missing {t4_path}"))

    # 5. Table 5: Multi-Severity Perturbation Robustness
    t5_path = tables_dir / "Table5_MultiSeverity_Robustness.csv"
    if t5_path.exists():
        t5 = pd.read_csv(t5_path)
        for label, substr, col, exp in [
            ("Table 5: Clean Proposed Quality-Gated F1 (0.9836)", "Clean held-out test", "4. Proposed Quality-Gated", 0.9836),
            ("Table 5: Noise sigma=5 Proposed F1 (0.9249)", "sigma=5", "4. Proposed Quality-Gated", 0.9249),
            ("Table 5: Noise sigma=10 Proposed F1 (0.8647)", "sigma=10", "4. Proposed Quality-Gated", 0.8647),
            ("Table 5: Noise sigma=10 Gain vs Raw (+0.3533)", "sigma=10", "Gain vs Raw", 0.3533),
            ("Table 5: Noise sigma=20 Proposed F1 (0.6104)", "sigma=20", "4. Proposed Quality-Gated", 0.6104),
            ("Table 5: Glare r=22px Proposed F1 (0.9812)", "radius=22", "4. Proposed Quality-Gated", 0.9812),
            ("Table 5: Glare r=35px Proposed F1 (0.9770)", "radius=35", "4. Proposed Quality-Gated", 0.9770),
            ("Table 5: Low-light -40% Proposed F1 (0.8012)", "Low-light", "4. Proposed Quality-Gated", 0.8012),
        ]:
            ok, msg = _check_val(t5, lambda d, s=substr: d["Perturbation Scenario"].astype(str).str.contains(s, regex=False), col, exp)
            checks.append((label, ok, msg))
    else:
        checks.append(("Table 5 exists", False, f"Missing {t5_path}"))

    # 6. Table 6: Fair End-to-End CPU Benchmark
    t6_path = tables_dir / "Table6_Fair_EndToEnd_CPU_Benchmark.csv"
    if t6_path.exists():
        t6 = pd.read_csv(t6_path)
        for label, substr, col, exp in [
            ("Table 6: Stage-1 Only Mean Latency (21.58 ms)", "Proposed Stage-1 Only", "End-to-End Mean (ms/img)", 21.58),
            ("Table 6: Stage-1 Only Throughput (46.3 img/s)", "Proposed Stage-1 Only", "Throughput (img/s)", 46.3),
            ("Table 6: Two-Stage Gated Mean Latency (21.74 ms)", "Proposed Two-Stage", "End-to-End Mean (ms/img)", 21.74),
            ("Table 6: Two-Stage Gated Throughput (46.0 img/s)", "Proposed Two-Stage", "Throughput (img/s)", 46.0),
            ("Table 6: MobileNetV3-Small Latency (48.70 ms)", "mobilenet_v3_small", "End-to-End Mean (ms/img)", 48.70),
            ("Table 6: MobileNetV2 Latency (56.42 ms)", "mobilenet_v2", "End-to-End Mean (ms/img)", 56.42),
            ("Table 6: ResNet50 Latency (272.04 ms)", "resnet50", "End-to-End Mean (ms/img)", 272.04),
        ]:
            ok, msg = _check_val(t6, lambda d, s=substr: d["Model / Pipeline"].astype(str).str.contains(s, regex=False), col, exp)
            checks.append((label, ok, msg))
    else:
        checks.append(("Table 6 exists", False, f"Missing {t6_path}"))

    # 7. Table 7: External Dataset Validation (MLDID, N=3,000)
    t7_path = tables_dir / "Table7_External_Dataset_Validation.csv"
    if t7_path.exists():
        t7 = pd.read_csv(t7_path)
        for label, substr, col, exp in [
            ("Table 7: Source-view In-Domain 5C Acc (99.34%)", "Source-view in-domain", "Accuracy (%)", 99.34),
            ("Table 7: Source-view In-Domain 5C F1 (0.9929)", "Source-view in-domain", "Macro-F1", 0.9929),
            ("Table 7: MLDID Source-view Zero-Shot Acc (31.27%)", "MLDID source-view Zero-Shot", "Accuracy (%)", 31.27),
            ("Table 7: MLDID Source-view Zero-Shot F1 (0.2886)", "MLDID source-view Zero-Shot", "Macro-F1", 0.2886),
            ("Table 7: Canonical-pose In-Domain 5C Acc (96.04%)", "Canonical-pose in-domain", "Accuracy (%)", 96.04),
            ("Table 7: MLDID Canonical-pose Zero-Shot Acc (42.63%)", "MLDID Canonical-pose Zero-Shot", "Accuracy (%)", 42.63),
            ("Table 7: MLDID Canonical-pose Zero-Shot F1 (0.4395)", "MLDID Canonical-pose Zero-Shot", "Macro-F1", 0.4395),
            ("Table 7: MLDID Canonical-pose + CORAL Acc (31.43%)", "MLDID Canonical-pose + CORAL", "Accuracy (%)", 31.43),
            ("Table 7: MLDID Canonical-pose + CORAL F1 (0.3274)", "MLDID Canonical-pose + CORAL", "Macro-F1", 0.3274),
        ]:
            ok, msg = _check_val(t7, lambda d, s=substr: d["Protocol"].astype(str).str.contains(s, regex=False), col, exp)
            checks.append((label, ok, msg))
    else:
        checks.append(("Table 7 exists", False, f"Missing {t7_path}"))

    print("=" * 92)
    print("LOCKED VERSION 5 (v5) GROUND-TRUTH METRIC VERIFICATION REPORT (SEED = 20261022)")
    print("=" * 92)
    passed = 0
    for name, ok, detail in checks:
        status = "[PASS]" if ok else "[FAIL]"
        if ok:
            passed += 1
        print(f"{status:6s} | {name:56s} | {detail}")
    print("-" * 92)
    print(f"Summary: {passed}/{len(checks)} checks passed.")
    print("=" * 92)
    return passed == len(checks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify locked v5 experimental tables.")
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("results/tables"),
        help="Path to directory containing Table1..Table8 CSV files.",
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path("results/manifests"),
        help="Path to directory containing manifest_MANGO_audited.csv.",
    )
    args = parser.parse_args()
    success = run_verification(args.tables_dir, args.manifests_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
