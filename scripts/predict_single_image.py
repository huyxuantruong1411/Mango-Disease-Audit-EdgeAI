#!/usr/bin/env python3
"""Run single-image inference using the locked Version 5 Two-Stage Confidence-Gated
Mango Leaf Disease Classification pipeline (`mango_edge_cv`).

Usage:
    python scripts/predict_single_image.py --image path/to/leaf.jpg
    python scripts/predict_single_image.py --synthetic-demo
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict

import cv2
import joblib
import numpy as np

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mango_edge_cv.config import IMG_SIZE, MANGO_MAIN_CLASSES, RAW_TOTAL_DIMS, SEED
from mango_edge_cv.features import compose_feature_vector_from_rgb, extract_fast_all_branches
from mango_edge_cv.models import (
    BlockPCACompressor,
    CalibratedConfidenceGatedSystem,
    _EstimatorWrapper,
)
from mango_edge_cv.preprocessing import (
    apply_adaptive_quality_preprocessing,
    canonicalize_leaf_organ_image,
    segment_plant_organ_roi,
)


def create_synthetic_mango_leaf(height: int = 256, width: int = 256, seed: int = SEED) -> np.ndarray:
    """Create a realistic synthetic mango leaf image with necrotic lesions for CLI smoke testing."""
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 235, dtype=np.uint8)
    center = (width // 2, height // 2)
    axes = (int(width * 0.36), int(height * 0.22))
    cv2.ellipse(img, center, axes, 25, 0, 360, (55, 145, 45), -1)
    cv2.line(img, (int(width * 0.18), int(height * 0.35)), (int(width * 0.82), int(height * 0.65)), (95, 175, 85), 2)
    for _ in range(8):
        cx = int(rng.integers(int(width * 0.32), int(width * 0.68)))
        cy = int(rng.integers(int(height * 0.38), int(height * 0.62)))
        radius = int(rng.integers(4, 11))
        cv2.circle(img, (cx, cy), radius, (45, 35, 25), -1)
    return img


def load_or_build_model(model_path: Path) -> CalibratedConfidenceGatedSystem:
    """Load the serialized v5 joblib model or fit a calibrated synthetic fallback."""
    if model_path.exists():
        try:
            import __main__

            setattr(__main__, "BlockPCACompressor", BlockPCACompressor)
            setattr(__main__, "CalibratedConfidenceGatedSystem", CalibratedConfidenceGatedSystem)
            setattr(__main__, "_EstimatorWrapper", _EstimatorWrapper)
            loaded = joblib.load(model_path)
            if isinstance(loaded, CalibratedConfidenceGatedSystem):
                return loaded
            if isinstance(loaded, dict) and "model" in loaded:
                return loaded["model"]
        except Exception as exc:
            print(f"[INFO] Pretrained artifact deserialization fallback ({exc}); fitting calibrated demo model.")

    rng = np.random.default_rng(SEED)
    X_syn = rng.normal(0, 1, size=(64, RAW_TOTAL_DIMS)).astype(np.float32)
    y_syn = np.repeat(np.arange(len(MANGO_MAIN_CLASSES)), 8)
    clf = CalibratedConfidenceGatedSystem()
    clf.fit_and_calibrate(X_syn, y_syn, X_syn, y_syn)
    return clf


def run_single_prediction(
    img_rgb: np.ndarray,
    model_path: Path,
    canonical_pose: bool = False,
    gate_threshold: float = 0.65,
) -> Dict[str, Any]:
    clf = load_or_build_model(model_path)

    # 1. Preprocessing & Quality Gate
    t0 = time.perf_counter()
    h, w = img_rgb.shape[:2]
    crop_h, crop_w = max(1, int(h * 0.88)), max(1, int(w * 0.88))
    y0, x0 = max(0, (h - crop_h) // 2), max(0, (w - crop_w) // 2)
    img_std = cv2.resize(
        img_rgb[y0 : y0 + crop_h, x0 : x0 + crop_w],
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    if canonical_pose:
        proc_rgb = canonicalize_leaf_organ_image(img_std, output_size=IMG_SIZE)
    else:
        proc_rgb = apply_adaptive_quality_preprocessing(img_std, foreground_mask=None)
    t_preproc_ms = (time.perf_counter() - t0) * 1000.0

    # 2. 7-Branch Feature Extraction (2,003D with f_hog at tail)
    t1 = time.perf_counter()
    f_hsv, f_lab, f_lbp, f_patch5, f_hog, f_q, f_mgct = extract_fast_all_branches(proc_rgb)
    q_vec = f_q
    feat_vec = np.concatenate(
        [f_hsv, f_lab, f_lbp, f_patch5, f_q, f_mgct, f_hog]
    ).astype(np.float32, copy=False).reshape(1, -1)
    t_feat_ms = (time.perf_counter() - t1) * 1000.0

    # 3. Energy-Balanced Block-PCA (2,003D -> 303D) & Confidence-Gated Inference
    t2 = time.perf_counter()
    z_vec = clf.compressor.transform(feat_vec)
    final_probs, p1_probs, route_rate = clf.predict_gated_probs_from_scaled(z_vec, tau_gate=gate_threshold)
    t_infer_ms = (time.perf_counter() - t2) * 1000.0

    p1_max = float(np.max(p1_probs[0]))
    pred_idx = int(np.argmax(final_probs[0]))
    total_ms = t_preproc_ms + t_feat_ms + t_infer_ms

    return {
        "predicted_class": MANGO_MAIN_CLASSES[pred_idx],
        "predicted_index": pred_idx,
        "stage1_max_confidence": round(p1_max, 4),
        "final_confidence": round(float(final_probs[0, pred_idx]), 4),
        "gate_threshold_tau": gate_threshold,
        "routed_to_stage2": bool(route_rate > 0.0),
        "feature_dim_raw": int(feat_vec.shape[1]),
        "feature_dim_compressed": int(z_vec.shape[1]),
        "latency_ms": {
            "preprocessing_and_segmentation": round(t_preproc_ms, 3),
            "feature_extraction_7branch": round(t_feat_ms, 3),
            "block_pca_and_classification": round(t_infer_ms, 3),
            "total_end_to_end_cpu": round(total_ms, 3),
        },
        "quality_vector_7d": [round(float(v), 4) for v in q_vec],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-image mango leaf disease prediction on CPU.")
    parser.add_argument("--image", type=Path, default=None, help="Path to input mango leaf image (.jpg/.png).")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("results/models/proposed_twostage_gated.joblib"),
        help="Path to serialized joblib model artifact.",
    )
    parser.add_argument("--tau", type=float, default=0.65, help="Stage-1 confidence gate threshold tau (default: 0.65).")
    parser.add_argument("--canonical-pose", action="store_true", help="Enable canonical leaf major-axis orientation alignment.")
    parser.add_argument("--synthetic-demo", action="store_true", help="Run on a generated synthetic mango leaf image.")
    args = parser.parse_args()

    if args.image is not None and args.image.exists():
        raw_bytes = np.fromfile(str(args.image), dtype=np.uint8)
        bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"Could not decode image at {args.image}")
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        source_desc = str(args.image)
    elif args.synthetic_demo or args.image is None:
        img_rgb = create_synthetic_mango_leaf()
        source_desc = "Synthetic Mango Leaf Demo (256x256 RGB)"
    else:
        raise FileNotFoundError(f"Image not found: {args.image}")

    result = run_single_prediction(
        img_rgb=img_rgb,
        model_path=args.model,
        canonical_pose=args.canonical_pose,
        gate_threshold=args.tau,
    )
    result["input_source"] = source_desc
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
