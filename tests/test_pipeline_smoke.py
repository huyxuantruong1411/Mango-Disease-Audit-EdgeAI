"""Unit and smoke tests for the `mango_edge_cv` package using synthetic images
and feature matrices (runs in <5 seconds without requiring raw datasets).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mango_edge_cv.audit import compute_phash_and_hsv_signature
from mango_edge_cv.config import (
    COMPRESSED_TOTAL_DIMS,
    FEATURE_BRANCH_DIMS,
    IMG_SIZE,
    MANGO_MAIN_CLASSES,
    RAW_TOTAL_DIMS,
    SEED,
)
from mango_edge_cv.features import (
    _LBP_RIU2_LUT,
    compose_feature_vector_from_rgb,
    compute_hog_128,
    extract_5patch_spatial_moments_45d,
    extract_fast_all_branches,
    extract_multigrid_chromatic_texture_50d,
    fast_vectorized_lbp_28d,
)
from mango_edge_cv.models import (
    BlockPCACompressor,
    CalibratedConfidenceGatedSystem,
    compute_ece_and_aurc,
    coral_align_features,
)
from mango_edge_cv.preprocessing import (
    apply_adaptive_quality_preprocessing,
    canonicalize_leaf_organ_image,
    segment_plant_organ_roi,
)


def _make_dummy_leaf(h: int = 256, w: int = 256) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    img = rng.integers(30, 220, size=(h, w, 3), dtype=np.uint8)
    img[40:216, 50:206, 0] = 50
    img[40:216, 50:206, 1] = 160
    img[40:216, 50:206, 2] = 45
    return img


def test_config_constants() -> None:
    assert SEED == 20261022
    assert IMG_SIZE == 256
    assert len(MANGO_MAIN_CLASSES) == 8
    assert RAW_TOTAL_DIMS == 2003
    assert COMPRESSED_TOTAL_DIMS == 303
    assert sum(FEATURE_BRANCH_DIMS.values()) == 2003


def test_lbp_lut_shape_and_range() -> None:
    assert _LBP_RIU2_LUT.shape == (256,)
    assert int(_LBP_RIU2_LUT.min()) == 0
    assert int(_LBP_RIU2_LUT.max()) == 9


def test_preprocessing_and_canonical_pose() -> None:
    img = _make_dummy_leaf(256, 256)
    fg_rgb, bg_rgb, mask_u8 = segment_plant_organ_roi(img)
    assert fg_rgb.shape == (256, 256, 3)
    assert bg_rgb.shape == (256, 256, 3)
    assert mask_u8.shape == (256, 256)

    proc_rgb = apply_adaptive_quality_preprocessing(img, foreground_mask=mask_u8)
    assert proc_rgb.shape == (256, 256, 3)

    canon_rgb = canonicalize_leaf_organ_image(img, output_size=256)
    assert canon_rgb.shape == (256, 256, 3)


def test_seven_feature_branches_exact_dimensions() -> None:
    img = _make_dummy_leaf(256, 256)
    proc_rgb = apply_adaptive_quality_preprocessing(img)

    f_hsv, f_lab, f_lbp, f_patch5, f_hog, f_q, f_mgct = extract_fast_all_branches(proc_rgb)
    assert f_hsv.shape == (52,)
    assert f_lab.shape == (57,)
    assert f_lbp.shape == (28,)
    assert f_patch5.shape == (45,)
    assert f_hog.shape == (1764,)
    assert f_q.shape == (7,)
    assert f_mgct.shape == (50,)

    full = compose_feature_vector_from_rgb(proc_rgb)
    assert full.shape == (RAW_TOTAL_DIMS,)
    assert np.all(np.isfinite(full))


def test_block_pca_and_confidence_gated_model() -> None:
    rng = np.random.default_rng(SEED)
    n_samples = 80
    X = rng.normal(0.0, 1.0, size=(n_samples, RAW_TOTAL_DIMS)).astype(np.float32)
    y = np.repeat(np.arange(8), n_samples // 8)

    comp = BlockPCACompressor(hog_dim=1764, n_hog_pca=64)
    Z = comp.fit_transform(X)
    assert Z.shape == (n_samples, COMPRESSED_TOTAL_DIMS)

    clf = CalibratedConfidenceGatedSystem()
    clf.fit_and_calibrate(X, y, X, y)
    final_probs, p1_probs, route_rate = clf.predict_gated_probs(X, tau_gate=0.65)
    assert final_probs.shape == (n_samples, 8)
    assert p1_probs.shape == (n_samples, 8)
    assert 0.0 <= route_rate <= 1.0
    assert np.allclose(final_probs.sum(axis=1), 1.0, atol=1e-4)

    ece, aurc = compute_ece_and_aurc(y, final_probs)
    assert 0.0 <= ece <= 1.0
    assert 0.0 <= aurc <= 1.0


def test_coral_domain_adaptation_and_phash(tmp_path: Path) -> None:
    rng = np.random.default_rng(SEED)
    X_s = rng.normal(0.0, 1.0, size=(40, 303)).astype(np.float32)
    X_t = rng.normal(1.5, 1.2, size=(30, 303)).astype(np.float32)
    X_aligned = coral_align_features(X_s, X_t)
    assert X_aligned.shape == X_t.shape
    assert np.all(np.isfinite(X_aligned))

    bgr_leaf = _make_dummy_leaf(120, 120)
    phash_bits, hsv_vec = compute_phash_and_hsv_signature(bgr_leaf)
    assert phash_bits.shape == (64,)
    assert hsv_vec.shape == (96,)
