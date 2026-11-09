"""
Seven-Branch Handcrafted Feature Extraction Engine with Shared Sobel Gradients
and Lookup-Table Rotation-Invariant Uniform LBP (`src/mango_edge_cv/features.py`).
Outputs 2,003 raw dimensions:
  - f_hsv (52D), f_lab (57D), f_lbp (28D), f_patch5 (45D),
    f_hog (1,764D), f_q (7D), f_mgct (50D).
"""

import os
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

from .config import FEATURE_CACHE_VERSION, HOG_BACKEND
from .preprocessing import (
    apply_adaptive_quality_preprocessing,
    apply_clahe_only,
    apply_glare_inpainting,
    apply_tophat_blackhat_enhancement,
    canonicalize_leaf_organ_image,
    read_and_standardize_rgb,
    read_rgb_preserve_aspect,
    segment_plant_organ_roi,
)

# Precomputed 256-entry lookup table for rotation-invariant uniform LBP (riu2)
_LBP_RIU2_LUT = np.zeros(256, dtype=np.uint8)
for _code in range(256):
    _bits = [(_code >> _i) & 1 for _i in range(8)]
    _transitions = sum(_bits[_i] != _bits[(_i + 1) % 8] for _i in range(8))
    _LBP_RIU2_LUT[_code] = sum(_bits) if _transitions <= 2 else 9


def compute_hog_128(
    gray_128: np.ndarray, gx: np.ndarray = None, gy: np.ndarray = None, grad_mag_128: np.ndarray = None
) -> np.ndarray:
    """
    Pure NumPy 4D-tensor vectorized Sobel HOG on a 128x128 grayscale image:
    8x8 cells (16x16 px), 9 orientation bins in [0, 180) with bilinear interpolation,
    7x7 overlapping 2x2-cell blocks with L2-Hys clipping at 0.2 -> 49 x 36 = 1,764D.
    """
    if gray_128.shape != (128, 128):
        gray_128 = cv2.resize(gray_128, (128, 128), interpolation=cv2.INTER_AREA)
    gray_f = np.ascontiguousarray(gray_128, dtype=np.float32)
    if gx is None or gy is None:
        gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
    else:
        gx = np.asarray(gx, dtype=np.float32)
        gy = np.asarray(gy, dtype=np.float32)
    if grad_mag_128 is None:
        grad_mag_128 = cv2.magnitude(gx, gy)
    ang = cv2.phase(gx, gy, angleInDegrees=True)
    ang = np.mod(ang, 180.0)

    bin_float = ang / 20.0
    bin_floor = np.floor(bin_float)
    b0 = bin_floor.astype(np.int32) % 9
    b1 = (b0 + 1) % 9
    w1 = (bin_float - bin_floor).astype(np.float32)
    w0 = 1.0 - w1

    def to_cells(a):
        return a.reshape(8, 16, 8, 16).transpose(0, 2, 1, 3).reshape(64, 256)

    mag_cells = to_cells(grad_mag_128)
    b0_cells = to_cells(b0)
    b1_cells = to_cells(b1)
    w0_cells = mag_cells * to_cells(w0)
    w1_cells = mag_cells * to_cells(w1)
    cell_idx = np.arange(64, dtype=np.int32)[:, None] * 9
    hist0 = np.bincount(
        (cell_idx + b0_cells).ravel(), weights=w0_cells.ravel(), minlength=64 * 9
    )
    hist1 = np.bincount(
        (cell_idx + b1_cells).ravel(), weights=w1_cells.ravel(), minlength=64 * 9
    )
    cell_hists = (hist0 + hist1).reshape(8, 8, 9).astype(np.float32)

    try:
        windows = np.lib.stride_tricks.sliding_window_view(cell_hists, (2, 2, 9))
        blocks = windows.reshape(49, 36).copy()
    except AttributeError:
        blocks = np.stack(
            [
                cell_hists[y : y + 2, x : x + 2, :].reshape(-1)
                for y in range(7)
                for x in range(7)
            ],
            axis=0,
        )

    blocks /= np.linalg.norm(blocks, axis=1, keepdims=True) + 1e-5
    blocks = np.minimum(blocks, 0.2)
    blocks /= np.linalg.norm(blocks, axis=1, keepdims=True) + 1e-5
    result = blocks.reshape(-1).astype(np.float32)
    if result.size != 1764 or not np.isfinite(result).all():
        raise ValueError(
            f"Invalid HOG vector: shape={result.shape}; finite={np.isfinite(result).all()}"
        )
    return result


def _lbp_code(gray: np.ndarray) -> np.ndarray:
    """Build 8-neighbour binary codes in vectorized NumPy operations."""
    g = np.asarray(gray, dtype=np.uint8)
    c = g[1:-1, 1:-1]
    neighbors = [
        g[0:-2, 0:-2],
        g[0:-2, 1:-1],
        g[0:-2, 2:],
        g[1:-1, 2:],
        g[2:, 2:],
        g[2:, 1:-1],
        g[2:, 0:-2],
        g[1:-1, 0:-2],
    ]
    code = np.zeros(c.shape, dtype=np.uint8)
    for bit, neighbor in enumerate(neighbors):
        code |= (neighbor >= c).astype(np.uint8) << bit
    return code


def fast_vectorized_lbp_28d(gray_128: np.ndarray, valid_mask_128: np.ndarray = None) -> np.ndarray:
    """
    28D multi-scale rotation-invariant uniform LBP using _LBP_RIU2_LUT:
      - LBP_{8,1}^{riu2} at 128x128 (10D)
      - LBP_{8,2}^{riu2} at 64x64 (10D)
      - 8-bin local micro-contrast residual histogram (8D).
    """
    g = np.asarray(gray_128, dtype=np.uint8)
    code1 = _lbp_code(g)
    lbp1 = _LBP_RIU2_LUT[code1]

    mask1 = None
    if valid_mask_128 is not None:
        m = np.asarray(valid_mask_128, dtype=np.uint8)
        if m.shape != g.shape:
            m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        eroded = cv2.erode(
            (m > 0).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1
        )
        candidate = eroded[1:-1, 1:-1] > 0
        if int(candidate.sum()) > 100:
            mask1 = candidate
    vals1 = lbp1[mask1] if mask1 is not None else lbp1.ravel()
    hist1 = np.bincount(vals1.ravel(), minlength=10).astype(np.float32)
    hist1 /= hist1.sum() + 1e-8

    g2 = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA)
    code2 = _lbp_code(g2)
    lbp2 = _LBP_RIU2_LUT[code2]
    mask2 = None
    if valid_mask_128 is not None:
        m2 = cv2.resize(
            (np.asarray(valid_mask_128) > 0).astype(np.uint8),
            (64, 64),
            interpolation=cv2.INTER_NEAREST,
        )
        m2 = cv2.erode(m2, np.ones((3, 3), np.uint8), iterations=1)
        candidate2 = m2[1:-1, 1:-1] > 0
        if int(candidate2.sum()) > 30:
            mask2 = candidate2
    vals2 = lbp2[mask2] if mask2 is not None else lbp2.ravel()
    hist2 = np.bincount(vals2.ravel(), minlength=10).astype(np.float32)
    hist2 /= hist2.sum() + 1e-8

    blur = cv2.GaussianBlur(g, (3, 3), 0)
    diff = cv2.absdiff(g[1:-1, 1:-1], blur[1:-1, 1:-1])
    diff_values = diff[mask1] if mask1 is not None else diff.ravel()
    bins = np.minimum(diff_values.astype(np.int32) // 8, 7)
    hist3 = np.bincount(bins.ravel(), minlength=8).astype(np.float32)
    hist3 /= hist3.sum() + 1e-8

    result = np.concatenate([hist1, hist2, hist3]).astype(np.float32)
    if result.size != 28 or not np.isfinite(result).all():
        raise ValueError(f"LBP dimension/value check failed: {result.size}")
    return result


def extract_5patch_spatial_moments_45d(
    hsv_128: np.ndarray,
    lab_128: np.ndarray,
    gray_128: np.ndarray,
    grad_mag_128: np.ndarray = None,
    valid_mask_128: np.ndarray = None,
) -> np.ndarray:
    """45D 5-Patch spatial moments: 5 zones (TL, TR, BL, BR, Center 64x64) x 9 chromatic/edge moments."""
    if grad_mag_128 is None:
        gx = cv2.Sobel(gray_128.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_128.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad_mag_128 = cv2.magnitude(gx, gy)
    patches = [
        (0, 64, 0, 64),
        (0, 64, 64, 128),
        (64, 128, 0, 64),
        (64, 128, 64, 128),
        (32, 96, 32, 96),
    ]
    feats = []
    for y1, y2, x1, x2 in patches:
        p_hsv = hsv_128[y1:y2, x1:x2]
        p_lab = lab_128[y1:y2, x1:x2]
        p_gray = gray_128[y1:y2, x1:x2].astype(np.float32)
        p_edge = grad_mag_128[y1:y2, x1:x2].astype(np.float32)
        valid = None if valid_mask_128 is None else (valid_mask_128[y1:y2, x1:x2] > 0)
        if valid is not None and int(valid.sum()) < 25:
            valid = None

        def vals(channel):
            arr = channel[valid] if valid is not None else channel.ravel()
            return arr.astype(np.float32)

        h_vals = vals(p_hsv[:, :, 0])
        s_vals = vals(p_hsv[:, :, 1])
        lab_l = vals(p_lab[:, :, 0])
        lab_a = vals(p_lab[:, :, 1])
        lab_b = vals(p_lab[:, :, 2])
        g_vals = vals(p_gray)
        edge_vals = vals(p_edge)
        feats.extend(
            [
                float(np.mean(h_vals) / 180.0),
                float(np.mean(s_vals) / 255.0),
                float(np.std(s_vals) / 255.0),
                float(np.mean(lab_l) / 255.0),
                float(np.mean(lab_a) / 255.0),
                float(np.std(lab_a) / 255.0),
                float(np.mean(lab_b) / 255.0),
                float(np.std(g_vals) / 255.0),
                float(np.mean(edge_vals) / 1442.0),
            ]
        )
    result = np.asarray(feats, dtype=np.float32)
    if result.size != 45 or not np.isfinite(result).all():
        raise ValueError(f"5-Patch dimension/value check failed: {result.size}")
    return result


def extract_multigrid_chromatic_texture_50d(
    hsv_128: np.ndarray,
    lab_128: np.ndarray,
    grad_mag_128: np.ndarray,
    valid_mask_128: np.ndarray = None,
) -> np.ndarray:
    """50D Multi-Grid Chromatic-Texture (MGCT): 5 spatial regions x 10 chromatic-gradient statistics."""
    a = (lab_128[:, :, 1].astype(np.float32) - 128.0) / 127.0
    b = (lab_128[:, :, 2].astype(np.float32) - 128.0) / 127.0
    saturation = hsv_128[:, :, 1].astype(np.float32) / 255.0
    grad_norm = np.clip(grad_mag_128.astype(np.float32) / 1442.0, 0.0, 1.0)
    chroma_mag = np.sqrt(a * a + b * b)
    regions = [
        (0, 128, 0, 128),
        (0, 64, 0, 64),
        (0, 64, 64, 128),
        (64, 128, 0, 64),
        (64, 128, 64, 128),
    ]
    values = []
    for y1, y2, x1, x2 in regions:
        region_mask = (
            None if valid_mask_128 is None else (valid_mask_128[y1:y2, x1:x2] > 0)
        )
        if region_mask is not None and int(region_mask.sum()) < 25:
            region_mask = None

        def rv(arr):
            part = arr[y1:y2, x1:x2]
            return (
                part[region_mask] if region_mask is not None else part.ravel()
            ).astype(np.float32)

        ss, aa, bb, cc, gg = rv(saturation), rv(a), rv(b), rv(chroma_mag), rv(grad_norm)
        cov = float(np.mean((cc - cc.mean()) * (gg - gg.mean()))) if len(cc) else 0.0
        corr = cov / (float(np.std(cc) * np.std(gg)) + 1e-6) if len(cc) else 0.0
        values.extend(
            [
                float(np.mean(ss)),
                float(np.std(ss)),
                float(np.mean(aa)),
                float(np.std(aa)),
                float(np.mean(bb)),
                float(np.std(bb)),
                float(np.mean(gg)),
                float(np.std(gg)),
                float(np.clip(corr, -1.0, 1.0)),
                float(np.mean(gg > (60.0 / 1442.0))),
            ]
        )
    result = np.asarray(values, dtype=np.float32)
    if result.size != 50:
        raise ValueError(f"MGCT dimension mismatch: {result.size}")
    return result


def extract_fast_all_branches(img_rgb_256: np.ndarray):
    """
    Extract all 7 feature branches from a single 128x128 RGB/HSV/LAB/Grayscale conversion
    and a single shared 3x3 Sobel gradient computation:
    Returns (f_hsv[52], f_lab[57], f_lbp[28], f_patch5[45], f_hog[1764], f_q[7], f_mgct[50]).
    """
    img_128 = cv2.resize(img_rgb_256, (128, 128), interpolation=cv2.INTER_AREA)
    hsv_128 = cv2.cvtColor(img_128, cv2.COLOR_RGB2HSV)
    lab_128 = cv2.cvtColor(img_128, cv2.COLOR_RGB2LAB)
    gray_128 = cv2.cvtColor(img_128, cv2.COLOR_RGB2GRAY)

    nonzero = (np.max(img_128, axis=2) > 3).astype(np.uint8) * 255
    zero_ratio = 1.0 - float(np.count_nonzero(nonzero)) / float(nonzero.size)
    hist_mask = nonzero if 0.05 < zero_ratio < 0.92 else None
    valid_px = (
        (hist_mask > 0)
        if hist_mask is not None
        else np.ones(gray_128.shape, dtype=bool)
    )

    hsv_hist_parts = [
        cv2.normalize(
            cv2.calcHist([hsv_128], [ch], hist_mask, [16], limits), None
        )
        .flatten()
        .astype(np.float32)
        for ch, limits in [(0, [0, 180]), (1, [0, 256]), (2, [0, 256])]
    ]
    avg_h = float(np.mean(hsv_128[:, :, 0][valid_px]) / 180.0)
    avg_s = float(np.mean(hsv_128[:, :, 1][valid_px]) / 255.0)
    avg_v = float(np.mean(hsv_128[:, :, 2][valid_px]) / 255.0)
    f_hsv = np.concatenate(
        hsv_hist_parts
        + [
            np.asarray(
                [avg_h, avg_s, avg_v, (avg_h + avg_s + avg_v) / 3.0], dtype=np.float32
            )
        ]
    )

    lab_hist_parts = [
        cv2.normalize(
            cv2.calcHist([lab_128], [ch], hist_mask, [16], [0, 256]), None
        )
        .flatten()
        .astype(np.float32)
        for ch in range(3)
    ]
    lab_moments = []
    for ch_id in range(3):
        values = lab_128[:, :, ch_id][valid_px].astype(np.float32)
        mean_ch = float(np.mean(values))
        std_ch = float(np.std(values)) + 1e-6
        skew_ch = float(np.mean(((values - mean_ch) / std_ch) ** 3))
        lab_moments.extend([mean_ch / 255.0, std_ch / 255.0, skew_ch])
    f_lab = np.concatenate(lab_hist_parts + [np.asarray(lab_moments, dtype=np.float32)])
    f_lbp = fast_vectorized_lbp_28d(gray_128, valid_mask_128=hist_mask)

    gray_for_edges = gray_128.copy()
    if hist_mask is not None and int(valid_px.sum()) > 100:
        gray_for_edges[~valid_px] = np.uint8(np.median(gray_128[valid_px]))
    gx = cv2.Sobel(gray_for_edges.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_for_edges.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag_128 = cv2.magnitude(gx, gy)

    f_patch5 = extract_5patch_spatial_moments_45d(
        hsv_128,
        lab_128,
        gray_for_edges,
        grad_mag_128=grad_mag_128,
        valid_mask_128=hist_mask,
    )
    f_hog = compute_hog_128(gray_for_edges, gx=gx, gy=gy, grad_mag_128=grad_mag_128)
    f_mgct = extract_multigrid_chromatic_texture_50d(
        hsv_128, lab_128, grad_mag_128, valid_mask_128=hist_mask
    )

    s_ch, v_ch = hsv_128[:, :, 1], hsv_128[:, :, 2]
    central_s, central_v = s_ch[20:108, 20:108], v_ch[20:108, 20:108]
    central_mask = valid_px[20:108, 20:108]
    if int(central_mask.sum()) < 25:
        central_mask = np.ones(central_s.shape, dtype=bool)
    valid_v = v_ch[valid_px]
    mu_v = float(np.mean(valid_v) / 255.0)
    sigma_v = float(np.std(valid_v) / 255.0)
    r_shadow = float(np.mean(valid_v < 40))
    r_glare = float(np.mean(((central_v > 240) & (central_s < 35))[central_mask]))
    var_lap = float(cv2.Laplacian(gray_for_edges, cv2.CV_32F).var() / 1000.0)
    p_lbp = f_lbp[:10] + 1e-10
    h_lbp = float(-np.sum(p_lbp * np.log(p_lbp)))
    mu_hog = float(np.mean(f_hog))
    f_q = np.asarray(
        [mu_v, sigma_v, r_shadow, r_glare, var_lap, h_lbp, mu_hog], dtype=np.float32
    )

    expected = [52, 57, 28, 45, 1764, 7, 50]
    actual = [
        f_hsv.size,
        f_lab.size,
        f_lbp.size,
        f_patch5.size,
        f_hog.size,
        f_q.size,
        f_mgct.size,
    ]
    if actual != expected:
        raise ValueError(
            f"Feature dimension mismatch: expected={expected}; actual={actual}"
        )
    vectors = [f_hsv, f_lab, f_lbp, f_patch5, f_hog, f_q, f_mgct]
    if not all(np.isfinite(v).all() for v in vectors):
        raise ValueError("Non-finite handcrafted feature encountered")
    return f_hsv, f_lab, f_lbp, f_patch5, f_hog, f_q, f_mgct


def compose_feature_vector_from_rgb(
    img_rgb_256: np.ndarray, view_mode: str = "adaptive"
) -> np.ndarray:
    """Preprocess an RGB image according to view_mode and return the 2,003D feature vector (with HOG at the tail)."""
    if view_mode == "segmented":
        fg_seg, _, mask = segment_plant_organ_roi(img_rgb_256)
        prepared = apply_adaptive_quality_preprocessing(fg_seg, foreground_mask=mask)
    elif view_mode == "canonical":
        prepared = canonicalize_leaf_organ_image(img_rgb_256)
    elif view_mode == "raw":
        prepared = img_rgb_256
    else:
        prepared = apply_adaptive_quality_preprocessing(
            img_rgb_256, foreground_mask=None
        )
    f_hsv, f_lab, f_lbp, f_patch5, f_hog, f_q, f_mgct = extract_fast_all_branches(
        prepared
    )
    return np.concatenate(
        [f_hsv, f_lab, f_lbp, f_patch5, f_q, f_mgct, f_hog]
    ).astype(np.float32)


def compose_feature_vector(path: str, canonical: bool = False, view_mode: str = "adaptive"):
    """Read an image file and compose its 2,003D feature vector."""
    if canonical:
        image = read_rgb_preserve_aspect(path)
        if image is None:
            return None
        image = canonicalize_leaf_organ_image(image)
    else:
        image = read_and_standardize_rgb(path)
        if image is None:
            return None
        if view_mode == "segmented":
            image, _, mask = segment_plant_organ_roi(image)
            image = apply_adaptive_quality_preprocessing(image, foreground_mask=mask)
        else:
            image = apply_adaptive_quality_preprocessing(image, foreground_mask=None)
    f_hsv, f_lab, f_lbp, f_patch5, f_hog, f_q, f_mgct = extract_fast_all_branches(image)
    return np.concatenate(
        [f_hsv, f_lab, f_lbp, f_patch5, f_q, f_mgct, f_hog]
    ).astype(np.float32)


def assemble_matrix_by_mask(
    feat_dict: dict, mask: np.ndarray, mode: str = "raw", branches=("hsv", "lbp", "hog")
):
    """Assemble horizontal feature matrix for a given split mask, preprocessing mode, and branch list."""
    mats = [feat_dict[f"{mode}_{branch}"][mask] for branch in branches]
    return np.hstack(mats).astype(np.float32), feat_dict["labels"][mask]


def extract_and_cache_all_variants(df_sub, cache_path: str, workers: int = 4):
    """Extract or load cached 6-mode x 7-branch feature matrices for the full audited dataset."""
    if cache_path and os.path.exists(cache_path):
        loaded = dict(np.load(cache_path, allow_pickle=True))
        cached_version_arr = loaded.get("cache_version", np.array([""]))
        cached_version = (
            str(cached_version_arr.reshape(-1)[0]) if cached_version_arr.size else ""
        )
        if cached_version == FEATURE_CACHE_VERSION and len(
            loaded.get("labels", [])
        ) == len(df_sub):
            return loaded

    modes = ["raw", "glare", "clahe", "morph", "adaptive", "segmented"]
    branches = ["hsv", "lab", "lbp", "patch5", "hog", "q", "mgct"]
    storage = {f"{m}_{b}": [] for m in modes for b in branches}
    valid_indices = []

    def _extract_one_row(item):
        idx, file_path = item
        img_raw = read_and_standardize_rgb(file_path)
        if img_raw is None:
            return idx, None
        fg_seg, _, fg_mask = segment_plant_organ_roi(img_raw)
        variants = {
            "raw": img_raw,
            "glare": apply_glare_inpainting(img_raw, foreground_mask=fg_mask),
            "clahe": apply_clahe_only(img_raw),
            "morph": apply_tophat_blackhat_enhancement(img_raw),
            "adaptive": apply_adaptive_quality_preprocessing(
                img_raw, foreground_mask=None
            ),
            "segmented": apply_adaptive_quality_preprocessing(
                fg_seg, foreground_mask=fg_mask
            ),
        }
        row_features = {}
        for mode_name, image_variant in variants.items():
            fhsv, flab, flbp, fp5, fhog, fq, fmgct = extract_fast_all_branches(
                image_variant
            )
            row_features[f"{mode_name}_hsv"] = fhsv
            row_features[f"{mode_name}_lab"] = flab
            row_features[f"{mode_name}_lbp"] = flbp
            row_features[f"{mode_name}_patch5"] = fp5
            row_features[f"{mode_name}_hog"] = fhog
            row_features[f"{mode_name}_q"] = fq
            row_features[f"{mode_name}_mgct"] = fmgct
        return idx, row_features

    indexed_paths = [
        (int(idx), str(row["file_path"])) for idx, row in df_sub.iterrows()
    ]
    worker_count = min(int(workers), max(1, len(indexed_paths)))
    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="agri-features"
    ) as pool:
        futures = pool.map(_extract_one_row, indexed_paths)
        for idx, row_features in tqdm(
            futures, total=len(indexed_paths), desc="Extracting All Feature Variants"
        ):
            if row_features is None:
                continue
            valid_indices.append(idx)
            for key, vector in row_features.items():
                storage[key].append(vector)

    np_storage = {k: np.array(v, dtype=np.float32) for k, v in storage.items()}
    np_storage["labels"] = df_sub.loc[valid_indices, "label_id"].values.astype(np.int32)
    np_storage["splits"] = df_sub.loc[valid_indices, "split_group"].values.astype(str)
    np_storage["splits_random"] = df_sub.loc[
        valid_indices, "split_random"
    ].values.astype(str)
    np_storage["fold_ids"] = df_sub.loc[valid_indices, "fold_id"].values.astype(
        np.int32
    )
    np_storage["group_ids"] = df_sub.loc[valid_indices, "group_id"].values.astype(
        np.int32
    )
    np_storage["file_paths"] = df_sub.loc[valid_indices, "file_path"].values.astype(str)
    np_storage["cache_version"] = np.array([FEATURE_CACHE_VERSION])

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        np.savez_compressed(cache_path, **np_storage)
    return np_storage
