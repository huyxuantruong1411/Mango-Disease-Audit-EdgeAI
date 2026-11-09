"""
Foreground-Aware Quality-Gated Optical Preprocessing, ROI Segmentation, and
Canonical Leaf Pose Normalization (`src/mango_edge_cv/preprocessing.py`).
"""

import os
import cv2
import numpy as np

from .config import CANONICAL_COLOR_MODE, IMG_SIZE


def read_and_standardize_rgb(img_path: str, img_size: int = IMG_SIZE, crop_ratio: float = 0.88):
    """Read an image file, center-crop by crop_ratio, and resize to (img_size, img_size) RGB."""
    path_str = str(img_path)
    ext = os.path.splitext(path_str)[1].casefold()
    try:
        file_size = os.path.getsize(path_str)
    except OSError:
        file_size = 0
    img_bgr = None
    reduced_flag = getattr(cv2, "IMREAD_REDUCED_COLOR_4", None)
    if (
        ext in {".jpg", ".jpeg", ".jpe", ".jfif"}
        and file_size > 350 * 1024
        and reduced_flag is not None
    ):
        img_bgr = cv2.imread(path_str, reduced_flag)
        if img_bgr is None or min(img_bgr.shape[:2]) < img_size:
            img_bgr = None
    if img_bgr is None:
        img_bgr = cv2.imread(path_str, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    crop_ratio = float(np.clip(crop_ratio, 0.1, 1.0))
    crop_h, crop_w = max(1, int(h * crop_ratio)), max(1, int(w * crop_ratio))
    y0, x0 = max(0, (h - crop_h) // 2), max(0, (w - crop_w) // 2)
    crop = img_rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_AREA)


def segment_plant_organ_roi(img_rgb: np.ndarray):
    """
    Heuristic largest-foreground leaf mask using combined HSV saturation and CIE-LAB chroma
    with Otsu thresholding and studio background suppression.
    Returns (foreground_rgb, background_rgb, selected_mask_u8).
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    s_ch = hsv[:, :, 1].astype(np.float32)
    v_ch = hsv[:, :, 2]
    a_ch = lab[:, :, 1].astype(np.float32) - 128.0
    b_ch = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a_ch * a_ch + b_ch * b_ch)
    score = np.clip(0.60 * s_ch + 1.25 * chroma, 0, 255).astype(np.uint8)
    _, mask = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    studio_bg = ((v_ch > 230) & (s_ch < 28)) | (v_ch < 15)
    mask[studio_bg] = 0
    h, w = mask.shape
    my, mx = max(1, int(0.03 * h)), max(1, int(0.03 * w))
    mask[:my, :] = 0
    mask[-my:, :] = 0
    mask[:, :mx] = 0
    mask[:, -mx:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    selected_mask = None
    if n_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = stats[largest, cv2.CC_STAT_AREA]
        if 0.04 * h * w <= area <= 0.92 * h * w:
            selected_mask = np.where(labels == largest, 255, 0).astype(np.uint8)
    if selected_mask is None:
        selected_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(
            selected_mask,
            (w // 2, h // 2),
            (max(1, int(w * 0.40)), max(1, int(h * 0.42))),
            0,
            0,
            360,
            255,
            -1,
        )
    foreground = cv2.bitwise_and(img_rgb, img_rgb, mask=selected_mask)
    background = cv2.bitwise_and(img_rgb, img_rgb, mask=cv2.bitwise_not(selected_mask))
    return foreground, background, selected_mask


def _fast_foreground_mask_128(hsv_128: np.ndarray):
    """Fast heuristic mask on 128x128 HSV image excluding studio white/gray background and dark borders."""
    s_ch = hsv_128[:, :, 1]
    v_ch = hsv_128[:, :, 2]
    fg = ~(((v_ch > 222) & (s_ch < 30)) | (v_ch < 12))
    h, w = fg.shape
    my, mx = max(1, int(0.025 * h)), max(1, int(0.025 * w))
    fg[:my, :] = False
    fg[-my:, :] = False
    fg[:, :mx] = False
    fg[:, -mx:] = False
    frac = float(np.mean(fg))
    if not (0.04 <= frac <= 0.94) or int(fg.sum()) < 150:
        return None
    return fg


def _usable_foreground_mask(img_rgb: np.ndarray, foreground_mask=None):
    """Normalize a supplied mask or estimate a fast HSV foreground mask without connectedComponents."""
    h, w = img_rgb.shape[:2]
    if foreground_mask is not None:
        mask = np.asarray(foreground_mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        if mask.shape != (h, w):
            mask = cv2.resize(
                mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
            )
        mask = (mask > 0).astype(np.uint8) * 255
        frac = float(np.mean(mask > 0))
        return mask if 0.02 <= frac <= 0.98 else None

    small = (
        cv2.resize(img_rgb, (128, 128), interpolation=cv2.INTER_AREA)
        if max(h, w) > 160
        else img_rgb
    )
    hsv_small = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    mask_small = (
        _fast_foreground_mask_128(hsv_small) if small.shape[:2] == (128, 128) else None
    )
    if mask_small is None:
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        s_ch, v_ch = hsv[:, :, 1], hsv[:, :, 2]
        candidate = ~(((v_ch > 222) & (s_ch < 30)) | (v_ch < 12))
        frac = float(candidate.mean())
        if not (0.04 <= frac <= 0.94):
            return None
        return candidate.astype(np.uint8) * 255
    if mask_small.shape != (h, w):
        mask = cv2.resize(
            mask_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )
    else:
        mask = mask_small.astype(np.uint8)
    frac = float(np.mean(mask > 0))
    return mask * 255 if 0.04 <= frac <= 0.94 else None


def _estimate_noise_sigma_mad(gray: np.ndarray, valid_mask=None) -> float:
    """
    Donoho-Johnstone Median Absolute Deviation (MAD) sensor grain estimator on
    3x3 median-filter high-frequency residuals within the foreground mask.
    """
    gray_u8 = np.asarray(gray, dtype=np.uint8)
    smooth = cv2.medianBlur(gray_u8, 3)
    residual = gray_u8.astype(np.float32) - smooth.astype(np.float32)
    if valid_mask is not None:
        valid = np.asarray(valid_mask) > 0
        if valid.shape != residual.shape:
            valid = (
                cv2.resize(
                    valid.astype(np.uint8),
                    (residual.shape[1], residual.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                > 0
            )
    else:
        valid = np.ones(residual.shape, dtype=bool)
    vals = residual[valid]
    if vals.size < 200:
        vals = residual.ravel()
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return mad / 0.6745


def apply_glare_inpainting(
    img_rgb: np.ndarray, tau_v: int = 238, tau_s: int = 38, foreground_mask=None
) -> np.ndarray:
    """Attenuate central specular glare highlights within the foreground mask via local Gaussian replacement."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    h, w = hsv.shape[:2]
    glare = (hsv[:, :, 2] > tau_v) & (hsv[:, :, 1] < tau_s)
    valid_mask = _usable_foreground_mask(img_rgb, foreground_mask)
    if valid_mask is not None:
        valid = valid_mask > 0
        glare &= valid
        denom = max(1, int(valid.sum()))
    else:
        valid = np.ones((h, w), dtype=bool)
        denom = glare.size

    border_y, border_x = int(0.14 * h), int(0.14 * w)
    glare[:border_y, :] = False
    glare[-border_y:, :] = False
    glare[:, :border_x] = False
    glare[:, -border_x:] = False
    ratio = float(np.count_nonzero(glare)) / denom
    if not (0.001 <= ratio <= 0.22):
        return img_rgb

    glare_dilated = (
        cv2.dilate(glare.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1) > 0
    )
    if valid_mask is not None:
        glare_dilated &= valid
    blurred = cv2.GaussianBlur(img_rgb, (11, 11), 2.5)
    return np.where(glare_dilated[:, :, None], blurred, img_rgb).astype(np.uint8)


def apply_clahe_only(img_rgb: np.ndarray, clip_limit: float = 1.8) -> np.ndarray:
    """Apply unconditional CLAHE on the CIE-LAB L* channel."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_eq = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)).apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_eq, a_ch, b_ch]), cv2.COLOR_LAB2RGB)


def apply_tophat_blackhat_enhancement(img_rgb: np.ndarray, weight: float = 0.40) -> np.ndarray:
    """Apply morphological Top-Hat and Black-Hat local contrast enhancement on CIE-LAB L*."""
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    top_hat = cv2.morphologyEx(l_ch, cv2.MORPH_TOPHAT, kernel)
    black_hat = cv2.morphologyEx(l_ch, cv2.MORPH_BLACKHAT, kernel)
    light = cv2.add(
        l_ch, np.clip(top_hat.astype(np.float32) * weight, 0, 255).astype(np.uint8)
    )
    light = cv2.subtract(
        light, np.clip(black_hat.astype(np.float32) * weight, 0, 255).astype(np.uint8)
    )
    return cv2.cvtColor(cv2.merge([light, a_ch, b_ch]), cv2.COLOR_LAB2RGB)


def apply_adaptive_quality_preprocessing(
    img_rgb: np.ndarray,
    foreground_mask=None,
    tau_v: int = 238,
    tau_s: int = 38,
    glare_ratio_thresh: float = 0.003,
    fg_contrast_std_thresh: float = 38.0,
    fg_dynamic_range_thresh: float = 132.0,
    noise_sigma_low: float = 5.5,
    noise_sigma_high: float = 10.0,
) -> np.ndarray:
    """
    Foreground-aware conditional optical quality gate:
      1) Robust MAD sensor grain estimation & conditional denoising;
      2) Central specular glare attenuation inside the foreground mask;
      3) Foreground L* dynamic range percentile stretch + mild CLAHE.
    """
    out = np.asarray(img_rgb, dtype=np.uint8).copy()
    valid_mask = _usable_foreground_mask(out, foreground_mask)
    valid = (
        (valid_mask > 0)
        if valid_mask is not None
        else np.ones(out.shape[:2], dtype=bool)
    )

    gray = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    noise_sigma = _estimate_noise_sigma_mad(gray, valid_mask)
    if noise_sigma >= noise_sigma_high:
        out = cv2.GaussianBlur(cv2.medianBlur(out, 3), (3, 3), 0.65)
    elif noise_sigma >= noise_sigma_low:
        out = cv2.GaussianBlur(out, (3, 3), 0.55)

    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
    h, w = hsv.shape[:2]
    y0, y1 = int(0.14 * h), int(0.86 * h)
    x0, x1 = int(0.14 * w), int(0.86 * w)
    central_s = hsv[y0:y1, x0:x1, 1]
    central_v = hsv[y0:y1, x0:x1, 2]
    central_valid = valid[y0:y1, x0:x1]
    candidate_glare = (central_v > tau_v) & (central_s < tau_s)
    if int(central_valid.sum()) >= 100:
        glare_ratio = float(
            np.count_nonzero(candidate_glare & central_valid)
            / max(1, central_valid.sum())
        )
    else:
        glare_ratio = float(np.mean(candidate_glare))
    if glare_ratio_thresh <= glare_ratio <= 0.22:
        out = apply_glare_inpainting(
            out, tau_v=tau_v, tau_s=tau_s, foreground_mask=valid_mask
        )

    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_values = l_ch[valid].astype(np.float32)
    if l_values.size < 200:
        valid = np.ones(l_ch.shape, dtype=bool)
        l_values = l_ch.ravel().astype(np.float32)
    fg_std = float(np.std(l_values))
    p5, p95 = np.percentile(l_values, [5, 95])
    fg_span = float(p95 - p5)

    if fg_std < fg_contrast_std_thresh or fg_span < fg_dynamic_range_thresh:
        p2, p98 = np.percentile(l_ch[valid].astype(np.float32), [2, 98])
        l_restored = l_ch.copy()
        if (p98 - p2) >= 8.0 and fg_span < fg_dynamic_range_thresh:
            stretch = np.clip(
                (l_ch.astype(np.float32) - p2) * (230.0 / (p98 - p2)) + 12.0, 0, 255
            )
            l_restored[valid] = stretch[valid].astype(np.uint8)
        l_eq = cv2.createCLAHE(clipLimit=1.55, tileGridSize=(8, 8)).apply(l_restored)
        l_final = l_restored.copy()
        l_final[valid] = l_eq[valid]
        out = cv2.cvtColor(cv2.merge([l_final, a_ch, b_ch]), cv2.COLOR_LAB2RGB)

    return out


def read_rgb_preserve_aspect(path: str, canvas_size: int = 320):
    """Read RGB image, preserve aspect ratio, and letterbox onto a white square canvas."""
    p = str(path)
    try:
        size_bytes = os.path.getsize(p)
    except OSError:
        size_bytes = 0
    ext = os.path.splitext(p)[1].casefold()
    bgr = None
    reduced_flag = getattr(cv2, "IMREAD_REDUCED_COLOR_2", None)
    if (
        ext in {".jpg", ".jpeg", ".jpe", ".jfif"}
        and size_bytes > 900 * 1024
        and reduced_flag is not None
    ):
        bgr = cv2.imread(p, reduced_flag)
    if bgr is None:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = min(canvas_size / max(1, w), canvas_size / max(1, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(rgb, (nw, nh), interpolation=interpolation)
    canvas = np.full((canvas_size, canvas_size, 3), 255, dtype=np.uint8)
    x0, y0 = (canvas_size - nw) // 2, (canvas_size - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _foreground_color_constancy_lab_l(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalize L* percentiles on foreground only while preserving chromatic a*/b* channels."""
    valid = mask > 0
    if int(valid.sum()) < 200:
        return img_rgb
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    vals = l_ch[valid].astype(np.float32)
    lo, hi = np.percentile(vals, [5, 95])
    if hi - lo >= 8:
        l32 = l_ch.astype(np.float32)
        l32[valid] = np.clip((l32[valid] - lo) * (220.0 / (hi - lo)) + 18.0, 0, 255)
        lab = cv2.merge([l32.astype(np.uint8), a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def _foreground_gray_world_bounded(
    img_rgb: np.ndarray, mask: np.ndarray, gain_limit=(0.82, 1.22)
) -> np.ndarray:
    """Bounded per-channel Gray-World foreground color constancy."""
    valid = mask > 0
    if int(valid.sum()) < 200:
        return img_rgb
    out = img_rgb.astype(np.float32)
    means = np.array([np.mean(out[:, :, c][valid]) for c in range(3)], dtype=np.float32)
    target = float(np.mean(means))
    gains = np.clip(target / np.maximum(means, 1e-5), gain_limit[0], gain_limit[1])
    for channel in range(3):
        plane = out[:, :, channel]
        plane[valid] = np.clip(plane[valid] * gains[channel], 0, 255)
        out[:, :, channel] = plane
    return out.astype(np.uint8)


def canonicalize_leaf_organ_image(
    img_rgb: np.ndarray, color_mode: str = None, output_size: int = 256
) -> np.ndarray:
    """
    Canonical Leaf Pose Normalization for Cross-Dataset Generalization:
      1) Segment foreground leaf mask;
      2) Align principal PCA axis horizontally (0 degrees);
      3) Tight-crop and aspect-preserving letterbox to output_size x output_size;
      4) Apply foreground L* luminance constancy and adaptive quality preprocessing.
    """
    mode = color_mode or CANONICAL_COLOR_MODE
    _, _, mask = segment_plant_organ_roi(img_rgb)
    ys, xs = np.where(mask > 0)
    h, w = img_rgb.shape[:2]
    if len(xs) < 200:
        return cv2.resize(img_rgb, (output_size, output_size), interpolation=cv2.INTER_AREA)

    pts = np.column_stack([xs, ys]).astype(np.float32)
    center = pts.mean(axis=0)
    centered = pts - center
    cov = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = float(np.degrees(np.arctan2(major[1], major[0])))
    rot_mat = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, 1.0)
    rgb_rot = cv2.warpAffine(
        img_rgb, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255)
    )
    mask_rot = cv2.warpAffine(
        mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
    )
    yy, xx = np.where(mask_rot > 0)
    if len(xx) < 100:
        return cv2.resize(img_rgb, (output_size, output_size), interpolation=cv2.INTER_AREA)
    pad = max(2, int(0.025 * max(h, w)))
    x0, x1 = max(0, int(xx.min()) - pad), min(w, int(xx.max()) + pad + 1)
    y0, y1 = max(0, int(yy.min()) - pad), min(h, int(yy.max()) + pad + 1)
    crop_rgb = rgb_rot[y0:y1, x0:x1]
    crop_mask = mask_rot[y0:y1, x0:x1]
    crop_fg = cv2.bitwise_and(crop_rgb, crop_rgb, mask=crop_mask)

    ch, cw = crop_fg.shape[:2]
    scale = min((output_size - 8) / max(1, cw), (output_size - 8) / max(1, ch))
    nw, nh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    fg_resized = cv2.resize(crop_fg, (nw, nh), interpolation=interp)
    mask_resized = cv2.resize(crop_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((output_size, output_size, 3), dtype=np.uint8)
    out_mask = np.zeros((output_size, output_size), dtype=np.uint8)
    ox, oy = (output_size - nw) // 2, (output_size - nh) // 2
    out[oy : oy + nh, ox : ox + nw] = fg_resized
    out_mask[oy : oy + nh, ox : ox + nw] = mask_resized

    if mode == "gray_world_bounded":
        out = _foreground_gray_world_bounded(out, out_mask)
    else:
        out = _foreground_color_constancy_lab_l(out, out_mask)
    out = cv2.bitwise_and(out, out, mask=out_mask)
    return apply_adaptive_quality_preprocessing(out, foreground_mask=out_mask)
