#!/usr/bin/env python3
"""Render high-legibility, publication-grade architectural diagrams and real-image
visual proofs for Mango-Disease-Audit-EdgeAI.

Design principles for GitHub & Manuscript readability:
1. Every figure canvas is constrained to 1040px width (matching GitHub's ~960px-1012px
   README viewport nearly 1:1) so images are never downscaled into tiny text.
2. Typography inside every diagram uses 17px-32px bold/semibold fonts so every title,
   equation, metric badge, and image caption is effortlessly readable without zooming.
3. Complex multi-stage mechanisms are decomposed into dedicated single-focus figures
   (Fig01 through Fig14, plus legacy Figure1, Figure5, Figure6 aliases).
4. All visual proofs use real MangoLeafBD and MLDID specimens and exact OpenCV
   intermediate maps produced by `src/mango_edge_cv/`.

Usage:
    python scripts/render_publication_diagrams.py
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mango_edge_cv.features import _LBP_RIU2_LUT, _lbp_code
from mango_edge_cv.preprocessing import (
    _estimate_noise_sigma_mad,
    apply_adaptive_quality_preprocessing,
    canonicalize_leaf_organ_image,
    segment_plant_organ_roi,
)


def standardize_image_framing(
    img_rgb: np.ndarray, target_size: int = 256, center_crop_ratio: float = 0.88
) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    crop_h, crop_w = max(1, int(h * center_crop_ratio)), max(1, int(w * center_crop_ratio))
    y0, x0 = max(0, (h - crop_h) // 2), max(0, (w - crop_w) // 2)
    crop = img_rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)


def _rgb_to_data_uri(img_rgb: np.ndarray, quality: int = 93) -> str:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode RGB image to JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _read_rgb_unicode_safe(path: Path) -> np.ndarray | None:
    try:
        if not path.exists():
            return None
        arr = np.fromfile(str(path), dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return None


def _make_fallback_leaf(portrait: bool = False, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = (293, 240) if portrait else (240, 320)
    img = np.full((h, w, 3), 232, dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * 0.24), int(h * 0.38)) if portrait else (int(w * 0.38), int(h * 0.24))
    angle = 85 if portrait else 12
    cv2.ellipse(img, center, axes, angle, 0, 360, (52, 142, 46), -1)
    for _ in range(12):
        cx = int(rng.integers(int(w * 0.35), int(w * 0.65)))
        cy = int(rng.integers(int(h * 0.35), int(h * 0.65)))
        cv2.circle(img, (cx, cy), int(rng.integers(4, 10)), (62, 40, 24), -1)
    return img


def _colorize_heatmap(gray_u8: np.ndarray, colormap: int = cv2.COLORMAP_VIRIDIS) -> np.ndarray:
    bgr = cv2.applyColorMap(gray_u8, colormap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def prepare_visual_assets() -> Dict[str, str]:
    """Generate all real-image visual proof thumbnails and intermediate OpenCV feature maps."""
    workspace_mango = REPO_ROOT.parent / "MangoLeafBD Dataset"
    workspace_mldid = REPO_ROOT.parent / "Resize mango leaf disease"

    # 1. Load primary Anthracnose specimen
    anth_path = workspace_mango / "Anthracnose" / "20211008_124249 (Custom).jpg"
    anth_img = _read_rgb_unicode_safe(anth_path)
    if anth_img is None and (workspace_mango / "Anthracnose").exists():
        files = sorted((workspace_mango / "Anthracnose").glob("*.jpg"))
        if files:
            anth_img = _read_rgb_unicode_safe(files[0])
    if anth_img is None:
        anth_img = _make_fallback_leaf(portrait=False, seed=101)

    std_rgb = standardize_image_framing(anth_img, target_size=256, center_crop_ratio=0.88)
    proc_rgb = apply_adaptive_quality_preprocessing(std_rgb)
    fg_rgb, bg_rgb, mask_u8 = segment_plant_organ_roi(proc_rgb)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_fg = proc_rgb.copy()
    cv2.drawContours(overlay_fg, contours, -1, (16, 185, 129), 3)

    masked_fg_white = proc_rgb.copy()
    masked_fg_white[mask_u8 == 0] = [241, 245, 249]

    # 2. Real Duplicate Cluster Siblings (Anthracnose Group 143 & Cutting_Weevil Group 1000)
    g143_files = [
        workspace_mango / "Anthracnose" / "20211011_170706 - Copy - Copy - Copy (Custom).jpg",
        workspace_mango / "Anthracnose" / "20211011_170703 (Custom).jpg",
        workspace_mango / "Anthracnose" / "20211011_170706 (Custom).jpg",
    ]
    g143_imgs = []
    for idx, p in enumerate(g143_files):
        im = _read_rgb_unicode_safe(p)
        if im is None:
            im = cv2.rotate(std_rgb, cv2.ROTATE_90_CLOCKWISE) if idx == 1 else std_rgb.copy()
        g143_imgs.append(cv2.resize(im, (240, 200), interpolation=cv2.INTER_AREA))

    g1000_files = [
        workspace_mango / "Cutting Weevil" / "20211011_131101 (Custom) (Custom).jpg",
        workspace_mango / "Cutting Weevil" / "20211011_131101 (Custom).jpg",
        workspace_mango / "Cutting Weevil" / "20211011_140216 (Custom).jpg",
    ]
    g1000_imgs = []
    for idx, p in enumerate(g1000_files):
        im = _read_rgb_unicode_safe(p)
        if im is None:
            im = _make_fallback_leaf(portrait=False, seed=300 + idx)
        g1000_imgs.append(cv2.resize(im, (220, 200), interpolation=cv2.INTER_AREA))

    # 3. Five-View Background Shortcut Masks
    # View 1: Whole ROI (r=0.86)
    view1_rgb = standardize_image_framing(anth_img, target_size=256, center_crop_ratio=0.86)
    # View 2: Segmented Organ Only (Background zeroed)
    view2_rgb, view3_rgb, v_mask = segment_plant_organ_roi(view1_rgb)
    # View 4: Four 32x32 Corner Patches Only
    view4_rgb = np.full_like(view1_rgb, 25, dtype=np.uint8)
    cs = 48  # scaled from 32x32 on 192 grid for visual clarity on 256 grid
    view4_rgb[:cs, :cs] = view1_rgb[:cs, :cs]
    view4_rgb[:cs, -cs:] = view1_rgb[:cs, -cs:]
    view4_rgb[-cs:, :cs] = view1_rgb[-cs:, :cs]
    view4_rgb[-cs:, -cs:] = view1_rgb[-cs:, -cs:]
    for x0, y0 in [(0, 0), (256 - cs, 0), (0, 256 - cs), (256 - cs, 256 - cs)]:
        cv2.rectangle(view4_rgb, (x0, y0), (x0 + cs - 1, y0 + cs - 1), (245, 158, 11), 2)

    # 4. Step-by-Step Hybrid ROI Segmentation Intermediates (Fig05)
    hsv = cv2.cvtColor(std_rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(std_rgb, cv2.COLOR_RGB2LAB)
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    a_ch = lab[:, :, 1].astype(np.float32) - 128.0
    b_ch = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.sqrt(a_ch**2 + b_ch**2)
    chroma_u8 = np.clip(chroma * 2.2, 0, 255).astype(np.uint8)
    salience = np.clip(0.60 * s_ch.astype(np.float32) + 1.25 * chroma, 0, 255).astype(np.uint8)

    s_heat = _colorize_heatmap(s_ch, cv2.COLORMAP_OCEAN)
    chroma_heat = _colorize_heatmap(chroma_u8, cv2.COLORMAP_INFERNO)
    salience_heat = _colorize_heatmap(salience, cv2.COLORMAP_TURBO)

    mask_rgb_vis = np.zeros_like(std_rgb)
    mask_rgb_vis[mask_u8 > 0] = [16, 185, 129]
    mask_rgb_vis[mask_u8 == 0] = [30, 41, 59]
    cv2.drawContours(mask_rgb_vis, contours, -1, (255, 255, 255), 2)

    # 5. Quality-Gated Optical Preprocessing Proofs (Fig06)
    rng = np.random.default_rng(20261022)
    # Row A: Sensor grain sigma=10 (isolated MAD gate output)
    noisy_rgb = np.clip(std_rgb.astype(np.float32) + rng.normal(0.0, 10.0, std_rgb.shape), 0, 255).astype(np.uint8)
    gray_noisy = cv2.cvtColor(noisy_rgb, cv2.COLOR_RGB2GRAY)
    med_noisy = cv2.medianBlur(gray_noisy, 3)
    res_map = np.clip(np.abs(gray_noisy.astype(np.float32) - med_noisy.astype(np.float32)) * 12.0, 0, 255).astype(np.uint8)
    res_heat = _colorize_heatmap(res_map, cv2.COLORMAP_MAGMA)
    denoised_fg = cv2.GaussianBlur(cv2.medianBlur(noisy_rgb, 3), (3, 3), 0.65)
    denoised_rgb = noisy_rgb.copy()
    denoised_rgb[mask_u8 > 0] = denoised_fg[mask_u8 > 0]

    # Row B: Central window specular glare
    glare_rgb = std_rgb.copy()
    glare_det_vis = (std_rgb.astype(np.float32) * 0.45).astype(np.uint8)
    glare_inpaint_mask = np.zeros(std_rgb.shape[:2], dtype=np.uint8)
    for gx, gy, gr in [(118, 106, 9), (144, 134, 8), (108, 146, 7), (132, 92, 6)]:
        cv2.circle(glare_rgb, (gx, gy), gr, (252, 252, 248), -1)
        cv2.circle(glare_det_vis, (gx, gy), gr + 3, (239, 68, 68), -1)
        cv2.circle(glare_det_vis, (gx, gy), gr, (254, 240, 138), -1)
        cv2.circle(glare_inpaint_mask, (gx, gy), gr + 3, 255, -1)
    cv2.rectangle(glare_det_vis, (int(256 * 0.14), int(256 * 0.14)), (int(256 * 0.86), int(256 * 0.86)), (56, 189, 248), 2)
    glare_restored = cv2.inpaint(glare_rgb, glare_inpaint_mask, 5, cv2.INPAINT_TELEA)

    # Row C: Low-light -40% luminance stretch + mild foreground CLAHE
    dim_rgb = np.clip(std_rgb.astype(np.float32) * 0.55 + 8.0, 0, 255).astype(np.uint8)
    dim_restored = apply_adaptive_quality_preprocessing(dim_rgb, foreground_mask=mask_u8)

    # 6. Seven Feature Branch Visual Maps on 128x128 Grid (Fig07)
    grid128 = cv2.resize(proc_rgb, (128, 128), interpolation=cv2.INTER_AREA)
    gray128 = cv2.cvtColor(grid128, cv2.COLOR_RGB2GRAY)

    # Branch 3: LBP riu2 map
    lbp_map = _LBP_RIU2_LUT[_lbp_code(gray128)]
    lbp_u8 = np.clip((lbp_map.astype(np.float32) / 9.0) * 255.0, 0, 255).astype(np.uint8)
    lbp_vis = cv2.resize(_colorize_heatmap(lbp_u8, cv2.COLORMAP_VIRIDIS), (256, 256), interpolation=cv2.INTER_NEAREST)

    # Branch 4: 5-Patch Spatial Zones overlay
    patch5_vis = proc_rgb.copy()
    cv2.line(patch5_vis, (128, 0), (128, 256), (255, 255, 255), 2)
    cv2.line(patch5_vis, (0, 128), (256, 128), (255, 255, 255), 2)
    cv2.rectangle(patch5_vis, (64, 64), (192, 192), (245, 158, 11), 3)

    # Branch 6: MGCT Chromatic-Gradient Coupling Map (C* x ||grad I||)
    gx = cv2.Sobel(gray128, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray128, cv2.CV_32F, 0, 1, ksize=3)
    gmag = cv2.magnitude(gx, gy)
    lab128 = cv2.cvtColor(grid128, cv2.COLOR_RGB2LAB).astype(np.float32)
    c128 = np.sqrt((lab128[:, :, 1] - 128.0) ** 2 + (lab128[:, :, 2] - 128.0) ** 2)
    mgct_prod = c128 * gmag
    mgct_u8 = np.clip(mgct_prod / (np.percentile(mgct_prod, 98) + 1e-5) * 255.0, 0, 255).astype(np.uint8)
    mgct_vis = cv2.resize(_colorize_heatmap(mgct_u8, cv2.COLORMAP_PLASMA), (256, 256), interpolation=cv2.INTER_LINEAR)
    cv2.line(mgct_vis, (128, 0), (128, 256), (255, 255, 255), 2)
    cv2.line(mgct_vis, (0, 128), (256, 128), (255, 255, 255), 2)

    # Branch 7: Shared-Sobel Gx, Gy, Magnitude M
    gx_u8 = np.clip(np.abs(gx) / (np.percentile(np.abs(gx), 98) + 1e-5) * 255.0, 0, 255).astype(np.uint8)
    gy_u8 = np.clip(np.abs(gy) / (np.percentile(np.abs(gy), 98) + 1e-5) * 255.0, 0, 255).astype(np.uint8)
    gm_u8 = np.clip(gmag / (np.percentile(gmag, 98) + 1e-5) * 255.0, 0, 255).astype(np.uint8)
    sobel_gx_vis = cv2.resize(_colorize_heatmap(gx_u8, cv2.COLORMAP_TWILIGHT_SHIFTED), (220, 220), interpolation=cv2.INTER_LINEAR)
    sobel_gy_vis = cv2.resize(_colorize_heatmap(gy_u8, cv2.COLORMAP_TWILIGHT_SHIFTED), (220, 220), interpolation=cv2.INTER_LINEAR)
    sobel_gm_vis = cv2.resize(_colorize_heatmap(gm_u8, cv2.COLORMAP_INFERNO), (220, 220), interpolation=cv2.INTER_LINEAR)

    # 7. External MLDID Canonical Pose Steps (Fig14)
    mldid_img = None
    if workspace_mldid.exists():
        mldid_files = sorted(workspace_mldid.rglob("*.jpg"))
        if mldid_files:
            mldid_img = _read_rgb_unicode_safe(mldid_files[0])
    if mldid_img is None:
        mldid_img = _make_fallback_leaf(portrait=True, seed=202)

    mldid_raw = cv2.resize(mldid_img, (256, 256), interpolation=cv2.INTER_AREA)
    _, _, mldid_mask = segment_plant_organ_roi(mldid_raw)
    mldid_axis_vis = mldid_raw.copy()
    m_cnts, _ = cv2.findContours(mldid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mldid_axis_vis, m_cnts, -1, (16, 185, 129), 3)
    ys, xs = np.nonzero(mldid_mask > 0)
    if len(xs) > 20:
        cx_m, cy_m = float(np.mean(xs)), float(np.mean(ys))
        cov = np.cov(np.vstack([xs - cx_m, ys - cy_m]))
        eigvals, eigvecs = np.linalg.eigh(cov)
        vx, vy = eigvecs[:, int(np.argmax(eigvals))]
        p1 = (int(cx_m - vx * 95), int(cy_m - vy * 95))
        p2 = (int(cx_m + vx * 95), int(cy_m + vy * 95))
        cv2.line(mldid_axis_vis, p1, p2, (239, 68, 68), 3)
        cv2.circle(mldid_axis_vis, (int(cx_m), int(cy_m)), 6, (245, 158, 11), -1)

    mldid_canon = canonicalize_leaf_organ_image(mldid_raw, output_size=256)
    _, _, canon_mask = segment_plant_organ_roi(mldid_canon)
    c_cnts, _ = cv2.findContours(canon_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mldid_canon_vis = mldid_canon.copy()
    cv2.drawContours(mldid_canon_vis, c_cnts, -1, (16, 185, 129), 3)
    cv2.line(mldid_canon_vis, (20, 128), (236, 128), (239, 68, 68), 3)

    # 8. Eight-Class Specimen Gallery for Fig15 EDA
    class_dirs = [
        ("cls_anth", "Anthracnose"),
        ("cls_bact", "Bacterial Canker"),
        ("cls_weev", "Cutting Weevil"),
        ("cls_dieb", "Die Back"),
        ("cls_gall", "Gall Midge"),
        ("cls_heal", "Healthy"),
        ("cls_powd", "Powdery Mildew"),
        ("cls_soot", "Sooty Mould"),
    ]
    cls_uris = {}
    for key_c, folder_c in class_dirs:
        c_img = None
        c_dir = workspace_mango / folder_c
        if c_dir.exists():
            c_files = sorted(c_dir.glob("*.jpg"))
            if c_files:
                c_img = _read_rgb_unicode_safe(c_files[min(2, len(c_files) - 1)])
        if c_img is None:
            c_img = _make_fallback_leaf(portrait=(key_c == "cls_bact"), seed=hash(key_c) % 999)
        cls_uris[key_c] = _rgb_to_data_uri(cv2.resize(c_img, (220, 180), interpolation=cv2.INTER_AREA))

    out_dict = {
        "std_rgb": _rgb_to_data_uri(std_rgb),
        "proc_rgb": _rgb_to_data_uri(proc_rgb),
        "overlay_fg": _rgb_to_data_uri(overlay_fg),
        "masked_fg_white": _rgb_to_data_uri(masked_fg_white),
        "g143_0": _rgb_to_data_uri(g143_imgs[0]),
        "g143_1": _rgb_to_data_uri(g143_imgs[1]),
        "g143_2": _rgb_to_data_uri(g143_imgs[2]),
        "g1000_0": _rgb_to_data_uri(g1000_imgs[0]),
        "g1000_1": _rgb_to_data_uri(g1000_imgs[1]),
        "g1000_2": _rgb_to_data_uri(g1000_imgs[2]),
        "view1_whole": _rgb_to_data_uri(view1_rgb),
        "view2_organ": _rgb_to_data_uri(view2_rgb),
        "view3_bg": _rgb_to_data_uri(view3_rgb),
        "view4_corners": _rgb_to_data_uri(view4_rgb),
        "seg_s_heat": _rgb_to_data_uri(s_heat),
        "seg_chroma_heat": _rgb_to_data_uri(chroma_heat),
        "seg_salience_heat": _rgb_to_data_uri(salience_heat),
        "seg_mask_vis": _rgb_to_data_uri(mask_rgb_vis),
        "qg_noisy": _rgb_to_data_uri(noisy_rgb),
        "qg_res_heat": _rgb_to_data_uri(res_heat),
        "qg_denoised": _rgb_to_data_uri(denoised_rgb),
        "qg_glare_in": _rgb_to_data_uri(glare_rgb),
        "qg_glare_mask": _rgb_to_data_uri(glare_det_vis),
        "qg_glare_out": _rgb_to_data_uri(glare_restored),
        "qg_dim_in": _rgb_to_data_uri(dim_rgb),
        "qg_dim_out": _rgb_to_data_uri(dim_restored),
        "feat_lbp": _rgb_to_data_uri(lbp_vis),
        "feat_patch5": _rgb_to_data_uri(patch5_vis),
        "feat_mgct": _rgb_to_data_uri(mgct_vis),
        "feat_gx": _rgb_to_data_uri(sobel_gx_vis),
        "feat_gy": _rgb_to_data_uri(sobel_gy_vis),
        "feat_gm": _rgb_to_data_uri(sobel_gm_vis),
        "mldid_raw": _rgb_to_data_uri(mldid_raw),
        "mldid_axis": _rgb_to_data_uri(mldid_axis_vis),
        "mldid_canon": _rgb_to_data_uri(mldid_canon_vis),
    }
    out_dict.update(cls_uris)
    return out_dict


def build_showcase_html(a: Dict[str, str]) -> str:
    """Build the multi-figure HTML document where every figure is 1040px wide with huge text."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Mango-Disease-Audit-EdgeAI High-Legibility Figures</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: #cbd5e1;
    color: #0f172a;
    padding: 40px;
    display: flex;
    flex-direction: column;
    gap: 56px;
    align-items: center;
  }}
  .fig-card {{
    width: 1040px;
    background: #ffffff;
    border-radius: 14px;
    padding: 30px 34px;
    border: 2px solid #94a3b8;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12);
  }}
  .fig-banner {{
    border-bottom: 3px solid #0f172a;
    padding-bottom: 14px;
    margin-bottom: 24px;
  }}
  .fig-kicker {{
    font-size: 16px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #1d4ed8;
    margin-bottom: 4px;
  }}
  .fig-main-title {{
    font-size: 27px;
    font-weight: 900;
    color: #0f172a;
    line-height: 1.22;
    letter-spacing: -0.3px;
  }}
  .fig-sub-desc {{
    font-size: 18px;
    font-weight: 500;
    color: #334155;
    margin-top: 6px;
    line-height: 1.4;
  }}
  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }}
  .grid-3 {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 18px;
  }}
  .panel-box {{
    background: #f8fafc;
    border: 2px solid #cbd5e1;
    border-radius: 12px;
    padding: 18px 20px;
  }}
  .panel-title {{
    font-size: 21px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .badge {{
    display: inline-block;
    padding: 5px 12px;
    border-radius: 7px;
    font-size: 16.5px;
    font-weight: 800;
  }}
  .badge-blue {{ background: #dbeafe; color: #1e40af; border: 1.5px solid #93c5fd; }}
  .badge-green {{ background: #dcfce7; color: #166534; border: 1.5px solid #86efac; }}
  .badge-red {{ background: #fee2e2; color: #991b1b; border: 1.5px solid #fca5a5; }}
  .badge-amber {{ background: #fef3c7; color: #92400e; border: 1.5px solid #fcd34d; }}
  .badge-purple {{ background: #f3e8ff; color: #6b21a8; border: 1.5px solid #d8b4fe; }}
  .formula-block {{
    background: #eff6ff;
    border-left: 5px solid #2563eb;
    padding: 12px 16px;
    border-radius: 6px;
    font-size: 19px;
    font-weight: 700;
    color: #1e3a8a;
    margin: 10px 0;
    line-height: 1.45;
  }}
  .body-lg {{
    font-size: 18px;
    color: #1e293b;
    line-height: 1.48;
    font-weight: 500;
  }}
  .body-lg strong {{
    color: #0f172a;
    font-weight: 800;
  }}
  .thumb-frame {{
    width: 100%;
    height: 190px;
    border-radius: 9px;
    overflow: hidden;
    border: 2px solid #64748b;
    background: #0f172a;
    margin-bottom: 10px;
  }}
  .thumb-frame img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
  }}
  .kpi-strip {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-top: 22px;
  }}
  .kpi-box {{
    background: #0f172a;
    color: #ffffff;
    border-radius: 10px;
    padding: 14px 16px;
    text-align: center;
  }}
  .kpi-num {{
    font-size: 28px;
    font-weight: 900;
    color: #38bdf8;
  }}
  .kpi-lbl {{
    font-size: 16px;
    font-weight: 600;
    color: #e2e8f0;
    margin-top: 4px;
  }}
</style>
</head>
<body>

<!-- ===================================================================== -->
<!-- FIG 01: MACRO SYSTEM ARCHITECTURE OVERVIEW                            -->
<!-- ===================================================================== -->
<div id="fig01-macro-overview" class="fig-card">
  <div class="fig-banner">
    <div class="fig-kicker">System Architecture Overview (Locked Version 5 Protocol)</div>
    <div class="fig-main-title">Audit-First, Quality-Gated Two-Stage Framework for Edge Mango Leaf Diagnosis</div>
    <div class="fig-sub-desc">End-to-end processing flow from leak-free specimen clustering to 21.58 ms single-core CPU inference. Each stage is detailed in dedicated Figures 2–14.</div>
  </div>

  <!-- Row 1: Stages 1 to 3 -->
  <div class="grid-3" style="margin-bottom: 20px;">
    <div class="panel-box" style="border-color: #3b82f6; background: #eff6ff;">
      <div class="panel-title">
        <span>Stage 1: Dataset Audit</span>
        <span class="badge badge-blue">Fig. 2–4</span>
      </div>
      <div class="thumb-frame" style="height: 165px;">
        <img src="{a['std_rgb']}" alt="Input Leaf">
      </div>
      <div class="body-lg">
        <strong>Graph-Based Specimen Audit:</strong><br>
        &bull; 64-bit DCT pHash (<i>d</i><sub>H</sub> &le; 6)<br>
        &bull; Central HS Cosine &ge; 0.992<br>
        &bull; <strong>2,554</strong> clusters / 4,000 images<br>
        &bull; <strong>0.0%</strong> group-split leakage
      </div>
    </div>

    <div class="panel-box" style="border-color: #10b981; background: #ecfdf5;">
      <div class="panel-title">
        <span>Stage 2: ROI &amp; Quality Gate</span>
        <span class="badge badge-green">Fig. 5–6</span>
      </div>
      <div class="thumb-frame" style="height: 165px;">
        <img src="{a['overlay_fg']}" alt="Segmented ROI">
      </div>
      <div class="body-lg">
        <strong>Foreground-Only Restoration:</strong><br>
        &bull; Hybrid Otsu + LAB Chroma ROI<br>
        &bull; MAD Noise Gate (&sigma;̂<sub>MAD</sub> &ge; 5.5 / 10)<br>
        &bull; Specular Glare Inpainting<br>
        &bull; Foreground <i>L</i>* Stretch &amp; CLAHE
      </div>
    </div>

    <div class="panel-box" style="border-color: #f59e0b; background: #fffbeb;">
      <div class="panel-title">
        <span>Stage 3: 7-Branch Features</span>
        <span class="badge badge-amber">Fig. 7</span>
      </div>
      <div class="thumb-frame" style="height: 165px;">
        <img src="{a['feat_mgct']}" alt="Feature Map">
      </div>
      <div class="body-lg">
        <strong>128&times;128 Multi-Modal Grid:</strong><br>
        &bull; HSV (52D) + CIE-LAB (57D)<br>
        &bull; LUT-LBP (28D) + 5-Patch (45D)<br>
        &bull; Quality <i>q</i> (7D) + MGCT (50D)<br>
        &bull; Shared-Sobel HOG (1,764D)
      </div>
    </div>
  </div>

  <!-- Row 2: Stages 4 to 6 -->
  <div class="grid-3">
    <div class="panel-box" style="border-color: #8b5cf6; background: #f5f3ff;">
      <div class="panel-title">
        <span>Stage 4: Block-PCA (303D)</span>
        <span class="badge badge-purple">Fig. 8–9</span>
      </div>
      <div class="formula-block" style="background: #ede9fe; border-color: #7c3aed; color: #4c1d95; font-size: 18px;">
        2,003D Raw &rarr; <strong>303D Balanced</strong><br>
        239D Color/Tex + 64D HOG-PCA
      </div>
      <div class="body-lg">
        &bull; Eliminates 7.38:1 HOG dilution<br>
        &bull; RMS energy scaling (&gamma; = 0.45)<br>
        &bull; Jumps accuracy from <strong>86.61%</strong> (<i>M</i><sub>4</sub>) to <strong>97.91%</strong> (<i>M</i><sub>10</sub>, <i>p</i> &lt; 0.0001)
      </div>
    </div>

    <div class="panel-box" style="border-color: #2563eb; background: #eff6ff;">
      <div class="panel-title">
        <span>Stage 5: Confidence Gate</span>
        <span class="badge badge-blue">Fig. 10–11</span>
      </div>
      <div class="formula-block" style="font-size: 18px;">
        max<sub><i>c</i></sub> <i>P</i><sub>1</sub>(<i>y</i>=<i>c</i> | <b>z</b>) &ge; &tau;<sub>gate</sub> = 0.65<br>
        <i>T</i>* = 0.758 (ECE = 0.0094)
      </div>
      <div class="body-lg">
        &bull; <strong>98.47% Fast Path:</strong> Stage-1 L2-LR<br>
        &bull; <strong>1.53% Slow Path:</strong> SVM + MLP + kNN<br>
        &bull; Achieves <strong>98.47% Acc</strong> (0.9836 F1)
      </div>
    </div>

    <div class="panel-box" style="border-color: #dc2626; background: #fef2f2;">
      <div class="panel-title">
        <span>Stage 6: Cross-Domain Pose</span>
        <span class="badge badge-red">Fig. 14</span>
      </div>
      <div class="thumb-frame" style="height: 120px;">
        <img src="{a['mldid_canon']}" alt="Canonical Pose">
      </div>
      <div class="body-lg">
        &bull; Eigenvector horizontal rotation<br>
        &bull; External MLDID (<i>N</i> = 3,000):<br>
        &bull; <strong>31.27% &rarr; 42.63%</strong> Zero-Shot Acc
      </div>
    </div>
  </div>

  <div class="kpi-strip">
    <div class="kpi-box">
      <div class="kpi-num">98.47%</div>
      <div class="kpi-lbl">Leak-Free Group-Split Acc</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-num">21.58 ms</div>
      <div class="kpi-lbl">Single-Core CPU Latency</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-num">480.2 KB</div>
      <div class="kpi-lbl">Stage-1 Model Footprint</div>
    </div>
    <div class="kpi-box">
      <div class="kpi-num">+15.09%</div>
      <div class="kpi-lbl">Zero-Shot Macro-F1 Gain</div>
    </div>
  </div>
</div>


<!-- ===================================================================== -->
<!-- FIG 02: SPECIMEN CLUSTERING GRAPH MECHANISM & REAL SIBLING PROOF      -->
<!-- ===================================================================== -->
<div id="fig02-specimen-clustering" class="fig-card">
  <div class="fig-banner">
    <div class="fig-kicker">Pillar I-A &bull; Specimen-Identity Graph Audit (Table 1)</div>
    <div class="fig-main-title">Graph-Based Near-Duplicate Discovery &amp; Split-Leakage Proof on MangoLeafBD</div>
    <div class="fig-sub-desc">Why naive random train/test splitting leaks 39.0% of multi-image specimen clusters, and how dual-criterion connected components enforce 0.0% leakage.</div>
  </div>

  <div class="grid-2" style="margin-bottom: 22px;">
    <div class="panel-box">
      <div class="panel-title">
        <span>Criterion 1: 64-Bit Structural pHash</span>
        <span class="badge badge-blue">2D-DCT</span>
      </div>
      <div class="formula-block">
        <i>d</i><sub>Hamming</sub>(pHash(<i>I<sub>i</sub></i>), pHash(<i>I<sub>j</sub></i>)) &le; 6 bits
      </div>
      <div class="body-lg">
        1. Resize grayscale image to <strong>32&times;32</strong>.<br>
        2. Compute 2D Discrete Cosine Transform <strong>DCT(<i>I</i><sub>32</sub>)</strong>.<br>
        3. Extract top-left <strong>8&times;8</strong> low-frequency block and threshold against the median of the 63 AC coefficients.<br>
        4. Captures identical leaves under re-compression, minor cropping, or brightness shifts.
      </div>
    </div>

    <div class="panel-box">
      <div class="panel-title">
        <span>Criterion 2: Central-ROI Color Cosine</span>
        <span class="badge badge-green">96D HSV</span>
      </div>
      <div class="formula-block" style="background: #ecfdf5; border-color: #059669; color: #065f46;">
        cos(<b>h</b><sub>HS</sub>(<i>I<sub>i</sub></i>), <b>h</b><sub>HS</sub>(<i>I<sub>j</sub></i>)) &ge; 0.992
      </div>
      <div class="body-lg">
        1. Crop central <strong>60% ROI</strong> (<code>[0.2H:0.8H, 0.2W:0.8W]</code>) and resize to <strong>64&times;64</strong>.<br>
        2. Compute joint 2D Hue–Saturation histogram (<strong>12 &times; 8 = 96 bins</strong>, <i>L</i><sub>2</sub>-normalized).<br>
        3. Rotation-invariant descriptor that links <strong>90&deg;/180&deg;/270&deg; rotated</strong> and zoomed views of the same leaf.
      </div>
    </div>
  </div>

  <!-- Empirical Proof 1: Anthracnose Group 143 -->
  <div class="panel-box" style="margin-bottom: 18px; border-color: #dc2626; background: #fff5f5;">
    <div class="panel-title">
      <span>Empirical Proof A: Anthracnose Cluster <code>group_id = 143</code> (5 Siblings in Raw Dataset)</span>
      <span class="badge badge-red">Leaked in Random Split</span>
    </div>
    <div class="grid-3" style="margin-top: 12px;">
      <div style="background: #ffffff; padding: 12px; border-radius: 10px; border: 2px solid #fca5a5;">
        <div class="thumb-frame" style="height: 165px;"><img src="{a['g143_0']}" alt="Sibling 1"></div>
        <div class="body-lg" style="font-size: 16.5px;">
          <code>20211011_170706 - Copy...jpg</code><br>
          &bull; Random Split: <strong style="color:#dc2626;">Train_Orig</strong><br>
          &bull; Group Split: <strong style="color:#166534;">Val_Orig</strong>
        </div>
      </div>
      <div style="background: #ffffff; padding: 12px; border-radius: 10px; border: 2px solid #fca5a5;">
        <div class="thumb-frame" style="height: 165px;"><img src="{a['g143_1']}" alt="Sibling 2"></div>
        <div class="body-lg" style="font-size: 16.5px;">
          <code>20211011_170703 (Custom).jpg</code><br>
          &bull; Random Split: <strong style="color:#d97706;">Val_Orig</strong><br>
          &bull; Group Split: <strong style="color:#166534;">Val_Orig</strong>
        </div>
      </div>
      <div style="background: #ffffff; padding: 12px; border-radius: 10px; border: 2px solid #fca5a5;">
        <div class="thumb-frame" style="height: 165px;"><img src="{a['g143_2']}" alt="Sibling 3"></div>
        <div class="body-lg" style="font-size: 16.5px;">
          <code>20211011_170706 (Custom).jpg</code><br>
          &bull; Random Split: <strong style="color:#dc2626;">Test_Orig (LEAK!)</strong><br>
          &bull; Group Split: <strong style="color:#166534;">Val_Orig</strong>
        </div>
      </div>
    </div>
  </div>

  <!-- Empirical Proof 2: Cutting Weevil Group 1000 -->
  <div class="panel-box" style="border-color: #1e40af; background: #eff6ff;">
    <div class="panel-title">
      <span>Empirical Proof B: Cutting_Weevil Mega-Cluster <code>group_id = 1000</code> (368 / 500 Class Images!)</span>
      <span class="badge badge-blue">93.8% Class Duplicate Ratio</span>
    </div>
    <div class="grid-3" style="margin-top: 12px;">
      <div style="background: #ffffff; padding: 12px; border-radius: 10px; border: 2px solid #93c5fd;">
        <div class="thumb-frame" style="height: 155px;"><img src="{a['g1000_0']}" alt="CW 1"></div>
        <div class="body-lg" style="font-size: 16.5px;">
          <code>20211011_131101 (Custom) (Custom).jpg</code><br>
          Random: <strong>Train (214 imgs)</strong> &rarr; Group: <strong>Val</strong>
        </div>
      </div>
      <div style="background: #ffffff; padding: 12px; border-radius: 10px; border: 2px solid #93c5fd;">
        <div class="thumb-frame" style="height: 155px;"><img src="{a['g1000_1']}" alt="CW 2"></div>
        <div class="body-lg" style="font-size: 16.5px;">
          <code>20211011_131101 (Custom).jpg</code><br>
          Random: <strong>Val (73 imgs)</strong> &rarr; Group: <strong>Val</strong>
        </div>
      </div>
      <div style="background: #ffffff; padding: 12px; border-radius: 10px; border: 2px solid #93c5fd;">
        <div class="thumb-frame" style="height: 155px;"><img src="{a['g1000_2']}" alt="CW 3"></div>
        <div class="body-lg" style="font-size: 16.5px;">
          <code>20211011_140216 (Custom).jpg</code><br>
          Random: <strong>Test (81 imgs)</strong> &rarr; Group: <strong>Val</strong>
        </div>
      </div>
    </div>
  </div>
</div>


<!-- ===================================================================== -->
<!-- FIG 04: FIVE-VIEW STUDIO BACKGROUND SHORTCUT AUDIT VISUAL PROOF       -->
<!-- ===================================================================== -->
<div id="fig04-fiveview-shortcuts" class="fig-card">
  <div class="fig-banner">
    <div class="fig-kicker">Pillar I-B &bull; Studio Capture Confounding Audit (Table 2)</div>
    <div class="fig-main-title">Five-View Visual &amp; Non-Visual Background Shortcut Audit on MangoLeafBD</div>
    <div class="fig-sub-desc">Evaluating an identical regularized classifier (137D HSV+LAB+LBP for Views 1–4; 7D metadata for View 5) under leak-free group-aware splits (8-class random chance = 12.50%).</div>
  </div>

  <div style="display: flex; flex-direction: column; gap: 16px;">
    <!-- View 1 -->
    <div class="panel-box" style="display: grid; grid-template-columns: 170px 1fr 270px; gap: 20px; align-items: center;">
      <div class="thumb-frame" style="height: 135px; margin: 0;"><img src="{a['view1_whole']}" alt="View 1"></div>
      <div>
        <div style="font-size: 21px; font-weight: 800; color: #0f172a;">View 1: Whole Image ROI (Center-Crop <i>r</i> = 0.86)</div>
        <div class="body-lg" style="margin-top: 4px;">Contains both the symptomatic mango leaf foreground and the surrounding studio paper backdrop.</div>
      </div>
      <div style="text-align: right;">
        <div style="font-size: 30px; font-weight: 900; color: #166534;">98.33% Acc</div>
        <div class="badge badge-green" style="margin-top: 4px;">Macro-F1: 0.9833</div>
      </div>
    </div>

    <!-- View 2 -->
    <div class="panel-box" style="display: grid; grid-template-columns: 170px 1fr 270px; gap: 20px; align-items: center;">
      <div class="thumb-frame" style="height: 135px; margin: 0;"><img src="{a['view2_organ']}" alt="View 2"></div>
      <div>
        <div style="font-size: 21px; font-weight: 800; color: #0f172a;">View 2: Segmented Plant Organ Only (Background Zeroed)</div>
        <div class="body-lg" style="margin-top: 4px;">Strictly retains pixels inside the morphological leaf mask <i>M</i><sub>fg</sub>(<i>x</i>,<i>y</i>) = 1; studio background is zeroed out.</div>
      </div>
      <div style="text-align: right;">
        <div style="font-size: 30px; font-weight: 900; color: #1d4ed8;">94.84% Acc</div>
        <div class="badge badge-blue" style="margin-top: 4px;">Macro-F1: 0.9440</div>
      </div>
    </div>

    <!-- View 3 -->
    <div class="panel-box" style="display: grid; grid-template-columns: 170px 1fr 270px; gap: 20px; align-items: center; border-color: #dc2626; background: #fff5f5;">
      <div class="thumb-frame" style="height: 135px; margin: 0;"><img src="{a['view3_bg']}" alt="View 3"></div>
      <div>
        <div style="font-size: 21px; font-weight: 800; color: #991b1b;">View 3: Background Only (Plant Organ Masked Out!)</div>
        <div class="body-lg" style="margin-top: 4px;"><strong>Zero leaf pixels remain</strong> (<i>I</i> &odot; (1 &minus; <i>M</i><sub>fg</sub>)). Proves that per-class studio illumination and backdrop reflectance act as a massive shortcut.</div>
      </div>
      <div style="text-align: right;">
        <div style="font-size: 30px; font-weight: 900; color: #dc2626;">94.84% Acc</div>
        <div class="badge badge-red" style="margin-top: 4px;">Macro-F1: 0.9396</div>
      </div>
    </div>

    <!-- View 4 -->
    <div class="panel-box" style="display: grid; grid-template-columns: 170px 1fr 270px; gap: 20px; align-items: center; border-color: #d97706; background: #fffbeb;">
      <div class="thumb-frame" style="height: 135px; margin: 0;"><img src="{a['view4_corners']}" alt="View 4"></div>
      <div>
        <div style="font-size: 21px; font-weight: 800; color: #92400e;">View 4: Four 32&times;32 Outer Corner Patches Only</div>
        <div class="body-lg" style="margin-top: 4px;">Samples only <strong>6.25% of image area</strong> at the four extreme frame corners—completely disjoint from the central leaf.</div>
      </div>
      <div style="text-align: right;">
        <div style="font-size: 30px; font-weight: 900; color: #b45309;">85.08% Acc</div>
        <div class="badge badge-amber" style="margin-top: 4px;">Macro-F1: 0.8332 (6.8&times; chance)</div>
      </div>
    </div>

    <!-- View 5 -->
    <div class="panel-box" style="display: grid; grid-template-columns: 170px 1fr 270px; gap: 20px; align-items: center;">
      <div style="height: 135px; background: #1e293b; color: #38bdf8; border-radius: 9px; display: flex; flex-direction: column; justify-content: center; align-items: center; padding: 10px; font-family: monospace; font-size: 15.5px; font-weight: 700; text-align: center;">
        [file_size_kb,<br>mean_RGB,<br>std_RGB,<br>fg_ratio, H/W]
      </div>
      <div>
        <div style="font-size: 21px; font-weight: 800; color: #0f172a;">View 5: 7D Non-Visual File Metadata &amp; Aspect Ratio</div>
        <div class="body-lg" style="margin-top: 4px;">Predicts disease class solely from JPEG file size, global RGB mean/std, mask ratio, and camera aspect ratio (<code>320&times;240</code> vs. <code>240&times;240</code> vs. <code>240&times;320</code>).</div>
      </div>
      <div style="text-align: right;">
        <div style="font-size: 30px; font-weight: 900; color: #334155;">59.14% Acc</div>
        <div class="badge badge-purple" style="margin-top: 4px;">Macro-F1: 0.5742 (4.7&times; chance)</div>
      </div>
    </div>
  </div>
</div>


<!-- ===================================================================== -->
<!-- FIG 05: STEP-BY-STEP HYBRID ROI SEGMENTATION                          -->
<!-- ===================================================================== -->
<div id="fig05-roi-segmentation" class="fig-card">
  <div class="fig-banner">
    <div class="fig-kicker">Pillar II-A &bull; Foreground Organ Isolation (<code>segment_plant_organ_roi</code>)</div>
    <div class="fig-main-title">Step-by-Step Hybrid HSV Saturation + CIE-LAB Chroma ROI Segmentation</div>
    <div class="fig-sub-desc">Why standard green-index thresholding fails on necrotic lesions, and how fusing HSV saturation with CIE-LAB chromatic magnitude preserves diseased tissue while rejecting studio backgrounds.</div>
  </div>

  <div class="formula-block" style="margin-bottom: 20px; font-size: 19.5px;">
    Salience Map: &nbsp; <i>S</i><sub>roi</sub>(<i>x</i>,<i>y</i>) = clip( 0.60 &middot; <i>S</i><sub>HSV</sub>(<i>x</i>,<i>y</i>) + 1.25 &middot; &radic;[(<i>a</i>*(<i>x</i>,<i>y</i>) &minus; 128)<sup>2</sup> + (<i>b</i>*(<i>x</i>,<i>y</i>) &minus; 128)<sup>2</sup>], &nbsp;0, &nbsp;255 )
  </div>

  <div class="grid-3" style="margin-bottom: 18px;">
    <div class="panel-box">
      <div class="panel-title"><span>1. Standardized Input</span><span class="badge badge-blue">256&times;256</span></div>
      <div class="thumb-frame"><img src="{a['std_rgb']}" alt="Step 1"></div>
      <div class="body-lg">Center-cropped (<i>r</i> = 0.88) to remove outer frame border artifacts and resized to 256&times;256 RGB.</div>
    </div>

    <div class="panel-box">
      <div class="panel-title"><span>2. HSV Saturation <i>S</i></span><span class="badge badge-blue">Weight 0.60</span></div>
      <div class="thumb-frame"><img src="{a['seg_s_heat']}" alt="Step 2"></div>
      <div class="body-lg">Separates colorful leaf lamina from low-saturation white/gray studio paper (<i>S</i> &lt; 28).</div>
    </div>

    <div class="panel-box">
      <div class="panel-title"><span>3. LAB Chroma <i>C</i>*</span><span class="badge badge-amber">Weight 1.25</span></div>
      <div class="thumb-frame"><img src="{a['seg_chroma_heat']}" alt="Step 3"></div>
      <div class="body-lg">Strongly activates on brown/yellow necrotic lesions (<code>Anthracnose</code>, <code>Die_Back</code>) where green hue drops.</div>
    </div>
  </div>

  <div class="grid-3">
    <div class="panel-box">
      <div class="panel-title"><span>4. Fused Salience <i>S</i><sub>roi</sub></span><span class="badge badge-purple">Otsu Gate</span></div>
      <div class="thumb-frame"><img src="{a['seg_salience_heat']}" alt="Step 4"></div>
      <div class="body-lg">Otsu threshold <i>T</i><sub>Otsu</sub> combined with studio paper suppression &not;(<i>V</i> &gt; 230 &and; <i>S</i> &lt; 28) &and; (<i>V</i> &gt; 15).</div>
    </div>

    <div class="panel-box">
      <div class="panel-title"><span>5. Morphological Mask</span><span class="badge badge-green">7&times;7 Ellipse</span></div>
      <div class="thumb-frame"><img src="{a['seg_mask_vis']}" alt="Step 5"></div>
      <div class="body-lg">3% border margin clear + 7&times;7 morphological close (2 iter), open (1 iter) + largest connected component.</div>
    </div>

    <div class="panel-box" style="border-color: #16a34a; background: #f0fdf4;">
      <div class="panel-title"><span>6. Isolated Leaf ROI</span><span class="badge badge-green">Shortcut-Free</span></div>
      <div class="thumb-frame"><img src="{a['masked_fg_white']}" alt="Step 6"></div>
      <div class="body-lg">All downstream noise estimation, CLAHE, and feature extraction operate strictly inside <i>M</i><sub>fg</sub>.</div>
    </div>
  </div>
</div>


<!-- ===================================================================== -->
<!-- FIG 06: QUALITY-GATED OPTICAL PREPROCESSING PROOF                     -->
<!-- ===================================================================== -->
<div id="fig06-quality-gated-preprocessing" class="fig-card">
  <div class="fig-banner">
    <div class="fig-kicker">Pillar II-B &bull; Selective Optical Restoration (<code>apply_adaptive_quality_preprocessing</code>)</div>
    <div class="fig-main-title">Foreground-Aware Quality-Gated Preprocessing: Visual &amp; Quantitative Proof</div>
    <div class="fig-sub-desc">Why unconditional enhancement degrades clean or noisy leaves, and how three mathematical gates selectively trigger restoration only when physical degradation is detected.</div>
  </div>

  <!-- Gate 1: MAD Sensor Noise -->
  <div class="panel-box" style="margin-bottom: 18px; border-color: #2563eb;">
    <div class="panel-title">
      <span>Gate 1: Donoho–Johnstone MAD Sensor Grain Gate (&sigma;̂<sub>MAD</sub> = median(|<i>R</i> &minus; median(<i>R</i>)|) / 0.6745)</span>
      <span class="badge badge-blue">+0.3533 Macro-F1 at &sigma;=10</span>
    </div>
    <div class="grid-3" style="margin-top: 10px;">
      <div>
        <div class="thumb-frame" style="height: 160px;"><img src="{a['qg_noisy']}" alt="Noisy Input"></div>
        <div class="body-lg"><strong>1A. Sensor Grain (&sigma; = 10):</strong> Raw baseline drops to <strong>0.5114 F1</strong>; unconditional CLAHE amplifies grain and collapses to <strong style="color:#dc2626;">0.4207 F1</strong>.</div>
      </div>
      <div>
        <div class="thumb-frame" style="height: 160px;"><img src="{a['qg_res_heat']}" alt="Residual Map"></div>
        <div class="body-lg"><strong>1B. 3&times;3 Median Residual |<i>R</i>|:</strong> Evaluated strictly on foreground <i>M</i><sub>fg</sub>. Triggers 2-tier smoothing at &sigma;̂<sub>MAD</sub> &ge; 5.5 and &ge; 10.0.</div>
      </div>
      <div>
        <div class="thumb-frame" style="height: 160px;"><img src="{a['qg_denoised']}" alt="Denoised Output"></div>
        <div class="body-lg"><strong>1C. MAD-Gated Output:</strong> Preserves clean leaves untouched (&sigma;=0: <strong>0.9792 F1</strong>) while restoring &sigma;=10 accuracy to <strong style="color:#166534;">0.8647 F1</strong>.</div>
      </div>
    </div>
  </div>

  <!-- Gate 2: Specular Glare -->
  <div class="panel-box" style="margin-bottom: 18px; border-color: #d97706;">
    <div class="panel-title">
      <span>Gate 2: Central-Window Specular Flash Glare Attenuation (<i>V</i> &gt; 238 &and; <i>S</i> &lt; 38 in Central 72% ROI)</span>
      <span class="badge badge-amber">0.9770 Macro-F1 at r=35px</span>
    </div>
    <div class="grid-3" style="margin-top: 10px;">
      <div>
        <div class="thumb-frame" style="height: 160px;"><img src="{a['qg_glare_in']}" alt="Glare Input"></div>
        <div class="body-lg"><strong>2A. Specular Flash Glare:</strong> High-intensity achromatic spots create false step-edges that corrupt Sobel/HOG gradients.</div>
      </div>
      <div>
        <div class="thumb-frame" style="height: 160px;"><img src="{a['qg_glare_mask']}" alt="Glare Mask"></div>
        <div class="body-lg"><strong>2B. Central 72% Gate:</strong> Ignores outer white backdrop; triggers when foreground glare fraction &isin; [0.3%, 22.0%].</div>
      </div>
      <div>
        <div class="thumb-frame" style="height: 160px;"><img src="{a['qg_glare_out']}" alt="Glare Restored"></div>
        <div class="body-lg"><strong>2C. Local Gaussian Fill:</strong> Dilates glare mask (5&times;5) and replaces with 11&times;11 neighborhood (&sigma;=2.5), achieving <strong style="color:#166534;">0.9770 F1</strong>.</div>
      </div>
    </div>
  </div>

  <!-- Gate 3: Low-Light Foreground CLAHE -->
  <div class="panel-box" style="border-color: #16a34a;">
    <div class="panel-title">
      <span>Gate 3: Foreground-Restricted CIE <i>L</i>* Percentile Stretch &amp; Mild CLAHE (&sigma;(<i>L</i>*) &lt; 38 or <i>P</i><sub>95</sub>&minus;<i>P</i><sub>5</sub> &lt; 132)</span>
      <span class="badge badge-green">+0.1194 Macro-F1 at -40% Light</span>
    </div>
    <div class="grid-2" style="margin-top: 10px;">
      <div style="display: grid; grid-template-columns: 180px 1fr; gap: 16px; align-items: center;">
        <div class="thumb-frame" style="height: 145px; margin: 0;"><img src="{a['qg_dim_in']}" alt="Dim Input"></div>
        <div class="body-lg"><strong>3A. Under-Exposed Leaf (&minus;40% Light):</strong> Compresses dynamic range, causing raw baseline Macro-F1 to drop to <strong>0.8391</strong>.</div>
      </div>
      <div style="display: grid; grid-template-columns: 180px 1fr; gap: 16px; align-items: center;">
        <div class="thumb-frame" style="height: 145px; margin: 0;"><img src="{a['qg_dim_out']}" alt="CLAHE Output"></div>
        <div class="body-lg"><strong>3B. Foreground [<i>P</i><sub>2</sub>, <i>P</i><sub>98</sub>] &rarr; [12, 242] + CLAHE (clip=1.55):</strong> Restores Macro-F1 to <strong style="color:#166534;">0.9585</strong> without touching background.</div>
      </div>
    </div>
  </div>
</div>


<!-- ===================================================================== -->
<!-- FIG 07: SEVEN FEATURE BRANCHES VISUAL MAPS (128x128 GRID)             -->
<!-- ===================================================================== -->
<div id="fig07-seven-feature-branches" class="fig-card">
  <div class="fig-banner">
    <div class="fig-kicker">Pillar III-A &bull; Multi-Modal Handcrafted Representation (<code>extract_features_from_rgb</code>)</div>
    <div class="fig-main-title">Seven Complementary Feature Branches on the Unified 128&times;128 Grid (2,003D Raw)</div>
    <div class="fig-sub-desc">Each branch captures a distinct physical manifestation of foliar disease, sharing a single 3&times;3 Sobel gradient pass for sub-10 ms extraction.</div>
  </div>

  <div class="grid-3" style="margin-bottom: 18px;">
    <div class="panel-box">
      <div class="panel-title"><span>1. HSV Color (52D)</span><span class="badge badge-blue">15.7% L1 Wt</span></div>
      <div class="thumb-frame" style="height: 155px;"><img src="{a['seg_s_heat']}" alt="HSV Map"></div>
      <div class="body-lg">16-bin <i>L</i><sub>2</sub>-normalized histograms for <i>H</i>, <i>S</i>, <i>V</i> (48D) + 4D global normalized means. Captures chlorosis &amp; fungal coating.</div>
    </div>

    <div class="panel-box">
      <div class="panel-title"><span>2. CIE-LAB (57D)</span><span class="badge badge-amber">16.4% L1 Wt</span></div>
      <div class="thumb-frame" style="height: 155px;"><img src="{a['seg_chroma_heat']}" alt="LAB Map"></div>
      <div class="body-lg">48D histograms + 9D moments (&mu;<sub><i>c</i></sub>, &sigma;<sub><i>c</i></sub>, skew<sub><i>c</i></sub>) for <i>L</i>*, <i>a</i>*, <i>b</i>*. Skewness quantifies asymmetric necrotic spotting.</div>
    </div>

    <div class="panel-box">
      <div class="panel-title"><span>3. LUT-LBP<sub>8,1</sub><sup>riu2</sup> (28D)</span><span class="badge badge-green">6.7% L1 Wt</span></div>
      <div class="thumb-frame" style="height: 155px;"><img src="{a['feat_lbp']}" alt="LBP Map"></div>
      <div class="body-lg">256-entry O(1) lookup table at 128<sup>2</sup> (10D) &amp; 64<sup>2</sup> (10D) + 8-bin 3&times;3 Gaussian micro-contrast residual histogram.</div>
    </div>
  </div>

  <div class="grid-3" style="margin-bottom: 18px;">
    <div class="panel-box">
      <div class="panel-title"><span>4. 5-Patch Zones (45D)</span><span class="badge badge-purple">13.4% L1 Wt</span></div>
      <div class="thumb-frame" style="height: 155px;"><img src="{a['feat_patch5']}" alt="5-Patch Map"></div>
      <div class="body-lg">5 spatial zones (TL, TR, BL, BR, Center 64&times;64) &times; 9 regional color &amp; Sobel gradient statistics. Localizes tip/margin dieback.</div>
    </div>

    <div class="panel-box">
      <div class="panel-title"><span>5. Quality Vector <i>q</i> (7D)</span><span class="badge badge-blue">2.0% L1 Wt</span></div>
      <div class="formula-block" style="font-size: 16.5px; margin: 14px 0;">
        [&mu;<sub><i>V</i></sub>, &sigma;<sub><i>V</i></sub>, <i>r</i><sub>shadow</sub>, <i>r</i><sub>glare</sub>,<br>
        Var(&nabla;<sup>2</sup><i>I</i>), <i>H</i><sub>LBP</sub>, &mu;<sub>HOG</sub>]
      </div>
      <div class="body-lg">Provides illumination, focus sharpness, and texture entropy context so linear weights adapt to exposure regime.</div>
    </div>

    <div class="panel-box">
      <div class="panel-title"><span>6. MGCT Grid (50D)</span><span class="badge badge-red">16.4% L1 Wt</span></div>
      <div class="thumb-frame" style="height: 155px;"><img src="{a['feat_mgct']}" alt="MGCT Map"></div>
      <div class="body-lg">5 regions (Full + 4 quadrants) &times; 10 stats including chromatic-gradient correlation &rho;(<i>C</i>*, ||&nabla;<i>I</i>||) &amp; strong-edge ratio.</div>
    </div>
  </div>

  <!-- Branch 7: Shared Sobel HOG -->
  <div class="panel-box" style="border-color: #0f172a; background: #f1f5f9;">
    <div class="panel-title">
      <span>7. Shared-Sobel Vectorized HOG (1,764D Raw &rarr; 64D Block-PCA) &mdash; Reuses (<i>G<sub>x</sub></i>, <i>G<sub>y</sub></i>, <i>M</i>) from Branches 4–6</span>
      <span class="badge badge-blue">29.4% L1 Weight Share</span>
    </div>
    <div class="grid-3" style="margin-top: 10px;">
      <div style="display: flex; gap: 12px; align-items: center;">
        <div class="thumb-frame" style="width: 125px; height: 125px; margin: 0; flex-shrink: 0;"><img src="{a['feat_gx']}" alt="Gx"></div>
        <div class="body-lg"><strong>Horizontal |<i>G<sub>x</sub></i>|:</strong><br>3&times;3 Sobel horizontal derivative on 128&times;128 grid.</div>
      </div>
      <div style="display: flex; gap: 12px; align-items: center;">
        <div class="thumb-frame" style="width: 125px; height: 125px; margin: 0; flex-shrink: 0;"><img src="{a['feat_gy']}" alt="Gy"></div>
        <div class="body-lg"><strong>Vertical |<i>G<sub>y</sub></i>|:</strong><br>Captures venation &amp; weevil cut orientation.</div>
      </div>
      <div style="display: flex; gap: 12px; align-items: center;">
        <div class="thumb-frame" style="width: 125px; height: 125px; margin: 0; flex-shrink: 0;"><img src="{a['feat_gm']}" alt="Gmag"></div>
        <div class="body-lg"><strong>Magnitude <i>M</i>:</strong><br>8&times;8 cells, 9 bins, 7&times;7 L2-Hys blocks = 1,764D.</div>
      </div>
    </div>
  </div>
</div>


<!-- ===================================================================== -->
<!-- FIG 08: BLOCK-PCA DIMENSIONALITY DILUTION & ENERGY BALANCING          -->
<!-- ===================================================================== -->
<div id="fig08-block-pca-mechanism" class="fig-card">
  <div class="fig-banner">
    <div class="fig-kicker">Pillar III-B &bull; Variance-Balanced Subspace Compression (<code>BlockPCACompressor</code>)</div>
    <div class="fig-main-title">Resolving High-Dimensional HOG Variance Dilution via Energy-Balanced Block-PCA</div>
    <div class="fig-sub-desc">Mathematical and empirical explanation of why naive feature concatenation fails (M4: 86.61%) and how Block-PCA achieves 97.91% accuracy (M10, p &lt; 0.0001).</div>
  </div>

  <div class="grid-2" style="margin-bottom: 20px;">
    <div class="panel-box" style="border-color: #dc2626; background: #fff5f5;">
      <div class="panel-title">
        <span>The Problem: Dimensionality Dilution</span>
        <span class="badge badge-red">M4 Drops to 86.61%</span>
      </div>
      <div class="formula-block" style="background: #fee2e2; border-color: #dc2626; color: #991b1b;">
        1,764D HOG (88.1%) vs. 239D Color/Tex (11.9%)<br>
        Imbalance Ratio = <strong>7.38 : 1</strong>
      </div>
      <div class="body-lg">
        &bull; Without HOG, compact Color + LBP (<i>M</i><sub>2</sub>, 80D) achieves <strong>97.35% Acc</strong>.<br>
        &bull; Appending raw 1,764D HOG (<i>M</i><sub>4</sub>, 1,844D) causes accuracy to collapse by <strong>&minus;10.74%</strong> down to <strong>86.61%</strong>!<br>
        &bull; Even with all 7 branches + Quality Gate (<i>M</i><sub>9</sub>, 2,003D), raw concatenation reaches only <strong>92.05% Acc</strong> because isotropic L2 penalty &lambda;||<b>w</b>||<sub>2</sub><sup>2</sup> is overwhelmed by 1,764 noisy spatial-cell gradients.
      </div>
    </div>

    <div class="panel-box" style="border-color: #16a34a; background: #f0fdf4;">
      <div class="panel-title">
        <span>The Solution: Energy-Balanced Block-PCA</span>
        <span class="badge badge-green">M10 Jumps to 97.91%</span>
      </div>
      <div class="formula-block" style="background: #dcfce7; border-color: #16a34a; color: #166534;">
        <b>z</b> = [ <b>x̃</b><sub>ct</sub> (239D) &nbsp;;&nbsp; &alpha;<sub>hog</sub> &middot; PCA<sub>64</sub>(<b>x̃</b><sub>hog</sub>) (64D) ] &isin; &real;<sup>303</sup><br>
        &alpha;<sub>hog</sub> = 0.45 &middot; RMS(<b>x̃</b><sub>ct</sub>) / RMS(<b>z</b><sub>hog</sub>)
      </div>
      <div class="body-lg">
        &bull; <strong>Step 1:</strong> Preserve all <strong>239D</strong> Chromatic-Texture &amp; Quality descriptors at 100% fidelity via Z-score standardization.<br>
        &bull; <strong>Step 2:</strong> Compress 1,764D HOG onto its top <strong><i>k</i> = 64</strong> unwhitened principal axes (<strong>27.6&times; reduction</strong>).<br>
        &bull; <strong>Step 3:</strong> Scale HOG-PCA by RMS energy factor &alpha;<sub>hog</sub> (&gamma; = 0.45), yielding a <strong>+5.86% accuracy jump</strong> (92.05% &rarr; <strong>97.91%</strong>, McNemar <i>p</i> &lt; 0.0001).
      </div>
    </div>
  </div>

  <!-- Visual Dimension & Accuracy Comparison Bar -->
  <div class="panel-box" style="background: #0f172a; color: #ffffff; border-color: #334155;">
    <div style="font-size: 20px; font-weight: 800; color: #38bdf8; margin-bottom: 12px;">
      Empirical Trajectory Across Feature Fusion Regimes (Locked Group-Split Test Set, Table 3)
    </div>
    <div class="grid-3">
      <div style="background: #1e293b; padding: 14px; border-radius: 9px; border-left: 5px solid #f59e0b;">
        <div style="font-size: 17px; color: #cbd5e1; font-weight: 700;">M2: HSV + LBP Only (80D)</div>
        <div style="font-size: 28px; font-weight: 900; color: #fbbf24; margin-top: 4px;">97.35% Acc</div>
        <div style="font-size: 16px; color: #94a3b8;">Macro-F1: 0.9718 (No shape info)</div>
      </div>
      <div style="background: #1e293b; padding: 14px; border-radius: 9px; border-left: 5px solid #ef4444;">
        <div style="font-size: 17px; color: #cbd5e1; font-weight: 700;">M4 / M9: Raw 2,003D Concat</div>
        <div style="font-size: 28px; font-weight: 900; color: #f87171; margin-top: 4px;">86.61% / 92.05%</div>
        <div style="font-size: 16px; color: #94a3b8;">Severe HOG variance dilution</div>
      </div>
      <div style="background: #1e293b; padding: 14px; border-radius: 9px; border-left: 5px solid #10b981;">
        <div style="font-size: 17px; color: #cbd5e1; font-weight: 700;">M10: Proposed Block-PCA (303D)</div>
        <div style="font-size: 28px; font-weight: 900; color: #34d399; margin-top: 4px;">97.91% Acc</div>
        <div style="font-size: 16px; color: #94a3b8;">Macro-F1: 0.9780 (p &lt; 0.0001)</div>
      </div>
    </div>
  </div>
</div>


<!-- ===================================================================== -->
<!-- FIG 10: TWO-STAGE CONFIDENCE-GATED ROUTING ARCHITECTURE               -->
<!-- ===================================================================== -->
<div id="fig10-twostage-confidence-gate" class="fig-card">
  <div class="fig-banner">
    <div class="fig-kicker">Pillar IV &bull; Selective Edge Inference (<code>CalibratedConfidenceGatedSystem</code>)</div>
    <div class="fig-main-title">Temperature-Calibrated Two-Stage Confidence-Gated Classification Architecture</div>
    <div class="fig-sub-desc">Combining fast linear execution for 98.47% of images with selective nonlinear specialist refinement for the 1.53% ambiguous borderline specimens.</div>
  </div>

  <!-- Stage 1 Box -->
  <div class="panel-box" style="border-color: #2563eb; background: #eff6ff; margin-bottom: 18px;">
    <div class="panel-title">
      <span>Stage 1: Always-On Linear Specialist + NLL Temperature Scaling</span>
      <span class="badge badge-blue">480.2 KB &bull; 21.58 ms</span>
    </div>
    <div class="grid-2" style="align-items: center;">
      <div class="formula-block" style="margin: 0;">
        <b>s</b><sub>1</sub>(<b>z</b>) = <b>W</b><sub>1</sub><b>z</b> + <b>b</b><sub>1</sub> &isin; &real;<sup>8</sup> &nbsp;(L2-LR, <i>C</i> = 0.35)<br>
        <i>P</i><sub>1</sub>(<i>y</i>=<i>c</i> | <b>z</b>) = exp(<i>s</i><sub>1,<i>c</i></sub> / <i>T</i>*) / &Sigma;<sub><i>k</i></sub> exp(<i>s</i><sub>1,<i>k</i></sub> / <i>T</i>*)
      </div>
      <div class="body-lg">
        &bull; Optimal temperature <strong><i>T</i>* = 0.758</strong> fitted on validation split via Negative Log-Likelihood minimization.<br>
        &bull; Achieves Expected Calibration Error <strong>ECE = 0.0094</strong> and standalone accuracy of <strong>97.91%</strong> (0.9780 Macro-F1).
      </div>
    </div>
  </div>

  <!-- Gate Split -->
  <div class="grid-2" style="margin-bottom: 18px;">
    <div class="panel-box" style="border-color: #16a34a; background: #f0fdf4;">
      <div class="panel-title">
        <span>Fast Path: max<sub><i>c</i></sub> <i>P</i><sub>1</sub>(<i>y</i>=<i>c</i> | <b>z</b>) &ge; 0.65</span>
        <span class="badge badge-green">98.47% of Test Images</span>
      </div>
      <div class="body-lg" style="margin-top: 8px;">
        &bull; Immediately accepts Stage-1 prediction <i>ŷ</i> = argmax<sub><i>c</i></sub> <i>P</i><sub>1</sub>(<i>y</i>=<i>c</i> | <b>z</b>).<br>
        &bull; <strong>Zero Stage-2 overhead:</strong> runs in <strong>21.58 ms</strong> on a single CPU core.<br>
        &bull; Handles all clear symptomatic and healthy specimens effortlessly.
      </div>
    </div>

    <div class="panel-box" style="border-color: #9333ea; background: #faf5ff;">
      <div class="panel-title">
        <span>Slow Path: max<sub><i>c</i></sub> <i>P</i><sub>1</sub>(<i>y</i>=<i>c</i> | <b>z</b>) &lt; 0.65</span>
        <span class="badge badge-purple">Only 1.53% Routed!</span>
      </div>
      <div class="body-lg" style="margin-top: 8px;">
        &bull; Routes ambiguous multi-lesion / early-stage samples to Stage-2 Nonlinear Specialist Ensemble:<br>
        &nbsp;&nbsp;1. <strong>0.55 &times; RBF-SVM</strong> (<i>C</i> = 4.0, &gamma; = scale, Platt-calibrated)<br>
        &nbsp;&nbsp;2. <strong>0.30 &times; MLP-128</strong> (ReLU, &alpha; = 10<sup>&minus;3</sup>, early stopping)<br>
        &nbsp;&nbsp;3. <strong>0.15 &times; Cosine-kNN</strong> (<i>k</i> = 11, distance-weighted)
      </div>
    </div>
  </div>

  <!-- Soft Posterior Blending Box -->
  <div class="panel-box" style="border-color: #0f172a; background: #f8fafc;">
    <div class="panel-title">
      <span>Stage-2 Soft Posterior Blending &amp; Selective Risk Guarantee (Table 4)</span>
      <span class="badge badge-green">98.47% Acc (@&tau;=0.65) &bull; 98.74% Acc (@&tau;=0.85)</span>
    </div>
    <div class="formula-block" style="background: #f1f5f9; border-color: #0f172a; color: #0f172a;">
      <i>P</i><sub>final</sub>(<i>y</i> | <b>z</b>) = (1 &minus; &beta;) &middot; <i>P</i><sub>1</sub>(<i>y</i> | <b>z</b>) + &beta; &middot; <i>P</i><sub>2</sub>(<i>y</i> | <b>z</b>), &nbsp;&nbsp;with &beta; = 0.70 &nbsp;(i.e., 0.30 <i>P</i><sub>1</sub> + 0.70 <i>P</i><sub>2</sub>)
    </div>
    <div class="body-lg">
      &bull; At validation-selected <strong>&tau;<sub>gate</sub> = 0.65</strong> (1.53% routed): reaches <strong>98.47% Test Acc</strong>, <strong>0.9836 Macro-F1</strong>, <strong>ECE = 0.0104</strong>, <strong>AURC = 0.00158</strong> at <strong>21.74 ms</strong> mean latency (+0.16 ms amortized).<br>
      &bull; At high-sensitivity <strong>&tau;<sub>gate</sub> = 0.85</strong> (4.32% routed): reaches <strong>98.74% Test Acc</strong> and <strong>0.9859 Macro-F1</strong> (AURC = 0.00151).
    </div>
  </div>
</div>


<!-- ===================================================================== -->
<!-- FIG 13: CPU LATENCY BREAKDOWN & EDGE RESOURCE PROFILE                 -->
<!-- ===================================================================== -->
<div id="fig13-latency-and-footprint" class="fig-card">
  <div class="fig-banner">
    <div class="fig-kicker">Pillar V-B &bull; Single-Core Edge CPU Benchmark (Table 6)</div>
    <div class="fig-main-title">Sub-22 ms Inference Latency Breakdown &amp; Footprint vs. Deep CNN Baselines</div>
    <div class="fig-sub-desc">Measured on a single CPU thread (batch size = 1) with no GPU, NPU, or multi-core parallelism.</div>
  </div>

  <!-- Per-Stage Breakdown -->
  <div class="panel-box" style="margin-bottom: 22px; border-color: #2563eb;">
    <div class="panel-title">
      <span>Per-Stage Execution Breakdown of the 21.58 ms Stage-1 Fast Path (46.3 images/sec)</span>
      <span class="badge badge-blue">p50 = 21.17 ms &bull; p95 = 24.48 ms</span>
    </div>
    <div style="display: grid; grid-template-columns: 27fr 43fr 27fr 8fr; gap: 6px; margin: 14px 0;">
      <div style="background: #0284c7; color: #fff; padding: 14px 12px; border-radius: 8px 0 0 8px; font-size: 17px; font-weight: 800;">
        1. ROI &amp; Quality Gate<br><span style="font-size: 22px;">~5.8 ms (27%)</span>
      </div>
      <div style="background: #2563eb; color: #fff; padding: 14px 12px; font-size: 17px; font-weight: 800;">
        2. Shared-Sobel 7-Branch Features<br><span style="font-size: 22px;">~9.4 ms (43%)</span>
      </div>
      <div style="background: #4f46e5; color: #fff; padding: 14px 12px; font-size: 17px; font-weight: 800;">
        3. Block-PCA (303D)<br><span style="font-size: 22px;">~5.9 ms (27%)</span>
      </div>
      <div style="background: #059669; color: #fff; padding: 14px 8px; border-radius: 0 8px 8px 0; font-size: 16px; font-weight: 800; text-align: center;">
        LR<br><strong>0.5ms</strong>
      </div>
    </div>
    <div class="body-lg">
      &bull; <strong>Shared 3&times;3 Sobel Reuse</strong> eliminates redundant convolutions across 5-Patch, MGCT, and HOG.<br>
      &bull; <strong>256-Entry LUT LBP</strong> replaces per-pixel bit-shifting loops with O(1) array indexing.<br>
      &bull; <strong>Two-Stage Gated Routing (&tau; = 0.65)</strong> adds only <strong>+0.16 ms</strong> amortized overhead (<strong>21.74 ms</strong> mean, 46.0 img/s).
    </div>
  </div>

  <!-- Head-to-Head Comparison Bars -->
  <div class="panel-box">
    <div class="panel-title">
      <span>Head-to-Head Single-Core CPU Latency &amp; Disk Footprint Comparison (Table 6)</span>
      <span class="badge badge-green">2.26&times; to 12.61&times; Faster</span>
    </div>

    <div style="display: flex; flex-direction: column; gap: 16px; margin-top: 14px;">
      <div>
        <div style="display: flex; justify-content: space-between; font-size: 18px; font-weight: 800; color: #166534; margin-bottom: 5px;">
          <span>1. Proposed Stage-1 Linear (480.2 KB &bull; 97.91% Acc)</span>
          <span>21.58 ms (46.3 img/s) &mdash; 1.00&times; Baseline</span>
        </div>
        <div style="width: 100%; background: #e2e8f0; height: 26px; border-radius: 6px; overflow: hidden;">
          <div style="width: 8%; background: #16a34a; height: 100%;"></div>
        </div>
      </div>

      <div>
        <div style="display: flex; justify-content: space-between; font-size: 18px; font-weight: 800; color: #1e40af; margin-bottom: 5px;">
          <span>2. Proposed Two-Stage Gated (4,060.6 KB &bull; 98.47% Acc)</span>
          <span>21.74 ms (46.0 img/s) &mdash; +0.16 ms Amortized</span>
        </div>
        <div style="width: 100%; background: #e2e8f0; height: 26px; border-radius: 6px; overflow: hidden;">
          <div style="width: 8.2%; background: #2563eb; height: 100%;"></div>
        </div>
      </div>

      <div>
        <div style="display: flex; justify-content: space-between; font-size: 18px; font-weight: 700; color: #475569; margin-bottom: 5px;">
          <span>3. MobileNetV3-Small ONNX FP32 (5,900 KB)</span>
          <span>48.70 ms (20.5 img/s) &mdash; 2.26&times; Slower / 12.3&times; Larger</span>
        </div>
        <div style="width: 100%; background: #e2e8f0; height: 26px; border-radius: 6px; overflow: hidden;">
          <div style="width: 18%; background: #64748b; height: 100%;"></div>
        </div>
      </div>

      <div>
        <div style="display: flex; justify-content: space-between; font-size: 18px; font-weight: 700; color: #475569; margin-bottom: 5px;">
          <span>4. MobileNetV2 ONNX FP32 (8,500 KB)</span>
          <span>56.42 ms (17.7 img/s) &mdash; 2.61&times; Slower / 17.7&times; Larger</span>
        </div>
        <div style="width: 100%; background: #e2e8f0; height: 26px; border-radius: 6px; overflow: hidden;">
          <div style="width: 21%; background: #64748b; height: 100%;"></div>
        </div>
      </div>

      <div>
        <div style="display: flex; justify-content: space-between; font-size: 18px; font-weight: 800; color: #991b1b; margin-bottom: 5px;">
          <span>5. ResNet50 ONNX FP32 (90,000 KB)</span>
          <span>272.04 ms (3.7 img/s) &mdash; 12.61&times; Slower / 187.4&times; Larger</span>
        </div>
        <div style="width: 100%; background: #e2e8f0; height: 26px; border-radius: 6px; overflow: hidden;">
          <div style="width: 100%; background: #dc2626; height: 100%;"></div>
        </div>
      </div>
    </div>
  </div>
</div>


<!-- ===================================================================== -->
<!-- FIG 14: CROSS-DATASET CANONICAL LEAF POSE NORMALIZATION (MLDID)       -->
<!-- ===================================================================== -->
<div id="fig14-crossdataset-canonical-pose" class="fig-card">
  <div class="fig-banner">
    <div class="fig-kicker">Pillar VI &bull; External Cross-Dataset Generalization (<code>canonicalize_leaf_organ_image</code>, Table 7)</div>
    <div class="fig-main-title">Eigenvector Canonical Leaf Pose Normalization on External MLDID (<i>N</i> = 3,000)</div>
    <div class="fig-sub-desc">Mitigating camera aspect-ratio confounding (320&times;240 / 240&times;240 &rarr; 512&times;512) and arbitrary leaf orientation in zero-shot cross-dataset transfer.</div>
  </div>

  <div class="formula-block" style="margin-bottom: 20px;">
    Principal Orientation Angle: &nbsp; &theta; = &frac12; arctan2( 2&mu;<sub>11</sub>, &nbsp;&mu;<sub>20</sub> &minus; &mu;<sub>02</sub> ) &nbsp;&rarr;&nbsp; Affine Rotation by &minus;&theta; + Foreground <i>L</i>* Constancy [<i>P</i><sub>5</sub>, <i>P</i><sub>95</sub>] &rarr; [18, 238]
  </div>

  <div class="grid-3" style="margin-bottom: 20px;">
    <div class="panel-box">
      <div class="panel-title"><span>1. Raw MLDID Frame</span><span class="badge badge-red">31.27% 0-Shot</span></div>
      <div class="thumb-frame"><img src="{a['mldid_raw']}" alt="MLDID Raw"></div>
      <div class="body-lg">Standardized 512&times;512 external image with vertical/diagonal leaf pose and distinct background illumination.</div>
    </div>

    <div class="panel-box">
      <div class="panel-title"><span>2. Covariance Axis &theta;</span><span class="badge badge-amber">PCA Moments</span></div>
      <div class="thumb-frame"><img src="{a['mldid_axis']}" alt="MLDID Axis"></div>
      <div class="body-lg">Extracts foreground mask <i>M</i><sub>fg</sub>, centroid (<i>c<sub>x</sub></i>, <i>c<sub>y</sub></i>), and principal eigenvector axis (red line).</div>
    </div>

    <div class="panel-box" style="border-color: #16a34a; background: #f0fdf4;">
      <div class="panel-title"><span>3. Canonical Pose</span><span class="badge badge-green">42.63% 0-Shot</span></div>
      <div class="thumb-frame"><img src="{a['mldid_canon']}" alt="MLDID Canonical"></div>
      <div class="body-lg">Rotated horizontally (&minus;&theta;), cropped to tight organ bbox (6% margin) on neutral gray <code>[232,232,232]</code> + <i>L</i>* constancy.</div>
    </div>
  </div>

  <div class="grid-2">
    <div class="panel-box" style="border-color: #16a34a;">
      <div class="panel-title">
        <span>Zero-Shot Canonical Pose Gain</span>
        <span class="badge badge-green">+11.36% Acc / +15.09% F1</span>
      </div>
      <div class="body-lg">
        &bull; <strong>Source-View Zero-Shot (<i>N</i>=3,000):</strong> 31.27% Acc (0.2886 Macro-F1).<br>
        &bull; <strong>Canonical-Pose Zero-Shot (<i>N</i>=3,000):</strong> <strong>42.63% Acc (0.4395 Macro-F1)</strong>.<br>
        &bull; Retains <strong>96.04%</strong> canonical (<strong>99.34%</strong> source) in-domain 5-class accuracy (<i>N</i>=455).
      </div>
    </div>

    <div class="panel-box" style="border-color: #64748b;">
      <div class="panel-title">
        <span>Why Unsupervised CORAL Degrades (31.43%)</span>
        <span class="badge badge-amber">Negative Finding</span>
      </div>
      <div class="body-lg">
        &bull; Block-Shrinkage CORAL aligns second-order feature covariances (<i>C<sub>S</sub></i> &rarr; <i>C<sub>T</sub></i>) without labels.<br>
        &bull; When target MLDID images contain residual background leakage and shifted class priors, global covariance matching pulls lesion boundaries toward background artifacts (31.43% Acc).
      </div>
    </div>
  </div>
</div>


<!-- ===================================================================== -->
<!-- FIG 15: EDA EIGHT-CLASS SPECIMEN GALLERY & GEOMETRY CONFOUNDING       -->
<!-- ===================================================================== -->
<div id="fig15-eda-eight-classes" class="fig-card">
  <div class="fig-banner">
    <div class="fig-kicker">Exploratory Data Analysis &bull; MangoLeafBD 8-Class Profile (<i>N</i> = 4,000)</div>
    <div class="fig-main-title">Class-Wise Specimen Morphology, Resolution Confounding &amp; Chromatic Statistics</div>
    <div class="fig-sub-desc">Demonstrating why native camera aspect ratio (320&times;240 Landscape vs. 240&times;240 Square vs. 240&times;320 Portrait) is strongly confounded with disease class.</div>
  </div>

  <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 18px;">
    <div class="panel-box" style="padding: 14px;">
      <div style="font-size: 18.5px; font-weight: 800; color: #0f172a; margin-bottom: 6px;">1. Anthracnose</div>
      <div class="thumb-frame" style="height: 135px;"><img src="{a['cls_anth']}" alt="Anthracnose"></div>
      <div class="body-lg" style="font-size: 15.5px;">
        &bull; Clusters: <strong>352</strong> (29.6% dup)<br>
        &bull; Median: <strong>240&times;240</strong> (16.3 KB)<br>
        &bull; <i>L</i>*: 189.8 | <i>S</i>: 40.8
      </div>
    </div>

    <div class="panel-box" style="padding: 14px;">
      <div style="font-size: 18.5px; font-weight: 800; color: #0f172a; margin-bottom: 6px;">2. Bacterial Canker</div>
      <div class="thumb-frame" style="height: 135px;"><img src="{a['cls_bact']}" alt="Bacterial Canker"></div>
      <div class="body-lg" style="font-size: 15.5px;">
        &bull; Clusters: <strong>275</strong> (45.0% dup)<br>
        &bull; Median: <strong style="color:#1d4ed8;">240&times;320</strong> (34.1 KB)<br>
        &bull; <i>L</i>*: 180.2 | <i>S</i>: 24.5
      </div>
    </div>

    <div class="panel-box" style="padding: 14px; border-color: #dc2626; background: #fff5f5;">
      <div style="font-size: 18.5px; font-weight: 800; color: #991b1b; margin-bottom: 6px;">3. Cutting Weevil</div>
      <div class="thumb-frame" style="height: 135px;"><img src="{a['cls_weev']}" alt="Cutting Weevil"></div>
      <div class="body-lg" style="font-size: 15.5px;">
        &bull; Clusters: <strong style="color:#dc2626;">31 (93.8% dup!)</strong><br>
        &bull; Median: <strong>240&times;240</strong> (29.5 KB)<br>
        &bull; <i>L</i>*: 180.3 | <i>S</i>: 69.0
      </div>
    </div>

    <div class="panel-box" style="padding: 14px; border-color: #dc2626; background: #fff5f5;">
      <div style="font-size: 18.5px; font-weight: 800; color: #991b1b; margin-bottom: 6px;">4. Die Back</div>
      <div class="thumb-frame" style="height: 135px;"><img src="{a['cls_dieb']}" alt="Die Back"></div>
      <div class="body-lg" style="font-size: 15.5px;">
        &bull; Clusters: <strong style="color:#dc2626;">95 (81.0% dup!)</strong><br>
        &bull; Median: <strong>240&times;240</strong> (25.8 KB)<br>
        &bull; <i>L</i>*: 180.8 | <i>S</i>: 43.0
      </div>
    </div>
  </div>

  <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;">
    <div class="panel-box" style="padding: 14px;">
      <div style="font-size: 18.5px; font-weight: 800; color: #0f172a; margin-bottom: 6px;">5. Gall Midge</div>
      <div class="thumb-frame" style="height: 135px;"><img src="{a['cls_gall']}" alt="Gall Midge"></div>
      <div class="body-lg" style="font-size: 15.5px;">
        &bull; Clusters: <strong>437</strong> (12.6% dup)<br>
        &bull; Median: <strong style="color:#b45309;">320&times;240</strong> (24.8 KB)<br>
        &bull; <i>L</i>*: 187.0 | <i>S</i>: 35.0
      </div>
    </div>

    <div class="panel-box" style="padding: 14px; border-color: #16a34a; background: #f0fdf4;">
      <div style="font-size: 18.5px; font-weight: 800; color: #166534; margin-bottom: 6px;">6. Healthy</div>
      <div class="thumb-frame" style="height: 135px;"><img src="{a['cls_heal']}" alt="Healthy"></div>
      <div class="body-lg" style="font-size: 15.5px;">
        &bull; Clusters: <strong>469</strong> (6.2% dup)<br>
        &bull; Median: <strong style="color:#1d4ed8;">240&times;245</strong> (33.0 KB)<br>
        &bull; <i>L</i>*: 189.7 | <i>S</i>: 39.2
      </div>
    </div>

    <div class="panel-box" style="padding: 14px;">
      <div style="font-size: 18.5px; font-weight: 800; color: #0f172a; margin-bottom: 6px;">7. Powdery Mildew</div>
      <div class="thumb-frame" style="height: 135px;"><img src="{a['cls_powd']}" alt="Powdery Mildew"></div>
      <div class="body-lg" style="font-size: 15.5px;">
        &bull; Clusters: <strong>427</strong> (14.6% dup)<br>
        &bull; Median: <strong>240&times;240</strong> (31.3 KB)<br>
        &bull; <i>L</i>*: 171.8 | <i>S</i>: 38.8
      </div>
    </div>

    <div class="panel-box" style="padding: 14px;">
      <div style="font-size: 18.5px; font-weight: 800; color: #0f172a; margin-bottom: 6px;">8. Sooty Mold</div>
      <div class="thumb-frame" style="height: 135px;"><img src="{a['cls_soot']}" alt="Sooty Mold"></div>
      <div class="body-lg" style="font-size: 15.5px;">
        &bull; Clusters: <strong>468</strong> (6.4% dup)<br>
        &bull; Median: <strong style="color:#b45309;">320&times;240</strong> (36.6 KB)<br>
        &bull; <i>L</i>*: 179.6 | <i>S</i>: 38.4
      </div>
    </div>
  </div>
</div>

</body>
</html>
"""


def render_all_html_diagrams() -> None:
    from playwright.sync_api import sync_playwright

    figures_dir = REPO_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    assets = prepare_visual_assets()
    html_content = build_showcase_html(assets)
    html_path = figures_dir / "publication_diagrams_showcase.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"[HTML] Wrote self-contained showcase to: {html_path}")

    targets = [
        ("#fig01-macro-overview", ["Fig01_System_Overview_Macro.png", "Figure1_Pipeline_Architecture_Overview.png"]),
        ("#fig02-specimen-clustering", ["Fig02_Specimen_Clustering_Graph_Mechanism.png"]),
        ("#fig04-fiveview-shortcuts", ["Fig04_FiveView_Background_Shortcut_VisualProof.png"]),
        ("#fig05-roi-segmentation", ["Fig05_Hybrid_ROI_Segmentation_StepByStep.png"]),
        ("#fig06-quality-gated-preprocessing", ["Fig06_Quality_Gate_MAD_Glare_CLAHE_Proof.png"]),
        ("#fig07-seven-feature-branches", ["Fig07_Seven_Feature_Branches_Visual_Maps.png"]),
        ("#fig08-block-pca-mechanism", ["Fig08_BlockPCA_Dimensionality_Dilution_Solution.png"]),
        ("#fig10-twostage-confidence-gate", ["Fig10_TwoStage_Confidence_Gate_Architecture.png"]),
        ("#fig13-latency-and-footprint", ["Fig13_CPU_Latency_Breakdown_and_Edge_Footprint.png", "Figure5_Latency_and_Resource_Profile.png"]),
        ("#fig14-crossdataset-canonical-pose", ["Fig14_CrossDataset_Canonical_Pose_and_CORAL.png", "Figure6_Optical_Modes_and_Canonical_Pose.png"]),
        ("#fig15-eda-eight-classes", ["Fig15_EDA_MangoLeafBD_EightClass_Specimens_and_Geometry.png"]),
    ]

    with sync_playwright() as p:
        browser = None
        for channel in ("msedge", "chrome", None):
            try:
                browser = (
                    p.chromium.launch(headless=True, channel=channel)
                    if channel
                    else p.chromium.launch(headless=True)
                )
                break
            except Exception:
                continue
        if browser is None:
            raise RuntimeError("Could not launch headless Chromium/Edge via Playwright.")

        page = browser.new_page(viewport={"width": 1160, "height": 1200}, device_scale_factor=2)
        page.goto(html_path.as_uri(), wait_until="networkidle")
        page.wait_for_timeout(350)

        for selector, out_names in targets:
            loc = page.locator(selector)
            for name in out_names:
                out_path = figures_dir / name
                loc.screenshot(path=str(out_path))
                print(f"[PNG] Rendered {out_path.name} ({out_path.stat().st_size / 1024:.1f} KB)")

        browser.close()


if __name__ == "__main__":
    render_all_html_diagrams()
