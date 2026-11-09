#!/usr/bin/env python3
"""Render publication-grade architecture, latency/resource, and optical/pose mode
diagrams by generating a self-contained HTML/CSS/SVG showcase and capturing
high-DPI PNG screenshots via Playwright headless Chromium.

Matches the visual design language of the research group's publication figures:
- Figure1_Pipeline_Architecture_Overview.png
- Figure5_Latency_and_Resource_Profile.png
- Figure6_Optical_Modes_and_Canonical_Pose.png

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

from mango_edge_cv.preprocessing import (
    apply_adaptive_quality_preprocessing,
    canonicalize_leaf_organ_image,
    segment_plant_organ_roi,
)


def _rgb_to_data_uri(img_rgb: np.ndarray, quality: int = 92) -> str:
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode RGB image to JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _make_fallback_leaf(portrait: bool = False, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = (293, 240) if portrait else (240, 320)
    img = np.full((h, w, 3), 232, dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(w * 0.24), int(h * 0.38)) if portrait else (int(w * 0.38), int(h * 0.24))
    angle = 85 if portrait else 12
    cv2.ellipse(img, center, axes, angle, 0, 360, (52, 142, 46), -1)
    for _ in range(10):
        cx = int(rng.integers(int(w * 0.35), int(w * 0.65)))
        cy = int(rng.integers(int(h * 0.35), int(h * 0.65)))
        cv2.circle(img, (cx, cy), int(rng.integers(4, 10)), (55, 38, 24), -1)
    return img


def _read_rgb_unicode_safe(path: Path) -> np.ndarray | None:
    try:
        arr = np.fromfile(str(path), dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return None


def prepare_visual_thumbnails() -> Dict[str, str]:
    """Load real mango leaf images from workspace if available (or fallback) and create visual pairs."""
    workspace_mango = REPO_ROOT.parent / "MangoLeafBD Dataset"
    workspace_mldid = REPO_ROOT.parent / "Resize mango leaf disease"

    anth_img = None
    if (workspace_mango / "Anthracnose").exists():
        files = sorted((workspace_mango / "Anthracnose").glob("*.jpg"))
        if files:
            anth_img = _read_rgb_unicode_safe(files[0])
    if anth_img is None:
        anth_img = _make_fallback_leaf(portrait=False, seed=101)

    sib_a = cv2.resize(anth_img, (256, 256), interpolation=cv2.INTER_AREA)
    proc_rgb = apply_adaptive_quality_preprocessing(sib_a)
    fg_rgb, _, mask_u8 = segment_plant_organ_roi(proc_rgb)
    masked_fg = proc_rgb.copy()
    masked_fg[mask_u8 == 0] = [241, 245, 249]
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_fg = proc_rgb.copy()
    cv2.drawContours(overlay_fg, contours, -1, (16, 185, 129), 3)

    sib_b = cv2.rotate(sib_a, cv2.ROTATE_90_CLOCKWISE)

    # Perturbed Low-Contrast + Sensor Grain + Specular Glare vs Restored Quality-Gated
    rng = np.random.default_rng(20261022)
    noise = rng.normal(0.0, 9.0, size=proc_rgb.shape).astype(np.float32)
    dark_glare = (proc_rgb.astype(np.float32) * 0.56 + 16.0 + noise).clip(0, 255).astype(np.uint8)
    glare_mask = np.zeros(proc_rgb.shape[:2], dtype=np.uint8)
    for gx, gy, gr in [(122, 108, 6), (134, 138, 5), (116, 152, 5)]:
        cv2.circle(dark_glare, (gx, gy), gr, (250, 250, 245), -1)
        cv2.circle(glare_mask, (gx, gy), gr + 2, 255, -1)
    inpainted_glare = cv2.inpaint(dark_glare, glare_mask, 5, cv2.INPAINT_TELEA)
    restored_rgb = apply_adaptive_quality_preprocessing(inpainted_glare, foreground_mask=mask_u8)
    restored_vis = restored_rgb.copy()
    cv2.drawContours(restored_vis, contours, -1, (37, 99, 235), 2)

    # External MLDID Portrait Leaf vs Canonical Pose Horizontal Alignment
    mldid_img = None
    if workspace_mldid.exists():
        mldid_files = sorted(workspace_mldid.rglob("*.jpg"))
        if mldid_files:
            mldid_img = _read_rgb_unicode_safe(mldid_files[0])
    if mldid_img is None:
        mldid_img = _make_fallback_leaf(portrait=True, seed=202)

    mldid_resized = cv2.resize(mldid_img, (256, 256), interpolation=cv2.INTER_AREA)
    canon_rgb = canonicalize_leaf_organ_image(mldid_resized, output_size=256)
    _, _, canon_mask = segment_plant_organ_roi(canon_rgb)
    canon_contours, _ = cv2.findContours(canon_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canon_vis = canon_rgb.copy()
    cv2.drawContours(canon_vis, canon_contours, -1, (16, 185, 129), 2)
    cv2.line(canon_vis, (24, 128), (232, 128), (239, 68, 68), 2)

    return {
        "input_leaf": _rgb_to_data_uri(sib_a),
        "segmented_overlay": _rgb_to_data_uri(overlay_fg),
        "masked_fg": _rgb_to_data_uri(masked_fg),
        "sibling_a": _rgb_to_data_uri(sib_a),
        "sibling_b": _rgb_to_data_uri(sib_b),
        "perturbed_glare": _rgb_to_data_uri(dark_glare),
        "restored_clahe": _rgb_to_data_uri(restored_vis),
        "mldid_portrait": _rgb_to_data_uri(mldid_resized),
        "mldid_canonical": _rgb_to_data_uri(canon_vis),
    }


def build_showcase_html(thumbs: Dict[str, str]) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Mango-Disease-Audit-EdgeAI Publication Figures</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    background: #e2e8f0;
    color: #0f172a;
    padding: 32px;
    display: flex;
    flex-direction: column;
    gap: 48px;
    align-items: center;
  }}
  .figure-canvas {{
    background: #ffffff;
    border-radius: 10px;
    padding: 26px 30px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.08);
    border: 1px solid #cbd5e1;
  }}

  /* ================= FIGURE 1: PIPELINE ARCHITECTURE ================= */
  #fig1-pipeline {{
    width: 1560px;
  }}
  .fig-header {{
    text-align: center;
    margin-bottom: 18px;
  }}
  .fig-title {{
    font-size: 19px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.2px;
  }}
  .fig-subtitle {{
    font-size: 12.5px;
    color: #475569;
    margin-top: 4px;
    font-weight: 500;
  }}
  .pipeline-row {{
    display: flex;
    align-items: stretch;
    justify-content: space-between;
    gap: 12px;
  }}
  .stage-col {{
    display: flex;
    flex-direction: column;
    justify-content: center;
    background: #f8fafc;
    border: 1.5px solid #cbd5e1;
    border-radius: 10px;
    padding: 12px;
    position: relative;
  }}
  .stage-badge {{
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    padding: 4px 8px;
    border-radius: 5px;
    text-align: center;
    margin-bottom: 8px;
  }}
  .img-box {{
    width: 126px;
    height: 116px;
    border-radius: 7px;
    overflow: hidden;
    border: 1.5px solid #94a3b8;
    margin: 0 auto 6px auto;
  }}
  .img-box img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
  }}
  .caption-sm {{
    font-size: 11px;
    text-align: center;
    color: #334155;
    line-height: 1.35;
  }}
  .caption-sm strong {{
    color: #0f172a;
  }}
  .audit-pill {{
    margin-top: 8px;
    background: #eff6ff;
    border: 1px solid #93c5fd;
    border-radius: 6px;
    padding: 6px 8px;
    font-size: 10.5px;
    color: #1e3a8a;
    line-height: 1.38;
  }}
  .arrow-connector {{
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    font-weight: 900;
    color: #475569;
    padding: 0 2px;
  }}
  .branches-container {{
    width: 470px;
    background: #f8fafc;
    border: 1.5px solid #94a3b8;
    border-radius: 10px;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .branch-card {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #ffffff;
    border-left: 5px solid #2563eb;
    border-top: 1px solid #e2e8f0;
    border-right: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 6px 10px;
  }}
  .branch-info {{
    display: flex;
    flex-direction: column;
  }}
  .branch-name {{
    font-size: 12px;
    font-weight: 700;
    color: #0f172a;
  }}
  .branch-desc {{
    font-size: 10.5px;
    color: #475569;
  }}
  .branch-dim {{
    font-size: 11.5px;
    font-weight: 800;
    padding: 3px 8px;
    border-radius: 5px;
    background: #f1f5f9;
    color: #0f172a;
    white-space: nowrap;
  }}
  .pca-box {{
    width: 195px;
    background: #fefce8;
    border: 1.5px solid #facc15;
  }}
  .pca-bars {{
    display: flex;
    flex-direction: column;
    gap: 5px;
    margin: 8px 0;
  }}
  .pca-bar-row {{
    display: flex;
    align-items: center;
    font-size: 10px;
    gap: 5px;
  }}
  .pca-bar-label {{ width: 52px; font-weight: 700; color: #334155; }}
  .pca-bar-track {{ flex: 1; height: 9px; background: #e2e8f0; border-radius: 4px; overflow: hidden; }}
  .pca-bar-fill {{ height: 100%; background: #ca8a04; }}
  .pca-bar-val {{ width: 38px; text-align: right; font-weight: 700; color: #713f12; }}

  .gate-col {{
    width: 285px;
    background: #f0fdf4;
    border: 1.5px solid #86efac;
    gap: 8px;
  }}
  .gate-card {{
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 7px;
    padding: 8px 10px;
    font-size: 11px;
    line-height: 1.38;
  }}
  .gate-diamond {{
    background: #fef3c7;
    border: 1.5px dashed #d97706;
    border-radius: 7px;
    padding: 7px 9px;
    text-align: center;
    font-size: 11px;
    font-weight: 700;
    color: #92400e;
  }}
  .output-col {{
    width: 190px;
    background: #eff6ff;
    border: 1.5px solid #93c5fd;
  }}
  .class-list {{
    display: flex;
    flex-direction: column;
    gap: 4px;
    margin: 6px 0;
  }}
  .class-pill {{
    font-size: 10.5px;
    padding: 3px 7px;
    border-radius: 4px;
    background: #ffffff;
    border: 1px solid #cbd5e1;
    color: #1e293b;
    display: flex;
    justify-content: space-between;
  }}
  .class-pill.active {{
    background: #16a34a;
    color: #ffffff;
    border-color: #15803d;
    font-weight: 700;
  }}

  /* ================= FIGURE 5: LATENCY & RESOURCE PROFILE ================= */
  #fig5-latency {{
    width: 1440px;
  }}
  .latency-grid {{
    display: grid;
    grid-template-columns: 1.55fr 1fr;
    gap: 28px;
    align-items: stretch;
  }}
  .panel-box {{
    background: #ffffff;
    border: 1.5px solid #cbd5e1;
    border-radius: 10px;
    padding: 20px 24px;
    position: relative;
  }}
  .panel-title {{
    font-size: 15.5px;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 4px;
  }}
  .panel-sub {{
    font-size: 12px;
    color: #475569;
    margin-bottom: 18px;
  }}
  .hbar-group {{
    display: flex;
    flex-direction: column;
    gap: 13px;
  }}
  .hbar-row {{
    display: grid;
    grid-template-columns: 245px 1fr 86px;
    align-items: center;
    gap: 12px;
  }}
  .hbar-label {{
    font-size: 12px;
    font-weight: 600;
    color: #1e293b;
    text-align: right;
    line-height: 1.25;
  }}
  .hbar-track {{
    height: 26px;
    background: #f1f5f9;
    border-radius: 5px;
    position: relative;
    overflow: hidden;
    border: 1px solid #e2e8f0;
  }}
  .hbar-fill {{
    height: 100%;
    border-radius: 4px;
  }}
  .hbar-val {{
    font-size: 12.5px;
    font-weight: 800;
    color: #0f172a;
  }}
  .callout-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 18px;
  }}
  .callout-note {{
    background: #fef9c3;
    border: 1.5px solid #eab308;
    border-radius: 7px;
    padding: 9px 12px;
    font-size: 11.5px;
    color: #713f12;
    line-height: 1.38;
  }}
  .resource-stack {{
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 14px;
  }}
  .res-card {{
    display: flex;
    align-items: center;
    gap: 16px;
    border-radius: 10px;
    padding: 16px 18px;
    border: 1.5px solid;
  }}
  .res-icon {{
    width: 54px;
    height: 54px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    font-weight: 900;
    flex-shrink: 0;
  }}
  .res-body h4 {{
    font-size: 14.5px;
    font-weight: 800;
    margin-bottom: 4px;
  }}
  .res-body p {{
    font-size: 12px;
    line-height: 1.42;
    color: #334155;
  }}

  /* ================= FIGURE 6: OPTICAL MODES & CANONICAL POSE ================= */
  #fig6-optical {{
    width: 1440px;
  }}
  .modes-table {{
    display: flex;
    flex-direction: column;
    gap: 16px;
  }}
  .mode-header {{
    display: grid;
    grid-template-columns: 220px 210px 210px 1fr;
    gap: 16px;
    padding: 8px 14px;
    background: #0f172a;
    color: #ffffff;
    border-radius: 7px;
    font-size: 12.5px;
    font-weight: 700;
  }}
  .mode-row {{
    display: grid;
    grid-template-columns: 220px 210px 210px 1fr;
    gap: 16px;
    align-items: center;
    background: #f8fafc;
    border: 1.5px solid #cbd5e1;
    border-radius: 10px;
    padding: 14px;
  }}
  .mode-tag {{
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}
  .mode-tag .badge {{
    display: inline-block;
    font-size: 11px;
    font-weight: 800;
    padding: 4px 8px;
    border-radius: 5px;
    width: fit-content;
  }}
  .mode-tag h4 {{
    font-size: 14px;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.3;
  }}
  .mode-tag p {{
    font-size: 11.5px;
    color: #475569;
    line-height: 1.35;
  }}
  .mode-img-wrap {{
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 5px;
  }}
  .mode-img-wrap img {{
    width: 185px;
    height: 142px;
    object-fit: cover;
    border-radius: 7px;
    border: 2px solid #64748b;
  }}
  .mode-img-wrap span {{
    font-size: 11px;
    font-weight: 700;
    color: #334155;
  }}
  .mode-analysis {{
    background: #ffffff;
    border-left: 5px solid #2563eb;
    border-radius: 7px;
    padding: 12px 16px;
    border-top: 1px solid #e2e8f0;
    border-right: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
    font-size: 12.5px;
    line-height: 1.5;
    color: #1e293b;
  }}
  .mode-analysis strong {{
    color: #0f172a;
  }}
  .metric-chip {{
    display: inline-block;
    background: #dcfce7;
    color: #166534;
    border: 1px solid #86efac;
    font-weight: 800;
    font-size: 11.5px;
    padding: 2px 8px;
    border-radius: 4px;
    margin-top: 6px;
  }}
</style>
</head>
<body>

<!-- =====================================================================
     FIGURE 1: PIPELINE ARCHITECTURE OVERVIEW
     ===================================================================== -->
<div class="figure-canvas" id="fig1-pipeline">
  <div class="fig-header">
    <div class="fig-title">Figure 1. Overview of the Proposed Audit-First, Quality-Gated 7-Branch Feature Fusion &amp; Confidence-Gated Edge AI Pipeline</div>
    <div class="fig-subtitle">End-to-end single-core CPU latency: 21.58 ms/img (Stage-1, 480.2 KB, 97.91% Acc) &bull; 21.74 ms/img (Two-Stage Gated, 98.47% Acc, 0.9836 Macro-F1 on Group-Disjoint Split)</div>
  </div>

  <div class="pipeline-row">
    <!-- Stage 1: Input & Audit -->
    <div class="stage-col" style="width: 175px;">
      <div class="stage-badge" style="background:#dbeafe; color:#1e40af;">1. Input &amp; Audit</div>
      <div class="img-box"><img src="{thumbs['input_leaf']}" alt="Input Leaf"></div>
      <div class="caption-sm"><strong>Raw RGB Image</strong><br>256 &times; 256 Standardized</div>
      <div class="audit-pill">
        <strong>Specimen Graph Audit:</strong><br>
        pHash (<i>d</i> &le; 6) | HSV (cos &ge; 0.992) &rarr; <strong>2,554 Unique Clusters</strong> (0% Split Leakage)
      </div>
    </div>

    <div class="arrow-connector">&rarr;</div>

    <!-- Stage 2: Quality Gate & Hybrid Segmentation -->
    <div class="stage-col" style="width: 215px;">
      <div class="stage-badge" style="background:#dcfce7; color:#166534;">2. Quality Gate &amp; ROI</div>
      <div style="display:flex; gap:6px; justify-content:center; margin-bottom:6px;">
        <div class="img-box" style="width:94px; height:94px; margin:0;"><img src="{thumbs['segmented_overlay']}" alt="Seg"></div>
        <div class="img-box" style="width:94px; height:94px; margin:0;"><img src="{thumbs['masked_fg']}" alt="Mask"></div>
      </div>
      <div class="caption-sm">
        <strong>HSV-Saturation + LAB-Chroma Otsu ROI</strong><br>
        + Largest Connected Leaf Organ Mask
      </div>
      <div class="audit-pill" style="background:#f0fdf4; border-color:#86efac; color:#14532d;">
        <strong>Selective Quality Gate:</strong><br>
        &bull; Donoho-Johnstone &sigma;&#770;<sub>MAD</sub> &ge; 5.5<br>
        &bull; Foreground CLAHE (<i>P</i><sub>95</sub> &minus; <i>P</i><sub>5</sub> &lt; 132)<br>
        &bull; Glare Attenuation (<i>V</i> &gt; 238, <i>S</i> &lt; 38)<br>
        &bull; Canonical PCA Major-Axis Pose
      </div>
    </div>

    <div class="arrow-connector">&rarr;</div>

    <!-- Stage 3: 7-Branch Feature Extraction -->
    <div class="branches-container">
      <div class="stage-badge" style="background:#e0e7ff; color:#3730a3; margin-bottom:2px;">
        3. Parallel 7-Branch Handcrafted Feature Extraction (2,003D Total)
      </div>
      <div class="branch-card" style="border-left-color:#2563eb;">
        <div class="branch-info">
          <span class="branch-name">Branch 1: HSV Chromatic Histograms &amp; Moments (f_hsv)</span>
          <span class="branch-desc">16+16+16 Bins + MeanH, MeanS, MeanV &amp; Composite Intensity</span>
        </div>
        <span class="branch-dim">52 D</span>
      </div>
      <div class="branch-card" style="border-left-color:#0891b2;">
        <div class="branch-info">
          <span class="branch-name">Branch 2: CIE-LAB Perceptual Color &amp; Skewness (f_lab)</span>
          <span class="branch-desc"><i>L*, a*, b*</i> Histograms (16+16+16) + Mean, Std &amp; Standardized Skewness</span>
        </div>
        <span class="branch-dim">57 D</span>
      </div>
      <div class="branch-card" style="border-left-color:#059669;">
        <div class="branch-info">
          <span class="branch-name">Branch 3: Lookup-Table Multi-Scale Uniform LBP (f_lbp)</span>
          <span class="branch-desc">256-Entry LUT LBP<sub>8,1</sub><sup>riu2</sup> (<i>R</i>=1, 2) + Local Variance, Entropy &amp; Sobel Ratio</span>
        </div>
        <span class="branch-dim">28 D</span>
      </div>
      <div class="branch-card" style="border-left-color:#d97706;">
        <div class="branch-info">
          <span class="branch-name">Branch 4: 5-Patch Spatial Moment Descriptors (f_patch5)</span>
          <span class="branch-desc">Center + 4 Quadrant Patches &times; 9 LAB &amp; Sobel Gradient Statistics</span>
        </div>
        <span class="branch-dim">45 D</span>
      </div>
      <div class="branch-card" style="border-left-color:#dc2626;">
        <div class="branch-info">
          <span class="branch-name">Branch 5: Optical Quality &amp; Glare Metadata Vector (f_q)</span>
          <span class="branch-desc">Mean <i>V</i>, Contrast <i>L*</i>, Glare Ratio, Sharpness, Border Contrast, MAD &sigma;&#770;</span>
        </div>
        <span class="branch-dim">7 D</span>
      </div>
      <div class="branch-card" style="border-left-color:#4f46e5;">
        <div class="branch-info">
          <span class="branch-name">Branch 6: Multi-Grid Chromatic-Texture Descriptor (f_mgct)</span>
          <span class="branch-desc">5 Spatial Regions &times; 10 Saturation, <i>a*, b*</i>, Chroma-Gradient Correlations</span>
        </div>
        <span class="branch-dim">50 D</span>
      </div>
      <div class="branch-card" style="border-left-color:#7c3aed;">
        <div class="branch-info">
          <span class="branch-name">Branch 7: Shared-Sobel Vectorized HOG Grid (f_hog)</span>
          <span class="branch-desc">128 &times; 128 Grid &bull; 8&times;8 Cells, 7&times;7 Overlapping 2&times;2 Blocks, 9 Bins</span>
        </div>
        <span class="branch-dim">1,764 D</span>
      </div>
    </div>

    <div class="arrow-connector">&rarr;</div>

    <!-- Stage 4: Block-PCA -->
    <div class="stage-col pca-box">
      <div class="stage-badge" style="background:#fef08a; color:#854d0e;">4. Energy Block-PCA</div>
      <div class="caption-sm">
        <strong>Chromatic-Texture + HOG-PCA</strong><br>
        (239 D + 64 D = 303 D)
      </div>
      <div class="pca-bars">
        <div class="pca-bar-row"><span class="pca-bar-label">f_hsv</span><div class="pca-bar-track"><div class="pca-bar-fill" style="width:82%"></div></div><span class="pca-bar-val">52&rarr;52</span></div>
        <div class="pca-bar-row"><span class="pca-bar-label">f_lab</span><div class="pca-bar-track"><div class="pca-bar-fill" style="width:90%"></div></div><span class="pca-bar-val">57&rarr;57</span></div>
        <div class="pca-bar-row"><span class="pca-bar-label">f_lbp</span><div class="pca-bar-track"><div class="pca-bar-fill" style="width:45%"></div></div><span class="pca-bar-val">28&rarr;28</span></div>
        <div class="pca-bar-row"><span class="pca-bar-label">f_patch5</span><div class="pca-bar-track"><div class="pca-bar-fill" style="width:70%"></div></div><span class="pca-bar-val">45&rarr;45</span></div>
        <div class="pca-bar-row"><span class="pca-bar-label">f_q</span><div class="pca-bar-track"><div class="pca-bar-fill" style="width:18%"></div></div><span class="pca-bar-val">7&rarr;7</span></div>
        <div class="pca-bar-row"><span class="pca-bar-label">f_mgct</span><div class="pca-bar-track"><div class="pca-bar-fill" style="width:78%"></div></div><span class="pca-bar-val">50&rarr;50</span></div>
        <div class="pca-bar-row"><span class="pca-bar-label">f_hog</span><div class="pca-bar-track"><div class="pca-bar-fill" style="width:100%; background:#7c3aed;"></div></div><span class="pca-bar-val">1764&rarr;64</span></div>
      </div>
      <div class="audit-pill" style="background:#fffbeb; border-color:#f59e0b; color:#92400e; text-align:center;">
        <strong>2,003 D &rarr; 303 D (+5.86% Acc)</strong><br>
        Eliminates 1,764D HOG dilution (M9 92.05% &rarr; M10 97.91%)
      </div>
    </div>

    <div class="arrow-connector">&rarr;</div>

    <!-- Stage 5: Two-Stage Confidence Gate -->
    <div class="stage-col gate-col">
      <div class="stage-badge" style="background:#bbf7d0; color:#14532d;">5. Confidence-Gated Routing</div>
      <div class="gate-card">
        <strong>Stage-1 Fast Linear Classifier (M10):</strong><br>
        L2 Multinomial Logistic Regression + Temperature Scaling (<i>T</i> = 0.758)<br>
        <span style="color:#15803d; font-weight:700;">&bull; Footprint: 480.2 KB &bull; Acc: 97.91%</span>
      </div>
      <div class="gate-diamond">
        Confidence Gate: max<sub><i>c</i></sub> <i>P</i><sub>1</sub>(<i>y=c</i> | <b>z</b>) &ge; &tau;<sub>gate</sub> ?<br>
        <span style="font-size:10.5px;">Val-Selected Threshold &tau;<sub>gate</sub> = 0.65</span>
      </div>
      <div class="gate-card" style="border-left:4px solid #16a34a;">
        <strong>Fast Exit (98.47% of Samples):</strong><br>
        Emit Stage-1 Calibrated Prediction immediately (Total: <strong>21.58 ms/img</strong>)
      </div>
      <div class="gate-card" style="border-left:4px solid #d97706;">
        <strong>Stage-2 Fallback (Only 1.53% Routed):</strong><br>
        RBF-SVM + Shallow-MLP + Cosine-KNN (0.30 <i>P</i><sub>1</sub> + 0.70 <i>P</i><sub>2</sub>, Amortized: <strong>21.74 ms</strong>)
      </div>
    </div>

    <div class="arrow-connector">&rarr;</div>

    <!-- Stage 6: Output Classes -->
    <div class="stage-col output-col">
      <div class="stage-badge" style="background:#bfdbfe; color:#1e3a8a;">6. Diagnosis Output</div>
      <div class="class-list">
        <div class="class-pill active"><span>Anthracnose</span><span>0.962</span></div>
        <div class="class-pill"><span>Bacterial Canker</span><span>0.012</span></div>
        <div class="class-pill"><span>Cutting Weevil</span><span>0.003</span></div>
        <div class="class-pill"><span>Die Back</span><span>0.005</span></div>
        <div class="class-pill"><span>Gall Midge</span><span>0.008</span></div>
        <div class="class-pill"><span>Healthy</span><span>0.002</span></div>
        <div class="class-pill"><span>Powdery Mildew</span><span>0.004</span></div>
        <div class="class-pill"><span>Sooty Mold</span><span>0.004</span></div>
      </div>
      <div class="audit-pill" style="background:#dcfce7; border-color:#22c55e; color:#14532d; text-align:center;">
        <strong>Group-Split Acc: 98.47%</strong><br>
        <strong>Macro-F1: 0.9836</strong><br>
        ECE: 0.0104 &bull; AURC: 0.00158
      </div>
    </div>
  </div>
</div>

<!-- =====================================================================
     FIGURE 5: END-TO-END LATENCY & RESOURCE USAGE PROFILE
     ===================================================================== -->
<div class="figure-canvas" id="fig5-latency">
  <div class="fig-header">
    <div class="fig-title">Figure 5. End-to-End Single-Core CPU Latency Breakdown and Edge Resource Usage Profile</div>
    <div class="fig-subtitle">Fair end-to-end single-threaded CPU benchmark (batch size = 1, including disk I/O + preprocessing + inference) from Locked Table 6</div>
  </div>

  <div class="latency-grid">
    <!-- Panel A -->
    <div class="panel-box">
      <div class="panel-title">A. End-to-End Single-Core CPU Latency Breakdown &amp; Deep CNN Comparison</div>
      <div class="panel-sub">Proposed Stage-1 (<strong>21.58 ms</strong>) &amp; Two-Stage Gated (<strong>21.74 ms</strong>) vs. Transfer-Learned CNN Backbones</div>

      <div class="hbar-group">
        <div class="hbar-row">
          <div class="hbar-label">Quality Gate &amp; Image Standardization</div>
          <div class="hbar-track"><div class="hbar-fill" style="width:12.0%; background:#3b82f6;"></div></div>
          <div class="hbar-val">~14.2 ms</div>
        </div>
        <div class="hbar-row">
          <div class="hbar-label">7-Branch Feature Extraction (2,003D)</div>
          <div class="hbar-track"><div class="hbar-fill" style="width:6.8%; background:#0ea5e9;"></div></div>
          <div class="hbar-val">~7.3 ms</div>
        </div>
        <div class="hbar-row">
          <div class="hbar-label">Block-PCA (303D) + Stage-1 OvR LR</div>
          <div class="hbar-track"><div class="hbar-fill" style="width:1.0%; background:#10b981;"></div></div>
          <div class="hbar-val">0.08 ms</div>
        </div>
        <div class="hbar-row">
          <div class="hbar-label">Amortized Stage-2 Gate (&tau;<sub>gate</sub> = 0.65)</div>
          <div class="hbar-track"><div class="hbar-fill" style="width:1.2%; background:#f59e0b;"></div></div>
          <div class="hbar-val">+0.16 ms</div>
        </div>
        <div class="hbar-row" style="padding-top:6px; border-top:1.5px dashed #cbd5e1;">
          <div class="hbar-label" style="color:#15803d; font-weight:800;">Proposed Stage-1 Only (480.2 KB)</div>
          <div class="hbar-track"><div class="hbar-fill" style="width:18.5%; background:#16a34a;"></div></div>
          <div class="hbar-val" style="color:#15803d;">21.58 ms</div>
        </div>
        <div class="hbar-row">
          <div class="hbar-label" style="color:#0f172a; font-weight:800;">Proposed Two-Stage Gated (4.0 MB)</div>
          <div class="hbar-track"><div class="hbar-fill" style="width:18.7%; background:#059669;"></div></div>
          <div class="hbar-val" style="color:#059669;">21.74 ms</div>
        </div>
        <div class="hbar-row">
          <div class="hbar-label" style="color:#64748b;">MobileNetV3-Small (5.9 MB, CPU)</div>
          <div class="hbar-track"><div class="hbar-fill" style="width:41.0%; background:#64748b;"></div></div>
          <div class="hbar-val" style="color:#475569;">48.70 ms</div>
        </div>
        <div class="hbar-row">
          <div class="hbar-label" style="color:#64748b;">MobileNetV2 (8.8 MB, CPU)</div>
          <div class="hbar-track"><div class="hbar-fill" style="width:47.5%; background:#94a3b8;"></div></div>
          <div class="hbar-val" style="color:#475569;">56.42 ms</div>
        </div>
        <div class="hbar-row">
          <div class="hbar-label" style="color:#dc2626; font-weight:700;">ResNet50 (90.0 MB, CPU)</div>
          <div class="hbar-track"><div class="hbar-fill" style="width:100.0%; background:#dc2626;"></div></div>
          <div class="hbar-val" style="color:#dc2626;">272.04 ms</div>
        </div>
      </div>

      <div class="callout-grid">
        <div class="callout-note">
          <strong>Shared-Sobel &amp; 256-Entry LUT-LBP:</strong><br>
          Reusing 128&times;128 Sobel gradients across HOG, LBP, and MGCT cuts feature extraction overhead, yielding <strong>p50 = 20.43 ms</strong> and <strong>p95 = 32.47 ms</strong>.
        </div>
        <div class="callout-note">
          <strong>2.2&times; to 12.5&times; Faster on CPU:</strong><br>
          Runs <strong>2.24&times; faster</strong> than MobileNetV3-Small (48.70 ms), <strong>2.60&times; faster</strong> than MobileNetV2 (56.42 ms), and <strong>12.51&times; faster</strong> than ResNet50 (272.04 ms).
        </div>
      </div>
    </div>

    <!-- Panel B -->
    <div class="panel-box resource-stack">
      <div>
        <div class="panel-title">B. Memory Footprint &amp; Edge Deployment Characteristics</div>
        <div class="panel-sub">Designed for offline agricultural smartphones, Raspberry Pi, and ARM Cortex-A nodes</div>
      </div>

      <div class="res-card" style="background:#f0fdf4; border-color:#86efac;">
        <div class="res-icon" style="background:#dcfce7; color:#15803d;">KB</div>
        <div class="res-body">
          <h4 style="color:#14532d;">Ultra-Compact Model Size (480.2 KB / 4.0 MB)</h4>
          <p>Stage-1 requires only <strong>480.2 KB</strong> (&lt;0.02M params, <strong>187&times; smaller</strong> than ResNet50's 90.0 MB). The full Two-Stage Gated bundle occupies just <strong>4.0 MB</strong> (&lt;0.05M params).</p>
        </div>
      </div>

      <div class="res-card" style="background:#eff6ff; border-color:#93c5fd;">
        <div class="res-icon" style="background:#dbeafe; color:#1d4ed8;">46</div>
        <div class="res-body">
          <h4 style="color:#1e3a8a;">Real-Time Single-Core Throughput (46.0 – 46.3 img/s)</h4>
          <p>Delivers <strong>46.3 img/s</strong> (Stage-1) and <strong>46.0 img/s</strong> (Two-Stage Gated) on CPU—compared to <strong>20.5 img/s</strong> for MobileNetV3-Small and <strong>3.7 img/s</strong> for ResNet50.</p>
        </div>
      </div>

      <div class="res-card" style="background:#fefce8; border-color:#fde047;">
        <div class="res-icon" style="background:#fef08a; color:#a16207;">1.5%</div>
        <div class="res-body">
          <h4 style="color:#713f12;">Selective Stage-2 Routing Efficiency (&tau;<sub>gate</sub> = 0.65)</h4>
          <p><strong>98.47%</strong> of test leaves exit early via temperature-calibrated linear Stage-1; only <strong>1.53%</strong> hard borderline samples invoke Stage-2 ensemble refinement (+0.16 ms amortized).</p>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- =====================================================================
     FIGURE 6: REPRESENTATIVE OPTICAL MODES & CANONICAL POSE NORMALIZATION
     ===================================================================== -->
<div class="figure-canvas" id="fig6-optical">
  <div class="fig-header">
    <div class="fig-title">Figure 6. Representative Visual Audit Findings, Quality-Gated Restoration, and Canonical Pose Alignment</div>
    <div class="fig-subtitle">Mechanistic illustration of the three core methodological pillars across MangoLeafBD (Bangladesh, N=4,000) and MLDID (India, N=3,000)</div>
  </div>

  <div class="modes-table">
    <div class="mode-header">
      <div>Methodological Pillar</div>
      <div style="text-align:center;">Raw / Unaligned Input View</div>
      <div style="text-align:center;">Audited / Restored / Aligned View</div>
      <div>Mechanistic Interpretation &amp; Quantitative Impact (Locked v5 Metrics)</div>
    </div>

    <!-- Row 1: Near-Duplicate & Rotation Sibling Audit -->
    <div class="mode-row">
      <div class="mode-tag">
        <span class="badge" style="background:#fee2e2; color:#991b1b;">Pillar 1 &bull; Dataset Audit</span>
        <h4>Near-Duplicate &amp; Zoom-Rotated Siblings</h4>
        <p>pHash (<i>d</i> &le; 6) | HSV Cosine (&ge; 0.992) Graph Connected Components</p>
      </div>
      <div class="mode-img-wrap">
        <img src="{thumbs['sibling_a']}" alt="Sibling A">
        <span>Specimen View A (0&deg;)</span>
      </div>
      <div class="mode-img-wrap">
        <img src="{thumbs['sibling_b']}" alt="Sibling B">
        <span>Zoom/Rotated Sibling B (90&deg;)</span>
      </div>
      <div class="mode-analysis">
        <strong>Why Un-Audited Random Splits Inflate Performance:</strong> Across MangoLeafBD's 4,000 images, our specimen-clustering audit discovers only <strong>2,554 unique leaf clusters</strong> (<strong>36.15% near-duplicate / augmentation-mix ratio</strong>, reaching 93.8% in Cutting Weevil and 81.0% in Die Back). Under a standard random split, <strong>39.0% of test images share a specimen cluster with training data</strong>. Enforcing StratifiedGroupKFold eliminates this leakage completely, while our Five-View Shortcut Audit (Table 2) proves segmented leaves alone retain <span class="metric-chip">94.84% Acc (0.9440 F1)</span> and M11 achieves <span class="metric-chip">96.37% Acc (0.9549 F1)</span> without background cues.
      </div>
    </div>

    <!-- Row 2: Selective Quality-Gated Restoration -->
    <div class="mode-row">
      <div class="mode-tag">
        <span class="badge" style="background:#dbeafe; color:#1e40af;">Pillar 2 &bull; Quality Gate</span>
        <h4>Foreground-Aware Optical Restoration</h4>
        <p>MAD Noise Gate (&sigma;&#770;<sub>MAD</sub> &ge; 5.5) + Glare Attenuation + Foreground CLAHE</p>
      </div>
      <div class="mode-img-wrap">
        <img src="{thumbs['perturbed_glare']}" alt="Perturbed Leaf">
        <span>Low-Light + Grain + Specular Glare</span>
      </div>
      <div class="mode-img-wrap">
        <img src="{thumbs['restored_clahe']}" alt="Restored Leaf">
        <span>Quality-Gated Restored ROI</span>
      </div>
      <div class="mode-analysis">
        <strong>Preventing Perturbation Collapse Without Distorting Clean Leaves:</strong> Unconditional CLAHE severely amplifies sensor noise (&sigma;=10 F1 drops from 0.5114 to 0.4207). By conditioning median/Gaussian denoising strictly on &sigma;&#770;<sub>MAD</sub> &ge; 5.5 and local glare replacement on central foreground glare masks (<i>V</i>&gt;238, <i>S</i>&lt;38), our Quality-Gated pipeline preserves clean test accuracy (<strong>0.9836 F1</strong>) while delivering massive gains under field degradations (Table 5): <span class="metric-chip">+0.3533 F1 at Noise &sigma;=10 (0.8647)</span> <span class="metric-chip">+0.2359 F1 at Glare r=35 (0.9770)</span> <span class="metric-chip">+0.1371 F1 at Low-Light -40%</span>.
      </div>
    </div>

    <!-- Row 3: Canonical PCA-Axis Leaf Pose Normalization -->
    <div class="mode-row">
      <div class="mode-tag">
        <span class="badge" style="background:#dcfce7; color:#166534;">Pillar 3 &bull; Domain Shift</span>
        <h4>Canonical Leaf Pose Normalization</h4>
        <p>PCA Major-Axis Horizontal Alignment + Foreground <i>L*</i> Constancy (lab_l)</p>
      </div>
      <div class="mode-img-wrap">
        <img src="{thumbs['mldid_portrait']}" alt="MLDID Portrait">
        <span>MLDID Portrait Leaf (240&times;293)</span>
      </div>
      <div class="mode-img-wrap">
        <img src="{thumbs['mldid_canonical']}" alt="MLDID Canonical">
        <span>Canonical Horizontal Pose (0&deg;)</span>
      </div>
      <div class="mode-analysis">
        <strong>Diagnosing and Resolving Cross-Dataset Orientation Shift:</strong> EDA reveals that MangoLeafBD leaves are captured predominantly in landscape (320&times;240, aspect ratio 1.33), whereas external MLDID leaves (<i>N</i>=3,000 across 5 shared classes) are captured in portrait (240&times;293, aspect ratio 0.82), misaligning 8&times;8 HOG cells and 5-Patch quadrants. Rotating every segmented leaf organ to its principal inertia axis and normalizing foreground <i>L*</i> boosts zero-shot cross-dataset accuracy without any target labels (Table 7): <span class="metric-chip">Zero-Shot Acc: 31.27% &rarr; 42.63% (+11.36%)</span> <span class="metric-chip">Macro-F1: 0.2886 &rarr; 0.4395 (+0.1509)</span>.
      </div>
    </div>
  </div>
</div>

</body>
</html>
"""


def render_diagrams_with_playwright(html_path: Path, figures_dir: Path) -> None:
    from playwright.sync_api import sync_playwright

    targets = [
        ("#fig1-pipeline", figures_dir / "Figure1_Pipeline_Architecture_Overview.png"),
        ("#fig5-latency", figures_dir / "Figure5_Latency_and_Resource_Profile.png"),
        ("#fig6-optical", figures_dir / "Figure6_Optical_Modes_and_Canonical_Pose.png"),
    ]
    with sync_playwright() as p:
        browser = None
        for ch in ("msedge", "chrome", None):
            try:
                browser = (
                    p.chromium.launch(channel=ch, headless=True)
                    if ch is not None
                    else p.chromium.launch(headless=True)
                )
                break
            except Exception:
                continue
        if browser is None:
            raise RuntimeError("Could not launch headless Chromium/Edge/Chrome via Playwright.")
        page = browser.new_page(viewport={"width": 1680, "height": 1200}, device_scale_factor=2)
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
        for selector, out_png in targets:
            loc = page.locator(selector)
            loc.screenshot(path=str(out_png))
            print(f"[OK] Rendered high-DPI publication figure: {out_png.relative_to(REPO_ROOT)}")
        browser.close()


def main() -> None:
    figures_dir = REPO_ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    thumbs = prepare_visual_thumbnails()
    html_content = build_showcase_html(thumbs)
    html_path = figures_dir / "publication_diagrams_showcase.html"
    html_path.write_text(html_content, encoding="utf-8")
    print(f"[OK] Wrote self-contained HTML diagram showcase: {html_path.relative_to(REPO_ROOT)}")
    render_diagrams_with_playwright(html_path, figures_dir)


if __name__ == "__main__":
    main()
