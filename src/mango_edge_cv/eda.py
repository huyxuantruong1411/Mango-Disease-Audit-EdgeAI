"""
Exploratory Data Analysis (EDA) Engine for MangoLeafBD and MLDID
(`src/mango_edge_cv/eda.py` — ported from v8_eda).
"""

import os
import cv2
import numpy as np
import pandas as pd
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable


def measure_image_for_eda(row: dict) -> dict:
    """Measure geometry, RGB/HSV/LAB moments, sharpness, glare/shadow ratios, and entropy for one image."""
    path = str(row["file_path"])
    try:
        file_size_kb = os.path.getsize(path) / 1024.0
    except OSError:
        file_size_kb = float(row.get("file_size_kb", np.nan))

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return {
            "file_path": path,
            "filename": row.get("filename", os.path.basename(path)),
            "label_name": row.get("label_name", "Unknown"),
            "split_group": row.get("split_group", "External"),
            "group_id": row.get("group_id", np.nan),
            "decoded": False,
        }

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    small = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    l_ch = lab[:, :, 0].astype(np.float32)
    p5, p95 = np.percentile(l_ch, [5, 95])
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    prob = hist / max(1.0, hist.sum())
    prob = prob[prob > 0]
    entropy = float(-np.sum(prob * np.log2(prob)))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())

    return {
        "file_path": path,
        "filename": row.get("filename", os.path.basename(path)),
        "label_name": row.get("label_name", "Unknown"),
        "split_group": row.get("split_group", "External"),
        "group_id": row.get("group_id", np.nan),
        "decoded": True,
        "width": int(w),
        "height": int(h),
        "resolution": f"{w}x{h}",
        "aspect_ratio": round(float(w) / max(1.0, float(h)), 4),
        "file_size_kb": round(float(file_size_kb), 2),
        "brightness_l": round(float(np.mean(l_ch)), 2),
        "contrast_l_p95_p05": round(float(p95 - p5), 2),
        "saturation_mean": round(float(np.mean(hsv[:, :, 1])), 2),
        "r_mean": round(float(np.mean(small[:, :, 0])), 2),
        "g_mean": round(float(np.mean(small[:, :, 1])), 2),
        "b_mean": round(float(np.mean(small[:, :, 2])), 2),
        "lab_a_mean": round(float(np.mean(lab[:, :, 1])), 2),
        "lab_b_mean": round(float(np.mean(lab[:, :, 2])), 2),
        "sharpness_lap_var": round(sharpness, 2),
        "gray_entropy": round(entropy, 3),
    }


def summarize_image_eda(df_metrics: pd.DataFrame, dataset_name: str, output_dir: str = None):
    """Aggregate class-level, resolution-level, and split-level EDA summaries."""
    valid = df_metrics[df_metrics["decoded"]].copy()
    total = max(1, len(valid))
    class_rows = []
    for cls_name, grp in valid.groupby("label_name", sort=True):
        n_img = len(grp)
        has_groups = grp["group_id"].notna().any()
        n_grp = int(grp["group_id"].nunique()) if has_groups else np.nan
        dup_ratio = (
            round((1.0 - n_grp / max(1, n_img)) * 100.0, 1)
            if has_groups
            else np.nan
        )
        class_rows.append(
            {
                "Class": cls_name,
                "Images": n_img,
                "Share (%)": round(100.0 * n_img / total, 2),
                "Unique groups": n_grp,
                "Near-duplicate ratio (%)": dup_ratio,
                "Median width": float(grp["width"].median()),
                "Median height": float(grp["height"].median()),
                "Median file size (KB)": round(float(grp["file_size_kb"].median()), 2),
                "Median aspect ratio": round(float(grp["aspect_ratio"].median()), 3),
                "Median brightness (L*)": round(float(grp["brightness_l"].median()), 2),
                "Median contrast (L* P95-P05)": round(
                    float(grp["contrast_l_p95_p05"].median()), 1
                ),
                "Median saturation": round(float(grp["saturation_mean"].median()), 2),
                "Median RGB (R/G/B)": (
                    f"{grp['r_mean'].median():.1f} / "
                    f"{grp['g_mean'].median():.1f} / "
                    f"{grp['b_mean'].median():.1f}"
                ),
                "Median LAB a*/b*": (
                    f"{grp['lab_a_mean'].median():.1f} / {grp['lab_b_mean'].median():.1f}"
                ),
                "Median sharpness": round(float(grp["sharpness_lap_var"].median()), 2),
                "Median gray entropy": round(float(grp["gray_entropy"].median()), 3),
            }
        )
    df_class = pd.DataFrame(class_rows)
    df_res = (
        valid.groupby("resolution")
        .size()
        .reset_index(name="Images")
        .sort_values("Images", ascending=False)
        .reset_index(drop=True)
    )
    df_split = (
        valid.groupby(["split_group", "label_name"])
        .size()
        .reset_index(name="Images")
        .rename(columns={"split_group": "Split", "label_name": "Class"})
    )

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df_class.to_csv(
            os.path.join(output_dir, f"{dataset_name}_ClassSummary.csv"), index=False
        )
        df_res.to_csv(
            os.path.join(output_dir, f"{dataset_name}_ResolutionSummary.csv"),
            index=False,
        )
        df_split.to_csv(
            os.path.join(output_dir, f"{dataset_name}_SplitSummary.csv"), index=False
        )
    return df_class, df_res, df_split


def run_image_dataset_eda(df_records: pd.DataFrame, dataset_name: str, output_dir: str = None):
    """Run full image-level EDA over a manifest DataFrame and save summary tables."""
    rows = [
        measure_image_for_eda(row)
        for row in tqdm(
            df_records.to_dict("records"),
            total=len(df_records),
            desc=f"Image EDA — {dataset_name}",
        )
    ]
    df_metrics = pd.DataFrame(rows)
    df_class, df_res, df_split = summarize_image_eda(
        df_metrics, dataset_name=dataset_name, output_dir=output_dir
    )
    return {
        "metrics": df_metrics,
        "class_summary": df_class,
        "resolution_summary": df_res,
        "split_summary": df_split,
    }
