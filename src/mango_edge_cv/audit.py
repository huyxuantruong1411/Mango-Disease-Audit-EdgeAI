"""
Near-Duplicate Specimen Clustering, StratifiedGroupKFold Partitioning, and
Five-View Background Shortcut Auditing (`src/mango_edge_cv/audit.py`).
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

from .config import MANIFEST_SCHEMA_VERSION, SEED, scan_dataset_records
from .features import fast_vectorized_lbp_28d
from .preprocessing import segment_plant_organ_roi


def compute_phash_and_hsv_signature(img_bgr: np.ndarray):
    """
    Compute 64-bit DCT Perceptual Hash (pHash) and L2-normalized central 60%-ROI
    2D Hue-Saturation histogram (12x8 = 96 bins) to capture rotated/zoomed near-duplicates.
    """
    gray_32 = cv2.resize(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
        (32, 32),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)
    dct_mat = cv2.dct(gray_32)[:8, :8]
    med_val = np.median(dct_mat[1:, 1:])
    bits = (dct_mat > med_val).flatten().astype(np.int8)

    h, w = img_bgr.shape[:2]
    crop = img_bgr[int(h * 0.2) : int(h * 0.8), int(w * 0.2) : int(w * 0.8)]
    hsv_small = cv2.cvtColor(
        cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2HSV,
    )
    hist_hs = (
        cv2.calcHist([hsv_small], [0, 1], None, [12, 8], [0, 180, 0, 256])
        .flatten()
        .astype(np.float32)
    )
    hist_hs /= np.linalg.norm(hist_hs) + 1e-8
    return bits, hist_hs


def write_manifest_audit_table(df: pd.DataFrame, output_dir: str = None) -> pd.DataFrame:
    """Rebuild Table 1 (Dataset Near-Duplicate & Leakage Audit) from a validated manifest."""
    rows = []
    for class_id, (class_name, group) in enumerate(df.groupby("label_name", sort=True)):
        n_images = len(group)
        n_clusters = group["group_id"].nunique()
        mode_values = group["resolution"].mode()
        rows.append(
            {
                "Class ID": class_id,
                "Class Name": class_name,
                "Total Images": n_images,
                "Unique Specimen Clusters (pHash<=6 | HSV cosine>=0.992)": n_clusters,
                "Near-Duplicate / Aug-Mix Ratio (%)": round(
                    (1.0 - n_clusters / max(1, n_images)) * 100.0, 2
                ),
                "Dominant Resolution": (
                    mode_values.iloc[0] if not mode_values.empty else "Unknown"
                ),
            }
        )
    train_groups = set(df.loc[df["split_random"] == "Train_Orig", "group_id"])
    random_test = df[df["split_random"] == "Test_Orig"]
    leaked = sum(g in train_groups for g in random_test["group_id"].values)
    leak_pct = 100.0 * leaked / max(1, len(random_test))
    summary = pd.DataFrame(rows)
    summary["Random-Split Test Leakage (%)"] = round(leak_pct, 2)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        summary.to_csv(
            os.path.join(output_dir, "Table1_Dataset_Leakage_and_Group_Audit.csv"),
            index=False,
        )
    return summary


def build_audited_manifest(
    data_root: str,
    track_name: str = "MANGO",
    manifest_csv_path: str = None,
    output_dir: str = None,
    seed: int = SEED,
):
    """
    Build an audited image manifest with connected-components specimen clustering
    (pHash Hamming <= 6 OR HSV cosine >= 0.992) and 5-fold StratifiedGroupKFold splits.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    records, discovered_folders = scan_dataset_records(
        data_root, track_name=track_name, is_external=False
    )
    if not records:
        raise RuntimeError(
            f"No valid images mapped under {data_root} for {track_name}. "
            f"Discovered folders: {sorted(discovered_folders)[:20]}"
        )

    df = (
        pd.DataFrame(records)
        .drop_duplicates(subset=["file_path"])
        .sort_values(["label_name", "file_path"], kind="stable")
        .reset_index(drop=True)
    )

    valid_rows, bits_list, hist_list = [], [], []
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc=f"Image signature audit — {track_name}"
    ):
        bgr = cv2.imread(row["file_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        bits, hsv_sig = compute_phash_and_hsv_signature(bgr)
        record = row.to_dict()
        record["resolution"] = f"{bgr.shape[1]}x{bgr.shape[0]}"
        valid_rows.append(record)
        bits_list.append(bits)
        hist_list.append(hsv_sig)

    df = pd.DataFrame(valid_rows).reset_index(drop=True)
    class_names = sorted(df["label_name"].unique().tolist())
    label_to_id = {name: idx for idx, name in enumerate(class_names)}
    df["label_id"] = df["label_name"].map(label_to_id).astype(int)
    df["manifest_schema"] = MANIFEST_SCHEMA_VERSION

    bits_mat_all = np.asarray(bits_list, dtype=np.int8)
    hist_mat_all = np.asarray(hist_list, dtype=np.float32)
    df["group_id"] = -1

    for class_id, _ in enumerate(class_names):
        class_idx = df.index[df["label_id"] == class_id].to_numpy()
        n = len(class_idx)
        if n == 0:
            continue
        bits_mat = bits_mat_all[class_idx]
        hsv_mat = hist_mat_all[class_idx]
        hamming = np.abs(bits_mat[:, None, :] - bits_mat[None, :, :]).sum(axis=2)
        cosine = hsv_mat @ hsv_mat.T
        adjacency = (hamming <= 6) | (cosine >= 0.992)

        visited = np.zeros(n, dtype=bool)
        for u in range(n):
            if visited[u]:
                continue
            cluster_id = int(class_idx[u])
            stack = [u]
            visited[u] = True
            while stack:
                cur = stack.pop()
                df.loc[class_idx[cur], "group_id"] = cluster_id
                neighbors = np.flatnonzero(adjacency[cur] & (~visited))
                for nb in neighbors:
                    visited[nb] = True
                    stack.append(int(nb))

    # Leak-free 5-fold StratifiedGroupKFold: Fold 0 -> Test_Orig, Fold 1 -> Val_Orig, Folds 2-4 -> Train_Orig
    df["split_group"] = "Train_Orig"
    df["fold_id"] = -1
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold_num, (_, test_pos) in enumerate(
        sgkf.split(df, df["label_id"].values, df["group_id"].values)
    ):
        df.loc[test_pos, "fold_id"] = fold_num
        if fold_num == 0:
            df.loc[test_pos, "split_group"] = "Test_Orig"
        elif fold_num == 1:
            df.loc[test_pos, "split_group"] = "Val_Orig"

    # Leakage-sensitive Random-Split reference (60/20/20)
    df["split_random"] = "Train_Orig"
    tr_idx, tmp_idx = train_test_split(
        df.index.values,
        test_size=0.40,
        stratify=df["label_id"].values,
        random_state=seed,
    )
    val_idx, test_idx = train_test_split(
        tmp_idx,
        test_size=0.50,
        stratify=df.loc[tmp_idx, "label_id"].values,
        random_state=seed,
    )
    df.loc[val_idx, "split_random"] = "Val_Orig"
    df.loc[test_idx, "split_random"] = "Test_Orig"

    audit_df = write_manifest_audit_table(df, output_dir=output_dir)
    if manifest_csv_path:
        os.makedirs(os.path.dirname(os.path.abspath(manifest_csv_path)), exist_ok=True)
        df.to_csv(manifest_csv_path, index=False)

    return df, class_names, audit_df


def _shortcut_color_texture_features(img_rgb_256: np.ndarray) -> np.ndarray:
    """Extract HSV(52) + CIE-LAB(57) + Multi-Scale LBP(28) = 137D for the 5-View Shortcut Audit."""
    img_128 = cv2.resize(img_rgb_256, (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(img_128, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(img_128, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(img_128, cv2.COLOR_RGB2GRAY)

    hsv_hist = [
        cv2.normalize(
            cv2.calcHist(
                [hsv], [channel], None, [16], [0, 180] if channel == 0 else [0, 256]
            ),
            None,
        ).flatten()
        for channel in range(3)
    ]
    avg_h = float(np.mean(hsv[:, :, 0]) / 180.0)
    avg_s = float(np.mean(hsv[:, :, 1]) / 255.0)
    avg_v = float(np.mean(hsv[:, :, 2]) / 255.0)
    f_hsv = np.concatenate(
        [
            *hsv_hist,
            np.asarray(
                [avg_h, avg_s, avg_v, (avg_h + avg_s + avg_v) / 3.0], dtype=np.float32
            ),
        ]
    )

    lab_hist = [
        cv2.normalize(
            cv2.calcHist([lab], [channel], None, [16], [0, 256]), None
        ).flatten()
        for channel in range(3)
    ]
    moments = []
    for channel in range(3):
        values = lab[:, :, channel].astype(np.float32).ravel()
        mean_value = float(np.mean(values))
        std_value = float(np.std(values)) + 1e-6
        moments.extend(
            [
                mean_value / 255.0,
                std_value / 255.0,
                float(np.mean(((values - mean_value) / std_value) ** 3)),
            ]
        )
    f_lab = np.concatenate([*lab_hist, np.asarray(moments, dtype=np.float32)])
    f_lbp = fast_vectorized_lbp_28d(gray)
    return np.concatenate([f_hsv, f_lab, f_lbp]).astype(np.float32)


def run_background_shortcut_audit(
    df_eval: pd.DataFrame,
    track_name: str = "MANGO",
    output_dir: str = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Evaluate the 5-View Background & Framing Shortcut Audit under leak-free Group-Split:
      1. Whole ROI (Center-Crop r=0.86)
      2. Segmented Organ Only (Background Masked Out)
      3. Background Only (Organ Masked Out)
      4. Four Corner Patches Only (32x32 Borders)
      5. Image File Metadata Only (Size + Global Stats + Aspect Ratio)
    """
    splits = df_eval["split_group"].values
    y_all = df_eval["label_id"].values

    feats_whole, feats_fg, feats_bg, feats_corners, feats_meta = [], [], [], [], []
    for _, row in tqdm(
        df_eval.iterrows(), total=len(df_eval), desc="Running Shortcut Audit"
    ):
        bgr = cv2.imread(row["file_path"])
        if bgr is None:
            raise FileNotFoundError(f"Cannot read image at {row['file_path']}")
        rgb_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_full.shape

        ch, cw = int(h * 0.86), int(w * 0.86)
        sy, sx = (h - ch) // 2, (w - cw) // 2
        roi_whole = cv2.resize(
            rgb_full[sy : sy + ch, sx : sx + cw],
            (256, 256),
            interpolation=cv2.INTER_AREA,
        )

        fg_im, bg_im, mask = segment_plant_organ_roi(roi_whole)

        c_tl = rgb_full[:32, :32]
        c_tr = rgb_full[:32, -32:]
        c_bl = rgb_full[-32:, :32]
        c_br = rgb_full[-32:, -32:]
        corners_mosaic = cv2.resize(
            np.vstack([np.hstack([c_tl, c_tr]), np.hstack([c_bl, c_br])]),
            (256, 256),
            interpolation=cv2.INTER_NEAREST,
        )

        f_meta = np.array(
            [
                row["file_size_kb"],
                float(np.mean(rgb_full[:, :, 0])),
                float(np.mean(rgb_full[:, :, 1])),
                float(np.mean(rgb_full[:, :, 2])),
                float(np.std(rgb_full)),
                float(np.mean(mask > 0)),
                float(h / (w + 1e-5)),
            ],
            dtype=np.float32,
        )

        feats_whole.append(_shortcut_color_texture_features(roi_whole))
        feats_fg.append(_shortcut_color_texture_features(fg_im))
        feats_bg.append(_shortcut_color_texture_features(bg_im))
        feats_corners.append(_shortcut_color_texture_features(corners_mosaic))
        feats_meta.append(f_meta)

    views = [
        ("1. Whole ROI (Center-Crop r=0.86)", np.array(feats_whole)),
        ("2. Segmented Organ Only (Background Masked Out)", np.array(feats_fg)),
        ("3. Background Only (Organ Masked Out)", np.array(feats_bg)),
        ("4. Four Corner Patches Only (32x32 Borders)", np.array(feats_corners)),
        ("5. Image File Metadata Only (Size + Global Stats)", np.array(feats_meta)),
    ]

    tr_m = splits == "Train_Orig"
    te_m = splits == "Test_Orig"
    audit_rows = []
    for v_name, X_mat in views:
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_mat[tr_m])
        X_te_s = scaler.transform(X_mat[te_m])
        clf = LogisticRegression(
            C=0.5, class_weight="balanced", max_iter=1000, random_state=seed
        )
        clf.fit(X_tr_s, y_all[tr_m])
        preds = clf.predict(X_te_s)
        audit_rows.append(
            {
                "Track": track_name,
                "Input Representation View": v_name,
                "Group-Split Test Acc (%)": round(
                    accuracy_score(y_all[te_m], preds) * 100.0, 2
                ),
                "Group-Split Macro-F1": round(
                    f1_score(y_all[te_m], preds, average="macro"), 4
                ),
            }
        )

    df_shortcut = pd.DataFrame(audit_rows)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df_shortcut.to_csv(
            os.path.join(output_dir, "Table2_Background_Shortcut_Audit.csv"),
            index=False,
        )
    return df_shortcut
