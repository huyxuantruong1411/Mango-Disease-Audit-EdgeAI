"""
Statistical Evaluation, Progressive Ablation (M1-M11), Confidence-Gated Routing,
Multi-Severity Robustness, End-to-End CPU Benchmarking, and External MLDID
Cross-Dataset Validation (`src/mango_edge_cv/evaluation.py`).
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import joblib
import numpy as np
import pandas as pd
import psutil
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # type: ignore[misc]
        return iterable

from .config import (
    CANONICAL_COLOR_MODE,
    FEATURE_CACHE_VERSION,
    HOG_BACKEND,
    SEED,
    scan_dataset_records,
)
from .features import (
    assemble_matrix_by_mask,
    compose_feature_vector,
    extract_fast_all_branches,
)
from .models import (
    BlockPCACompressor,
    CalibratedConfidenceGatedSystem,
    compute_ece_and_aurc,
    coral_align_features,
)
from .preprocessing import (
    _estimate_noise_sigma_mad,
    _usable_foreground_mask,
    apply_adaptive_quality_preprocessing,
    apply_clahe_only,
    read_and_standardize_rgb,
    segment_plant_organ_roi,
)


def bootstrap_macro_f1_ci(
    y_true: np.ndarray, y_pred: np.ndarray, n_boot: int = 1000, seed: int = SEED
):
    """Non-parametric percentile bootstrap 95% confidence interval for Macro-F1."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan"), float("nan"), float("nan")
    labels = np.unique(y_true)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        scores.append(
            f1_score(
                y_true[idx], y_pred[idx], labels=labels, average="macro", zero_division=0
            )
        )
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(np.mean(scores)), float(lo), float(hi)


def mcnemar_exact_pvalue(
    y_true: np.ndarray, pred_base: np.ndarray, pred_prop: np.ndarray
) -> float:
    """Exact two-sided binomial McNemar test on discordant test-set predictions."""
    correct_base = np.asarray(pred_base) == np.asarray(y_true)
    correct_prop = np.asarray(pred_prop) == np.asarray(y_true)
    b = int(np.sum(correct_base & (~correct_prop)))
    c = int(np.sum((~correct_base) & correct_prop))
    if b + c == 0:
        return 1.0
    return float(binomtest(min(b, c), n=b + c, p=0.5, alternative="two-sided").pvalue)


def run_progressive_ablation(
    feat_dict: dict, track_name: str = "MANGO", output_dir: str = None, seed: int = SEED
) -> pd.DataFrame:
    """Run the 14-step Progressive Ablation Study (M1 to M11) under leak-free Group-Split."""
    branch_cost_ms = {
        "hsv": 0.52,
        "lab": 0.58,
        "lbp": 0.44,
        "patch5": 0.38,
        "hog": 0.40,
        "q": 0.14,
        "mgct": 0.45,
    }
    preproc_cost_ms = {
        "raw": 0.80,
        "glare": 3.20,
        "clahe": 1.20,
        "morph": 2.05,
        "adaptive": 1.05,
        "segmented": 2.40,
    }

    steps = [
        ("M1. HSV Only (OvR LR)", "raw", ["hsv"], False),
        ("M2. HSV + Fast Multi-Scale LBP (OvR LR)", "raw", ["hsv", "lbp"], False),
        ("M3. HSV + Vectorized Sobel HOG (OvR LR)", "raw", ["hsv", "hog"], False),
        ("M4. Baseline: HSV + LBP + HOG (OvR LR)", "raw", ["hsv", "lbp", "hog"], False),
        (
            "M5. M4 + CIE LAB Color Space",
            "raw",
            ["hsv", "lab", "lbp", "hog"],
            False,
        ),
        (
            "M6. M5 + 5-Patch Spatial Moments",
            "raw",
            ["hsv", "lab", "lbp", "patch5", "hog"],
            False,
        ),
        (
            "M7a. M6 + Specular Glare Inpainting Only",
            "glare",
            ["hsv", "lab", "lbp", "patch5", "hog"],
            False,
        ),
        (
            "M7b. M6 + CLAHE Contrast Only",
            "clahe",
            ["hsv", "lab", "lbp", "patch5", "hog"],
            False,
        ),
        (
            "M7c. M6 + Morphological Top/Black-Hat Only",
            "morph",
            ["hsv", "lab", "lbp", "patch5", "hog"],
            False,
        ),
        (
            "M7d. M6 + Quality-Gated Conditional Preproc",
            "adaptive",
            ["hsv", "lab", "lbp", "patch5", "hog"],
            False,
        ),
        (
            "M8. M7d + Image Quality Vector q",
            "adaptive",
            ["hsv", "lab", "lbp", "patch5", "q", "hog"],
            False,
        ),
        (
            "M9. M8 + Multi-Grid Chromatic-Texture (MGCT)",
            "adaptive",
            ["hsv", "lab", "lbp", "patch5", "q", "mgct", "hog"],
            False,
        ),
        (
            "M10. M9 + Block-PCA Compression (Proposed Stage-1)",
            "adaptive",
            ["hsv", "lab", "lbp", "patch5", "q", "mgct", "hog"],
            True,
        ),
        (
            "M11. M10 on Segmented Organ ROI (No-BG Audit)",
            "segmented",
            ["hsv", "lab", "lbp", "patch5", "q", "mgct", "hog"],
            True,
        ),
    ]

    splits = feat_dict["splits"]
    folds = feat_dict["fold_ids"]
    tr_m = splits == "Train_Orig"
    val_m = splits == "Val_Orig"
    te_m = splits == "Test_Orig"

    X_tr_m4, y_tr_m4 = assemble_matrix_by_mask(
        feat_dict, tr_m, "raw", ["hsv", "lbp", "hog"]
    )
    X_te_m4, y_te_m4 = assemble_matrix_by_mask(
        feat_dict, te_m, "raw", ["hsv", "lbp", "hog"]
    )
    sc_m4 = StandardScaler()
    clf_m4 = LogisticRegression(
        C=0.5, class_weight="balanced", max_iter=1000, random_state=seed
    ).fit(sc_m4.fit_transform(X_tr_m4), y_tr_m4)
    preds_m4_te = clf_m4.predict(sc_m4.transform(X_te_m4))
    base_m4_lat = (
        preproc_cost_ms["raw"]
        + sum(branch_cost_ms[b] for b in ["hsv", "lbp", "hog"])
        + 0.03
    )

    rows = []
    for cfg_name, mode, branches, use_pca in steps:
        X_tr, y_tr = assemble_matrix_by_mask(feat_dict, tr_m, mode, branches)
        X_val, y_val = assemble_matrix_by_mask(feat_dict, val_m, mode, branches)
        X_te, y_te = assemble_matrix_by_mask(feat_dict, te_m, mode, branches)

        if use_pca:
            comp = BlockPCACompressor(hog_dim=1764, n_hog_pca=64)
            X_tr_s = comp.fit_transform(X_tr)
            X_val_s = comp.transform(X_val)
            X_te_s = comp.transform(X_te)
        else:
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_val_s = sc.transform(X_val)
            X_te_s = sc.transform(X_te)

        best_c, best_vf1 = 1.0, -1.0
        for c_cand in [0.05, 0.2, 0.5, 1.0, 2.5]:
            m_cand = LogisticRegression(
                C=c_cand, class_weight="balanced", max_iter=1000, random_state=seed
            ).fit(X_tr_s, y_tr)
            vf1 = f1_score(y_val, m_cand.predict(X_val_s), average="macro")
            if vf1 > best_vf1:
                best_vf1, best_c = vf1, c_cand

        clf = LogisticRegression(
            C=best_c, class_weight="balanced", max_iter=1000, random_state=seed
        ).fit(X_tr_s, y_tr)
        t0 = time.perf_counter()
        for _ in range(10):
            y_pred = clf.predict(X_te_s)
        infer_ms = ((time.perf_counter() - t0) * 1000.0) / (10.0 * len(X_te_s))

        cv_f1s = []
        for f_id in sorted(np.unique(folds[tr_m]).tolist()):
            f_te = (folds == f_id) & tr_m
            f_tr = tr_m & (folds != f_id)
            if f_te.sum() == 0 or f_tr.sum() == 0:
                continue
            Xf_tr, yf_tr = assemble_matrix_by_mask(feat_dict, f_tr, mode, branches)
            Xf_te, yf_te = assemble_matrix_by_mask(feat_dict, f_te, mode, branches)
            if use_pca:
                cp_f = BlockPCACompressor(hog_dim=1764, n_hog_pca=64)
                Xf_tr_s, Xf_te_s = cp_f.fit_transform(Xf_tr), cp_f.transform(Xf_te)
            else:
                sc_f = StandardScaler()
                Xf_tr_s, Xf_te_s = sc_f.fit_transform(Xf_tr), sc_f.transform(Xf_te)
            clf_f = LogisticRegression(
                C=best_c, class_weight="balanced", max_iter=1000, random_state=seed
            ).fit(Xf_tr_s, yf_tr)
            cv_f1s.append(f1_score(yf_te, clf_f.predict(Xf_te_s), average="macro"))

        _, ci_lo, ci_hi = bootstrap_macro_f1_ci(y_te, y_pred, n_boot=1500, seed=seed)
        p_mc = mcnemar_exact_pvalue(y_te, preds_m4_te, y_pred)
        est_ms = (
            preproc_cost_ms[mode]
            + sum(branch_cost_ms[b] for b in branches)
            + infer_ms * 10.0
        )
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
            y_te, y_pred, average="macro"
        )
        rows.append(
            {
                "Configuration": cfg_name,
                "Dims": X_tr_s.shape[1],
                "3-Fold Train Group-CV F1": f"{np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}",
                "Test Acc (%)": round(accuracy_score(y_te, y_pred) * 100.0, 2),
                "Macro-Precision": round(prec_m, 4),
                "Macro-Recall": round(rec_m, 4),
                "Test Macro-F1": round(f1_m, 4),
                "95% Bootstrap CI": f"[{ci_lo:.4f}, {ci_hi:.4f}]",
                "McNemar p-val": round(p_mc, 4),
                "Estimated Relative Latency (proxy)": f"{est_ms / base_m4_lat:.2f}x",
            }
        )

    df_abl = pd.DataFrame(rows)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        df_abl.to_csv(
            os.path.join(output_dir, "Table3_Progressive_Ablation_Study.csv"),
            index=False,
        )
    return df_abl


def evaluate_confidence_gating(
    feat_data: dict, track_name: str = "MANGO", output_dir: str = None, seed: int = SEED
):
    """Evaluate Table 4 (Confidence-Gated Two-Stage Routing & Probability Calibration)."""
    branches_full = ["hsv", "lab", "lbp", "patch5", "q", "mgct", "hog"]
    train_mask = feat_data["splits"] == "Train_Orig"
    val_mask = feat_data["splits"] == "Val_Orig"
    test_mask = feat_data["splits"] == "Test_Orig"

    candidates = []
    for view_mode in ["adaptive", "segmented"]:
        X_train, y_train = assemble_matrix_by_mask(
            feat_data, train_mask, view_mode, branches_full
        )
        X_val, y_val = assemble_matrix_by_mask(
            feat_data, val_mask, view_mode, branches_full
        )
        if len(X_train) == 0 or len(X_val) == 0:
            continue
        system = CalibratedConfidenceGatedSystem(
            hog_dim=1764, n_hog_pca=64
        ).fit_and_calibrate(X_train, y_train, X_val, y_val)
        p_val, _, route_rate = system.predict_gated_probs(
            X_val, tau_gate=system.best_tau_gate
        )
        val_f1 = f1_score(
            y_val, np.argmax(p_val, axis=1), average="macro", zero_division=0
        )
        candidates.append((val_f1, -route_rate, view_mode, system))

    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    _, _, best_mode, gated_system = candidates[0]
    gated_system.selected_view_mode = best_mode

    X_test, y_test = assemble_matrix_by_mask(
        feat_data, test_mask, best_mode, branches_full
    )
    gate_rows = []
    for tau in [0.0, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]:
        p_final, _, route_rate = gated_system.predict_gated_probs(X_test, tau_gate=tau)
        predictions = np.argmax(p_final, axis=1)
        ece_value, aurc_value = compute_ece_and_aurc(y_test, p_final)
        _, ci_lo, ci_hi = bootstrap_macro_f1_ci(
            y_test, predictions, n_boot=1000, seed=seed
        )
        mode_desc = (
            "Stage-1 LR Only"
            if tau == 0.0
            else (
                "Full Stage-2 Ensemble"
                if tau == 1.0
                else f"Confidence-Gated (tau={tau:.2f})"
            )
        )
        gate_rows.append(
            {
                "Routing Mode": mode_desc,
                "Gate Threshold (tau_gate)": tau,
                "Selected on Validation": bool(
                    np.isclose(float(tau), float(gated_system.best_tau_gate))
                ),
                "Stage-2 Routed Samples (%)": round(route_rate * 100.0, 2),
                "Test Accuracy (%)": round(
                    accuracy_score(y_test, predictions) * 100.0, 2
                ),
                "Group-Split Macro-F1": round(
                    f1_score(y_test, predictions, average="macro", zero_division=0), 4
                ),
                "95% Bootstrap CI": f"[{ci_lo:.4f}, {ci_hi:.4f}]",
                "Calibration ECE": round(ece_value, 4),
                "Selective Risk AURC": round(aurc_value, 5),
            }
        )

    random_train = feat_data["splits_random"] == "Train_Orig"
    random_val = feat_data["splits_random"] == "Val_Orig"
    random_test = feat_data["splits_random"] == "Test_Orig"
    X_rtrain, y_rtrain = assemble_matrix_by_mask(
        feat_data, random_train, best_mode, branches_full
    )
    X_rval, y_rval = assemble_matrix_by_mask(
        feat_data, random_val, best_mode, branches_full
    )
    X_rtest, y_rtest = assemble_matrix_by_mask(
        feat_data, random_test, best_mode, branches_full
    )
    if len(X_rtrain) and len(X_rval) and len(X_rtest):
        random_system = CalibratedConfidenceGatedSystem(
            hog_dim=1764, n_hog_pca=64
        ).fit_and_calibrate(X_rtrain, y_rtrain, X_rval, y_rval)
        p_random, _, random_route_rate = random_system.predict_gated_probs(
            X_rtest, tau_gate=random_system.best_tau_gate
        )
        pred_random = np.argmax(p_random, axis=1)
        ece_r, aurc_r = compute_ece_and_aurc(y_rtest, p_random)
        _, r_lo, r_hi = bootstrap_macro_f1_ci(
            y_rtest, pred_random, n_boot=1000, seed=seed
        )
        gate_rows.append(
            {
                "Routing Mode": "[REFERENCE] Leakage-sensitive Standard Random-Split",
                "Gate Threshold (tau_gate)": random_system.best_tau_gate,
                "Selected on Validation": True,
                "Stage-2 Routed Samples (%)": round(random_route_rate * 100.0, 2),
                "Test Accuracy (%)": round(
                    accuracy_score(y_rtest, pred_random) * 100.0, 2
                ),
                "Group-Split Macro-F1": round(
                    f1_score(y_rtest, pred_random, average="macro", zero_division=0), 4
                ),
                "95% Bootstrap CI": f"[{r_lo:.4f}, {r_hi:.4f}]",
                "Calibration ECE": round(ece_r, 4),
                "Selective Risk AURC": round(aurc_r, 5),
            }
        )

    result = pd.DataFrame(gate_rows)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result.to_csv(
            os.path.join(output_dir, "Table4_Confidence_Gated_and_Calibration.csv"),
            index=False,
        )
    return gated_system, result


def apply_parametric_perturbation(
    img_rgb: np.ndarray,
    pert_type: str = "gaussian_noise",
    severity: float = 10.0,
    rng=None,
) -> np.ndarray:
    """Apply parametric synthetic sensor noise, specular glare, or low-light contrast loss."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    img = img_rgb.copy()
    h, w, _ = img.shape
    if severity == 0:
        return img
    if pert_type == "gaussian_noise":
        noise = rng.normal(0, float(severity), img.shape).astype(np.float32)
        return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if pert_type == "glare_radius":
        overlay = img.copy()
        cx = int(w // 2 + rng.integers(-24, 24))
        cy = int(h // 2 + rng.integers(-24, 24))
        cv2.circle(
            overlay,
            (cx, cy),
            radius=int(severity),
            color=(255, 255, 252),
            thickness=-1,
        )
        return cv2.addWeighted(overlay, 0.82, img, 0.18, 0)
    if pert_type == "contrast_loss":
        alpha = max(0.35, 1.0 - float(severity) / 100.0)
        return np.clip(128.0 + alpha * (img.astype(np.float32) - 128.0), 0, 255).astype(
            np.uint8
        )
    return img


def evaluate_multilevel_robustness(
    feat_dict: dict,
    gated_sys: CalibratedConfidenceGatedSystem,
    track_name: str = "MANGO",
    output_dir: str = None,
    seed: int = SEED,
) -> pd.DataFrame:
    """Evaluate Table 5 (Multi-Severity Optical & Sensor Perturbation Robustness)."""
    branches_full = ["hsv", "lab", "lbp", "patch5", "q", "mgct", "hog"]
    tr_m = feat_dict["splits"] == "Train_Orig"
    te_m = feat_dict["splits"] == "Test_Orig"
    y_tr = feat_dict["labels"][tr_m]
    y_te_all = feat_dict["labels"][te_m]
    te_paths = feat_dict["file_paths"][te_m]
    use_seg = getattr(gated_sys, "selected_view_mode", "adaptive") == "segmented"

    loaded_images, valid_y = [], []
    for path, label in zip(te_paths, y_te_all):
        img = read_and_standardize_rgb(path)
        if img is not None:
            loaded_images.append(img)
            valid_y.append(int(label))
    y_te = np.asarray(valid_y, dtype=np.int32)

    X_tr_raw, _ = assemble_matrix_by_mask(feat_dict, tr_m, "raw", ["hsv", "lbp", "hog"])
    scaler_raw = StandardScaler().fit(X_tr_raw)
    clf_raw = LogisticRegression(
        C=0.5, class_weight="balanced", max_iter=1200, random_state=seed
    ).fit(scaler_raw.transform(X_tr_raw), y_tr)

    X_tr_clahe, _ = assemble_matrix_by_mask(feat_dict, tr_m, "clahe", branches_full)
    comp_clahe = BlockPCACompressor(1764, 64)
    X_tr_clahe_s = comp_clahe.fit_transform(X_tr_clahe)
    clf_clahe = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1200, random_state=seed
    ).fit(X_tr_clahe_s, y_tr)

    scenarios = [
        ("gaussian_noise", 0, "Clean held-out test (sigma=0)"),
        ("gaussian_noise", 5, "Gaussian sensor noise (sigma=5)"),
        ("gaussian_noise", 10, "Gaussian sensor noise (sigma=10)"),
        ("gaussian_noise", 20, "Gaussian sensor noise (sigma=20)"),
        ("glare_radius", 22, "Synthetic specular glare (radius=22 px)"),
        ("glare_radius", 35, "Synthetic specular glare (radius=35 px)"),
        ("contrast_loss", 40, "Low-light contrast loss (-40%)"),
    ]
    rows = []
    for p_type, severity, description in tqdm(scenarios, desc="Table 5 robustness"):
        rng = np.random.default_rng(
            seed + int(severity) + sum(ord(ch) for ch in p_type)
        )
        X_raw, X_blur, X_clahe, X_prop = [], [], [], []
        for image in loaded_images:
            clean_mask = None
            if use_seg:
                _, _, clean_mask = segment_plant_organ_roi(image)
            perturbed = apply_parametric_perturbation(image, p_type, severity, rng)

            fh, _, flbp, _, fhog, _, _ = extract_fast_all_branches(perturbed)
            X_raw.append(np.concatenate([fh, flbp, fhog]))

            gray = cv2.cvtColor(perturbed, cv2.COLOR_RGB2GRAY)
            mask_for_noise = (
                clean_mask if use_seg else _usable_foreground_mask(perturbed, None)
            )
            noise_est = _estimate_noise_sigma_mad(gray, mask_for_noise)
            blur_image = (
                cv2.GaussianBlur(perturbed, (3, 3), 0.55)
                if noise_est >= 5.5
                else perturbed
            )
            fhb, _, flbpb, _, fhogb, _, _ = extract_fast_all_branches(blur_image)
            X_blur.append(np.concatenate([fhb, flbpb, fhogb]))

            clahe_image = apply_clahe_only(perturbed)
            fhc, flabc, flbpc, fp5c, fhogc, fqc, fmgctc = extract_fast_all_branches(
                clahe_image
            )
            X_clahe.append(
                np.concatenate([fhc, flabc, flbpc, fp5c, fqc, fmgctc, fhogc])
            )

            if use_seg:
                prop_base, _, prop_mask = segment_plant_organ_roi(perturbed)
                prop_image = apply_adaptive_quality_preprocessing(
                    prop_base, foreground_mask=prop_mask
                )
            else:
                prop_image = apply_adaptive_quality_preprocessing(
                    perturbed, foreground_mask=None
                )
            fhp, flabp, flbp, fp5p, fhogp, fqp, fmgctp = extract_fast_all_branches(
                prop_image
            )
            X_prop.append(np.concatenate([fhp, flabp, flbp, fp5p, fqp, fmgctp, fhogp]))

        X_raw = np.asarray(X_raw, dtype=np.float32)
        X_blur = np.asarray(X_blur, dtype=np.float32)
        X_clahe = np.asarray(X_clahe, dtype=np.float32)
        X_prop = np.asarray(X_prop, dtype=np.float32)
        pred_raw = clf_raw.predict(scaler_raw.transform(X_raw))
        pred_blur = clf_raw.predict(scaler_raw.transform(X_blur))
        pred_clahe = clf_clahe.predict(comp_clahe.transform(X_clahe))
        probs_prop, _, _ = gated_sys.predict_gated_probs(X_prop)
        pred_prop = np.argmax(probs_prop, axis=1)
        f1_raw = f1_score(y_te, pred_raw, average="macro", zero_division=0)
        f1_blur = f1_score(y_te, pred_blur, average="macro", zero_division=0)
        f1_clahe = f1_score(y_te, pred_clahe, average="macro", zero_division=0)
        f1_prop = f1_score(y_te, pred_prop, average="macro", zero_division=0)
        rows.append(
            {
                "Perturbation Scenario": description,
                "1. Baseline Raw (HSV+LBP+HOG)": round(f1_raw, 4),
                "2. Unconditional CLAHE": round(f1_clahe, 4),
                "3. Conditional Blur Control": round(f1_blur, 4),
                "4. Proposed Quality-Gated": round(f1_prop, 4),
                "Gain vs Raw": round(f1_prop - f1_raw, 4),
                "Selected view": gated_sys.selected_view_mode,
            }
        )

    result = pd.DataFrame(rows)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result.to_csv(
            os.path.join(output_dir, "Table5_MultiSeverity_Robustness.csv"), index=False
        )
    return result


def benchmark_proposed_pipeline_cpu(
    df_all: pd.DataFrame,
    gated_sys: CalibratedConfidenceGatedSystem,
    output_dir: str = None,
    seed: int = SEED,
):
    """Profile per-stage CPU latency and memory footprint for the proposed pipeline."""
    te_sub = df_all[df_all["split_group"] == "Test_Orig"].reset_index(drop=True)
    use_seg = getattr(gated_sys, "selected_view_mode", "adaptive") == "segmented"

    kb_s1, kb_s2 = 480.2, 4020.4
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        p_s1 = os.path.join(output_dir, "proposed_stage1_lr.joblib")
        p_s2 = os.path.join(output_dir, "proposed_twostage_gated.joblib")
        joblib.dump(
            {
                "compressor": gated_sys.compressor,
                "lr": gated_sys.base_lr,
                "temperature": gated_sys.temperature_,
                "feature_cache_version": FEATURE_CACHE_VERSION,
                "selected_view_mode": gated_sys.selected_view_mode,
            },
            p_s1,
            compress=3,
        )
        joblib.dump(gated_sys, p_s2, compress=3)
        kb_s1 = os.path.getsize(p_s1) / 1024.0
        kb_s2 = os.path.getsize(p_s2) / 1024.0

    proc = psutil.Process(os.getpid())
    psutil.cpu_percent(interval=None)
    t_io_l, t_pre_l, t_feat_l, t_s1_l, t_gate_l, t_tot_l = [], [], [], [], [], []
    y_eval, preds_s1, preds_gated = [], [], []

    for row in tqdm(
        te_sub.itertuples(index=False),
        total=len(te_sub),
        desc=f"CPU latency — {gated_sys.selected_view_mode}",
    ):
        path = str(row.file_path)
        y_true = int(row.label_id)
        t_start = time.perf_counter()

        t0 = time.perf_counter()
        image = read_and_standardize_rgb(path)
        t_io = (time.perf_counter() - t0) * 1000.0
        if image is None:
            continue

        t0 = time.perf_counter()
        if use_seg:
            image, _, fg_mask = segment_plant_organ_roi(image)
            image_prepared = apply_adaptive_quality_preprocessing(
                image, foreground_mask=fg_mask
            )
        else:
            image_prepared = apply_adaptive_quality_preprocessing(
                image, foreground_mask=None
            )
        t_pre = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        f_hsv, f_lab, f_lbp, f_patch5, f_hog, f_q, f_mgct = extract_fast_all_branches(
            image_prepared
        )
        x_raw = (
            np.concatenate([f_hsv, f_lab, f_lbp, f_patch5, f_q, f_mgct, f_hog])
            .reshape(1, -1)
            .astype(np.float32)
        )
        x_scaled = gated_sys.compressor.transform(x_raw)
        t_feat = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        p_stage1 = gated_sys._stage1_probs(x_scaled)
        t_s1 = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        p_final, _, _ = gated_sys.predict_gated_probs_from_scaled(
            x_scaled, p1=p_stage1
        )
        t_gate = (time.perf_counter() - t0) * 1000.0
        t_total = (time.perf_counter() - t_start) * 1000.0

        y_eval.append(y_true)
        preds_s1.append(int(np.argmax(p_stage1[0])))
        preds_gated.append(int(np.argmax(p_final[0])))
        t_io_l.append(t_io)
        t_pre_l.append(t_pre)
        t_feat_l.append(t_feat)
        t_s1_l.append(t_s1)
        t_gate_l.append(t_gate)
        t_tot_l.append(t_total)

    y_eval = np.asarray(y_eval, dtype=np.int32)
    preds_s1 = np.asarray(preds_s1, dtype=np.int32)
    preds_gated = np.asarray(preds_gated, dtype=np.int32)
    cpu_pct = psutil.cpu_percent(interval=None)
    rss_mb = proc.memory_info().rss / (1024.0 * 1024.0)

    _, s1_lo, s1_hi = bootstrap_macro_f1_ci(y_eval, preds_s1, n_boot=1000, seed=seed)
    _, g_lo, g_hi = bootstrap_macro_f1_ci(y_eval, preds_gated, n_boot=1000, seed=seed)
    stage1_latency = (
        np.asarray(t_io_l)
        + np.asarray(t_pre_l)
        + np.asarray(t_feat_l)
        + np.asarray(t_s1_l)
    )

    row_s1 = {
        "Model / Pipeline": "Proposed Stage-1 Only (HSV/LAB+LBP+5Patch+MGCT+HOG-PCA -> LR)",
        "Params / Disk Size": f"<0.02M ({kb_s1:.1f} KB)",
        "Group-Split Acc (%)": round(accuracy_score(y_eval, preds_s1) * 100.0, 2),
        "Group-Split Macro-F1": round(
            f1_score(y_eval, preds_s1, average="macro", zero_division=0), 4
        ),
        "95% Bootstrap CI": f"[{s1_lo:.4f}, {s1_hi:.4f}]",
        "End-to-End Mean (ms/img)": round(float(np.mean(stage1_latency)), 2),
        "End-to-End p50 (ms)": round(float(np.percentile(stage1_latency, 50)), 2),
        "End-to-End p95 (ms)": round(float(np.percentile(stage1_latency, 95)), 2),
        "Throughput (img/s)": round(
            1000.0 / max(float(np.mean(stage1_latency)), 1e-9), 1
        ),
        "Process RSS RAM (MB)": round(rss_mb, 1),
    }
    row_gated = {
        "Model / Pipeline": f"Proposed Two-Stage Confidence-Gated (tau={gated_sys.best_tau_gate:.2f})",
        "Params / Disk Size": f"<0.05M ({kb_s2:.1f} KB)",
        "Group-Split Acc (%)": round(accuracy_score(y_eval, preds_gated) * 100.0, 2),
        "Group-Split Macro-F1": round(
            f1_score(y_eval, preds_gated, average="macro", zero_division=0), 4
        ),
        "95% Bootstrap CI": f"[{g_lo:.4f}, {g_hi:.4f}]",
        "End-to-End Mean (ms/img)": round(float(np.mean(t_tot_l)), 2),
        "End-to-End p50 (ms)": round(float(np.percentile(t_tot_l, 50)), 2),
        "End-to-End p95 (ms)": round(float(np.percentile(t_tot_l, 95)), 2),
        "Throughput (img/s)": round(1000.0 / max(float(np.mean(t_tot_l)), 1e-9), 1),
        "Process RSS RAM (MB)": round(rss_mb, 1),
    }
    breakdown = {
        "Input decode + resize (ms)": round(float(np.mean(t_io_l)), 3),
        "Segmentation + adaptive preprocess (ms)": round(float(np.mean(t_pre_l)), 3),
        "Feature extraction + Block-PCA (ms)": round(float(np.mean(t_feat_l)), 3),
        "Stage-1 LR inference (ms)": round(float(np.mean(t_s1_l)), 3),
        "Gated decision including Stage-2 if routed (ms)": round(
            float(np.mean(t_gate_l)), 3
        ),
        "CPU utilization snapshot (%)": round(float(cpu_pct), 1),
        "Stage-1 artifact size (KB)": round(kb_s1, 1),
        "Two-stage artifact size (KB)": round(kb_s2, 1),
        "Measured samples": len(y_eval),
        "Selected view mode": gated_sys.selected_view_mode,
        "Selected gate threshold": gated_sys.best_tau_gate,
        "HOG backend": HOG_BACKEND,
    }
    return row_s1, row_gated, breakdown


def run_external_dataset_validation(
    ext_root: str,
    main_feat_dict: dict,
    main_class_names: list,
    track_name: str = "MANGO",
    output_dir: str = None,
    source_view_mode: str = "adaptive",
    seed: int = SEED,
):
    """
    Evaluate Cross-Dataset Generalization (MangoLeafBD -> MLDID, N=3,000, 5 shared classes):
      - Protocol A: Source-view Zero-Shot
      - Protocol B: Canonical-pose Zero-Shot (PCA-axis horizontal alignment + foreground L* constancy)
      - Protocol C: Transductive Block-Shrinkage CORAL
    """
    if ext_root is None or not os.path.isdir(ext_root):
        return None

    ext_records, _ = scan_dataset_records(ext_root, track_name, is_external=True)
    df_ext = (
        pd.DataFrame(ext_records)
        .drop_duplicates(subset=["file_path"])
        .reset_index(drop=True)
        if ext_records
        else pd.DataFrame()
    )
    if df_ext.empty:
        return None

    main_label_names = np.asarray(
        [main_class_names[int(i)] for i in main_feat_dict["labels"]], dtype=object
    )
    shared_classes = sorted(
        set(main_label_names).intersection(set(df_ext["label_name"].unique()))
    )
    shared_map = {name: idx for idx, name in enumerate(shared_classes)}
    source_shared = np.isin(main_label_names, shared_classes)
    split_names = main_feat_dict["splits"]
    train_mask = (split_names == "Train_Orig") & source_shared
    val_mask = (split_names == "Val_Orig") & source_shared
    test_mask = (split_names == "Test_Orig") & source_shared

    y_train = np.asarray(
        [shared_map[n] for n in main_label_names[train_mask]], dtype=np.int32
    )
    y_val = np.asarray(
        [shared_map[n] for n in main_label_names[val_mask]], dtype=np.int32
    )
    y_test = np.asarray(
        [shared_map[n] for n in main_label_names[test_mask]], dtype=np.int32
    )

    rows = []
    workers = 4
    branches = ["hsv", "lab", "lbp", "patch5", "q", "mgct", "hog"]
    view_mode = (
        source_view_mode if source_view_mode in {"adaptive", "segmented"} else "adaptive"
    )
    X_train_a, _ = assemble_matrix_by_mask(
        main_feat_dict, train_mask, view_mode, branches
    )
    X_val_a, _ = assemble_matrix_by_mask(main_feat_dict, val_mask, view_mode, branches)
    X_test_a, _ = assemble_matrix_by_mask(
        main_feat_dict, test_mask, view_mode, branches
    )
    system_a = CalibratedConfidenceGatedSystem(
        hog_dim=1764, n_hog_pca=64
    ).fit_and_calibrate(X_train_a, y_train, X_val_a, y_val)

    ext_paths = [str(r.file_path) for r in df_ext.itertuples(index=False)]
    ext_names = [str(r.label_name) for r in df_ext.itertuples(index=False)]
    X_ext_a, y_ext_a = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for vec, name in tqdm(
            zip(
                pool.map(
                    lambda p: compose_feature_vector(
                        p, canonical=False, view_mode=view_mode
                    ),
                    ext_paths,
                ),
                ext_names,
            ),
            total=len(ext_paths),
            desc="MLDID source-view features",
        ):
            if vec is not None and name in shared_map:
                X_ext_a.append(vec)
                y_ext_a.append(shared_map[name])

    if X_ext_a:
        X_ext_a = np.asarray(X_ext_a, dtype=np.float32)
        y_ext_a = np.asarray(y_ext_a, dtype=np.int32)
        p_in, _, _ = system_a.predict_gated_probs(X_test_a)
        p_ext_a, _, _ = system_a.predict_gated_probs(X_ext_a)
        pred_in = np.argmax(p_in, axis=1)
        pred_ext_a = np.argmax(p_ext_a, axis=1)
        _, in_lo, in_hi = bootstrap_macro_f1_ci(y_test, pred_in, n_boot=1000, seed=seed)
        _, ext_a_lo, ext_a_hi = bootstrap_macro_f1_ci(
            y_ext_a, pred_ext_a, n_boot=1000, seed=seed
        )
        rows.extend(
            [
                {
                    "Protocol": "Source-view in-domain Group-Split Test",
                    "Adaptation": "None; source-only training",
                    "Samples": len(y_test),
                    "Shared classes": len(shared_classes),
                    "Accuracy (%)": round(accuracy_score(y_test, pred_in) * 100, 2),
                    "Balanced Acc (%)": round(
                        balanced_accuracy_score(y_test, pred_in) * 100, 2
                    ),
                    "Macro-F1": round(
                        f1_score(y_test, pred_in, average="macro", zero_division=0), 4
                    ),
                    "95% Bootstrap CI": f"[{in_lo:.4f}, {in_hi:.4f}]",
                    "Source view": view_mode,
                },
                {
                    "Protocol": "MLDID source-view Zero-Shot",
                    "Adaptation": "No target statistics used; source-trained shared-label model",
                    "Samples": len(y_ext_a),
                    "Shared classes": len(shared_classes),
                    "Accuracy (%)": round(
                        accuracy_score(y_ext_a, pred_ext_a) * 100, 2
                    ),
                    "Balanced Acc (%)": round(
                        balanced_accuracy_score(y_ext_a, pred_ext_a) * 100, 2
                    ),
                    "Macro-F1": round(
                        f1_score(y_ext_a, pred_ext_a, average="macro", zero_division=0),
                        4,
                    ),
                    "95% Bootstrap CI": f"[{ext_a_lo:.4f}, {ext_a_hi:.4f}]",
                    "Source view": view_mode,
                },
            ]
        )

    source_indices = np.flatnonzero(source_shared)
    source_paths = [str(main_feat_dict["file_paths"][i]) for i in source_indices]
    source_splits = split_names[source_indices]
    source_labels_names = main_label_names[source_indices]
    canonical_rows, canonical_splits, canonical_y = [], [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for vec, split, name in tqdm(
            zip(
                pool.map(
                    lambda p: compose_feature_vector(p, canonical=True), source_paths
                ),
                source_splits,
                source_labels_names,
            ),
            total=len(source_paths),
            desc="Canonical source features",
        ):
            if vec is not None:
                canonical_rows.append(vec)
                canonical_splits.append(str(split))
                canonical_y.append(shared_map[str(name)])

    X_src = np.asarray(canonical_rows, dtype=np.float32)
    canonical_splits = np.asarray(canonical_splits, dtype=object)
    canonical_y = np.asarray(canonical_y, dtype=np.int32)
    c_train = canonical_splits == "Train_Orig"
    c_val = canonical_splits == "Val_Orig"
    c_test = canonical_splits == "Test_Orig"
    y_ctr, y_cval, y_ctest = (
        canonical_y[c_train],
        canonical_y[c_val],
        canonical_y[c_test],
    )
    canonical_system = CalibratedConfidenceGatedSystem(
        hog_dim=1764, n_hog_pca=64
    ).fit_and_calibrate(X_src[c_train], y_ctr, X_src[c_val], y_cval)

    target_vecs, target_y = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for vec, name in tqdm(
            zip(
                pool.map(lambda p: compose_feature_vector(p, canonical=True), ext_paths),
                ext_names,
            ),
            total=len(ext_paths),
            desc="Canonical MLDID features",
        ):
            if vec is not None and name in shared_map:
                target_vecs.append(vec)
                target_y.append(shared_map[name])

    if target_vecs:
        X_ext_c = np.asarray(target_vecs, dtype=np.float32)
        y_ext_c = np.asarray(target_y, dtype=np.int32)
        p_source_c, _, _ = canonical_system.predict_gated_probs(X_src[c_test])
        p_ext_c, _, _ = canonical_system.predict_gated_probs(X_ext_c)
        pred_source_c = np.argmax(p_source_c, axis=1)
        pred_ext_c = np.argmax(p_ext_c, axis=1)
        _, source_c_lo, source_c_hi = bootstrap_macro_f1_ci(
            y_ctest, pred_source_c, n_boot=1000, seed=seed + 1
        )
        _, ext_c_lo, ext_c_hi = bootstrap_macro_f1_ci(
            y_ext_c, pred_ext_c, n_boot=1000, seed=seed + 1
        )
        rows.extend(
            [
                {
                    "Protocol": "Canonical-pose in-domain Group-Split Test",
                    "Adaptation": f"Same source/target transform; color mode={CANONICAL_COLOR_MODE}",
                    "Samples": len(y_ctest),
                    "Shared classes": len(shared_classes),
                    "Accuracy (%)": round(
                        accuracy_score(y_ctest, pred_source_c) * 100, 2
                    ),
                    "Balanced Acc (%)": round(
                        balanced_accuracy_score(y_ctest, pred_source_c) * 100, 2
                    ),
                    "Macro-F1": round(
                        f1_score(
                            y_ctest, pred_source_c, average="macro", zero_division=0
                        ),
                        4,
                    ),
                    "95% Bootstrap CI": f"[{source_c_lo:.4f}, {source_c_hi:.4f}]",
                    "Source view": "canonical_pose",
                },
                {
                    "Protocol": "MLDID Canonical-pose Zero-Shot",
                    "Adaptation": f"No target stats used; PCA-axis pose + foreground L* mode={CANONICAL_COLOR_MODE}",
                    "Samples": len(y_ext_c),
                    "Shared classes": len(shared_classes),
                    "Accuracy (%)": round(
                        accuracy_score(y_ext_c, pred_ext_c) * 100, 2
                    ),
                    "Balanced Acc (%)": round(
                        balanced_accuracy_score(y_ext_c, pred_ext_c) * 100, 2
                    ),
                    "Macro-F1": round(
                        f1_score(
                            y_ext_c, pred_ext_c, average="macro", zero_division=0
                        ),
                        4,
                    ),
                    "95% Bootstrap CI": f"[{ext_c_lo:.4f}, {ext_c_hi:.4f}]",
                    "Source view": "canonical_pose",
                },
            ]
        )

        if len(X_ext_c) >= 2:
            Z_source_train = canonical_system.compressor.transform(X_src[c_train])
            Z_target = canonical_system.compressor.transform(X_ext_c)
            Z_target_coral = coral_align_features(
                Z_source_train, Z_target, reg=0.35, anchor=0.20
            )
            p_coral, _, _ = canonical_system.predict_gated_probs_from_scaled(
                Z_target_coral
            )
            pred_coral = np.argmax(p_coral, axis=1)
            _, coral_lo, coral_hi = bootstrap_macro_f1_ci(
                y_ext_c, pred_coral, n_boot=1000, seed=seed + 2
            )
            rows.append(
                {
                    "Protocol": "MLDID Canonical-pose + CORAL",
                    "Adaptation": "Transductive; unsupervised covariance alignment on compact features",
                    "Samples": len(y_ext_c),
                    "Shared classes": len(shared_classes),
                    "Accuracy (%)": round(accuracy_score(y_ext_c, pred_coral) * 100, 2),
                    "Balanced Acc (%)": round(
                        balanced_accuracy_score(y_ext_c, pred_coral) * 100, 2
                    ),
                    "Macro-F1": round(
                        f1_score(y_ext_c, pred_coral, average="macro", zero_division=0),
                        4,
                    ),
                    "95% Bootstrap CI": f"[{coral_lo:.4f}, {coral_hi:.4f}]",
                    "Source view": "canonical_pose + CORAL",
                }
            )

    result = pd.DataFrame(rows)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        result.to_csv(
            os.path.join(output_dir, "Table7_External_Dataset_Validation.csv"),
            index=False,
        )
    return result
