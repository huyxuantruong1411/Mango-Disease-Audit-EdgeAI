"""
Energy-Balanced Block-PCA Compressor, Calibrated Two-Stage Confidence-Gated
Classifier, and Transductive Block-Shrinkage CORAL (`src/mango_edge_cv/models.py`).
"""

import numpy as np
from scipy.special import softmax as scipy_softmax
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import SEED


def compute_ece_and_aurc(y_true: np.ndarray, probs_2d: np.ndarray, n_bins: int = 10):
    """
    Compute Expected Calibration Error (ECE, 10 equal-width bins) and
    Area Under the Risk-Coverage Curve (AURC) for selective classification.
    """
    y_true = np.asarray(y_true)
    probs_2d = np.asarray(probs_2d)
    confidences = np.max(probs_2d, axis=1)
    predictions = np.argmax(probs_2d, axis=1)
    accuracies = (predictions == y_true).astype(np.float32)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b_idx in range(n_bins):
        in_bin = (confidences > bin_boundaries[b_idx]) & (
            confidences <= bin_boundaries[b_idx + 1]
        )
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            ece += (
                np.abs(np.mean(accuracies[in_bin]) - np.mean(confidences[in_bin]))
                * prop_in_bin
            )

    order = np.argsort(-confidences)
    sorted_errors = 1.0 - accuracies[order]
    cum_risks = np.cumsum(sorted_errors) / np.arange(1, len(y_true) + 1)
    aurc = float(np.mean(cum_risks))
    return float(ece), aurc


class BlockPCACompressor:
    """
    Energy-Balanced Block-PCA Compressor (2,003D -> 303D):
      - Splits raw feature vector x in R^2003 into interpretable chromatic-texture block
        x_ct in R^239 (52+57+28+45+7+50) and HOG block x_hog in R^1764.
      - Standardizes x_ct via StandardScaler (x_tilde_ct).
      - Projects standardized x_hog onto top K=64 principal components (z_hog in R^64)
        without whitening tail noise.
      - Scales z_hog by energy-balancing scalar alpha_hog = (RMS(x_tilde_ct) / RMS(z_hog)) * 0.45.
    """

    def __init__(
        self, hog_dim: int = 1764, n_hog_pca: int = 64, hog_weight: float = 0.45
    ):
        self.hog_dim = int(hog_dim)
        self.n_hog_pca = int(n_hog_pca)
        self.hog_weight = float(hog_weight)
        self.scaler_ct = StandardScaler()
        self.scaler_hog = StandardScaler()
        self.pca_hog = None
        self.hog_scale_ = 1.0
        self.n_components_ = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] <= self.hog_dim:
            raise ValueError(
                f"Expected 2-D input with HOG tail of {self.hog_dim} columns; got {X.shape}"
            )
        X_ct, X_hog = X[:, : -self.hog_dim], X[:, -self.hog_dim :]
        ct_s = self.scaler_ct.fit_transform(X_ct).astype(np.float32)
        hog_s = self.scaler_hog.fit_transform(X_hog).astype(np.float32)
        n_comp = min(self.n_hog_pca, max(1, X_hog.shape[0] - 1), X_hog.shape[1])
        self.pca_hog = PCA(n_components=n_comp, random_state=SEED, svd_solver="auto")
        hog_p = self.pca_hog.fit_transform(hog_s).astype(np.float32)
        self.n_components_ = int(n_comp)
        ct_rms = float(np.sqrt(np.mean(np.square(ct_s)))) + 1e-6
        hog_rms = float(np.sqrt(np.mean(np.square(hog_p)))) + 1e-6
        self.hog_scale_ = (ct_rms / hog_rms) * self.hog_weight
        return np.hstack([ct_s, hog_p * self.hog_scale_]).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X_ct, X_hog = X[:, : -self.hog_dim], X[:, -self.hog_dim :]
        ct_s = self.scaler_ct.transform(X_ct).astype(np.float32)
        hog_s = self.scaler_hog.transform(X_hog).astype(np.float32)
        hog_p = self.pca_hog.transform(hog_s).astype(np.float32)
        return np.hstack([ct_s, hog_p * self.hog_scale_]).astype(np.float32)


class _EstimatorWrapper:
    """Compatibility wrapper used by the feature-weight attribution reporter."""

    def __init__(self, estimator):
        self.estimator = estimator


class CalibratedConfidenceGatedSystem:
    """
    Two-Stage Confidence-Gated Classification System:
      - Stage 1: Multinomial Temperature-Scaled Logistic Regression (T=0.758 on Val_Orig).
      - Confidence Gate: Routes samples with p_max(x) < tau_gate (tau_gate=0.65) to Stage 2.
      - Stage 2: Nonlinear Specialist Ensemble (0.55 * RBF-SVM + 0.30 * Shallow-MLP + 0.15 * Cosine-KNN),
        blended with Stage-1 posterior as P_final = 0.30 * P_1 + 0.70 * P_2.
    """

    def __init__(
        self,
        hog_dim: int = 1764,
        n_hog_pca: int = 64,
        route_penalty: float = 0.0002,
    ):
        self.compressor = BlockPCACompressor(hog_dim=hog_dim, n_hog_pca=n_hog_pca)
        self.base_lr = None
        self.calibrated_classifiers_ = []
        self.stage1_lr = self
        self.temperature_ = 1.0
        self.stage2_svm = None
        self.stage2_mlp = None
        self.stage2_knn = None
        self.best_tau_gate = 0.0
        self.selected_view_mode = "adaptive"
        self.route_penalty = float(route_penalty)
        self.classes_ = None

    def _stage1_logits(self, X_scaled: np.ndarray) -> np.ndarray:
        scores = np.asarray(self.base_lr.decision_function(X_scaled))
        if scores.ndim == 1:
            scores = np.column_stack([np.zeros_like(scores), scores])
        return scores

    def _stage1_probs(self, X_scaled: np.ndarray) -> np.ndarray:
        logits = self._stage1_logits(X_scaled)
        probs = scipy_softmax(logits / max(float(self.temperature_), 1e-4), axis=1)
        return np.asarray(probs, dtype=np.float64)

    def fit_and_calibrate(
        self, X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ):
        X_tr_s = self.compressor.fit_transform(X_tr)
        X_val_s = self.compressor.transform(X_val)
        y_tr = np.asarray(y_tr, dtype=np.int32)
        y_val = np.asarray(y_val, dtype=np.int32)
        self.classes_ = np.unique(y_tr)
        if len(self.classes_) < 2:
            raise ValueError("At least two classes are required to fit the classifier.")
        if not set(np.unique(y_val)).issubset(set(self.classes_)):
            raise ValueError("Validation contains a class absent from training.")

        best_c, best_val_f1 = 1.0, -1.0
        for c_value in [0.02, 0.05, 0.15, 0.5, 1.0, 2.5]:
            candidate = LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=1500,
                solver="lbfgs",
                random_state=SEED,
            ).fit(X_tr_s, y_tr)
            val_pred = candidate.predict(X_val_s)
            score = f1_score(y_val, val_pred, average="macro", zero_division=0)
            if score > best_val_f1:
                best_val_f1, best_c = score, c_value

        self.base_lr = LogisticRegression(
            C=best_c,
            class_weight="balanced",
            max_iter=1500,
            solver="lbfgs",
            random_state=SEED,
        ).fit(X_tr_s, y_tr)
        self.calibrated_classifiers_ = [_EstimatorWrapper(self.base_lr)]
        self.stage1_lr = self

        val_logits = self._stage1_logits(X_val_s)
        best_t, best_nll = 1.0, float("inf")
        for t_candidate in np.geomspace(0.25, 4.0, 61):
            probs = scipy_softmax(val_logits / float(t_candidate), axis=1)
            indices = y_val.astype(int)
            nll = -float(np.mean(np.log(probs[np.arange(len(y_val)), indices] + 1e-12)))
            if nll < best_nll:
                best_nll, best_t = nll, float(t_candidate)
        self.temperature_ = best_t

        self.stage2_svm = SVC(
            C=3.0,
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=SEED,
        ).fit(X_tr_s, y_tr)
        self.stage2_mlp = MLPClassifier(
            hidden_layer_sizes=(128,),
            alpha=0.5,
            max_iter=250,
            early_stopping=True,
            random_state=SEED,
        ).fit(X_tr_s, y_tr)
        self.stage2_knn = KNeighborsClassifier(
            n_neighbors=min(7, len(y_tr)),
            weights="distance",
            metric="cosine",
            n_jobs=1,
        ).fit(X_tr_s, y_tr)

        p1_val = self._stage1_probs(X_val_s)
        base_pred = np.argmax(p1_val, axis=1)
        baseline_f1 = f1_score(y_val, base_pred, average="macro", zero_division=0)
        _, baseline_aurc = compute_ece_and_aurc(y_val, p1_val)
        best_objective = float(baseline_f1 - 0.15 * baseline_aurc)
        best_route_rate = 0.0
        self.best_tau_gate = 0.0
        gate_candidates = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
        for tau in gate_candidates:
            p_final, _, route_rate = self._predict_from_scaled(X_val_s, tau_gate=tau)
            val_pred = np.argmax(p_final, axis=1)
            val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
            _, val_aurc = compute_ece_and_aurc(y_val, p_final)
            objective = float(val_f1 - 0.15 * val_aurc - self.route_penalty * route_rate)
            if (objective > best_objective + 1e-6) or (
                abs(objective - best_objective) <= 1e-6
                and route_rate < best_route_rate - 1e-12
            ):
                best_objective = objective
                best_route_rate = float(route_rate)
                self.best_tau_gate = float(tau)
        return self

    def _predict_from_scaled(self, X_scaled: np.ndarray, tau_gate=None, p1=None):
        X_scaled = np.asarray(X_scaled, dtype=np.float32)
        if len(X_scaled) == 0:
            n_classes = len(self.classes_) if self.classes_ is not None else 0
            empty = np.empty((0, n_classes), dtype=np.float64)
            return empty, empty, 0.0
        if tau_gate is None:
            tau_gate = self.best_tau_gate
        p1 = (
            self._stage1_probs(X_scaled)
            if p1 is None
            else np.asarray(p1, dtype=np.float64)
        )
        if float(tau_gate) <= 0.0:
            return p1, p1, 0.0
        uncertain = np.flatnonzero(np.max(p1, axis=1) < float(tau_gate))
        route_rate = len(uncertain) / max(1, len(X_scaled))
        final_probs = p1.copy()
        if len(uncertain):
            X_uncertain = X_scaled[uncertain]
            p_svm = self.stage2_svm.predict_proba(X_uncertain)
            p_mlp = self.stage2_mlp.predict_proba(X_uncertain)
            p_knn = self.stage2_knn.predict_proba(X_uncertain)
            p_stage2 = 0.55 * p_svm + 0.30 * p_mlp + 0.15 * p_knn
            final_probs[uncertain] = 0.30 * p1[uncertain] + 0.70 * p_stage2
            final_probs[uncertain] /= (
                final_probs[uncertain].sum(axis=1, keepdims=True) + 1e-12
            )
        return final_probs, p1, float(route_rate)

    def predict_gated_probs_from_scaled(
        self, X_scaled: np.ndarray, tau_gate=None, p1=None
    ):
        return self._predict_from_scaled(X_scaled, tau_gate=tau_gate, p1=p1)

    def predict_gated_probs(self, X_raw: np.ndarray, tau_gate=None):
        X_scaled = self.compressor.transform(X_raw)
        return self._predict_from_scaled(X_scaled, tau_gate=tau_gate)

    def predict_proba(self, X_raw: np.ndarray) -> np.ndarray:
        return self._stage1_probs(self.compressor.transform(X_raw))


def coral_align_features(
    X_src: np.ndarray, X_tgt: np.ndarray, reg: float = 0.35, anchor: float = 0.20
) -> np.ndarray:
    """
    Transductive Block-Shrinkage CORAL:
      1) Coordinate-wise empirical quantile matching;
      2) Block-diagonal covariance shrinkage and alignment within each semantic branch
         [0, 52, 109, 137, 182, 189, 239, 303] without cross-branch mixing.
    """
    X_src = np.asarray(X_src, dtype=np.float64)
    X_tgt = np.asarray(X_tgt, dtype=np.float64)
    if X_src.ndim != 2 or X_tgt.ndim != 2 or X_src.shape[1] != X_tgt.shape[1]:
        raise ValueError(
            f"CORAL expects matching 2-D feature matrices, got {X_src.shape} and {X_tgt.shape}"
        )
    n_s, d = X_src.shape
    n_t = X_tgt.shape[0]
    quantile_pct = np.linspace(0.0, 100.0, min(201, n_s, n_t))
    X_q = np.empty_like(X_tgt)

    for j in range(d):
        src_q = np.percentile(X_src[:, j], quantile_pct)
        tgt_q = np.percentile(X_tgt[:, j], quantile_pct)
        tgt_unique, first_idx = np.unique(tgt_q, return_index=True)
        src_at_unique = src_q[first_idx]
        if len(tgt_unique) >= 3:
            X_q[:, j] = np.interp(X_tgt[:, j], tgt_unique, src_at_unique)
        else:
            mu_s, mu_t = float(np.mean(X_src[:, j])), float(np.mean(X_tgt[:, j]))
            sd_s = max(float(np.std(X_src[:, j])), 1e-4)
            sd_t = max(float(np.std(X_tgt[:, j])), 1e-4)
            X_q[:, j] = (X_tgt[:, j] - mu_t) * (sd_s / sd_t) + mu_s

    if d >= 303:
        boundaries = [0, 52, 109, 137, 182, 189, 239, d]
    else:
        boundaries = sorted(
            set([0, min(52, d), min(109, d), min(137, d), min(182, d), min(189, d), d])
        )

    blocks = [(a, b) for a, b in zip(boundaries[:-1], boundaries[1:]) if b > a]
    X_out = X_q.copy()
    eps = 1e-4

    def sym_power(mat, power):
        mat = (mat + mat.T) * 0.5
        vals, vecs = np.linalg.eigh(mat)
        vals = np.maximum(vals, eps)
        return (vecs * (vals**power)) @ vecs.T

    for left, right in blocks:
        S = X_src[:, left:right]
        T = X_q[:, left:right]
        mu_s = np.mean(S, axis=0, keepdims=True)
        mu_t = np.mean(T, axis=0, keepdims=True)
        Zs = S - mu_s
        Zt = T - mu_t
        width = right - left
        if width <= 1 or min(len(S), len(T)) <= 2:
            X_out[:, left:right] = Zt + mu_s
            continue
        cov_s = (Zs.T @ Zs) / max(1, len(S) - 1)
        cov_t = (Zt.T @ Zt) / max(1, len(T) - 1)
        cov_s = (1.0 - reg) * cov_s + reg * np.diag(np.diag(cov_s)) + eps * np.eye(width)
        cov_t = (1.0 - reg) * cov_t + reg * np.diag(np.diag(cov_t)) + eps * np.eye(width)
        coral_transform = sym_power(cov_t, -0.5) @ sym_power(cov_s, 0.5)
        transform = (1.0 - anchor) * np.eye(width) + anchor * coral_transform
        X_out[:, left:right] = Zt @ transform + mu_s

    return X_out.astype(np.float32)

