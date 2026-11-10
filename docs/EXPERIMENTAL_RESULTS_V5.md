# Locked Version 5 (`v5`) Experimental Results and Statistical Analysis

All experimental metrics in this repository are locked to **Version 5 (`v5`, `SEED = 20261022`)** and verified automatically by [`scripts/verify_locked_results.py`](../scripts/verify_locked_results.py) (`57/57` ground-truth checks passing).

---

## 1. Specimen-Identity Leakage Audit and Background Shortcut Quantification (Tables 1 & 2)

- **Specimen-Cluster Audit (Table 1):** Across 4,000 images in `MangoLeafBD` (500 per class), dual-criterion graph clustering ($d_{\text{Hamming}}(\text{pHash}) \le 6 \lor \cos_{\text{HS}} \ge 0.992$) identifies **2,554 unique specimen clusters** and **1,446 near-duplicate/rotated siblings (36.15%)**. Under a naive stratified random split, **85 / 218 (39.0%) of multi-image clusters leak across train and test sets**, whereas `StratifiedGroupKFold` enforces **0.0% leakage** (2,795 train / 595 validation / 610 test images).

- **Five-View Shortcut Audit (Table 2):** Under the leak-free group split (12.50% random chance):
  - **Whole ROI:** achieves **98.33% Acc** (0.9833 Macro-F1).
  - **Segmented Organ Only:** achieves **94.84% Acc** (0.9440 Macro-F1).
  - **Background Only (leaf masked out):** achieves **94.84% Acc** (0.9396 Macro-F1) — matching leaf organ accuracy without a single leaf pixel!
  - **Four $32\times32$ Corner Patches:** achieve **85.08% Acc** (0.8332 Macro-F1) — predicting class at $6.8\times$ random chance.
  - **7D File Metadata:** achieves **59.14% Acc** (0.5742 Macro-F1) — predicting class at $4.7\times$ chance from file size and aspect ratio alone.

<div align="center">
  <img src="../results/figures/Fig03_Cluster_Distribution_and_Split_Leakage.png" width="94%" alt="Figure 3: Cluster Distribution and Leakage" />
  <p><em>Figure 3: Class-wise unique specimen clusters versus near-duplicate siblings across all 4,000 images.</em></p>
</div>

<div align="center">
  <img src="../results/figures/Fig04_FiveView_Background_Shortcut_VisualProof.png" width="94%" alt="Figure 4: Five-View Shortcut Audit" />
  <p><em>Figure 4: Five-view visual and quantitative shortcut audit demonstrating severe background confounding.</em></p>
</div>

---

## 2. Progressive 14-Step Ablation and Block-PCA Recovery (Table 3)

- **Dimensionality Dilution ($M_2 \to M_4 \to M_9$):** Compact HSV + LBP ($M_2$, 80D) achieves **97.35% Acc** (0.9739 Macro-F1). Appending raw 1,764D HOG ($M_4$, 1,844D) dilutes chromatic weights and drops accuracy to **86.61%** (0.8453 Macro-F1). Adding CIE-LAB, 5-Patch, Quality Gate, Quality Vector $q$, and MGCT in raw 2,003D space ($M_9$) recovers accuracy only to **92.05%** (0.9236 Macro-F1).

- **Energy-Balanced Block-PCA Recovery ($M_9 \to M_{10}$):** Compressing 1,764D HOG to 64D PCA and balancing RMS energy ($\gamma = 0.45 \to 303\text{D}$) drives a **+5.86% accuracy jump** to **97.91% Acc** (0.9789 Macro-F1, 95% CI $[0.9656, 0.9891]$, exact McNemar $p < 0.0001$).

- **Pure Biological Organ-Only Benchmark ($M_{11}$):** Evaluating $M_{10}$ strictly on segmented leaf organs with studio background zeroed achieves **96.37% Acc** (0.9549 Macro-F1).

<div align="center">
  <img src="../results/figures/Fig09_Progressive_Ablation_Trajectory_M1_M11.png" width="94%" alt="Figure 9: Progressive Ablation Trajectory" />
  <p><em>Figure 9: Progressive 14-step ablation trajectory from single-branch HSV to Energy-Balanced Block-PCA.</em></p>
</div>

---

## 3. Temperature-Calibrated Two-Stage Confidence-Gated Routing (Table 4)

- **Stage-1 Alone ($\tau_{\text{gate}} = 0.00$):** 0.0% routed, **97.91% Acc**, **0.9789 Macro-F1**, $\text{ECE} = 0.0094$, $\text{AURC} = 0.00164$.
- **Validation-Selected Operating Point ($\tau_{\text{gate}} = 0.65$):** Routes only **1.53%** of test samples to Stage-2, achieving **98.47% Acc**, **0.9836 Macro-F1** (95% CI $[0.9712, 0.9932]$), $\text{ECE} = 0.0104$, and $\text{AURC} = 0.00158$.
- **High-Sensitivity Operating Point ($\tau_{\text{gate}} = 0.85$):** Routes **4.32%** of test samples to Stage-2, achieving **98.74% Acc**, **0.9859 Macro-F1** (95% CI $[0.9744, 0.9950]$), and $\text{AURC} = 0.00159$.

<div align="center">
  <img src="../results/figures/Fig11_Gate_Threshold_Sweep_and_Calibration.png" width="94%" alt="Figure 11: Gate Threshold Sweep and Calibration" />
  <p><em>Figure 11: Stage-2 routing rate versus accuracy/Macro-F1, reliability calibration (ECE), and selective risk (AURC).</em></p>
</div>

---

## 4. Multi-Severity Perturbation Robustness and Single-Core CPU Profiling (Tables 5 & 6)

- **Multi-Severity Perturbations (Table 5):** Across 7 degradations, Proposed Quality-Gated preprocessing achieves **0.9836 Macro-F1** on clean images ($\sigma=0$), **0.9249** at $\sigma=5$ (+0.1461 vs. raw), **0.8647** at $\sigma=10$ (**+0.3533 vs. raw**; +0.4440 vs. unconditional CLAHE 0.4207), **0.6104** at $\sigma=20$ (+0.4090 vs. raw), **0.9812** at glare $r=22\text{px}$, **0.9770** at glare $r=35\text{px}$, and **0.8012** under $-40\%$ low light (+0.0767 vs. raw).

- **Single-Core CPU Latency (Table 6):** Stage-1 executes in **21.58 ms** (46.3 FPS, 480.2 KB) and Two-Stage Gated executes in **21.74 ms** (46.0 FPS, 4.08 MB) — outperforming `MobileNetV3-Small` (48.70 ms, 9.80 MB), `MobileNetV2` (56.42 ms, 13.60 MB), and `ResNet50` (272.04 ms, 97.80 MB).

<div align="center">
  <img src="../results/figures/Fig12_MultiSeverity_Perturbation_Robustness.png" width="94%" alt="Figure 12: Multi-Severity Perturbation Robustness" />
  <p><em>Figure 12: Macro-F1 resilience across noise, specular glare, and low-light underexposure.</em></p>
</div>

<div align="center">
  <img src="../results/figures/Fig13_CPU_Latency_Breakdown_and_Edge_Footprint.png" width="94%" alt="Figure 13: CPU Latency and Footprint" />
  <p><em>Figure 13: End-to-end CPU runtime breakdown across pipeline stages and on-disk model footprint.</em></p>
</div>

---

## 5. External Cross-Dataset Generalization on `MLDID` ($N = 3,000$, Table 7)

- Evaluating on 3,000 external images from India (`MLDID`, 5 shared classes), **Eigenvector Canonical Leaf Pose Normalization** improves zero-shot accuracy from **31.27% (0.2886 Macro-F1) to 42.63% (0.4395 Macro-F1)** (+11.36% accuracy and +15.09% Macro-F1 gain), while achieving **96.04%** canonical-pose and **99.34%** source-view in-domain 5-class accuracy on `MangoLeafBD` ($N=455$).

<div align="center">
  <img src="../results/figures/Fig14_CrossDataset_Canonical_Pose_and_CORAL.png" width="94%" alt="Figure 14: Canonical Leaf Pose on External MLDID" />
  <p><em>Figure 14: Eigenvector canonical leaf pose normalization and zero-shot transfer comparison on external MLDID field images.</em></p>
</div>
