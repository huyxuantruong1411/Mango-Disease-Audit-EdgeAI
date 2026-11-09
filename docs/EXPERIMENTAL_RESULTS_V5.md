# Locked Version 5 (`v5`) Experimental Results and Statistical Analysis

All experimental metrics in this repository are locked to **Version 5 (`v5`, `SEED = 20261022`)** and verified automatically by [`scripts/verify_locked_results.py`](../scripts/verify_locked_results.py) (`57/57` ground-truth checks passing).

---

## 1. Specimen-Identity Leakage Audit and Background Shortcut Quantification (Tables 1 & 2)

- **Specimen-Cluster Audit (`Table 1`):** Across $4,000$ images in `MangoLeafBD` ($500$ per class), dual-criterion graph clustering ($d_{\text{Hamming}}(\text{pHash}) \le 6 \lor \cos_{\text{HS}} \ge 0.992$) identifies **$2,554$ unique specimen clusters** and **$1,446$ near-duplicate/rotated siblings ($36.15\%$)**. Under a naive stratified random split, **$85 / 218$ ($39.0\%$) of multi-image clusters leak across train and test sets**, whereas `StratifiedGroupKFold` enforces **$0.0\%$ leakage** ($2,795$ train / $595$ validation / $610$ test images).
- **Five-View Shortcut Audit (`Table 2`):** Under the leak-free group split ($12.50\%$ random chance), Whole ROI achieves **$98.33\%$ Acc** ($0.9833$ F1), Segmented Organ Only achieves **$94.84\%$ Acc** ($0.9440$ F1), **Background Only (leaf masked out) achieves $94.84\%$ Acc ($0.9396$ F1)**, Four $32\times32$ Corner Patches achieve **$85.08\%$ Acc** ($0.8332$ F1), and $7\text{D}$ File Metadata achieves **$59.14\%$ Acc** ($0.5742$ F1).

![Figure 3: Class-Wise Unique Specimen Clusters vs. Near-Duplicate Siblings](../results/figures/Fig03_Cluster_Distribution_and_Split_Leakage.png)

![Figure 4: Five-View Studio Background Shortcut Visual Proof](../results/figures/Fig04_FiveView_Background_Shortcut_VisualProof.png)

---

## 2. Progressive 14-Step Ablation and Block-PCA Recovery (Table 3)

- **Dimensionality Dilution (`M2` $\to$ `M4` $\to$ `M9`):** Compact HSV + LBP ($M_2$, $80\text{D}$) achieves **$97.35\%$ Acc** ($0.9739$ Macro-F1). Appending raw $1,764\text{D}$ HOG ($M_4$, $1,844\text{D}$) dilutes chromatic weights and drops accuracy to **$86.61\%$** ($0.8453$ Macro-F1). Adding CIE-LAB, 5-Patch, Quality Gate, Quality Vector $q$, and MGCT in raw $2,003\text{D}$ space ($M_9$) recovers accuracy only to **$92.05\%$** ($0.9236$ Macro-F1).
- **Energy-Balanced Block-PCA Recovery (`M9` $\to$ `M10`):** Compressing $1,764\text{D}$ HOG to $64\text{D}$ PCA and balancing RMS energy ($\gamma = 0.45 \to 303\text{D}$) drives a **+5.86% accuracy jump** to **$97.91\%$ Acc** ($0.9789$ Macro-F1, $95\%$ CI $[0.9656, 0.9891]$, exact McNemar $p < 0.0001$).
- **Pure Biological Organ-Only Benchmark (`M11`):** Evaluating $M_{10}$ strictly on segmented leaf organs with studio background zeroed achieves **$96.37\%$ Acc** ($0.9549$ Macro-F1).

![Figure 9: Progressive 14-Step Ablation Trajectory (M1 to M11)](../results/figures/Fig09_Progressive_Ablation_Trajectory_M1_M11.png)

---

## 3. Temperature-Calibrated Two-Stage Confidence-Gated Routing (Table 4)

- **Stage-1 Alone ($\tau_{\text{gate}} = 0.00$):** $0.0\%$ routed, **$97.91\%$ Acc**, **$0.9789$ Macro-F1**, $\text{ECE} = 0.0094$, $\text{AURC} = 0.00164$.
- **Validation-Selected Operating Point ($\tau_{\text{gate}} = 0.65$):** Routes only **$1.53\%$** of test samples to Stage-2, achieving **$98.47\%$ Acc**, **$0.9836$ Macro-F1** ($95\%$ CI $[0.9712, 0.9932]$), $\text{ECE} = 0.0104$, and $\text{AURC} = 0.00158$.
- **High-Sensitivity Operating Point ($\tau_{\text{gate}} = 0.85$):** Routes **$4.32\%$** of test samples to Stage-2, achieving **$98.74\%$ Acc**, **$0.9859$ Macro-F1** ($95\%$ CI $[0.9742, 0.9952]$), and $\text{AURC} = 0.00151$.

![Figure 11: Selective Stage-2 Routing Rate vs. Accuracy/Macro-F1 and Calibration ECE/AURC](../results/figures/Fig11_Gate_Threshold_Sweep_and_Calibration.png)

---

## 4. Multi-Severity Perturbation Robustness and Single-Core CPU Profiling (Tables 5 & 6)

- **Multi-Severity Perturbations (`Table 5`):** Across 7 degradations, Proposed Quality-Gated preprocessing achieves **$0.9836$** Macro-F1 on clean images ($\sigma=0$), **$0.9249$** at $\sigma=5$ ($+0.2170$ vs. raw), **$0.8647$** at $\sigma=10$ (**$+0.3533$ vs. raw**; $+0.4440$ vs. unconditional CLAHE $0.4207$), **$0.6104$** at $\sigma=20$ ($+0.2367$ vs. raw), **$0.9812$** at glare $r=22\text{ px}$, **$0.9770$** at glare $r=35\text{ px}$, and **$0.8012$** under $-40\%$ low light ($+0.1371$ vs. raw).
- **Single-Core CPU Latency (`Table 6`):** Stage-1 executes in **$21.58\text{ ms}$** ($46.3\text{ img/s}$, $480.2\text{ KB}$) and Two-Stage Gated executes in **$21.74\text{ ms}$** ($46.0\text{ img/s}$, $4,060.6\text{ KB}$)—outperforming `MobileNetV3-Small` ($48.70\text{ ms}$, $5.9\text{ MB}$), `MobileNetV2` ($56.42\text{ ms}$, $8.5\text{ MB}$), and `ResNet50` ($272.04\text{ ms}$, $90.0\text{ MB}$).

![Figure 12: Multi-Severity Optical and Sensor Perturbation Robustness](../results/figures/Fig12_MultiSeverity_Perturbation_Robustness.png)

![Figure 13: Single-Core Edge CPU Latency Breakdown and Footprint Comparison](../results/figures/Fig13_CPU_Latency_Breakdown_and_Edge_Footprint.png)

---

## 5. External Cross-Dataset Generalization on `MLDID` ($N = 3,000$, Table 7)

- Evaluating on $3,000$ external images from India (`MLDID`, 5 shared classes), **Eigenvector Canonical Leaf Pose Normalization** improves zero-shot accuracy from **$31.27\%$ ($0.2886$ Macro-F1) to $42.63\%$ ($0.4395$ Macro-F1)** ($+11.36\%$ accuracy and $+15.09\%$ Macro-F1 gain), while achieving **$96.04\%$** canonical-pose and **$99.34\%$** source-view in-domain 5-class accuracy on `MangoLeafBD` ($N=455$).

![Figure 14: Eigenvector Canonical Leaf Pose Normalization on External MLDID](../results/figures/Fig14_CrossDataset_Canonical_Pose_and_CORAL.png)
