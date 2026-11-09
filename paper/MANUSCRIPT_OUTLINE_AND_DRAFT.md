# Manuscript Outline & Draft (Locked Version 5 Results)

## Title
**An Interpretable, Leak-Free, Quality-Gated Classical Computer Vision Framework for Mango Leaf Disease Classification and Cross-Dataset Generalization on Edge CPUs**

---

## Abstract
Recent deep learning studies on the public `MangoLeafBD` benchmark ($N=4,000$ across 8 classes) report near-perfect classification accuracies ($95.5\%$–$99.8\%$) under standard random train-test splits. However, three foundational questions remain unaddressed in the literature: (1) to what extent do near-duplicate or augmentation-derived sibling images leak across random splits, (2) how strongly do studio background and framing shortcuts confound disease recognition, and (3) can an ultra-compact, interpretable classical computer vision pipeline match deep convolutional backbones while running in real time on single-core edge CPUs and generalizing across geographic domains?

In this study, we first conduct a graph-based specimen-clustering audit (`pHash` Hamming distance $\le 6$ or HSV cosine similarity $\ge 0.992$), revealing that `MangoLeafBD` contains only **2,554 unique specimen clusters** (**36.15% near-duplicate/augmentation-mix ratio**, reaching **93.8%** in *Cutting Weevil* and **81.0%** in *Die Back*), causing **39.0% test-set cluster leakage** under naive random splits. Furthermore, our Five-View Background Shortcut Audit demonstrates that studio background pixels alone (`94.84%`) or four $32\times32$ corner border patches (`85.08%`) yield high accuracy without any leaf tissue.

Under a strict leak-free `StratifiedGroupKFold` protocol ($0\%$ cluster overlap), we propose a **Foreground-Aware Quality-Gated 7-Branch Feature Fusion & Two-Stage Confidence-Gated Edge AI Pipeline**. Our pipeline combines conditional MAD sensor noise filtering, central specular glare replacement, and foreground $L^*$ CLAHE with a shared-Sobel, lookup-table uniform LBP, 5-patch spatial, and multi-grid chromatic-texture (`MGCT`) extractor ($2,003\text{D}$). Using **Energy-Balanced Block-PCA** ($2,003\text{D} \to 303\text{D}$), we eliminate high-dimensional HOG dilution, boosting Stage-1 linear accuracy from **92.05% ($M_9$) to 97.91% ($M_{10}$, Macro-F1 = 0.9789)** inside a **480.2 KB** model running in **21.58 ms/img** (**46.3 img/s**) on a single CPU core—**2.24$\times$ to 12.51$\times$ faster** than MobileNetV3-Small (`48.70 ms`), MobileNetV2 (`56.42 ms`), and ResNet50 (`272.04 ms`). Routing only the **1.53%** most uncertain samples ($\tau_{\text{gate}} = 0.65$) to a nonlinear Stage-2 specialist ensemble elevates group-disjoint accuracy to **98.47%** (**0.9836 Macro-F1**, $\text{ECE}=0.0104$) at **21.74 ms/img** (`4.0 MB`), while segmented leaf organs without background (`M11`) retain **96.37% accuracy** (**0.9549 Macro-F1**). Finally, on the external Indian `MLDID` dataset ($N=3,000$ across 5 shared classes), our exploratory data analysis diagnoses a $90^\circ$ landscape-to-portrait orientation shift; our **Canonical Leaf Pose Normalization** increases zero-shot cross-dataset accuracy from **31.27% (0.2886 F1) to 42.63% (0.4395 F1)** (+11.36% accuracy, +0.1509 F1) without target labels.

---

## 1. Introduction & Key Contributions
1. **Specimen-Cluster Leakage & Five-View Background Shortcut Audit**: First rigorous quantification of specimen cluster duplication ($2,554$ unique clusters across $4,000$ images; $36.15\%$ near-duplicate ratio; $39.0\%$ random-split test leakage) and studio background/framing shortcuts on `MangoLeafBD` (Tables 1–2, Figure 2).
2. **Foreground-Aware Conditional Quality Gate**: Prevents unconditional CLAHE noise amplification on clean or grainy sensors ($\sigma=10$ F1 improved from `0.5114` raw / `0.4207` unconditional CLAHE to **`0.8647`**, a **`+0.3533`** gain; glare $r=35$ improved by **`+0.2359`** to **`0.9770`**, Table 5, Figure 6).
3. **Shared-Sobel 7-Branch Feature Fusion & Energy-Balanced Block-PCA**: Resolves the dimensionality dilution of $1,764\text{D}$ HOG against $239\text{D}$ chromatic-texture descriptors, recovering **+5.86% accuracy** (`92.05%` $\to$ `97.91%`, McNemar $p < 0.0001$, Table 3, Figure 3–4).
4. **Calibrated Two-Stage Confidence-Gated Edge Inference**: Achieves **98.47% group-disjoint accuracy** (`0.9836` Macro-F1) at **21.74 ms/img** (`46.0 img/s`) by routing only **1.53%** of samples ($\tau_{\text{gate}}=0.65$) to Stage-2 (`98.74%` / `0.9859` F1 at $\tau_{\text{gate}}=0.85$; `99.00%` on standard random split, Tables 4 & 6, Figure 5).
5. **Cross-Dataset Orientation Diagnosis & Canonical Leaf Pose Normalization**: Identifies portrait-vs-landscape aspect ratio shift between `MangoLeafBD` and `MLDID` ($N=3,000$), improving zero-shot accuracy from **31.27% to 42.63%** (+11.36%) and Macro-F1 from **0.2886 to 0.4395** (+0.1509) without target training data (Table 7, Figure 7).

---

## 2. Mapping of Manuscript Tables & Figures to Repository Assets

### LaTeX Tables (`paper/tables_latex/`)
- `\input{tables_latex/table1_audit.tex}` $\leftarrow$ `results/tables/Table1_Dataset_Leakage_and_Group_Audit.csv`
- `\input{tables_latex/table2_background_shortcut.tex}` $\leftarrow$ `results/tables/Table2_Background_Shortcut_Audit.csv`
- `\input{tables_latex/table3_progressive_ablation.tex}` $\leftarrow$ `results/tables/Table3_Progressive_Ablation_Study.csv`
- `\input{tables_latex/table4_confidence_gate_calibration.tex}` $\leftarrow$ `results/tables/Table4_Confidence_Gated_and_Calibration.csv`
- `\input{tables_latex/table5_perturbation_robustness.tex}` $\leftarrow$ `results/tables/Table5_MultiSeverity_Robustness.csv`
- `\input{tables_latex/table6_cpu_benchmark.tex}` $\leftarrow$ `results/tables/Table6_Fair_EndToEnd_CPU_Benchmark.csv`
- `\input{tables_latex/table7_external_validation_mldid.tex}` $\leftarrow$ `results/tables/Table7_External_Dataset_Validation.csv`
- `\input{tables_latex/table8_literature_comparison.tex}` $\leftarrow$ `results/tables/Table8_Published_Literature_Comparison.csv`

### Main Manuscript Figures (`results/figures/`)
- **Figure 1**: `results/figures/Figure1_Pipeline_Architecture_Overview.png`
- **Figure 2**: `results/figures/Figure2_Audit_and_Background_Shortcuts.png`
- **Figure 3**: `results/figures/Figure3_Feature_Block_Attribution.png`
- **Figure 4**: `results/figures/Figure4_Ablation_and_Confidence_Gate.png`
- **Figure 5**: `results/figures/Figure5_Latency_and_Resource_Profile.png`
- **Figure 6**: `results/figures/Figure6_Optical_Modes_and_Canonical_Pose.png`
- **Figure 7**: `results/figures/Figure7_Perturbation_Robustness_and_CrossDomain.png`
- **Supplementary EDA Figures S1–S16**: `results/figures/eda/*.png`
