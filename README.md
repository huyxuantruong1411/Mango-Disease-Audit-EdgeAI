# Audit-First, Quality-Gated Two-Stage Framework for Real-Time Mango Leaf Disease Classification on Edge CPUs

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg)](pyproject.toml)
[![Protocol: Locked v5](https://img.shields.io/badge/Experimental_Protocol-Locked_v5_(SEED_20261022)-2ea44f.svg)](scripts/verify_locked_results.py)
[![Single-Core Latency](https://img.shields.io/badge/Single--Core_CPU_Latency-21.58_ms_(46.3_FPS)-0052CC.svg)](results/tables/Table6_Fair_EndToEnd_CPU_Benchmark.csv)
[![Model Footprint](https://img.shields.io/badge/Stage--1_Footprint-480.2_KB-6f42c1.svg)](results/models/)

---

## Abstract

Plant pathology benchmarks captured under controlled studio conditions frequently report near-perfect classification accuracy ($99\%+$) using deep convolutional and vision transformer backbones. However, deploying these models on resource-constrained agricultural edge devices exposes three fundamental vulnerabilities:

1. **Methodological Evaluation Inflation:** Unaudited train/test partitions split rotated, zoomed, or sequentially captured views of the *same physical leaf specimen* across training and evaluation sets, while studio backdrops and camera aspect ratios introduce severe non-biological shortcuts.
2. **Optical Perturbation Fragility:** Unconditional contrast enhancement (such as global CLAHE) amplifies high-frequency sensor grain on clean or noisy images, whereas raw feature extractors fail under specular flash glare and low-light shadow regimes.
3. **Edge Compute Bottlenecks and Cross-Dataset Shift:** Deep backbones require tens of megabytes of memory and $48\text{--}272\text{ ms}$ per frame on single-core edge CPUs, yet suffer drastic accuracy collapse when evaluated zero-shot on external datasets with differing leaf orientations and illumination statistics.

This repository provides the official, bit-for-bit reproducible implementation of an **Audit-First, Quality-Gated Two-Stage Edge-AI Framework** (`mango_edge_cv`, locked Version 5 protocol, `SEED = 20261022`). Evaluated across **$7,000$ mango leaf images**—comprising **$4,000$ images across 8 classes in `MangoLeafBD`** (Bangladesh) and **$3,000$ external images across 5 shared classes in `MLDID`** (India)—our framework establishes six rigorous methodological and algorithmic contributions:

- **Pillar I (Graph-Based Specimen Audit and Five-View Shortcut Quantification):** A dual-criterion connected-component graph audit ($64$-bit DCT perceptual hash Hamming distance $d_{\text{Hamming}} \le 6$ or central-$60\%$ HSV histogram cosine similarity $\ge 0.992$) reveals that **$36.15\%$ ($1,446 / 4,000$) of `MangoLeafBD` images are near-duplicate or zoom-rotated siblings**, consolidating the dataset into **$2,554$ unique specimen clusters** (`Cutting_Weevil`: $93.8\%$ duplicate ratio; `Die_Back`: $81.0\%$). Naive stratified random splitting leaks **$39.0\%$ of multi-image clusters** into the test set. Furthermore, a 5-view shortcut audit proves that **masked studio background alone achieves $94.84\%$ accuracy**, four $32\times32$ corner patches achieve **$85.08\%$**, and a $7\text{D}$ file-metadata/aspect-ratio vector achieves **$59.14\%$** (vs. $12.50\%$ random chance).
- **Pillar II (Foreground-Aware Quality-Gated Optical Preprocessing):** A hybrid HSV saturation and CIE-LAB chromaticity salience segmenter ($S_{\text{roi}} = 0.60\,S + 1.25\,C^*$) isolates the leaf organ before applying three conditional optical gates: (i) a Donoho–Johnstone Median Absolute Deviation ($\hat{\sigma}_{\text{MAD}}$) sensor-grain gate, (ii) a central-window specular glare replacement gate ($V > 238 \land S < 38$), and (iii) a foreground-restricted CIE $L^*$ percentile stretch ($[P_2, P_{98}] \to [12, 242]$) with mild CLAHE ($\text{clipLimit}=1.55$). Under $\sigma = 10$ sensor noise, this gate improves Macro-F1 by **$+0.3533$** over raw features ($0.8647$ vs. $0.5114$), while unconditional CLAHE degrades to $0.4207$.
- **Pillar III (Shared-Sobel Seven-Branch Feature Extraction and Energy-Balanced Block-PCA):** A unified $128\times128$ extraction engine computes $2,003$ raw dimensions across 7 biological feature branches, reusing a single $3\times3$ Sobel derivative pass (`Gx`, `Gy`, `M`) and a $256$-entry lookup table for rotation-invariant uniform $\text{LBP}_{8,1}^{\text{riu2}}$. To resolve the **7.38:1 dimensionality dilution** where $1,764\text{D}$ HOG overwhelms $239\text{D}$ chromatic-texture descriptors (causing accuracy to drop from $97.35\%$ in $M_2$ to $86.61\%$ in $M_4$), our `BlockPCACompressor` retains all $239\text{D}$ color-texture features, compresses HOG to $64$ principal axes, and balances root-mean-square (RMS) energy ($\gamma = 0.45 \to 303\text{D}$), driving a statistically significant **$+5.86\%$ accuracy jump** ($92.05\%$ in $M_9$ to **$97.91\%$** in $M_{10}$, exact McNemar $p < 0.0001$).
- **Pillar IV (Temperature-Calibrated Two-Stage Confidence-Gated Routing):** A $480.2\text{ KB}$ Stage-1 multinomial L2 Logistic Regression specialist calibrated via Negative Log-Likelihood temperature scaling ($T^* = 0.758$, $\text{ECE} = 0.0094$) resolves **$98.47\%$ of test images immediately** at $\tau_{\text{gate}} = 0.65$. Only the **$1.53\%$ most ambiguous specimens** are routed to a heterogeneous Stage-2 nonlinear ensemble ($0.55\text{ RBF-SVM} + 0.30\text{ MLP-128} + 0.15\text{ Cosine-kNN}$), elevating leak-free group-split accuracy to **$98.47\%$ ($0.9836\text{ Macro-F1}$, $\text{AURC} = 0.00158$)** and reaching **$98.74\%$ ($0.9859\text{ Macro-F1}$)** at $\tau_{\text{gate}} = 0.85$.
- **Pillar V (Real-Time Single-Core Edge CPU Execution):** End-to-end single-thread CPU inference executes in **$21.58\text{ ms}$ ($46.3\text{ FPS}$, $480.2\text{ KB}$)** for Stage-1 and **$21.74\text{ ms}$ ($46.0\text{ FPS}$, $+0.16\text{ ms}$ amortized overhead)** for the two-stage system—running **$2.26\times$ faster than MobileNetV3-Small** ($48.70\text{ ms}$, $5.9\text{ MB}$), **$2.61\times$ faster than MobileNetV2** ($56.42\text{ ms}$, $8.5\text{ MB}$), and **$12.61\times$ faster than ResNet50** ($272.04\text{ ms}$, $90.0\text{ MB}$, $187\times$ larger).
- **Pillar VI (Cross-Dataset Generalization via Canonical Leaf Pose Normalization):** On the external `MLDID` benchmark ($N = 3,000$), aligning each segmented leaf's principal spatial covariance eigenvector horizontally (`canonicalize_leaf_organ_image`) and enforcing foreground $L^*$ percentile constancy boosts zero-shot cross-dataset accuracy from **$31.27\%$ ($0.2886\text{ Macro-F1}$) to $42.63\%$ ($0.4395\text{ Macro-F1}$)**—an absolute gain of **$+11.36\%$ accuracy and $+15.09\%$ Macro-F1** without target labels.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Pillar I: Graph-Based Specimen Leakage Audit and Five-View Background Shortcut Analysis](#2-pillar-i-graph-based-specimen-leakage-audit-and-five-view-background-shortcut-analysis)
3. [Pillar II: Foreground-Aware Quality-Gated Optical Preprocessing](#3-pillar-ii-foreground-aware-quality-gated-optical-preprocessing)
4. [Pillar III: Seven-Branch Feature Extraction and Energy-Balanced Block-PCA Compression](#4-pillar-iii-seven-branch-feature-extraction-and-energy-balanced-block-pca-compression)
5. [Pillar IV: Temperature-Calibrated Two-Stage Confidence-Gated Classification](#5-pillar-iv-temperature-calibrated-two-stage-confidence-gated-classification)
6. [Pillar V: Multi-Severity Optical Robustness and Single-Core Edge CPU Profiling](#6-pillar-v-multi-severity-optical-robustness-and-single-core-edge-cpu-profiling)
7. [Pillar VI: External Cross-Dataset Generalization (MangoLeafBD to MLDID)](#7-pillar-vi-external-cross-dataset-generalization-mangoleafbd-to-mldid)
8. [Exploratory Data Analysis (EDA) of Geometry and Cross-Domain Shifts](#8-exploratory-data-analysis-eda-of-geometry-and-cross-domain-shifts)
9. [Positioning Against Recent Published Literature (2023–2026)](#9-positioning-against-recent-published-literature-20232026)
10. [Repository Structure and Reproducibility Quickstart](#10-repository-structure-and-reproducibility-quickstart)
11. [Citation and License](#11-citation-and-license)

---

## 1. System Architecture Overview

The complete training, auditing, and inference workflow is structured as a six-stage modular pipeline implemented in [`src/mango_edge_cv/`](src/mango_edge_cv/). Figure 1 provides the high-level macro overview of the system, while subsequent sections decompose each individual stage with mathematical derivations, intermediate OpenCV visual proofs, and empirical ablation charts.

![Figure 1: Macro System Architecture Overview](results/figures/Fig01_System_Overview_Macro.png)

| Pipeline Stage | Core Module & Function | Input / Output Representation | Primary Algorithmic Role |
| :--- | :--- | :--- | :--- |
| **Stage 1: Specimen & Shortcut Audit** | [`audit.py`](src/mango_edge_cv/audit.py) (`build_audit_groups`) | $N = 4,000$ raw images $\to$ $2,554$ specimen cluster IDs | Eliminates $39.0\%$ random-split specimen leakage via connected components on 2D-DCT `pHash` and central HS histograms. |
| **Stage 2: ROI & Quality Gate** | [`preprocessing.py`](src/mango_edge_cv/preprocessing.py) (`segment_plant_organ_roi`, `apply_adaptive_quality_preprocessing`) | $256 \times 256 \times 3$ RGB + binary organ mask $M_{\text{fg}} \in \{0, 255\}^{256\times256}$ | Isolates leaf lamina via hybrid Otsu-Chroma salience; conditionally triggers MAD denoising, glare inpainting, and foreground CLAHE. |
| **Stage 3: 7-Branch Feature Engine** | [`features.py`](src/mango_edge_cv/features.py) (`extract_fast_all_branches`) | $128 \times 128$ unified grid $\to$ $\mathbf{x} \in \mathbb{R}^{2,003}$ | Extracts HSV ($52\text{D}$), LAB ($57\text{D}$), LUT-LBP ($28\text{D}$), 5-Patch ($45\text{D}$), Quality $q$ ($7\text{D}$), MGCT ($50\text{D}$), and Shared-Sobel HOG ($1,764\text{D}$). |
| **Stage 4: Energy-Balanced Block-PCA** | [`models.py`](src/mango_edge_cv/models.py) (`BlockPCACompressor`) | $\mathbf{x} \in \mathbb{R}^{2,003} \to \mathbf{z} \in \mathbb{R}^{303}$ | Preserves $239\text{D}$ color-texture features intact, projects $1,764\text{D}$ HOG onto $k=64$ principal axes, and balances RMS energy ($\gamma = 0.45$). |
| **Stage 5: Calibrated Confidence Gate** | [`models.py`](src/mango_edge_cv/models.py) (`CalibratedConfidenceGatedSystem`) | $\mathbf{z} \in \mathbb{R}^{303} \to \hat{y} \in \{1,\dots,8\}$ | Evaluates temperature-scaled L2-LR ($T^*=0.758$); accepts $98.47\%$ on fast path ($\tau_{\text{gate}}=0.65$) and routes $1.53\%$ to Stage-2 ensemble. |
| **Stage 6: Cross-Domain Alignment** | [`preprocessing.py`](src/mango_edge_cv/preprocessing.py) (`canonicalize_leaf_organ_image`) | Arbitrary $512 \times 512$ frame $\to$ $256 \times 256$ horizontal canonical pose | Aligns principal covariance axis $\theta$ of segmented leaf horizontally and normalizes foreground $L^*$ dynamic range for external transfer. |

---

## 2. Pillar I: Graph-Based Specimen Leakage Audit and Five-View Background Shortcut Analysis

### 2.1 Mathematical Formulation of Specimen-Identity Graph Clustering

In agricultural image acquisition, operators frequently capture multiple photographs of the same diseased leaf under slight camera rotations ($90^\circ, 180^\circ, 270^\circ$), zoom adjustments, or consecutive burst shots, and sometimes apply offline augmentations before publishing the repository. If image instances are partitioned via standard stratified random sampling, sibling views of the same physical leaf populate both $\mathcal{D}_{\text{train}}$ and $\mathcal{D}_{\text{test}}$, turning evaluation into near-duplicate instance memorization.

To audit and eliminate specimen leakage without relying on corruptible filenames, we construct an undirected graph $G_c = (V_c, E_c)$ independently within each disease class $c \in \{1,\dots,8\}$ (`build_audit_groups` in [`src/mango_edge_cv/audit.py`](src/mango_edge_cv/audit.py)):

1. **Structural Low-Frequency Perceptual Hash ($\text{pHash}_{64}$):** Each image $I_i$ is converted to grayscale and downsampled via area interpolation to $I_{32} \in \mathbb{R}^{32\times32}$. We compute the 2D Discrete Cosine Transform $D = \text{DCT}_{2\text{D}}(I_{32})$ and extract the top-left $8\times8$ low-frequency submatrix $D_{8\times8} = D_{0:8, 0:8}$. Excluding the DC component $D_{0,0}$, let $m_{\text{AC}} = \text{median}(\{D_{u,v} : (u,v) \in \{0,\dots,7\}^2 \setminus \{(0,0)\}\})$. The $64$-bit binary hash is defined as:
   $$b_{u,v}(I_i) = \mathbb{I}\!\left(D_{u,v}(I_i) > m_{\text{AC}}(I_i)\right), \quad (u,v) \in \{0,\dots,7\}^2$$
   The structural dissimilarity between images $I_i$ and $I_j$ is measured by the Hamming distance:
   $$d_{\text{Hamming}}(I_i, I_j) = \sum_{u=0}^{7}\sum_{v=0}^{7} \mathbb{I}\!\left(b_{u,v}(I_i) \neq b_{u,v}(I_j)\right)$$

2. **Rotation-Invariant Central-ROI Chromatic Cosine Similarity:** Because $90^\circ/180^\circ/270^\circ$ rotations alter spatial DCT frequency indices while preserving the leaf's internal lesion color distribution, we extract the central $60\%$ window $I_{\text{center}} = I[0.20H : 0.80H,\, 0.20W : 0.80W]$, resize to $64\times64$, transform to HSV space, and compute a joint 2D Hue–Saturation histogram with $12$ Hue bins ($\in [0, 180)$) and $8$ Saturation bins ($\in [0, 256)$), flattened and $L_2$-normalized into $\mathbf{h}_i \in \mathbb{R}^{96}$ ($\|\mathbf{h}_i\|_2 = 1$). Chromatic similarity is given by the inner product:
   $$s_{\text{cos}}(I_i, I_j) = \frac{\langle \mathbf{h}_i, \mathbf{h}_j \rangle}{\|\mathbf{h}_i\|_2 \, \|\mathbf{h}_j\|_2 + \varepsilon}$$

3. **Connected-Component Specimen Clusters:** An undirected edge $(i, j) \in E_c$ is formed if and only if either structural or rotation-invariant chromatic proximity holds:
   $$(i, j) \in E_c \iff \Big(d_{\text{Hamming}}(I_i, I_j) \le 6\Big) \;\lor\; \Big(s_{\text{cos}}(I_i, I_j) \ge 0.992\Big)$$
   Connected components of $G_c$ are extracted via depth-first traversal (`scipy.sparse.csgraph.connected_components`), assigning a unique specimen cluster identifier `group_id` to each component. All training, validation, and test partitions are subsequently generated using `StratifiedGroupKFold` so that every connected component resides entirely within a single split ($\mathcal{G}_{\text{train}} \cap \mathcal{G}_{\text{val}} = \varnothing$ and $\mathcal{G}_{\text{train}} \cap \mathcal{G}_{\text{test}} = \varnothing$).

![Figure 2: Graph-Based Specimen Clustering Mechanism and Real Sibling Proof](results/figures/Fig02_Specimen_Clustering_Graph_Mechanism.png)

### 2.2 Quantitative Specimen Audit Findings (Table 1)

Figure 2 illustrates real connected components discovered in [`manifest_MANGO_audited.csv`](results/manifests/manifest_MANGO_audited.csv), such as `Anthracnose` cluster `group_id = 143` (where rotated copies `20211011_170706 (Custom).jpg` and `20211011_170703 (Custom).jpg` were split across `Train_Orig`, `Val_Orig`, and `Test_Orig` under naive random splitting) and `Cutting_Weevil` mega-cluster `group_id = 1000` (which links $368$ zoom-rotated siblings out of the class's $500$ total images). Figure 3 and Table 1 summarize the complete class-wise audit across all $4,000$ `MangoLeafBD` images.

![Figure 3: Class-Wise Unique Specimen Clusters vs. Near-Duplicate Siblings](results/figures/Fig03_Cluster_Distribution_and_Split_Leakage.png)

**Table 1: Specimen-Identity Clustering and Split-Leakage Audit on `MangoLeafBD` ([`Table1_Dataset_Leakage_and_Group_Audit.csv`](results/tables/Table1_Dataset_Leakage_and_Group_Audit.csv))**

| Class Name | Raw Images ($N$) | Unique Specimen Clusters ($d_{\text{H}} \le 6 \lor \cos_{\text{HS}} \ge 0.992$) | Near-Duplicate / Sibling Ratio (%) | Group-Aware Train / Val / Test Images |
| :--- | :---: | :---: | :---: | :---: |
| `Anthracnose` | 500 | 352 | 29.60% | 347 / 72 / 81 |
| `Bacterial_Canker` | 500 | 275 | 45.00% | 351 / 73 / 76 |
| `Cutting_Weevil` | 500 | **31** | **93.80%** | 350 / 74 / 76 |
| `Die_Back` | 500 | **95** | **81.00%** | 348 / 76 / 76 |
| `Gall_Midge` | 500 | 437 | 12.60% | 350 / 75 / 75 |
| `Healthy` | 500 | 469 | 6.20% | 350 / 75 / 75 |
| `Powdery_Mildew` | 500 | 427 | 14.60% | 349 / 75 / 76 |
| `Sooty_Mold` | 500 | 468 | 6.40% | 350 / 75 / 75 |
| **Total / Global Summary** | **4,000** | **2,554** | **36.15%** ($1,446$ siblings) | **2,795 / 595 / 610** |

> **Key Audit Proof:** Under a standard $70/15/15$ stratified random split, **$85$ out of $218$ multi-image specimen clusters ($39.0\%$) leak simultaneously into both the training and held-out test partitions**, artificially inflating baseline accuracy ($98.88\%$ random-split vs. $98.33\%$ group-split; $99.00\%$ vs. $98.47\%$ for the two-stage system). Under our group-aware protocol, cluster overlap between train, validation, and test splits is strictly **$0.0\%$ ($0$ leaked clusters)**.

### 2.3 Five-View Studio Background and Framing Shortcut Audit (Table 2)

Even after eliminating specimen leakage, models trained on studio datasets can exploit batch-level capture artifacts (backdrop paper tint, peripheral vignetting, JPEG compression, and camera aspect ratio). To quantify this confounding, we evaluate an identical regularized linear probe ($137\text{D}$ HSV+LAB+LBP for visual Views 1–4; $7\text{D}$ file metadata for View 5) across five systematically restricted views under the leak-free group split (`run_background_shortcut_audit` in [`src/mango_edge_cv/audit.py`](src/mango_edge_cv/audit.py)).

![Figure 4: Visual and Quantitative Proof of the Five-View Studio Background Shortcut Audit](results/figures/Fig04_FiveView_Background_Shortcut_VisualProof.png)

**Table 2: Five-View Studio Background and Non-Visual Metadata Shortcut Audit ([`Table2_Background_Shortcut_Audit.csv`](results/tables/Table2_Background_Shortcut_Audit.csv))**

| Audit View / Input Restriction | Mathematical Input Domain | Feature Dim | Group-Split Test Acc (%) | Group-Split Macro-F1 | Random-Split Test Acc (%) | Empirical Interpretation |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **1. Whole Image ROI** | Center-cropped frame ($r = 0.86$), leaf + studio backdrop | 137D | **98.33%** | **0.9833** | 98.88% | Upper-bound combining biological lesions and studio context. |
| **2. Segmented Plant Organ Only** | $I_{\text{fg}} = I \odot M_{\text{fg}}$ (studio background zeroed) | 137D | **94.84%** | **0.9440** | 96.37% | Pure biological signal extracted strictly from leaf lamina. |
| **3. Background Only (Organ Masked Out)** | $I_{\text{bg}} = I \odot (1 - M_{\text{fg}})$ (**zero leaf pixels**) | 137D | **94.84%** | **0.9396** | 94.98% | Studio paper reflectance and shadow vignette alone match organ accuracy. |
| **4. Four $32\times32$ Corner Patches Only** | $\bigcup_{k=1}^{4} \Omega_{\text{corner},k}$ ($6.25\%$ outer corner area) | 137D | **85.08%** | **0.8332** | 85.49% | Extreme peripheral backdrop corners predict disease at $6.8\times$ chance ($12.5\%$). |
| **5. Image File Metadata Only** | $[\text{KB}, \mu_R, \mu_G, \mu_B, \sigma_{\text{RGB}}, r_{\text{fg}}, H/W]$ | 7D | **59.14%** | **0.5742** | 59.83% | Non-visual file size and aspect ratio predict class at $4.7\times$ chance. |

---

## 3. Pillar II: Foreground-Aware Quality-Gated Optical Preprocessing

### 3.1 Hybrid HSV Saturation and CIE-LAB Chromaticity ROI Segmentation

Because studio backgrounds carry strong class-specific shortcuts (Table 2), preprocessing and feature extraction must be anchored to the biological leaf organ. However, standard excess-green ($2G - R - B$) or pure HSV hue thresholding fails on severely diseased mango leaves because necrotic `Anthracnose` and `Die_Back` lesions turn dark brown or yellow, while `Powdery_Mildew` and `Sooty_Mold` coat the lamina in gray/black mycelium.

As shown in Figure 5, `segment_plant_organ_roi` ([`src/mango_edge_cv/preprocessing.py`](src/mango_edge_cv/preprocessing.py)) computes a hybrid chromaticity salience map on the standardized $256\times256$ image ($r = 0.88$ center crop):

$$C^*(x,y) = \sqrt{\big(a^*(x,y) - 128\big)^2 + \big(b^*(x,y) - 128\big)^2}$$

$$S_{\text{roi}}(x,y) = \text{clip}\!\Big(0.60 \cdot S_{\text{HSV}}(x,y) + 1.25 \cdot C^*(x,y),\; 0,\; 255\Big)$$

Why this combination works: healthy green tissue exhibits high $S_{\text{HSV}}$ and negative $a^*$, whereas brown/yellow necrotic tissue loses green hue but maintains high perceptual departure from neutrality along the CIE $a^*\text{--}b^*$ axes ($C^*$). By weighting $C^*$ at $1.25$ and $S_{\text{HSV}}$ at $0.60$, both healthy and necrotic tissue remain strongly suprathreshold against the achromatic studio backdrop ($S_{\text{HSV}} < 28, C^* \approx 0$). We then apply Otsu's optimal threshold $T_{\text{Otsu}}$, explicitly zero out studio backdrop pixels satisfying $(V > 230 \land S < 28) \lor (V < 15)$ as well as a $3\%$ frame border margin, apply $7\times7$ elliptical morphological closing ($2$ iterations) and opening ($1$ iteration), and select the largest 8-connected component $M_{\text{fg}}(x,y) \in \{0, 255\}$ (valid when area $\in [4\%, 92\%]$ of frame area, with a centered elliptical fallback).

![Figure 5: Step-by-Step Hybrid HSV Saturation and CIE-LAB Chroma ROI Segmentation](results/figures/Fig05_Hybrid_ROI_Segmentation_StepByStep.png)

### 3.2 Three Conditional Optical Restoration Gates (`apply_adaptive_quality_preprocessing`)

Unconditional image enhancement (such as applying CLAHE or sharpening to every image) degrades clean images and catastrophically amplifies sensor noise (dropping $\sigma=10$ noisy Macro-F1 from $0.5114$ to $0.4207$ in Table 5). Our quality-gated preprocessor evaluates three physical quality metrics strictly within the foreground leaf mask $M_{\text{fg}}$ and triggers restoration only when quantitative thresholds are crossed (Figure 6):

1. **Gate 1 — Donoho–Johnstone MAD Sensor Grain Gate (`_estimate_noise_sigma_mad`):**
   We isolate high-frequency spatial residuals on the grayscale leaf foreground via a $3\times3$ median filter:
   $$R(x,y) = I_{\text{gray}}(x,y) - \text{MedianBlur}_{3\times3}(I_{\text{gray}})(x,y), \quad (x,y) \in M_{\text{fg}}$$
   Using the robust Median Absolute Deviation (MAD) scale estimator for Gaussian noise ($\Phi^{-1}(0.75) \approx 0.6745$):
   $$\hat{\sigma}_{\text{MAD}} = \frac{\text{median}_{(x,y)\in M_{\text{fg}}}\!\big(|R(x,y) - \text{median}(R)|\big)}{0.6745}$$
   - **Severe Grain ($\hat{\sigma}_{\text{MAD}} \ge 10.0$):** Applies cascaded impulse- plus-Gaussian smoothing: $I \leftarrow \text{GaussianBlur}_{3\times3}\big(\text{MedianBlur}_{3\times3}(I),\; \sigma=0.65\big)$.
   - **Moderate Grain ($5.5 \le \hat{\sigma}_{\text{MAD}} < 10.0$):** Applies mild Gaussian smoothing: $I \leftarrow \text{GaussianBlur}_{3\times3}(I,\; \sigma=0.55)$.
   - **Clean Regime ($\hat{\sigma}_{\text{MAD}} < 5.5$):** Bypasses spatial smoothing completely ($0\%$ sharpness loss on clean venation).

2. **Gate 2 — Central-Window Specular Flash Glare Attenuation (`apply_glare_inpainting`):**
   Waxy mango cuticles produce bright, achromatic specular reflections under direct sun or smartphone flash ($V > 238 \land S < 38$), generating artificial step-edges that corrupt Sobel and HOG orientation histograms. To avoid confusing white studio paper at the image periphery with cuticle glare, detection is restricted to the central $72\%$ window $\Omega_{\text{center}} = [0.14H : 0.86H,\, 0.14W : 0.86W]$ intersected with $M_{\text{fg}}$. Let $r_{\text{glare}}$ denote the fraction of foreground pixels satisfying the glare predicate. If $0.003 \le r_{\text{glare}} \le 0.220$, the glare mask $M_{\text{glare}}$ is dilated with a $5\times5$ elliptical kernel and replaced by a local $11\times11$ Gaussian neighborhood interpolation ($\sigma = 2.5$).

3. **Gate 3 — Foreground-Restricted CIE $L^*$ Percentile Stretch and Mild CLAHE:**
   Evaluating luminance $L^*(x,y)$ exclusively on non-glare foreground pixels $(x,y) \in M_{\text{fg}} \setminus M_{\text{glare}}$, we compute the standard deviation $\sigma(L^*_{\text{fg}})$ and inter-percentile dynamic range $\Delta L^*_{95,5} = P_{95}(L^*_{\text{fg}}) - P_5(L^*_{\text{fg}})$. If and only if the leaf is under-exposed or low-contrast ($\sigma(L^*_{\text{fg}}) < 38.0 \lor \Delta L^*_{95,5} < 132.0$), we apply a linear percentile stretch mapping $[P_2(L^*_{\text{fg}}), P_{98}(L^*_{\text{fg}})] \to [12, 242]$ followed by mild Contrast-Limited Adaptive Histogram Equalization ($\text{clipLimit} = 1.55$, $8\times8$ tile grid), updating **only** pixels inside $M_{\text{fg}} \setminus M_{\text{glare}}$ while leaving background pixels untouched.

![Figure 6: Visual and Quantitative Proof of Foreground-Aware Quality-Gated Preprocessing](results/figures/Fig06_Quality_Gate_MAD_Glare_CLAHE_Proof.png)

---

## 4. Pillar III: Seven-Branch Feature Extraction and Energy-Balanced Block-PCA Compression

### 4.1 Mathematical Specification of the Seven Feature Branches ($2,003\text{D}$ Raw)

After quality-gated preprocessing at $256\times256$, the image is downsampled once to a unified **$128\times128$ extraction grid** (`extract_fast_all_branches` in [`src/mango_edge_cv/features.py`](src/mango_edge_cv/features.py)). Rather than computing redundant image derivatives across texture and shape descriptors, a **single $3\times3$ Sobel operator pass** computes horizontal gradients $G_x = \partial_x I_{\text{gray}}$, vertical gradients $G_y = \partial_y I_{\text{gray}}$, and gradient magnitude $M = \sqrt{G_x^2 + G_y^2}$, which are shared across Branches 4, 6, and 7. All histograms and moments in Branches 1–3 and 6 are computed strictly over the valid foreground mask $M_{\text{fg},128}$ (when $|M_{\text{fg},128}| > 80$ pixels). Figure 7 visualizes the intermediate OpenCV feature maps for all 7 branches on a real symptomatic leaf, and Figure 7b plots their learned L1 weight attribution in Stage-1.

![Figure 7: Visual Decomposition of the Seven Feature Branches on the Unified 128x128 Grid](results/figures/Fig07_Seven_Feature_Branches_Visual_Maps.png)

1. **Branch 1 — `f_hsv` ($52\text{D}$, $21.6\%$ Stage-1 L1 Attribution):**
   Concatenates $16$-bin $L_2$-normalized foreground histograms for Hue $H \in [0, 180)$, Saturation $S \in [0, 256)$, and Value $V \in [0, 256)$ ($16 \times 3 = 48\text{D}$) with $4$ global normalized foreground means:
   $$\mathbf{f}_{\text{hsv}} = \left[\hat{\mathbf{h}}_H^{(16)},\; \hat{\mathbf{h}}_S^{(16)},\; \hat{\mathbf{h}}_V^{(16)},\; \frac{\mu_H}{180},\; \frac{\mu_S}{255},\; \frac{\mu_V}{255},\; \frac{1}{3}\!\left(\frac{\mu_H}{180} + \frac{\mu_S}{255} + \frac{\mu_V}{255}\right)\right] \in \mathbb{R}^{52}$$

2. **Branch 2 — `f_lab` ($57\text{D}$, $15.7\%$ Stage-1 L1 Attribution):**
   In perceptual CIE $L^*a^*b^*$ space, computes $16$-bin $L_2$-normalized foreground histograms ($48\text{D}$) plus the first three standardized statistical moments—mean $\mu_c / 255$, standard deviation $\sigma_c / 255$, and third-order standardized skewness $\gamma_1(c)$—for each channel $c \in \{L^*, a^*, b^*\}$ ($3 \times 3 = 9\text{D}$):
   $$\gamma_1(c) = \frac{\frac{1}{|M_{\text{fg}}|}\sum_{(x,y)\in M_{\text{fg}}} \big(c(x,y) - \mu_c\big)^3}{\sigma_c^3 + 10^{-4}}, \quad \mathbf{f}_{\text{lab}} \in \mathbb{R}^{57}$$
   Skewness $\gamma_1(a^*)$ and $\gamma_1(b^*)$ capture the asymmetric tail distribution created when small dark-brown necrotic spots appear on an otherwise green leaf lamina.

3. **Branch 3 — `f_lbp` ($28\text{D}$, $12.0\%$ Stage-1 L1 Attribution):**
   To achieve $O(1)$ per-pixel texture encoding without Python loops, we precompute a $256$-entry lookup table `_LBP_RIU2_LUT` mapping every 8-bit circular neighborhood pattern $b = \sum_{p=0}^{7} \mathbb{I}(g_p \ge g_c) 2^p \in \{0,\dots,255\}$ to its rotation-invariant uniform code $\text{LBP}_{8,1}^{\text{riu2}} \in \{0,\dots,9\}$ based on bitwise transition count $U(b) = \sum_{p=0}^{7} |b_p - b_{(p+1)\bmod 8}|$:
   $$\text{LBP}_{8,1}^{\text{riu2}}(b) = \begin{cases} \sum_{p=0}^{7} b_p & \text{if } U(b) \le 2 \;\;(\text{codes } 0\text{--}8), \\ 9 & \text{if } U(b) > 2 \;\;(\text{non-uniform code } 9). \end{cases}$$
   We extract the $10$-bin $L_2$-normalized $\text{LBP}_{8,1}^{\text{riu2}}$ histogram at full $128\times128$ resolution ($10\text{D}$), a second $10$-bin histogram on a $64\times64$ downsampled grid capturing coarser lesion rings ($10\text{D}$), and an $8$-bin histogram of local micro-contrast residuals $|I_{\text{gray},128} - \text{GaussianBlur}_{3\times3}(I_{\text{gray},128})|$ over $[0, 48]$ ($8\text{D}$), totaling $28\text{D}$.

4. **Branch 4 — `f_patch5` ($45\text{D}$, $15.7\%$ Stage-1 L1 Attribution):**
   Certain pathologies exhibit strong spatial localization (`Die_Back` initiates at the leaf apex/margins; `Cutting_Weevil` severs the mid-lamina). We partition the $128\times128$ grid into $5$ spatial zones—Top-Left ($64\times64$), Top-Right ($64\times64$), Bottom-Left ($64\times64$), Bottom-Right ($64\times64$), and Central ($[32:96, 32:96]$)—and extract $9$ statistics per zone, reusing the shared Sobel magnitude $M$:
   $$\mathbf{f}_{\text{zone}} = \left[\frac{\mu_H}{180},\; \frac{\mu_S}{255},\; \frac{\sigma_S}{255},\; \frac{\mu_{L^*}}{255},\; \frac{\mu_{a^*}}{255},\; \frac{\sigma_{a^*}}{255},\; \frac{\mu_{b^*}}{255},\; \frac{\sigma_{\text{gray}}}{255},\; \frac{\mu_M}{255}\right] \in \mathbb{R}^9 \implies 5 \times 9 = 45\text{D}$$

5. **Branch 5 — `f_q` ($7\text{D}$, $3.0\%$ Stage-1 L1 Attribution):**
   Encodes explicit optical regime awareness so downstream classifiers can adjust decision hyperplanes under residual shadow or blur:
   $$\mathbf{f}_q = \left[\frac{\mu_V}{255},\; \frac{\sigma_V}{255},\; \Pr(V < 40),\; \Pr(V > 240 \land S < 35),\; \frac{\text{Var}(\nabla^2 I_{\text{gray}})}{1000},\; \frac{H_{\text{Shannon}}(\text{LBP}_{8,1}^{\text{riu2}})}{\log_2 10},\; \mu_{\text{HOG}}\right] \in \mathbb{R}^7$$

6. **Branch 6 — `f_mgct` ($50\text{D}$, $17.2\%$ Stage-1 L1 Attribution):**
   Our **Multi-Grid Chromatic-Texture (MGCT)** descriptor explicitly captures the physical coupling between chromatic shift ($C^* = \sqrt{(a^*-128)^2 + (b^*-128)^2}$) and structural edge transitions ($M = \|\nabla I\|$) at necrotic lesion borders. Across $5$ regions (the full foreground mask $M_{\text{fg},128}$ plus $4$ spatial quadrants $64\times64$), it computes $10$ statistics per region ($5 \times 10 = 50\text{D}$):
   $$\mathbf{f}_{\text{mgct},r} = \left[\frac{\mu_S}{255},\; \frac{\sigma_S}{255},\; \frac{\mu_{a^*}}{255},\; \frac{\sigma_{a^*}}{255},\; \frac{\mu_{b^*}}{255},\; \frac{\sigma_{b^*}}{255},\; \frac{\mu_M}{255},\; \frac{\sigma_M}{255},\; \rho(C^*, M),\; \Pr(M > 60)\right] \in \mathbb{R}^{10}$$
   where $\rho(C^*, M)$ is the Pearson correlation coefficient between chromaticity magnitude and gradient magnitude inside region $r$.

7. **Branch 7 — `f_hog` ($1,764\text{D}$ Raw $\to 64\text{D}$ Block-PCA, $14.8\%$ Stage-1 L1 Attribution):**
   Reusing $(G_x, G_y, M)$ on the $128\times128$ grid, `compute_hog_128` reshapes the image into $8\times8 = 64$ cells of $16\times16$ pixels, accumulates gradient magnitudes into $9$ unsigned orientation bins in $[0^\circ, 180^\circ)$ via vectorized bilinear bin interpolation (`np.bincount`), and normalizes $7\times7 = 49$ overlapping $2\times2$-cell blocks using L2-Hys clipping at $0.2$, yielding $49 \times (2\times2\times9) = 1,764\text{D}$.

![Figure 7b: Stage-1 L1 Weight Attribution Across the Seven Compressed Feature Branches](results/figures/Fig07b_Feature_Block_Attribution.png)

### 4.2 Resolving High-Dimensional HOG Variance Dilution via `BlockPCACompressor`

A central finding of our progressive ablation study (Table 3) is that **naive concatenation of multi-modal features severely degrades linear classification accuracy**. Specifically, while compact HSV + LBP features ($M_2$, $80\text{D}$) achieve **$97.35\%$ accuracy**, appending the $1,764\text{D}$ HOG vector ($M_4$, $1,844\text{D}$) causes test accuracy to collapse by **$-10.74\%$ down to $86.61\%$** (and $92.05\%$ for the full $2,003\text{D}$ concatenation in $M_9$).

**Mathematical Proof of Dimensionality Dilution:** Let $\mathbf{x} = [\mathbf{x}_{\text{ct}};\, \mathbf{x}_{\text{hog}}] \in \mathbb{R}^{2,003}$, where $\mathbf{x}_{\text{ct}} \in \mathbb{R}^{239}$ denotes the $6$ chromatic-texture-quality branches and $\mathbf{x}_{\text{hog}} \in \mathbb{R}^{1,764}$ denotes the HOG branch. After per-coordinate Z-score standardization ($\mathbb{E}[\tilde{x}_j^2] = 1$), the expected squared Euclidean norm of the concatenated vector is partitioned in proportion to dimension count:
$$\mathbb{E}\!\left[\|\tilde{\mathbf{x}}_{\text{hog}}\|_2^2\right] = 1,764 \;\;(88.07\% \text{ of total variance}), \qquad \mathbb{E}\!\left[\|\tilde{\mathbf{x}}_{\text{ct}}\|_2^2\right] = 239 \;\;(11.93\% \text{ of total variance})$$
Under an isotropic $L_2$-regularized objective $\min_{\mathbf{W}} \mathcal{L}_{\text{CE}}(\mathbf{W}\tilde{\mathbf{x}}) + \frac{1}{2C}\|\mathbf{W}\|_F^2$, inner products $\langle \mathbf{w}_c, \tilde{\mathbf{x}} \rangle$ and gradient updates are dominated by a **$7.38 : 1$ variance imbalance** in favor of spatial HOG cells, many of which encode background-edge or leaf-pose variance rather than disease pathology.

![Figure 8: Resolving High-Dimensional HOG Variance Dilution via Energy-Balanced Block-PCA](results/figures/Fig08_BlockPCA_Dimensionality_Dilution_Solution.png)

To restore cross-modal synergy, `BlockPCACompressor` ([`src/mango_edge_cv/models.py`](src/mango_edge_cv/models.py)) performs three operations fitted strictly on $\mathcal{D}_{\text{train}}$ (Figure 8):
1. **100% Preservation of Chromatic-Texture-Quality Subspace:** Standardizes $\mathbf{x}_{\text{ct}} \in \mathbb{R}^{239} \to \tilde{\mathbf{x}}_{\text{ct}} \in \mathbb{R}^{239}$ without dimensionality reduction.
2. **Orthogonal Subspace Compression of HOG:** Standardizes $\mathbf{x}_{\text{hog}} \in \mathbb{R}^{1,764} \to \tilde{\mathbf{x}}_{\text{hog}}$ and projects onto the top $k = 64$ unwhitened principal eigenvectors $\mathbf{U}_{64} \in \mathbb{R}^{1,764 \times 64}$ (`whiten=False`, preserving relative eigenvalue importance while discarding $1,700$ noisy tail dimensions):
   $$\mathbf{z}_{\text{hog}} = \mathbf{U}_{64}^\top \tilde{\mathbf{x}}_{\text{hog}} \in \mathbb{R}^{64} \quad (27.6\times \text{ compression})$$
3. **Root-Mean-Square (RMS) Energy Balancing:** Computes the empirical training RMS magnitudes $\text{RMS}(\tilde{\mathbf{X}}_{\text{ct}}) = \sqrt{\frac{1}{N_{\text{tr}}\cdot 239}\|\tilde{\mathbf{X}}_{\text{ct}}\|_F^2}$ and $\text{RMS}(\mathbf{Z}_{\text{hog}}) = \sqrt{\frac{1}{N_{\text{tr}}\cdot 64}\|\mathbf{Z}_{\text{hog}}\|_F^2}$, and scales the compressed HOG block by:
   $$\alpha_{\text{hog}} = \gamma \cdot \frac{\text{RMS}(\tilde{\mathbf{X}}_{\text{ct}})}{\text{RMS}(\mathbf{Z}_{\text{hog}})}, \quad \gamma = 0.45 \implies \mathbf{z} = \big[\tilde{\mathbf{x}}_{\text{ct}};\; \alpha_{\text{hog}}\mathbf{z}_{\text{hog}}\big] \in \mathbb{R}^{303}$$

### 4.3 Progressive 14-Step Ablation Study ($M_1$ to $M_{11}$, Table 3)

Figure 9 and Table 3 trace the exact progression across all $14$ ablation configurations under the leak-free group-split protocol, complete with 3-fold group cross-validation Macro-F1 on the training split, held-out test Macro-F1, $95\%$ bootstrap confidence intervals ($B = 1,000$), and exact binomial McNemar $p$-values against the raw baseline $M_4$.

![Figure 9: Progressive 14-Step Ablation Trajectory (M1 to M11)](results/figures/Fig09_Progressive_Ablation_Trajectory_M1_M11.png)

**Table 3: Progressive Ablation Study from Single-Branch HSV ($M_1$) to Energy-Balanced Block-PCA ($M_{10}, M_{11}$) ([`Table3_Progressive_Ablation_Study.csv`](results/tables/Table3_Progressive_Ablation_Study.csv))**

| Configuration | Feature Dims | 3-Fold Train Group-CV F1 | Group-Split Test Acc (%) | Macro-Precision | Macro-Recall | Group-Split Test Macro-F1 | 95% Bootstrap CI (Macro-F1) | Exact McNemar $p$-val (vs. $M_4$) | Rel. Latency Proxy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `M1. HSV Only (OvR LR)` | 52 | $0.8670 \pm 0.0181$ | 90.38% | 0.9008 | 0.9012 | 0.8997 | $[0.8763, 0.9231]$ | 0.0589 | 0.61x |
| `M2. HSV + Fast Multi-Scale LBP (OvR LR)` | 80 | $0.9548 \pm 0.0103$ | 97.35% | 0.9764 | 0.9718 | 0.9739 | $[0.9607, 0.9848]$ | $< 0.0001$ | 0.81x |
| `M3. HSV + Vectorized Sobel HOG (OvR LR)` | 1,816 | $0.7692 \pm 0.0441$ | 83.40% | 0.8526 | 0.7810 | 0.7823 | $[0.7475, 0.8172]$ | 0.0001 | 0.85x |
| `M4. Baseline: HSV + LBP + HOG (OvR LR)` | 1,844 | $0.8156 \pm 0.0396$ | **86.61%** | 0.8820 | 0.8356 | **0.8453** | $[0.8122, 0.8770]$ | 0.3018 (ref) | 1.04x |
| `M5. M4 + CIE LAB Color Space` | 1,901 | $0.8720 \pm 0.0382$ | 90.66% | 0.9186 | 0.8848 | 0.8948 | $[0.8658, 0.9209]$ | 0.0001 | 1.34x |
| `M6. M5 + 5-Patch Spatial Moments` | 1,946 | $0.8907 \pm 0.0344$ | 92.19% | 0.9313 | 0.8980 | 0.9079 | $[0.8814, 0.9314]$ | $< 0.0001$ | 1.51x |
| `M7a. M6 + Specular Glare Inpainting Only` | 1,946 | $0.8894 \pm 0.0337$ | 92.33% | 0.9324 | 0.8996 | 0.9092 | $[0.8824, 0.9330]$ | $< 0.0001$ | 2.58x |
| `M7b. M6 + CLAHE Contrast Only` | 1,946 | $0.8824 \pm 0.0362$ | 92.19% | 0.9307 | 0.8978 | 0.9074 | $[0.8797, 0.9312]$ | $< 0.0001$ | 1.70x |
| `M7c. M6 + Morphological Top/Black-Hat Only` | 1,946 | $0.8892 \pm 0.0357$ | 92.05% | 0.9276 | 0.8967 | 0.9053 | $[0.8784, 0.9297]$ | $< 0.0001$ | 2.06x |
| `M7d. M6 + Quality-Gated Conditional Preproc` | 1,946 | $0.8735 \pm 0.0287$ | 91.07% | 0.9205 | 0.8862 | 0.8964 | $[0.8696, 0.9216]$ | 0.0016 | 1.60x |
| `M8. M7d + Image Quality Vector q` | 1,953 | $0.8711 \pm 0.0302$ | 90.93% | 0.9187 | 0.8850 | 0.8951 | $[0.8680, 0.9198]$ | 0.0022 | 1.67x |
| `M9. M8 + Multi-Grid Chromatic-Texture (MGCT)` | 2,003 | $0.8935 \pm 0.0291$ | **92.05%** | 0.9309 | 0.9180 | **0.9236** | $[0.9012, 0.9423]$ | 0.0001 | 1.87x |
| **`M10. M9 + Block-PCA Compression (Proposed Stage-1)`** | **303** | **$0.9713 \pm 0.0064$** | **97.91%** | **0.9824** | **0.9758** | **0.9789** | **$[0.9656, 0.9891]$** | **$< 0.0001$** | **1.85x** |
| `M11. M10 on Segmented Organ ROI (No-BG Audit)` | 303 | $0.9375 \pm 0.0078$ | **96.37%** | 0.9537 | 0.9570 | **0.9549** | $[0.9363, 0.9715]$ | $< 0.0001$ | 2.44x |

---

## 5. Pillar IV: Temperature-Calibrated Two-Stage Confidence-Gated Classification

### 5.1 Mathematical Formulation of Selective Two-Stage Routing (`CalibratedConfidenceGatedSystem`)

While the linear Stage-1 specialist ($M_{10}$) classifies $97.91\%$ of test specimens accurately in $0.48\text{ ms}$ of classifier time, a small fraction ($\sim 1.5\%\text{--}4.3\%$) of specimens exhibit early-stage or overlapping symptoms (e.g., `Anthracnose` vs. `Die_Back` necrosis, or `Cutting_Weevil` vs. `Bacterial_Canker` margin damage) that lie near linear decision boundaries. Conversely, running a heavy nonlinear ensemble on *every* frame would multiply model size by $8.5\times$ and degrade probability calibration ($\text{ECE} = 0.0387$).

We formulate inference as a **Selective Confidence-Gated Cascade** ([`src/mango_edge_cv/models.py`](src/mango_edge_cv/models.py), Figure 10):

1. **Stage-1 Multinomial L2 Logistic Regression with NLL Temperature Scaling:**
   Given $\mathbf{z} \in \mathbb{R}^{303}$, Stage 1 computes linear logits $\mathbf{s}_1(\mathbf{z}) = \mathbf{W}_1 \mathbf{z} + \mathbf{b}_1 \in \mathbb{R}^8$ ($C = 0.35$, `lbfgs`). To ensure that the maximum predicted probability $\hat{p}_{\max}(\mathbf{z})$ reflects true empirical accuracy, a scalar temperature $T > 0$ is optimized on the validation split $\mathcal{D}_{\text{val}}$ by minimizing cross-entropy Negative Log-Likelihood (NLL) without altering $\arg\max_c s_{1,c}(\mathbf{z})$:
   $$T^* = \arg\min_{T \in [0.05, 5.0]} -\frac{1}{|\mathcal{D}_{\text{val}}|} \sum_{(\mathbf{z}_i, y_i) \in \mathcal{D}_{\text{val}}} \log \frac{\exp\!\big(s_{1,y_i}(\mathbf{z}_i) / T\big)}{\sum_{k=1}^{8} \exp\!\big(s_{1,k}(\mathbf{z}_i) / T\big)} = 0.758$$
   yielding calibrated Stage-1 posteriors $P_1(y = c \mid \mathbf{z}) = \text{softmax}(\mathbf{s}_1(\mathbf{z}) / T^*)_c$ with Expected Calibration Error $\text{ECE} = 0.0094$.

2. **Confidence Gate and Stage-2 Heterogeneous Nonlinear Specialist Ensemble:**
   Let $\hat{p}_1(\mathbf{z}) = \max_{c \in \{1,\dots,8\}} P_1(y = c \mid \mathbf{z})$. For a confidence threshold $\tau_{\text{gate}} \in [0, 1]$ selected on $\mathcal{D}_{\text{val}}$ ($\tau_{\text{gate}} = 0.65$):
   - **Fast Path ($\hat{p}_1(\mathbf{z}) \ge \tau_{\text{gate}}$, $98.47\%$ of test samples):** Immediately outputs $\hat{y} = \arg\max_c P_1(y=c\mid\mathbf{z})$ with zero Stage-2 computation.
   - **Slow Path ($\hat{p}_1(\mathbf{z}) < \tau_{\text{gate}}$, $1.53\%$ of test samples):** Invokes three diverse nonlinear learners on $\mathbf{z} \in \mathbb{R}^{303}$:
     - **Platt-Calibrated RBF-SVM ($w_{\text{svm}} = 0.55$):** `SVC(kernel='rbf', C=4.0, gamma='scale')` wrapped in 3-fold sigmoid `CalibratedClassifierCV`, capturing localized radial lesion boundaries.
     - **Single-Hidden-Layer Neural MLP ($w_{\text{mlp}} = 0.30$):** `MLPClassifier(hidden_layer_sizes=(128,), activation='relu', alpha=1e-3, early_stopping=True)`, capturing smooth cross-branch feature interactions.
     - **Distance-Weighted Cosine-kNN ($w_{\text{knn}} = 0.15$):** `KNeighborsClassifier(n_neighbors=11, weights='distance', metric='cosine')`, providing non-parametric neighborhood retrieval.
     The Stage-2 posterior $P_2(y\mid\mathbf{z}) = 0.55\,P_{\text{svm}} + 0.30\,P_{\text{mlp}} + 0.15\,P_{\text{knn}}$ is softly blended with Stage 1 ($\beta = 0.70$):
     $$P_{\text{final}}(y \mid \mathbf{z}) = \begin{cases} P_1(y \mid \mathbf{z}) & \text{if } \max_c P_1(y=c \mid \mathbf{z}) \ge \tau_{\text{gate}}, \\ (1 - \beta)\,P_1(y \mid \mathbf{z}) + \beta\,P_2(y \mid \mathbf{z}) & \text{if } \max_c P_1(y=c \mid \mathbf{z}) < \tau_{\text{gate}}. \end{cases}$$

![Figure 10: Temperature-Calibrated Two-Stage Confidence-Gated Classification Architecture](results/figures/Fig10_TwoStage_Confidence_Gate_Architecture.png)

### 5.2 Threshold Sweep, Probability Calibration (ECE), and Selective Risk (AURC) (Table 4)

We evaluate probability calibration via $15$-bin Expected Calibration Error ($\text{ECE} = \sum_{m=1}^{15} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$) and selective classification quality via the **Area Under the Risk–Coverage Curve (AURC)**, defined by sorting test predictions by confidence $\hat{p}(\mathbf{z}_i)$ and integrating empirical $0\text{--}1$ error across coverage levels $c \in (0, 1]$. Figure 11 and Table 4 report the complete operating trajectory across $\tau_{\text{gate}} \in [0.0, 1.0]$.

![Figure 11: Selective Stage-2 Routing Rate vs. Accuracy/Macro-F1 and Calibration ECE/AURC](results/figures/Fig11_Gate_Threshold_Sweep_and_Calibration.png)

**Table 4: Two-Stage Confidence-Gated Routing Sweep, Calibration ECE, and Selective Risk AURC ([`Table4_Confidence_Gated_and_Calibration.csv`](results/tables/Table4_Confidence_Gated_and_Calibration.csv))**

| Routing Mode / Operating Point | Gate Threshold ($\tau_{\text{gate}}$) | Val Selected | Stage-2 Routed Samples (%) | Group-Split Test Acc (%) | Group-Split Macro-F1 | 95% Bootstrap CI (Macro-F1) | Calibration ECE ($\downarrow$) | Selective Risk AURC ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `Stage-1 LR Only (Fast Path Always)` | 0.00 | False | **0.00%** | 97.91% | 0.9789 | $[0.9655, 0.9891]$ | **0.0094** | 0.00164 |
| `Confidence-Gated (tau=0.60)` | 0.60 | False | 0.98% | 98.33% | 0.9825 | $[0.9697, 0.9921]$ | 0.0108 | 0.00160 |
| **`Confidence-Gated (tau=0.65, Val-Selected)`** | **0.65** | **True** | **1.53%** | **98.47%** | **0.9836** | **$[0.9712, 0.9932]$** | **0.0104** | **0.00158** |
| `Confidence-Gated (tau=0.70)` | 0.70 | False | 1.95% | 98.61% | 0.9847 | $[0.9729, 0.9940]$ | 0.0122 | 0.00157 |
| `Confidence-Gated (tau=0.75)` | 0.75 | False | 2.23% | 98.61% | 0.9847 | $[0.9729, 0.9940]$ | 0.0120 | 0.00156 |
| `Confidence-Gated (tau=0.80)` | 0.80 | False | 3.07% | 98.61% | 0.9847 | $[0.9729, 0.9940]$ | 0.0124 | 0.00157 |
| **`Confidence-Gated (tau=0.85, High-Sensitivity)`** | **0.85** | **False** | **4.32%** | **98.74%** | **0.9859** | **$[0.9742, 0.9952]$** | **0.0144** | **0.00151** |
| `Confidence-Gated (tau=0.90)` | 0.90 | False | 5.86% | 98.74% | 0.9859 | $[0.9742, 0.9952]$ | 0.0142 | 0.00151 |
| `Confidence-Gated (tau=0.95)` | 0.95 | False | 7.53% | 98.74% | 0.9859 | $[0.9742, 0.9952]$ | 0.0173 | 0.00151 |
| `Full Stage-2 Ensemble (Uncalibrated Slow Path)` | 1.00 | False | 100.00% | 98.74% | 0.9859 | $[0.9742, 0.9952]$ | 0.0387 | 0.00259 |
| `[REFERENCE] Leakage-Sensitive Random-Split` | 0.70 | True | 1.50% | *99.00%* | *0.9900* | $[0.9828, 0.9963]$ | 0.0106 | 0.00046 |

---

## 6. Pillar V: Multi-Severity Optical Robustness and Single-Core Edge CPU Profiling

### 6.1 Multi-Severity Optical and Sensor Perturbation Benchmark (Table 5)

To verify that our quality-gated preprocessing (Pillar II) and energy-balanced Block-PCA representation (Pillar III) generalize under harsh field imaging conditions, we evaluate four controlled pipelines across **7 multi-severity optical and sensor degradations** on the held-out group-split test set (`run_perturbation_benchmark` in [`src/mango_edge_cv/evaluation.py`](src/mango_edge_cv/evaluation.py), Figure 12 and Table 5).

![Figure 12: Multi-Severity Optical and Sensor Perturbation Robustness Across 7 Degradations](results/figures/Fig12_MultiSeverity_Perturbation_Robustness.png)

**Table 5: Multi-Severity Optical and Sensor Perturbation Benchmark (Group-Split Test Macro-F1, [`Table5_MultiSeverity_Robustness.csv`](results/tables/Table5_MultiSeverity_Robustness.csv))**

| Perturbation Scenario | 1. Baseline Raw (`M4`, HSV+LBP+HOG) | 2. Unconditional CLAHE | 3. Conditional Blur Control | 4. Proposed Quality-Gated | Absolute Gain vs. Raw ($\Delta\text{F1}$) | Selected Mode |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `Clean held-out test (sigma=0)` | 0.8519 | 0.9841 | 0.8519 | **0.9836** | **+0.1317** | `adaptive` |
| `Gaussian sensor noise (sigma=5)` | 0.7079 | 0.8045 | 0.7079 | **0.9249** | **+0.2170** | `adaptive` |
| `Gaussian sensor noise (sigma=10)` | 0.5114 | *0.4207* | 0.5758 | **0.8647** | **+0.3533** | `adaptive` |
| `Gaussian sensor noise (sigma=20)` | 0.3737 | *0.2373* | 0.3641 | **0.6104** | **+0.2367** | `adaptive` |
| `Synthetic specular glare (radius=22 px)` | 0.7601 | 0.9818 | 0.7601 | **0.9812** | **+0.2211** | `adaptive` |
| `Synthetic specular glare (radius=35 px)` | 0.7412 | 0.9785 | 0.7412 | **0.9770** | **+0.2359** | `adaptive` |
| `Low-light contrast loss (-40%)` | 0.6641 | 0.8071 | 0.6641 | **0.8012** | **+0.1371** | `adaptive` |

> **Why Unconditional CLAHE Fails Under Sensor Grain:** On clean images ($\sigma=0$), unconditional CLAHE appears competitive ($0.9841$). However, under moderate ($\sigma=10$) and severe ($\sigma=20$) sensor noise, unconditional CLAHE amplifies random pixel fluctuations across local $8\times8$ tiles, corrupting LBP micro-patterns and Sobel orientations and plunging Macro-F1 to **$0.4207$** and **$0.2373$** (worse than doing nothing). Our Donoho–Johnstone $\hat{\sigma}_{\text{MAD}}$ gate detects the elevated residual variance, suppresses the grain prior to feature extraction, and achieves **$0.8647$** ($+0.4440$ above unconditional CLAHE at $\sigma=10$).

### 6.2 Fair Single-Core Edge CPU Latency and Footprint Benchmark (Table 6)

All models are benchmarked end-to-end (including image resize, preprocessing, feature extraction, and classifier forward pass) on a **single CPU thread (`batch_size = 1`)** without GPU acceleration (`run_cpu_latency_benchmark` in [`src/mango_edge_cv/evaluation.py`](src/mango_edge_cv/evaluation.py)). Figure 13 and Table 6 compare our Stage-1 and Two-Stage pipelines against transfer-learned deep CNN baselines (`MobileNetV3-Small`, `MobileNetV2`, and `ResNet50`).

![Figure 13: Single-Core Edge CPU Latency Breakdown and Footprint Comparison](results/figures/Fig13_CPU_Latency_Breakdown_and_Edge_Footprint.png)

**Table 6: Fair End-to-End Single-Core CPU Latency, Memory Footprint, and Accuracy Comparison ([`Table6_Fair_EndToEnd_CPU_Benchmark.csv`](results/tables/Table6_Fair_EndToEnd_CPU_Benchmark.csv))**

| Model / Pipeline Architecture | Execution Backend | End-to-End CPU Mean Latency (ms) | Latency Std (ms) | p50 / p95 Latency (ms) | Throughput (img/s) | Model Size on Disk (KB) | Group-Split Test Acc (%) | Group-Split Macro-F1 | External `MLDID` 0-Shot Acc (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`1. Proposed Stage-1 (Block-PCA + L2-LR)`** | `OpenCV + NumPy + Sklearn (1- thread)` | **21.58 ms** | **2.21** | **21.17 / 24.48** | **46.3** | **480.2 KB** | **97.91%** | **0.9789** | **42.63%** |
| **`2. Proposed Two-Stage Confidence-Gated`** | `OpenCV + NumPy + Sklearn (1-thread)` | **21.74 ms** | **2.65** | **21.22 / 24.91** | **46.0** | **4,060.6 KB** | **98.47%** | **0.9836** | **42.63%** |
| `3. MobileNetV3-Small (Transfer-Learned)` | `PyTorch / ONNX CPU (1-thread FP32)` | 48.70 ms | 4.85 | 47.90 / 56.40 | 20.5 | 5,900.0 KB | 98.19% | 0.9815 | 38.40% |
| `4. MobileNetV2 (Transfer-Learned)` | `PyTorch / ONNX CPU (1-thread FP32)` | 56.42 ms | 5.60 | 55.20 / 65.10 | 17.7 | 8,500.0 KB | 98.47% | 0.9842 | 39.80% |
| `5. ResNet50 (Transfer-Learned)` | `PyTorch / ONNX CPU (1-thread FP32)` | 272.04 ms | 19.40 | 268.50 / 305.20 | 3.7 | 90,000.0 KB | 98.61% | 0.9855 | 41.10% |

---

## 7. Pillar VI: External Cross-Dataset Generalization (`MangoLeafBD` to `MLDID`)

### 7.1 Mathematical Derivation of Eigenvector Canonical Leaf Pose Normalization

When models trained on `MangoLeafBD` ($N = 4,000$, Bangladesh, mixed $320\times240$ / $240\times240$ / $240\times320$ frames) are evaluated zero-shot on the independent **Mango Leaf Disease Identification Dataset (`MLDID`, $N = 3,000$, India, standardized $512\times512$ frames across 5 shared classes: `Anthracnose`, `Bacterial_Canker`, `Die_Back`, `Gall_Midge`, `Healthy`)**, source-view accuracy drops to **$31.27\%$ ($0.2886\text{ Macro-F1}$)**.

Why does zero-shot transfer fail on raw frames? In `MangoLeafBD`, leaf orientation is strongly correlated with camera aspect ratio (`Gall_Midge` and `Sooty_Mold` are horizontal $320\times240$; `Bacterial_Canker` is vertical $240\times320$). Because `f_patch5`, `f_mgct`, and `f_hog` use fixed spatial grids and orientation bins ($[0^\circ, 180^\circ)$), an external `MLDID` leaf photographed vertically or diagonally activates completely different HOG orientation bins and quadrant statistics than a horizontal training leaf!

To achieve rotation and aspect-ratio invariance prior to feature extraction, `canonicalize_leaf_organ_image` ([`src/mango_edge_cv/preprocessing.py`](src/mango_edge_cv/preprocessing.py), Figure 14) maps every leaf into a **Canonical Horizontal Organ Coordinate Frame**:

1. **Spatial Covariance Moment Calculation:** Given the binary foreground leaf mask $M_{\text{fg}}(x,y) \in \{0, 1\}$ with foreground pixel coordinates $\Omega_{\text{fg}} = \{(x_i, y_i)\}_{i=1}^{K}$ ($K = |\Omega_{\text{fg}}|$), we compute the centroid $(\bar{x}, \bar{y})$ and second-order central spatial moments:
   $$\bar{x} = \frac{1}{K}\sum_{i=1}^{K} x_i, \quad \bar{y} = \frac{1}{K}\sum_{i=1}^{K} y_i, \quad \mu_{20} = \frac{1}{K}\sum_{i=1}^{K}(x_i - \bar{x})^2, \quad \mu_{02} = \frac{1}{K}\sum_{i=1}^{K}(y_i - \bar{y})^2, \quad \mu_{11} = \frac{1}{K}\sum_{i=1}^{K}(x_i - \bar{x})(y_i - \bar{y})$$
2. **Principal Eigenvector Axis Alignment:** The angle $\theta \in (-\frac{\pi}{2}, \frac{\pi}{2}]$ of the principal elongation eigenvector of the $2\times2$ spatial covariance matrix $\Sigma_{xy} = \begin{pmatrix} \mu_{20} & \mu_{11} \\ \mu_{11} & \mu_{02} \end{pmatrix}$ is given in closed form by:
   $$\theta = \frac{1}{2}\operatorname{atan2}\!\big(2\mu_{11},\; \mu_{20} - \mu_{02}\big)$$
   When the leaf exhibits non-degenerate elongation ($\lambda_{\max} / \lambda_{\min} \ge 1.12$), both the RGB image and foreground mask are rotated by $-\theta$ about $(\bar{x}, \bar{y})$ via an affine transformation matrix $R_{-\theta} \in \mathbb{R}^{2\times3}$, aligning the leaf's midrib strictly along the horizontal axis ($0^\circ$).
3. **Aspect-Preserving Tight Organ Crop and Foreground $L^*$ Constancy:** We extract the tight axis-aligned bounding box of the rotated mask $M_{\text{fg}}^{\text{rot}}$ with a $6\%$ padding margin, replace all exterior background pixels with a neutral constant gray $[232, 232, 232]$ (eliminating studio background color shifts), and linearly normalize foreground CIE $L^*$ percentiles $[P_5(L^*_{\text{fg}}), P_{95}(L^*_{\text{fg}})] \to [18, 238]$.

![Figure 14: Eigenvector Canonical Leaf Pose Normalization on External MLDID](results/figures/Fig14_CrossDataset_Canonical_Pose_and_CORAL.png)

### 7.2 Quantitative Cross-Dataset Validation and Analysis of Unsupervised CORAL (Table 7)

Table 7 reports both in-domain performance on the 5 shared classes of `MangoLeafBD` ($N = 455$ group-split test images) and external cross-dataset transfer to all $N = 3,000$ images of `MLDID` ($600$ images per class).

**Table 7: External Cross-Dataset Generalization (`MangoLeafBD` $\to$ `MLDID`, $N = 3,000$, 5 Shared Classes, [`Table7_External_Dataset_Validation.csv`](results/tables/Table7_External_Dataset_Validation.csv))**

| Evaluation Track & Protocol | Evaluated Samples ($N$) | Shared Classes | Accuracy (%) | Balanced Acc (%) | Macro-F1 | 95% Bootstrap CI (Macro-F1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `1. Source-View In-Domain Group-Split Test` | 455 | 5 | **99.34%** | 99.30% | **0.9929** | $[0.9841, 1.0000]$ |
| `2. Canonical-Pose In-Domain Group-Split Test` | 455 | 5 | **96.04%** | 95.68% | **0.9589** | $[0.9394, 0.9762]$ |
| `3. External MLDID (Zero-Shot Source-View, Unaligned)` | 3,000 | 5 | 31.27% | 31.27% | 0.2886 | $[0.2767, 0.3005]$ |
| **`4. External MLDID (Zero-Shot Canonical Leaf Pose, Proposed)`** | **3,000** | **5** | **42.63%** | **42.63%** | **0.4395** | **$[0.4247, 0.4527]$** |
| `5. External MLDID (Canonical Pose + Unsupervised Block-CORAL)` | 3,000 | 5 | 31.43% | 31.43% | 0.3274 | $[0.3121, 0.3424]$ |

> **Rigorous Analysis of Why Unsupervised CORAL Underperforms (31.43% vs. 42.63%):** We also investigated transductive Block-Shrinkage CORAL ($\mathbf{X}_{\text{aligned}} = (\mathbf{X}_T - \boldsymbol{\mu}_T)\,\hat{\Sigma}_T^{-1/2}\hat{\Sigma}_S^{1/2} + \boldsymbol{\mu}_S$ with Ledoit–Wolf shrinkage $\lambda = 0.15$). While CORAL improves accuracy slightly over raw unaligned frames ($31.43\%$ vs. $31.27\%$), it degrades canonical-pose performance ($31.43\%$ vs. $42.63\%$). This provides an important methodological insight: unsupervised second-order moment matching assumes that source and target feature covariances differ primarily by a global linear sensor shift. When target images exhibit complex, non-Gaussian background clutter and class-conditional lesion scale differences (e.g., diffuse `MLDID` `Bacterial_Canker` vs. localized `MangoLeafBD` lesions), forcing global covariance equality rotates discriminative lesion hyperplanes into alignment with background noise. In contrast, geometric **Canonical Leaf Pose Normalization** operates on the physical organ prior to feature extraction, delivering a **+11.36% accuracy and +15.09% Macro-F1 zero-shot improvement**.

---

## 8. Exploratory Data Analysis (EDA) of Geometry and Cross-Domain Shifts

To provide full transparency into the physical properties of both datasets, [`src/mango_edge_cv/eda.py`](src/mango_edge_cv/eda.py) measures image-level geometry, file size, CIE $L^*$ brightness and contrast ($P_{95} - P_5$), HSV saturation, RGB/LAB channel medians, Laplacian variance sharpness, and Shannon grayscale entropy across all $7,000$ images (`results/eda/*.csv`).

### 8.1 Native Resolution Confounding in `MangoLeafBD` ($N = 4,000$)

As shown in Figure 15 and [`MangoLeafBD_ResolutionSummary.csv`](results/eda/MangoLeafBD_ResolutionSummary.csv), `MangoLeafBD` is not geometrically uniform: it comprises **$1,415$ Landscape ($320\times240$) images**, **$1,187$ Square ($240\times240$) images**, **$1,271$ Portrait ($240\times320$ / $240\times245$ / $240\times293$) images**, and $127$ miscellaneous frames. Critically, native resolution aligns almost deterministically with disease class (`Cutting_Weevil`, `Die_Back`, `Powdery_Mildew`, and `Anthracnose` are predominantly $240\times240$; `Gall_Midge` and `Sooty_Mold` are $320\times240$; `Bacterial_Canker` is $240\times320$). This explains why View 5 in Table 2 achieves $59.14\%$ accuracy from metadata and aspect ratio alone, and underscores why resizing to a square grid without canonical organ alignment creates severe orientation bias.

![Figure 15: Class-Wise Specimen Morphology, Resolution Confounding, and Chromatic Statistics on MangoLeafBD](results/figures/Fig15_EDA_MangoLeafBD_EightClass_Specimens_and_Geometry.png)

### 8.2 Cross-Dataset Optical Distribution Shift (`MangoLeafBD` vs. `MLDID`)

Figure 16 compares the class-wise median CIE $L^*$ dynamic range ($P_{95} - P_5$) and high-frequency optical sharpness (Laplacian variance on the $128\times128$ grid) between `MangoLeafBD` ($N = 4,000$) and `MLDID` ($N = 3,000$, where all $3,000$ images have standardized $512\times512$ resolution). Notice the sharp inversion in contrast and sharpness across classes—for example, `Bacterial_Canker` has low contrast ($107.0$) and high sharpness ($196.7$) in `MangoLeafBD` but high contrast ($160.0$) in `MLDID`, whereas `Healthy` leaves drop in sharpness from $199.9$ in `MangoLeafBD` to $104.9$ in `MLDID`.

![Figure 16: Cross-Dataset Luminance Contrast and Optical Sharpness Shift (MangoLeafBD vs. MLDID)](results/figures/Fig16_EDA_CrossDataset_Distribution_Shift_MangoLeafBD_vs_MLDID.png)

---

## 9. Positioning Against Recent Published Literature (2023–2026)

Table 8 contrasts our audit-first framework against recent deep learning studies on mango leaf disease classification (`2023–2026`). While prior works focus on architectural complexity under standard random splits, our work is the first to simultaneously enforce specimen-cluster disjointness ($0.0\%$ leakage), quantify studio background shortcuts across 5 views, evaluate multi-severity optical robustness, validate zero-shot transfer on $3,000$ external images, and achieve sub-$22\text{ ms}$ single-core CPU inference in a $480.2\text{ KB}$ footprint.

**Table 8: Comparison with Recent Mango Leaf Disease Classification Literature ([`Table8_Published_Literature_Comparison.csv`](results/tables/Table8_Published_Literature_Comparison.csv))**

| Study / Reference | Year | Dataset ($N$, Classes) | Core Model Family | Split Protocol & Leakage Audit | Background Shortcut Audit | Reported In-Domain Acc (%) | Model Size / Params | Edge CPU Latency | External Cross-Dataset Validation |
| :--- | :---: | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| ** Ali et al. (*MangoLeafBD*)** | 2022 | `MangoLeafBD` ($4,000$, 8 cls) | CNN / ResNet50 / VGG16 | Random split (No specimen audit) | No | $97.00\text{--}99.00\%$ | $\sim 23\text{--}138\text{M}$ params | Not profiled | No |
| ** Rizvee et al. (*LeafNet*)** | 2023 | `MangoLeafBD` ($4,000$, 8 cls) | Custom Deep CNN | Random split (No specimen audit) | No | $99.55\%$ | $\sim 14.5\text{ MB}$ | Not profiled | No |
| ** Salamai et al.** | 2023 | `MangoLeafBD` ($4,000$, 8 cls) | Vision Transformer (ViT) | Random split (No specimen audit) | No | $99.25\%$ | $>80\text{ MB}$ | GPU required | No |
| ** Gautam et al.** | 2024 | `MangoLeafBD` ($4,000$, 8 cls) | Stacking Ensemble CNN | Random split (No specimen audit) | No | $98.88\%$ | $>50\text{ MB}$ | Not profiled | No |
| ** Bezabh et al.** | 2024 | `MangoLeafBD` + Custom | Hybrid CNN-ViT | Random split (No specimen audit) | No | $99.10\%$ | $>45\text{ MB}$ | Not profiled | No |
| ** Proposed Stage-1 (`M10`)** | **2026** | `MangoLeafBD` ($4,000$) + `MLDID` ($3,000$) | **Quality-Gated 7-Branch + Block-PCA + Calibrated L2-LR** | **Graph Specimen Audit ($2,554$ clusters, $0\%$ leak)** | **5-View Shortcut Audit (Table 2)** | **97.91%** *(Random: 98.88%)* | **480.2 KB** | **21.58 ms** *(46.3 FPS)* | **`MLDID` ($N=3,000$): 42.63% Zero-Shot** |
| ** Proposed Two-Stage Gated** | **2026** | `MangoLeafBD` ($4,000$) + `MLDID` ($3,000$) | **Stage-1 L2-LR + Selective Stage-2 ($\tau=0.65 / 0.85$)** | **Graph Specimen Audit ($2,554$ clusters, $0\%$ leak)** | **5-View Shortcut Audit (Table 2)** | **98.47% / 98.74%** *(Random: 99.00%)* | **4.06 MB** *(1.53% routed)* | **21.74 ms** *(46.0 FPS)* | **`MLDID` ($N=3,000$): +11.36% Pose Gain** |

---

## 10. Repository Structure and Reproducibility Quickstart

### 10.1 Directory Organization

```text
Mango-Disease-Audit-EdgeAI/
├── README.md                                  # Comprehensive academic monograph & visual documentation
├── LICENSE                                    # MIT License
├── CITATION.cff                               # Formal citation metadata
├── requirements.txt                           # Locked Python dependencies
├── pyproject.toml                             # Installable package configuration (`mango_edge_cv`)
├── configs/
│   └── default_v5_config.yaml                 # Locked Version 5 hyperparameters (SEED = 20261022)
├── src/
│   └── mango_edge_cv/
│       ├── __init__.py                        # Package exports
│       ├── config.py                          # Global constants, class mappings, and locked parameters
│       ├── audit.py                           # Graph-based pHash/HSV specimen clustering & 5-view audit
│       ├── preprocessing.py                   # Hybrid Otsu-Chroma ROI segmentation, MAD/Glare/CLAHE gates & Canonical Pose
│       ├── features.py                        # Shared-Sobel 7-branch feature engine (2,003D) & 256-entry LUT LBP
│       ├── models.py                          # BlockPCACompressor (303D), Temperature Scaling & Two-Stage Confidence Gate
│       ├── evaluation.py                      # Bootstrap CIs, exact McNemar test, ECE, AURC, perturbation & CPU benchmarks
│       ├── eda.py                             # Dataset-wide geometry, chromaticity, contrast & sharpness profiling
│       └── visualization.py                   # High-legibility publication analytical charts (Fig03, Fig07b, Fig09, Fig11, Fig12, Fig16)
├── scripts/
│   ├── verify_locked_results.py               # Verifies 57/57 ground-truth v5 metrics across Tables 1-8 & models
│   ├── predict_single_image.py                # Single-image CLI inference with latency & routing diagnostics
│   ├── generate_paper_assets.py               # Generates LaTeX booktabs tables & Matplotlib analytical charts
│   ├── render_publication_diagrams.py         # Renders high-DPI HTML/OpenCV architectural & visual proof diagrams
│   └── run_full_audit_and_ablation.py         # End-to-end training, auditing & evaluation runner
├── results/
│   ├── tables/                                # Locked Version 5 CSV tables (Table1_*.csv through Table8_*.csv)
│   ├── figures/                               # Complete 16-figure high-legibility visual suite (Fig01-Fig16 & Figure1-Figure7)
│   ├── eda/                                   # Exploratory Data Analysis summary CSVs for MangoLeafBD & MLDID
│   ├── manifests/                             # Sanitized manifest_MANGO_audited.csv (4,000 rows, 2,554 cluster IDs)
│   └── models/                                # Pretrained Stage-1 (480.2 KB) & Two-Stage Gated (4.0 MB) joblib bundles
├── docs/
│   ├── DATASET_PREPARATION.md                 # Dataset acquisition, folder layout & specimen audit guide
│   ├── METHODOLOGY_AND_MATH.md                # Full mathematical derivations for all 6 methodological pillars
│   ├── EXPERIMENTAL_RESULTS_V5.md             # Complete walkthrough of locked Version 5 tables & statistical tests
│   └── REPRODUCIBILITY_GUIDE.md               # Step-by-step verification, testing & figure regeneration instructions
├── paper/
│   ├── MANUSCRIPT_OUTLINE_AND_DRAFT.md        # Full Q1 journal manuscript draft (Abstract through Conclusion)
│   └── tables_latex/                          # Publication-ready LaTeX booktabs tables (table1_audit.tex - table8_*.tex)
├── notebooks/
│   ├── Agri_EdgeAI_Mango_Research_fixed_v5.ipynb      # Core Version 5 experimental notebook
│   └── Agri_EdgeAI_Mango_Research_fixed_v8_eda.ipynb  # Exploratory Data Analysis (EDA) notebook
└── tests/
    └── test_pipeline_smoke.py                 # Automated unit & integration tests (6/6 passing)
```

### 10.2 Installation and Verification Quickstart

```bash
# 1. Clone the repository and install in editable mode
git clone https://github.com/huyxuantruong1411/Mango-Disease-Audit-EdgeAI.git
cd Mango-Disease-Audit-EdgeAI
pip install -r requirements.txt
pip install -e .

# 2. Verify all 57 locked Version 5 experimental metrics, manifests, and pretrained models
python scripts/verify_locked_results.py

# 3. Run the automated unit test suite (tests segmentation, quality gates, 2,003D features, and models)
python -m pytest tests/ -v

# 4. Run single-image CLI inference (on a real leaf image or synthetic test specimen)
python scripts/predict_single_image.py --synthetic-demo
python scripts/predict_single_image.py --image "path/to/mango_leaf.jpg" --tau-gate 0.65

# 5. Regenerate all LaTeX booktabs tables and high-legibility publication figures
python scripts/generate_paper_assets.py
python scripts/render_publication_diagrams.py
```

---

## 11. Citation and License

If you use this codebase, the specimen-cluster audit manifest (`manifest_MANGO_audited.csv`), the 5-view shortcut audit protocol, or the quality-gated two-stage edge architecture in your research, please cite:

```bibtex
@article{truong2026mango_audit_edgeai,
  title   = {Audit-First, Quality-Gated Two-Stage Framework for Real-Time Mango Leaf Disease Classification on Edge CPUs},
  author  = {Truong, Huy Xuan},
  journal = {Computers and Electronics in Agriculture (Submitted)},
  year    = {2026},
  url     = {https://github.com/huyxuantruong1411/Mango-Disease-Audit-EdgeAI}
}
```

This project is released under the [MIT License](LICENSE).
