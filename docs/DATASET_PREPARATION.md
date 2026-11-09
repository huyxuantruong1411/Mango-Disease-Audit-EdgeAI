# Dataset Preparation & Exploratory Data Analysis (EDA)

This document provides step-by-step instructions for downloading, organizing, auditing, and exploring the two mango leaf disease datasets evaluated in **Mango-Disease-Audit-EdgeAI**:
1. **Primary Benchmark (`MangoLeafBD`)**: $N = 4,000$ images across 8 classes (Bangladesh).
2. **External Cross-Dataset Benchmark (`MLDID` / `Resize mango leaf disease`)**: $N = 3,000$ images across 5 shared classes (India).

> [!IMPORTANT]
> Raw image archives are excluded from Git tracking via `.gitignore` to keep the repository lightweight. However, all pre-computed **audited manifests (`results/manifests/manifest_MANGO_audited.csv`)**, **EDA summary tables (`results/eda/*.csv`)**, and **16 publication-grade EDA figures (`results/figures/eda/*.png`)** are committed directly in the repository for immediate inspection and verification.

---

## 1. Benchmark Datasets & Official Sources

### 1.1 Primary Dataset: `MangoLeafBD` ($N = 4,000$, 8 Classes)
- **Authors**: S. I. Ahmed et al. (*Data in Brief*, Vol. 47, 108941, 2023)
- **DOI**: [`10.1016/j.dib.2023.108941`](https://doi.org/10.1016/j.dib.2023.108941) / Mendeley Data: [`10.17632/hxsnvwty3r.1`](https://doi.org/10.17632/hxsnvwty3r.1)
- **Geographic Origin**: Four mango orchards across Bangladesh (Sher-e-Bangla Agricultural University, Kashimpur, Rajshahi, and BARI).
- **Class Balance**: Exactly $500$ images per class across $8$ categories ($4,000$ total):
  1. `Anthracnose` ($500$)
  2. `Bacterial_Canker` (`Bacterial Canker`, $500$)
  3. `Cutting_Weevil` (`Cutting Weevil`, $500$)
  4. `Die_Back` (`Die Back`, $500$)
  5. `Gall_Midge` (`Gall Midge`, $500$)
  6. `Healthy` ($500$)
  7. `Powdery_Mildew` (`Powdery Mildew`, $500$)
  8. `Sooty_Mold` (`Sooty Mould`, $500$)

### 1.2 External Cross-Dataset Benchmark: `MLDID` ($N = 3,000$, 5 Shared Classes)
- **Dataset Name**: Mango Leaf Disease Identification Dataset (`MLDID`, distributed as `Resize mango leaf disease`)
- **DOI / Source**: Mendeley Data [`10.17632/jpwtpv2c4s.1`](https://doi.org/10.17632/jpwtpv2c4s.1)
- **Geographic Origin**: Indian orchard field and semi-controlled captures.
- **Class Balance**: Exactly $600$ images per class across $5$ shared categories ($3,000$ total):
  1. `Anthracnose` ($600$)
  2. `Bacterial_Canker` ($600$)
  3. `Die_Back` ($600$)
  4. `Gall_Midge` ($600$)
  5. `Healthy` ($600$)

---

## 2. Expected Local Directory Layout

Place or symlink the extracted datasets under `data/` at the repository root (or pass custom paths via `--mango-root` and `--mldid-root` to `scripts/run_full_pipeline.py`):

```text
Mango-Disease-Audit-EdgeAI/
└── data/
    ├── MangoLeafBD/
    │   ├── Anthracnose/          # 500 .jpg images
    │   ├── Bacterial Canker/     # 500 .jpg images
    │   ├── Cutting Weevil/       # 500 .jpg images
    │   ├── Die Back/             # 500 .jpg images
    │   ├── Gall Midge/           # 500 .jpg images
    │   ├── Healthy/              # 500 .jpg images
    │   ├── Powdery Mildew/       # 500 .jpg images
    │   └── Sooty Mould/          # 500 .jpg images
    └── MLDID/
        ├── Anthracnose/          # 600 .jpg images
        ├── Bacterial Canker/     # 600 .jpg images
        ├── Die Back/             # 600 .jpg images
        ├── Gall Midge/           # 600 .jpg images
        └── Healthy/              # 600 .jpg images
```

> [!NOTE]
> The label standardizer (`mango_edge_cv.config.standardize_label`) automatically normalizes folder names regardless of spaces, underscores, or British/American spelling (`Sooty Mould` $\to$ `Sooty_Mold`, `Bacterial Canker` $\to$ `Bacterial_Canker`).

---

## 3. Specimen-Cluster Leakage Audit (`MangoLeafBD`)

A foundational finding of our study is that `MangoLeafBD` contains substantial near-duplicate and zoom/rotation-augmented sibling groups that were generated prior to public release. Using our graph connected-component audit (`pHash` Hamming distance $\le 6$ or 36-bin HSV cosine similarity $\ge 0.992$), we partition all $4,000$ images into **$2,554$ disjoint specimen clusters**:

| Class ID | Class Name | Total Images | Unique Specimen Clusters (`pHash<=6` \| `HSV cos>=0.992`) | Near-Duplicate / Aug-Mix Ratio (%) | Dominant Resolution | Random-Split Test Leakage (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 0 | `Anthracnose` | 500 | 352 | 29.6% | `320x240` | 39.0% |
| 1 | `Bacterial_Canker` | 500 | 275 | 45.0% | `240x320` | 39.0% |
| 2 | `Cutting_Weevil` | 500 | 31 | **93.8%** | `240x240` | 39.0% |
| 3 | `Die_Back` | 500 | 95 | **81.0%** | `240x240` | 39.0% |
| 4 | `Gall_Midge` | 500 | 437 | 12.6% | `320x240` | 39.0% |
| 5 | `Healthy` | 500 | 469 | 6.2% | `240x320` | 39.0% |
| 6 | `Powdery_Mildew` | 500 | 427 | 14.6% | `240x240` | 39.0% |
| 7 | `Sooty_Mold` | 500 | 468 | 6.4% | `320x240` | 39.0% |
| **Total / Mean** | **8 Classes** | **4,000** | **2,554** | **36.15%** | **Mixed** | **39.0%** |

### Split Partition Summary (`results/eda/MangoLeafBD_SplitSummary.csv`)
- **Group-Disjoint Split (`StratifiedGroupKFold`, 0% Leakage)**:
  - `Train_Orig`: $2,693$ images ($1,720$ clusters)
  - `Val_Orig`: $590$ images ($373$ clusters)
  - `Test`: $717$ images ($461$ clusters)
- **Standard Stratified Random Split (39.0% Cluster Leakage Reference)**:
  - `Train_Orig`: $2,680$ images ($1,984$ clusters)
  - `Val_Orig`: $520$ images ($474$ clusters)
  - `Test`: $800$ images ($680$ clusters; $39.0\%$ share a cluster with `Train_Orig`/`Val_Orig`)

---

## 4. Exploratory Data Analysis (EDA) & Cross-Dataset Covariate Shift

### 4.1 Resolution & Aspect-Ratio Confounding (`MangoLeafBD` vs. `MLDID`)
- **Within `MangoLeafBD` (`results/eda/MangoLeafBD_ResolutionSummary.csv`)**: Image dimensions are strongly correlated with class identity (`320x240` landscape in *Anthracnose*, *Gall Midge*, and *Sooty Mold*; `240x240` square in *Cutting Weevil*, *Die Back*, and *Powdery Mildew*; `240x320` portrait in *Bacterial Canker* and *Healthy*). Consequently, file metadata and aspect ratio alone achieve **59.14% accuracy** (Table 2, View 5 vs. 12.5% chance).
- **Between `MangoLeafBD` and `MLDID` (`results/eda/MLDID_ResolutionSummary.csv`)**: While `MangoLeafBD` is predominantly landscape/square, `MLDID` consists almost exclusively of **portrait-oriented frames** (`240x293`, `240x270`, `240x285`; mean aspect ratio $0.82$ vs. $1.07$), rotated by $\sim 90^\circ$ relative to `MangoLeafBD`. This finding directly motivated our **Canonical Leaf Pose Normalization** (`mango_edge_cv.preprocessing.canonicalize_leaf_organ_image`), which boosts zero-shot cross-dataset accuracy from **31.27% to 42.63%** (+11.36% gain, Table 7).

### 4.2 Visual EDA Gallery (`results/figures/eda/`)
All 16 high-resolution EDA plots are stored in `results/figures/eda/`:
- `MangoLeafBD_EDA_Fig1_ClassAndResolution.png` & `MLDID_EDA_Fig1_ClassAndResolution.png`
- `MangoLeafBD_EDA_Fig2_AuditAndSplitLeakage.png` & `MLDID_EDA_Fig2_AuditAndSplitLeakage.png`
- `MangoLeafBD_EDA_Fig3_OpticalQualityMetrics.png` & `MLDID_EDA_Fig3_OpticalQualityMetrics.png`
- `MangoLeafBD_EDA_Fig4_ColorAndForegroundStats.png` & `MLDID_EDA_Fig4_ColorAndForegroundStats.png`
- `MangoLeafBD_EDA_Fig5_CorrelationAndPCA.png` & `MLDID_EDA_Fig5_CorrelationAndPCA.png`
- `MangoLeafBD_EDA_Fig6_SampleGallery.png` & `MLDID_EDA_Fig6_SampleGallery.png`
- `MangoLeafBD_EDA_Fig7_ROISegmentationGallery.png` & `MLDID_EDA_Fig7_ROISegmentationGallery.png`
- `MangoLeafBD_EDA_Fig8_DuplicateClusterPairs.png` & `MLDID_EDA_Fig8_DuplicateClusterPairs.png`
