# Reproducibility & Verification Guide

This guide explains how to reproduce and verify every table, figure, LaTeX asset, and model prediction in **Mango-Disease-Audit-EdgeAI** (`SEED = 20261022`), both **without** downloading raw image archives (using the committed manifests and locked Version 5 artifacts) and **with** the raw datasets (`MangoLeafBD` and `MLDID`).

---

## 1. Environment Setup

We recommend Python `3.10+` (`3.10`–`3.13` tested). From the repository root:

```bash
# Install core scientific dependencies and editable package
pip install -r requirements.txt
pip install -e .
```

---

## 2. Instant Verification of Locked Version 5 (`v5`) Results (< 5 Seconds)

You can verify all 42 ground-truth manuscript metrics across `manifest_MANGO_audited.csv` and `Table1`–`Table7` without needing the raw image datasets:

```bash
python scripts/verify_locked_results.py
```

Expected output:
```text
============================================================================================
LOCKED VERSION 5 (v5) GROUND-TRUTH METRIC VERIFICATION REPORT (SEED = 20261022)
============================================================================================
[PASS] | Manifest: 4,000 total images across 8 classes            | rows=4000, classes=8
[PASS] | Manifest: 2,554 unique specimen clusters                 | unique_clusters=2554, expected=2554
...
[PASS] | Table 7: MLDID Canonical-pose Zero-Shot Acc (42.63%)     | actual=42.63, expected=42.63
[PASS] | Table 7: MLDID Canonical-pose Zero-Shot F1 (0.4395)      | actual=0.4395, expected=0.4395
--------------------------------------------------------------------------------------------
Summary: 42/42 checks passed.
============================================================================================
```

---

## 3. Running Unit & Smoke Tests (`pytest`)

Our unit test suite (`tests/test_pipeline_smoke.py`) validates the 256-entry uniform LBP lookup table, hybrid Otsu-Chroma segmentation, Canonical Leaf Pose Normalization, exact 7-branch feature dimensionality ($52 + 57 + 28 + 45 + 7 + 50 + 1,764 = 2,003\text{D}$), `BlockPCACompressor` ($2,003\text{D} \to 303\text{D}$), `CalibratedConfidenceGatedSystem`, and `coral_align_features`:

```bash
pytest tests/ -v
```

---

## 4. Single-Image Edge Inference Demo (`predict_single_image.py`)

Run the serialized Version 5 model (`results/models/proposed_twostage_gated.joblib`) on any local leaf image or on a synthetic demo leaf:

```bash
# Run on a synthetic mango leaf (no dataset required)
python scripts/predict_single_image.py --synthetic-demo

# Run on a real mango leaf image with Canonical Pose Normalization
python scripts/predict_single_image.py --image "data/MangoLeafBD/Anthracnose/20211008_124249 (Custom).jpg" --tau 0.65
```

---

## 5. Regenerating LaTeX Tables & Publication Figures

1. **Generate LaTeX `booktabs` Tables (`paper/tables_latex/*.tex`) & Analytical Charts (`Figure2`, `Figure4`, `Figure7`)**:
   ```bash
   python scripts/generate_paper_assets.py
   ```
2. **Render High-DPI HTML/CSS/SVG Architecture & Latency Diagrams (`Figure1`, `Figure5`, `Figure6`) via Playwright**:
   ```bash
   python scripts/render_publication_diagrams.py
   ```
   This script writes the self-contained interactive showcase to `results/figures/publication_diagrams_showcase.html` and captures 2x-DPI PNG screenshots directly to `results/figures/Figure1_Pipeline_Architecture_Overview.png`, `results/figures/Figure5_Latency_and_Resource_Profile.png`, and `results/figures/Figure6_Optical_Modes_and_Canonical_Pose.png`.

---

## 6. Full End-to-End Execution on Raw Datasets

Once `MangoLeafBD` ($4,000$ images) and `MLDID` ($3,000$ images) are placed in `data/MangoLeafBD` and `data/MLDID` (see [`docs/DATASET_PREPARATION.md`](DATASET_PREPARATION.md)), run:

```bash
python scripts/run_full_pipeline.py --mango-root "data/MangoLeafBD" --mldid-root "data/MLDID" --run-eda
```

Or open the interactive research notebooks in `notebooks/`:
- [`notebooks/Agri_EdgeAI_Mango_Research_v5.ipynb`](../notebooks/Agri_EdgeAI_Mango_Research_v5.ipynb): Locked Version 5 core experimental benchmark.
- [`notebooks/Agri_EdgeAI_Mango_Research_v8_eda.ipynb`](../notebooks/Agri_EdgeAI_Mango_Research_v8_eda.ipynb): Extended 8-figure Exploratory Data Analysis (`EDA_MangoLeafBD` and `EDA_MLDID`).
