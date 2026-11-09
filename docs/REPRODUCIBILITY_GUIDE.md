# Reproducibility and Verification Guide (Locked Version 5 Protocol)

This repository enforces deterministic execution (`SEED = 20261022`) across specimen-cluster auditing, train/val/test group partitioning, feature extraction, Block-PCA projection, temperature scaling, bootstrap confidence intervals ($B = 1,000$), and exact McNemar hypothesis testing.

---

## 1. Environment Installation

```bash
git clone https://github.com/huyxuantruong1411/Mango-Disease-Audit-EdgeAI.git
cd Mango-Disease-Audit-EdgeAI
pip install -r requirements.txt
pip install -e .
```

---

## 2. Automated 57-Check Ground-Truth Verification (`verify_locked_results.py`)

Run the automated verifier to confirm that all 8 CSV tables (`Table1`–`Table8`), the audited manifest (`manifest_MANGO_audited.csv`), the EDA summaries (`results/eda/*.csv`), and the pretrained model bundles (`proposed_stage1_lr.joblib` and `proposed_twostage_gated.joblib`) match the locked Version 5 numbers bit-for-bit:

```bash
python scripts/verify_locked_results.py
```

Expected output:
```text
ALL 57 LOCKED VERSION 5 (v5) GROUND-TRUTH CHECKS PASSED SUCCESSFULLY.
```

---

## 3. Unit and Smoke Test Suite (`pytest`)

Run the automated test suite to verify ROI segmentation, quality-gated preprocessing, 7-branch $2,003\text{D}$ feature extraction, `BlockPCACompressor` ($303\text{D}$), `CalibratedConfidenceGatedSystem`, and locked CSV integrity:

```bash
python -m pytest tests/ -v
```

---

## 4. Single-Image CLI Inference (`predict_single_image.py`)

Evaluate the pretrained Two-Stage Confidence-Gated model on any RGB image file or run the built-in synthetic specimen demo:

```bash
# Synthetic leaf smoke test
python scripts/predict_single_image.py --synthetic-demo

# Real mango leaf image inference with custom gate threshold
python scripts/predict_single_image.py --image "path/to/leaf.jpg" --tau-gate 0.65 --canonical-pose
```

---

## 5. Regenerating Publication Figures and LaTeX Booktabs Tables

1. **Generate LaTeX Booktabs Tables (`paper/tables_latex/*.tex`) and Analytical Charts (`Fig03`, `Fig07b`, `Fig09`, `Fig11`, `Fig12`, `Fig16`):**
   ```bash
   python scripts/generate_paper_assets.py
   ```
2. **Render High-DPI Architectural and Real-Image Visual Proof Diagrams (`Fig01`, `Fig02`, `Fig04`, `Fig05`, `Fig06`, `Fig07`, `Fig08`, `Fig10`, `Fig13`, `Fig14`, `Fig15`):**
   ```bash
   python scripts/render_publication_diagrams.py
   ```
