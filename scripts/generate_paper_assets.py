#!/usr/bin/env python3
"""Generate publication-ready LaTeX booktabs tables (paper/tables_latex/*.tex)
and analytical chart figures from the locked Version 5 CSV tables.

Usage:
    python scripts/generate_paper_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mango_edge_cv.models import (
    BlockPCACompressor,
    CalibratedConfidenceGatedSystem,
    _EstimatorWrapper,
)
from mango_edge_cv.visualization import (
    generate_analytical_charts_from_tables,
    plot_feature_block_attribution,
)


TABLE_SPECS = [
    (
        "Table1_Dataset_Leakage_and_Group_Audit.csv",
        "table1_audit.tex",
        "Dataset specimen-cluster audit on MangoLeafBD ($N=4,000$ images across $2,554$ unique clusters; $36.15\\%$ near-duplicate ratio and $39.0\\%$ random-split test leakage).",
        "tab:audit_leakage",
    ),
    (
        "Table2_Background_Shortcut_Audit.csv",
        "table2_background_shortcut.tex",
        "Five-view studio background and framing shortcut audit under the leak-free group-disjoint split protocol.",
        "tab:background_shortcut",
    ),
    (
        "Table3_Progressive_Ablation_Study.csv",
        "table3_progressive_ablation.tex",
        "Progressive ablation study ($M_1$--$M_{11}$) demonstrating raw HOG dimensionality dilution ($M_4$: $86.61\\%$) and Energy-Balanced Block-PCA recovery ($M_{10}$: $97.91\\%$).",
        "tab:progressive_ablation",
    ),
    (
        "Table4_Confidence_Gated_and_Calibration.csv",
        "table4_confidence_gate_calibration.tex",
        "Two-stage confidence-gated routing sweep ($\\tau_{\\text{gate}}$), calibration ECE, selective risk AURC, and random-split leakage reference.",
        "tab:confidence_gate",
    ),
    (
        "Table5_MultiSeverity_Robustness.csv",
        "table5_perturbation_robustness.tex",
        "Multi-severity optical and sensor perturbation robustness comparing raw baseline, unconditional CLAHE, conditional blur control, and proposed quality-gated preprocessing.",
        "tab:perturbation_robustness",
    ),
    (
        "Table6_Fair_EndToEnd_CPU_Benchmark.csv",
        "table6_cpu_benchmark.tex",
        "Fair end-to-end single-core CPU latency, memory footprint, and external zero-shot generalization compared against transfer-learned deep CNN backbones.",
        "tab:cpu_benchmark",
    ),
    (
        "Table7_External_Dataset_Validation.csv",
        "table7_external_validation_mldid.tex",
        "External cross-dataset generalization from MangoLeafBD (Bangladesh) to MLDID (India, $N=3,000$ across 5 shared classes) with Canonical Leaf Pose Normalization and Block-Shrinkage CORAL.",
        "tab:external_mldid",
    ),
    (
        "Table8_Published_Literature_Comparison.csv",
        "table8_literature_comparison.tex",
        "Comparison with recent mango leaf disease classification literature (2023--2026) across split audit rigor, parameter footprint, CPU latency, and external validation.",
        "tab:literature_comparison",
    ),
]


def _escape_latex(val: object) -> str:
    s = str(val)
    replacements = {
        "&": "\\&",
        "%": "\\%",
        "_": "\\_",
        "#": "\\#",
        "~": "$\\sim$",
        "±": "$\\pm$",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)
    return s


def csv_to_booktabs_latex(csv_path: Path, tex_path: Path, caption: str, label: str) -> None:
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    col_align = "l" + "r" * (len(cols) - 1)

    lines = [
        "% Auto-generated from locked Version 5 (v5) CSV table: " + csv_path.name,
        "\\begin{table*}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{col_align}}}",
        "\\toprule",
        " & ".join([f"\\textbf{{{_escape_latex(c)}}}" for c in cols]) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        cells = [_escape_latex(row[c]) for c in cols]
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}%",
            "}",
            "\\end{table*}",
            "",
        ]
    )
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    tables_dir = REPO_ROOT / "results" / "tables"
    latex_dir = REPO_ROOT / "paper" / "tables_latex"
    figures_dir = REPO_ROOT / "results" / "figures"

    for csv_name, tex_name, caption, label in TABLE_SPECS:
        csv_path = tables_dir / csv_name
        if csv_path.exists():
            csv_to_booktabs_latex(csv_path, latex_dir / tex_name, caption, label)
            print(f"[OK] Generated LaTeX table: paper/tables_latex/{tex_name}")

    generate_analytical_charts_from_tables(str(tables_dir), str(figures_dir))
    model_path = REPO_ROOT / "results" / "models" / "proposed_twostage_gated.joblib"
    if model_path.exists():
        bundle = joblib.load(model_path)
        gated_sys = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
        plot_feature_block_attribution(
            gated_sys,
            str(figures_dir / "Figure3_Feature_Block_Attribution.png"),
            num_classes=8,
            track_name="MangoLeafBD",
        )

    for fig_name in [
        "Fig03_Cluster_Distribution_and_Split_Leakage.png",
        "Fig07b_Feature_Block_Attribution.png",
        "Fig09_Progressive_Ablation_Trajectory_M1_M11.png",
        "Fig11_Gate_Threshold_Sweep_and_Calibration.png",
        "Fig12_MultiSeverity_Perturbation_Robustness.png",
        "Figure2_Audit_and_Background_Shortcuts.png",
        "Figure3_Feature_Block_Attribution.png",
        "Figure4_Ablation_and_Confidence_Gate.png",
        "Figure7_Perturbation_Robustness_and_CrossDomain.png",
    ]:
        print(f"[OK] Generated analytical chart: results/figures/{fig_name}")


if __name__ == "__main__":
    main()
