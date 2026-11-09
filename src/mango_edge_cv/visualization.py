"""
High-Legibility Publication-Grade Visualization Suite for Mango-Disease-Audit-EdgeAI
(`src/mango_edge_cv/visualization.py`).

All analytical figures use constrained widths (10.5-10.8 in), generous headroom,
`bbox_inches="tight"`, and large bold typography (12.5pt-16pt) so every label, tick,
bar value, and callout is effortlessly readable on GitHub without zooming.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_feature_block_attribution(
    gated_system, output_path: str, num_classes: int = 8, track_name: str = "MangoLeafBD"
):
    """Plot relative L1 weight attribution across the 7 compressed feature blocks (303D)."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    base_lr_est = gated_system.stage1_lr.calibrated_classifiers_[0].estimator
    w_abs = np.mean(np.abs(base_lr_est.coef_), axis=0)
    slices = [
        ("1. HSV Color\n(52D)", slice(0, 52)),
        ("2. CIE-LAB\n(57D)", slice(52, 109)),
        ("3. Multi-Scale\nLBP (28D)", slice(109, 137)),
        ("4. 5-Patch\nZones (45D)", slice(137, 182)),
        ("5. Quality\nVector q (7D)", slice(182, 189)),
        ("6. MGCT Grid\n(50D)", slice(189, 239)),
        ("7. Shared-Sobel\nHOG-PCA (64D)", slice(239, min(303, len(w_abs)))),
    ]
    b_names = [s[0] for s in slices]
    weight_total = float(np.sum(w_abs))
    b_pcts = [
        (float(np.sum(w_abs[s[1]])) / weight_total * 100.0) if weight_total > 0 else 0.0
        for s in slices
    ]

    fig, ax = plt.subplots(figsize=(10.8, 6.2), dpi=300)
    colors = [
        "#2563eb",
        "#d97706",
        "#16a34a",
        "#7c3aed",
        "#64748b",
        "#dc2626",
        "#0f172a",
    ]
    bars = ax.bar(
        b_names, b_pcts, color=colors, width=0.55, edgecolor="#0f172a", linewidth=1.2
    )
    for b in bars:
        h_v = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h_v + 0.7,
            f"{h_v:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=13.5,
            color="#0f172a",
        )
    ax.set_ylim(0, max(1.0, max(b_pcts) * 1.25))
    ax.set_ylabel(
        "Relative Stage-1 L1 Weight Attribution (%)",
        fontweight="bold",
        fontsize=14.0,
    )
    ax.set_title(
        f"Stage-1 Linear Classifier L1 Weight Attribution Across 7 Feature Branches\n"
        f"(303D Energy-Balanced Subspace across {num_classes} {track_name} Classes)",
        fontweight="bold",
        fontsize=15.0,
        pad=14,
    )
    ax.tick_params(axis="x", labelsize=12.0)
    ax.tick_params(axis="y", labelsize=13.0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.grid(axis="y", linestyle="--", alpha=0.45, linewidth=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.22)
    alt_path = os.path.join(
        os.path.dirname(os.path.abspath(output_path)),
        "Fig07b_Feature_Block_Attribution.png",
    )
    fig.savefig(alt_path, dpi=300, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    return b_pcts


def generate_analytical_charts_from_tables(tables_dir: str, figures_dir: str):
    """
    Generate high-legibility single-focus and vertically stacked analytical figures
    from the locked Version 5 CSV tables:
      - Fig03_Cluster_Distribution_and_Split_Leakage.png
      - Fig09_Progressive_Ablation_Trajectory_M1_M11.png
      - Fig11_Gate_Threshold_Sweep_and_Calibration.png
      - Fig12_MultiSeverity_Perturbation_Robustness.png
      - Figure2_Audit_and_Background_Shortcuts.png
      - Figure4_Ablation_and_Confidence_Gate.png
      - Figure7_Perturbation_Robustness_and_CrossDomain.png
    """
    os.makedirs(figures_dir, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"

    df_t1 = pd.read_csv(
        os.path.join(tables_dir, "Table1_Dataset_Leakage_and_Group_Audit.csv")
    )
    df_t2 = pd.read_csv(
        os.path.join(tables_dir, "Table2_Background_Shortcut_Audit.csv")
    )
    df_t3 = pd.read_csv(
        os.path.join(tables_dir, "Table3_Progressive_Ablation_Study.csv")
    )
    df_t4 = pd.read_csv(
        os.path.join(tables_dir, "Table4_Confidence_Gated_and_Calibration.csv")
    )
    df_t5 = pd.read_csv(
        os.path.join(tables_dir, "Table5_MultiSeverity_Robustness.csv")
    )
    df_t7 = pd.read_csv(
        os.path.join(tables_dir, "Table7_External_Dataset_Validation.csv")
    )

    # =========================================================================
    # 1. Fig03_Cluster_Distribution_and_Split_Leakage.png (Single-Focus Large)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10.8, 6.8), dpi=300)
    classes = [c.replace("_", "\n") for c in df_t1["Class Name"]]
    clusters = df_t1[
        "Unique Specimen Clusters (pHash<=6 | HSV cosine>=0.992)"
    ].values
    dups = 500 - clusters
    x = np.arange(len(classes))

    ax.bar(
        x,
        clusters,
        color="#1d4ed8",
        label="Unique Physical Specimen Clusters (Total = 2,554 / 4,000)",
        width=0.58,
        edgecolor="#0f172a",
        linewidth=1.0,
    )
    ax.bar(
        x,
        dups,
        bottom=clusters,
        color="#dc2626",
        alpha=0.88,
        label="Near-Duplicate / Zoom-Rotated Siblings (1,446 imgs = 36.15%)",
        width=0.58,
        edgecolor="#0f172a",
        linewidth=1.0,
    )
    for idx, (cl, ratio) in enumerate(
        zip(clusters, df_t1["Near-Duplicate / Aug-Mix Ratio (%)"])
    ):
        if cl >= 110:
            ax.text(
                idx,
                cl / 2,
                f"{cl}\nclusters",
                ha="center",
                va="center",
                fontsize=11.5,
                fontweight="bold",
                color="white",
            )
        else:
            ax.text(
                idx,
                cl + 16,
                f"{cl} cl.",
                ha="center",
                va="bottom",
                fontsize=11.5,
                fontweight="bold",
                color="white",
            )
        ax.text(
            idx,
            510,
            f"{ratio:.1f}%\ndup",
            ha="center",
            va="bottom",
            fontsize=11.5,
            fontweight="bold",
            color="#991b1b" if ratio > 50 else "#1e293b",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=12.5, fontweight="bold")
    ax.tick_params(axis="y", labelsize=13.0)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.set_ylim(0, 685)
    ax.set_ylabel("Images per Class (Raw N = 500)", fontweight="bold", fontsize=14.0)
    ax.set_title(
        "Class-Wise Unique Specimen Clusters vs. Near-Duplicate Siblings on MangoLeafBD\n"
        "(Naive Random Split Leaks 39.0% of Multi-Image Clusters vs. 0.0% in StratifiedGroupKFold)",
        fontweight="bold",
        fontsize=14.5,
        pad=14,
    )
    ax.legend(loc="upper center", fontsize=12.5, framealpha=0.96)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            figures_dir, "Fig03_Cluster_Distribution_and_Split_Leakage.png"
        ),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.22,
    )
    plt.close(fig)

    # =========================================================================
    # 2. Fig09_Progressive_Ablation_Trajectory_M1_M11.png (Single-Focus Large)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10.8, 6.8), dpi=300)
    short_labels = [c.split(".")[0] for c in df_t3["Configuration"]]
    test_f1 = df_t3["Test Macro-F1"].values * 100.0
    test_acc = df_t3["Test Acc (%)"].values
    x3 = np.arange(len(short_labels))

    ax.plot(
        x3,
        test_acc,
        marker="o",
        markersize=9,
        linewidth=3.0,
        color="#1d4ed8",
        label="Group-Split Test Accuracy (%)",
    )
    ax.plot(
        x3,
        test_f1,
        marker="s",
        markersize=8,
        linewidth=2.6,
        linestyle="--",
        color="#16a34a",
        label="Group-Split Test Macro-F1 (%)",
    )

    key_indices = {
        0: ("90.38%\n(M1 HSV)", (0.2, 3.2)),
        1: ("97.35%\n(M2 +LBP)", (0.0, 2.2)),
        2: ("83.40%\n(M3 +HOG)", (-0.1, -6.5)),
        3: ("86.61%\n(M4 Raw Concat)", (0.9, -5.5)),
        6: ("92.33%\n(M7a)", (0.0, 2.5)),
        11: ("92.05%\n(M9 2,003D Raw)", (-0.6, -5.5)),
        12: ("97.91%\n(M10 Block-PCA)", (-0.9, 2.6)),
        13: ("96.37%\n(M11 No-BG)", (-0.3, -5.5)),
    }
    for idx_pt, (txt_lbl, (dx, dy)) in key_indices.items():
        val = test_acc[idx_pt]
        col = "#dc2626" if idx_pt in (2, 3, 11) else "#0f172a"
        if idx_pt == 12:
            col = "#15803d"
        ax.annotate(
            txt_lbl,
            xy=(idx_pt, val),
            xytext=(idx_pt + dx, val + dy),
            ha="center",
            fontsize=11.5,
            fontweight="bold",
            color=col,
            arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
        )

    ax.scatter([2, 3, 11], [test_acc[2], test_acc[3], test_acc[11]], color="#dc2626", s=130, zorder=6)
    ax.scatter([12], [test_acc[12]], color="#16a34a", s=165, zorder=6)

    ax.set_xticks(x3)
    ax.set_xticklabels(short_labels, fontsize=12.0, fontweight="bold")
    ax.tick_params(axis="y", labelsize=13.0)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.set_ylim(72.0, 104.0)
    ax.set_ylabel("Leak-Free Group-Split Performance (%)", fontweight="bold", fontsize=14.0)
    ax.set_xlabel("Progressive Ablation Configuration (Table 3)", fontweight="bold", fontsize=14.0)
    ax.set_title(
        "Progressive 14-Step Ablation Trajectory (M1 to M11): Raw HOG Dilution vs. Block-PCA Recovery\n"
        "(M9 Raw 2,003D: 92.05% Acc -> M10 Energy-Balanced Block-PCA 303D: 97.91% Acc, p < 0.0001)",
        fontweight="bold",
        fontsize=14.2,
        pad=14,
    )
    ax.legend(loc="lower right", fontsize=12.5, framealpha=0.96)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        os.path.join(
            figures_dir, "Fig09_Progressive_Ablation_Trajectory_M1_M11.png"
        ),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.22,
    )
    plt.close(fig)

    # =========================================================================
    # 3. Fig11_Gate_Threshold_Sweep_and_Calibration.png (Stacked 2-Row Large)
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.8, 11.2), dpi=300)
    df_t4_group = df_t4[~df_t4["Routing Mode"].str.contains("REFERENCE")].copy()
    routed = df_t4_group["Stage-2 Routed Samples (%)"].values
    f1_g = df_t4_group["Group-Split Macro-F1"].values * 100.0
    acc_g = df_t4_group["Test Accuracy (%)"].values
    taus = df_t4_group["Gate Threshold (tau_gate)"].values

    mask_zoom = taus <= 0.95
    ax1.plot(
        routed[mask_zoom],
        acc_g[mask_zoom],
        marker="o",
        markersize=9,
        linewidth=3.0,
        color="#1d4ed8",
        label="Group-Split Test Accuracy (%)",
    )
    ax1.plot(
        routed[mask_zoom],
        f1_g[mask_zoom],
        marker="D",
        markersize=8,
        linewidth=2.8,
        linestyle="--",
        color="#7c3aed",
        label="Group-Split Test Macro-F1 (%)",
    )

    # Clean non-overlapping callouts for the 3 key operating points
    ax1.annotate(
        "Stage-1 Only (tau=0.0)\n0.0% routed | 97.91% Acc",
        xy=(0.0, 97.91),
        xytext=(0.7, 97.62),
        fontsize=12.0,
        fontweight="bold",
        color="#0f172a",
        arrowprops=dict(arrowstyle="->", color="#0f172a", lw=1.5),
    )
    ax1.annotate(
        "Val-Selected Gate (tau=0.65)\n1.53% routed | 98.47% Acc (98.36% F1)",
        xy=(1.53, 98.47),
        xytext=(0.4, 98.80),
        fontsize=12.0,
        fontweight="bold",
        color="#dc2626",
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.6),
    )
    ax1.annotate(
        "High-Sensitivity Gate (tau=0.85)\n4.32% routed | 98.74% Acc (98.59% F1)",
        xy=(4.32, 98.74),
        xytext=(4.5, 98.28),
        fontsize=12.0,
        fontweight="bold",
        color="#b45309",
        arrowprops=dict(arrowstyle="->", color="#b45309", lw=1.6),
    )

    ax1.scatter([1.53], [98.47], color="#dc2626", s=165, zorder=6, label="Val-Selected Gate (tau=0.65: 1.53% routed, 98.47% Acc)")
    ax1.scatter([4.32], [98.74], color="#d97706", s=155, zorder=6, label="High-Sensitivity Gate (tau=0.85: 4.32% routed, 98.74% Acc)")
    ax1.set_xlim(-0.4, 8.8)
    ax1.set_ylim(97.45, 99.10)
    ax1.set_xlabel("Samples Routed to Stage-2 Nonlinear Ensemble (%)", fontweight="bold", fontsize=13.5)
    ax1.set_ylabel("Group-Split Performance (%)", fontweight="bold", fontsize=13.5)
    ax1.set_title(
        "A. Selective Stage-2 Routing Percentage vs. Group-Split Accuracy & Macro-F1 (Table 4)",
        fontweight="bold",
        fontsize=14.5,
        pad=10,
    )
    ax1.tick_params(axis="both", labelsize=12.5)
    for lbl in ax1.get_xticklabels() + ax1.get_yticklabels():
        lbl.set_fontweight("bold")
    ax1.legend(loc="lower right", fontsize=11.5, framealpha=0.96)
    ax1.grid(True, linestyle="--", alpha=0.45)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel B: Calibration ECE & Selective Risk AURC Comparison
    modes_cal = [
        "Stage-1 Linear\nOnly (T*=0.758)",
        "Two-Stage Gated\n(tau=0.65, 1.53%)",
        "Two-Stage Gated\n(tau=0.85, 4.32%)",
        "Stage-2 Always\n(Uncalibrated)",
    ]
    ece_vals = [0.94, 1.04, 1.44, 3.87]
    aurc_vals = [0.164, 0.158, 0.151, 0.259]
    x_cal = np.arange(len(modes_cal))
    w_cal = 0.34
    b1 = ax2.bar(
        x_cal - w_cal / 2,
        ece_vals,
        width=w_cal,
        color="#2563eb",
        edgecolor="#0f172a",
        label="Expected Calibration Error (ECE x 100, lower is better)",
    )
    b2 = ax2.bar(
        x_cal + w_cal / 2,
        aurc_vals,
        width=w_cal,
        color="#16a34a",
        edgecolor="#0f172a",
        label="Area Under Risk-Coverage Curve (AURC x 100, lower is better)",
    )
    for b in b1:
        hv = b.get_height()
        ax2.text(
            b.get_x() + b.get_width() / 2,
            hv + 0.08,
            f"ECE:\n{hv/100:.4f}",
            ha="center",
            va="bottom",
            fontsize=12.0,
            fontweight="bold",
            color="#1e3a8a",
        )
    for b in b2:
        hv = b.get_height()
        ax2.text(
            b.get_x() + b.get_width() / 2,
            hv + 0.08,
            f"AURC:\n{hv/100:.5f}",
            ha="center",
            va="bottom",
            fontsize=12.0,
            fontweight="bold",
            color="#14532d",
        )
    ax2.set_xticks(x_cal)
    ax2.set_xticklabels(modes_cal, fontsize=12.5, fontweight="bold")
    ax2.tick_params(axis="y", labelsize=12.5)
    for lbl in ax2.get_yticklabels():
        lbl.set_fontweight("bold")
    ax2.set_ylim(0, 4.95)
    ax2.set_ylabel("Error / Selective Risk (x 100)", fontweight="bold", fontsize=13.5)
    ax2.set_title(
        "B. Probability Calibration (ECE) & Selective Risk (AURC) Across Operating Points",
        fontweight="bold",
        fontsize=14.5,
        pad=10,
    )
    ax2.legend(loc="upper left", fontsize=12.0, framealpha=0.96)
    ax2.grid(axis="y", linestyle="--", alpha=0.45)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.tight_layout(pad=2.0)
    fig.savefig(
        os.path.join(
            figures_dir, "Fig11_Gate_Threshold_Sweep_and_Calibration.png"
        ),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.22,
    )
    plt.close(fig)

    # =========================================================================
    # 4. Fig12_MultiSeverity_Perturbation_Robustness.png (Single-Focus Large)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10.8, 6.8), dpi=300)
    scen_labels = [
        "Clean\n(sigma=0)",
        "Gaussian\nsigma=5",
        "Gaussian\nsigma=10",
        "Gaussian\nsigma=20",
        "Glare\nr=22px",
        "Glare\nr=35px",
        "Low-Light\n(-40%)",
    ]
    x5 = np.arange(len(scen_labels))
    w = 0.20
    raw_v = df_t5["1. Baseline Raw (HSV+LBP+HOG)"].values
    clahe_v = df_t5["2. Unconditional CLAHE"].values
    blur_v = df_t5["3. Conditional Blur Control"].values
    prop_v = df_t5["4. Proposed Quality-Gated"].values

    ax.bar(
        x5 - 1.5 * w,
        raw_v,
        width=w,
        color="#94a3b8",
        edgecolor="#0f172a",
        label="1. Baseline Raw (M4)",
    )
    ax.bar(
        x5 - 0.5 * w,
        clahe_v,
        width=w,
        color="#f59e0b",
        edgecolor="#0f172a",
        label="2. Unconditional CLAHE",
    )
    ax.bar(
        x5 + 0.5 * w,
        blur_v,
        width=w,
        color="#3b82f6",
        edgecolor="#0f172a",
        label="3. Conditional Blur Control",
    )
    bars_prop = ax.bar(
        x5 + 1.5 * w,
        prop_v,
        width=w,
        color="#16a34a",
        edgecolor="#0f172a",
        label="4. Proposed Quality-Gated",
    )

    for b_p, p_val, r_val in zip(bars_prop, prop_v, raw_v):
        delta = p_val - r_val
        lbl_top = f"{p_val:.3f}\n(+{delta:.2f})"
        ax.text(
            b_p.get_x() + b_p.get_width() / 2,
            p_val + 0.015,
            lbl_top,
            ha="center",
            va="bottom",
            fontsize=11.0,
            fontweight="bold",
            color="#14532d",
        )

    ax.set_xticks(x5)
    ax.set_xticklabels(scen_labels, fontsize=12.5, fontweight="bold")
    ax.tick_params(axis="y", labelsize=13.0)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")
    ax.set_ylim(0.15, 1.24)
    ax.set_ylabel("Group-Split Test Macro-F1", fontweight="bold", fontsize=14.0)
    ax.set_title(
        "Multi-Severity Optical & Sensor Perturbation Robustness Across 7 Degradations (Table 5)\n"
        "(Quality-Gated Preprocessing Gains +0.3533 Macro-F1 at sigma=10 and +0.2367 at sigma=20)",
        fontweight="bold",
        fontsize=14.2,
        pad=14,
    )
    ax.legend(loc="upper center", fontsize=12.0, ncol=2, framealpha=0.96)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, "Fig12_MultiSeverity_Perturbation_Robustness.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.22,
    )
    plt.close(fig)

    # =========================================================================
    # 5. Legacy Composite Figures (Figure2, Figure4, Figure7) Upgraded to
    #    Vertically Stacked Full-Width Layouts with Huge Typography
    # =========================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.8, 11.4), dpi=300)
    ax1.bar(
        x,
        clusters,
        color="#1d4ed8",
        label="Unique Specimen Clusters (Total = 2,554)",
        width=0.58,
        edgecolor="#0f172a",
    )
    ax1.bar(
        x,
        dups,
        bottom=clusters,
        color="#dc2626",
        alpha=0.88,
        label="Near-Duplicate / Zoom-Rotated Siblings (36.15%)",
        width=0.58,
        edgecolor="#0f172a",
    )
    for idx, (cl, ratio) in enumerate(
        zip(clusters, df_t1["Near-Duplicate / Aug-Mix Ratio (%)"])
    ):
        ax1.text(
            idx,
            510,
            f"{cl} cl.\n({ratio:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11.5,
            fontweight="bold",
            color="#0f172a",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, fontsize=12.0, fontweight="bold")
    ax1.tick_params(axis="y", labelsize=12.5)
    ax1.set_ylim(0, 675)
    ax1.set_ylabel("Images per Class (N = 500)", fontweight="bold", fontsize=13.5)
    ax1.set_title(
        "A. Near-Duplicate Specimen Audit on MangoLeafBD (39.0% Random-Split Leakage)",
        fontweight="bold",
        fontsize=14.2,
    )
    ax1.legend(loc="upper center", fontsize=12.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.45)

    views_short = [
        "1. Whole ROI (Center r=0.86)",
        "2. Segmented Organ Only",
        "3. Background Only (Masked)",
        "4. Four 32x32 Corner Patches",
        "5. File Metadata & Aspect Ratio",
    ]
    acc_short = df_t2["Group-Split Test Acc (%)"].values
    f1_short = df_t2["Group-Split Macro-F1"].values * 100.0
    y_pos = np.arange(len(views_short))
    bar_colors = ["#16a34a", "#2563eb", "#dc2626", "#d97706", "#64748b"]
    bars2 = ax2.barh(
        y_pos, acc_short, color=bar_colors, height=0.56, edgecolor="#0f172a"
    )
    ax2.axvline(
        12.5,
        color="#991b1b",
        linestyle="--",
        linewidth=2.0,
        label="8-Class Random Chance (12.5%)",
    )
    for b, acc_v, f1_v in zip(bars2, acc_short, f1_short):
        ax2.text(
            acc_v + 1.2,
            b.get_y() + b.get_height() / 2,
            f"{acc_v:.2f}% (Macro-F1: {f1_v:.2f}%)",
            va="center",
            fontsize=12.0,
            fontweight="bold",
            color="#0f172a",
        )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(views_short, fontsize=12.5, fontweight="bold")
    ax2.tick_params(axis="x", labelsize=12.5)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 126)
    ax2.set_xlabel("Leak-Free Group-Split Test Accuracy (%)", fontweight="bold", fontsize=13.5)
    ax2.set_title(
        "B. Five-View Studio Background & Framing Shortcut Audit",
        fontweight="bold",
        fontsize=14.2,
    )
    ax2.legend(loc="lower right", fontsize=12.0)
    ax2.grid(axis="x", linestyle="--", alpha=0.45)
    fig.tight_layout(pad=2.0)
    fig.savefig(
        os.path.join(figures_dir, "Figure2_Audit_and_Background_Shortcuts.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.22,
    )
    plt.close(fig)

    # Figure4_Ablation_and_Confidence_Gate.png
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.8, 11.2), dpi=300)
    ax1.plot(
        x3,
        test_acc,
        marker="o",
        markersize=8,
        linewidth=2.8,
        color="#1d4ed8",
        label="Group-Split Test Accuracy (%)",
    )
    ax1.plot(
        x3,
        test_f1,
        marker="s",
        markersize=7,
        linewidth=2.4,
        linestyle="--",
        color="#16a34a",
        label="Group-Split Test Macro-F1 (%)",
    )
    ax1.scatter([3], [test_acc[3]], color="#dc2626", s=130, zorder=5)
    ax1.annotate(
        "M4 Raw HOG Dilution (1,844D: 86.61%)",
        xy=(3, test_acc[3]),
        xytext=(2.3, 79.0),
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.5),
        fontsize=12.0,
        fontweight="bold",
        color="#dc2626",
    )
    ax1.scatter([12], [test_acc[12]], color="#16a34a", s=140, zorder=5)
    ax1.annotate(
        "M10 Proposed Block-PCA (303D: 97.91% Acc, p<0.0001)",
        xy=(12, test_acc[12]),
        xytext=(6.2, 99.5),
        arrowprops=dict(arrowstyle="->", color="#15803d", lw=1.5),
        fontsize=12.0,
        fontweight="bold",
        color="#15803d",
    )
    ax1.set_xticks(x3)
    ax1.set_xticklabels(short_labels, fontsize=12.0, fontweight="bold")
    ax1.tick_params(axis="y", labelsize=12.5)
    ax1.set_ylim(74, 103.0)
    ax1.set_ylabel("Group-Split Performance (%)", fontweight="bold", fontsize=13.5)
    ax1.set_title(
        "A. Progressive Ablation Study (M1 to M11) & Block-PCA Recovery",
        fontweight="bold",
        fontsize=14.2,
    )
    ax1.legend(loc="lower right", fontsize=12.0)
    ax1.grid(True, linestyle="--", alpha=0.45)

    ax2.plot(
        routed[mask_zoom],
        f1_g[mask_zoom],
        marker="D",
        markersize=8,
        linewidth=2.8,
        color="#7c3aed",
        label="Two-Stage Confidence-Gated Trajectory",
    )
    ax2.scatter([1.53], [98.36], color="#dc2626", s=140, zorder=6, label="Val-Selected Gate (tau=0.65: 98.47% Acc / 98.36% F1)")
    ax2.scatter([4.32], [98.59], color="#d97706", s=130, zorder=6, label="High-Sensitivity Gate (tau=0.85: 98.74% Acc / 98.59% F1)")
    ax2.set_xlim(-0.4, 8.8)
    ax2.set_ylim(97.60, 98.85)
    ax2.set_xlabel("Samples Routed to Stage-2 Nonlinear Ensemble (%)", fontweight="bold", fontsize=13.5)
    ax2.set_ylabel("Group-Split Test Macro-F1 (%)", fontweight="bold", fontsize=13.5)
    ax2.set_title(
        "B. Selective Stage-2 Routing Rate vs. Macro-F1 Gain",
        fontweight="bold",
        fontsize=14.2,
    )
    ax2.tick_params(axis="both", labelsize=12.5)
    ax2.legend(loc="lower right", fontsize=12.0)
    ax2.grid(True, linestyle="--", alpha=0.45)
    fig.tight_layout(pad=2.0)
    fig.savefig(
        os.path.join(figures_dir, "Figure4_Ablation_and_Confidence_Gate.png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.22,
    )
    plt.close(fig)

    # Figure7_Perturbation_Robustness_and_CrossDomain.png
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.8, 11.4), dpi=300)
    ax1.bar(
        x5 - 1.5 * w,
        raw_v,
        width=w,
        color="#94a3b8",
        edgecolor="#0f172a",
        label="1. Baseline Raw (M4)",
    )
    ax1.bar(
        x5 - 0.5 * w,
        clahe_v,
        width=w,
        color="#f59e0b",
        edgecolor="#0f172a",
        label="2. Unconditional CLAHE",
    )
    ax1.bar(
        x5 + 0.5 * w,
        blur_v,
        width=w,
        color="#3b82f6",
        edgecolor="#0f172a",
        label="3. Conditional Blur Control",
    )
    ax1.bar(
        x5 + 1.5 * w,
        prop_v,
        width=w,
        color="#16a34a",
        edgecolor="#0f172a",
        label="4. Proposed Quality-Gated",
    )
    ax1.set_xticks(x5)
    ax1.set_xticklabels(scen_labels, fontsize=12.0, fontweight="bold")
    ax1.tick_params(axis="y", labelsize=12.5)
    ax1.set_ylim(0.15, 1.22)
    ax1.set_ylabel("Group-Split Test Macro-F1", fontweight="bold", fontsize=13.5)
    ax1.set_title(
        "A. Multi-Severity Optical & Sensor Perturbation Robustness (Table 5)",
        fontweight="bold",
        fontsize=14.2,
    )
    ax1.legend(loc="upper center", fontsize=11.5, ncol=2)
    ax1.grid(axis="y", linestyle="--", alpha=0.45)

    prot_labels = [
        "In-Domain\nSource-View\n(N=455)",
        "In-Domain\nCanonical-Pose\n(N=455)",
        "MLDID External\nSource 0-Shot\n(N=3,000)",
        "MLDID External\nCanonical 0-Shot\n(N=3,000)",
        "MLDID External\nCanonical+CORAL\n(N=3,000)",
    ]
    idx_order = [0, 2, 1, 3, 4]
    acc_t7 = df_t7.iloc[idx_order]["Accuracy (%)"].values
    f1_t7 = df_t7.iloc[idx_order]["Macro-F1"].values * 100.0
    x7 = np.arange(len(prot_labels))
    w7 = 0.34
    b_acc = ax2.bar(
        x7 - w7 / 2,
        acc_t7,
        width=w7,
        color="#1d4ed8",
        edgecolor="#0f172a",
        label="Accuracy (%)",
    )
    b_f1 = ax2.bar(
        x7 + w7 / 2,
        f1_t7,
        width=w7,
        color="#16a34a",
        edgecolor="#0f172a",
        label="Macro-F1 (%)",
    )
    for bar in list(b_acc) + list(b_f1):
        hv = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            hv + 1.5,
            f"{hv:.1f}%",
            ha="center",
            fontsize=11.5,
            fontweight="bold",
            color="#0f172a",
        )
    ax2.annotate(
        "+11.36% Acc / +15.09% F1\nZero-Shot Canonical Pose Gain",
        xy=(3.17, 48.0),
        xytext=(2.0, 72.0),
        arrowprops=dict(arrowstyle="->", color="#15803d", lw=1.8),
        fontsize=12.5,
        fontweight="bold",
        color="#15803d",
    )
    ax2.set_xticks(x7)
    ax2.set_xticklabels(prot_labels, fontsize=11.5, fontweight="bold")
    ax2.tick_params(axis="y", labelsize=12.5)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Performance on 5 Shared Classes (%)", fontweight="bold", fontsize=13.5)
    ax2.set_title(
        "B. Cross-Dataset Generalization (MangoLeafBD -> MLDID, N=3,000, Table 7)",
        fontweight="bold",
        fontsize=14.2,
    )
    ax2.legend(loc="upper right", fontsize=12.0)
    ax2.grid(axis="y", linestyle="--", alpha=0.45)
    fig.tight_layout(pad=2.0)
    fig.savefig(
        os.path.join(
            figures_dir, "Figure7_Perturbation_Robustness_and_CrossDomain.png"
        ),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.22,
    )
    plt.close(fig)

    # =========================================================================
    # 6. Fig16_EDA_CrossDataset_Distribution_Shift_MangoLeafBD_vs_MLDID.png
    # =========================================================================
    eda_dir = os.path.join(os.path.dirname(os.path.abspath(tables_dir)), "eda")
    m_cls_path = os.path.join(eda_dir, "MangoLeafBD_ClassSummary.csv")
    x_cls_path = os.path.join(eda_dir, "MLDID_ClassSummary.csv")
    if os.path.exists(m_cls_path) and os.path.exists(x_cls_path):
        df_m = pd.read_csv(m_cls_path).set_index("Class")
        df_x = pd.read_csv(x_cls_path).set_index("Class")
        shared = ["Anthracnose", "Bacterial_Canker", "Die_Back", "Gall_Midge", "Healthy"]
        shared_lbls = [s.replace("_", "\n") for s in shared]
        xs = np.arange(len(shared))
        ws = 0.35

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.8, 10.8), dpi=300)
        c_m = df_m.loc[shared, "Median contrast (L* P95-P05)"].values
        c_x = df_x.loc[shared, "Median contrast (L* P95-P05)"].values
        b1 = ax1.bar(
            xs - ws / 2,
            c_m,
            width=ws,
            color="#1d4ed8",
            edgecolor="#0f172a",
            label="MangoLeafBD Source (Bangladesh, N=4,000, Mixed 240p/320p)",
        )
        b2 = ax1.bar(
            xs + ws / 2,
            c_x,
            width=ws,
            color="#dc2626",
            edgecolor="#0f172a",
            label="MLDID External Target (India, N=3,000, Standardized 512x512)",
        )
        for b in list(b1) + list(b2):
            hv = b.get_height()
            ax1.text(
                b.get_x() + b.get_width() / 2,
                hv + 2.5,
                f"{hv:.0f}",
                ha="center",
                va="bottom",
                fontsize=12.0,
                fontweight="bold",
                color="#0f172a",
            )
        ax1.set_xticks(xs)
        ax1.set_xticklabels(shared_lbls, fontsize=12.5, fontweight="bold")
        ax1.tick_params(axis="y", labelsize=12.5)
        for lbl in ax1.get_yticklabels():
            lbl.set_fontweight("bold")
        ax1.set_ylim(0, 215)
        ax1.set_ylabel("Median L* Dynamic Range (P95 - P05)", fontweight="bold", fontsize=13.5)
        ax1.set_title(
            "A. Cross-Dataset Luminance Contrast Shift (CIE L* P95-P05) Across 5 Shared Classes",
            fontweight="bold",
            fontsize=14.5,
            pad=10,
        )
        ax1.legend(loc="upper center", fontsize=12.0, framealpha=0.96)
        ax1.grid(axis="y", linestyle="--", alpha=0.45)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        s_m = df_m.loc[shared, "Median sharpness"].values
        s_x = df_x.loc[shared, "Median sharpness"].values
        b3 = ax2.bar(
            xs - ws / 2,
            s_m,
            width=ws,
            color="#0284c7",
            edgecolor="#0f172a",
            label="MangoLeafBD Source Median Sharpness (Laplacian Var)",
        )
        b4 = ax2.bar(
            xs + ws / 2,
            s_x,
            width=ws,
            color="#d97706",
            edgecolor="#0f172a",
            label="MLDID Target Median Sharpness (Laplacian Var)",
        )
        for b in list(b3) + list(b4):
            hv = b.get_height()
            ax2.text(
                b.get_x() + b.get_width() / 2,
                hv + 3.0,
                f"{hv:.1f}",
                ha="center",
                va="bottom",
                fontsize=12.0,
                fontweight="bold",
                color="#0f172a",
            )
        ax2.set_xticks(xs)
        ax2.set_xticklabels(shared_lbls, fontsize=12.5, fontweight="bold")
        ax2.tick_params(axis="y", labelsize=12.5)
        for lbl in ax2.get_yticklabels():
            lbl.set_fontweight("bold")
        ax2.set_ylim(0, 270)
        ax2.set_ylabel("Median Laplacian Variance (128x128)", fontweight="bold", fontsize=13.5)
        ax2.set_title(
            "B. Cross-Dataset High-Frequency Optical Sharpness Shift (MangoLeafBD vs. MLDID)",
            fontweight="bold",
            fontsize=14.5,
            pad=10,
        )
        ax2.legend(loc="upper center", fontsize=12.0, framealpha=0.96)
        ax2.grid(axis="y", linestyle="--", alpha=0.45)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        fig.tight_layout(pad=2.0)
        fig.savefig(
            os.path.join(
                figures_dir,
                "Fig16_EDA_CrossDataset_Distribution_Shift_MangoLeafBD_vs_MLDID.png",
            ),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.22,
        )
        plt.close(fig)

