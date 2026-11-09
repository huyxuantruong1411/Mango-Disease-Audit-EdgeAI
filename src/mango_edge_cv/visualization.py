"""
Publication-Grade Visualization Suite for Mango-Disease-Audit-EdgeAI
(`src/mango_edge_cv/visualization.py`).
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_feature_block_attribution(
    gated_system, output_path: str, num_classes: int = 8, track_name: str = "MANGO"
):
    """Plot relative L1 weight attribution across the 7 compressed feature blocks (303D)."""
    base_lr_est = gated_system.stage1_lr.calibrated_classifiers_[0].estimator
    w_abs = np.mean(np.abs(base_lr_est.coef_), axis=0)
    slices = [
        ("HSV Color\n(52d)", slice(0, 52)),
        ("CIE LAB\n(57d)", slice(52, 109)),
        ("Multi-Scale LBP\n(28d)", slice(109, 137)),
        ("5-Patch Spatial\n(45d)", slice(137, 182)),
        ("Quality Vector q\n(7d)", slice(182, 189)),
        ("MGCT Chromatic-Texture\n(50d)", slice(189, 239)),
        ("HOG Block-PCA\n(64d)", slice(239, min(303, len(w_abs)))),
    ]
    b_names = [s[0] for s in slices]
    weight_total = float(np.sum(w_abs))
    b_pcts = [
        (float(np.sum(w_abs[s[1]])) / weight_total * 100.0) if weight_total > 0 else 0.0
        for s in slices
    ]

    fig, ax = plt.subplots(figsize=(9.5, 4.4), dpi=300)
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#9467bd",
        "#8c564b",
        "#d62728",
        "#7f7f7f",
    ]
    bars = ax.bar(b_names, b_pcts, color=colors, width=0.52, edgecolor="#222222", linewidth=0.6)
    for b in bars:
        h_v = b.get_height()
        ax.text(
            b.get_x() + b.get_width() / 2,
            h_v + 0.6,
            f"{h_v:.1f}%",
            ha="center",
            fontweight="bold",
            fontsize=9.5,
        )
    ax.set_ylim(0, max(1.0, max(b_pcts) * 1.18))
    ax.set_ylabel("Relative L1 Weight Attribution (%)", fontweight="bold")
    ax.set_title(
        f"Feature Block Attribution Across {num_classes} Classes ({track_name})",
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return b_pcts


def generate_analytical_charts_from_tables(tables_dir: str, figures_dir: str):
    """
    Generate publication-grade analytical figures from locked v5 CSV tables:
      - Figure2_Audit_and_Background_Shortcuts.png
      - Figure4_Ablation_and_Confidence_Gate.png
      - Figure7_Perturbation_Robustness_and_CrossDomain.png
    """
    os.makedirs(figures_dir, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Serif"

    # 1. Figure 2: Dataset Audit (Unique Clusters vs Duplicate Ratio) + 5-View Background Shortcut Audit
    df_t1 = pd.read_csv(
        os.path.join(tables_dir, "Table1_Dataset_Leakage_and_Group_Audit.csv")
    )
    df_t2 = pd.read_csv(
        os.path.join(tables_dir, "Table2_Background_Shortcut_Audit.csv")
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.2), dpi=300)
    classes = [c.replace("_", "\n") for c in df_t1["Class Name"]]
    clusters = df_t1["Unique Specimen Clusters (pHash<=6 | HSV cosine>=0.992)"].values
    dups = 500 - clusters

    x = np.arange(len(classes))
    ax1.bar(
        x,
        clusters,
        color="#2b6cb0",
        label="Unique Specimen Clusters (Total = 2,554)",
        width=0.64,
    )
    ax1.bar(
        x,
        dups,
        bottom=clusters,
        color="#e53e3e",
        alpha=0.85,
        label="Near-Duplicate / Zoom-Rotated Siblings (36.15%)",
        width=0.64,
    )
    for idx, (cl, ratio) in enumerate(
        zip(clusters, df_t1["Near-Duplicate / Aug-Mix Ratio (%)"])
    ):
        ax1.text(
            idx,
            cl / 2 if cl > 70 else cl + 22,
            f"{cl}\n({ratio:.1f}%)",
            ha="center",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            color="white" if cl > 70 else "#1a202c",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, fontsize=8.5, fontweight="bold")
    ax1.set_ylim(0, 565)
    ax1.set_ylabel("Images per Class (N = 500)", fontweight="bold")
    ax1.set_title(
        "A. Near-Duplicate Specimen Audit on MangoLeafBD (39.0% Random-Split Leakage)",
        fontweight="bold",
        fontsize=10.5,
    )
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    views_short = [
        "1. Whole ROI\n(Center r=0.86)",
        "2. Segmented\nOrgan Only",
        "3. Background\nOnly (Masked)",
        "4. Four 32x32\nCorner Patches",
        "5. File Metadata\n& Aspect Ratio",
    ]
    acc_short = df_t2["Group-Split Test Acc (%)"].values
    f1_short = df_t2["Group-Split Macro-F1"].values * 100.0
    y_pos = np.arange(len(views_short))
    bar_colors = ["#2f855a", "#3182ce", "#dd6b20", "#d69e2e", "#718096"]
    bars2 = ax2.barh(y_pos, acc_short, color=bar_colors, height=0.55, edgecolor="#222")
    ax2.axvline(12.5, color="#c53030", linestyle="--", linewidth=1.3, label="Random Chance (12.5%)")
    for b, acc_v, f1_v in zip(bars2, acc_short, f1_short):
        ax2.text(
            acc_v + 1.2,
            b.get_y() + b.get_height() / 2,
            f"{acc_v:.2f}% (F1: {f1_v:.2f}%)",
            va="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(views_short, fontsize=9, fontweight="bold")
    ax2.invert_yaxis()
    ax2.set_xlim(0, 116)
    ax2.set_xlabel("Leak-Free Group-Split Test Accuracy (%)", fontweight="bold")
    ax2.set_title(
        "B. Five-View Studio Background & Framing Shortcut Audit",
        fontweight="bold",
        fontsize=10.5,
    )
    ax2.legend(loc="lower right", fontsize=8.5)
    ax2.grid(axis="x", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, "Figure2_Audit_and_Background_Shortcuts.png"),
        dpi=300,
    )
    plt.close(fig)

    # 2. Figure 4: Progressive Ablation (M1-M11) & Confidence-Gated Routing Trade-off
    df_t3 = pd.read_csv(
        os.path.join(tables_dir, "Table3_Progressive_Ablation_Study.csv")
    )
    df_t4 = pd.read_csv(
        os.path.join(tables_dir, "Table4_Confidence_Gated_and_Calibration.csv")
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.4), dpi=300)
    short_labels = [c.split(".")[0] for c in df_t3["Configuration"]]
    test_f1 = df_t3["Test Macro-F1"].values * 100.0
    test_acc = df_t3["Test Acc (%)"].values
    x3 = np.arange(len(short_labels))

    ax1.plot(
        x3,
        test_acc,
        marker="o",
        linewidth=2.2,
        color="#2b6cb0",
        label="Group-Split Test Accuracy (%)",
    )
    ax1.plot(
        x3,
        test_f1,
        marker="s",
        linewidth=2.0,
        linestyle="--",
        color="#2f855a",
        label="Group-Split Test Macro-F1 (%)",
    )
    ax1.scatter([3], [test_acc[3]], color="#e53e3e", s=90, zorder=5)
    ax1.annotate(
        "M4 Raw HOG Dilution\n(1,844D: 86.61%)",
        xy=(3, test_acc[3]),
        xytext=(2.1, 80.5),
        arrowprops=dict(arrowstyle="->", color="#c53030", lw=1.2),
        fontsize=8,
        fontweight="bold",
        color="#c53030",
    )
    ax1.scatter([12], [test_acc[12]], color="#2f855a", s=100, zorder=5)
    ax1.annotate(
        "M10 Proposed Block-PCA\n(303D: 97.91% Acc, p<0.0001)",
        xy=(12, test_acc[12]),
        xytext=(6.8, 98.4),
        arrowprops=dict(arrowstyle="->", color="#2f855a", lw=1.2),
        fontsize=8,
        fontweight="bold",
        color="#22543d",
    )
    ax1.set_xticks(x3)
    ax1.set_xticklabels(short_labels, fontsize=8.5, fontweight="bold")
    ax1.set_ylim(75, 101.5)
    ax1.set_ylabel("Group-Split Performance (%)", fontweight="bold")
    ax1.set_xlabel("Ablation Configuration (M1 to M11)", fontweight="bold")
    ax1.set_title(
        "A. Progressive Ablation Study & Block-PCA Recovery",
        fontweight="bold",
        fontsize=10.5,
    )
    ax1.legend(loc="lower right", fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.4)

    df_t4_group = df_t4[~df_t4["Routing Mode"].str.contains("REFERENCE")].copy()
    routed = df_t4_group["Stage-2 Routed Samples (%)"].values
    f1_g = df_t4_group["Group-Split Macro-F1"].values * 100.0
    taus = df_t4_group["Gate Threshold (tau_gate)"].values

    mask_zoom = taus <= 0.95
    ax2.plot(
        routed[mask_zoom],
        f1_g[mask_zoom],
        marker="D",
        linewidth=2.3,
        color="#805ad5",
        label="Two-Stage Confidence-Gated Trajectory",
    )
    for r_v, f_v, t_v in zip(routed[mask_zoom], f1_g[mask_zoom], taus[mask_zoom]):
        if t_v in (0.0, 0.65, 0.85, 0.95):
            lbl = (
                f"Stage-1 Only\n(0% routed, {f_v:.2f}%)"
                if t_v == 0.0
                else f"tau={t_v:.2f}\n({r_v:.2f}% routed, {f_v:.2f}%)"
            )
            offset_y = -0.18 if t_v == 0.0 else (0.08 if t_v == 0.65 else 0.06)
            ax2.annotate(
                lbl,
                xy=(r_v, f_v),
                xytext=(r_v + 0.35, f_v + offset_y),
                fontsize=7.8,
                fontweight="bold",
                color="#2d3748",
            )
    ax2.scatter([1.53], [98.36], color="#e53e3e", s=110, zorder=6, label="Val-Selected Gate (tau=0.65: 98.47% Acc)")
    ax2.scatter([4.32], [98.59], color="#dd6b20", s=95, zorder=6, label="High-Sensitivity Gate (tau=0.85: 98.74% Acc)")
    ax2.set_xlim(-0.4, 9.2)
    ax2.set_ylim(97.65, 98.85)
    ax2.set_xlabel("Samples Routed to Stage-2 Nonlinear Ensemble (%)", fontweight="bold")
    ax2.set_ylabel("Group-Split Test Macro-F1 (%)", fontweight="bold")
    ax2.set_title(
        "B. Selective Stage-2 Routing Rate vs. Macro-F1 Gain",
        fontweight="bold",
        fontsize=10.5,
    )
    ax2.legend(loc="lower right", fontsize=8.2)
    ax2.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(
        os.path.join(figures_dir, "Figure4_Ablation_and_Confidence_Gate.png"),
        dpi=300,
    )
    plt.close(fig)

    # 3. Figure 7: Multi-Severity Perturbation Robustness (Table 5) & MLDID Cross-Dataset Validation (Table 7)
    df_t5 = pd.read_csv(
        os.path.join(tables_dir, "Table5_MultiSeverity_Robustness.csv")
    )
    df_t7 = pd.read_csv(
        os.path.join(tables_dir, "Table7_External_Dataset_Validation.csv")
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.3), dpi=300)
    scen_labels = [
        "Clean\n(sigma=0)",
        "Noise\nsigma=5",
        "Noise\nsigma=10",
        "Noise\nsigma=20",
        "Glare\nr=22px",
        "Glare\nr=35px",
        "Low-Light\n-40%",
    ]
    x5 = np.arange(len(scen_labels))
    w = 0.20
    ax1.bar(
        x5 - 1.5 * w,
        df_t5["1. Baseline Raw (HSV+LBP+HOG)"],
        width=w,
        color="#a0aec0",
        label="1. Baseline Raw (M4)",
    )
    ax1.bar(
        x5 - 0.5 * w,
        df_t5["2. Unconditional CLAHE"],
        width=w,
        color="#ed8936",
        label="2. Unconditional CLAHE",
    )
    ax1.bar(
        x5 + 0.5 * w,
        df_t5["3. Conditional Blur Control"],
        width=w,
        color="#4299e1",
        label="3. Conditional Blur Control",
    )
    ax1.bar(
        x5 + 1.5 * w,
        df_t5["4. Proposed Quality-Gated"],
        width=w,
        color="#2f855a",
        label="4. Proposed Quality-Gated",
    )
    ax1.set_xticks(x5)
    ax1.set_xticklabels(scen_labels, fontsize=8.2, fontweight="bold")
    ax1.set_ylim(0.15, 1.08)
    ax1.set_ylabel("Group-Split Test Macro-F1", fontweight="bold")
    ax1.set_title(
        "A. Multi-Severity Optical & Sensor Perturbation Robustness",
        fontweight="bold",
        fontsize=10.5,
    )
    ax1.legend(loc="upper right", fontsize=8.0, ncol=2)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    prot_labels = [
        "In-Domain\nSource-View\n(N=455)",
        "In-Domain\nCanonical-Pose\n(N=455)",
        "MLDID External\nSource-View 0-Shot\n(N=3,000)",
        "MLDID External\nCanonical 0-Shot\n(N=3,000)",
        "MLDID External\nCanonical + CORAL\n(N=3,000)",
    ]
    idx_order = [0, 2, 1, 3, 4]
    acc_t7 = df_t7.iloc[idx_order]["Accuracy (%)"].values
    f1_t7 = df_t7.iloc[idx_order]["Macro-F1"].values * 100.0
    x7 = np.arange(len(prot_labels))
    w7 = 0.34
    b_acc = ax2.bar(
        x7 - w7 / 2, acc_t7, width=w7, color="#2b6cb0", label="Accuracy (%)"
    )
    b_f1 = ax2.bar(
        x7 + w7 / 2, f1_t7, width=w7, color="#38a169", label="Macro-F1 (%)"
    )
    for bar in list(b_acc) + list(b_f1):
        hv = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            hv + 1.2,
            f"{hv:.1f}%",
            ha="center",
            fontsize=7.5,
            fontweight="bold",
        )
    ax2.annotate(
        "+11.36% Acc / +15.09% F1\nZero-Shot Canonical Pose Gain",
        xy=(3.17, 47.5),
        xytext=(2.05, 66.0),
        arrowprops=dict(arrowstyle="->", color="#22543d", lw=1.4),
        fontsize=8.2,
        fontweight="bold",
        color="#22543d",
    )
    ax2.set_xticks(x7)
    ax2.set_xticklabels(prot_labels, fontsize=8.0, fontweight="bold")
    ax2.set_ylim(0, 112)
    ax2.set_ylabel("Performance on 5 Shared Classes (%)", fontweight="bold")
    ax2.set_title(
        "B. Cross-Dataset Generalization (MangoLeafBD -> MLDID, N=3,000)",
        fontweight="bold",
        fontsize=10.5,
    )
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(
        os.path.join(
            figures_dir, "Figure7_Perturbation_Robustness_and_CrossDomain.png"
        ),
        dpi=300,
    )
    plt.close(fig)
