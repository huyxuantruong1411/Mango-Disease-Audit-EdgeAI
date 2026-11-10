# Mathematical Formulation and Algorithmic Architecture (`mango_edge_cv`)

This document provides the complete mathematical specification and algorithmic rationale for all six pillars of the **Audit-First, Quality-Gated Two-Stage Edge-AI Framework** implemented in [`src/mango_edge_cv/`](../src/mango_edge_cv/). Every equation and constant documented below corresponds directly to the locked Version 5 (`v5`, `SEED = 20261022`) implementation.

<div align="center">
  <img src="../results/figures/Fig01_System_Overview_Macro.png" width="94%" alt="Figure 1: Macro System Architecture Overview" />
  <p><em>Figure 1: Macro system architecture overview illustrating the six-pillar design from dataset auditing to zero-shot edge inference.</em></p>
</div>

---

## 1. Pillar I: Graph-Based Specimen-Identity Clustering and Shortcut Auditing (`src/mango_edge_cv/audit.py`)

### 1.1 Connected-Component Specimen Discovery (`build_audit_groups`)

Let the plant pathology dataset be denoted by:

$$
\mathcal{D} = \lbrace (I_i, y_i) \rbrace_{i=1}^{N}
$$

comprising $N = 4,000$ images across $C = 8$ classes, where each class contains $N_c = 500$ images ($c \in \lbrace 1, \dots, 8 \rbrace$). Because multiple photographs in studio benchmarks originate from burst captures, 90°, 180°, 270° rotations, or crop-zoom variations of the *same physical leaf*, standard random train/test partitioning violates specimen independence.

Within each class $c$, we construct an undirected graph:

$$
G_c = (V_c, E_c), \quad V_c = \lbrace 1, \dots, N_c \rbrace
$$

indexing the images in class $c$:

#### 1. 64-Bit Structural 2D-DCT Perceptual Hash (pHash-64)

Each image is converted to grayscale and resized via area interpolation to $32 \times 32$ pixels ($I_{32}$). Applying the orthogonal 2D Discrete Cosine Transform yields:

$$
D(I_i) = \text{DCT}_{2\text{D}}(I_{32}) \in \mathbb{R}^{32\times32}
$$

We retain the top-left $8\times8$ low-frequency submatrix $D_{0:8,0:8}$ and compute the median of its 63 AC coefficients (excluding the DC term $D_{0,0}$):

$$
m_{\text{AC}}(I_i) = \text{median}_{(u, v) \neq (0, 0)} D_{u,v}(I_i), \quad (u, v) \in \lbrace 0, \dots, 7 \rbrace^2
$$

Quantizing each coefficient against the median threshold yields a 64-bit binary descriptor $\mathbf{b}(I_i) \in \lbrace 0, 1 \rbrace^{64}$ with Hamming distance:

$$
b_{u,v}(I_i) = \mathbb{I}\big(D_{u,v}(I_i) > m_{\text{AC}}(I_i)\big), \quad (u, v) \in \lbrace 0, \dots, 7 \rbrace^2
$$

$$
d_{\text{Hamming}}(I_i, I_j) = \Vert \mathbf{b}(I_i) \oplus \mathbf{b}(I_j) \Vert_1 = \sum_{u=0}^{7}\sum_{v=0}^{7} \mathbb{I}\big(b_{u,v}(I_i) \neq b_{u,v}(I_j)\big)
$$

#### 2. Rotation-Invariant Central-ROI Hue–Saturation Cosine Similarity

Because planar rotations permute 2D-DCT spatial frequencies while leaving the leaf's internal lesion color distribution invariant, we extract the central 60% sub-window:

$$
I_{\text{center}} = I[0.20H : 0.80H,\, 0.20W : 0.80W]
$$

We resize to $64\times64$, convert to HSV, and compute a 2D Hue–Saturation histogram with $12 \times 8 = 96$ bins, L2-normalized into unit vector $\mathbf{h}_i \in \mathbb{R}^{96}$ ($\Vert \mathbf{h}_i \Vert = 1$). Pairwise chromatic similarity is evaluated as:

$$
s_{\text{cos}}(I_i, I_j) = \frac{\langle \mathbf{h}_i, \mathbf{h}_j \rangle}{\Vert \mathbf{h}_i \Vert_2 \, \Vert \mathbf{h}_j \Vert_2 + 10^{-8}}
$$

#### 3. Adjacency Criterion and Connected Components

An edge $(i,j) \in E_c$ connects images $I_i$ and $I_j$ if and only if:

$$
(i, j) \in E_c \iff \Big(d_{\text{Hamming}}(I_i, I_j) \le 6\Big) \;\lor\; \Big(s_{\text{cos}}(I_i, I_j) \ge 0.992\Big)
$$

Extracting connected components of $G_c$ yields 2,554 unique physical specimen clusters across the 4,000 images (36.15% near-duplicate sibling ratio). Enforcing `StratifiedGroupKFold` over these cluster IDs guarantees zero specimen overlap between $\mathcal{D}_{\text{train}}$, $\mathcal{D}_{\text{val}}$, and $\mathcal{D}_{\text{test}}$:

$$
\mathcal{G}_{\text{train}} \cap \mathcal{G}_{\text{val}} = \varnothing, \quad \mathcal{G}_{\text{train}} \cap \mathcal{G}_{\text{test}} = \varnothing
$$

<div align="center">
  <img src="../results/figures/Fig02_Specimen_Clustering_Graph_Mechanism.png" width="94%" alt="Figure 2: Specimen-Identity Graph Clustering" />
  <p><em>Figure 2: Specimen-identity graph clustering mechanism and empirical proof of near-duplicate detection.</em></p>
</div>

### 1.2 Five-View Studio Background Shortcut Protocol (`run_background_shortcut_audit`)

To decouple biological foliar symptoms from studio capture artifacts, we evaluate five experimental perspectives on each image $I$ and its foreground organ mask:

- **View 1 (Whole ROI, $r=0.86$):** Center-cropped frame $\implies \mathbf{f} \in \mathbb{R}^{137}$ (98.33% Acc).
- **View 2 (Segmented Organ Only):** `I_fg = I * M_fg` (studio background zeroed) $\implies \mathbf{f} \in \mathbb{R}^{137}$ (94.84% Acc).
- **View 3 (Background Only, Organ Masked Out):** `I_bg = I * (1 - M_fg)` (**zero leaf pixels**) $\implies \mathbf{f} \in \mathbb{R}^{137}$ (94.84% Acc).
- **View 4 (Four $32\times32$ Corner Patches Only):** Retains only the four $32\times32$ corner regions $\implies \mathbf{f} \in \mathbb{R}^{137}$ (85.08% Acc).
- **View 5 (File Metadata Only):** Non-visual metadata vector (59.14% Acc):

$$
\mathbf{m} = [\text{FileSizeKB},\, \bar{R},\, \bar{G},\, \bar{B},\, \sigma_{\text{RGB}},\, r_{\text{fg}},\, \text{AspectRatio}]^\top \in \mathbb{R}^7
$$

---

## 2. Pillar II: Foreground-Aware Quality-Gated Optical Preprocessing (`src/mango_edge_cv/preprocessing.py`)

### 2.1 Hybrid Otsu–Chroma Foreground Organ Segmentation (`segment_plant_organ_roi`)

Given an input image standardized via center-crop ($r = 0.88$) to $256 \times 256 \times 3$, let $(H, S, V)$ and $(L^{\ast}, a^{\ast}, b^{\ast})$ denote its 8-bit HSV and CIE-LAB representations. The per-pixel chromaticity salience score is defined as:

$$
C^{\ast}(x, y) = \sqrt{\big(a^{\ast}(x, y) - 128\big)^2 + \big(b^{\ast}(x, y) - 128\big)^2}
$$

$$
S_{\text{roi}}(x, y) = \text{clip}\Big(\text{round}\big(0.60\,S(x, y) + 1.25\,C^{\ast}(x, y)\big), \; 0, \; 255\Big)
$$

An initial binary mask is generated via Otsu thresholding:

$$
M_0(x, y) = \mathbb{I}\big(S_{\text{roi}}(x, y) > T_{\text{Otsu}}\big)
$$

and refined by suppressing studio backdrop pixels where:

$$
\big(V(x, y) > 230 \land S(x, y) < 28\big) \lor \big(V(x, y) < 15\big)
$$

clearing a 3% frame border margin, and applying morphological closing ($7\times7$ ellipse, 2 iterations) followed by opening (1 iteration). Among all 8-connected foreground components $\lbrace C_k \rbrace$, we select the largest component $C_{k^{\ast}}$ provided its area satisfies $0.04 \le \frac{|C_{k^{\ast}}|}{256^2} \le 0.92$.

<div align="center">
  <img src="../results/figures/Fig05_Hybrid_ROI_Segmentation_StepByStep.png" width="94%" alt="Figure 5: Hybrid ROI Segmentation" />
  <p><em>Figure 5: Step-by-step hybrid HSV saturation and CIE-LAB chroma ROI segmentation pipeline.</em></p>
</div>

### 2.2 Three Conditional Optical Restoration Gates (`apply_adaptive_quality_preprocessing`)

All three restoration gates are conditioned on physical measurements evaluated strictly inside $M_{\text{fg}}$:

#### 1. Donoho–Johnstone MAD Sensor Grain Gate (`_estimate_noise_sigma_mad`)

We isolate high-frequency spatial residuals on the grayscale leaf foreground:

$$
R(x, y) = I_{\text{gray}}(x, y) - \text{MedianBlur}_{3\times3}(I_{\text{gray}})(x, y), \quad (x, y) \in M_{\text{fg}}
$$

The robust noise scale is:

$$
\hat{\sigma}_{\text{MAD}} = \frac{\text{median}_{(x, y) \in M_{\text{fg}}}\big(|R(x, y) - \text{median}(R)|\big)}{0.6745}
$$

$$
I^{(1)} = \begin{cases}
\text{GaussianBlur}_{3\times3}\big(\text{MedianBlur}_{3\times3}(I), \; \sigma = 0.65\big) & \text{if } \hat{\sigma}_{\text{MAD}} \ge 10.0, \\
\text{GaussianBlur}_{3\times3}(I, \; \sigma = 0.55) & \text{if } 5.5 \le \hat{\sigma}_{\text{MAD}} < 10.0, \\
I & \text{if } \hat{\sigma}_{\text{MAD}} < 5.5.
\end{cases}
$$

#### 2. Central-Window Specular Flash Glare Gate (`apply_glare_inpainting`)

Inside the central 72% window:

$$
\Omega_{\text{center}} = [0.14H : 0.86H, \; 0.14W : 0.86W] \cap M_{\text{fg}}
$$

candidate specular glare pixels are identified by $V(x, y) > 238 \land S(x, y) < 38$. If the foreground glare ratio satisfies $0.003 \le r_{\text{glare}} \le 0.220$, $M_{\text{glare}}$ is dilated with a $5\times5$ elliptical element $M_{\text{glare}}^{\oplus}$ and replaced by an $11\times11$ Gaussian neighborhood fill ($\sigma = 2.5$):

$$
I^{(2)}(x, y) = \begin{cases}
\text{GaussianBlur}_{11\times11}(I^{(1)}, \; \sigma = 2.5)(x, y) & \text{if } (x, y) \in M_{\text{glare}}^{\oplus} \land 0.003 \le r_{\text{glare}} \le 0.220, \\
I^{(1)}(x, y) & \text{otherwise.}
\end{cases}
$$

#### 3. Foreground-Restricted CIE $L^{\ast}$ Percentile Stretch and Mild CLAHE

Let non-glare foreground pixels be denoted by:

$$
\Omega_{\text{valid}} = M_{\text{fg}} \setminus M_{\text{glare}}^{\oplus}
$$

If foreground contrast is compressed ($\sigma(L^{\ast}) < 38.0$ or $P(95) - P(5) < 132.0$), we linearly stretch $[P(2), P(98)] \to [12, 242]$ and apply CLAHE ($\text{clipLimit} = 1.55$, $8\times8$ tiles) exclusively on $\Omega_{\text{valid}}$.

<div align="center">
  <img src="../results/figures/Fig06_Quality_Gate_MAD_Glare_CLAHE_Proof.png" width="94%" alt="Figure 6: Quality Gate Proof" />
  <p><em>Figure 6: Quality-gated optical preprocessing visual and quantitative proof.</em></p>
</div>

---

## 3. Pillar III: Shared-Sobel Seven-Branch Feature Engine and `BlockPCACompressor` (`src/mango_edge_cv/features.py`, `models.py`)

### 3.1 Unified $128\times128$ Grid and Shared Derivative Reuse

On the downsampled $128\times128$ image $I_{128}$, a single $3\times3$ Sobel pass computes:

$$
G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix} \ast I_{\text{gray},128}, \quad
G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix} \ast I_{\text{gray},128}
$$

$$
M = \sqrt{G_x^2 + G_y^2}, \quad \Theta = \left(\text{atan2}(G_y, G_x) \cdot \frac{180}{\pi}\right) \bmod 180^\circ
$$

The concatenated raw feature vector $\mathbf{x} \in \mathbb{R}^{2,003}$ comprises:

$$
\mathbf{x} = \Big[\underbrace{\mathbf{f}_{\text{hsv}}}_{52\text{D}};\; \underbrace{\mathbf{f}_{\text{lab}}}_{57\text{D}};\; \underbrace{\mathbf{f}_{\text{lbp}}}_{28\text{D}};\; \underbrace{\mathbf{f}_{\text{patch5}}}_{45\text{D}};\; \underbrace{\mathbf{f}_q}_{7\text{D}};\; \underbrace{\mathbf{f}_{\text{mgct}}}_{50\text{D}};\; \underbrace{\mathbf{f}_{\text{hog}}}_{1,764\text{D}}\Big] \in \mathbb{R}^{2,003}
$$

<div align="center">
  <img src="../results/figures/Fig07_Seven_Feature_Branches_Visual_Maps.png" width="94%" alt="Figure 7: Seven Feature Branches" />
  <p><em>Figure 7: Visual maps of the seven complementary feature branches extracted on a symptomatic mango leaf.</em></p>
</div>

### 3.2 Energy-Balanced Block-PCA Subspace Projection (`BlockPCACompressor`)

Partitioning raw features $\mathbf{x}$ into the 239D chromatic-texture-quality block and the 1,764D spatial gradient block:

$$
\mathbf{x} = [\mathbf{x}_{\text{ct}}; \; \mathbf{x}_{\text{hog}}] \in \mathbb{R}^{2,003}, \quad \mathbf{x}_{\text{ct}} \in \mathbb{R}^{239}, \quad \mathbf{x}_{\text{hog}} \in \mathbb{R}^{1,764}
$$

`BlockPCACompressor` fits two separate Z-score standardizers on the training split, projects centered HOG features onto their top $k = 64$ unwhitened principal components $\mathbf{z}_{\text{hog}} \in \mathbb{R}^{64}$, and balances block RMS energy with $\gamma = 0.45$:

$$
\alpha_{\text{hog}} = 0.45 \cdot \frac{\sqrt{\frac{1}{239 N_{\text{tr}}}\sum_{i=1}^{N_{\text{tr}}} \Vert\tilde{\mathbf{x}}_{\text{ct},i}\Vert_2^2}}{\sqrt{\frac{1}{64 N_{\text{tr}}}\sum_{i=1}^{N_{\text{tr}}} \Vert\mathbf{z}_{\text{hog},i}\Vert_2^2} + 10^{-8}}, \qquad \mathbf{z} = \begin{bmatrix} \tilde{\mathbf{x}}_{\text{ct}} \\ \alpha_{\text{hog}}\mathbf{z}_{\text{hog}} \end{bmatrix} \in \mathbb{R}^{303}
$$

<div align="center">
  <img src="../results/figures/Fig08_BlockPCA_Dimensionality_Dilution_Solution.png" width="94%" alt="Figure 8: Block-PCA Solution" />
  <p><em>Figure 8: Block-PCA dimensionality dilution and energy balancing solution.</em></p>
</div>

---

## 4. Pillar IV: Temperature-Calibrated Two-Stage Confidence Gate (`src/mango_edge_cv/models.py`)

Given $\mathbf{z} \in \mathbb{R}^{303}$, Stage 1 evaluates a multinomial L2-regularized Logistic Regression classifier ($C = 0.35$) whose logits $\mathbf{s}_1(\mathbf{z}) \in \mathbb{R}^8$ are calibrated via validation NLL minimization ($T^{\ast} = 0.758$):

$$
P_1(y = c \mid \mathbf{z}) = \frac{\exp\big(s_{1,c}(\mathbf{z}) / T^{\ast}\big)}{\sum_{k=1}^{8} \exp\big(s_{1,k}(\mathbf{z}) / T^{\ast}\big)}
$$

When calibrated Stage-1 confidence is below threshold $\tau_{\text{gate}} = 0.65$ (1.53% of test samples), the sample is routed to the Stage-2 nonlinear specialist ensemble:

$$
P_2(y \mid \mathbf{z}) = 0.55\,P_{\text{RBF-SVM}}(y \mid \mathbf{z}) + 0.30\,P_{\text{MLP-128}}(y \mid \mathbf{z}) + 0.15\,P_{\text{Cosine-kNN}}(y \mid \mathbf{z})
$$

$$
P_{\text{final}}(y \mid \mathbf{z}) = 0.30\,P_1(y \mid \mathbf{z}) + 0.70\,P_2(y \mid \mathbf{z})
$$

<div align="center">
  <img src="../results/figures/Fig10_TwoStage_Confidence_Gate_Architecture.png" width="94%" alt="Figure 10: Two-Stage Gate Architecture" />
  <p><em>Figure 10: Two-stage confidence-gated routing architecture.</em></p>
</div>

---

## 5. Pillar VI: Eigenvector Canonical Leaf Pose Normalization (`src/mango_edge_cv/preprocessing.py`)

For cross-dataset transfer (`MangoLeafBD` $\to$ `MLDID`), `canonicalize_leaf_organ_image` computes the $2\times2$ spatial covariance matrix of foreground coordinates $(x, y) \in M_{\text{fg}}$, extracts the principal eigenvector orientation:

$$
\theta = \frac{1}{2}\text{atan2}(2\mu_{11}, \; \mu_{20} - \mu_{02})
$$

rotates the organ by $-\theta$ so the midrib lies horizontally, crops tightly to the organ bounding box (6% margin) over neutral gray $[232, 232, 232]$, and normalizes foreground luminance percentiles $[P(5), P(95)] \to [18, 238]$.

<div align="center">
  <img src="../results/figures/Fig14_CrossDataset_Canonical_Pose_and_CORAL.png" width="94%" alt="Figure 14: Canonical Leaf Pose" />
  <p><em>Figure 14: Cross-dataset canonical leaf pose normalization and zero-shot transfer comparison on external MLDID field images.</em></p>
</div>
