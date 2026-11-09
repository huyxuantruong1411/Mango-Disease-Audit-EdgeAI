# Methodology & Mathematical Formulations

This document presents the complete mathematical formulation of the **Mango-Disease-Audit-EdgeAI** framework as implemented in `src/mango_edge_cv/` and locked in Version 5 (`v5`, `SEED = 20261022`).

![Pipeline Architecture Overview](../results/figures/Figure1_Pipeline_Architecture_Overview.png)

---

## 1. Specimen-Cluster Graph Audit & Five-View Shortcut Protocol (`audit.py`)

### 1.1 Perceptual Hash & Chromatic Graph Clustering
Given an input dataset $\mathcal{D} = \{(I_i, y_i)\}_{i=1}^N$, each image $I_i$ is mapped to:
1. **64-bit Discrete Cosine Transform Perceptual Hash (`pHash`)**: Computed on the $32 \times 32$ grayscale luminance plane via 2D DCT, retaining the top-left $8 \times 8$ low-frequency coefficients relative to their median $m$:
   $$h_i(u,v) = \mathbb{I}\!\left(\text{DCT}_{32}(I_i)_{u,v} > m\right), \quad (u,v) \in \{0,\dots,7\}^2$$
2. **36-D HSV Chromatic Signature ($\mathbf{c}_i \in \mathbb{R}^{36}$)**: Concatenation of $L_1$-normalized histograms over Hue ($16$ bins), Saturation ($10$ bins), and Value ($10$ bins), standardized to unit $L_2$ norm.

Within each class $y$, an undirected graph $\mathcal{G}_y = (\mathcal{V}_y, \mathcal{E}_y)$ is constructed where an edge $(i, j) \in \mathcal{E}_y$ exists if and only if:
$$d_{\text{Hamming}}(h_i, h_j) \le 6 \quad \lor \quad \frac{\langle \mathbf{c}_i, \mathbf{c}_j \rangle}{\|\mathbf{c}_i\|_2 \|\mathbf{c}_j\|_2} \ge 0.992$$
Connected components of $\bigcup_{y=1}^C \mathcal{G}_y$ define the disjoint specimen cluster IDs $g_i \in \{1, \dots, G\}$. On `MangoLeafBD` ($N=4,000$), this yields $G = 2,554$ unique clusters ($36.15\%$ near-duplicate/augmentation-mix ratio). Partitioning via `StratifiedGroupKFold` guarantees:
$$\mathcal{G}_{\text{train}} \cap \mathcal{G}_{\text{val}} = \varnothing, \quad \mathcal{G}_{\text{train}} \cap \mathcal{G}_{\text{test}} = \varnothing, \quad \mathcal{G}_{\text{val}} \cap \mathcal{G}_{\text{test}} = \varnothing$$

### 1.2 Five-View Background Shortcut Audit
To quantify non-biological studio background and image-framing confounders, we evaluate identical linear classifiers across five spatial representations under the group-disjoint split (Table 2):
1. **View 1 — Whole ROI (`Center-Crop r=0.86`)**: Standard center-cropped RGB frame ($256 \times 256$).
2. **View 2 — Segmented Organ Only**: Foreground leaf pixels retained ($\mathbf{M}_{\text{fg}} = 1$); background zeroed out.
3. **View 3 — Background Only**: Leaf interior masked out ($\mathbf{M}_{\text{fg}} = 0$); only studio backdrop retained.
4. **View 4 — Four Corner Patches Only**: Only the four $32 \times 32$ outer border corners retained.
5. **View 5 — Image File Metadata Only**: 11-D vector of file size (KB), native width, height, aspect ratio, and global RGB mean/std.

---

## 2. Foreground-Aware Quality-Gated Preprocessing (`preprocessing.py`)

### 2.1 Hybrid HSV-Saturation & CIE-LAB Chroma Otsu Segmentation
Given an RGB image $I \in \{0,\dots,255\}^{H \times W \times 3}$, we transform to HSV $(H, S, V)$ and CIE-LAB $(L^*, a^*, b^*)$. Centered opponent chroma is computed as:
$$C^*(x,y) = \sqrt{\left(a^*(x,y) - 128\right)^2 + \left(b^*(x,y) - 128\right)^2}$$
We fuse saturation and perceptual chroma into a foreground salience map:
$$S_{\text{fg}}(x,y) = \text{clip}\!\left(0.60\, S(x,y) + 1.25\, C^*(x,y),\; 0,\; 255\right)$$
Applying Otsu's threshold $\tau_{\text{Otsu}}$ followed by studio backdrop suppression ($V > 230 \land S < 28$, or $V < 15$), morphological opening/closing ($7 \times 7$ elliptical structuring element), and largest connected-component selection yields the binary leaf organ mask $\mathbf{M}_{\text{fg}} \in \{0, 255\}^{H \times W}$.

### 2.2 7-D Optical Quality Vector ($\mathbf{q} \in \mathbb{R}^7$) & Conditional Restoration Gates
Before modifying any pixel intensities, we extract a 7-D diagnostic quality vector $\mathbf{q}$:
$$\mathbf{q} = \left[\mu_L,\; \sigma_L,\; P_{95}(L^*) - P_5(L^*),\; \rho_{\text{glare}},\; \log_{10}(1 + \text{Var}(\nabla^2 I_{\text{gray}})),\; \mu_S,\; \hat{\sigma}_{\text{MAD}}\right]^\top$$
where $\hat{\sigma}_{\text{MAD}}$ is the **Donoho–Johnstone Median Absolute Deviation (MAD)** high-frequency sensor noise estimator computed from the $3 \times 3$ Laplacian residual $R = I_{\text{gray}} - \text{GaussianBlur}_{3\times 3}(I_{\text{gray}})$:
$$\hat{\sigma}_{\text{MAD}} = 1.4826 \times \text{median}\!\left(|R - \text{median}(R)|\right)$$

Each restoration operator is triggered **selectively** based on $\mathbf{q}$:
1. **Conditional Bilateral Denoising**: Triggered if $\hat{\sigma}_{\text{MAD}} > 2.2$ (adaptive spatial/range $\sigma$ scaled by $\hat{\sigma}_{\text{MAD}}$).
2. **Localized Specular Glare Telea Inpainting**: Triggered if glare pixel fraction $\rho_{\text{glare}} = \Pr(V > 242 \land S < 38) \in (0.002, 0.10)$, inpainting dilated glare regions via Alexandru Telea's fast marching method ($r = 3\text{ px}$).
3. **Conditional LAB-Luminance CLAHE**: Triggered if dynamic range $P_{95}(L^*) - P_5(L^*) < 95$ or $\mu_L < 95$ or $\mu_L > 185$ (clip limit $1.8$, $8 \times 8$ tiles).
4. **Conditional Unsharp Masking**: Triggered if Laplacian sharpness $\text{Var}(\nabla^2 I_{\text{gray}}) < 55.0$ and $\hat{\sigma}_{\text{MAD}} \le 3.0$.

### 2.3 Canonical Leaf Pose Normalization (`canonicalize_leaf_organ_image`)
To resolve the $90^\circ$ landscape-vs-portrait acquisition shift between `MangoLeafBD` and `MLDID`, we compute the $2 \times 2$ covariance matrix of foreground pixel coordinates $\{(x_k, y_k)\}_{k \in \mathbf{M}_{\text{fg}}}$ and extract the principal eigenvector angle $\theta$:
$$\theta = \frac{1}{2} \arctan2\!\left(2\mu_{11},\; \mu_{20} - \mu_{02}\right)$$
Rotating both image and mask by $\theta$ aligns the leaf's major inertia axis horizontally, followed by tight bounding-box cropping ($6\%$ margin), aspect-preserving resize to $256 \times 256$, and foreground $L^*$ standardization (`CANONICAL_COLOR_MODE = "lab_l"`):
$$L^*_{\text{canon}}(x,y) = \text{clip}\!\left(\frac{L^*(x,y) - \mu_{L^*,\text{fg}}}{\max(\sigma_{L^*,\text{fg}}, 6.0)} \cdot 38.0 + 132.0,\; 0,\; 255\right)$$

---

## 3. Seven-Branch Handcrafted Feature Extraction (`features.py`, $D = 2,003$)

All spatial descriptors are computed on a standardized $128 \times 128$ foreground-masked grid where the Sobel gradients $(G_x, G_y)$ and gradient magnitude $M = \sqrt{G_x^2 + G_y^2}$ are **computed once and shared** across branches:

| Branch | Symbol | Dim | Mathematical Formulation |
| :--- | :---: | :---: | :--- |
| **1. HSV Chromatic** | $\mathbf{f}_{\text{hsv}}$ | $52\text{D}$ | Density histograms ($16 H + 12 S + 12 V = 40\text{D}$) + $\mu, \sigma, \gamma_1$ per channel ($9\text{D}$) + necrotic/chlorotic/saturation pixel ratios ($3\text{D}$). |
| **2. CIE-LAB Color** | $\mathbf{f}_{\text{lab}}$ | $57\text{D}$ | Density histograms ($12 L^* + 16 a^* + 16 b^* = 44\text{D}$) + $\mu, \sigma, Q_{10}, Q_{90}$ per channel ($12\text{D}$) + mean chroma $\bar{C}^*$ ($1\text{D}$). |
| **3. LUT Multi-Scale LBP** | $\mathbf{f}_{\text{lbp}}$ | $28\text{D}$ | Precomputed 256-entry lookup table $\text{LBP}_{8,1}^{\text{riu2}} \in \{0,\dots,9\}$ at $128\times128$ ($10\text{D}$) and $64\times64$ ($10\text{D}$) + multi-scale gradient mean/std, edge densities, and Shannon entropy ($8\text{D}$). |
| **4. 5-Patch Spatial** | $\mathbf{f}_{\text{patch5}}$ | $45\text{D}$ | 5 overlapping spatial regions (4 quadrants + central $50\%$ patch) $\times$ 9 chromatic moments ($\mu_H, \sigma_H, \mu_S, \sigma_S, \mu_V, \sigma_V, \mu_{L^*}, \mu_{a^*}, \mu_{b^*}$). |
| **5. Quality Vector** | $\mathbf{f}_{\text{q}}$ | $7\text{D}$ | Standardized optical quality metadata vector $\mathbf{q} \in \mathbb{R}^7$ from Section 2.2. |
| **6. Multi-Grid MGCT** | $\mathbf{f}_{\text{mgct}}$ | $50\text{D}$ | $4 \times 4$ spatial grid means of $(a^*, b^*, M)$ ($48\text{D}$) + foreground-to-center contrast ratios in $a^*$ and gradient magnitude $M$ ($2\text{D}$). |
| **7. Vectorized Sobel HOG** | $\mathbf{f}_{\text{hog}}$ | $1,764\text{D}$ | $8 \times 8$ cells ($16\times16\text{ px}$), 9 orientation bins in $[0^\circ, 180^\circ)$ with bilinear angle interpolation, $7 \times 7$ overlapping $2\times2$-cell blocks with L2-Hys normalization ($\epsilon = 0.2$) $\to 49 \times 36 = 1,764\text{D}$. |

Concatenating all seven branches produces the raw vector:
$$\mathbf{x} = \left[\underbrace{\mathbf{f}_{\text{hsv}}^\top,\; \mathbf{f}_{\text{lab}}^\top,\; \mathbf{f}_{\text{lbp}}^\top,\; \mathbf{f}_{\text{patch5}}^\top,\; \mathbf{f}_{\text{q}}^\top,\; \mathbf{f}_{\text{mgct}}^\top}_{\mathbf{x}_{\text{ct}} \in \mathbb{R}^{239}},\; \underbrace{\mathbf{f}_{\text{hog}}^\top}_{\mathbf{x}_{\text{hog}} \in \mathbb{R}^{1,764}}\right]^\top \in \mathbb{R}^{2,003}$$

---

## 4. Energy-Balanced Block-PCA & Two-Stage Confidence Gate (`models.py`)

### 4.1 Energy-Balanced Block-PCA Compression ($2,003\text{D} \to 303\text{D}$)
Directly feeding $\mathbf{x} \in \mathbb{R}^{2,003}$ into a regularized linear classifier causes **dimensionality dilution**: the $1,764\text{D}$ HOG block represents $88.1\%$ of all coordinates and drowns out the $239\text{D}$ chromatic-texture descriptors (causing accuracy to drop from $97.35\%$ in $M_2$ to $86.61\%$ in $M_4$, Table 3).

`BlockPCACompressor` resolves this by:
1. Standardizing the $239\text{D}$ chromatic-texture block independently: $\tilde{\mathbf{x}}_{\text{ct}} = \text{StandardScaler}(\mathbf{x}_{\text{ct}}) \in \mathbb{R}^{239}$.
2. Projecting standardized $\tilde{\mathbf{x}}_{\text{hog}} \in \mathbb{R}^{1,764}$ onto its top $K = 64$ principal components $\mathbf{z}_{\text{hog}} = \mathbf{W}_{\text{PCA}}^\top \tilde{\mathbf{x}}_{\text{hog}} \in \mathbb{R}^{64}$ (preserving structural shape energy without whitening noisy tail eigenvectors).
3. Scaling $\mathbf{z}_{\text{hog}}$ by an RMS energy-balancing factor $\alpha_{\text{hog}}$ with weight $w_{\text{hog}} = 0.45$:
   $$\alpha_{\text{hog}} = 0.45 \times \frac{\text{RMS}(\tilde{\mathbf{x}}_{\text{ct}})}{\text{RMS}(\mathbf{z}_{\text{hog}})}, \qquad \mathbf{z} = \left[\tilde{\mathbf{x}}_{\text{ct}}^\top,\; \alpha_{\text{hog}} \mathbf{z}_{\text{hog}}^\top\right]^\top \in \mathbb{R}^{303}$$
This compression jumps group-disjoint test accuracy from **92.05% ($M_9$, $2,003\text{D}$) to 97.91% ($M_{10}$, $303\text{D}$, $p < 0.0001$)**.

### 4.2 Temperature-Calibrated Stage-1 & Selective Stage-2 Routing
- **Stage-1 Fast Linear Classifier**: An $L_2$-regularized multinomial Logistic Regression produces logits $\boldsymbol{\ell}_1(\mathbf{z}) \in \mathbb{R}^C$. Temperature scaling $T^* > 0$ is optimized via grid search on `Val_Orig` to minimize Negative Log-Likelihood (NLL):
  $$P_1(y = c \mid \mathbf{z}) = \frac{\exp(\ell_{1,c}(\mathbf{z}) / T^*)}{\sum_{k=1}^C \exp(\ell_{1,k}(\mathbf{z}) / T^*)}$$
- **Confidence Gate ($\tau_{\text{gate}} = 0.65$)**: Let $p_{\max}(\mathbf{z}) = \max_{c} P_1(y = c \mid \mathbf{z})$.
  - If $p_{\max}(\mathbf{z}) \ge \tau_{\text{gate}}$ (**98.47% of test images**), the pipeline exits immediately with $P_{\text{final}}(\mathbf{z}) = P_1(\mathbf{z})$ in **21.58 ms/img**.
  - If $p_{\max}(\mathbf{z}) < \tau_{\text{gate}}$ (**only 1.53% of test images**), the sample is routed to a **Stage-2 Nonlinear Specialist Ensemble** combining an RBF-kernel SVM ($0.55$), a 128-unit shallow MLP ($0.30$), and a distance-weighted Cosine $k$-NN ($0.15$):
    $$P_2(\mathbf{z}) = 0.55\, P_{\text{SVM}}(\mathbf{z}) + 0.30\, P_{\text{MLP}}(\mathbf{z}) + 0.15\, P_{\text{kNN}}(\mathbf{z})$$
    $$P_{\text{final}}(\mathbf{z}) = 0.30\, P_1(\mathbf{z}) + 0.70\, P_2(\mathbf{z})$$

### 4.3 Transductive Block-Shrinkage CORAL (`coral_align_features`)
For unsupervised domain adaptation on `MLDID`, we first align marginal feature distributions via 201-point empirical quantile interpolation, then apply **block-diagonal covariance shrinkage and whitening/re-coloring** separately within each semantic feature block $b \in \{[0{:}52], [52{:}109], [109{:}137], [137{:}182], [182{:}189], [189{:}239], [239{:}303]\}$:
$$\hat{\boldsymbol{\Sigma}}_b = (1 - \lambda)\boldsymbol{\Sigma}_b + \lambda\,\text{diag}(\boldsymbol{\Sigma}_b) + \epsilon\mathbf{I}, \qquad \mathbf{A}_b = (1 - \eta)\mathbf{I} + \eta\,\hat{\boldsymbol{\Sigma}}_{t,b}^{-1/2}\hat{\boldsymbol{\Sigma}}_{s,b}^{1/2}$$
where $\lambda = 0.35$ and anchor weight $\eta = 0.20$.
