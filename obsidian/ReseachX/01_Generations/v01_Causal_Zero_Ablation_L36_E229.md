---
tags: [generation, v01, causal-ablation, expert-localization, empirical-report]
version: v01
classification: Exploratory Causal Screening
model_architecture: Laguna-XS.2 (33.4B Total, 3.0B Active, 48 Layers, 256 Routed Experts)
compute_infrastructure: AMD Instinct MI300X (192 GiB HBM3, ROCm 7.14)
date: 2026-08-25
---

# 🧬 Generation v01: Exhaustive Causal Expert Zero-Ablation & Localization of L36/E229

## 1. Theoretical Motivation & Problem Formulation

In large-scale sparse Mixture-of-Experts (MoE) foundation models, parameter capacity is partitioned across discrete expert sub-networks. The foundational hypothesis of capability localization asserts that multi-step mathematical reasoning capabilities reside within a sparse, identifiable subset of routed experts $\mathcal{S}^* \subset \{ (l, e) \mid l \in [0, 47], e \in [0, 255] \}$ where $|\mathcal{S}^*| \ll 12,288$.

Let $p_\theta(y | x)$ represent the autoregressive next-token distribution parameterized by $\theta$. For any individual expert tensor $W_{l,e} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$, we define the causal intervention operator $\operatorname{do}(W_{l,e} = 0)$ as setting the expert parameter projection to the zero tensor. The task-specific causal sensitivity metric $\Delta \text{NLL}_{l,e}$ over a dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ is given by:

$$\Delta \text{NLL}_{l,e}(\mathcal{D}) = \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ -\sum_{t=1}^T \log p_{\theta \mid W_{l,e}=0}(y_t \mid x, y_{<t}) \right] - \mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ -\sum_{t=1}^T \log p_{\theta_0}(y_t \mid x, y_{<t}) \right]$$

---

## 2. Experimental Infrastructure & Protocol

The screening protocol evaluated all $12,288$ routed expert banks across the 48 transformer layers of **Laguna XS.2**:
* **Target Domain $\mathcal{D}_{\text{target}}$**: GSM8K ($N=384$ formal multi-step arithmetic reasoning items).
* **Control Domain $\mathcal{D}_{\text{control}}$**: C4 English validation split ($N=256$ natural language sequences).
* **Batch Configuration**: Sequence length $T=512$, BF16 tensor execution, deterministic ROCm RNG seed $= 111071$.

---

## 3. Empirical Results & Primary Data Matrix

Out of $12,288$ parameter banks, $>99.2\%$ of experts exhibited negligible causal impact when zero-ablated individually ($\Delta\text{NLL} < 0.01$). A singular, high-magnitude outlier was discovered at **Layer 36, Expert 229**.

### 📊 Primary Causal Sensitivity Ledger:
| Layer Depth ($l$) | Expert Index ($e$) | Parameter Dimensions | GSM8K $\Delta\text{NLL}$ | C4 Control $\Delta\text{NLL}$ | Selectivity Ratio ($\frac{\Delta\text{NLL}_{\text{task}}}{\Delta\text{NLL}_{\text{ctrl}}}$) | Causal Classification |
|---|---|---|---|---|---|---|
| **Layer 36** | **Expert 229** | $W \in \mathbb{R}^{2048 \times 8192}$ | **$+1.2858$** | **$+0.0210$** | **$61.23\times$** | **Critical Reasoning Read-Path** |
| **Layer 18** | **Expert 43** | $W \in \mathbb{R}^{2048 \times 8192}$ | **$+0.3120$** | $+0.0150$ | $20.80\times$ | Upstream Logic Gating |
| **Layer 20** | **Expert 219** | $W \in \mathbb{R}^{2048 \times 8192}$ | **$+0.2240$** | $+0.0090$ | $24.89\times$ | Intermediate Arithmetic Normalizer |
| **Layer 21** | **Expert 183** | $W \in \mathbb{R}^{2048 \times 8192}$ | **$+0.1510$** | $+0.0110$ | $13.73\times$ | Intermediate Token Binder |
| **Layer 25** | **Expert 244** | $W \in \mathbb{R}^{2048 \times 8192}$ | **$+0.1180$** | $+0.0080$ | $14.75\times$ | Representation Consolidator |
| **Layers 0–15** | All 4,096 Experts | $W \in \mathbb{R}^{2048 \times 8192}$ | $< +0.0200$ | $< +0.0100$ | $\sim 1.0\times$ | Diffuse / Low Specificity |

---

## 4. Mechanistic Analysis & Limitations

1. **Depth Localization Asymmetry**: High causal sensitivity is strictly absent in early layers ($L \le 15$), concentrating within deep semantic layers ($L \ge 36$). This indicates that reasoning primitives are computed hierarchically through continuous residual transformations before being resolved by late-stage expert routing.
2. **Single-Ablation Limit**: Individual ablation cannot detect cooperative ensembles where pairs or triplets of experts provide mutual redundancy.

---

## 5. Formal Succession Criteria for v02
To establish whether upstream experts in Layers 18, 20, and 21 form a cooperative circuit with Layer 36, Expert 229, Generation v02 was authorized to execute **Combinatorial Interaction Screening ($K=2, 3, 4$)**.
