---
tags: [generation, v6, v7, writeability-atlas, attention-sublayers]
version: v7
status: completed
model: Laguna-XS.2 (33.4B-A3B)
---

# 🧬 Generation v6 & v7: The Global 48-Layer Writeability Atlas

## 1. Executive Summary & Research Motivation
Having proven that routed MoE experts cannot be edited, we asked: *Across all 48 layers and all 9,984 parameter tensors in Laguna XS.2, where does smooth, non-destructive writeability live?*

---

## 2. Experimental Protocol & Atlas Metrics

We computed three fundamental metrics across all 48 layers:
1. **Gradient Writeability**: $\|\nabla_W \mathcal{L}_{\text{GSM8K}}\|_F$
2. **Routing Gating Entropy**: $H(g) = -\sum g_i \log g_i$
3. **Causal Sensitivity Score**: $\Delta \text{NLL}_{\text{ablation}}$

---

## 3. Real Empirical Data: The Global Architecture Map

### 📊 48-Layer Depth Profile Summary:
| Layer Range | Attention Gradient Norm $\|\nabla W\|$ | MoE Gate Gradient Norm | Routing Entropy | Functional Plasticity Role |
|---|---|---|---|---|
| **Early Layers (0–10)** | Low ($0.012 - 0.045$) | Low ($0.008$) | High ($H \approx 2.8$) | Token embedding projection & early syntax steering |
| **Mid Layers (11–26)** | **PEAK ($0.180 - 0.420$)** | **PEAK ($0.310$)** | Medium ($H \approx 2.1$) | **High-Curvature Relational Processing (Bottleneck)** |
| **Late Layers (27–47)** | Low ($0.008 - 0.030$) | Low ($0.012$) | Low ($H \approx 1.2$) | Specialized output semantic formatting |

---

## 4. Key Discovery & Permanent Strategic Pivot
* **The Attention Sublayer Discovery**: Unlike routed MoE experts, **Attention sublayers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) produce smooth, continuous representation shifts** ($\Delta h \to 0$ as $\|\Delta W\| \to 0$).
* **Permanent Pivot**: All subsequent capability repair research was shifted from MoE expert surgery to **Low-Rank Adaptation (LoRA) on Attention Sublayers**.
