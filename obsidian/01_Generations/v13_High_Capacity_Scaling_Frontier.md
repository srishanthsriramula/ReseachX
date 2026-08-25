---
tags: [generation, v13, scaling, high-capacity, frontier, roadmap]
version: v13
status: planned
model: Laguna-XS.2 (33.4B-A3B)
---

# 🧬 Generation v13: High-Capacity Scaling & Multi-Capability Generalization

## 1. Executive Summary & Research Motivation
Having mathematically and empirically established both the **Stratified Depth Hierarchy** (v11: $+1.48\text{ pp}$) and the **Soft Riemannian Safety Shield** (v12: up to $88\%$ drift reduction on MBPP), we now have the necessary safety guarantees to scale parameter capacity.

### The Core Research Question:
> *"Does scaling LoRA rank from $r=63 \to r=128–256$ ($25\text{M}–50\text{M}$ parameters) across Stratified Layers unlock $+5\text{ to }+8\text{ percentage point}$ breakthrough gains without compromising general capabilities?"*

---

## 2. Experimental Design & Scaling Matrix

| Experimental Arm | Layer Distribution | LoRA Rank ($r$) | Trainable Parameters | Riemannian Damping ($\alpha$) | Target Benchmark | Retained Control | Target Accuracy Gain |
|---|---|---|---|---|---|---|---|
| **v12 Baseline** | `[1, 2, 8, 11, 12, 16, 21, 26]` | $r=63$ | $12.64\text{M}$ | $0.01$ | GSM8K ($N=384$) | MBPP ($N=160$) | $+0.78\text{ pp} \to +1.48\text{ pp}$ |
| **v13 Scale A** | `[1, 2, 8, 11, 12, 16, 21, 26]` | $r=128$ | $25.69\text{M}$ | $0.01$ | GSM8K ($N=384$) | MBPP ($N=160$) | **Target: $+3.5\text{ to }+5.0\text{ pp}$** |
| **v13 Scale B** | `[1, 2, 8, 11, 12, 16, 21, 26]` | $r=256$ | $51.38\text{M}$ | $0.01$ | GSM8K ($N=384$) | MBPP ($N=160$) | **Target: $+5.0\text{ to }+8.0\text{ pp}$** |
| **v13 Multi-Task** | `[1, 2, 8, 11, 12, 16, 21, 26]` | $r=128$ | $25.69\text{M}$ | $0.01$ | GSM8K + MBPP | C4 English Fluency | **Simultaneous Math + Code Repair** |

---

## 3. The 3 Scaling Laws to Be Tested in v13

1. **Rank Saturation vs. Scaling Linearity**: Does accuracy scale logarithmically with rank $r$, or does the Stratified Hierarchy allow linear scaling up to $r=256$?
2. **Riemannian Invariance Under High Rank**: Does $\alpha = 0.01$ continue to provide $>80\%$ drift suppression when adapter norm $\|\Delta W\|_F$ quadruples?
3. **Cross-Capability Superposition**: Can orthogonal LoRA vectors $\Delta W_{\text{math}}$ and $\Delta W_{\text{code}}$ be linearly merged via task arithmetic without mutual interference?
