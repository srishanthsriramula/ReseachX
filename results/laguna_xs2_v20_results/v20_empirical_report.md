# Empirical Report: Laguna-XS.2 v20 & v20.1 Frontier Science SFT

> **Model**: Laguna-XS.2 (33.4B MoE)  
> **Benchmark**: GPQA Diamond (198 questions)  
> **Training**: SciQ (1,000 questions)  
> **Date**: August 29–30, 2026

---

## Executive Summary

The v20 and v20.1 experiments evaluated early implementations of Geodesic SFT on GPQA Diamond. While initial v20 runs suggested strong geodesic performance (+3.1% gain on seed 107), subsequent forensic auditing uncovered two critical implementation flaws that compromised the validity of early comparisons.

---

## Forensic Audit Findings

### 1. Inactive Geodesic Constraint in v20
In `laguna_xs2_v20_frontier_science_system2_geodesic_repair.ipynb`, the function `apply_whitened_initialization_to_model` copied the whitened basis $A_0$ into LoRA $A$, but failed to set `sA.weight.requires_grad = False`. Consequently, both $A$ and $B$ were updated freely by AdamW, rendering v20 an unconstrained Warm LoRA run rather than a true Geodesic constraint.

### 2. Norm Amplification (2× Scale Disparity)
The unregularized whitened basis $A_0 = U_r^T C_{	ext{code}}^{-1/2}$ exhibited a Frobenius norm of $\|A_0\|_F pprox 72.4$, compared to Kaiming uniform initialization with $\|A_{	ext{Kaiming}}\|_F pprox 36.2$. This 2× norm inflation gave the whitened initialization an artificial effective learning rate advantage at step 1.

### 3. The Chicken-and-Egg Initialization Problem in Standard LoRA
Standard LoRA initializes $B=0$ and $A \sim \mathcal{N}(0, \sigma^2)$. At step 1:
$$rac{\partial L}{\partial A} = B^T rac{\partial L}{\partial y} x^T = 0$$
With only 32 training steps, standard LoRA required several steps for $B$ to grow before $A$ received meaningful gradients, leading to training loss divergence (Loss 1.04 → 1.83) in under-parameterized 32-step regimes.

---

## v20.1 Corrected Implementation
In v20.1, explicit Kaiming norm matching was implemented:
$$A_0 \leftarrow A_0 \cdot rac{\|A_{	ext{Kaiming}}\|_F}{\|A_0\|_F}$$
and `sA.weight.requires_grad = False` was strictly enforced. This stabilized loss progression (0.88 → 0.54) and isolated the true geometric constraint.
