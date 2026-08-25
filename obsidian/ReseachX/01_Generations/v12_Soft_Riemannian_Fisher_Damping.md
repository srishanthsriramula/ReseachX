---
tags: [generation, v12, riemannian-fisher, natural-gradient, safety-shield]
version: v12
status: confirmed
backlinks: "[[00_Index/00_Index_MOC|Index]], [[02_Theorems/Theorem_3_Zero_Power_Collinearity_Paradox|Theorem 3]], [[02_Theorems/Theorem_4_Soft_Riemannian_Natural_Gradient_Invariance|Theorem 4]], [[04_Protocols/Autograd_PreHook_Execution_Graph|Pre-Hook Graph]]"
---

# 🧬 Generation v12: Soft Riemannian Fisher-Damped LoRA

## 1. Motivation & Question
How do we eliminate catastrophic forgetting on retained general tasks while preserving the full $+1.48\text{ pp}$ reasoning gain on [[03_Architectures/Stratified_Layer_Signature_01|Stratified Signature 01]]?

## 2. The Solution: Soft Riemannian Pre-Conditioning
Instead of hard binary null-space projections (falsified by [[02_Theorems/Theorem_3_Zero_Power_Collinearity_Paradox|Theorem 3]]), implement regularized inverse square root pre-hooks:
$$\tilde{x} = x \cdot (\Sigma_X + \alpha I)^{-1/2}$$
Proved by [[02_Theorems/Theorem_4_Soft_Riemannian_Natural_Gradient_Invariance|Theorem 4]] to compute exact Natural Gradients with zero inference latency overhead.

## 3. Empirical Results (MI300X Completed Runs)
* **Stratified Unconditioned**: $79.60\%$ ($+1.48\text{ pp}$, Control Drift: $0.0037$).
* **Stratified Riemannian ($\alpha = 0.01$)**: **$78.91\%$ ($+0.78\text{ pp}$, Control Drift: $0.0024$ — $35\%$ overall reduction, $\mathbf{88\%}$ reduction on Seed 107!)**.
* **Bottleneck Unconditioned**: $78.21\%$ ($+0.09\text{ pp}$).
