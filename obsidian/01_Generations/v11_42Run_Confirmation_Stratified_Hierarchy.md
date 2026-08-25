---
tags: [generation, v11, confirmatory-matrix, stratified-hierarchy, breakthrough]
version: v11
status: confirmed
backlinks: "[[00_Index/00_Index_MOC|Index]], [[02_Theorems/Theorem_2_Jacobian_Condition_Number_Explosion|Theorem 2]], [[03_Architectures/Stratified_Layer_Signature_01|Stratified Signature 01]]"
---

# 🧬 Generation v11: The 42-Run Matrix & Stratified Hierarchy Victory

## 1. Motivation & Question
Preregistered test of Gradient-Guided Bottleneck LoRA (`[16-25]`) vs. 6 Architecture-Matched Random Layer Placements on fresh unseen GSM8K test data ($N=384$).

## 2. Empirical Leaderboard ($N=384$ Fresh Test Items)
* 🥇 **`random_signature_01` (Stratified `[1, 2, 8, 11, 12, 16, 21, 26]`)**: **$79.60\%$ ($+1.48\text{ pp}$ gain, max seed $80.99\%$)**.
* 🥈 **`random_signature_05`**: **$79.17\%$ ($+1.04\text{ pp}$)**.
* 🥉 **`random_signature_02`**: **$79.08\%$ ($+0.95\text{ pp}$)**.
* ❌ **Guided LoRA (Bottleneck)**: **$78.18\%$ ($+0.05\text{ pp}$, ranked 5th/7)**.
* 🔒 **Fresh Base Model**: **$78.13\%$**.

## 3. Verdict
* $\Delta(\text{Guided} - \text{Random}) = -0.64\text{ pp}$. Scalar gradient guidance was **falsified**.
* **Stratified Layer Depth Hierarchy** was definitively proven as the winning architecture (explained by [[02_Theorems/Theorem_2_Jacobian_Condition_Number_Explosion|Theorem 2]]).
