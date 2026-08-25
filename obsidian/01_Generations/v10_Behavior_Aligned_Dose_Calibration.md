---
tags: [generation, v10, dose-calibration, gradient-peaking, hypothesis]
version: v10
status: completed
model: Laguna-XS.2 (33.4B-A3B)
---

# 🧬 Generation v10: Behavior-Aligned Dose Calibration & The Gradient Hypothesis

## 1. Executive Summary & Research Motivation
With optimization dose calibrated to 8 updates, we formulated the **Scalar Gradient-Guided Layer Selection Hypothesis**:
> *"Concentrating rank capacity into the 8 layers with the highest gradient norms (`guided_lora`: `[16, 18, 19, 20, 21, 23, 24, 25]`, rank 63, ~12.64M params) will outperform spreading rank thinly across all 40 layers (rank 12, ~12.29M params)."*

---

## 2. Experimental Setup & Protocol

* **Dose Calibration**: Locked to exactly **8 updates @ LR $1 \times 10^{-5}$**, Batch Accumulation = 8.
* **Architecture Comparison**:
  * **Standard LoRA**: 40 layers $\times$ Rank 12 = $12,288,000$ parameters.
  * **Guided LoRA**: 8 mid-bottleneck layers $\times$ Rank 63 = $12,644,352$ parameters.

---

## 3. Initial Development Findings & The Need for v11
* Exploratory development runs showed Guided LoRA reaching $\approx 78.5\%$ vs. Standard LoRA $\approx 77.8\%$.
* **The Limitation**: Development sets were small ($N=64$). We needed a **preregistered, double-blind confirmatory matrix on fresh unseen test data** ($N=384$) with architecture-matched random controls.
* **Transition Logic**: We transitioned to [[01_Generations/v11_42Run_Confirmation_Stratified_Victory|Generation v11]].
