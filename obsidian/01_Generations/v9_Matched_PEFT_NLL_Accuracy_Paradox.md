---
tags: [generation, v9, lora, nll-accuracy-paradox]
version: v9
status: completed
backlinks: "[[00_Index/00_Index_MOC|Index]], [[01_Generations/v10_Behavior_Aligned_Writeability_Dose|v10]]"
---

# 🧬 Generation v9: Matched PEFT & The NLL-Accuracy Decoupling Paradox

## 1. Motivation & Question
Evaluate Standard Low-Rank Adaptation (LoRA) on Attention sublayers (`q_proj`, `k_proj`, `v_proj`, `o_proj`) across all 40 layers.

## 2. The Discovery of the NLL-Accuracy Paradox
Minimizing cross-entropy loss does NOT monotonically improve multi-step reasoning accuracy:
* **Step 8**: $\text{NLL} = 0.7610 \implies \text{Accuracy} = \mathbf{79.60\%}$ (Peak accuracy).
* **Step 16**: $\text{NLL} = 0.6120 \implies \text{Accuracy} = \mathbf{77.81\%}$ (Collapsed below base model!).
* **Step 32**: $\text{NLL} = 0.4890 \implies \text{Accuracy} = \mathbf{74.20\%}$ (Severe overfitting).

## 3. Mechanistic Cause
Cross-entropy optimization fits surface formatting tokens, degrading arithmetic token calibration.
Transitioned to [[01_Generations/v10_Behavior_Aligned_Writeability_Dose|v10]] to lock the dose to **8 updates @ LR $1\times 10^{-5}$**.
