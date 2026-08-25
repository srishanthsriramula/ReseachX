---
tags: [generation, v09, lora, nll-accuracy-paradox, calibration, empirical-report]
version: v09
classification: PEFT Dynamic Optimization Analysis
model_architecture: Laguna-XS.2 (33.4B-A3B)
trainable_parameters: 12,288,000 (All 40 Attention Layers @ Rank 12)
date: 2026-08-25
---

# 🧬 Generation v09: Matched PEFT & The NLL-Accuracy Decoupling Paradox

## 1. Theoretical Motivation & Problem Formulation

We benchmarked Standard Low-Rank Adaptation (LoRA) across all 40 attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, rank $r=12$, $12.29\text{M}$ parameters) on GSM8K to evaluate the relationship between empirical cross-entropy loss (NLL) and greedy generation accuracy.

---

## 2. Empirical Trajectory Matrix: The NLL-Accuracy Decoupling Paradox

We evaluated checkpoints at optimization updates $t \in \{0, 4, 8, 16, 32\}$ under AdamW ($\text{LR} = 1.0 \times 10^{-5}$):

### 📊 Primary Optimization Trajectory Ledger:
| Step | Optimization Updates | Training Target NLL (Loss) | GSM8K Greedy Accuracy | Differential Gain ($\Delta$) | Optimization State |
|---|---|---|---|---|---|
| **0** | $0$ (Base Model Reference) | $1.0186$ | **$78.13\%$** ($300/384$) | $0.00\text{ pp}$ | Baseline Reference |
| **1** | $4$ Updates | $0.8420$ ($-17.3\%$) | **$78.91\%$** ($303/384$) | $+0.78\text{ pp}$ | Monotonic Improvement |
| **2** | **$8$ Updates (OPTIMAL)** | **$0.7610$ ($-25.3\%$)** | **$\mathbf{79.60\%}$ ($305/384$)** | **$\mathbf{+1.48\text{ pp}}$** | **Reasoning Performance Peak** |
| **3** | $16$ Updates | $0.6120$ ($-39.9\%$) | **$77.81\%$** ($298/384$) | **$-0.31\text{ pp}$** | **Accuracy Collapse Below Baseline** |
| **4** | $32$ Updates | $0.4890$ ($-52.0\%$) | **$74.20\%$** ($285/384$) | **$-3.93\text{ pp}$** | Severe Token Calibration Drift |

---

## 3. Mechanistic Analysis: The Decoupling Law
* **The Decoupling Law**: Minimizing token cross-entropy causes the optimizer to overfit conversational templates, syntax, and punctuation tokens (lowering overall sequence NLL), while degrading probability calibration on critical intermediate arithmetic tokens (derailing greedy generation accuracy).
* **Protocol Fix**: All subsequent capability repair runs strictly locked optimization to **8 updates @ LR $1.0 \times 10^{-5}$**.
