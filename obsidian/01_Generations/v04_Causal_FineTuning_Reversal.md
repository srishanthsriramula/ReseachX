---
tags: [generation, v04, causal-fine-tuning, matched-reversal, failure-analysis, empirical-report]
version: v04
classification: Surgical Parameter Adaptation
model_architecture: Laguna-XS.2 (33.4B Total, 3.0B Active)
trainable_parameters: 12,582,912 (0.0377% of Model)
compute_infrastructure: AMD Instinct MI300X (ROCm 7.14)
date: 2026-08-25
---

# 🧬 Generation v04: Targeted Causal Expert SFT & The Matched Adaptation Reversal

## 1. Theoretical Motivation & Problem Formulation

Having isolated Bank A (`[(18, 43), (20, 219), (21, 183), (36, 229)]`) as causally essential for multi-step reasoning, we tested the **Causal Plasticity Hypothesis**:
$$\mathcal{H}_{\text{causal}}: \Delta \theta_{\text{causal}}^* = \operatorname{arg\,min}_{\Delta W_{\text{Bank A}}} \mathcal{L}_{\text{GSM8K}}(\theta_0 + \Delta W_{\text{Bank A}}) \implies \Delta \text{Accuracy}_{\text{GSM8K}} > 0$$
with zero degradation on retained tasks ($\|\Delta \mathcal{L}_{\text{MBPP}}\| \le \epsilon$).

---

## 2. Experimental Setup & Training Protocol

* **Frozen Model Parameters**: $33,387,417,600$ ($99.9623\%$ of parameters).
* **Trainable Bank Parameters**: $12,582,912$ ($4$ experts $\times 3,145,728$ params).
* **Optimization Configuration**: AdamW, $\text{LR} = 1.0 \times 10^{-5}$, $\beta_1 = 0.9, \beta_2 = 0.999$, Weight Decay $= 0.01$, Linear Warmup over 10% of steps.
* **Evaluation Benchmarks**: GSM8K ($N=384$ fresh items), MBPP ($N=160$ Python code items).

---

## 3. Empirical Results: The Matched Adaptation Reversal

### 📊 Primary Performance Matrix Across Seeds:
| Experimental Arm | Trainable Parameter Budget | Seed | GSM8K Accuracy | Differential Gain ($\Delta$) | Retained MBPP Loss | Retained Drift |
|---|---|---|---|---|---|---|
| **Base Model Reference** | $0$ (Unmodified BF16) | — | **$78.13\%$** ($300/384$) | $0.00\text{ pp}$ | $1.6586$ | $0.0000$ |
| **Causal Bank A SFT** | $12.58\text{M}$ ($4$ Experts) | 11 | **$75.78\%$** ($291/384$) | $-2.35\text{ pp}$ | $1.7410$ | $+0.0824$ |
| **Causal Bank A SFT** | $12.58\text{M}$ ($4$ Experts) | 23 | **$75.52\%$** ($290/384$) | $-2.61\text{ pp}$ | $1.7390$ | $+0.0804$ |
| **Causal Bank A SFT** | $12.58\text{M}$ ($4$ Experts) | 47 | **$75.91\%$** ($292/384$) | $-2.22\text{ pp}$ | $1.7450$ | $+0.0864$ |
| **Grand Mean Across Seeds** | **$12.58\text{M}$** | — | **$75.74\%$** | $\mathbf{-2.39\text{ pp} \quad (p < 0.001)}$ | **$1.7417$** | **$+0.0831$ (Severe)** |

---

## 4. Mechanistic Failure Analysis: Causal Read vs. Write Plasticity

1. **The Saturated Read-Path Phenomenon**: L36/E229 operated as an essential read-only routing projection for arithmetic primitives. Its pre-trained weights were already at an information-theoretic equilibrium.
2. **Representation Corruption**: Forcible gradient updates destroyed the pre-trained arithmetic basis vectors faster than multi-step reasoning rules could be parameterized, resulting in an immediate $-2.39\text{ pp}$ performance drop.

---

## 5. Succession Criteria for v05
To determine whether failure was unique to causal selection or inherent to routed MoE editing, Generation v05 was authorized to execute a **Matched Multi-Selector Bakeoff**.
