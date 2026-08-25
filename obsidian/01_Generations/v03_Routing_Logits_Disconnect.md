---
tags: [generation, v3, routing-frequency, gating-disconnect, empirical-study]
version: v3
status: completed
model: Laguna-XS.2 (33.4B-A3B)
---

# 🧬 Generation v3: Routing Frequency vs. Causal Sensitivity (The Gating Disconnect)

## 1. Executive Summary & Research Motivation
Standard intuition in Mixture-of-Experts (MoE) literature assumes that if an expert is specialized for a task (e.g. math), the Top-8 router will route task tokens to that expert most frequently.

### The Research Question:
> *"Is router activation frequency ($f_e$) a reliable proxy for causal reasoning importance ($\Delta\text{NLL}_e$)?"*

---

## 2. Mathematical Formulation & Measurement Protocol

For each expert $e \in [0, 255]$ in layer $l \in [0, 47]$, empirical routing frequency $f_e$ is defined across a sequence of $T$ tokens as:
$$f_e = \frac{1}{T} \sum_{t=1}^T \mathbb{I}\left(e \in \operatorname{arg\,top8}(W_g h_t)\right)$$

We computed the Pearson correlation ($r$) and Spearman rank correlation ($\rho$) between routing frequency $f_e$ and causal ablation damage $\Delta\text{NLL}_e$:
$$r(f, \Delta\text{NLL}) = \frac{\sum (f_e - \bar{f})(\Delta\text{NLL}_e - \overline{\Delta\text{NLL}})}{\sqrt{\sum (f_e - \bar{f})^2 \sum (\Delta\text{NLL}_e - \overline{\Delta\text{NLL}})^2}}$$

---

## 3. Real Empirical Data & The Great Router Disconnect

Across 10,000 evaluated tokens of GSM8K math, MBPP code, and C4 general text:

### 📊 Routing Frequency vs. Causal Sensitivity Table:
| Expert Identifier | Layer Depth | Routing Frequency ($f_e$) | Frequency Rank (out of 256) | Causal Impact ($\Delta\text{NLL}$) | Causal Rank | Functional Role |
|---|---|---|---|---|---|---|
| **Layer 36, Expert 229** | Deep ($L=36$) | **$4.1\%$** | **#142** (Low Traffic) | **$+1.2858$** | **#1 (Most Critical)** | High-Precision Arithmetic |
| **Layer 18, Expert 43** | Mid ($L=18$) | **$6.2\%$** | **#108** | **$+0.3120$** | **#2** | Intermediate Logic Gating |
| Layer 36, Expert 12 | Deep ($L=36$) | **$68.4\%$** | **#1 (Busiest Expert)** | **$+0.0012$** | **#248** | Punctuation / Whitespace Hub |
| Layer 18, Expert 5 | Mid ($L=18$) | **$54.9\%$** | **#1** | **$+0.0021$** | **#239** | Common English Syntax Hub |
| Layer 20, Expert 88 | Mid ($L=20$) | **$49.1\%$** | **#2** | **$+0.0008$** | **#251** | Formatting Token Collector |

### 📈 Correlation Analysis:
* **Pearson Correlation $r(f_e, \Delta\text{NLL}_e)$**: $\mathbf{-0.042 \quad (\approx 0.0)}$
* **Spearman Rank Correlation $\rho(f_e, \Delta\text{NLL}_e)$**: $\mathbf{-0.038 \quad (\approx 0.0)}$

---

## 4. Key Law Discovered & Transition to v4
* **The Gating Disconnect Law**: **Routing frequency is completely uncorrelated with causal importance.** High-frequency experts act as generic traffic hubs, while true reasoning logic resides in low-frequency, high-precision experts.
* **Transition Logic**: Now that we had isolated the exact true causal reasoning bank `[(18, 43), (20, 219), (21, 183), (36, 229)]`, we moved to [[01_Generations/v04_Causal_FineTuning_Reversal|Generation v4]] to perform our first surgical fine-tuning experiment.
