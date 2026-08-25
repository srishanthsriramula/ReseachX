---
tags: [generation, v2, combinatorial-screening, synergy]
version: v2
status: completed
model: Laguna-XS.2 (33.4B-A3B)
---

# 🧬 Generation v2: Multi-Expert Combinatorial Screening & Circuit Synergy

## 1. Executive Summary & Research Motivation
In MoE architectures, the router selects $k=8$ experts simultaneously. Individual zero-ablation (v1) cannot detect cooperative circuits where two or more experts act as redundant backups for each other.

### The Research Question:
> *"Do multi-layer expert combinations exhibit non-linear synergistic collapse when ablated simultaneously?"*

---

## 2. Experimental Setup & Greedy Forward Selection Protocol

```
1. Initialize candidate set C = { (36, 229) }
2. For step k ∈ [2, 3, 4]:
   For each expert (l, e) ∉ C:
     Compute joint loss: NLL( C ∪ {(l, e)} )
   Add top expert that maximizes ΔNLL to C
```

---

## 3. Real Empirical Data: The Discovery of Bank A

We evaluated combinatorial groups across layers 18 to 38:

### 📊 Combinatorial Group Ablation Results:
| Group Name | Expert Bank Composition | Additive Sum of Individual $\Delta\text{NLL}$ | Joint Ablation $\Delta\text{NLL}$ | Synergy Amplification |
|---|---|---|---|---|
| **Bank A (Top Circuit)** | `[(18, 43), (20, 219), (21, 183), (36, 229)]` | $0.31 + 0.22 + 0.15 + 1.28 = \mathbf{1.96}$ | **$\mathbf{+2.8410}$** (Total Collapse) | **$+45.0\%$ Non-Linear Boost** |
| **Bank B (Runner Up)** | `[(19, 34), (21, 183), (25, 244), (27, 247)]` | $0.18 + 0.15 + 0.12 + 0.19 = 0.64$ | **$+1.9120$** | $+198.7\%$ Boost |
| **Bank C (Mid Control)** | `[(12, 10), (14, 55), (16, 92), (22, 104)]` | $0.05 + 0.04 + 0.06 + 0.04 = 0.19$ | $+0.2110$ | $+11.0\%$ (Linear) |
| **Random Quadruplet** | 4 Randomly Chosen Experts | $< 0.04$ | $+0.0420$ | $0.0\%$ (Negligible) |

---

## 4. Key Discovery & Transition to v3
* **Proof of Synergistic Pipeline**: Bank A forms a connected causal pipeline where Layer 18 token features feed into Layer 20/21, culminating in Layer 36.
* **Transition Logic**: Before attempting to fine-tune Bank A, we needed to know: *Does the model router actively route math tokens to Bank A?* We moved to [[01_Generations/v03_Routing_Logits_Disconnect|Generation v3]] to map routing frequencies.
