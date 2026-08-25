---
tags: [generation, v5, matched-bakeoff, router-avalanche, falsification]
version: v5
status: falsified
backlinks: "[[00_Index/00_Index_MOC|Index]], [[02_Theorems/Theorem_1_Discontinuous_MoE_Routing_Bifurcation|Theorem 1]]"
---

# 🧬 Generation v5: Matched Bakeoff & The Router Avalanche

## 1. Motivation & Question
Does *any* routed expert selection policy (Causal, Gradient, Routing Frequency, Random) produce positive generalization gains on GSM8K?

## 2. Experimental Results
* Tested 4 matched policies ($K=4$, ~12.58M params) across seeds 11, 23, 47 on MI300X:
  * **Causal Experts (E229 bank)**: **$-2.39\text{ pp}$** degradation.
  * **Gradient Experts**: **$-1.82\text{ pp}$** degradation.
  * **Routing Experts**: **$-3.12\text{ pp}$** degradation.
  * **Random Experts**: **$-2.60\text{ pp}$** degradation.

## 3. Theoretical Proof: The Router Avalanche
Proved [[02_Theorems/Theorem_1_Discontinuous_MoE_Routing_Bifurcation|Theorem 1 (Discontinuous Router Bifurcation)]]. Modifying routed weights shifts downstream activations across softmax routing boundaries, triggering a discrete permutation avalanche across all 48 layers.

> [!WARNING]
> Parameter surgery on routed MoE experts was **permanently falsified and abandoned**.
