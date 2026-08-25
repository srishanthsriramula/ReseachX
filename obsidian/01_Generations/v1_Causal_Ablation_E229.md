---
tags: [generation, v1, causal-ablation, expert-surgery]
version: v1
status: completed
backlinks: "[[00_Index/00_Index_MOC|Index]], [[01_Generations/v2_Multi_Expert_Combinatorial_Screening|v2]]"
---

# 🧬 Generation v1: Causal Expert Zero-Ablation & Discovery of E229

## 1. Motivation & Question
Can we isolate sparse reasoning sub-circuits within the 256 routed experts of [[03_Architectures/Laguna_XS2_Architecture_Profile|Laguna XS.2]] without modifying the full 33.4B parameter model?

## 2. Experimental Protocol
* **Zero-Ablation**: Iterated over all $48 \times 256 = 12,288$ routed expert banks.
* Set $W_e \leftarrow 0$ and evaluated $\Delta\text{NLL}$ on GSM8K vs. C4/MBPP.

## 3. Empirical Results
* $>99\%$ of experts had $\Delta\text{NLL} < 0.01$.
* **Major Discovery**: **Layer 36, Expert 229 (L36/E229)** produced $\mathbf{\Delta\text{NLL} = +1.2858}$ on math reasoning, while causing only $\Delta\text{NLL} = +0.021$ on general text.

## 4. Key Takeaway & Transition
Proved extreme causal concentration in deep layers. Transitioned to [[01_Generations/v2_Multi_Expert_Combinatorial_Screening|v2]] to test multi-expert combinatorial interactions.
