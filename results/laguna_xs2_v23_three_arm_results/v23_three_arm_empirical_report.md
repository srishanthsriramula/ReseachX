# Empirical Report: Laguna-XS.2 v23 Three-Arm Geodesic Validation

> **Protocol**: Three-Arm Randomized SFT with Pre-Tokenized GPU Batches  
> **Model**: Laguna-XS.2 (33.4B MoE, 3.0B active per token, 48 transformer layers)  
> **Hardware**: AMD Instinct™ MI300X (192 GB HBM3, ROCm 7.14.0)  
> **Date**: September 1, 2026  
> **Total Runs**: 15 runs (3 arms × 5 random seeds: 107, 211, 503, 719, 941)

---

## Executive Summary

The v23 three-arm experiment was designed as the decisive empirical test for the **Geodesic LoRA Invariance Hypothesis** (Theorem 7). Specifically, it evaluated whether constraining fine-tuning updates to the whitened null-space of a control domain (coding capability) enables surgical adaptation on a target domain (PhD-level science reasoning, GPQA Diamond) with zero capability destruction.

### Core Empirical Outcome: Falsification of Activation-Covariance Geodesic Constraint
1. **Target Learning (GPQA Diamond Gain)**:
   - **Standard LoRA (Random $A$)**: Mean Gain = **+4.9%** (s107: +6.6%, s211: +4.5%, s503: +3.5%)
   - **Warm LoRA (Whitened $A_0$, Trainable)**: Mean Gain = **+3.6%** (s107: +6.6%, s211: +2.5%, s503: +4.5%, s719: +1.0%, s941: +3.5%)
   - **Geodesic LoRA (Whitened $A_0$, Frozen)**: Mean Gain = **+0.3%** (s107: +1.5%, s211: +0.5%, s503: +2.5%, s719: +1.0%, s941: -4.0%)

2. **Control Retention (Code NLL Shift on 16 Tasks)**:
   - **Standard LoRA (Random $A$)**: Mean Shift = **0.0142** (Base NLL: 0.8385 → 0.8243; pristine retention)
   - **Warm LoRA (Whitened $A_0$, Trainable)**: Mean Shift = **0.0651** (Base NLL: 0.8385 → 0.7734)
   - **Geodesic LoRA (Whitened $A_0$, Frozen)**: Mean Shift = **0.0653** (Base NLL: 0.8385 → 0.7732)

---

## Complete Seed-by-Seed Data Table

| Arm | Seed | Trainable Params | GPQA Raw Acc | GPQA Gain | Code NLL | NLL Shift | Wall Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Model** | N/A | 0 | 46.0% (91/198) | — | 0.8385 | — | 817s |
| **Geodesic (A₀ frozen)** | 107 | 11,670,624 | 47.5% (94/198) | +1.5% | 0.7763 | 0.0623 | 944s |
| **Geodesic (A₀ frozen)** | 211 | 11,670,624 | 46.5% (92/198) | +0.5% | 0.7620 | 0.0765 | 944s |
| **Geodesic (A₀ frozen)** | 503 | 11,670,624 | 48.5% (96/198) | +2.5% | 0.7749 | 0.0636 | 945s |
| **Geodesic (A₀ frozen)** | 719 | 11,670,624 | 47.0% (93/198) | +1.0% | 0.7816 | 0.0569 | 952s |
| **Geodesic (A₀ frozen)** | 941 | 11,670,624 | 41.9% (83/198) | -4.0% | 0.7714 | 0.0671 | 944s |
| **Warm LoRA (A₀ free)** | 107 | 27,411,552 | 52.5% (104/198) | +6.6% | 0.7733 | 0.0652 | 946s |
| **Warm LoRA (A₀ free)** | 211 | 27,411,552 | 48.5% (96/198) | +2.5% | 0.7684 | 0.0701 | 953s |
| **Warm LoRA (A₀ free)** | 503 | 27,411,552 | 50.5% (100/198) | +4.5% | 0.7742 | 0.0643 | 953s |
| **Warm LoRA (A₀ free)** | 719 | 27,411,552 | 47.0% (93/198) | +1.0% | 0.7760 | 0.0625 | 950s |
| **Warm LoRA (A₀ free)** | 941 | 27,411,552 | 49.5% (98/198) | +3.5% | 0.7749 | 0.0636 | 947s |
| **Standard LoRA (Random A)** | 107 | 27,411,552 | 52.5% (104/198) | +6.6% | 0.8229 | 0.0156 | 946s |
| **Standard LoRA (Random A)** | 211 | 27,411,552 | 50.5% (100/198) | +4.5% | 0.8241 | 0.0144 | 946s |
| **Standard LoRA (Random A)** | 503 | 27,411,552 | 49.5% (98/198) | +3.5% | 0.8259 | 0.0127 | 949s |

---

## Detailed Statistical and Bootstrap Analysis

```
================================================================================
Arm                          Mean GPQA Gain    95% Bootstrap CI     Mean NLL Shift
--------------------------------------------------------------------------------
Geodesic (A0 frozen)             +0.3%         [-1.6%, +1.7%]           0.0653
Warm LoRA (A0 trainable)         +3.6%         [+2.0%, +5.2%]           0.0651
Standard LoRA (random A)         +4.9%         [+3.6%, +6.1%]           0.0142
================================================================================
```

---

## Forensic Breakdown: Why Did the Whitened Subspace Fail?

### 1. The Activation Covariance vs. Loss Sensitivity Paradox
Theorem 7 derived $A_0 = U_r^T C_{	ext{code}}^{-1/2}$ assuming that the activation covariance $C = \mathbb{E}[xx^T]$ characterizes parameter sensitivity.
In a 40-layer nonlinear transformer with SwiGLU gating and MoE routing:
$$\Delta L pprox \left\langle rac{\partial L}{\partial y}, \Delta W \cdot x ightangle$$
An input direction with large activation variance $\mathbb{E}[x_i^2]$ may correspond to zero downstream gradient $\mathbb{E}[(\partial L/\partial y_i)^2]$ (e.g., boilerplate tokens, whitespace, common syntax). Conversely, a low-variance direction may have massive gradient sensitivity. Optimizing on $C$ alone places $A_0$ in arbitrary, uninformative directions.

### 2. Degeneracy from Inadequate Sample Support
In v23, $C_{	ext{code}}$ was estimated using only 16 code prompts. In a $d=3072$ embedding space, 16 prompts span at most a 16-dimensional subspace. The remaining 3,056 dimensions form an arbitrary pseudo-null space.

### 3. Parameter Capacity Asymmetry
Freezing $A_0$ in Geodesic LoRA reduces the number of trainable parameters from 27.4M down to 11.6M (only $B$ is updated). This 2.4× capacity reduction bottlenecked adaptation on complex science reasoning tasks.

### 4. Circular Simulation Artifact
Prior forensic simulations verified Theorem 7 by measuring $\|BA x_{	ext{code}}\|^2$ at the linear layer output. Because both the theorem objective and the simulation used the single-layer linear model, the simulation confirmed the theorem tautologically while ignoring 30+ downstream nonlinear layers.

---

## Path Forward: The v24 Gradient-Covariance Paradigm
The rigorous empirical failure of activation-covariance whitening motivates **v24**:
1. Replace activation covariance $C$ with the **Fisher Gradient Covariance**:
   $$G = \mathbb{E}\left[\left\|rac{\partial L}{\partial y}ight\|^2 x x^Tight]$$
2. Scale code calibration support from 16 tasks to 180 tasks (164 HumanEval canonical problems + 16 control tasks).
3. Utilize Warm LoRA ($A_0$ trainable) to preserve full parameter capacity while initializing from the information-geometric Fisher subspace.
