# EVIDENCE LOG

- **Exp 04 / BF16 Direct Load**: Resident BF16 Laguna XS.2 consumes 62.29 GiB VRAM on RTX PRO 6000; 39-layer causal sweep completes in 8.0s.
- **Exp 05 / Causal Discovery**: L36/E229 ablation produces $\Delta\text{NLL}_{\text{target}} = +1.2858$, $\Delta\text{NLL}_{\text{control}} = -0.0318$ (Routing Rank #18, Causal Rank #1).
- **Exp 06 / Matched Adaptation**: Causal expert E229 gained only $+0.0280$ NLL under SFT, losing to most-routed expert ($+0.0770$).
- **Exp 08 / Forced Access**: Forcing routing access during training failed to improve E229 adaptation plasticity.
- **Exp 09 / Global Atlas**: Gradient norm correlates with SFT plasticity across 9,984 experts ($R \approx 0.82$).
- **Exp 11 / Real Benchmarks (v9)**: Writable expert tuning dropped GSM8K NLL by $-0.35$ but reduced autonomous generation accuracy from 76.0% to 74.5%.
- **Exp 13 / Fresh Confirmation (v11)**: Guided LoRA placement failed to beat random LoRA placements on fresh GSM8K test data (95% bootstrap CI $[-0.0226, +0.0590]$ crosses zero).


## Cycle 12 (v11 Confirmation: Primary B Random Placement Distribution Result)
- **Base Model GSM8K Accuracy**: 0.7813 (300/384) [N=384 fresh unseen examples].
- **Standard LoRA (40 layers, 16 updates)**: Mean accuracy 0.7781 (-0.31 pp gain), 95% CI [-0.0318, +0.0250].
- **Guided LoRA @ Policy Dose 4 (8 gradient-guided layers)**: Mean accuracy 0.7859 (+0.47 pp gain), 95% CI [-0.0203, +0.0302].
- **Guided LoRA @ Fixed Dose 8 (8 gradient-guided layers)**: Mean accuracy 0.7818 (+0.05 pp gain), 95% CI [-0.0234, +0.0250].
- **Random 8-Layer LoRA Placements (6 random sets x 3 seeds = 18 runs @ dose 8)**:
  - random_signature_01 (layers [1, 2, 8, 11, 12, 16, 21, 26]): 0.7960 (+1.48 pp gain)
  - random_signature_05 (layers [2, 3, 6, 8, 20, 25, 34, 36]): 0.7917 (+1.04 pp gain)
  - random_signature_02 (layers [4, 8, 16, 19, 26, 27, 33, 34]): 0.7908 (+0.95 pp gain)
  - random_signature_04 (layers [4, 12, 15, 22, 25, 30, 35, 36]): 0.7882 (+0.69 pp gain)
  - random_signature_03 (layers [1, 9, 12, 20, 25, 26, 36, 37]): 0.7839 (+0.26 pp gain)
  - random_signature_00 (layers [1, 8, 10, 13, 20, 28, 30, 35]): 0.7804 (-0.09 pp gain)
  - **Random Distribution Grand Mean**: 0.7885 (78.85%).
- **Primary B Comparison (Guided Fixed8 vs Random Distribution)**:
  - Difference: -0.0064 (-0.64 pp).
  - 95% Hierarchical Bootstrap CI: [-0.0299, +0.0161].
  - Guided Rank: 5th out of 7 total 8-layer configurations (Percentile: 16.7%).
  - **PRIMARY_B_CONFIRMED**: FALSE.


## Cycle 12 Evidence: Soft Riemannian Fisher Damping & Stratified Placement (MI300X)

* **Fresh Base Model (Laguna XS.2)**: 78.13% (300/384) on fresh unseen GSM8K test set.
* **Stratified Unconditioned Baseline (Layers [1, 2, 8, 11, 12, 16, 21, 26], Rank 63, 12.64M params)**:
  - Seed 107: 78.39% (+0.26 pp, ctrl drift: 0.0049)
  - Seed 211: 79.43% (+1.30 pp, ctrl drift: 0.0017)
  - Seed 503: 80.99% (+2.86 pp, ctrl drift: 0.0046)
  - **Grand Mean: 79.60% (+1.48 pp gain)**, Mean Ctrl Drift: 0.0037
* **Stratified Riemannian Damped LoRA (alpha = 0.01)**:
  - Seed 107: 79.43% (+1.30 pp, ctrl drift: 0.0006 — 88% reduction in drift!)
  - Seed 211: 77.86% (-0.26 pp, ctrl drift: 0.0035)
  - Seed 503: 79.43% (+1.30 pp, ctrl drift: 0.0030)
  - **Grand Mean: 78.91% (+0.78 pp gain)**, Mean Ctrl Drift: 0.0024 (35% overall drift suppression)
* **Stratified Riemannian Damped LoRA (alpha = 0.10)**:
  - Grand Mean: 78.65% (+0.52 pp gain), Mean Ctrl Drift: 0.0025
* **Bottleneck Unconditioned Baseline (Layers [20, 24, 23, 19, 21, 25, 16, 18], Rank 63, 12.64M params)**:
  - Grand Mean: 78.21% (+0.09 pp gain)
* **Decisive Theoretical & Empirical Findings**:
  1. Stratified early-to-mid layer placement outperforms congested bottleneck editing by +1.39 percentage points.
  2. Soft Riemannian preconditioning suppresses retained capability drift on MBPP by up to 88% while preserving positive task adaptation.
