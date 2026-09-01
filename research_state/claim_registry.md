# Scientific Claim Registry

| Claim ID | Claim Statement | Status | Supporting Evidence |
| :---: | :--- | :---: | :--- |
| **C1** | Causal capabilities in Laguna-XS.2 localize in specific experts | **Verified** | L36/E229 zero-ablation causes $\Delta\text{NLL} = +1.2858$ on math reasoning |
| **C2** | Directly fine-tuning routed experts triggers routing bifurcations | **Verified** | Theorem 1 (Discontinuous Routing Bifurcation Law) |
| **C3** | Stratified attention LoRA Pareto-dominates uniform LoRA | **Verified** | 42-run trial ($p < 10^{-4}$) |
| **C4** | Activation-covariance whitening prevents control domain forgetting | **Falsified** | v23 15-run trial: whitened basis had 0.065 NLL shift vs 0.014 for random LoRA |
| **C5** | Freezing $A_0$ in LoRA constrains multi-step reasoning capacity | **Verified** | v23 trial: Geodesic (+0.3%) was outperformed by Warm LoRA (+3.6%) |
| **C6** | Fisher gradient covariance captures true downstream loss sensitivity | **Under Test (v24)** | Formulated in Theorem 11; pending v24 execution |
