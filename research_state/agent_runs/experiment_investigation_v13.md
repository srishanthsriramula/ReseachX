# 🧪 Specialist Investigation Report: Experiment & Protocol
**Agent**: `experiment`  
**Timestamp**: 2026-08-25T19:06:55.663362+00:00  
**Investigation**: Protocol Design, Hyperparameter Specification, and Checkpoint Manifest for Generation v13.1

---

## 1. Complete Hyperparameter Ledger (v13.1 Production)

| Hyperparameter | Value | Rationale & Derivation |
|---|---|---|
| **Base Model** | `poolside/Laguna-XS.2` (33.4B-A3B) | Frozen BF16 Checkpoint (14 shards, 66.8 GB) |
| **Target Task** | GSM8K Multi-Step Math ($N=384$ fresh) | Fresh test split disjoint from all prior generations |
| **Retained Control Task** | MBPP Python Code ($N=160$ items) | Pinned reference standard for control drift |
| **Stratified Layers** | `[1, 2, 8, 11, 12, 16, 21, 26]` | Theorem 2 Bound: $\kappa(J) = \mathcal{O}(K)$ |
| **Target Modules** | `q_proj, k_proj, v_proj, o_proj` | GQA Attention Circuit (4 projections per layer) |
| **LoRA Ranks Evaluated** | $r \in \{63, 128, 256\}$ | High-capacity scaling frontier |
| **Update Horizon ($T$)** | **$8$ Updates strictly locked** | Prevents cumulative dose leakage |
| **Batch Size ($B$)** | **$16$** ($128$ examples per seed) | Standardized calibrated batch dose |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.95, \text{wd}=0.01$) | Standardized second-moment tracking |
| **LR for $r=63$** | **$1.0 \times 10^{-5}$** | Baseline calibrated learning rate |
| **LR for $r=128$** | **$7.0 \times 10^{-6}$** | $\mu\text{P}$ Scaled: $1.0\text{e-5} \times \sqrt{63/128}$ |
| **LR for $r=256$** | **$5.0 \times 10^{-6}$** | $\mu\text{P}$ Scaled: $1.0\text{e-5} \times \sqrt{63/256}$ |
| **Early Damping (L1-2)** | **$\alpha_{\text{early}} = 0.05$** | Strong protection for syntax and token foundation |
| **Mid Damping (L8-12)** | **$\alpha_{\text{mid}} = 0.01$** | Balanced relational anchor invariance |
| **Deep Damping (L16-26)**| **$\alpha_{\text{deep}} = 0.002$** | Maximum reasoning torque in deep layers |
