# EXPERIMENTAL TIMELINE

This document provides a comprehensive, rigorous autopsy of every experimental iteration conducted across the Laguna research program.

---

### Exp 01: Kaggle T4 × 2 Initial MoE Atlas
- **Notebook**: `laguna_xs2_expert_atlas_kaggle_t4x2.ipynb`
- **Question**: Can we load Laguna XS.2 in Kaggle 2×16GB environment and run a complete routing sweep across all 9,984 experts?
- **Setup & Hardware**: Kaggle 2× NVIDIA T4 GPUs (16GB each = 32GB total VRAM).
- **Method**: Tensor-parallel / pipeline sharding of custom Laguna architecture via Hugging Face Transformers.
- **Result & Validity**: FAILED due to CUDA OOM and disk space exhaustion when downloading multi-shard FP8/BF16 checkpoints.
- **Scientific & Engineering Meaning**: Sharded execution of a 33.4B MoE on 32GB consumer GPUs introduces high serialization latency and fragile hook injection surfaces.

---

### Exp 02: T4 × 2 vLLM Inference & Causal Exploration (v1 – v4)
- **Notebooks**: `laguna_xs2_causal_vllm_t4x2.ipynb`, `v2.ipynb`, `v3.ipynb`, `v4.ipynb`
- **Question**: Can vLLM enable fast batched causal sweeps and fixed-routing ablations on T4 GPUs?
- **Method**: Custom vLLM engine integration with custom routing modification kernels.
- **Result & Validity**: FAILED due to vLLM FP8 KV-cache incompatibility with custom sliding-window/GQA heads and vLLM API version mismatches.
- **Scientific & Engineering Meaning**: vLLM's optimized fused kernels bypass intermediate activation hooks required for causal intervention; direct PyTorch execution is required for surgical research.

---

### Exp 03: L40S Causal Atlas & Loader Fix (v6 – v9)
- **Notebooks**: `laguna_xs2_causal_atlas_L40_48GB_...v6..v9.ipynb`
- **Question**: Can a single 48GB GPU (NVIDIA L40/L40S) run full causal sweeps with custom PyTorch forward hooks?
- **Method**: Direct Transformers loading with custom code hook registration.
- **Result & Validity**: Revealed the `DecompressExperts` problem: custom FP8 unpacking on the fly interfered with hook registration and backpropagation.

---

### Exp 04: RTX PRO 6000 96GB Direct BF16 Causal Surgery
- **Notebook**: `laguna_xs2_final_causal_surgery_RTX_PRO_6000_96GB_32vCPU_256GB (1).ipynb`
- **Question**: Can uncompressed BF16 weights fit on 96GB VRAM and execute rapid causal sweeps?
- **Setup & Hardware**: AWS `g7e.2xlarge` (1× NVIDIA RTX PRO 6000 Ada/Blackwell 96GB VRAM, 8 vCPUs, 62.3 GiB RAM).
- **Result & Validity**: SUCCESSFUL INFRASTRUCTURE. Resident BF16 model occupied 62.29 GiB VRAM with 29.6 GiB free headroom. Full 39-layer causal sweep executed in 8.0 seconds.

---

### Exp 05: Scoring Fix & First Valid Causal Map (Discovery of E229)
- **Notebook**: `laguna_xs2_causal_surgery_g7e_2xlarge_v2_scoringfix (2).ipynb`
- **Question**: After fixing the `</think>\n` teacher-forcing alignment and validating selective vs full-logit cross-entropy, where does frontend CSS capability causally reside?
- **Method**: Hierarchical search (39-layer routed branch ablation -> individual expert sweep in top layers) under fixed routing.
- **Result**:
  - Layer 36 routed branch ablation: $\Delta\text{NLL}_{\text{target}} = +1.3797$.
  - Layer 36, Expert 229 (E229) alone: $\Delta\text{NLL}_{\text{target}} = +1.2858$, $\Delta\text{NLL}_{\text{control}} = -0.0318$ (Specificity = $+1.2858$).
  - Bootstrap 95% CI strictly positive ($P > 0.99$).
  - Routing Rank: #18 (only ~2.1% token routing). Causal Rank: #1.
- **Interpretation**: Strong evidence that causal necessity is decoupled from routing frequency.

---

### Exp 06: Matched Baselines & The Critical Reversal
- **Notebook**: `laguna_xs2_causal_surgery_g7e_2xlarge_v3_matched_baselines (3).ipynb`
- **Question**: Does fine-tuning the causally necessary expert (E229) yield superior adaptation compared to top-routed or random experts?
- **Setup**: Matched parameter budget (1 expert = ~3.15M params), identical SFT training on frontend capability data.
- **Result**:
  - *Most-Routed Expert (L38/E60)*: $+0.0770$ to $+0.1100$ adaptation NLL gain.
  - *Causal Expert (L36/E229)*: $+0.0280$ adaptation NLL gain.
  - *Random Same-Layer (L36/E45)*: $+0.0171$ gain.
  - *Random Global (L12/E110)*: $+0.0031$ gain.
- **Decisive Conclusion**: Causal Expert lost to Most-Routed Expert. **Causal Necessity $\neq$ Adaptation Plasticity**.

---

### Exp 07: Clean End-to-End Surgery (v5)
- **Notebook**: `laguna_xs2_causal_surgery_g7e_2xlarge_v5_clean_end_to_end (2).ipynb`
- **Question**: Replicate the causal vs routing adaptation reversal with clean preflight checks, native BF16 weight merging, and frozen base evaluations.
- **Result**: Confirmed that causal experts are rigid read-heavy memory structures that do not adapt well when fine-tuned in isolation.

---

### Exp 08: Falsification-Grade Search & Forced Access (v6)
- **Notebook**: `laguna_xs2_v6_falsification_grade (1).ipynb`
- **Question**: Did E229 fail to adapt because the router starved it of training tokens (Routing Blocker Hypothesis)?
- **Method**: Forced-access training: forcing router routing probabilities to direct tokens to E229 during training.
- **Result**: Falsified the Routing Blocker Hypothesis. Even when given 100% routing access during training, E229 failed to show high adaptation plasticity.

---

### Exp 09: Global Writeability Atlas (v7 & v7.1)
- **Notebooks**: `laguna_xs2_v7_1_global_writeability_atlas_fixed (3).ipynb`
- **Question**: What parameter property predicts adaptation plasticity across all 9,984 experts?
- **Method**: Evaluated (1) Causal Specificity, (2) Routing Frequency, (3) Target Gradient Norm $\|\nabla_\theta \mathcal{L}\|$, and (4) SFT Plasticity Gain across all layers.
- **Result**: Gradient Norm exhibited a high population-level correlation ($R \approx 0.82$) with adaptation plasticity. Causal necessity showed weak/negative correlation.

---

### Exp 10: Cross-Capability Replication (v8 & v8.2)
- **Notebooks**: `laguna_xs2_v8_2_1b_cross_capability_full_checkpoint_fixed (1).ipynb`, `laguna_v8_cross_capability_experiment.csv`
- **Question**: Does gradient-based writeability generalize across diverse domains (Frontend, Python Systems, Math)?
- **Setup**: Evaluated parameter budget curves ($K=1, 2, 4, 8$ writable experts vs controls) across 3 capabilities.
- **Result**: Population correlation held across domains, but top-K selector reliability was noisy. Top-1 gradient expert was not consistently the top-1 adapter.

---

### Exp 11: Real-Benchmark Matched PEFT on GSM8K (v9)
- **Notebook**: `laguna_xs2_v9_matched_peft_gsm8k (1).ipynb`
- **Question**: How do writable experts compare to Attention LoRA on real multi-step reasoning benchmarks (GSM8K) under matched parameter budgets (~12.6M params)?
- **Result**:
  - *Teacher-Forced NLL*: Writable experts improved NLL dramatically ($\Delta\text{NLL} = -0.35$).
  - *Free-Running Autonomous Accuracy*: Accuracy dropped or stagnated (76.0% $\to$ 74.5% / 76.0%).
- **Critical Discovery**: **NLL Improvement $\neq$ Free-Running Reasoning Accuracy**. Teacher-forcing on gold prefixes suffers from severe Exposure Bias / Distribution Shift.

---

### Exp 12: Behavior-Aware Dose Selection & Contrastive LoRA (v10)
- **Notebook**: `laguna_xs2_v10_behavior_aligned_writeability (1).ipynb`
- **Question**: Does behavior-based dose calibration on free-running accuracy and contrastive gradient projection rescue reasoning performance?
- **Method**: Checkpoint evaluation grid on free-running GSM8K; attempted contrastive selector: $\|\nabla_\theta \mathcal{L}_{\text{GSM8K}}\| - \lambda \|\nabla_\theta \mathcal{L}_{\text{MBPP}}\|$.
- **Result**: Apparent $+0.047$ accuracy gain on calibration set.
- **Methodological Breakdown**: Contrastive selector selected identical experts as raw gradient because subtracting one MBPP vector in 12M dimensions reduced norm by $<0.5\%$. Small validation set ($N=48$) led to Winner's Curse.

---

### Exp 13: Fresh Confirmation & Random-Placement Distribution (v11)
- **Notebook**: `laguna_xs2_v11_fresh_confirmation_random_placement.ipynb`
- **Question**: Does Guided LoRA / Expert placement statistically outperform an architecture-matched distribution of random LoRA placements on fresh, untouched test data?
- **Setup**: Frozen experimental policy, untouched fresh GSM8K ($N=384$), 5 random training seeds, 6 independent random LoRA placements matching module layout.
- **Result**: Guided LoRA failed to statistically beat random LoRA. Paired 2-way bootstrap 95% CI crossed zero: Mean gain $+0.0174$, 95% CI $[-0.0226, +0.0590]$.
- **Decisive Conclusion**: The positive result from v10 did not reliably replicate.

---

### Exp 14: Gemma 2 2B IT Dense Repair Subspace Protocol
- **Model**: `google/gemma-2-2b-it` (Dense 2B instruction-tuned model).
- **Question**: Does failure-conditioned, preservation-projected subspace repair succeed in a dense model where MoE routing artifacts and AdamW optimizer dynamics are completely eliminated?
- **Method**: Self-generated failure trajectories ($y^-$) vs gold corrections ($y^+$); preservation gradients on self-generated correct solutions ($y^{\text{corr}}$); null-space Fisher projection; rank-8 low-rank algebraic update; functional KL trust region; rescue vs damage metric.
