# FAILURE REGISTRY & POST-MORTEM

This registry documents every technical blocker, methodological mistake, and scientific failure in the research program.

---

### Failure 1: The `</think>\n` Formatting & Aligned Scoring Bug
- **Type**: Methodological / Implementation Bug.
- **Root Cause**: Early teacher-forcing evaluation scripts omitted the trailing newline character after the `</think>` token and used selective-logit cross-entropy that diverged from full-logit calculations.
- **Scientific Impact**: Produced distorted causal rankings (e.g., Layer 20/22/27 artifacts).
- **Resolution**: Implemented exact token-level aligned full-logit scoring and verified cross-entropy against native PyTorch loss.

---

### Failure 2: The `DecompressExperts` Autograd Breakdown
- **Type**: Infrastructure / Model Architecture Incompatibility.
- **Root Cause**: Custom Laguna model code dynamically uncompressed FP8 weights on the fly inside the forward pass, severing the autograd backward graph.
- **Scientific Impact**: Blocked gradient backpropagation and hook-based causal intervention.
- **Resolution**: Migrated to native, uncompressed BF16 weights loaded directly into VRAM on RTX PRO 6000 96GB.

---

### Failure 3: vLLM Custom Kernel & KV-Cache Incompatibility
- **Type**: Infrastructure Incompatibility.
- **Root Cause**: vLLM's optimized FP8 KV-cache and grouped-attention kernels failed on Laguna's custom 36-sliding / 12-global layer hybrid layout.
- **Scientific Impact**: Blocked high-throughput batch inference during causal search.
- **Resolution**: Abandoned vLLM in favor of direct PyTorch / Transformers execution on large-memory GPUs.

---

### Failure 4: Causal Expert Adaptation Failure (Exp 06)
- **Type**: Scientific Falsification.
- **Root Cause**: Assuming causal necessity equals adaptation plasticity (Read-Write Equivalence). Essential experts store foundational representations; fine-tuning them causes brittle interference.
- **Scientific Impact**: Shattered the original causal surgery thesis and led to the distinction between causal necessity, routing access, gradient accessibility, and plasticity.

---

### Failure 5: Real-Benchmark Autonomous Reasoning Collapse (v9)
- **Type**: Methodological / Conceptual Mismatch.
- **Root Cause**: Optimizing teacher-forced cross-entropy on gold reference trajectories (Exposure Bias). Modifying high-gradient parameters destroyed multi-step reasoning capabilities during autonomous autoregressive generation.
- **Scientific Impact**: Proved that NLL is an invalid surrogate for reasoning accuracy.

---

### Failure 6: Contrastive Selector Norm Collapse (v10)
- **Type**: Mathematical / Methodological Flaw.
- **Root Cause**: Subtracting a single scalar MBPP control gradient vector from a target gradient in $\mathbb{R}^{12,000,000}$. The projection removed $<0.5\%$ of the vector norm, causing the contrastive score to collapse into raw gradient ranking.
- **Scientific Impact**: Exposed that single-vector contrastive heuristics are mathematically futile in high-dimensional parameter spaces.

---

### Failure 7: v11 Fresh Confirmation Replication Failure
- **Type**: Scientific Replication Failure.
- **Root Cause**: Overfitting to small calibration sets in v10 (Winner's Curse). Under frozen confirmatory protocols with 5 seeds and 6 random placement controls, Guided LoRA failed to beat random placement.
- **Scientific Impact**: Triggered the complete Root-Cause Reassessment and the reformulation of the research program into failure-conditioned subspace geometry.
