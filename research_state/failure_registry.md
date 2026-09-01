# Comprehensive Failure Registry (16 Technical Bugs + 5 Theoretical Mistakes)

## Technical Implementation Bugs

| # | Bug Name | Version | Failure Mechanism | Root Cause | Discovery Method | Resolution |
| :-: | :--- | :---: | :--- | :--- | :--- | :--- |
| **1** | Safetensor Loading Error | v18–v19 | Model failed to initialize from remote weights | Missing tensor shards in local cache | Stack trace inspection | Implemented custom memory-mapped shard loader |
| **2** | Config AttributeError | v19 | Crash during model property access | Discrepancy between HuggingFace config and model definition | Runtime error | Wrapped config access in defensive attribute resolvers |
| **3** | Missing Chat Prefix Formatting | v20 | Model output degenerate text in evaluation | Missing `<user>` and `<assistant>` delimiters | Qualitative output inspection | Implemented `chat_prefix_text` wrapper |
| **4** | 4-Module Attention Omission | v20 | Unconstrained gradient drift in attention | Laguna-XS.2 has 5 attention projections (`g_proj` omitted) | Architecture code audit | Added auto-discovery for all 5 attention projections |
| **5** | Layer Target Index Mismatch | v20 | LoRA attached to non-existent layers | PEFT config indexing did not match Laguna layers | PEFT module inspection | Formatted layer indices as `layers_to_transform` |
| **6** | CUDA Out-of-Memory (Training) | v20, v23 | OutOfMemoryError during backward pass | Batch size 8 on long GPQA prompts (384 tokens) exceeded VRAM | CUDA allocation error | Reduced `TRAIN_BATCH=4` + enabled gradient checkpointing |
| **7** | Inactive Geodesic Constraint | v20 | Geodesic was actually Warm LoRA | `sA.weight.requires_grad = False` was omitted in helper function | Forensic code audit | Explicitly set `requires_grad = False` on $A$ |
| **8** | 2× Norm Amplification | v20.1 | Unfair effective LR advantage for whitened basis | $A_0 = U_r^T C_{\text{code}}^{-1/2}$ had $\|A_0\|_F \approx 72.4$ vs Kaiming $\|A\|_F \approx 36.2$ | Forensic simulation | Implemented exact Kaiming norm matching |
| **9** | Substring Regex Match Inflation | v20.1 | False positive accuracy inflation | Substring matching matched `"c"` in words like `"calculation"` | Prediction CSV inspection | Implemented strict boundary regex matching |
| **10** | 256-Token CoT Truncation | v22 | 8.59% apparent accuracy | CoT reasoning severed before reaching boxed answer | Output length inspection | Scaled `max_new_tokens` to 1024 |
| **11** | MMLU Blank Output Collapse | v23 | 25.0% random score on MMLU | Laguna-XS.2 is code-only; outputs blank text on non-code tasks | Prompt test inspection | Replaced MMLU with Code NLL and HumanEval |
| **12** | `c_he` NameError in Cell 9 | v23 | Evaluation loop crashed after run 1 | Cell 9 referenced deleted HumanEval variable | Runtime traceback | Removed legacy variable references; unified return values |
| **13** | HumanEval Subprocess Block | v23 | Sandbox execution failed on remote VM | Container environment restricted subprocess spawning | Execution test | Implemented safe in-process fallback with timeout handling |
| **14** | 128-Step Science Overfitting | v23 | GPQA accuracy dropped 13% (46% → 33%) | 128 steps on 250 GPQA Main questions caused verbatim memorization | Loss curve inspection | Reduced training horizon to 32 steps |
| **15** | SciQ Difficulty Mismatch | v23 | Zero transfer to GPQA Diamond | SciQ is high-school level; GPQA Diamond is PhD-level | Evaluation analysis | Combined SciQ (reasoning structure) with GPQA Main (domain depth) |
| **16** | "Think Step-by-Step" Degradation | v23 | Base accuracy dropped from 53.5% to 46.0% | Model was trained on LaTeX `\boxed{}` formatting | A/B prompt comparison | Reverted prompt instruction to `\boxed{}` format |

---

## Theoretical Conceptual Mistakes

| # | Theoretical Mistake | Manifestation in Practice | Why It Failed Mathematically | Resolution |
| :-: | :--- | :--- | :--- | :--- |
| **1** | Activation Covariance Equated with Loss Sensitivity | Whitened basis caused 4.6× MORE code forgetting than random LoRA | $C = \mathbb{E}[xx^T]$ measures input variance, ignoring downstream gradient sensitivity $\partial L/\partial y$ | Replaced $C$ with Fisher Gradient Covariance $G = \mathbb{E}[\|\partial L/\partial y\|^2 xx^T]$ (v24) |
| **2** | Degenerate Calibration Sample Support | Rank-16 pseudo-null space in $d=3072$ dimensions | 16 prompts cannot span a 3072-dimensional manifold | Scaled calibration support to 180 code tasks + 200 STEM tasks (v24) |
| **3** | Circular Simulation Validation | False confidence in Theorem 7 | Linear 1-layer simulation evaluated the exact objective optimized by Theorem 7, ignoring 30 downstream nonlinear layers | Evaluated full-network loss and generation accuracy |
| **4** | Norm-Matching Scaling Artifact | Null-space directions amplified into active space | Scaling $A_0$ up to match Kaiming norm amplified small residual projections in the null space | Integrated Fisher weighting to naturally calibrate singular values |
| **5** | Confounded Capacity Asymmetry | Geodesic had 2.4× fewer parameters than Standard LoRA | Freezing $A$ reduced trainable parameters from 27.4M to 11.6M | Introduced Warm LoRA ($A_0$ initialized from Fisher subspace, trainable) |
