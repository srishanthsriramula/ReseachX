# Failed Approaches & Deprecated Concepts

1. **Direct MoE Causal Weight Ablation (v1–v4)**:
   - *Concept*: Zero-ablating or fine-tuning individual routed experts.
   - *Failure*: Triggered catastrophic routing avalanches and router disconnects.
2. **Reinforcement Learning via GRPO on Sparse MoE (v22)**:
   - *Concept*: Group Relative Policy Optimization with rule-based verifiers on single GPU.
   - *Failure*: Severe policy collapse after step 13 due to extreme gradient variance under small group sizes ($G=4$).
3. **Activation-Covariance Whitened Subspace Invariance (v15–v23)**:
   - *Concept*: Theorem 7 null-space initialization $A_0 = U_r^T C_{\text{code}}^{-1/2}$.
   - *Failure*: Proved empirically inferior to standard random LoRA (+0.3% vs +4.9% gain; 0.065 vs 0.014 forgetting) due to disconnect between activation variance and loss sensitivity.
4. **General-Domain MMLU as MoE Control Benchmark (v23)**:
   - *Concept*: Evaluating code models on general knowledge MMLU.
   - *Failure*: Model produced empty strings across non-STEM subjects because its pretraining was purely code-focused.
