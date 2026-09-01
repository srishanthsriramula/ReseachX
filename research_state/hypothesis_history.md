# Chronological Hypothesis History (v1 → v24)

1. **H1 (Causal Localization)**: Math capabilities reside in a sparse sub-network of MoE experts. (Confirmed: E229).
2. **H3 (Direct Expert Tuning)**: Fine-tuning E229 repairs math reasoning. (Falsified: Router bypass).
3. **H5 (Joint Router Tuning)**: Tuning gates and experts simultaneously stabilizes routing. (Falsified: Routing avalanche).
4. **H6 (Attention Channel Stability)**: Attention linear layers provide continuous, non-bifurcating adaptation channels. (Confirmed).
5. **H11 (Stratified Hierarchy)**: Targeting middle-depth representation layers $[1, 26]$ dominates uniform layer allocations. (Confirmed, $p < 10^{-4}$).
6. **H15 (Theorem 7 Activation Whitening)**: Constraining updates to the null-space of activation covariance $C_{\text{code}}$ eliminates forgetting. (Falsified in v23: $C$ ignores downstream gradients).
7. **H22 (GRPO Policy Optimization)**: Reinforcement learning improves reasoning without forgetting. (Falsified: Reward collapse at step 13).
8. **H24 (Theorem 11 Fisher Gradient Subspace)**: Fisher gradient covariance $G = \mathbb{E}[\|\partial L/\partial y\|^2 xx^T]$ captures true loss sensitivity and eliminates interference. (Active in v24).
