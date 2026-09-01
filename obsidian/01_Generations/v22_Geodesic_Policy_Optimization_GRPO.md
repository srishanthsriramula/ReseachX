# Generation v22: Geodesic Policy Optimization (GRPO)

## 1. Objective
Attempted to transition from Supervised Fine-Tuning (SFT) to reinforcement learning using Group Relative Policy Optimization (GRPO) with rule-based mathematical/science verifiers and dense format rewards.

## 2. The Catastrophic Failure Mode
* Training showed initial reward improvement, but suffered catastrophic reward collapse after step 13.
* **Root Cause**: With single-GPU group sizes ($G=4$) and sparse MoE routing dynamics, policy gradient variance exploded. The policy mode-collapsed onto trivial reward-hacking strings.
* **Strategic Decision**: Reverted from RL to clean, deterministic SFT to isolate representation mechanics without policy optimization confounders.
