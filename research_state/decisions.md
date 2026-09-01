# Strategic Decisions Log

1. **Revert from RL (GRPO) to SFT (Aug 30, 2026)**: Single-GPU MoE policy gradient variance caused catastrophic collapse at step 13. SFT provides deterministic, clean measurements of representation mechanics.
2. **Switch from MMLU to Code NLL & HumanEval (Sep 1, 2026)**: Laguna-XS.2 is a specialized coding agent; MMLU collapses to blank text.
3. **Revert Prompt Format to `\boxed{}` (Sep 1, 2026)**: Laguna-XS.2 was trained on LaTeX boxed outputs. Standardizing to "Let's think step by step" dropped base accuracy by 7%.
4. **Transition from Activation Covariance to Fisher Gradient Covariance (Sep 1, 2026)**: Discovered that activation covariance ignores downstream loss sensitivity, motivating v24.
