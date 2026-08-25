# CONFLICT & DISAGREEMENT REGISTRY

This registry tracks explicit contradictions, competing findings, and opposing evidence discovered during the research.

---

### Conflict 1: Causal Necessity vs Adaptation Plasticity
- **Side A (Original Causal Thesis)**: The parameter block most causally responsible for a capability is the optimal target for modifying that capability.
- **Side B (Matched Adaptation Finding)**: Fine-tuning causal expert E229 yields almost no adaptation gain ($+0.0280$), whereas the most-routed expert adapts effectively ($+0.0770$).
- **Resolution**: Decisively resolved in favor of Side B. Causal necessity reflects read-heavy execution, not writeable plasticity.

---

### Conflict 2: Teacher-Forced NLL vs Free-Running Generation Accuracy
- **Side A (SFT Loss Metric)**: Writable expert fine-tuning produces a large $-0.35$ NLL improvement on GSM8K.
- **Side B (Behavioral Generation Metric)**: Autonomous generation accuracy drops from 76.0% to 74.5%.
- **Resolution**: Resolved in favor of Side B. Exposure bias decouples teacher-forced likelihood from autonomous multi-step reasoning accuracy.

---

### Conflict 3: Population Correlation vs Top-K Selector Reliability
- **Side A (v7 Global Atlas)**: Gradient norm correlates strongly with plasticity across all 9,984 experts ($R \approx 0.82$).
- **Side B (v8/v11 Selector Audits)**: Selecting the top-1 or top-4 gradient experts does not reliably outperform random parameter placements.
- **Resolution**: Both are true: gradient magnitude provides a coarse macro-level filter, but fine-grained top-K ranking is dominated by noise, curvature, and directional incoherence.

---

### Conflict 4: v10 Apparent Gain vs v11 Fresh Confirmation Failure
- **Side A (v10 Findings)**: Guided LoRA achieved $+0.047$ accuracy gain on calibration data.
- **Side B (v11 Confirmatory Test)**: Under 5 seeds and 6 architecture-matched random placements on fresh test data, 95% bootstrap CI $[-0.0226, +0.0590]$ crossed zero.
- **Resolution**: Resolved in favor of Side B. v10 suffered from Winner's Curse due to small validation sample sizes.
