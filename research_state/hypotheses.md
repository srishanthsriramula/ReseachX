# ACTIVE RESEARCH HYPOTHESES

### Hypothesis 1 (Leading): The Controllable Repair Subspace Hypothesis
Model failures are heterogeneous and cannot be repaired by unprojected scalar gradient tuning. Effective repair requires computing preference margin gradients between actual wrong trajectories and desired completions, projected onto the null space of preserved capability Fisher information, applied via low-rank conditional gating.

### Hypothesis 2: Routing-Limited Failure Dominance in Sparse MoEs
A substantial fraction ($\ge 30\%$) of multi-step reasoning failures in Laguna XS.2 arise because the router fails to select capable existing experts, meaning router adaptation or counterfactual routing outperforms expert weight editing.

### Hypothesis 3: Dense Null-Space Repair Generality
The failure-conditioned null-space repair mechanism is a fundamental property of transformer representations and operates effectively in dense architectures (e.g. Gemma 2 2B IT) without MoE routing.
