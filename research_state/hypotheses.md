# Active Hypotheses

## Hypothesis 24 (Fisher Gradient Covariance Subspace Invariance - Theorem 11)
$$\text{Weighting covariance by backpropagated output gradient norm } G = \mathbb{E}[\|\partial L/\partial y\|^2 xx^T] \text{ aligns initializations with true loss sensitivity,}$$
resolving the activation-covariance breakdown observed in v23.

## Hypothesis 25 (Unified Surgical Constrained LoRA)
$$\text{Combining Fisher-gradient initialization with adaptive Riemannian trust-region damping achieves Pareto-optimal adaptation}$$
with strictly bounded control capability degradation.
