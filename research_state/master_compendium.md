# ResearchX Master Compendium (v1 → v25)

## 1. Overview
This compendium serves as the single source of truth for all mathematical formulations, architectural definitions, and empirical results in ResearchX.

## 2. Model Profile: Laguna-XS.2
- 33.4B MoE (3.0B active per token), 48 layers, 256 routed experts + 1 shared expert.
- 5 attention projections per layer (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `g_proj`).
- Agentic code specialization.

## 3. Mathematical Foundations
- **Theorems 1–6**: MoE routing bifurcations, Jacobian condition numbers, $\mu P$ rank coupling.
- **Theorem 7**: Activation-covariance subspace invariance (falsified in v23 due to downstream loss blindness).
- **Theorems 8–10**: Riemannian metric equivalence, geodesic capacity horizons, representation arithmetic.
- **Theorem 11**: Fisher-weighted gradient covariance subspace invariance (v24 theoretical foundation).
