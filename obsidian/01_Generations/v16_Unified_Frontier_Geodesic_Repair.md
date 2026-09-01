# Generation v16: Unified Frontier Geodesic Scaling & Rank Coupling

## 1. Scientific Motivation
To test whether the whitened geodesic subspace scales with LoRA rank $r \in \{8, 16, 32, 63\}$, evaluating capacity horizons and information preservation across diverse multi-task suites.

## 2. Core Investigations
* Evaluated rank-scaling laws under $\mu P$ parameterization: $\alpha = r$.
* Tested whether increasing rank from $r=8$ to $r=63$ causes control domain leakage or improves multi-step reasoning capacity.

## 3. Findings & Limitations
* Rank $r=63$ provided substantial expressive capacity without numerical divergence under BF16.
* Discovered that standard 4-module attention coverage ($W_q, W_k, W_v, W_o$) leaked gradients into unadapted attention gating mechanisms in Laguna-XS.2.
