# CLAIM REGISTRY

This registry categorizes all empirical and theoretical claims within the Laguna research corpus by verification status.

---

### Category A: Established Claims (Decisively Verified)
1. **Capacity vs Active Compute Decoupling**: Sparse MoE architectures (e.g. Laguna 118B-A8B, 33.4B-A3B) deliver frontier-level coding performance at fractional active inference cost.
2. **Causal Necessity $\neq$ Routing Frequency**: Frequently routed experts are often general syntactic/lexical routers; causally essential capability experts can have low routing rank (e.g., E229 routing rank #18, causal rank #1).
3. **Causal Necessity $\neq$ Adaptation Plasticity**: Causally necessary experts (e.g., E229) resist SFT adaptation and perform worse during fine-tuning than heavily routed experts.
4. **Forced Routing Does Not Restore Plasticity**: Forcing router tokens into rigid causal experts during training does not increase their adaptation plasticity.
5. **Teacher-Forced NLL $\neq$ Autonomous Generation Accuracy**: Substantial NLL reduction under teacher forcing can coexist with degraded free-running reasoning accuracy due to exposure bias.
6. **Single-Vector Contrastive Projection Collapses in High Dimensions**: Subtracting a single scalar control gradient in $\mathbb{R}^{10^7}$ reduces gradient norm by $<0.5\%$ and fails to protect retention benchmarks.

---

### Category B: Strongly Supported Claims (Multiple Replications)
1. **Population-Level Gradient-Plasticity Correlation**: Across full MoE layers, parameter gradient norm correlates with SFT adaptation plasticity ($R \approx 0.82$).
2. **Top-K Selector Instability**: Selecting top-K parameter blocks via scalar gradient norm does not reliably pick the optimal adaptation subset on fresh held-out data.
3. **Hardware Headroom for Direct BF16**: Resident 33.4B BF16 model occupies 62.3 GiB VRAM on an RTX PRO 6000 (96GB), leaving ~30 GiB headroom for causal hooks and activation caching.

---

### Category C: Plausible Claims (Theoretically Grounded, Awaiting Final Validation)
1. **Failure Mode Heterogeneity**: GSM8K and reasoning failures consist of distinct causal mechanisms (arithmetic, entity tracking, plan maintenance) whose repair gradients are mutually orthogonal or antagonistic.
2. **Preservation Subspace Null-Projection**: Projecting repair gradients orthogonally to the empirical Fisher information matrix of correct behaviors prevents collateral damage.
3. **Functional Trust Region Fairness**: PEFT methods should be compared at equal functional policy shift ($\mathbb{E}[\text{KL}(P_0 \| P_\theta)]$) rather than equal raw trainable parameter count.

---

### Category D: Contradicted & Disproven Claims
1. **Read-Write Equivalence**: The claim that causally essential experts are the best targets for capability adaptation is DISPROVEN.
2. **Routing Blocker Hypothesis**: The claim that low routing frequency causes poor adaptation plasticity in causal experts is DISPROVEN.
3. **v10 Guided LoRA Superiority**: The claim that v10 guided LoRA placement statistically outperforms random placement is DISPROVEN by v11.
