---
tags: [protocol, statistics, bootstrap, confidence-intervals]
---

# ⚙️ Two-Way Hierarchical Bootstrap Confidence Intervals

## 1. Formal Statistical Formulation
To evaluate whether a capability repair method achieves statistically significant gains without data leakage, we perform a **Two-Way Hierarchical Bootstrap** over both **Random Seeds ($S$)** and **Evaluation Test Items ($N=384$)**.

```
Algorithm: Two-Way Hierarchical Bootstrap
Input: Method correctness matrix M ∈ {0, 1}^{S × N}, Base correctness B ∈ {0, 1}^N
Draws: B_draws = 20,000, RNG Seed = 111071

For b = 1 to B_draws:
  1. Sample S seed indices with replacement: s* ~ Uniform(1, S)
  2. Sample N test item indices with replacement: i* ~ Uniform(1, N)
  3. Compute bootstrap method accuracy: Acc*_method = mean(M[s*, i*])
  4. Compute bootstrap base accuracy:   Acc*_base   = mean(B[i*])
  5. Compute bootstrap gain: ΔAcc*_b = Acc*_method - Acc*_base

Output: Observed Mean Gain, 95% CI Low (2.5% quantile), 95% CI High (97.5% quantile)
```

---

## 2. Empirical Verification Across Methods ($N=384$ Fresh Items)

| Method Name | Number of Seeds ($S$) | Mean Accuracy | Base Accuracy ($78.13\%$) | Observed Mean Gain | $95\%$ Two-Way Bootstrap CI | Positive Seed Count |
|---|---|---|---|---|---|---|
| **`stratified_lora_unconditioned`** | $3$ | **$79.60\%$** | $78.13\%$ | **$+1.48	ext{ pp}$** | **$[+0.12	ext{ pp}, +2.86	ext{ pp}]$** | **$3 / 3$ ($100\%$)** |
| **`stratified_lora_riemannian_alpha001`** | $3$ | **$78.91\%$** | $78.13\%$ | **$+0.78	ext{ pp}$** | **$[-0.42	ext{ pp}, +1.98	ext{ pp}]$** | **$2 / 3$ ($66.7\%$)** |
| **`bottleneck_lora_unconditioned`** | $3$ | **$78.21\%$** | $78.13\%$ | **$+0.09	ext{ pp}$** | **$[-1.42	ext{ pp}, +1.60	ext{ pp}]$** | **$1 / 3$ ($33.3\%$)** |
| **`standard_lora_40layers`** | $5$ | **$77.81\%$** | $78.13\%$ | **$-0.31	ext{ pp}$** | **$[-1.84	ext{ pp}, +1.20	ext{ pp}]$** | **$1 / 5$ ($20.0\%$)** |
