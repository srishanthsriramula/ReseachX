---
tags: [protocol, statistics, bootstrap, confidence-intervals]
aliases: [Two-Way Bootstrap Protocol, Significance Testing]
---

# ⚙️ Two-Way Hierarchical Bootstrap Confidence Intervals

> [!NOTE]
> **Purpose**: Eliminates sample-selection bias and evaluation variance by jointly resampling over **Random Seeds ($S$)** and **Fresh Evaluation Items ($N=384$)**.

---

## 📋 Algorithm Specification

> [!ALGORITHM] Two-Way Hierarchical Bootstrap Protocol
> * **Inputs**: Method correctness matrix $M \in \{0, 1\}^{S \times N}$, Base correctness vector $B \in \{0, 1\}^N$
> * **Parameters**: Bootstrap draws $B_{\text{draws}} = 20,000$, RNG Seed $= 111071$
> 
> 1. For $b = 1$ to $B_{\text{draws}}$:
>    * Sample $S$ seed indices with replacement: $s^* \sim \operatorname{Uniform}(1, S)$
>    * Sample $N$ test item indices with replacement: $i^* \sim \operatorname{Uniform}(1, N)$
>    * Compute bootstrap method accuracy: $\text{Acc}^*_{\text{method}} = \frac{1}{S \times N} \sum_{s, i} M[s^*, i^*]$
>    * Compute bootstrap baseline accuracy: $\text{Acc}^*_{\text{base}} = \frac{1}{N} \sum_i B[i^*]$
>    * Record bootstrap gain: $\Delta\text{Acc}^*_b = \text{Acc}^*_{\text{method}} - \text{Acc}^*_{\text{base}}$
> 2. Compute 95% Confidence Interval: $[\text{Quantile}_{0.025}(\Delta\text{Acc}^*), \text{Quantile}_{0.975}(\Delta\text{Acc}^*)]$

---

## 📊 Empirical Verification Matrix ($N=384$ Fresh Test Items)

| Method Name | Seeds ($S$) | Mean Accuracy | Baseline ($78.13\%$) | Observed Gain | 95% Bootstrap CI | Positive Seeds |
|---|---|---|---|---|---|---|
| **`stratified_lora_unconditioned`** | 3 | **$79.60\%$** | $78.13\%$ | **$+1.48\text{ pp}$** | **$[+0.12\text{ pp}, +2.86\text{ pp}]$** | **$3 / 3$ ($100\%$)** |
| **`stratified_lora_riemannian_alpha001`** | 3 | **$78.91\%$** | $78.13\%$ | **$+0.78\text{ pp}$** | **$[-0.42\text{ pp}, +1.98\text{ pp}]$** | **$2 / 3$ ($66.7\%$)** |
| **`bottleneck_lora_unconditioned`** | 3 | **$78.21\%$** | $78.13\%$ | **$+0.09\text{ pp}$** | **$[-1.42\text{ pp}, +1.60\text{ pp}]$** | **$1 / 3$ ($33.3\%$)** |
| **`standard_lora_40layers`** | 5 | **$77.81\%$** | $78.13\%$ | **$-0.31\text{ pp}$** | **$[-1.84\text{ pp}, +1.20\text{ pp}]$** | **$1 / 5$ ($20.0\%$)** |
