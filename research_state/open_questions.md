# OPEN QUESTIONS

1. **Failure Incoherence in Dense vs Sparse Models**: What is the empirical effective rank $r_{\text{eff}}(G)$ of failure repair gradients on GSM8K in Gemma 2 2B IT vs Laguna XS.2?
2. **Routing vs Content Failure Ratio**: What percentage of Laguna GSM8K failures are strictly routing-limited according to the Counterfactual Routing Oracle?
3. **Null-Space Preservation Efficacy**: Does projecting repair gradients onto $P_\perp$ eliminate damage on already-correct problems while maintaining non-zero rescue rates?
4. **Local Subspace vs Distributed Representation**: Does a rank-8 localized subspace update outperform distributed attention updates when both are constrained to identical functional KL trust regions?
5. **Conditional Gating Feasibility**: Can a lightweight gating vector $q(h) = \sigma(w^\top h)$ accurately trigger repair branches only on true failure states at inference time?
