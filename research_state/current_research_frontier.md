# Current Research Frontier: Laguna XS.2 v12

**Last Updated**: 2026-08-25 01:47:12 UTC
**Active Protocol**: `v12.0-riemannian-fisher-stratified-lora`

## Summary of Established Findings (v1-v11)
1. **MoE Expert Surgery Failure**: Direct parameter surgery on routed MoE experts suffers from discontinuous router bifurcations, causing negative gains across all seeds.
2. **Scalar Gradient Localization Failure**: Raw task gradient norm measures bottleneck curvature/traffic, not plastic writeability. Contiguous gradient-guided LoRA placed in bottom 16.7% of configurations.
3. **Stratified Layer Hierarchy Success**: Early-to-mid stratified layer spans (`random_signature_01: [1, 2, 8, 11, 12, 16, 21, 26]`) achieved +1.48 pp gain (79.60%, max seed 80.99%) by avoiding bottleneck congestion.
4. **Hard Null-Space Trap**: Hard binary null-space projectors destroy 99.9% of task learning signal due to language representation collinearity.

## Active Investigation (v12)
* **Hypothesis**: Soft Riemannian Fisher Damping on Stratified Layer Spans maximizes task adaptation while strictly suppressing catastrophic interference along principal retained directions.
* **Notebook**: `laguna/laguna_xs2_v12_riemannian_fisher_stratified_lora.ipynb`
