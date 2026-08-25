# Laguna XS.2 v12: Soft Riemannian Fisher Damping Confirmation Report

Protocol version: `v12.0-riemannian-fisher-stratified-lora`
Fresh snapshot SHA256: `96b439d19505c1c895ed5535cef4653eade325e93091c1629bc4a458babef6e5`

## Key Hypotheses Tested
1. **Stratified Layer Geometry**: Evaluates layers `[1, 2, 8, 11, 12, 16, 21, 26]` vs bottleneck layers `[20, 24, 23, 19, 21, 25, 16, 18]`.
2. **Soft Riemannian Fisher Damping**: Pre-conditions LoRA updates via Riemannian operator to suppress interference along principal retained axes.

## Summary Leaderboard

| method                              |   n_seeds |   mean_accuracy |   base_accuracy |   mean_accuracy_gain |   two_way_ci_low |   two_way_ci_high |   positive_seed_count |   target_improvement_mean |   control_abs_shift_mean |   trainable_params |
|:------------------------------------|----------:|----------------:|----------------:|---------------------:|-----------------:|------------------:|----------------------:|--------------------------:|-------------------------:|-------------------:|
| bottleneck_lora_unconditioned       |         3 |        0.782118 |         0.78125 |          0.000868056 |       -0.0251736 |         0.0269097 |                     1 |                0.00549587 |               0.00250407 |           12644352 |
| stratified_lora_unconditioned       |         3 |        0.796007 |         0.78125 |          0.0147569   |       -0.0147569 |         0.0451389 |                     3 |                0.00185529 |               0.00373069 |           12644352 |
| stratified_lora_riemannian_alpha001 |         3 |        0.789062 |         0.78125 |          0.0078125   |       -0.0217014 |         0.0364583 |                     2 |                0.0280937  |               0.00235478 |           12644352 |
| stratified_lora_riemannian_alpha010 |         3 |        0.786458 |         0.78125 |          0.00520833  |       -0.0217014 |         0.0329861 |                     2 |                0.00713936 |               0.00247391 |           12644352 |

## Guardrails & Verification
- Fresh final GSM8K test set (N=384) has 0 overlap with prior training/selection splits.
- Retained control benchmark: MBPP (N=160).
- Zero inference latency overhead: pre-conditioning baked into evaluation weights.