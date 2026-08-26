# Laguna XS.2 v13: High-Capacity Adaptive Riemannian Stratified LoRA Report

Protocol version: `v13.0-high-capacity-adaptive-riemannian`
Fresh snapshot SHA256: `96b439d19505c1c895ed5535cef4653eade325e93091c1629bc4a458babef6e5`

## Key Hypotheses Tested
1. **Layer-Adaptive Damping (alpha_l)**: alpha=0.05 on early layers (L1-2) protects syntax; alpha=0.002 on deep layers (L16-26) unlocks maximum reasoning torque.
2. **High-Capacity Scaling (r=128 -> r=256)**: Scaling parameter capacity from 12.6M to 51.4M with muP learning rate scaling (LR_128=7e-6, LR_256=5e-6).
3. **Strict 8-Step Dose Calibration**: Prevents cumulative residual leakage and preserves the 88% invariance shield.

## Summary Leaderboard

| method                                   |   n_seeds |   mean_accuracy |   base_accuracy |   mean_accuracy_gain |   two_way_ci_low |   two_way_ci_high |   positive_seed_count |   target_improvement_mean |   control_abs_shift_mean |   trainable_params |
|:-----------------------------------------|----------:|----------------:|----------------:|---------------------:|-----------------:|------------------:|----------------------:|--------------------------:|-------------------------:|-------------------:|
| stratified_lora_baseline_r63             |         3 |        0.796007 |         0.78125 |           0.0147569  |      -0.0147786  |         0.0451389 |                     3 |                0.00185529 |               0.00373069 |           12644352 |
| stratified_lora_adaptive_riemannian_r63  |         3 |        0.798611 |         0.78125 |           0.0173611  |      -0.00954861 |         0.0451389 |                     3 |                0.0540895  |               0.00351942 |           12644352 |
| stratified_lora_adaptive_riemannian_r128 |         3 |        0.789931 |         0.78125 |           0.00868056 |      -0.0251736  |         0.0442708 |                     1 |                0.0704328  |               0.0103212  |           25690112 |
| stratified_lora_adaptive_riemannian_r256 |         3 |        0.78559  |         0.78125 |           0.00434028 |      -0.0304036  |         0.0381944 |                     2 |                0.0982674  |               0.0130251  |           51380224 |

## Guardrails & Verification
- Fresh final GSM8K test set (N=384) with 0 overlap with training/selection splits.
- Retained control benchmark: MBPP (N=160).
- Zero inference latency: Pre-hook damping operator baked into evaluation weights.
