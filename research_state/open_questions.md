# Critical Open Questions

1. **Does Fisher Gradient Covariance Beat Standard LoRA?**: Will v24's gradient-weighted subspace achieve higher GPQA gain than standard LoRA (+4.9%) while matching or exceeding its pristine 0.014 code retention?
2. **Optimal Calibration Sample Threshold**: Is $N=180$ code samples sufficient to estimate the Fisher metric in $d=3072$ dimensions, or is full dataset streaming required?
3. **Generalization to Other Architectures**: Does the Fisher gradient subspace transfer seamlessly to dense models (e.g. LLaMA-3) or is it specific to sparse MoE attention backbones?
