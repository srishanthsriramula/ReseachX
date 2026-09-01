# Current Research Frontier: v24 / v25

## Frontier Status
1. **Falsification Established**: v23 conclusively proved that activation covariance $C = \mathbb{E}[xx^T]$ fails to protect control capabilities or guide target adaptation because it ignores backpropagated downstream loss gradients $\partial L/\partial y$.
2. **Theoretical Reformulation**: v24 formulates the **Empirical Fisher Gradient Covariance** $G = \mathbb{E}[\|\partial L/\partial y\|^2 xx^T]$ (Theorem 11).
3. **Execution Ready**: `laguna_xs2_v24_gradient_warm_lora.ipynb` is fully compiled with 180 code calibration samples (HumanEval + control tasks) and 200 STEM reasoning samples.
