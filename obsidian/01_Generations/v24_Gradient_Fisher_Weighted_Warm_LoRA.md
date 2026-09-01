# Generation v24: Gradient & Fisher-Weighted Warm LoRA

## 1. Theoretical Formulation (Theorem 11)
To correct the fundamental flaw of activation covariance, v24 reformulates the subspace selection using the **Empirical Fisher Gradient Covariance**:
$$G = \mathbb{E}\left[\left\|\frac{\partial L}{\partial y}\right\|^2 x x^T\right]$$
Where $\partial L/\partial y$ is the output gradient flowing back through all 40 downstream layers, exactly encoding true loss sensitivity.

## 2. Implementation Specifications
* **Code Calibration Dataset**: Scaled from 16 to 180 tasks (164 HumanEval canonical problems + 16 control tasks).
* **STEM Calibration Dataset**: 200 labeled science reasoning questions.
* **Warm LoRA Architecture**: Initialize $A_0$ from the Fisher-whitened subspace, but keep $A$ fully trainable ($27.4\text{M params}$) to preserve learning capacity.
