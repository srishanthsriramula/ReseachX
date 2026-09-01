# Theoretical Breakdown: Activation Covariance vs. Loss Sensitivity

## 1. The Core Disconnect
Standard covariance analysis computes:
$$C = \frac{1}{N} \sum_{i=1}^N x_i x_i^T$$
This measures purely where input signals have geometric energy.

However, the neural network's loss landscape satisfies:
$$\nabla_W \mathcal{L} = \frac{\partial \mathcal{L}}{\partial y} x^T$$

The expected outer product of weight gradients (the Fisher Information) is:
$$\mathcal{F}_W = \mathbb{E}\left[ \left(\frac{\partial \mathcal{L}}{\partial y} x^T\right) \otimes \left(\frac{\partial \mathcal{L}}{\partial y} x^T\right) \right] = \mathbb{E}\left[ \left\| \frac{\partial \mathcal{L}}{\partial y} \right\|^2 x x^T \right]$$

## 2. Comparative Pathology

| Metric | Property Measured | Vulnerability |
| :--- | :--- | :--- |
| **Activation Covariance $C$** | Input token variance | Captures high-frequency syntax / whitespace with zero loss gradient |
| **Fisher Covariance $G$** | Downstream loss sensitivity $\times$ input variance | Directly identifies parameters whose alteration shifts domain performance |

## 3. Why Standard LoRA Outperformed Geodesic LoRA in v23
Standard LoRA updates $W = W_0 + B A$ with gradients:
$$\frac{\partial \mathcal{L}}{\partial B} = \frac{\partial \mathcal{L}}{\partial y} (A x)^T, \quad \frac{\partial \mathcal{L}}{\partial A} = B^T \frac{\partial \mathcal{L}}{\partial y} x^T$$
Both updates are explicitly driven by the true loss gradient $\frac{\partial \mathcal{L}}{\partial y}$. Because Geodesic LoRA froze $A_0$ in directions defined purely by $C$, it forced $B$ to operate in sub-optimal directions, degrading learning efficiency.
