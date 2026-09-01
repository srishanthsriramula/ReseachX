# Theorem 11: Fisher-Weighted Gradient Covariance Subspace Invariance

## Mathematical Formulation
Let $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ be a linear transformation within an $L$-layer neural network $f_\theta$. Let $\mathcal{L}_{\text{control}}$ and $\mathcal{L}_{\text{target}}$ denote the loss functions over the respective domain distributions.

For an input token $x_t \in \mathbb{R}^{d_{\text{in}}}$ with pre-activation output $y_t = W_0 x_t$, the first-order loss perturbation under weight change $\Delta W = B A$ is:
$$\Delta \mathcal{L} \approx \sum_{t} \left\langle \frac{\partial \mathcal{L}}{\partial y_t}, B A x_t \right\rangle$$

The expected squared loss sensitivity under the control domain is governed by the **Empirical Fisher Gradient Covariance Matrix**:
$$G_{\text{control}} = \mathbb{E}_{\text{control}}\left[ \sum_{t} \left\| \frac{\partial \mathcal{L}_{\text{control}}}{\partial y_t} \right\|^2 x_t x_t^T \right] \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$$

Similarly, the target domain loss sensitivity is:
$$G_{\text{target}} = \mathbb{E}_{\text{target}}\left[ \sum_{t} \left\| \frac{\partial \mathcal{L}_{\text{target}}}{\partial y_t} \right\|^2 x_t x_t^T \right] \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$$

The optimal rank-$r$ adaptation subspace $A_0^*$ that maximizes target gradient alignment while minimizing control gradient interference solves the Rayleigh quotient:
$$A_0^* = \arg\max_{A \in \mathbb{R}^{r \times d_{\text{in}}}} \frac{\text{tr}(A G_{\text{target}} A^T)}{\text{tr}(A (G_{\text{control}} + \alpha I) A^T)}$$

### Theorem 11 Guarantee:
Unlike Theorem 7, Theorem 11 explicitly incorporates backpropagated sensitivity $\frac{\partial \mathcal{L}}{\partial y_t}$ from all downstream layers $l+1 \dots L$, rendering the subspace invariant to downstream nonlinearities, layer norms, and MoE routing.
