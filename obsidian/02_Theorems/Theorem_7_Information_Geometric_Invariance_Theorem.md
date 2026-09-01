# Theorem 7: Information-Geometric Subspace Invariance (Activation Covariance Formulation)

## Mathematical Statement
Let $\mathcal{M}$ be the parameter manifold of a linear layer $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$. Let $C_{\text{code}} = \mathbb{E}[x_{\text{code}} x_{\text{code}}^T]$ and $C_{\text{target}} = \mathbb{E}[x_{\text{target}} x_{\text{target}}^T]$ denote the empirical second-moment activation covariance matrices.

The optimal rank-$r$ adaptation subspace $A_0 \in \mathbb{R}^{r \times d_{\text{in}}}$ that maximizes target activation variance relative to control activation variance is given by:
$$A_0 = \arg\max_{A \in \mathbb{R}^{r \times d_{\text{in}}}} \frac{\text{tr}(A C_{\text{target}} A^T)}{\text{tr}(A (C_{\text{code}} + \alpha I) A^T)}$$

### Closed-Form Solution:
1. Compute the regularized whitening matrix:
   $$G_{\text{code}} = C_{\text{code}} + \alpha I = E \Lambda E^T \implies G_{\text{code}}^{-1/2} = E \Lambda^{-1/2} E^T$$
2. Construct the transformed target covariance:
   $$\Sigma_{\text{whitened}} = G_{\text{code}}^{-1/2} C_{\text{target}} G_{\text{code}}^{-1/2}$$
3. Perform eigendecomposition $\Sigma_{\text{whitened}} = V \Omega V^T$.
4. Select the top-$r$ eigenvectors $U_r = V_{:, -r:}$.
5. Construct the unnormalized basis $A_0^{\text{raw}} = U_r^T G_{\text{code}}^{-1/2}$.
6. Apply Kaiming norm matching:
   $$A_0 = A_0^{\text{raw}} \cdot \frac{\sqrt{2/d_{\text{in}}} \sqrt{r \cdot d_{\text{in}}}}{\|A_0^{\text{raw}}\|_F}$$

---

## Empirical Boundary Conditions & Failure Analysis (from v23)
While the linear algebra of Theorem 7 is mathematically sound, its physical application to deep transformers fails due to three unstated assumptions:
1. **Single-Layer Linearity Assumption**: In a deep network, $\Delta y = \Delta W x$ does not translate directly to loss change. The downstream gradient $\frac{\partial L}{\partial y}$ modulates the output effect.
2. **Activation Variance ≠ Loss Gradient**: Input directions with high activation variance often correspond to invariant boilerplate tokens with zero gradient.
3. **Sample Support Degeneracy**: Estimating $C \in \mathbb{R}^{3072 \times 3072}$ from $N=16$ samples results in a rank-16 matrix, making the pseudo-null space degenerate.
