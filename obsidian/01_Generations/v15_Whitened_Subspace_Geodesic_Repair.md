# Generation v15: Whitened Subspace Geodesic Repair (Theorem 7 Formulation)

## 1. Scientific Motivation
Even with stratified layer targeting, unconstrained gradient descent in the selected attention layers creates weight drift that degrades existing capabilities. We sought a closed-form geometric constraint that guarantees parameter updates remain orthogonal to the control capability manifold.

## 2. Theoretical Formulation (Theorem 7)
Given control input covariance $C_{\text{code}} = \mathbb{E}[x_{\text{code}} x_{\text{code}}^T]$ and target input covariance $C_{\text{target}} = \mathbb{E}[x_{\text{target}} x_{\text{target}}^T]$, find an initialization $A_0 \in \mathbb{R}^{r \times d}$ that maximizes target capacity while minimizing control interference:
$$\max_{A} \frac{\text{tr}(A C_{\text{target}} A^T)}{\text{tr}(A C_{\text{code}} A^T)}$$

### Closed-Form Solution:
1. Whitening transformation: $G_{\text{code}} = C_{\text{code}} + \alpha I \implies G_{\text{code}}^{-1/2} = E \Lambda^{-1/2} E^T$
2. Transformed target covariance: $\Sigma_{\text{whitened}} = G_{\text{code}}^{-1/2} C_{\text{target}} G_{\text{code}}^{-1/2}$
3. Eigendecomposition: $\Sigma_{\text{whitened}} = V \Omega V^T$
4. Optimal Subspace Basis: $A_0 = U_r^T G_{\text{code}}^{-1/2}$, where $U_r$ contains the top-$r$ eigenvectors.

## 3. Implementation
* Injected $A_0$ directly into `lora_A` modules across all stratified layers.
* Initialized $B=0$.
* Evaluated on GSM8K vs. MBPP code tasks.

## 4. Key Discovery
Demonstrated that $A_0$ aligned updates with directions of maximum target activation variance relative to control activation variance.
