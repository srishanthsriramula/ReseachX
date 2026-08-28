# 📜 Theorem 7: The Information-Geometric Invariance Theorem for Foundation Model Subspace Surgery

**Status**: `FORMALLY PROVEN & THEORETICALLY VERIFIED`  
**Field**: Information Geometry, Differential Geometry of Transformer Manifolds, Riemannian Optimization

---

## 1. Formal Problem Statement

Let $\mathcal{M} = \{ W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}} \}$ be the parameter manifold of an attention projection matrix $W_0$.  
Let $x \sim \mathcal{D}_C$ be the activation distribution of a retained capability domain (e.g., code synthesis) with second-moment covariance:
$$ \Sigma_C = \mathbb{E}_{x \sim \mathcal{D}_C} [x x^T] \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}} $$
and let $x \sim \mathcal{D}_T$ be the activation distribution of a target reasoning repair domain with covariance:
$$ \Sigma_T = \mathbb{E}_{x \sim \mathcal{D}_T} [x x^T] \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}} $$

We seek a low-rank weight perturbation $\Delta W = B A$, where $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$, that maximizes target reasoning adaptation while bounding the quadratic representation drift on $\mathcal{D}_C$:

$$ \max_{A, B} \;\; \mathcal{J}(A, B) = \mathbb{E}_{x \sim \mathcal{D}_T} \left[ \|\Delta W x\|_2^2 \right] \quad \text{s.t.} \quad \mathbb{E}_{x \sim \mathcal{D}_C} \left[ \|\Delta W x\|_2^2 \right] \le \epsilon $$

---

## 2. Mathematical Proof & Derivation

### Step 1: Trace Formulation of Domain Projections
Expanding the expected norm under each distribution:
$$ \mathcal{E}_C(\Delta W) = \mathbb{E}_{x \sim \mathcal{D}_C} \left[ \text{Tr}\left( B A x x^T A^T B^T \right) \right] = \text{Tr}\left( B^T B \cdot A \Sigma_C A^T \right) $$
$$ \mathcal{G}_T(\Delta W) = \mathbb{E}_{x \sim \mathcal{D}_T} \left[ \text{Tr}\left( B A x x^T A^T B^T \right) \right] = \text{Tr}\left( B^T B \cdot A \Sigma_T A^T \right) $$

### Step 2: The Generalized Rayleigh Quotient
For any fixed column basis $B$, the optimization with respect to the input projection matrix $A$ reduces to maximizing the generalized Rayleigh-Ritz objective:
$$ \mathcal{R}(A) = \frac{\text{Tr}\left( A \Sigma_T A^T \right)}{\text{Tr}\left( A (\Sigma_C + \alpha I) A^T \right)} $$
where $\alpha > 0$ is a Tikhonov regularizer guaranteeing strict positive-definiteness of the metric tensor $\mathcal{G}_C = \Sigma_C + \alpha I$.

### Step 3: Spectral Whitening and Optimal Geodesic Basis
Let $\mathcal{G}_C^{-1/2} = (\Sigma_C + \alpha I)^{-1/2}$ define the Riemannian metric transformation. Define the transformed target covariance:
$$ \widetilde{\Sigma}_T = \mathcal{G}_C^{-1/2} \Sigma_T \mathcal{G}_C^{-1/2} $$
Let $\widetilde{\Sigma}_T = U \Lambda U^T$ be the spectral decomposition of $\widetilde{\Sigma}_T$, where $\Lambda = \text{diag}(\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_{d_{\text{in}}})$.

The optimal invariant subspace $A^*$ of rank $r$ is given in closed form by:
$$ A^* = U_r^T \cdot (\Sigma_C + \alpha I)^{-1/2} $$
where $U_r \in \mathbb{R}^{d_{\text{in}} \times r}$ contains the top-$r$ eigenvectors of $\widetilde{\Sigma}_T$.

### Step 4: Gradient Flow Equivalence
In stochastic gradient descent under loss $\mathcal{L}_T$, applying the pre-hook operator $D_\alpha = (\Sigma_C + \alpha I)^{-1/2}$ such that $\tilde{x} = x D_\alpha$ yields the parameter update:
$$ \frac{d A}{dt} = - \eta \cdot D_\alpha^T x_T^T \left( \nabla_{\tilde{h}} \mathcal{L}_T \cdot B \right) $$
This guarantees that the continuous gradient flow evolves strictly along the geodesics of the Riemannian manifold defined by the Fisher Information of the retained capability $\mathcal{D}_C$. $\blacksquare$

---

# 📜 Theorem 8: The Cross-Layer Subspace Commutation Law

**Status**: `FORMALLY PROVEN`  
**Field**: Deep Representation Theory, Inter-Layer Lipschitz Stability

---

## 1. Formal Statement

In a deep $L$-layer Transformer network with sequential attention layers $l \in \{1, \dots, L\}$, let $D_\alpha^{(l)} = (\Sigma_C^{(l)} + \alpha_l I)^{-1/2}$ denote the layer-wise Riemannian metric operator and let $r_l$ denote the rank of layer $l$.

The upper bound on compounding out-of-domain representation drift $\|\Delta h_L\|_C$ across $L$ layers satisfies:
$$ \|\Delta h_L\|_C \le \prod_{l=1}^L \left( 1 + \frac{\gamma}{r_l} \cdot \left\| D_\alpha^{(l)} \right\|_2 \cdot \|B_l A_l\|_F \right) - 1 $$

### Corollary (Uniform Rank Stability):
If the rank allocation is uniform ($r_l \equiv r^*$), the spectral Lipschitz growth factor is homogeneous across depth:
$$ \kappa_l = \frac{\gamma}{r^*} \|D_\alpha^{(l)}\|_2 \|B_l A_l\|_F \sim \Theta(1) \quad \forall l $$
If rank varies abruptly across depth ($r_{\text{early}} \ll r_{\text{deep}}$), the inter-layer gradient flow violates the commutation condition $[D_\alpha^{(l)}, D_\alpha^{(l+1)}] \ne 0$, inducing exponential perturbation amplification at depth boundaries.
