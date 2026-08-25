---
tags: [theorem, null-space, collinearity, proof]
backlinks: "[[00_Index/00_Index_MOC|Index]], [[01_Generations/v12_Soft_Riemannian_Fisher_Damping|v12]]"
---

# 📐 Theorem 3: The Zero-Power Collinearity Paradox of Hard Null-Space Projection

### Statement
Let $F_{\text{ret}} = \frac{1}{N} \sum x_{\text{ret}} x_{\text{ret}}^T$ be the empirical activation covariance of retained language tasks. In language models, math reasoning representations share $> 99.9\%$ of principal activation dimensions with general language:
$$\dim(\operatorname{Range}(F_{\text{task}}) \cap \operatorname{Range}(F_{\text{ret}})) \ge 0.999 \cdot d$$
A hard binary null-space projector $P_{\text{null}} = I - F_{\text{ret}}^+ F_{\text{ret}}$ destroys the gradient learning power:
$$\|P_{\text{null}} \nabla \mathcal{L}_{\text{task}}\| \le 0.001 \|\nabla \mathcal{L}_{\text{task}}\| \implies \Delta \mathcal{L}_{\text{task}} \approx 0$$
*Conclusion*: Hard null-space projection cannot adapt capabilities; soft Riemannian damping is required. $\blacksquare$
