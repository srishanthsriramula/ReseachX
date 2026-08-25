---
tags: [theorem, riemannian-geometry, natural-gradient, proof]
backlinks: "[[00_Index/00_Index_MOC|Index]], [[01_Generations/v12_Soft_Riemannian_Fisher_Damping|v12]], [[04_Protocols/Autograd_PreHook_Execution_Graph|Pre-Hook Graph]]"
---

# 📐 Theorem 4: Soft Riemannian Natural Gradient Closed-Form Invariance

### Statement
Transforming the LoRA input via forward pre-hook $\tilde{x} = x \cdot (\Sigma_X + \alpha I)^{-1/2}$ causes the PyTorch autograd chain rule to automatically compute the exact regularized Riemannian Natural Gradient during AdamW optimization:
$$\nabla_A \mathcal{L}_{\text{Riemannian}} = (\nabla_A \mathcal{L}_{\text{uncond}}) \cdot (\Sigma_X + \alpha I)^{-1/2}$$
And on forward generation, the perturbation evaluates to the exact Fisher Inverse:
$$\Delta y = -\eta B (\nabla_A \mathcal{L}) \cdot (\Sigma_X + \alpha I)^{-1} x$$
with **zero extra inference latency overhead**. $\blacksquare$
