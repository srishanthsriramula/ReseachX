---
tags: [protocol, autograd, pre-hook, pytorch]
backlinks: "[[00_Index/00_Index_MOC|Index]], [[02_Theorems/Theorem_4_Soft_Riemannian_Natural_Gradient_Invariance|Theorem 4]]"
---

# ⚙️ PyTorch Autograd Forward Pre-Hook Architecture

```
Forward Training:  x ──► [Pre-Hook: D_α] ──► x_damped ──► z = x_damped @ A^T ──► Δy = z @ B^T
Backward Autograd: ∇_A L = (∇_z L)^T @ (x @ D_α) = (∇_A L_uncond) @ D_α
AdamW Update:      ΔA = -η (∇_A L_uncond) @ D_α
Forward Output:    Δy = -η B (∇_A L_uncond) @ (Σ_X + α·I)^(-1) x
```

### Invariants
* Attached dynamically during training and evaluation.
* Removed cleanly in `finally:` blocks, restoring base model probes to $< 10^{-5}$ $\Delta\text{NLL}$.
