---
tags: [protocol, autograd, pre-hook, pytorch]
backlinks: "[[00_Index/00_Index_MOC|Index]], [[02_Theorems/Theorem_4_Soft_Riemannian_Natural_Gradient_Invariance|Theorem 4]]"
---

# ⚙️ PyTorch Autograd Forward Pre-Hook Architecture

```mermaid
flowchart LR
    subgraph Forward ["Forward Training Pass"]
        X["Input x"] --> PRE["Pre-Hook: D_α"]
        PRE --> XD["x_damped"]
        XD --> LORA_A["z = x_damped @ A^T"]
        LORA_A --> LORA_B["Δy = z @ B^T"]
    end

    subgraph Backward ["Backward Autograd Pass"]
        CHAIN["∇_A L = (∇_z L)^T @ (x @ D_α)"] --> NAT["= (∇_A L_uncond) @ D_α"]
        NAT --> STEP["AdamW Update: ΔA ∝ (∇_A L) @ D_α"]
    end

    subgraph Output ["Forward Evaluation Output"]
        STEP -.-> OUT["Δy = -η · B · (∇_A L_uncond) · (Σ_X + α·I)^(-1) · x"]
    end

    Forward --> Backward
```

### Invariants
* Attached dynamically during training and evaluation.
* Removed cleanly in `finally:` blocks, restoring base model probes to $< 10^{-5}$ $\Delta\text{NLL}$.
