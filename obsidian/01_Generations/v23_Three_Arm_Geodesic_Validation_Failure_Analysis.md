# Generation v23: Three-Arm Validation & The Theoretical Breakdown

## 1. Experimental Design
* **3 Arms × 5 Random Seeds = 15 Complete Runs**:
  * **Arm 1 (Geodesic)**: $A_0$ whitened basis, frozen ($11.6\text{M trainable params}$).
  * **Arm 2 (Warm LoRA)**: $A_0$ whitened basis, trainable ($27.4\text{M trainable params}$).
  * **Arm 3 (Standard LoRA)**: $A_0$ Kaiming random, trainable ($27.4\text{M trainable params}$).
* **Target Benchmark**: GPQA Diamond (198 questions, 1024 max tokens, CoT greedy).
* **Control Benchmark**: Code NLL (16 diverse algorithmic tasks, 1,107 tokens).
* **Training**: 32 SFT steps, batch size 4, gradient checkpointing ON.

## 2. Empirical Results

```
================================================================================
Arm                          Mean GPQA Gain    95% Bootstrap CI     Mean NLL Shift
--------------------------------------------------------------------------------
Geodesic (A0 frozen)             +0.3%         [-1.6%, +1.7%]           0.0653
Warm LoRA (A0 trainable)         +3.6%         [+2.0%, +5.2%]           0.0651
Standard LoRA (random A)         +4.9%         [+3.6%, +6.1%]           0.0142
================================================================================
```

## 3. Theoretical Post-Mortem
1. **Activation Covariance ≠ Loss Sensitivity**:
   $$C = \mathbb{E}[xx^T] \quad \text{vs.} \quad \Delta L \approx \left\langle \frac{\partial L}{\partial y}, \Delta W \cdot x \right\rangle$$
   $C$ identifies input directions with high variance (e.g. whitespace, syntax boilerplate), which have near-zero gradient $\partial L/\partial y$. It is blind to low-variance directions with critical loss sensitivity.
2. **Inadequate Calibration Support**: 16 code prompts in $d=3072$ created a rank-deficient pseudo-null space.
3. **Parameter Bottleneck**: Freezing $A$ reduced parameter capacity by 2.4×, crippling complex multi-step reasoning.
