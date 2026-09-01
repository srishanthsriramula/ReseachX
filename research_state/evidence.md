# Empirical Evidence Compendium (v1 → v23)

## 1. v23 Three-Arm Randomized Trial (15 Runs, N=15)
* **Base Model Baseline**: GPQA Diamond = **46.0%** (91/198), Code NLL = **0.8385** (1,107 tokens).

### Seed-by-Seed Results:

| Arm | Seed | Trainable Params | GPQA Acc | GPQA Gain | Code NLL | NLL Shift |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Geodesic (A₀ frozen)** | 107 | 11.6M | 47.5% | +1.5% | 0.7763 | 0.0623 |
| **Geodesic (A₀ frozen)** | 211 | 11.6M | 46.5% | +0.5% | 0.7620 | 0.0765 |
| **Geodesic (A₀ frozen)** | 503 | 11.6M | 48.5% | +2.5% | 0.7749 | 0.0636 |
| **Geodesic (A₀ frozen)** | 719 | 11.6M | 47.0% | +1.0% | 0.7816 | 0.0569 |
| **Geodesic (A₀ frozen)** | 941 | 11.6M | 41.9% | -4.0% | 0.7714 | 0.0671 |
| **Warm LoRA (A₀ free)** | 107 | 27.4M | 52.5% | +6.6% | 0.7733 | 0.0652 |
| **Warm LoRA (A₀ free)** | 211 | 27.4M | 48.5% | +2.5% | 0.7684 | 0.0701 |
| **Warm LoRA (A₀ free)** | 503 | 27.4M | 50.5% | +4.5% | 0.7742 | 0.0643 |
| **Warm LoRA (A₀ free)** | 719 | 27.4M | 47.0% | +1.0% | 0.7760 | 0.0625 |
| **Warm LoRA (A₀ free)** | 941 | 27.4M | 49.5% | +3.5% | 0.7749 | 0.0636 |
| **Standard LoRA (Random A)** | 107 | 27.4M | 52.5% | +6.6% | 0.8229 | 0.0156 |
| **Standard LoRA (Random A)** | 211 | 27.4M | 50.5% | +4.5% | 0.8241 | 0.0144 |
| **Standard LoRA (Random A)** | 503 | 27.4M | 49.5% | +3.5% | 0.8259 | 0.0127 |

### Summary Statistics (5,000 Bootstrap Resamples):
* **Geodesic**: Mean Gain = **+0.3%** $[-1.6\%, +1.7\%]$, Mean NLL Shift = **0.0653**
* **Warm LoRA**: Mean Gain = **+3.6%** $[+2.0\%, +5.2\%]$, Mean NLL Shift = **0.0651**
* **Standard LoRA**: Mean Gain = **+4.9%** $[+3.6\%, +6.1\%]$, Mean NLL Shift = **0.0142**
