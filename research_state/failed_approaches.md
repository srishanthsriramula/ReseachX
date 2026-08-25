# FAILED APPROACHES & NEGATIVE RESULTS

1. **Routing Frequency Localization**: Assuming frequently routed experts contain capability circuits (Disproven in Exp 05).
2. **Causal Expert Fine-Tuning**: Assuming causally necessary experts are optimal adaptation sites (Disproven in Exp 06).
3. **Forced-Access SFT**: Attempting to fix causal expert rigidity via forced routing (Disproven in Exp 08).
4. **Teacher-Forced NLL Optimization on Real Benchmarks**: Using SFT cross-entropy on gold prefixes as a proxy for multi-step reasoning accuracy (Disproven in Exp 11).
5. **Single-Vector Contrastive Subtraction**: Using $\|\nabla \mathcal{L}_{\text{target}}\| - \lambda \|\nabla \mathcal{L}_{\text{control}}\|$ as a contrastive selector in high dimensions (Disproven in Exp 12).
6. **Scalar Gradient-Guided PEFT Placement**: Selecting LoRA or expert locations via scalar gradient heuristics without null-space projection (Disproven in Exp 13).
