# Generation v20.1: Forensic Code Audit & Norm-Matched SFT

## 1. Forensic Discoveries
1. **The Inactive Constraint**: In v20, `apply_whitened_initialization_to_model` initialized $A_0$ with the whitened basis but never set `requires_grad = False` on $A$. As a result, both $A$ and $B$ were updated freely by AdamW.
2. **Norm Amplification**: The unregularized whitened basis had Frobenius norm $\|A_0\|_F \approx 72.4$ vs. Kaiming norm $\|A_{\text{Kaiming}}\|_F \approx 36.2$. This 2× norm inflation gave the whitened initialization an artificial learning rate advantage.
3. **Extraction Substring Artifact**: Substring matching (`"c" in "calculation"`) created false positives in earlier base evals, artificially inflating base accuracy to 53.5%.

## 2. Corrected Protocol
* Implemented strict Kaiming norm matching:
  $$A_0 \leftarrow A_0 \cdot \frac{\|A_{\text{Kaiming}}\|_F}{\|A_0\|_F}$$
* Strictly enforced `sA.weight.requires_grad = False` for Geodesic runs.
* Fixed extraction to multi-tiered regex: `\boxed{X}` -> `The answer is (X)` -> last `(X)` -> standalone letter.
