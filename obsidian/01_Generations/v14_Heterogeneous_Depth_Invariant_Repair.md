# Generation v14: Heterogeneous Depth-Invariant Representation Repair

## 1. Scientific Motivation
Laguna-XS.2 possesses a non-uniform transformer depth: out of 40 layers, 10 are global full-context attention layers and 30 are sliding-window attention (SWA, 512-token local window) layers. Previous uniform PEFT allocations distributed adaptation ranks equally across all layers, causing semantic instability in early input embedding layers and output token de-biasing heads.

## 2. Core Hypothesis
$$\text{Hypothesis 14: Restricting adaptation exclusively to middle-to-deep representation layers } \mathcal{L}_{\text{stratified}} \subset [1, 26] \text{ minimizes representation corruption in foundational embeddings.}$$

## 3. Implementation Details
* **Stratified Layer Signature**:
  $$\mathcal{L}_{\text{stratified}} = [1, 2, 4, 6, 8, 10, 11, 12, 14, 16, 18, 20, 21, 22, 24, 26]$$
* **Target Projections**: Standard attention linear operators ($W_q, W_v, W_o$).
* **Evaluation Domains**: GSM8K target math vs. C4 / Python code control perplexity.

## 4. Empirical Findings
* Completely freezing layers $l < 1$ and $l > 26$ prevented output distribution collapse.
* Maintained baseline code perplexity ($\Delta\text{NLL}_{\text{code}} < 0.05$) while enabling learning on mathematical reasoning.

## 5. Failure Analysis & Next Steps
While layer stratification localized the depth dimension, within each layer the parameter updates still drifted along arbitrary directions in parameter space. This necessitated explicit geometric subspace constraints within each selected layer (v15).
