---
tags: [architecture, laguna-xs2, sparse-moe, hardware, tensor-shapes, blueprint]
aliases: [Laguna XS.2 Blueprint, MoE Architecture Diagram]
---

# 🏛️ Laguna XS.2 (33.4B-A3B) Full Architecture Blueprint

## 1. Model Macro-Topology

Laguna XS.2 is a 33.4-Billion parameter Sparse Mixture-of-Experts (MoE) foundation model operating with 3.0-Billion active parameters per token.

```mermaid
flowchart TD
    IN["Input Tokens x ∈ ℕ^T"] --> EMB["Embedding: W_emb ∈ ℝ^(128256 × 2048)"]
    EMB --> BLOCKS["48 Transformer Layers (12 Global + 36 Sliding Window)"]
    BLOCKS --> NORM["Final RMSNorm: ℝ^2048 → ℝ^2048"]
    NORM --> HEAD["LM Head: W_head ∈ ℝ^(128256 × 2048)"]
    HEAD --> OUT["Logits y ∈ ℝ^(T × 128256)"]
```

---

## 2. Layer Micro-Architecture (Compact Flow)

Each transformer layer contains an Attention Sublayer (Grouped-Query Attention) followed by a Sparse MoE Sublayer (Top-8 Gating over 256 Experts + 1 Shared Expert):

```mermaid
flowchart TD
    IN["Residual State h_l ∈ ℝ^(B × T × 2048)"] --> N1["RMSNorm 1 (Attention Norm)"]
    N1 --> GQA["GQA Projections: q (8192), k (1024), v (1024)"]
    GQA --> ATTN["FlashAttention Engine → o_proj (2048)"]
    ATTN --> ADD1["Residual Add 1: h'_l = h_l + Attn(h_l)"]
    ADD1 --> N2["RMSNorm 2 (MoE Norm)"]
    N2 --> ROUTE["Top-8 Router: W_g ∈ ℝ^(256 × 2048)"]
    ROUTE --> MOE["Top-8 SwiGLU Experts + Shared Expert"]
    MOE --> ADD2["Residual Add 2: h_(l+1) = h'_l + MoE(h'_l)"]
    ADD2 --> OUT["Output State h_(l+1) ∈ ℝ^(B × T × 2048)"]
```

---

## 3. Physical Parameter & Tensor Dimension Ledger

| Component | PyTorch Tensor Identifier | $d_{\text{in}}$ | $d_{\text{out}}$ | Parameters / Layer | Full 48-Layer Total |
|---|---|---|---|---|---|
| **Query Projection** | `self_attn.q_proj.weight` | $2048$ ($d_{\text{model}}$) | $8192$ ($64 \times 128$) | $16,777,216$ | $805,306,368$ |
| **Key Projection** | `self_attn.k_proj.weight` | $2048$ ($d_{\text{model}}$) | $1024$ ($8 \times 128$) | $2,097,152$ | $100,663,296$ |
| **Value Projection** | `self_attn.v_proj.weight` | $2048$ ($d_{\text{model}}$) | $1024$ ($8 \times 128$) | $2,097,152$ | $100,663,296$ |
| **Output Projection** | `self_attn.o_proj.weight` | $8192$ ($64 \times 128$) | $2048$ ($d_{\text{model}}$) | $16,777,216$ | $805,306,368$ |
| **MoE Router Gate** | `block_sparse_moe.gate.weight` | $2048$ ($d_{\text{model}}$) | $256$ (Experts) | $524,288$ | $25,165,824$ |
| **256 Routed Experts** | `block_sparse_moe.experts[e]` | $2048$ ($d_{\text{model}}$) | $8192$ ($d_{\text{ffn}}$) | $256 \times 50.33\text{M}$ | $618,240,000,000$ |
| **Shared Expert** | `block_sparse_moe.shared_expert` | $2048$ ($d_{\text{model}}$) | $8192$ ($d_{\text{ffn}}$) | $50,331,648$ | $2,415,919,104$ |
