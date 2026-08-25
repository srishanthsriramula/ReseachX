---
tags: [architecture, laguna-xs2, sparse-moe, hardware, tensor-shapes, blueprint]
aliases: [Laguna XS.2 Blueprint, MoE Architecture Diagram]
---

# 🏛️ Laguna XS.2 (33.4B-A3B) Full Architecture Blueprint

## 1. Macro Architecture & Layer Hierarchy

Laguna XS.2 is a 33.4-Billion parameter Sparse Mixture-of-Experts (MoE) foundation model operating with 3.0-Billion active parameters per forward token.

```mermaid
flowchart TD
    IN["Input Token IDs x"] --> EMB["Embedding Layer: W_emb ∈ ℝ^(128256 × 2048)"]
    EMB --> L0["Transformer Layer 0 (Global Attention)"]
    L0 --> L1["Transformer Layer 1 (Sliding Window Attention)"]
    L1 --> LDOT["... Layers 2 to 46 ..."]
    LDOT --> L47["Transformer Layer 47 (Global Attention)"]
    L47 --> NORM["Final RMSNorm: ℝ^2048 → ℝ^2048"]
    NORM --> HEAD["LM Head: W_head ∈ ℝ^(128256 × 2048)"]
    HEAD --> OUT["Logits y ∈ ℝ^(T × 128256)"]
```

---

## 2. Micro-Architecture: Detailed Layer Block Structure

Each of the 48 transformer layers consists of an **Attention Sublayer (Grouped-Query Attention)** followed by a **Sparse MoE Sublayer (256 Routed Experts + 1 Shared Expert)**:

```mermaid
flowchart TD
    H_IN["Input Residual State: h_l ∈ ℝ^(B × T × 2048)"] --> NORM1["RMSNorm 1 (Attention Norm)"]
    
    NORM1 --> Q_PROJ["q_proj: ℝ^2048 → ℝ^8192 (64 Heads)"]
    NORM1 --> K_PROJ["k_proj: ℝ^2048 → ℝ^1024 (8 KV Heads)"]
    NORM1 --> V_PROJ["v_proj: ℝ^2048 → ℝ^1024 (8 KV Heads)"]
    
    Q_PROJ & K_PROJ & V_PROJ --> ATTN["FlashAttention / RoPE GQA Engine"]
    ATTN --> O_PROJ["o_proj: ℝ^8192 → ℝ^2048"]
    
    H_IN --> ADD1["Residual Add 1: h'_l = h_l + Attn(h_l)"]
    O_PROJ --> ADD1
    
    ADD1 --> NORM2["RMSNorm 2 (MoE Norm)"]
    NORM2 --> ROUTER["Top-8 Softmax Router: W_g ∈ ℝ^(256 × 2048)"]
    ROUTER --> GATING["G(x) = Top8(Softmax(W_g · x))"]
    
    GATING --> EXP["Active 8 Routed SwiGLU Experts"]
    NORM2 --> SHARED["Always-Active Shared Expert (MLP)"]
    
    EXP & SHARED --> MOE_OUT["MoE Output: ∑ g_i E_i(x) + E_shared(x)"]
    ADD1 --> ADD2["Residual Add 2: h_(l+1) = h'_l + MoE_Output"]
    MOE_OUT --> ADD2
    ADD2 --> H_OUT["Output to Layer l+1: h_(l+1) ∈ ℝ^(B × T × 2048)"]
```

---

## 3. Parameter Dimensions & Hardware Memory Ledger

| Architectural Component | PyTorch Tensor Name | Input Dimension ($d_{\text{in}}$) | Output Dimension ($d_{\text{out}}$) | Total Parameters Per Layer |
|---|---|---|---|---|
| **Query Projection** | `self_attn.q_proj.weight` | $2048$ ($d_{\text{model}}$) | $8192$ ($64 \times 128$) | $16,777,216$ |
| **Key Projection** | `self_attn.k_proj.weight` | $2048$ ($d_{\text{model}}$) | $1024$ ($8 \times 128$) | $2,097,152$ |
| **Value Projection** | `self_attn.v_proj.weight` | $2048$ ($d_{\text{model}}$) | $1024$ ($8 \times 128$) | $2,097,152$ |
| **Output Projection** | `self_attn.o_proj.weight` | $8192$ ($64 \times 128$) | $2048$ ($d_{\text{model}}$) | $16,777,216$ |
| **MoE Router Gate** | `block_sparse_moe.gate.weight` | $2048$ ($d_{\text{model}}$) | $256$ (Experts) | $524,288$ |
| **256 Routed Experts** | `block_sparse_moe.experts[e]` | $2048$ ($d_{\text{model}}$) | $8192$ ($d_{\text{ffn}}$) | $256 \times 50.33\text{M} = 12.88\text{B}$ (Full Model) |
| **Shared Expert** | `block_sparse_moe.shared_expert` | $2048$ ($d_{\text{model}}$) | $8192$ ($d_{\text{ffn}}$) | $50,331,648$ |
