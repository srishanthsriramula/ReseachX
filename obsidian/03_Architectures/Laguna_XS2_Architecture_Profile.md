---
tags: [architecture, laguna-xs2, sparse-moe, hardware, specs]
aliases: [Laguna XS.2 Profile, Model Specs]
---

# 🏛️ Laguna XS.2 (33.4B-A3B) Architectural Specification

## 1. Physical Model Parameters & Geometry

| Architectural Parameter | Numerical Value | Description / Engineering Role |
|---|---|---|
| **Total Parameter Count** | $33,400,000,000$ ($33.4\text{B}$) | Total resident model footprint in memory |
| **Active Parameters Per Token** | $3,000,000,000$ ($3.0\text{B}$) | Compute FLOPs executed per forward token |
| **Number of Transformer Layers** | $48$ | Total depth of residual stream |
| **Global Attention Layers** | $12$ | Full context attention across sequence |
| **Sliding Window Attention Layers** | $36$ | Local 512-token receptive field attention |
| **Hidden Dimension ($d_{\text{model}}$)** | $2048$ ($d=2048$) | Residual stream vector width |
| **Attention Head Dimension ($d_{\text{head}}$)** | $128$ | Per-head projection dimension |
| **Number of Attention Heads** | $64$ | Query projection width ($64 \times 128 = 8192$) |
| **Key-Value Heads (GQA)** | $8$ | Grouped Query Attention KV width ($8 \times 128 = 1024$) |
| **Total Routed Experts Per Layer** | $256$ | Sparse routed SwiGLU expert banks |
| **Always-Active Shared Experts** | $1$ | Common baseline MLP layer |
| **Top-$k$ Expert Routing** | $k=8$ | Top-8 softmax router selection |
| **Intermediate FFN Dimension** | $8192$ | Hidden dimension inside each SwiGLU expert |

---

## 2. Memory Footprint & Hardware Constraints on AMD Instinct MI300X

| Precision / Component | VRAM Footprint | MI300X Capacity ($192\text{ GiB}$) | Execution Headroom |
|---|---|---|---|
| **Base Model Weights (BF16)** | $62.29\text{ GiB}$ | $32.5\%$ of VRAM | $129.4\text{ GiB}$ free for KV cache & activations |
| **LoRA Adapters ($r=63$)** | $0.05\text{ GiB}$ ($12.64\text{M}$ params) | $<0.1\%$ of VRAM | Negligible footprint |
| **Generation KV Cache (Batch 32, Seq 512)** | $18.40\text{ GiB}$ | $9.6\%$ of VRAM | Enables high-throughput parallel generation |
| **Peak Training VRAM (Batch 8 Accum)** | $80.50\text{ GiB}$ | $42.0\%$ of VRAM | **100% OOM-Safe with Zero Paging** |
