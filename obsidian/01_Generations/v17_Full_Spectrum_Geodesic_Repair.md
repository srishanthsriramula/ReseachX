# Generation v17: Full-Spectrum 5-Module Geodesic Repair

## 1. Architectural Discovery
Laguna-XS.2's attention block contains five distinct linear projections per layer, not four:
$$\text{Attention Linear Modules: } \{q\_proj, k\_proj, v\_proj, o\_proj, g\_proj\}$$
where `g_proj` is a SwiGLU gating projection operating directly on the post-attention hidden state:
$$\text{gate} = \text{softplus}(\text{g\_proj}(h))$$

## 2. The Vulnerability
Previous generations (v14–v16) only applied LoRA to 4 linear projections. Unadapted `g_proj` layers experienced massive backpropagation gradient shock, functioning as an unconstrained bypass channel that eroded control capabilities.

## 3. The Full-Spectrum Fix
* Auto-discovered all attention projections including `g_proj`.
* Computed whitened geodesic bases $A_0$ across all 5 attention modules for all 16 stratified layers ($16 \times 5 = 80$ linear layers).
* Total LoRA parameters scaled to 27,411,552 ($r=63$).
