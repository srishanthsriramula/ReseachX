# Laguna-XS.2 Architecture & Behavior Profile

## 1. Architectural Specifications
* **Total Parameters**: 33.4 Billion
* **Active Parameters per Token**: 3.0 Billion (sparse top-8 MoE routing out of 256 experts + 1 shared expert)
* **Layer Composition**: 40 transformer layers
  * 10 Global Attention layers
  * 30 Sliding-Window Attention (SWA) layers (512-token window)
* **Attention Mechanism**: 5 linear projections per layer: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `g_proj` (SwiGLU gate).
* **Context Horizon**: 1,048,576 tokens

## 2. Training Domain & Behavioral Specialization
* Laguna-XS.2 was trained specifically as an **agentic coding model** (SWE-bench, Terminal-Bench).
* It was **not** pretrained on general knowledge encyclopedias (MMLU, trivia, humanities).
* **Blank Output Phenomenon**: When presented with non-STEM / non-code multiple choice questions (e.g. MMLU philosophy, history), the model produces empty strings / whitespace because its conditional generation distribution assigns zero probability to non-code completions.
* **STEM Reasoning Viability**: The model possesses strong mathematical and scientific reasoning capability acquired through code and algorithmic training, achieving 46.0% baseline accuracy on PhD-level GPQA Diamond under Chain-of-Thought prompting.
