# Evaluation Methodology & Empirical Flaw Registry

## 1. Evaluator Architecture
* **Target Benchmark**: **GPQA Diamond** (198 PhD-level science questions, Physics/Chemistry/Biology).
  * Prompt format: CoT reasoning ending with `\boxed{X}`.
  * Generation: Greedy decoding (`do_sample=False`), `max_new_tokens=1024`.
* **Control Benchmark**: **Code NLL** (16 diverse Python algorithms, 1,107 tokens).
  * Measures exact teacher-forced negative log-likelihood on standard algorithmic implementations.

## 2. Five Fatal Evaluation Flaws Identified and Fixed

| # | Flaw | Manifestation | Root Cause | Resolution |
| :---: | :--- | :--- | :--- | :--- |
| **1** | 256-Token Truncation | 8.59% GPQA score | CoT reasoning cut off before reaching answer | Scaled to `max_new_tokens=1024` |
| **2** | Substring Matching | False positive inflation | Regex matched `"c"` in `"calculation"` | Exact boundary regex matching `\boxed{X}` |
| **3** | MMLU Log-Likelihood | 25.8% (random chance) | Prompt format misaligned with MoE logits | Discarded MMLU; used Code NLL |
| **4** | MMLU Generation | Blank text output | Model is code-only; no general knowledge | Replaced with Code NLL & HumanEval |
| **5** | "Think Step-by-Step" | 46.0% vs 53.5% base | Model trained on LaTeX `\boxed{}` formatting | Reverted prompts to `\boxed{}` format |
