# Generation v18: Frontier Science Shift to GPQA Diamond

## 1. Scientific Motivation
GSM8K grade-school arithmetic had become saturated and prone to surface-level pattern matching. To evaluate genuine deep reasoning and surgical capability expansion, the benchmark suite was shifted to **GPQA Diamond** (198 PhD-level, Google-proof multiple-choice science questions in physics, chemistry, and biology).

## 2. The Token Length Truncation Crisis
* Initial evaluations generated with `max_new_tokens = 256` yielded an apparent accuracy of **8.59%**.
* Inspection revealed that Laguna-XS.2 generates extensive Chain-of-Thought (CoT) reasoning traces (~600–900 tokens). At 256 tokens, generation was severed mid-derivation before reaching any final boxed answer letter.
* **Resolution**: Scaled evaluation generation horizon to `max_new_tokens = 1024` with greedy decoding (`do_sample=False`). Accuracy immediately rose to ~46.5%.
