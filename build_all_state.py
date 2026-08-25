import json
import os
from pathlib import Path

ROOT = Path('/Users/srishanthsriramula/Downloads/Research-/research_state')
ROOT.mkdir(parents=True, exist_ok=True)

def write_file(name, content):
    p = ROOT / name
    p.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"Written {name} ({len(content)} bytes)")

# 1. knowledge_timeline.md
write_file("knowledge_timeline.md", """# CHRONOLOGICAL KNOWLEDGE TIMELINE

This timeline chronicles the empirical and conceptual trajectory of the Laguna research program from initial architecture analysis through the shift to causal expert surgery, experimental reversals, benchmark scaling, and the final formulation of failure-conditioned repair geometry.

---

### Phase 1: Architecture, Capacity, and Synthetic Data Limits (PDF pp. 1–45)
- **Laguna Model Family Architecture**:
  - *Laguna XS.2*: 33.4B total parameters, 3.0B active per token, 256 routed experts + 1 shared always-active expert, top-8 routing, 48 transformer layers (12 global-attention layers, 36 sliding-window layers with 512-token window), Grouped-Query Attention (GQA) with 8 KV heads, 1,048,576-token context window.
  - *Laguna S 2.1*: 118B total parameters, 8.0B active per token, 256 routed experts + 1 shared expert, top-10 routing.
  - *Compute & Pretraining*: Trained on >30T tokens (>27T unique tokens) heavily weighted toward code, syntax, and execution. Post-training used 38,000 tasks across ~17,000 real git repositories with commit-reproduction RL, incentivizing persistence, verification, backtracking, and long-horizon tool execution.
- **Parametric Capacity vs Active Compute Paradox**:
  - Laguna achieves frontier-tier coding performance because it possesses a massive 118B-capacity parametric memory reservoir while paying the runtime inference cost of only an ~8B active model.
  - However, inference efficiency does not imply cheap training: pretraining required 4,096 H200 GPUs.
- **The Limits of Synthetic Data**:
  - *Ungrounded Synthetic Data*: Direct LLM self-generation suffers from *Diversity Collapse* (mode collapse onto high-frequency knowledge, erasing obscure APIs, race conditions, edge cases, and syntax bugs).
  - *Grounded Synthetic Data*: Conditioned on real external artifacts (actual git commits, issue tickets, diffs) to anchor the distribution in real-world entropy.
  - *Verifiable Synthetic Data*: Programmatic code and mathematical deductions subjected to deterministic compiler, interpreter, or unit test verification.
- **Multimodal & Computer-Use Proposal**:
  - Proposed adding vision and OS/browser interaction to give the agent visual debugging and GUI execution capability rather than naively expanding active parameters from 8B to 16B.
- **Hardware Strategy**:
  - Identified a single workstation with 1× NVIDIA RTX PRO 6000 (96GB VRAM) / AWS `g7e.2xlarge` as the target platform for running BF16 Laguna XS.2 (62.3 GiB resident VRAM) with ~30 GiB headroom for causal surgery hooks, gradients, and activation caching.

---

### Phase 2: The Shift to Causal Expert Surgery (PDF pp. 45–100)
- **The Core Scientific Pivot**:
  - Shifted from macro architectural characterization to mechanistic capability localization: *Can we identify where a capability is causally implemented in a sparse MoE and modify only that computation without full model fine-tuning?*
- **Localization Concepts Evaluated**:
  1. *Routing Frequency*: Initially tested whether experts most frequently selected by the router during a task contain the capability.
  2. *Gradient Magnitude*: Tested whether experts receiving the largest gradient norm under task loss identify writeable parameters.
  3. *Causal Ablation under Fixed Routing*: Realized that routing frequency is purely correlational. Developed causal ablation: setting an expert's output to zero while forcing routing gates to remain frozen to isolate true counterfactual necessity ($\Delta\text{NLL}$).
  4. *Target vs. Control Specificity*: Defined causal specificity as $\Delta\text{NLL}_{\text{target}} - \Delta\text{NLL}_{\text{control}}$ to filter out generic computational hubs.
  5. *Hierarchical Search Protocol*: Two-stage sweep (layer-level routed-branch ablation, followed by individual expert ablation within top layers) to avoid evaluating all 9,984 experts individually.
  6. *Coalition Dynamics*: Evaluated pairwise and multi-expert ablations to detect non-linear interactions (positive synergy vs destructive interference).

---

### Phase 3: Infrastructure Engineering & Diagnostic Failures (PDF pp. 100–140)
- **Kaggle T4 × 2 Limitations**: OOM failures and extreme latency when sharding MoE layers across 2×16GB GPUs.
- **Transformers Compressed Execution Problem**: Custom Laguna model code used `DecompressExperts` to dynamically unpack FP8 weights on the fly during the forward pass, which broke PyTorch backward autograd and custom intervention hooks.
- **vLLM Incompatibilities**:
  - vLLM FP8 KV-cache kernels failed on Laguna's custom sliding-window/GQA architecture.
  - Custom grouped INT8 kernels failed to execute reliably on target architectures.
- **Storage Bottlenecks**: Ephemeral disk exhaustion during multi-shard safetensors downloads on Kaggle/AWS instances.
- **Resolution via Direct BF16 on RTX PRO 6000 96GB (`g7e.2xlarge`)**:
  - Converted model to uncompressed native BF16 weights loaded directly into 62.3 GiB VRAM on RTX PRO 6000, achieving 8-second 39-layer causal sweeps and enabling direct PyTorch hook manipulation.

---

### Phase 4: First Causal Results & The Scoring Bug Fix (PDF pp. 135–160)
- **Scoring & Prompt Formatting Bug**:
  - Early teacher-forcing evaluations missed the mandatory newline following the `</think>` token in Laguna S2.1's reasoning format, shifting token alignment.
  - Selective-logit scorers differed from full-logit cross-entropy; required an exact token-level aligned cross-entropy verification check.
- **Discovery of E229 (Layer 36, Expert 229)**:
  - In frontend CSS/Flexbox capability search, Layer 36 ablation showed $\Delta\text{NLL}_{\text{target}} = +1.3797$.
  - Individual expert sweep revealed Expert 229 accounted for almost the entire damage: $\Delta\text{NLL}_{\text{target}} = +1.2858$, $\Delta\text{NLL}_{\text{control}} = -0.0318$ (Specificity = $+1.2858$).
  - E229 survived across search seeds, bootstrap 95% CI was strictly positive ($P > 0.99$), and renormalized causal specificity remained massive.
- **The Routing vs Causal Disconnect**:
  - E229 was routing rank #18 (routed on only ~2.1% of tokens) but was causal rank #1.
  - In contrast, L38/E60 was routing rank #1 but had modest causal specificity ($\Delta\text{NLL} = +0.1917$).
  - *Initial Excitement*: Appeared to validate the core thesis that routing frequency is misleading and causal ablation isolates the true capability circuit.

---

### Phase 5: The Critical Reversal — Matched Adaptation (PDF pp. 160–220)
- **The Matched Adaptation Experiment**:
  - Fine-tuned candidate expert weights on target capability data under identical parameter budgets (~3.15M params per expert):
    - *Causal Expert*: Layer 36, Expert 229
    - *Most-Routed Expert*: Layer 38, Expert 60 (or Layer 36 top-routed)
    - *Random Same-Layer Expert*: Layer 36, Expert 45
    - *Random Global Expert*: Layer 12, Expert 110
- **The Decisive Reversal**:
  - *Most-Routed Expert* achieved $+0.0770$ to $+0.1100$ adaptation NLL improvement.
  - *Causal Expert (E229)* achieved only $+0.0280$, barely exceeding random baselines ($+0.0171$ same-layer, $+0.0031$ global).
- **Core Scientific Insight**:
  - **Causal Necessity $\\neq$ Routing Access $\\neq$ Gradient Accessibility $\\neq$ Adaptation Plasticity**.
  - A parameter essential for existing capability execution acts as a brittle, read-heavy computational structure; modifying it destroys baseline competence rather than writing new capability.

---

### Phase 6: Systematic Characterization from v6 to v8 (PDF pp. 220–280)
- **v6 (Falsification-Grade Search)**:
  - Tested the "Routing Blocker Hypothesis": Did E229 fail to adapt simply because the router didn't route training tokens to it?
  - Evaluated *Forced-Access Training* (forcing router probabilities to route to E229 during SFT).
  - *Result*: Even with forced routing access, E229's adaptation plasticity remained poor, falsifying the routing blocker hypothesis.
- **v7 & v7.1 (Global Writeability Atlas)**:
  - Mapped all 9,984 experts across 39 layers for: (1) Causal Specificity, (2) Routing Frequency, (3) Gradient Norm, (4) Adaptation Plasticity.
  - Discovered a strong population-level correlation ($R \\approx 0.82$) between Gradient Norm and Adaptation Plasticity.
- **v8 & v8.2 (Cross-Capability Replication)**:
  - Replicated across Frontend (CSS), Python Systems, and Math Reasoning.
  - Evaluated budget curves ($K=1, 2, 4, 8$ experts).
  - *Crucial Finding*: Population-level correlation ($R \\approx 0.8$) was robust globally, but top-K selector reliability was noisy: the top-1 gradient expert was not reliably the top-1 adapter.

---

### Phase 7: Real Benchmark Transition — v9 (PDF pp. 280–340)
- **Transition to GSM8K and MBPP**:
  - Scaled from synthetic micro-benchmarks to GSM8K (multi-step math) and MBPP (Python programming).
  - Evaluated matched parameter budgets: 4 writable experts (~12.6M parameters) vs Attention LoRA (~12.6M parameters) across 3 training seeds.
- **The NLL vs Autonomous Generation Paradox**:
  - Writable expert training dropped teacher-forced NLL dramatically ($\Delta\text{NLL} = -0.35$ on GSM8K).
  - *Autonomous Free-Running Accuracy*: Accuracy dropped from 76.0% down to 74.5% or stagnated at 76.0%.
  - *Root Cause*: Teacher forcing evaluates on gold prefixes that the model never generates at test time (Exposure Bias / State Distribution Shift). Modifying high-gradient parameters caused severe collateral damage to multi-step reasoning trajectories.

---

### Phase 8: Behavior-Aware Dose Selection — v10 (PDF pp. 340–365)
- **v10 Methodology**:
  - Introduced behavior-based checkpoint evaluation and dose calibration on free-running GSM8K accuracy rather than teacher-forced NLL.
  - Tested an attempted Contrastive Selector: $\\text{Score} = \\|\\nabla_\\theta \\mathcal{L}_{\\text{GSM8K}}\\| - \\lambda \\|\\nabla_\\theta \\mathcal{L}_{\\text{MBPP}}\\|$.
  - Compared Contrastive LoRA vs Random LoRA vs Writable Experts.
- **Apparent Success & Subsequent Methodological Breakdown**:
  - Showed an apparent $+0.047$ accuracy gain on calibration data.
  - *Critical Flaw*: Contrastive and raw-gradient selectors picked identical parameter blocks because subtracting one scalar MBPP gradient vector in 12-million-dimensional space reduced norm by $<0.5\\%$.
  - Small validation sets ($N=48$) introduced the *Winner's Curse* (overfitting hyperparameters to stochastic validation noise).

---

### Phase 9: Fresh Confirmation & Replication Failure — v11 (PDF pp. 365–385)
- **v11 Strict Protocol**:
  - Frozen experimental policy, untouched fresh GSM8K test set ($N=384$), 5 random training seeds (107, 211, 503, 887, 1597), architecture-signature matched random LoRA distribution (6 independent random placements).
- **Decisive Empirical Result**:
  - Guided LoRA and Writable Experts failed to statistically outperform random LoRA placements.
  - Paired 2-way bootstrap 95% CI crossed zero: Mean gain $+0.0174$, CI $[-0.0226, +0.0590]$.
  - The positive finding from v10 failed to replicate under rigorous confirmatory conditions.

---

### Phase 10: Root-Cause Reassessment & New Mathematical Framework (PDF pp. 385–450)
- **The 24 Reassessment Hypotheses**:
  - Deconstructed why scalar gradient magnitude, unprojected SFT, and full-expert editing fail.
  - Identified failure heterogeneity, parameter sensitivity $\\neq$ controllability, exposure bias, absence of preservation null spaces, and rescue vs damage trade-offs.
- **The New Mathematical Object**:
  - Shifted from "Writable Expert" to **"Failure-Conditioned Controllable Repair Subspace"**.
  - Formulated the **Behavioral Repair Kernel (BRK)**, Preference Margin Gradients, Null-Space Fisher Projection, Repair-to-Interference Ratio (RIR), and Conditional Write Gating.
- **Counterfactual Routing Diagnostic**:
  - Partitioned failure modes into: *representation-limited*, *routing-limited*, *expert-content-limited*, *decoding-limited*, and *formatting-limited*.
- **Dense Model Experiment (Gemma 2 2B IT)**:
  - Formulated a clean non-MoE experiment to isolate the core algebra of repair vs preservation on self-generated failure trajectories without MoE routing artifacts or AdamW optimizer noise.
""")
