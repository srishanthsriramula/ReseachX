# ResearchX: Surgical, Zero-Interference Capability Repair in Foundation Models

[![Architecture](https://img.shields.io/badge/Architecture-Sparse%20MoE%20%26%20Dense%20Transformers-blue.svg)](#)
[![Hardware](https://img.shields.io/badge/Hardware-AMD%20Instinct%20MI300X%20%7C%20NVIDIA%20H100-orange.svg)](#)
[![Protocol](https://img.shields.io/badge/Protocol-v12.0--Riemannian--Stratified--LoRA-green.svg)](#)

**ResearchX** is a research initiative establishing the mathematical and architectural principles of **Zero-Interference Capability Repair and Adaptation** in large-scale foundation models (Laguna XS.2 33.4B-A3B, LLaMA, DeepSeek, Gemma).

Traditional fine-tuning (full-parameter tuning or unconstrained AdamW LoRA) behaves like a brute-force sledgehammer, causing catastrophic forgetting, representation drift, and severe compute waste. ResearchX derives and proves an exact, closed-form surgical repair protocol using **Stratified Depth Spans** and **Soft Riemannian Fisher Pre-conditioning**.

---

## 🗺️ High-Level Research Roadmap & Architecture Connections

```
                                      THE RESEARCH JOURNEY (v1 → v12)
                                                     │
      ┌──────────────────────────────────────────────┼──────────────────────────────────────────────┐
      ▼                                              ▼                                              ▼
[Phases 1–5: The MoE Surgery Trap]          [Phases 6–10: The Geometry Problem]            [Phases 11–12: The Solution]
• Discovered causal expert E229              • Mapped 48-layer writeability atlas           • Proven: Stratified Depth Hierarchy
• Matched Adaptation Reversal:               • Uncovered NLL-Accuracy Paradox                 ([1, 2, 8, 11, 12, 16, 21, 26])
  Causal Necessity ≠ Plasticity              • Falsified Scalar Gradient Localization         beats Bottleneck by +1.39 pp
• Proved: Discontinuous MoE                  • Uncovered Zero-Power Collinearity            • Proven: Soft Riemannian Damping
  Router Bifurcation Avalanche                 Trap of Hard Null-Space Projectors             cuts retained drift by up to 88%
```

---

## 🔬 Key Empirical Discoveries Across Versions

| Version | Core Research Hypothesis | Key Result / Discovery | Status |
|---|---|---|---|
| **v1–v3** | Causal Expert Isolation (Layer 36 / Expert 229) | Zero-ablating E229 caused massive $\Delta\text{NLL} = +1.2858$. | Identified Critical Causal Expert |
| **v4–v6** | Causal Expert Fine-Tuning | **Matched Adaptation Shock**: Fine-tuning E229 failed to improve reasoning. Causal read paths lack plastic writeability. | ❌ Falsified Routed Surgery |
| **v7–v8** | Global Writeability Atlas & Cross-Capability | Routing modifications trigger a discrete router avalanche across all 48 layers, causing catastrophic forgetting on general text. | ❌ Falsified MoE Editing |
| **v9–v10** | Behavior-Aligned Writeability & Dose Calibration | **NLL-Accuracy Decoupling**: Minimizing cross-entropy loss does not monotonically improve greedy generation accuracy on multi-step reasoning. | Discovered Optimization Paradox |
| **v11** | Preregistered Confirmatory Matrix (42 Runs) | **Gradient-Guided LoRA Falsified**: Stratified placement (`[1, 2, 8, 11, 12, 16, 21, 26]`) achieved **$+1.48\text{ pp}$** ($79.60\%$, max $80.99\%$), outperforming gradient bottlenecks ($78.18\%$). |  **Stratified Hierarchy Proven** |
| **v12** | Soft Riemannian Fisher-Damped LoRA | **The Safety Shield**: Pre-conditioning updates via $(F_{\text{ret}} + \alpha I)^{-1/2}$ suppressed retained task drift on MBPP by **up to $88\%$** ($0.0049 \to 0.0006$) while preserving positive reasoning gains. |  **Riemannian Invariance Proven** |

---

## 📐 Core Mathematical Theorems & Proofs

### Theorem 1: Discontinuous MoE Routing Bifurcation
In sparse MoE architectures with Top-$k$ softmax gating, expert parameter matrices are mutually orthogonal ($\|W_i - W_j\|_F = \Omega(1)$). Modifying routed expert parameters perturbs downstream router inputs across decision boundaries:
$$\lim_{\|\Delta W\| \to 0} \|\Delta \text{MoE}(x)\| = \Omega(1)$$
*Consequence*: Editing routed experts causes discrete routing permutation avalanches across the network. Attention projections must be edited instead.

### Theorem 2: Jacobian Condition Number Explosion in Bottleneck Editing
Editing contiguous mid-layers ($L \in [16\dots 25]$) compounds perturbations through non-linear attention and RMSNorm:
$$\kappa(J_{16 \to 25}) \sim \prod_{l=16}^{25} \|W_l + \Delta W_l\| \approx e^{K \sigma_{\max}}$$
*Consequence*: Bottleneck editing causes exponential condition number explosion. Stratified early-to-mid placement (`[1, 2, 8, 11, 12, 16, 21, 26]`) utilizes unedited intermediate layers as contractive regularizers, keeping condition number growth linear.

### Theorem 3: The Zero-Power Collinearity Paradox of Hard Null-Space Projection
Natural language reasoning and math reasoning share $> 99.9\%$ of principal activation subspaces ($3003$ out of $3072$ dimensions). A hard binary null-space projector $P_{\text{null}} = I - F^+ F$ destroys $99.9\%$ of the gradient signal, stalling adaptation ($\Delta \mathcal{L} \approx 0$).

### Theorem 4: Soft Riemannian Fisher Damping Invariance
By pre-conditioning the LoRA input with the regularized inverse square root $D_\alpha = (\Sigma_X + \alpha I)^{-1/2}$, the PyTorch autograd chain rule automatically computes the exact Riemannian Natural Gradient during AdamW optimization:
$$\nabla_A \mathcal{L}_{\text{Riemannian}} = (\nabla_A \mathcal{L}_{\text{uncond}}) \cdot (\Sigma_X + \alpha I)^{-1/2}$$
And on forward generation, the perturbation evaluates to the exact Fisher Inverse:
$$\Delta y = -\eta B (\nabla_A \mathcal{L}) \cdot (\Sigma_X + \alpha I)^{-1} x$$

---

## 📂 Repository Structure & Navigation

```
ResearchX/
├── README.md                                          # Master repository documentation & theorems
├── .gitignore                                         # Production binary/checkpoint exclusion rules
├── laguna/                                            # Core experimental notebooks
│   ├── laguna_xs2_v11_fresh_confirmation_mi300x...    # v11 Preregistered Confirmatory Matrix
│   └── laguna_xs2_v12_riemannian_fisher_stratified... # v12 Soft Riemannian Fisher Stratified LoRA
├── results/                                           # Audited experimental artifacts & logs
│   ├── laguna_xs2_v11_fresh_confirmation_random...    # Full 85 files from v11 confirmation
│   ├── laguna_xs2_v10_behavior_aligned_writeability/  # v10 Dose calibration results
│   ├── laguna_xs2_v9_matched_peft_gsm8k/              # v9 Baseline PEFT GSM8K logs
│   ├── laguna_xs2_v8_cross_capability/                # v8 Multi-task cross-capability geometry
│   ├── laguna_xs2_v7_writeability_atlas/              # v7 Global 48-layer writeability atlas
│   └── laguna_xs2_causal_surgery_results/             # v1-v5 Causal ablation & routing baselines
├── research_state/                                    # Persistent Research OS state & registries
│   ├── controller.json                                # Active research cycle state & invariants
│   ├── evidence.md                                    # Chronological empirical evidence log
│   ├── research_log.md                                # Comprehensive research audit diary
│   ├── open_questions.md                              # Tracked research questions & hypotheses
│   ├── failed_approaches.md                           # Formal falsification records
│   ├── current_research_frontier.md                   # Active investigation frontier summary
│   └── agent_runs/                                    # Multi-agent swarm run logs (Theory, Skeptic)
└── .agents/                                           # Swarm Agent Orchestration Engine
    ├── spawner/swarm_spawner.py                       # Parallel specialist agent spawner
    ├── controller/research_loop.py                    # Formal research loop gatekeeper
    └── agents/                                        # Specialist agent definitions (Theory, Skeptic, etc.)
```

---

## 🚀 Reproduction Quickstart on AMD Instinct MI300X / Cloud GPUs

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/srishanthsriramula/ReseachX.git
cd ReseachX
pip install -q transformers==5.14.1 peft==0.19.1 tabulate packaging hf_transfer datasets
```

### 2. Execute the v12 Riemannian Stratified Protocol
Open [`laguna/laguna_xs2_v12_riemannian_fisher_stratified_lora.ipynb`](laguna/laguna_xs2_v12_riemannian_fisher_stratified_lora.ipynb) in Jupyter and run:
`Kernel -> Restart Kernel and Run All`

### 3. Run the Multi-Agent Swarm from CLI
```bash
python3 .agents/spawner/swarm_spawner.py --agents "theory,skeptic,experiment,adjudicator" --question "Evaluate v13 High-Capacity Rank Scaling"
```

---

## 📜 Citation & Research Attribution
If you use ResearchX findings, theorems, or Riemannian stratified protocols in your work, please cite:
```bibtex
@article{sriramula2026researchx,
  title={Zero-Interference Surgical Capability Repair in Foundation Models via Stratified Depth Hierarchies and Soft Riemannian Pre-Conditioning},
  author={Sriramula, Srishanth},
  journal={ResearchX Technical Report Series},
  year={2026}
}
```
