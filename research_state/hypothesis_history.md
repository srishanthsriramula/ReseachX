# HYPOTHESIS EVOLUTION HISTORY

This registry traces every major hypothesis formulated, tested, revised, or falsified throughout the Laguna research program.

---

### Hypothesis 1: Active Parameter Count Determines Model Capability
- **Formulation**: A model activating ~8B parameters per token performs like a standard dense 8B model.
- **Why We Believed It**: Standard scaling laws correlate capability directly with active FLOPs/parameters per token.
- **Evidence/Experiment**: Architectural inspection of Laguna S 2.1 (118B total, 8B active) and XS.2 (33.4B total, 3B active) trained on >30T tokens with agentic RL.
- **Result**: Falsified. Laguna S 2.1 achieves 70.2% on Terminal-Bench 2.1 and 78.5% on SWE-bench Multilingual, dramatically outperforming dense 8B baselines.
- **Current Status**: SUPERSEDED. Sparse MoEs access a massive parametric memory reservoir while paying low per-token inference FLOPs.

---

### Hypothesis 2: Routing Frequency Identifies Capability-Specific Experts
- **Formulation**: The experts most frequently selected by the router during execution of a capability constitute its causal implementation.
- **Why We Believed It**: Intuition suggested that frequently routed experts are the ones performing the task computation.
- **Evidence/Experiment**: Routing sweeps on Frontend Flexbox/CSS benchmarks.
- **Result**: Falsified in Exp 05. Expert L38/E60 was routing rank #1 (~8.4% routing) but had minor causal necessity ($\Delta\text{NLL} = +0.1917$). Expert L36/E229 was routing rank #18 (~2.1% routing) but had massive causal necessity ($\Delta\text{NLL} = +1.2858$).
- **Current Status**: DISPROVEN. Routing frequency is correlational and reflects high-frequency lexical/syntactic routing, not causal necessity.

---

### Hypothesis 3: Causal Necessity Implies Adaptation Plasticity (Read-Write Equivalence)
- **Formulation**: The parameter blocks most causally essential for an existing capability (highest causal ablation $\Delta\text{NLL}$) are the optimal locations to fine-tune/modify that capability.
- **Why We Believed It**: The "Read-Write Equivalence" assumption: where knowledge is stored is where new knowledge should be written.
- **Evidence/Experiment**: Matched adaptation experiment in Exp 06 (L36/E229 vs L38/E60 vs random experts).
- **Result**: Falsified. E229 gained only $+0.0280$ in adaptation NLL, losing to the most-routed expert ($+0.0770$ to $+0.1100$) and barely beating random baselines ($+0.0171$).
- **Current Status**: DISPROVEN. Causal necessity $\neq$ adaptation plasticity. Essential experts act as rigid read-heavy memory structures.

---

### Hypothesis 4: Routing Starvation Explains Poor Causal Expert Plasticity
- **Formulation**: E229 failed to adapt because the router did not assign enough training tokens to it during SFT.
- **Why We Believed It**: E229 had low natural routing frequency (~2.1%). If gradients are gated by routing probability, low routing could starve the expert of updates.
- **Evidence/Experiment**: Forced-Access Training in Exp 08 (v6), forcing router tokens to E229 during training.
- **Result**: Falsified. Even with 100% forced routing access, E229's adaptation plasticity remained poor.
- **Current Status**: DISPROVEN. Lack of routing access is not the root cause of E229's rigidity.

---

### Hypothesis 5: Parameter Gradient Norm Identifies Writable Experts
- **Formulation**: Parameter blocks with the largest gradient norm $\|\nabla_\theta \mathcal{L}\|$ under target loss identify the most plastic, writeable parameters.
- **Why We Believed It**: Large gradients indicate parameters that strongly influence the loss landscape.
- **Evidence/Experiment**: Global Writeability Atlas (v7/v7.1) and Cross-Capability Replication (v8).
- **Result**: Supported at the population correlation level ($R \approx 0.82$), but top-K selector reliability was unstable.
- **Current Status**: PARTIALLY VALID BUT INSUFFICIENT. Gradient norm reflects sensitivity/curvature, but not controllability or directional coherence.

---

### Hypothesis 6: Teacher-Forced NLL Reduction Translates to Free-Running Reasoning Accuracy
- **Formulation**: Minimizing teacher-forced cross-entropy loss on target benchmarks directly improves autonomous multi-step reasoning.
- **Why We Believed It**: Standard SFT paradigm assumes lower perplexity/NLL leads to higher task accuracy.
- **Evidence/Experiment**: Real-benchmark evaluation on GSM8K in Exp 11 (v9).
- **Result**: Falsified. Writable expert tuning improved GSM8K NLL by $-0.35$, but autonomous generation accuracy dropped from 76.0% to 74.5%.
- **Current Status**: DISPROVEN. Teacher forcing evaluates on gold prefixes the model never generates at test time (Exposure Bias / Distribution Shift).

---

### Hypothesis 7: Contrastive Gradient Projection Isolates Task-Specific Parameters
- **Formulation**: Projecting the target gradient orthogonally to a control gradient ($\nabla \mathcal{L}_{\text{GSM8K}} - \lambda \nabla \mathcal{L}_{\text{MBPP}}$) discovers specific repair parameters.
- **Why We Believed It**: Contrastive objectives prevent catastrophic forgetting on retain benchmarks.
- **Evidence/Experiment**: Exp 12 (v10) and algebraic analysis.
- **Result**: Falsified. In 12-million-dimensional space, subtracting a single control gradient vector removes $<0.5\%$ of norm, collapsing back to raw gradient ranking.
- **Current Status**: DISPROVEN. A single vector cannot span the preservation subspace in high dimensions.

---

### Hypothesis 8: Guided LoRA/Expert Placement Statistically Outperforms Random Placement
- **Formulation**: Gradient-guided placement of PEFT adapters or expert surgery reliably outperforms architecture-matched random placement on fresh test data.
- **Why We Believed It**: Apparent positive gains in v10 calibration runs.
- **Evidence/Experiment**: Exp 13 (v11) fresh confirmation across 5 seeds, 6 random placements, 384 fresh test examples.
- **Result**: Falsified. 95% paired bootstrap CI $[-0.0226, +0.0590]$ crossed zero.
- **Current Status**: DISPROVEN FOR SCALAR GUIDANCE. Unprojected scalar gradient heuristics do not reliably outperform random placement.

---

### Hypothesis 9: The Controllable Repair Subspace Hypothesis
- **Formulation**: Model failures are heterogeneous. Effective repair requires computing preference margin gradients between actual wrong trajectories and desired completions, projected onto the null space of preserved capability Fisher information, applied via low-rank conditional gating.
- **Why We Believe It**: Derived from first principles to resolve exposure bias, failure gradient incoherence, and catastrophic collateral damage.
- **Evidence/Experiment**: Mathematical formulation of Behavioral Repair Kernel (BRK) and dense model protocol (Exp 14).
- **Current Status**: ACTIVE LEADING HYPOTHESIS.
