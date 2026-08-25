---
tags: [handover, onboarding, guide, master-index, start-here]
aliases: [Master Handover Guide, Complete Onboarding Guide]
---

# 🚀 ResearchX: The Master Handover & Onboarding Guide
### *Everything You Need to Know to Understand, Run, and Extend the Research*

---

## 🌟 1. What Is ResearchX in 60 Seconds?

* **The Problem**: When you fine-tune large AI models on new reasoning skills (like multi-step math), standard fine-tuning destroys their pre-existing abilities (like code synthesis or English grammar).
* **The Goal**: Invent a surgical laser scalpel that repairs or teaches new capabilities with **Zero Interference / Zero Forgetting**.
* **The Model Studied**: **Laguna XS.2** (33.4 Billion total parameters, 3.0 Billion active per token, 48 transformer layers, 256 routed experts + 1 shared expert).
* **The Hardware**: AMD Instinct MI300X ($192\text{ GiB}$ HBM3) and NVIDIA GPUs running ROCm 7.14 / PyTorch 2.12.

---

## 🗺️ 2. The 12-Generation Journey at a Glance

| Generation | Plain English Goal | What Actually Happened | The Law Learned |
|---|---|---|---|
| **v1 → v3** | Can we find the math spark plug among 12,288 experts? | Found **L36/E229** (ablation damage $+1.28$). But router only sends $4.1\%$ of tokens there! | **Routing Frequency $\neq$ Causality** |
| **v4 → v5** | Fine-tuning the causal experts to repair math. | **Matched Reversal**: Accuracy dropped $-2.39\text{ pp}$. Every MoE policy failed! | **Theorem 1: Router Avalanches** |
| **v6 → v8** | Mapping all 48 layers to find smooth control. | Attention sublayers shift smoothly; Math and English share $>90\%$ directional energy. | **Pivot from MoE to Attention LoRA** |
| **v9 → v10** | Testing Standard LoRA on math across 40 layers. | **NLL-Accuracy Paradox**: Step 8 peaked ($79.60\%$); Step 16 collapsed ($77.81\%$). | **Calibrate dose to 8 updates @ 1e-5** |
| **v11** | 42-run double-blind trial of peak layers vs random. | **Gradient guidance falsified** (ranked bot $16.7\%$). **Stratified LoRA WON (+1.48 pp)**! | **Theorem 2: Stratified Hierarchy** |
| **v12** | Building the mathematical safety shield for retained tasks. | Hard projection destroys $99.9\%$ signal. **Soft Riemannian Damping cut drift by up to 88%**! | **Theorems 3 & 4: Riemannian Invariance** |

---

## 📐 3. The 4 Fundamental Theorems

1. **[[02_Theorems/Theorem_1_Discontinuous_MoE_Routing_Bifurcation|Theorem 1 (Discontinuous MoE Routing Bifurcation)]]**:
   * *Why MoE surgery fails*: Continuous weight changes in routed experts cause $\mathcal{O}(1)$ discrete routing avalanches across all 48 downstream layers.
2. **[[02_Theorems/Theorem_2_Jacobian_Condition_Number_Explosion|Theorem 2 (Jacobian Condition Number Explosion)]]**:
   * *Why Stratified placement wins*: Contiguous mid-layer editing causes exponential Jacobian explosion ($\kappa \sim e^{K \sigma_{\max}}$). Spacing edits across early-to-mid spans (`[1, 2, 8, 11, 12, 16, 21, 26]`) keeps condition numbers linear.
3. **[[02_Theorems/Theorem_3_Zero_Power_Collinearity_Paradox|Theorem 3 (The Zero-Power Collinearity Paradox)]]**:
   * *Why hard null-space projection fails*: Math and language share $>99.9\%$ of principal activation dimensions ($3003/3072$). Hard projection destroys $99.9\%$ of the learning gradient.
4. **[[02_Theorems/Theorem_4_Soft_Riemannian_Natural_Gradient_Invariance|Theorem 4 (Soft Riemannian Closed-Form Invariance)]]**:
   * *The winning solution*: Transforming LoRA inputs with forward pre-hook $\tilde{x} = x \cdot (\Sigma_X + \alpha I)^{-1/2}$ evaluates to the exact closed-form Natural Gradient on every AdamW step with **zero extra inference latency**.

---

## 🛠️ 4. How to Reproduce & Run the Protocol in 3 Commands

### Step 1: Clone the Repo
```bash
git clone https://github.com/srishanthsriramula/ReseachX.git
cd ReseachX
```

### Step 2: Open the Pristine v12 Notebook
Open `laguna/laguna_xs2_v12_riemannian_fisher_stratified_lora.ipynb` in VS Code or JupyterLab.

### Step 3: Run the Stratified Riemannian Protocol
Execute cells 1 through 74. The notebook will:
1. Load **Laguna XS.2** in BF16.
2. Collect covariance $\Sigma_X$ across attention projections on retained MBPP prompts.
3. Attach forward pre-hooks ($D_{\alpha=0.01}$) to Stratified Signature 01 (`[1, 2, 8, 11, 12, 16, 21, 26]`).
4. Execute 8 AdamW updates.
5. Generate greedy completions on fresh test items ($N=384$) and verify $>79.4\%$ accuracy with $<0.0006$ drift!

---

## 🧭 5. Knowledge Vault Directory Map

* **`00_Index/`**: Master MOC, Monograph, Academic Treatise, and Handover Guide.
* **`01_Generations/`**: 12 detailed version notes (`v01` to `v12`) + `v13` scaling roadmap.
* **`02_Theorems/`**: First-principles mathematical proofs with step-by-step derivations.
* **`03_Architectures/`**: Model blueprints and layer geometry specifications.
* **`04_Protocols/`**: PyTorch autograd engine and bootstrap statistical testing.
* **`05_Swarm/`**: Multi-agent specialist framework and persona contracts.
