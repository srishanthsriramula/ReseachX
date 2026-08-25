# NOTEBOOK REGISTRY

Detailed registry of all Jupyter notebooks in the Laguna research repository.

| Notebook Filename | Target Hardware | Execution State | Key Purpose & Outputs | Lessons & Status |
|---|---|---|---|---|
| `laguna_xs2_expert_atlas_kaggle_t4x2.ipynb` | Kaggle 2×T4 (32GB) | Unexecuted | Initial attempt at 9,984-expert routing atlas. | OOM / Sharding bottleneck. Superseded. |
| `laguna_xs2_causal_vllm_t4x2.ipynb` (v1–v4) | Kaggle 2×T4 (32GB) | Unexecuted | Attempted vLLM fast batched causal sweeps. | vLLM FP8 KV-cache incompatibility. Abandoned. |
| `laguna_xs2_causal_atlas_L40_...` (v6–v9) | NVIDIA L40S (48GB) | Unexecuted | Transformers loading with custom code. | Exposed `DecompressExperts` autograd bug. Superseded. |
| `laguna_xs2_final_causal_surgery_RTX_PRO_6000_... (1).ipynb` | RTX PRO 6000 (96GB) | Executed (25 outputs) | First successful BF16 direct load. 8s causal sweep. | Established 96GB BF16 platform. Valid. |
| `laguna_xs2_causal_surgery_g7e_2xlarge_v2_scoringfix (2).ipynb` | AWS g7e.2xlarge | Executed | Scored fixed-routing causal sweep; found E229. | Discovered E229 (L36/E229 $\Delta\text{NLL} = +1.2858$). Valid. |
| `laguna_xs2_causal_surgery_g7e_2xlarge_v3_matched_baselines (3).ipynb` | AWS g7e.2xlarge | Executed | Matched adaptation of E229 vs routed vs random. | The Critical Reversal: E229 lost to routed expert. Valid. |
| `laguna_xs2_causal_surgery_g7e_2xlarge_v5_clean_end_to_end (2).ipynb` | AWS g7e.2xlarge | Executed | Clean end-to-end replication of causal reversal. | Confirmed causal expert rigidity. Valid. |
| `laguna_xs2_v6_falsification_grade (1).ipynb` | AWS g7e.2xlarge | Executed (68 outputs) | Forced-access training on E229. | Falsified Routing Blocker Hypothesis. Valid. |
| `laguna_xs2_v7_1_global_writeability_atlas_fixed (3).ipynb` | AWS g7e.2xlarge | Executed (289 outputs) | Complete 9,984-expert atlas of gradients vs plasticity. | Established $R \approx 0.82$ gradient-plasticity correlation. Valid. |
| `laguna_xs2_v8_2_1b_cross_capability_full_checkpoint_fixed (1).ipynb` | AWS g7e.2xlarge | Executed (714 outputs) | Cross-capability replication (Frontend, Python, Math). | Validated population correlation across domains. Valid. |
| `laguna_xs2_v9_matched_peft_gsm8k (1).ipynb` | AWS g7e.2xlarge | Executed (103 outputs) | Matched PEFT on GSM8K (Writable Experts vs LoRA). | The NLL vs Accuracy Paradox. Valid. |
| `laguna_xs2_v10_behavior_aligned_writeability (1).ipynb` | AWS g7e.2xlarge | Executed (104 outputs) | Behavior dose calibration & contrastive LoRA. | Apparent $+0.047$ win; flawed contrastive projection. Validated flaw. |
| `laguna_xs2_v11_fresh_confirmation_random_placement.ipynb` | AWS g7e.2xlarge | Frozen / Protocol | Confirmatory test against random LoRA distribution. | Replication failed (CI crosses zero). Definitively closed scalar search. |
|  | AMD Instinct MI300X (192GB HBM3) | Production Ready | Optimized v11 confirmatory matrix: hf_transfer multi-worker downloads, ROCm HIP tuning, batch size scaled to 32, SDPA attention. | Full protocol fidelity with ~1.0h execution time (down from ~12h on RTX 6000). |


### 12. `laguna_xs2_v12_riemannian_fisher_stratified_lora.ipynb`
* **Protocol**: `v12.0-riemannian-fisher-stratified-lora`
* **Target Hardware**: AMD Instinct MI300X (ROCm) / NVIDIA RTX PRO 6000
* **Core Investigation**: Soft Riemannian Fisher Damping on Stratified Layer Geometries (`[1, 2, 8, 11, 12, 16, 21, 26]`).
* **Design**: Pre-computes retained activation covariance Sigma_X on MBPP/General text, applies right-sided Riemannian pre-conditioning to LoRA adapters, traces Pareto frontier across damping scales alpha in [10^-3, 10^-2, 10^-1], and evaluates fresh GSM8K (N=384) math reasoning gains with zero inference overhead.
