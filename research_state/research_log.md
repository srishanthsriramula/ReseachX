# RESEARCH LOG

- **2026-08-14 10:58**: Initialized Laguna S2.1 architectural capability analysis.
- **2026-08-14 15:30**: Infrastructure failures on Kaggle T4x2 (vLLM FP8 incompatibilities, `DecompressExperts`).
- **2026-08-14 18:50**: Migrated to RTX PRO 6000 96GB direct BF16 path.
- **2026-08-14 19:30**: Discovered E229 in Layer 36 for Frontend CSS capability.
- **2026-08-14 20:00**: Conducted matched adaptation experiment; observed the Critical Reversal (E229 rigidity).
- **2026-08-14 22:30**: Executed v6 forced-access experiment; falsified Routing Blocker hypothesis.
- **2026-08-15 15:30**: Completed v7.1 Global Writeability Atlas across all 9,984 experts ($R \approx 0.82$).
- **2026-08-15 17:30**: Completed v8.2 Cross-Capability Replication (Frontend, Python, Math).
- **2026-08-16 09:40**: Executed v9 matched PEFT on GSM8K; discovered the NLL vs Accuracy Paradox.
- **2026-08-16 17:45**: Conducted v10 behavior-aligned dose experiment; identified contrastive norm collapse.
- **2026-08-17 08:45**: Executed v11 confirmatory test; proved Guided LoRA fails to beat random LoRA distribution.
- **2026-08-17 12:00**: Conducted Root-Cause Reassessment across 24 hypotheses.
- **2026-08-17 14:30**: Formulated Behavioral Repair Kernel, Margin Gradients, Null-Space Fisher Projection, and Gemma 2 2B IT protocol.
- **2026-08-24 19:55**: Research Director complete onboarding and state memory reconstruction.


### [2026-08-25 01:17:31 UTC] v11 Final Confirmation Results Ingested
- Extracted and audited  executed on AMD Instinct MI300X.
- Confirmed Primary B result: Guided LoRA failed to outperform the matched random layer distribution (Guided 78.21% vs Random 78.85%, diff -0.64 pp, 95% CI [-0.0299, +0.0161]).
- Falsified the scalar Gradient-Guided Layer Placement hypothesis.
- Discovered that early/mid distributed random placements (e.g.  [1, 2, 8, 11, 12, 16, 21, 26]) achieve superior math reasoning gains (+1.48 pp, 79.60%) by avoiding high-gradient bottleneck congestion.
