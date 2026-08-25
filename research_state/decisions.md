# RESEARCH DECISIONS

- **Decision 01 (Aug 14)**: Adopt RTX PRO 6000 96GB (`g7e.2xlarge`) as primary hardware platform for uncompressed BF16 execution.
- **Decision 02 (Aug 14)**: Enforce exact aligned full-logit cross-entropy scoring and require `</think>\n` formatting check.
- **Decision 03 (Aug 15)**: Abandon the Read-Write Equivalence assumption following Exp 06 reversal.
- **Decision 04 (Aug 16)**: Forbid teacher-forced NLL as primary success criterion for reasoning benchmarks; require autonomous generation accuracy.
- **Decision 05 (Aug 17)**: Terminate unprojected scalar gradient selector search following v11 replication failure.
- **Decision 06 (Aug 17)**: Pivot primary research program to Failure-Conditioned Controllable Repair Subspaces with Null-Space Fisher Projection.
- **Decision 07 (Aug 17)**: Implement the Gemma 2 2B IT dense model experiment to isolate the core algebra of repair geometry.
