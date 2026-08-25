# Laguna XS.2 v11 confirmation report

Protocol hash: `75482335de3ffac7`
Fresh snapshot SHA256: `96b439d19505c1c895ed5535cef4653eade325e93091c1629bc4a458babef6e5`

## Mandatory v10 interpretation correction

v10 raw-gradient and contrastive expert selectors were identical. v11 therefore tests one frozen writable-expert location and makes no contrastive-vs-raw expert selector claim.

## Primary A — writable expert behavioral replication

Mean accuracy gain: N/A (Skipped)
95% CI: N/A
Prespecified confirmation criterion met: SKIPPED (Historical v10 negative)

## Primary B — guided LoRA placement vs matched random placement

Guided fixed8 mean accuracy: 0.7821
Random-placement mean accuracy: 0.7885
Guided minus random mean: -0.0064
95% hierarchical placement/seed/example bootstrap CI: [-0.0299, +0.0161]
Prespecified confirmation criterion met: False

## Guardrails

- No v11 final data were used for selector, LR, rank, dose, placement, or seed selection.
- Random placements match the guided layer-shape signature histogram.
- Guided and random placement use exactly the same trainable parameter count.
- Guided/random placement use identical LR, rank, update count, training rows, and common seeds.
- Positive means are descriptive unless the prespecified CI criterion is met.
- This remains one model/checkpoint and one target benchmark.