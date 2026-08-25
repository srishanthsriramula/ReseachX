# Laguna XS.2 v10 behavior-aligned writeability report

Protocol hash: `6448478bbc7880aa`
Fresh snapshot SHA256: `37fe52c5e03cb38a2fb3e17dc8e8f6018ff1f5519515381b32423c64bb56d12d`
v9 exclusion SHA256: `eba2f16fd8d1e5cc7af43b6504258372e34521f5418c393b8e51001b9e748eef`

## Freshness

- Every v9-used GSM8K/MBPP row was reconstructed from the pinned datasets and excluded.
- v9/v10 example-ID overlap: **0**.
- Fresh final GSM8K rows: **192**.
- Fresh final MBPP controls: **192**.

## Behavioral calibration

- Expert-family LR: `1e-05`.
- LoRA-family LR: `1e-05`.
- Candidate doses: `[0, 1, 2, 4, 8, 16, 32, 64]`.
- LR selection and dose selection used autonomous GSM8K accuracy first.
- `abs(control_improvement)` was the first tie-breaker, so broad control movement is never rewarded.

## Frozen doses

- `gradient_experts`: 0 optimizer updates.
- `gradient_specific_experts`: 8 optimizer updates.
- `contrastive_experts`: 8 optimizer updates.
- `standard_lora`: 16 optimizer updates.
- `random_layer_lora`: 8 optimizer updates.
- `contrastive_lora`: 4 optimizer updates.

## Fresh final

Base GSM8K exact accuracy: **0.7604**.

| method                    | family      |   n_runs |   selected_updates |    lr |   trainable_params |   generation_accuracy_mean |   generation_accuracy_std |   generation_accuracy_gain_mean |   target_improvement_mean |   control_abs_shift_mean |   selective_target_gain_mean |   parameter_delta_l2_mean |   train_wall_seconds_mean |   paired_mean_accuracy_gain |   paired_two_way_ci_low |   paired_two_way_ci_high |   positive_seed_count |   nonnegative_seed_count |
|:--------------------------|:------------|---------:|-------------------:|------:|-------------------:|---------------------------:|--------------------------:|--------------------------------:|--------------------------:|-------------------------:|-----------------------------:|--------------------------:|--------------------------:|----------------------------:|------------------------:|-------------------------:|----------------------:|-------------------------:|
| random_layer_lora         | lora        |        3 |                  8 | 1e-05 |        1.25338e+07 |                   0.78125  |                0.0238676  |                      0.0208333  |                0.00463442 |               0.0045104  |                  0.000124017 |                 0.134298  |                   41.2637 |                  0.0208333  |              -0.0260417 |                0.0659722 |                     2 |                        3 |
| contrastive_experts       | full_expert |        3 |                  8 | 1e-05 |        1.25829e+07 |                   0.779514 |                0.0131074  |                      0.0190972  |                0.0579759  |               0.0338345  |                  0.0241414   |                 0.174897  |                   40.7595 |                  0.0190972  |              -0.0295139 |                0.0677083 |                     3 |                        3 |
| contrastive_lora          | lora        |        3 |                  4 | 1e-05 |        1.26444e+07 |                   0.777778 |                0.0131074  |                      0.0173611  |                0.00223966 |               0.00481176 |                 -0.0025721   |                 0.0709213 |                   20.6461 |                  0.0173611  |              -0.0225694 |                0.0590278 |                     3 |                        3 |
| gradient_specific_experts | full_expert |        3 |                  8 | 1e-05 |        1.25829e+07 |                   0.765625 |                0.0090211  |                      0.00520833 |                0.0363997  |               0.00370836 |                  0.0326914   |                 0.150977  |                   40.8437 |                  0.00520833 |              -0.03125   |                0.0416667 |                     1 |                        3 |
| standard_lora             | lora        |        3 |                 16 | 1e-05 |        1.2288e+07  |                   0.762153 |                0.00795586 |                      0.00173611 |                0.00213699 |               0.00578725 |                 -0.00365027  |                 0.2555    |                   87.5543 |                  0.00173611 |              -0.0399306 |                0.0451389 |                     1 |                        2 |
| gradient_experts          | full_expert |        3 |                  0 | 1e-05 |        1.25829e+07 |                   0.760417 |                0          |                      0          |                0          |               0          |                  0           |                 0         |                    0      |                  0          |               0         |                0         |                     0 |                        3 |

## Pairwise frozen-policy comparisons

| method_a            | method_b                  |   mean_accuracy_difference |   bootstrap_ci_low |   bootstrap_ci_high |
|:--------------------|:--------------------------|---------------------------:|-------------------:|--------------------:|
| contrastive_experts | gradient_experts          |                 0.0190972  |         -0.0295139 |           0.0677083 |
| contrastive_experts | gradient_specific_experts |                 0.0138889  |         -0.03125   |           0.0625    |
| contrastive_lora    | standard_lora             |                 0.015625   |         -0.0295139 |           0.0572917 |
| contrastive_lora    | random_layer_lora         |                -0.00347222 |         -0.046875  |           0.0416667 |

## Pre-registered interpretation

A method is a strong behavioral success only if its fresh-final mean accuracy gain is positive and its paired two-way bootstrap interval excludes zero.
Teacher-forced target NLL remains secondary: v9 showed that large NLL movement can coexist with worse autonomous reasoning.
Control movement is reported as an absolute shift because either direction indicates a less task-specific intervention.

### contrastive_experts
Mean accuracy gain: +0.0191; 95% paired two-way bootstrap CI [-0.0295, +0.0677].
Strong behavioral success criterion: **NOT MET**.

### contrastive_lora
Mean accuracy gain: +0.0174; 95% paired two-way bootstrap CI [-0.0226, +0.0590].
Strong behavioral success criterion: **NOT MET**.