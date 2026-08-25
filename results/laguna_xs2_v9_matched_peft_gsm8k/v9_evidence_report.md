# Laguna XS.2 v9 matched-PEFT report

Protocol hash: `792fd9d4a19c5f37`
Benchmark snapshot SHA256: `8cedb762f6d18ef0dc2f3a828bae64d68da03643238aa2659d93a65a289e361b`

## Budget

Target trainable budget: 12,582,912 parameters (K=4 full routed experts).

## Base GSM8K

Untouched generation-subset accuracy: 0.7969 (n=128).

## Best matched method

`random_experts`: GSM8K exact accuracy 0.7969, accuracy gain +0.0000, target NLL improvement +0.0126.

## Full table

| method                    | family      |   n_runs |   trainable_params |   budget_rel_error |   target_improvement_mean |   target_improvement_std |   control_improvement_mean |   specific_gain_mean |   relative_target_improvement_mean |   relative_specific_gain_mean |   generation_accuracy_mean |   generation_accuracy_std |   generation_accuracy_gain_mean |   peak_gpu_gib_mean |   train_wall_seconds_mean |
|:--------------------------|:------------|---------:|-------------------:|-------------------:|--------------------------:|-------------------------:|---------------------------:|---------------------:|-----------------------------------:|------------------------------:|---------------------------:|--------------------------:|--------------------------------:|--------------------:|--------------------------:|
| random_experts            | full_expert |        3 |        1.25829e+07 |         0          |                 0.0126467 |              0.00154294  |                 0.00943148 |           0.00321519 |                          0.0124184 |                    0.00736047 |                   0.796875 |                 0.0078125 |                      0          |             64.6794 |                   325.045 |
| causal_experts            | full_expert |        3 |        1.25829e+07 |         0          |                 0.103382  |              0.00152145  |                 0.00400635 |           0.0993757  |                          0.101516  |                    0.0993676  |                   0.791667 |                 0.0251137 |                     -0.00520833 |             64.6794 |                   324.179 |
| gradient_specific_experts | full_expert |        3 |        1.25829e+07 |         0          |                 0.18503   |              0.00111993  |                 0.00343919 |           0.181591   |                          0.18169   |                    0.179846   |                   0.789062 |                 0.0206699 |                     -0.0078125  |             64.6794 |                   324.653 |
| routing_experts           | full_expert |        3 |        1.25829e+07 |         0          |                 0.102712  |              0.00276619  |                 0.00374405 |           0.0989682  |                          0.100858  |                    0.0988505  |                   0.78125  |                 0         |                     -0.015625   |             64.6794 |                   324.771 |
| random_layer_lora         | lora        |        3 |        1.25338e+07 |         0.00390625 |                 0.580603  |              0.00159591  |                 0.778203   |          -0.1976     |                          0.570123  |                    0.152787   |                   0.757812 |                 0.0078125 |                     -0.0390625  |             64.6559 |                   323.824 |
| standard_lora             | lora        |        3 |        1.2288e+07  |         0.0234375  |                 0.587594  |              0.000916393 |                 0.785502   |          -0.197908   |                          0.576988  |                    0.155738   |                   0.752604 |                 0.0238676 |                     -0.0442708  |             64.6554 |                   342.815 |
| gradient_guided_lora      | lora        |        3 |        1.26444e+07 |         0.00488281 |                 0.581418  |              0.00329324  |                 0.776194   |          -0.194776   |                          0.570924  |                    0.154665   |                   0.736979 |                 0.0119338 |                     -0.0598958  |             64.6561 |                   324.348 |
| gradient_experts          | full_expert |        3 |        1.25829e+07 |         0          |                 0.347617  |              0.00263691  |                 0.235033   |           0.112583   |                          0.341342  |                    0.215298   |                   0.729167 |                 0.019661  |                     -0.0677083  |             64.6794 |                   324.865 |

## Interpretation guardrails

- Final test was not used for expert/LoRA location selection or LR tuning.
- Standard broad LoRA alone selected one validation LR; all final LoRA location arms share it.
- Expert arms use the v6-v8 established 1e-5 LR, giving LoRA the benefit of a validation LR sweep.
- Expert primary metrics are from the physically merged native BF16 Laguna fused expert kernel.
- MBPP is a teacher-forced collateral-damage control here; this notebook does not claim MBPP execution accuracy.

### gradient_experts
Accuracy 0.7292; target improvement +0.3476; specific gain +0.1126.

### gradient_specific_experts
Accuracy 0.7891; target improvement +0.1850; specific gain +0.1816.

### standard_lora
Accuracy 0.7526; target improvement +0.5876; specific gain -0.1979.

### gradient_guided_lora
Accuracy 0.7370; target improvement +0.5814; specific gain -0.1948.