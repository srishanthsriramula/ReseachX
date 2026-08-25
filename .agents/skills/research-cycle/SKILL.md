---
name: research-cycle
description: Executes repeated research cycles with mandatory synthesis, uncertainty tracking, adversarial review, state updates, and escalation until a defensible stopping condition is reached.
---

# RESEARCH CYCLE PROTOCOL

A difficult research problem is not one task.
It is a sequence of evidence-producing cycles.

## CYCLE 0 — ORIENTATION

Before investigating:

1. Read research_state/state.md.
2. Read research_state/hypotheses.md.
3. Read research_state/evidence.md.
4. Read research_state/open_questions.md.
5. Read research_state/failed_approaches.md.
6. Define the precise research question.
7. Identify the decision or conclusion the research must support.
8. Identify assumptions already being made.

Do not start by writing the final answer.

## CYCLE 1 — EXPLORATION

Build an initial model of the problem.

Determine:
- major hypotheses
- unknowns
- evidence requirements
- likely failure modes
- relevant literature
- possible experiments

For difficult tasks, use independent specialists.

Do not allow the initial leading hypothesis to become unquestioned truth.

## CYCLE 2+ — TARGETED INVESTIGATION

After the first synthesis, identify the most consequential remaining uncertainty.

Then choose the action with the highest expected information gain.

Possible actions:
- targeted literature investigation
- theory derivation
- counterexample search
- independent reproduction
- discriminating experiment
- adversarial review
- adjudication

Do NOT simply repeat the previous research.

Each new cycle must target a specific remaining uncertainty.

## MANDATORY POST-CYCLE SYNTHESIS

After specialists return:

1. Compare conclusions.
2. Identify agreement.
3. Identify disagreement.
4. Identify shared assumptions.
5. Identify contradictory evidence.
6. Identify unsupported claims.
7. Identify failed approaches.
8. Update confidence.
9. Identify what remains unknown.
10. Select the highest-value next action.

Do not merely summarize agent outputs.

The important question is:

"What changed in the research state?"

## STATE UPDATE REQUIREMENT

After every meaningful cycle update:

research_state/state.md
research_state/hypotheses.md
research_state/evidence.md
research_state/open_questions.md
research_state/failed_approaches.md
research_state/decisions.md
research_state/research_log.md
research_state/controller.json

Research state must survive conversation truncation and agent replacement.

## EVIDENCE DISCIPLINE

For important claims distinguish:

ESTABLISHED
Multiple strong independent evidence sources or direct experiment.

STRONGLY SUPPORTED
Substantial evidence with limited unresolved uncertainty.

PLAUSIBLE
Reasonable explanation but insufficient verification.

UNCERTAIN
Material evidence is missing.

CONTRADICTED
Important evidence conflicts with the claim.

DISPROVEN
Evidence or experiment rules it out.

Never silently promote a plausible claim to established.

## ADVERSARIAL CYCLE

Before a major conclusion:

1. State the strongest version of the hypothesis.
2. State the strongest competing hypothesis.
3. Ask the skeptic to attack the preferred hypothesis.
4. Identify what observation would discriminate them.
5. Obtain that observation when practical.

If the preferred hypothesis survives, record WHY.

## EXPERIMENTAL CYCLE

When an experiment can resolve uncertainty:

1. Define hypothesis.
2. Define prediction.
3. Define controls.
4. Define metric.
5. Define expected outcomes.
6. Execute.
7. Record actual outcome.
8. Explain discrepancy.
9. Update research state.

Never use experiments merely as a demonstration.

## ESCALATION

Continue into another cycle when ANY is true:

- a high-impact uncertainty remains
- competing hypotheses remain viable
- an important claim lacks independent support
- agents disagree materially
- the strongest counterargument remains unanswered
- a discriminating experiment remains available
- important external evidence has not been reproduced
- new evidence contradicts the current model

## COMPLETION

A difficult research task may finish only when:

- the question is precise
- the leading conclusion is explicit
- important alternatives were investigated
- major claims have evidence
- adversarial testing occurred
- useful experiments were run or ruled out
- disagreements are resolved
- failed approaches are documented
- remaining uncertainty is explicit
- another reasonable research cycle is unlikely to materially change the conclusion

Do not use "I think this is enough" as a stopping criterion.

Use evidence and expected information gain.

## CORE PRINCIPLE

The research system should spend effort where uncertainty is highest.

The goal is not:

"produce an answer."

The goal is:

"produce a conclusion that survives serious attempts to disprove it."
