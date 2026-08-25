---
name: swarm-orchestration
description: Adaptive multi-agent research orchestration for difficult problems. Decomposes uncertainty, delegates independent investigations, runs parallel research, detects disagreement, escalates depth, and prevents premature convergence.
---

# SWARM ORCHESTRATION

The purpose of the swarm is not to produce multiple answers.
It is to produce independent evidence that reduces uncertainty.

## CORE RULE

For difficult research problems:

DECOMPOSE
→ IDENTIFY UNCERTAINTIES
→ GENERATE COMPETING HYPOTHESES
→ DELEGATE DISTINCT INVESTIGATIONS
→ RUN INDEPENDENT WORK IN PARALLEL
→ COMBINE FINDINGS
→ DETECT CONTRADICTIONS
→ ATTACK THE LEADING HYPOTHESIS
→ RUN DISCRIMINATING EXPERIMENTS
→ REASSESS
→ ESCALATE IF NECESSARY
→ VERIFY
→ COMPLETE

A plausible first answer is never a reason to stop.

## DEPTH LEVELS

LEVEL 0 — DIRECT
Simple deterministic task.
No swarm.

LEVEL 1 — FOCUSED
One meaningful uncertainty.
Use one specialist.

LEVEL 2 — DEEP
Several independent uncertainties.
Use at least two independent specialists.

LEVEL 3 — RESEARCH
Substantial uncertainty or novelty.
Use:
- theory
- literature
- skeptic
- experiment when applicable

LEVEL 4 — CRITICAL
High-impact research conclusion.
Use:
- theory
- literature
- skeptic
- experiment or reproducer
- adjudicator when disagreement exists

## DELEGATION

Every subagent must receive a distinct question.

Bad:
"Research this topic."

Good:
"Theory: determine whether mechanism X can produce Y under assumptions A and B."

Good:
"Skeptic: identify the strongest counterexample to hypothesis H."

Good:
"Experiment: design the cheapest experiment that distinguishes H1 from H2."

Never create duplicate agents simply to increase the apparent amount of work.

## INDEPENDENCE

When independent evidence is needed, prevent early conclusions from contaminating later investigations.

Prefer:

question
→ independent investigations
→ compare afterward

over:

question
→ agent A concludes X
→ agent B is shown X
→ agent B agrees with X

Agreement based on shared assumptions is not independent confirmation.

## PARALLELISM

Independent investigations should execute concurrently when practical.

Do not serialize tasks that do not depend on each other.

Dependent investigation must wait for the information it requires.

## POST-CYCLE ANALYSIS

After every swarm cycle:

1. Collect findings.
2. Separate agreement from disagreement.
3. Identify assumptions shared across agents.
4. Identify contradictions.
5. Identify unsupported critical claims.
6. Update confidence.
7. Update open questions.
8. Determine the highest-value next investigation.

Do not simply summarize the agents.

Ask:

"What did we learn that changes the research state?"

## ESCALATION

Escalate research depth when:

- confidence remains low
- important evidence is missing
- competing hypotheses survive
- agents disagree
- the strongest counterargument remains unresolved
- a discriminating experiment is available
- new evidence contradicts the leading hypothesis
- the first approach failed
- the conclusion would materially affect research direction

Possible escalation:

new independent agent
→ targeted literature investigation
→ stronger skeptic
→ experiment
→ reproduction
→ adjudication
→ another research cycle

## INFORMATION VALUE

Choose the next action using:

EXPECTED INFORMATION GAIN / COST

Prefer actions that can actually change the conclusion.

Typical priority:

discriminating experiment
>
independent verification
>
strong counterexample search
>
targeted primary-source search
>
new theoretical derivation
>
repetitive reasoning

## DISAGREEMENT

Never resolve disagreement by majority vote.

When agents disagree:

1. Identify exactly where their reasoning diverges.
2. Identify which assumptions differ.
3. Determine what evidence would discriminate between them.
4. Obtain that evidence if practical.
5. Use adjudication when necessary.
6. Update the research state.

## FALSIFICATION

Every major research hypothesis should face an explicit attempt at falsification.

Ask:

- Under what conditions should this fail?
- What observation would disprove it?
- Has that condition been tested?
- Is there contradictory literature?
- Is there an alternative mechanism explaining the same observation?

If the hypothesis survives, record why.

## EXPERIMENTS

Prefer experiments that distinguish hypotheses.

Before execution define:

- hypothesis
- prediction
- variables
- controls
- metric
- expected outcomes
- interpretation

An experiment with no potential to change the research state is low-value.

## ANTI-PREMATURE-CONVERGENCE

Do NOT terminate because:

- the explanation is coherent
- the implementation works once
- one paper supports it
- no counterexample appeared immediately
- multiple agents agree without independent evidence
- the answer is probably correct

Before completion:

"What remaining uncertainty could still change the conclusion?"

If the answer is materially yes:

CONTINUE.

## RESEARCH CYCLES

Treat difficult research as iterative cycles.

Cycle N:
investigate
→ synthesize
→ identify uncertainty
→ choose next action

Cycle N+1:
target the remaining uncertainty.

Continue until:

- high-impact uncertainties are resolved or explicitly irreducible
- important claims are supported
- major alternatives have been considered
- adversarial testing has occurred
- useful experiments have been performed or ruled out
- disagreements are resolved

The goal is not maximum agent count.

The goal is maximum justified confidence.
