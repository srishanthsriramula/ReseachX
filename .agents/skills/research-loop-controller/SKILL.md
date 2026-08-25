---
name: research-loop-controller
description: Adaptive research control loop that determines investigation depth, selects specialist subagents, drives repeated research cycles, detects unresolved uncertainty, and prevents premature completion.
---

# RESEARCH LOOP CONTROLLER

The controller determines WHAT should happen next.

The Research Director determines HOW to execute it using Antigravity's native agent collaboration tools.

## CONTROL LOOP

1. Run:
   python3 .agents/controller/research_loop.py

2. Inspect:
   - level
   - critical uncertainties
   - evidence gaps
   - competing hypotheses
   - disagreements
   - next action

3. Select the minimum set of specialist agents that can materially reduce those uncertainties.

4. Invoke independent specialists concurrently using native `invoke_subagent`.

5. Compare returned results.

6. Record:
   - evidence
   - disagreements
   - failures
   - confidence
   - new questions

7. Update controller.json.

8. Run the controller again.

9. Continue another research cycle whenever the controller says completion is not justified.

## SPECIALIST SELECTION

Level 0:
No subagent.

Level 1:
One appropriate specialist.

Level 2:
Theory + Literature.

Level 3:
Theory + Literature + Skeptic + Experiment when empirical discrimination is possible.

Level 4:
Theory + Literature + Skeptic + Experiment/Reproducer + Adjudicator when required.

## PRINCIPLE

The objective is not maximum agent count.

The objective is maximum justified uncertainty reduction.

A first plausible answer is an input to the next research cycle, not a stopping condition.
