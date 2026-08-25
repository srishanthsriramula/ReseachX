#!/usr/bin/env python3
"""
Adaptive Research Loop Controller.

This controller does not solve the research problem itself.
It converts research state into the next required investigation.

State is stored in:
research_state/controller.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = ROOT / "research_state" / "controller.json"
STATE_MD = ROOT / "research_state" / "state.md"
QUESTIONS_MD = ROOT / "research_state" / "open_questions.md"
HYPOTHESES_MD = ROOT / "research_state" / "hypotheses.md"


def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def n(state: dict, key: str) -> int:
    value = state.get(key, [])
    return len(value) if isinstance(value, list) else int(value or 0)


def choose_level(state: dict) -> int:
    """
    Adaptive depth score.

    We intentionally bias upward for difficult research.
    The objective is not efficiency first; it is avoiding premature convergence.
    """

    confidence = state.get("confidence")
    novelty = state.get("novelty")
    consequence = state.get("consequence")
    uncertainty = state.get("uncertainty")

    score = 0

    if confidence is None:
        score += 2
    elif isinstance(confidence, (int, float)):
        if confidence < 0.60:
            score += 2
        elif confidence < 0.80:
            score += 1

    for value in (novelty, consequence, uncertainty):
        if isinstance(value, str):
            value = value.lower()
            if value in {"high", "very high", "critical"}:
                score += 2
            elif value == "medium":
                score += 1

    score += min(n(state, "critical_uncertainties"), 3)
    score += min(n(state, "competing_hypotheses"), 2)
    score += min(n(state, "evidence_gaps"), 2)
    score += min(n(state, "agent_disagreements"), 2)

    if int(state.get("research_cycles", 0) or 0) == 0:
        score += 1

    if score >= 8:
        return 4
    if score >= 5:
        return 3
    if score >= 3:
        return 2
    if score >= 1:
        return 1
    return 0


def plan_for_level(level: int, state: dict) -> dict:
    """
    Produce an explicit investigation plan.
    """

    if level == 0:
        return {
            "level": 0,
            "mode": "direct",
            "agents": [],
            "actions": [
                "Solve directly.",
                "Do not manufacture unnecessary depth."
            ]
        }

    if level == 1:
        return {
            "level": 1,
            "mode": "focused",
            "agents": ["one appropriate specialist"],
            "actions": [
                "Identify the single highest-impact uncertainty.",
                "Run one focused investigation.",
                "Reassess before stopping."
            ]
        }

    if level == 2:
        return {
            "level": 2,
            "mode": "deep",
            "agents": ["theory", "literature"],
            "actions": [
                "Run theory and literature investigations independently.",
                "Compare findings.",
                "Identify contradictions and evidence gaps."
            ]
        }

    if level == 3:
        return {
            "level": 3,
            "mode": "research",
            "agents": ["theory", "literature", "skeptic", "experiment"],
            "actions": [
                "Run independent investigations in parallel where possible.",
                "Attempt explicit falsification.",
                "Run a discriminating experiment when practical.",
                "Perform a second synthesis cycle after results return."
            ]
        }

    return {
        "level": 4,
        "mode": "critical",
        "agents": [
            "theory",
            "literature",
            "skeptic",
            "experiment",
            "reproducer",
            "adjudicator"
        ],
        "actions": [
            "Run independent investigations in parallel.",
            "Require adversarial challenge.",
            "Independently reproduce important external claims.",
            "Use adjudication for disagreement.",
            "Do not complete until high-impact uncertainties are resolved."
        ]
    }


def next_action(state: dict, plan: dict) -> str:
    if n(state, "agent_disagreements") > 0:
        return "adjudicate disagreements"

    if n(state, "critical_uncertainties") > 0:
        return "target the highest-impact unresolved uncertainty"

    if n(state, "evidence_gaps") > 0:
        return "perform targeted evidence acquisition"

    if n(state, "competing_hypotheses") > 1:
        return "run a discriminating investigation or experiment"

    if plan["level"] >= 3 and int(state.get("falsification_attempts", 0) or 0) == 0:
        return "perform explicit falsification attempt"

    if plan["level"] >= 3 and int(state.get("experiments_completed", 0) or 0) == 0:
        return "design or execute a discriminating experiment"

    return "perform independent verification of the leading conclusion"


def completion_check(state: dict) -> tuple[bool, list[str]]:
    reasons = []

    if int(state.get("level", 0) or 0) == 0:
        return True, reasons

    if n(state, "critical_uncertainties") > 0:
        reasons.append("critical uncertainties remain")

    if n(state, "evidence_gaps") > 0:
        reasons.append("evidence gaps remain")

    if n(state, "agent_disagreements") > 0:
        reasons.append("agent disagreements remain")

    level = int(state.get("level", 0) or 0)

    if level >= 2 and int(state.get("falsification_attempts", 0) or 0) < 1:
        reasons.append("no falsification attempt recorded")

    if (
        level >= 3
        and int(state.get("experiments_completed", 0) or 0) < 1
        and not state.get("experiment_not_applicable", False)
    ):
        reasons.append("no discriminating experiment recorded")

    if int(state.get("independent_validations", 0) or 0) < 1 and level >= 2:
        reasons.append("no independent validation recorded")

    return len(reasons) == 0, reasons


def render_state(state: dict, plan: dict, action: str, allowed: bool, reasons: list[str]) -> None:
    lines = [
        "# CURRENT RESEARCH STATE",
        "",
        f"## Research Level",
        str(plan["level"]),
        "",
        f"## Mode",
        plan["mode"],
        "",
        "## Confidence",
        str(state.get("confidence")),
        "",
        "## Critical Uncertainties",
        str(n(state, "critical_uncertainties")),
        "",
        "## Competing Hypotheses",
        str(n(state, "competing_hypotheses")),
        "",
        "## Evidence Gaps",
        str(n(state, "evidence_gaps")),
        "",
        "## Agent Disagreements",
        str(n(state, "agent_disagreements")),
        "",
        "## Research Cycles",
        str(state.get("research_cycles", 0)),
        "",
        "## Required Agents",
    ]

    for agent in plan["agents"]:
        lines.append(f"- {agent}")

    lines += [
        "",
        "## Required Actions",
    ]

    for item in plan["actions"]:
        lines.append(f"- {item}")

    lines += [
        "",
        "## Highest-Value Next Action",
        action,
        "",
        "## Completion Allowed",
        str(allowed),
        "",
        "## Completion Blockers",
    ]

    for reason in reasons:
        lines.append(f"- {reason}")

    STATE_MD.write_text("\n".join(lines) + "\n")


def main() -> int:
    state = load_state()

    level = choose_level(state)
    state["level"] = level

    plan = plan_for_level(level, state)
    action = next_action(state, plan)

    allowed, reasons = completion_check(state)

    # Research director must explicitly earn completion.
    state["completion_allowed"] = allowed
    state["next_action"] = action

    save_state(state)
    render_state(state, plan, action, allowed, reasons)

    output = {
        "research_level": level,
        "mode": plan["mode"],
        "agents": plan["agents"],
        "actions": plan["actions"],
        "next_action": action,
        "completion_allowed": allowed,
        "completion_blockers": reasons,
        "research_cycles": state.get("research_cycles", 0),
    }

    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
