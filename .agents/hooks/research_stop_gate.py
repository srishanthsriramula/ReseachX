#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        print(json.dumps({"decision": "allow"}))
        return

    paths = payload.get("workspacePaths") or []
    if not paths:
        print(json.dumps({"decision": "allow"}))
        return

    state_file = Path(paths[0]) / "research_state" / "controller.json"

    try:
        state = json.loads(state_file.read_text())
    except Exception:
        print(json.dumps({"decision": "allow"}))
        return

    if state.get("completion_allowed") is True:
        print(json.dumps({"decision": "allow"}))
        return

    level = int(state.get("level", 0) or 0)
    if level <= 0:
        print(json.dumps({"decision": "allow"}))
        return

    reasons = []

    if state.get("critical_uncertainties"):
        reasons.append("critical uncertainties remain")

    if state.get("evidence_gaps"):
        reasons.append("evidence gaps remain")

    if state.get("agent_disagreements"):
        reasons.append("agent disagreements remain")

    if level >= 2 and int(state.get("falsification_attempts", 0) or 0) < 1:
        reasons.append("no falsification attempt recorded")

    if level >= 3 and int(state.get("experiments_completed", 0) or 0) < 1 and not state.get("experiment_not_applicable", False):
        reasons.append("no discriminating experiment recorded")

    if reasons:
        print(json.dumps({
            "decision": "continue",
            "reason": "RESEARCH STOP GATE: " + "; ".join(reasons) +
                      ". Continue investigating and update controller.json."
        }))
    else:
        print(json.dumps({"decision": "allow"}))

if __name__ == "__main__":
    main()
