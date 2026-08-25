#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        print(json.dumps({}))
        return

    paths = payload.get("workspacePaths") or []
    if not paths:
        print(json.dumps({}))
        return

    state_file = Path(paths[0]) / "research_state" / "controller.json"

    try:
        state = json.loads(state_file.read_text())
    except Exception:
        state = {}

    message = f"""
RESEARCH DEPTH CONTROLLER — LEVEL {state.get("level", 0)}

Do not optimize for shortest-path completion.

Confidence: {state.get("confidence")}
Critical uncertainties: {len(state.get("critical_uncertainties", []))}
Competing hypotheses: {len(state.get("competing_hypotheses", []))}

Before acting:
1. Identify the highest-impact uncertainty.
2. Prefer the action that most reduces that uncertainty.
3. Do not treat a plausible first solution as completion.
4. Consider alternative explanations.
5. Use independent investigation, literature, tools, adversarial testing, or experiments when they can materially change the conclusion.
6. Continue when the conclusion could still materially change.

You are being optimized for research quality, not speed.
"""

    print(json.dumps({
        "injectSteps": [
            {
                "ephemeralMessage": message
            }
        ]
    }))

if __name__ == "__main__":
    main()
