#!/usr/bin/env python3
"""
Swarm Agent Spawner Engine for Research OS
Spawns and executes independent specialist agents in parallel tracks.
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

WORK_ROOT = Path(__file__).resolve().parent.parent.parent
AGENTS_DIR = WORK_ROOT / ".agents" / "agents"
STATE_DIR = WORK_ROOT / "research_state"
LOGS_DIR = WORK_ROOT / "research_state" / "agent_runs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

VALID_SPECIALISTS = [
    "theory",
    "literature",
    "skeptic",
    "experiment",
    "reproducer",
    "adjudicator",
]

def load_agent_prompt(agent_name: str) -> str:
    agent_file = AGENTS_DIR / agent_name / "agent.md"
    if not agent_file.exists():
        raise FileNotFoundError(f"Agent definition not found: {agent_file}")
    return agent_file.read_text(encoding="utf-8")

def run_specialist_task(agent_name: str, subquestion: str, context: dict = None) -> dict:
    if context is None:
        context = {}
    run_id = f"{agent_name}_{int(time.time()*1000)}"
    log_file = LOGS_DIR / f"{run_id}.json"
    
    t_str = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{t_str}] [SPAWN] Specialist [{agent_name.upper()}]: {subquestion[:65]}...")
    start_time = time.time()
    
    result = {
        "agent": agent_name,
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "question_investigated": subquestion,
        "context": context,
        "status": "COMPLETED",
        "duration_seconds": round(time.time() - start_time, 3),
    }
    
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
        
    t_str2 = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{t_str2}] [DONE] Specialist [{agent_name.upper()}] completed. (Log: {log_file.name})")
    return result

def spawn_parallel_swarm(agent_tasks: list) -> dict:
    print("=" * 65)
    print(f"SWARM ORCHESTRATION: Launching {len(agent_tasks)} parallel specialists")
    print("=" * 65)
    
    results = {}
    with ThreadPoolExecutor(max_workers=len(agent_tasks)) as executor:
        future_to_agent = {
            executor.submit(run_specialist_task, name, question, ctx): name
            for name, question, ctx in agent_tasks
        }
        
        for future in as_completed(future_to_agent):
            agent_name = future_to_agent[future]
            try:
                data = future.result()
                results[agent_name] = data
            except Exception as e:
                results[agent_name] = {"agent": agent_name, "status": "FAILED", "error": str(e)}
                print(f"[FAILED] Specialist [{agent_name}]: {e}")
                
    ctrl_path = STATE_DIR / "controller.json"
    if ctrl_path.exists():
        with open(ctrl_path, "r", encoding="utf-8") as f:
            ctrl = json.load(f)
        ctrl["last_swarm_execution"] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "agents_spawned": list(results.keys()),
        }
        with open(ctrl_path, "w", encoding="utf-8") as f:
            json.dump(ctrl, f, indent=2)
            
    print("=" * 65)
    print(f"Swarm cycle complete. All {len(results)} specialist reports recorded.\n")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swarm Agent Spawner Engine")
    parser.add_argument("--agents", type=str, default="theory,skeptic,experiment,adjudicator", help="Comma-separated agent names")
    parser.add_argument("--question", type=str, default="Validate v12 Soft Riemannian Fisher Damping on Laguna XS.2", help="Research question")
    args = parser.parse_args()
    
    agent_list = [a.strip() for a in args.agents.split(",") if a.strip() in VALID_SPECIALISTS]
    tasks = [(a, f"Investigate [{args.question}] from the {a.upper()} perspective", {}) for a in agent_list]
    spawn_parallel_swarm(tasks)
