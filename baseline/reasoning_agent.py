import requests
import json
from typing import Dict, Any, List

API_BASE_URL = "http://localhost:8000"

def decide_action(state: Dict[str, Any]) -> Dict[str, Any]:
    logs = state.get("logs", [])
    services = state.get("services", [])

    # Rule 1: Check for database-related keywords
    db_keywords = ["database", "db", "connection timeout"]
    if any(keyword in log.lower() for log in logs for keyword in db_keywords):
        db_service = next((s for s in services if s["name"] == "db"), None)
        if db_service and db_service["status"] == "degraded":
            return {"action_type": "optimize_db", "target": "db"}

    # Rule 2: Check for services that are down
    for service in services:
        if service["status"] == "down":
            return {"action_type": "restart_service", "target": service["name"]}

    # Rule 3: Check for services that are degraded
    for service in services:
        if service["status"] == "degraded":
            return {"action_type": "scale", "target": service["name"]}

    # Default action
    return {"action_type": "ignore", "target": "none"}

def run_task(task_id: str):
    print(f"--- Running task: {task_id} ---")
    
    response = requests.post(f"{API_BASE_URL}/reset/{task_id}")
    state = response.json()
    
    done = False
    while not done:
        action = decide_action(state)
        print(f"Step {state['time_step']}: Action: {action['action_type']}, Target: {action['target']}")
        
        response = requests.post(f"{API_BASE_URL}/step/{task_id}", json=action)
        try:
            state = response.json()
            done = state["done"]
        except json.JSONDecodeError:
            print("Error decoding JSON from response. Assuming episode is done.")
            done = True

    response = requests.post(f"{API_BASE_URL}/grade/{task_id}")
    score = response.json()
    print("Final Score:", json.dumps(score, indent=2))

def main():
    run_task("easy-auth-down")
    print("\n")
    run_task("medium-payments-degraded")
    print("\n")
    run_task("hard-cascading-failure")

if __name__ == "__main__":
    main()