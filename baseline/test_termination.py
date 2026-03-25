import requests
import json

API_BASE_URL = "http://localhost:8000"
TASK_ID = "hard-cascading-failure"

def run_simulation(actions):
    requests.post(f"{API_BASE_URL}/reset/{TASK_ID}")
    
    final_state = None
    for i, action in enumerate(actions):
        response = requests.post(f"{API_BASE_URL}/step/{TASK_ID}", json=action)
        try:
            state = response.json()
            print(f"Step {i+1}: Action: {action['action_type']}, Target: {action['target']}, Done: {state['done']}")
            if state["done"]:
                final_state = state
                break
        except json.JSONDecodeError:
            # Handle cases where the response may be empty
            print(f"Step {i+1}: Action: {action['action_type']}, Target: {action['target']}, Done: True (Empty Response)")
            break
    
    if not final_state:
        final_state = requests.get(f"{API_BASE_URL}/state/{TASK_ID}").json()

    response = requests.post(f"{API_BASE_URL}/grade/{TASK_ID}")
    return response.json(), final_state

def main():
    # Scenario 1: Agent does nothing
    print("--- Scenario 1: Agent does nothing ---")
    actions = [{"action_type": "ignore", "target": "none"}] * 5
    score, state = run_simulation(actions)
    print("Final Score:", json.dumps(score, indent=2))
    print("Termination reason: Exceeded max_steps")

    # Scenario 2: Agent does wrong actions repeatedly
    print("\n--- Scenario 2: Agent does wrong actions repeatedly ---")
    actions = [
        {"action_type": "restart_service", "target": "auth"}, 
        {"action_type": "scale", "target": "payments"}, 
        {"action_type": "restart_service", "target": "frontend"}
    ]
    score, state = run_simulation(actions)
    print("Final Score:", json.dumps(score, indent=2))

    # Scenario 3: Correct agent
    print("\n--- Scenario 3: Correct agent ---")
    actions = [{"action_type": "optimize_db", "target": "db"}]
    score, state = run_simulation(actions)
    print("Final Score:", json.dumps(score, indent=2))
    print("Termination reason: All services up")

if __name__ == "__main__":
    main()