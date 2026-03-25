import requests
import json

API_BASE_URL = "http://localhost:8000"
TASK_ID = "hard-cascading-failure"

def run_simulation(actions):
    requests.post(f"{API_BASE_URL}/reset/{TASK_ID}")
    
    for action in actions:
        response = requests.post(f"{API_BASE_URL}/step/{TASK_ID}", json=action)
        try:
            if response.json().get("done"):
                break
        except json.JSONDecodeError:
            # Handle cases where the response may be empty
            pass
            
    response = requests.post(f"{API_BASE_URL}/grade/{TASK_ID}")
    return response.json()

def main():
    # Scenario 1: Perfect run
    perfect_actions = [
        {"action_type": "optimize_db", "target": "db"}
    ]
    perfect_score = run_simulation(perfect_actions)
    print("--- Perfect Run ---")
    print(json.dumps(perfect_score, indent=2))

    # Scenario 2: Wrong actions
    wrong_actions = [
        {"action_type": "restart_service", "target": "auth"}, # Wrong fix
        {"action_type": "scale", "target": "payments"}, # Wrong fix
        {"action_type": "optimize_db", "target": "db"} # Correct fix
    ]
    wrong_score = run_simulation(wrong_actions)
    print("\n--- Wrong Actions Run ---")
    print(json.dumps(wrong_score, indent=2))

    # Scenario 3: Inefficient run
    inefficient_actions = [
        {"action_type": "check_logs", "target": "auth"},
        {"action_type": "check_logs", "target": "payments"},
        {"action_type": "check_logs", "target": "db"},
        {"action_type": "optimize_db", "target": "db"}
    ]
    inefficient_score = run_simulation(inefficient_actions)
    print("\n--- Inefficient Run ---")
    print(json.dumps(inefficient_score, indent=2))

if __name__ == "__main__":
    main()