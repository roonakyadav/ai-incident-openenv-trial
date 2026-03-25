import requests
import json
import os
from typing import List, Dict, Any

API_BASE_URL = "http://localhost:8000"

def run_task(task_id: str):
    print(f"--- Running task: {task_id} ---")
    
    # Reset environment
    response = requests.post(f"{API_BASE_URL}/reset/{task_id}")
    state = response.json()
    
    done = False
    total_reward = 0.0
    
    while not done:
        print(f"\nStep {state['time_step']}:")
        print(f"Services: {json.dumps(state['services'], indent=2)}")
        print(f"Logs: {state['logs'][-2:] if state['logs'] else []}")
        print(f"Alerts: {state['alerts']}")
        
        # Simple rule-based agent for baseline (could be replaced with LLM call)
        action = decide_action(state)
        print(f"Action: {action}")
        
        # Step in environment
        response = requests.post(f"{API_BASE_URL}/step/{task_id}", json=action)
        result = response.json()
        
        state = result['state']
        reward = result['reward']
        done = result['done']
        total_reward += reward
        
        print(f"Reward: {reward}")
        
    print(f"\nTask {task_id} finished. Total reward: {total_reward:.2f}")

def decide_action(state: Dict[str, Any]) -> Dict[str, Any]:
    # A simple rule-based agent to demonstrate the environment interaction
    for service in state['services']:
        if service['status'] == 'down':
            return {"action_type": "restart_service", "target": service['name']}
        if service['status'] == 'degraded':
            return {"action_type": "scale", "target": service['name']}
    
    return {"action_type": "ignore", "target": "none"}

if __name__ == "__main__":
    # Test with easy task
    try:
        run_task("easy-auth-down")
    except Exception as e:
        print(f"Error connecting to environment: {e}. Is the API running?")
