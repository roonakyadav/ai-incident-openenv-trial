import requests
import json
import os
import sys
from typing import List, Dict, Any

# Ensure we can import from the root
sys.path.append(os.getcwd())

# Import baseline agent logic
try:
    from baseline.run_agent import rule_based_decide_action, llm_decide_action, API_BASE_URL
except ImportError:
    # Fallback if imports fail
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
    def rule_based_decide_action(state): return {"action_type": "ignore", "target": "none"}
    def llm_decide_action(state): return {"action_type": "ignore", "target": "none"}, False

def run_inference(task_id: str, agent_type: str = "llm"):
    """Function to run a single task with a specified agent type."""
    print(f"\n--- Running inference for task: {task_id} with {agent_type} agent ---")
    
    # Reset environment
    try:
        response = requests.post(f"{API_BASE_URL}/reset/{task_id}")
        response.raise_for_status()
        state = response.json()
    except Exception as e:
        print(f"Error connecting to environment: {e}. Is the API running at {API_BASE_URL}?")
        return 0.0, 0.0

    done = False
    total_reward = 0.0
    steps = 0
    
    while not done and steps < 20: # Safety cap
        if agent_type == "llm":
            action, _ = llm_decide_action(state)
        else:
            action = rule_based_decide_action(state)
            
        try:
            response = requests.post(f"{API_BASE_URL}/step/{task_id}", json=action)
            response.raise_for_status()
            result = response.json()
            
            state = result['state']
            reward = result['reward']
            done = result['done']
            
            total_reward += reward
            steps += 1
            
        except Exception as e:
            print(f"Error during step: {e}")
            break
            
    # Get final score from grader
    try:
        response = requests.post(f"{API_BASE_URL}/grade/{task_id}")
        response.raise_for_status()
        final_score = response.json().get("final_score", 0.0)
    except Exception as e:
        print(f"Error getting final grade: {e}")
        final_score = 0.0

    print(f"Task {task_id} finished. Total reward: {total_reward:.2f}, Final Score: {final_score:.2f}")
    return total_reward, final_score

if __name__ == "__main__":
    # This script is meant to be called with a task_id, but we can run a default for testing
    task_id = "easy-auth-down"
    if len(sys.argv) > 1:
        task_id = sys.argv[1]
        
    # Run with LLM agent by default
    run_inference(task_id, agent_type="llm")
