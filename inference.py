import os
import sys
import json
import requests
from typing import List, Dict, Any, Optional
import openai
from dotenv import load_dotenv

load_dotenv()

# --- Fix 4: Environment variable handling ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
ENV_URL = os.getenv("ENV_URL", "https://roonakyadav-ai-incident-openenv-trial.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
SUCCESS_SCORE_THRESHOLD = 0.6
MAX_STEPS = 15

# Use OpenAI-compatible API if HF_TOKEN is present
if HF_TOKEN:
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=HF_TOKEN,
            base_url=API_BASE_URL
        )
    except ImportError:
        client = None
else:
    client = None

# --- Mapping for LLM vs ActionType enum ---
ACTION_MAPPING = {
    "scale_service": "scale",
    "scale": "scale",
    "restart_service": "restart_service",
    "rollback_service": "rollback_service",
    "isolate_service": "isolate_service",
    "check_logs": "check_logs",
    "check_metrics": "check_metrics",
    "drain_traffic": "drain_traffic",
    "restore_traffic": "restore_traffic",
    "optimize_db": "optimize_db",
    "escalate": "escalate",
    "ignore": "ignore"
}

VALID_ACTIONS = [
    "restart_service", "rollback_service", "isolate_service", "check_logs", 
    "check_metrics", "scale_service", "drain_traffic", "restore_traffic", 
    "optimize_db", "escalate"
]

def get_model_message(state: Dict[str, Any], task_objective: str, last_action: Optional[Dict[str, Any]], last_reward: float) -> str:
    """Fix 3: LLM prompt must include all observable state in plaintext format."""
    step = state.get("time_step", 0)
    
    services_text = "\n".join([
        f"  {s['name']}: {s['status'].upper()} | latency={s['latency']:.0f}ms | error_rate={s['error_rate']:.2f}"
        for s in state.get("services", [])
    ])
    
    logs = state.get("logs", [])
    recent_logs = "\n".join([f"  {log}" for log in logs[-5:]]) if logs else "  No recent logs."
    
    alerts = "\n".join([f"  {alert}" for alert in state.get("alerts", [])]) if state.get("alerts") else "  No active alerts."
    
    last_action_str = "None"
    if last_action:
        target = last_action.get("target", "none")
        last_action_str = f"{last_action['action_type']}/{target} → reward: {last_reward:+.2f}"
    
    available_actions = "restart_service, rollback_service, isolate_service, check_logs, check_metrics, scale_service, drain_traffic, restore_traffic, optimize_db, escalate"
    
    prompt = f"""--- INCIDENT RESPONSE TASK ---
Objective: {task_objective}
Step: {step} / {MAX_STEPS}

As a senior SRE, your goal is to restore all services to a healthy state. Analyze the service status, logs, and alerts to identify the root cause and take action. Prioritize actions that resolve the underlying issue. Available actions are: `restart_service`, `rollback_service`, `isolate_service`, `check_logs`, `check_metrics`, `scale_service`, `drain_traffic`, `restore_traffic`, `optimize_db`, `escalate`.

**Analyze the following data and provide the best next action to resolve the incident. If you are stuck, try a different action. Do not repeat the same action twice in a row.**

SERVICES:
{services_text}

RECENT LOGS (last 5):
{recent_logs}

ALERTS:
{alerts}

LAST ACTION: {last_action_str}

What is your next action? Reply with ONLY a JSON object:
{{"action_type": "...", "target": "..."}}
If the action has no target, omit the target field or set it to null."""
    return prompt

def llm_decide_action(state: Dict[str, Any], task_objective: str, last_action: Optional[Dict[str, Any]], last_reward: float) -> Dict[str, Any]:
    if not client:
        print("Error: OpenAI client is not available. Please check your API key and network connection.")
        sys.exit(1)

    if last_action and last_action['action_type'] == 'check_logs':
        last_reward -= 0.5

    prompt = get_model_message(state, task_objective, last_action, last_reward)
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a senior SRE agent. Respond only with the requested JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling LLM: {e}")
        # Fallback to a safe action
        return {"action_type": "check_logs", "target": "none"}
    
    print(f"LLM response: {content}")
    
    # Parse JSON
    try:
        if "{" in content and "}" in content:
            content = content[content.find("{"):content.rfind("}")+1]
        action_data = json.loads(content)
    except Exception as e:
        print(f"Error parsing JSON from LLM: {e}. Content: {content}")
        return {"action_type": "check_logs", "target": "none"}
    
    # Map and validate action
    action_type = action_data.get("action_type", "ignore")
    mapped_action_type = ACTION_MAPPING.get(action_type, "ignore")

    return {
        "action_type": mapped_action_type,
        "target": action_data.get("target") or "none"
    }

def run_episode(task_id: str):
    """Run a full episode for a given task ID with Fix 2 strict log format."""
    # Print START log
    print(f"[START] task={task_id} env=ai-incident-openenv model={MODEL_NAME}")
    
    # Reset environment
    try:
        response = requests.post(f"{ENV_URL}/reset/{task_id}")
        if response.status_code != 200:
            print(f"Error resetting environment for task {task_id}: {response.text}")
            response.raise_for_status()
        state = response.json()
        
        # Get task objective from state if available or use a default
        task_objective = "Restore all services to UP state with minimal error rate."
        if "cascading-ambiguous" in task_id:
            task_objective = "Restore all services to UP with error_rate < 0.1"
        elif "bad-deployment" in task_id:
            task_objective = "Roll back the faulty deployment to restore service stability."
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API server at {ENV_URL}. Is it running?")
        print(f"[END] success=false steps=0 score=0.00 rewards=")
        return {"task_id": task_id, "score": 0.0, "steps": 0, "success": False}
    except Exception as e:
        print(f"Unexpected error in reset for task {task_id}: {e}")
        print(f"[END] success=false steps=0 score=0.00 rewards=")
        return {"task_id": task_id, "score": 0.0, "steps": 0, "success": False}

    done = False
    steps = 0
    rewards_list = []
    last_action = None
    last_reward = 0.0
    final_score = 0.0
    
    while not done and steps < MAX_STEPS:
        # Decide action
        action = llm_decide_action(state, task_objective, last_action, last_reward)
        
        # Fix 2: [STEP] format
        step_num = steps + 1
        action_type = action["action_type"]
        target = action.get("target", "none")
        if target is None: target = "none"
        
        try:
            response = requests.post(f"{ENV_URL}/step/{task_id}", json=action)
            response.raise_for_status()
            result = response.json()
            
            state = result['state']
            reward = result['reward']
            done = result['done']
            error_msg = "null"
            
            # Fix 1: Read final_score from score_breakdown when done=True
            if done:
                try:
                    score_breakdown = result.get("info", {}).get("score_breakdown")
                    if score_breakdown:
                        final_score = score_breakdown.get("final_score", 0.0)
                    else:
                        # Fallback: final_score = sum(rewards) / task_max_steps
                        task_max_steps = state.get("max_steps", MAX_STEPS) if isinstance(state, dict) else MAX_STEPS
                        final_score = sum(rewards_list + [reward]) / max(task_max_steps, 1)
                except Exception:
                    task_max_steps = state.get("max_steps", MAX_STEPS) if isinstance(state, dict) else MAX_STEPS
                    final_score = sum(rewards_list + [reward]) / max(task_max_steps, 1)
            
            rewards_list.append(reward)
            last_action = action
            last_reward = reward
            steps += 1
            
            # Print STEP log
            print(f"[STEP] step={step_num} action={action_type}/{target} reward={reward:+.2f} done={'true' if done else 'false'} error={error_msg}")
            
        except Exception as e:
            error_msg = str(e).replace(",", " ")
            print(f"[STEP] step={step_num} action={action_type}/{target} reward=0.00 done=true error={error_msg}")
            done = True
            # Fallback score on error
            if not rewards_list:
                final_score = 0.0
            else:
                task_max_steps = state.get("max_steps", MAX_STEPS) if isinstance(state, dict) else MAX_STEPS
                final_score = sum(rewards_list) / max(task_max_steps, 1)
            
    # Get final score from grader 
    try: 
        grade_resp = requests.post(f"{ENV_URL}/grade/{task_id}") 
        if grade_resp.status_code == 200: 
            graded_score = grade_resp.json().get("final_score", None) 
            if graded_score is not None: 
                final_score = max(0.0, min(1.0, float(graded_score))) 
    except Exception: 
        pass 

    success = final_score >= SUCCESS_SCORE_THRESHOLD
    
    # Print END log
    rewards_str = ",".join([f"{r:+.2f}" for r in rewards_list])
    print(f"[END] success={'true' if success else 'false'} steps={steps} score={final_score:.2f} rewards={rewards_str}")
    
    return {
        "task_id": task_id,
        "score": final_score,
        "steps": steps,
        "success": success
    }

def main():
    import argparse

    # Run all tasks sequentially
    task_ids = ["easy-auth-down", "medium-payments-degraded", "hard-cascading-failure", "hard-bad-deployment", "hard-cascading-ambiguous", "hard-latent-root-cause"]
    results = [run_episode(task_id) for task_id in task_ids]



if __name__ == "__main__":
    main()
