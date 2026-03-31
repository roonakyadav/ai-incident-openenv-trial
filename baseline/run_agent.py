import requests
import json
import os
from typing import List, Dict, Any
from openai import OpenAI

# Environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Initialize OpenAI client
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

def run_task(task_id: str, agent_type: str = "llm"):
    print(f"\n--- Running task: {task_id} with {agent_type} agent ---")
    
    # Reset environment
    try:
        response = requests.post(f"{API_BASE_URL}/reset/{task_id}")
        response.raise_for_status()
        state = response.json()
    except Exception as e:
        print(f"Error connecting to environment: {e}. Is the API running at {API_BASE_URL}?")
        return 0.0, False

    done = False
    total_reward = 0.0
    fallback_triggered = False
    
    while not done:
        print(f"\nStep {state['time_step']}:")
        print(f"Services: {json.dumps(state['services'], indent=2)}")
        print(f"Logs: {state['logs'][-2:] if state['logs'] else []}")
        print(f"Alerts: {state['alerts']}")
        
        if agent_type == "llm":
            action, triggered = llm_decide_action(state)
            if triggered:
                fallback_triggered = True
        else:
            action = rule_based_decide_action(state)
            
        print(f"Action: {action}")
        
        # Step in environment
        try:
            response = requests.post(f"{API_BASE_URL}/step/{task_id}", json=action)
            response.raise_for_status()
            result = response.json()
            
            state = result['state']
            reward = result['reward']
            done = result['done']
            total_reward += reward
            
            print(f"Reward: {reward}")
        except Exception as e:
            print(f"Error during step: {e}")
            break
        
    print(f"\nTask {task_id} finished. Total reward: {total_reward:.2f}")
    return total_reward, fallback_triggered

def rule_based_decide_action(state: Dict[str, Any]) -> Dict[str, Any]:
    # A simple rule-based agent to demonstrate the environment interaction
    for service in state['services']:
        if service['status'] == 'down':
            return {"action_type": "restart_service", "target": service['name']}
        if service['status'] == 'degraded':
            return {"action_type": "scale", "target": service['name']}
    
    return {"action_type": "ignore", "target": "none"}

def llm_decide_action(state: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
    """
    Decide action using LLM with fallback to rule-based logic.
    Returns (action, fallback_triggered)
    """
    # Define valid actions and targets for the prompt and validation
    valid_action_types = ["restart_service", "scale", "optimize_db", "check_logs", "ignore"]
    valid_targets = [s['name'] for s in state['services']] + ["none"]

    system_prompt = (
        "You are an SRE agent solving incident response tasks. "
        "Your goal is to fix the root cause of issues, avoid unnecessary actions, and minimize cost. "
        "You MUST return the next action in JSON format exactly as follows: "
        '{"action_type": "...", "target": "..."}\n\n'
        f"Allowed action_type: {', '.join(valid_action_types)}\n"
        f"Allowed target: {', '.join(valid_targets)}"
    )
    
    # Construct state representation for prompt
    services_info = "\n".join([
        f"- {s['name']}: status={s['status']}, latency={s.get('latency', 'N/A')}, error_rate={s.get('error_rate', 'N/A')}"
        for s in state['services']
    ])
    recent_logs = "\n".join(state['logs'][-5:]) if state['logs'] else "No recent logs."
    alerts = ", ".join(state['alerts']) if state['alerts'] else "No active alerts."
    
    user_prompt = f"""Current System State:
Services:
{services_info}

Recent Logs:
{recent_logs}

Active Alerts:
{alerts}

What is the best next action to resolve the issues?
"""

    try:
        # Check if the model supports JSON mode (Groq models might vary)
        extra_args = {}
        if "gpt" in MODEL_NAME.lower():
            extra_args["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2,
            **extra_args
        )
        
        content = response.choices[0].message.content.strip()
        
        # Robust parsing for potential Markdown blocks or whitespace
        if content.startswith("```"):
            # Extract content from markdown code block
            parts = content.split("```")
            if len(parts) >= 3:
                content = parts[1]
                if content.startswith("json"):
                    content = content[4:].strip()
                else:
                    content = content.strip()
        
        # In case of leading/trailing junk
        if "{" in content and "}" in content:
            content = content[content.find("{"):content.rfind("}")+1]

        action = json.loads(content)
        
        # Basic validation of LLM output
        if "action_type" in action and "target" in action:
            # Strict validation against allowed values
            if action["action_type"] in valid_action_types:
                return action, False
            else:
                print(f"LLM returned invalid action_type: {action['action_type']}, falling back.")
                return rule_based_decide_action(state), True
        else:
            print(f"LLM returned invalid structure: {action}, falling back to rules.")
            return rule_based_decide_action(state), True
            
    except Exception as e:
        print(f"LLM call failed or parsing error: {e}. Falling back to rule-based logic.")
        return rule_based_decide_action(state), True

if __name__ == "__main__":
    task_id = "easy-auth-down"
    
    # Run Rule-based agent
    rule_reward, _ = run_task(task_id, agent_type="rule")
    
    # Run LLM-based agent
    llm_reward, fallback_triggered = run_task(task_id, agent_type="llm")
    
    print("\n" + "="*40)
    print("FINAL REPORT")
    print("="*40)
    print(f"Task: {task_id}")
    print(f"Rule-based Reward: {rule_reward:.2f}")
    print(f"LLM-based Reward:  {llm_reward:.2f}")
    print(f"LLM Valid Actions: {'Yes (mostly)' if not fallback_triggered else 'No (fallback triggered)'}")
    print(f"Fallback Triggered: {fallback_triggered}")
    print(f"Total Reward Comparison: {'LLM outperformed' if llm_reward > rule_reward else 'Rule-based outperformed' if rule_reward > llm_reward else 'Both performed equally'}")
    print("="*40)

