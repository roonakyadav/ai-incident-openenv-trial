import sys
import os
import json
sys.path.append(os.getcwd())

from env.core import IncidentEnv
from env.tasks import get_task
from models.schemas import Action, ActionType

def print_step(step_num, res):
    info = res['info']
    print(f"STEP {step_num}:")
    print(f"action = {res['action'].action_type} {res['action'].target}")
    print(f"reward = {res['reward']:.2f}")
    print(f"done = {res['done']}")
    metrics = {
        "symptom_fixes": info.get("symptom_fixes"),
        "root_cause_step": info.get("root_cause_step"),
        "delayed_failures": info.get("delayed_failures"),
        "wasted_actions": info.get("wasted_actions")
    }
    print(f"metrics = {json.dumps(metrics, indent=2)}")
    print("-" * 20)

def run_naive_agent():
    print("\n" + "="*40)
    print("CASE A: NAIVE AGENT (Symptom Chaser)")
    print("="*40)
    task = get_task("hard-latent-root-cause")
    env = IncidentEnv(task)
    
    # Naive behavior:
    # 1. Check logs (streak 1)
    # 2. Check logs (streak 2)
    # 3. Check logs (streak 3)
    # 4. Check logs (streak 4 -> wasted +1)
    # 5. Fix symptom 'auth' (symptom_fix +1, sets delay)
    # 6. Check metrics (streak 1)
    # 7. Check metrics (streak 2, wait for delay -> delayed_failure +1)
    # 8. Fix symptom 'auth' again (symptom_fix +2, repeat -> wasted +1)
    
    actions = [
        Action(action_type=ActionType.CHECK_LOGS, target="auth"),
        Action(action_type=ActionType.CHECK_LOGS, target="auth"),
        Action(action_type=ActionType.CHECK_LOGS, target="auth"),
        Action(action_type=ActionType.CHECK_LOGS, target="auth"),
        Action(action_type=ActionType.RESTART_SERVICE, target="auth"),
        Action(action_type=ActionType.CHECK_METRICS, target="frontend"),
        Action(action_type=ActionType.CHECK_METRICS, target="frontend"),
        Action(action_type=ActionType.RESTART_SERVICE, target="auth"),
        Action(action_type=ActionType.ESCALATE, target="all") # End episode
    ]
    
    for i, action in enumerate(actions):
        res = env.step(action)
        res['action'] = action
        print_step(i + 1, res)
        if res['done']:
            break
            
    final_state = env.get_state()
    episode_result = env.grader.grade_episode(final_state, task)
    
    print("\nDIAGNOSTIC SUMMARY:")
    print(f"- symptom_fixes: {final_state.symptom_fix_count}")
    print(f"- root_cause_step: {final_state.root_cause_step}")
    print(f"- delayed_failures: {final_state.delayed_failure_count}")
    print(f"- wasted_actions: {final_state.wasted_action_count}")
    print(f"- failure_type: {episode_result.failure_type}")

def run_reasoning_agent():
    print("\n" + "="*40)
    print("CASE B: REASONING AGENT (Efficient Reasoner)")
    print("="*40)
    task = get_task("hard-latent-root-cause")
    env = IncidentEnv(task)
    
    # Reasoning behavior:
    # 1. Target root cause immediately (root_cause_step = 1)
    # 2. Correct fix (no symptom fixes, no delay)
    
    actions = [
        Action(action_type=ActionType.CHECK_LOGS, target="payments"),
        Action(action_type=ActionType.ROLLBACK_SERVICE, target="payments")
    ]
    
    for i, action in enumerate(actions):
        res = env.step(action)
        res['action'] = action
        print_step(i + 1, res)
        if res['done']:
            break
            
    final_state = env.get_state()
    episode_result = env.grader.grade_episode(final_state, task)
    
    print("\nDIAGNOSTIC SUMMARY:")
    print(f"- symptom_fixes: {final_state.symptom_fix_count}")
    print(f"- root_cause_step: {final_state.root_cause_step}")
    print(f"- delayed_failures: {final_state.delayed_failure_count}")
    print(f"- wasted_actions: {final_state.wasted_action_count}")
    print(f"- failure_type: {episode_result.failure_type}")

if __name__ == "__main__":
    run_naive_agent()
    run_reasoning_agent()
