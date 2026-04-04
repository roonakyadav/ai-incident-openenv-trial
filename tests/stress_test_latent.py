import sys
import os
import json
import statistics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.core import IncidentEnv
from env.tasks import get_task
from models.schemas import Action, ActionType, ServiceStatus

def format_state(state):
    services = ", ".join([f"{s.name}:{s.status.value}(err:{s.error_rate:.2f})" for s in state.services])
    return f"Health:{state.system_health:.2f} | Stability:{state.system_stability:.2f} | Services: [{services}]"

def run_trace(agent_name, actions_sequence, seed=42):
    print(f"\n=== TRACE: {agent_name} (Seed: {seed}) ===")
    task = get_task("hard-latent-root-cause")
    env = IncidentEnv(task, seed=seed)
    
    state = env.get_state()
    print(f"STEP 0: Initial State -> {format_state(state)}")
    
    total_reward = 0
    for i, action in enumerate(actions_sequence):
        result = env.step(action)
        state = result["state"]
        reward = result["reward"]
        total_reward += reward
        done = result["done"]
        
        print(f"STEP {i+1}: Action({action.action_type.value}, {action.target}) -> Reward: {reward:+.2f}")
        print(f"        State: {format_state(state)}")
        if result['info'].get('message'):
             print(f"        Info: {result['info']['message']}")
        
        # Print logs that happened in this step
        new_logs = state.logs[-3:] # Show last 3 logs
        for log in new_logs:
            print(f"        LOG: {log}")
            
        if done:
            print(f"--- Episode Finished at Step {i+1} ---")
            break
            
    final_grade = env.grader.grade_episode(state, task)
    print(f"FINAL SCORE: {final_grade.final_score:.4f}")
    return final_grade

def test_failure_modes():
    print("\n=== FAILURE MODE ANALYSIS ===")
    task = get_task("hard-latent-root-cause")
    
    # 1. Repeated wrong fixes
    print("\n1. Case: Repeated Wrong Fixes (Loop)")
    env = IncidentEnv(task, seed=42)
    for i in range(5):
        action = Action(action_type=ActionType.RESTART_SERVICE, target="auth")
        result = env.step(action)
        print(f"Step {i+1}: Restart auth -> Timer: {env.fake_recovery_timer}, Stability: {result['state'].system_stability:.2f}")
    
    # 2. Late correct fix
    print("\n2. Case: Late Correct Fix (after re-degradation)")
    env = IncidentEnv(task, seed=42)
    env.step(Action(action_type=ActionType.RESTART_SERVICE, target="auth")) # Step 1: Temp fix
    env.step(Action(action_type=ActionType.CHECK_LOGS, target="none"))      # Step 2: Timer 1
    env.step(Action(action_type=ActionType.CHECK_METRICS, target="none"))   # Step 3: Re-degrade
    print(f"After re-degrade: auth status={env._get_service('auth').status.value}")
    result = env.step(Action(action_type=ActionType.RESTART_SERVICE, target="payments")) # Step 4: Correct fix
    print(f"Step 4: Restart payments -> auth status={env._get_service('auth').status.value}, Score: {env.grader.grade_episode(env.get_state(), task).final_score:.2f}")

def verify_mechanism():
    print("\n=== MECHANISM VERIFICATION ===")
    task = get_task("hard-latent-root-cause")
    env = IncidentEnv(task, seed=42)
    
    # 1. Hidden State Integrity
    state_dict = env.get_state().model_dump()
    leaked = "true_root_cause" in state_dict or "fake_recovery_timer" in state_dict
    print(f"1. Hidden state leaked in observation? {'YES (FAIL)' if leaked else 'NO (PASS)'}")
    
    # 2. No Explits: Repeat restart auth
    print("2. Testing exploit: repeating restart auth during recovery")
    env.step(Action(action_type=ActionType.RESTART_SERVICE, target="auth"))
    print(f"   Timer after first restart: {env.fake_recovery_timer}")
    env.step(Action(action_type=ActionType.RESTART_SERVICE, target="auth"))
    print(f"   Timer after second restart: {env.fake_recovery_timer}")

def agent_diff_test():
    print("\n=== AGENT DIFFERENTIATION TEST ===")
    task = get_task("hard-latent-root-cause")
    
    # Agent 1: Naive (restarts auth)
    env1 = IncidentEnv(task, seed=42)
    env1.step(Action(action_type=ActionType.RESTART_SERVICE, target="auth"))
    env1.step(Action(action_type=ActionType.CHECK_LOGS, target="none"))
    env1.step(Action(action_type=ActionType.CHECK_LOGS, target="none"))
    score1 = env1.grader.grade_episode(env1.get_state(), task).final_score
    
    # Agent 2: Reasoning (restarts payments)
    env2 = IncidentEnv(task, seed=42)
    env2.step(Action(action_type=ActionType.RESTART_SERVICE, target="payments"))
    score2 = env2.grader.grade_episode(env2.get_state(), task).final_score
    
    print(f"Agent 1 (Naive) Score: {score1:.4f}")
    print(f"Agent 2 (Reasoning) Score: {score2:.4f}")
    print(f"Difference: {score2 - score1:.4f}")

def score_distribution():
    print("\n=== SCORE DISTRIBUTION (10 Episodes) ===")
    task = get_task("hard-latent-root-cause")
    scores = []
    for s in range(10):
        env = IncidentEnv(task, seed=s)
        env.step(Action(action_type=ActionType.RESTART_SERVICE, target="payments"))
        scores.append(env.grader.grade_episode(env.get_state(), task).final_score)
    
    print(f"Mean Score: {statistics.mean(scores):.4f}")
    print(f"Variance: {statistics.variance(scores):.4f}")
    print(f"Scores: {[round(s, 4) for s in scores]}")

if __name__ == "__main__":
    # Part 1: Traces
    wrong_actions = [
        Action(action_type=ActionType.RESTART_SERVICE, target="auth"),
        Action(action_type=ActionType.CHECK_LOGS, target="none"),
        Action(action_type=ActionType.CHECK_METRICS, target="none"),
        Action(action_type=ActionType.CHECK_LOGS, target="none")
    ]
    run_trace("WRONG AGENT (Symptom Fixer)", wrong_actions)
    
    correct_actions = [
        Action(action_type=ActionType.RESTART_SERVICE, target="payments"),
        Action(action_type=ActionType.CHECK_LOGS, target="none")
    ]
    run_trace("CORRECT AGENT (Root Cause reasoning)", correct_actions)
    
    # Part 3: Failure Modes
    test_failure_modes()
    
    # Part 4: Verification
    verify_mechanism()
    
    # Part 5: Differentiation
    agent_diff_test()
    
    # Part 6: Distribution
    score_distribution()
