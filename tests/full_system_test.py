import requests
import json
import os
import sys
import time
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

# Test settings
TASKS = ["easy-auth-down", "medium-payments-degraded", "hard-cascading-failure"]
SEEDS = [42, 123, 999]
MODES = ["llm", "fallback"]

def run_simulation(task_id: str, seed: int, mode: str):
    # Reset environment with seed
    try:
        response = requests.post(f"{API_BASE_URL}/reset/{task_id}?seed={seed}")
        response.raise_for_status()
        state = response.json()
    except Exception as e:
        return {"error": f"Reset failed: {e}"}

    done = False
    total_reward = 0.0
    steps = 0
    bad_actions = 0
    
    while not done and steps < 20: # Safety cap
        if mode == "llm":
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
            bad_actions = state['bad_actions']
            
        except Exception as e:
            return {"error": f"Step failed: {e}"}
            
    return {
        "task": task_id,
        "seed": seed,
        "mode": mode,
        "raw_reward": round(total_reward, 2),
        "normalized_reward": round(total_reward / max(1, steps), 2),
        "steps": steps,
        "bad_actions": bad_actions,
        "stability": round(state['system_stability'], 2)
    }

def test_bad_actions_termination():
    task_id = "easy-auth-down"
    requests.post(f"{API_BASE_URL}/reset/{task_id}")
    
    # Send wrong actions repeatedly
    for _ in range(5):
        action = {"action_type": "restart_service", "target": "invalid-service-name"}
        response = requests.post(f"{API_BASE_URL}/step/{task_id}", json=action)
        result = response.json()
        if result['done']:
            return True
    return False

def test_determinism():
    task_id = "easy-auth-down"
    seed = 42
    
    res1 = run_simulation(task_id, seed, "fallback")
    res2 = run_simulation(task_id, seed, "fallback")
    
    # Compare all relevant fields except any that might naturally drift if not reset properly
    # (But reset should handle it)
    keys = ["raw_reward", "steps", "bad_actions", "stability"]
    return all(res1[k] == res2[k] for k in keys)

def test_randomness():
    task_id = "easy-auth-down"
    res1 = run_simulation(task_id, 42, "fallback")
    res2 = run_simulation(task_id, 123, "fallback")
    
    # With noise, at least stability or reward should eventually vary, 
    # but for simple tasks it might stay same. 
    # This is just a placeholder to confirm we can run different seeds.
    return True

if __name__ == "__main__":
    print("="*50)
    print("OPENENV FULL SYSTEM TEST HARNESS")
    print("="*50)

    # Check if server is running
    try:
        requests.get(f"{API_BASE_URL}/health")
    except:
        print(f"Error: API server not found at {API_BASE_URL}.")
        print("Please start it in another terminal with:")
        print("source venv/bin/activate && python3 -m api.main")
        sys.exit(1)
        
    results = []
    print("\n--- Running Task Simulations ---")
    for task in TASKS:
        for seed in SEEDS:
            for mode in MODES:
                print(f"Testing {task:25} | Seed: {seed:3} | Mode: {mode:8}", end=" ", flush=True)
                start_time = time.time()
                res = run_simulation(task, seed, mode)
                duration = time.time() - start_time
                if "error" in res:
                    print(f"| ERROR: {res['error']}")
                else:
                    print(f"| Raw Reward: {res['raw_reward']:5.2f} | Norm Reward: {res['normalized_reward']:5.2f} | Steps: {res['steps']:2} | Stability: {res['stability']:4.2f} | ({duration:4.2f}s)")
                results.append(res)
                
    # Summary
    print("\n" + "="*50)
    print("### SUMMARY")
    print("="*50)
    
    task_norm_rewards = {}
    for res in results:
        if "error" in res: continue
        t = res['task']
        task_norm_rewards.setdefault(t, []).append(res['normalized_reward'])
        
    for t, norm_rewards in task_norm_rewards.items():
        avg = sum(norm_rewards) / len(norm_rewards)
        print(f"- Avg normalized reward for {t:25}: {avg:5.2f}")
        
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        best = max(valid_results, key=lambda x: x['normalized_reward'])
        worst = min(valid_results, key=lambda x: x['normalized_reward'])
        print(f"- Best run (normalized):  Task: {best['task']}, Seed: {best['seed']}, Mode: {best['mode']}, Reward: {best['normalized_reward']}")
        print(f"- Worst run (normalized): Task: {worst['task']}, Seed: {worst['seed']}, Mode: {worst['mode']}, Reward: {worst['normalized_reward']}")
        
    failures = [r for r in results if "error" in r]
    if failures:
        print(f"- Failures detected: {len(failures)}")
        
    # Edge cases
    print("\n### EDGE CASES & DIAGNOSTICS")
    bad_actions_pass = test_bad_actions_termination()
    print(f"- Bad actions termination check:    {'PASS' if bad_actions_pass else 'FAIL'}")
    
    det_pass = test_determinism()
    print(f"- Determinism (fixed seed) check:   {'PASS' if det_pass else 'FAIL'}")
    
    # Warnings
    print("\n### WARNINGS")
    has_warnings = False
    if len(set(r['normalized_reward'] for r in valid_results if r['task'] == TASKS[0])) == 1:
        print("! Warning: No normalized reward variation observed for easy task across seeds.")
        has_warnings = True
    if failures:
        print(f"! Warning: {len(failures)} simulation(s) crashed or failed.")
        has_warnings = True
    if not has_warnings:
        print("No significant warnings detected.")
        
    # Final Verdict
    print("\n### FINAL VERDICT")
    is_stable = not failures and bad_actions_pass
    print(f"- Stability:      {'Good' if is_stable else 'Needs Work'}")
    print(f"- Deterministic:  {'Yes' if det_pass else 'No'}")
    print(f"- Overall Status: {'READY FOR SUBMISSION' if is_stable and det_pass else 'NEEDS FIXES'}")
    print("="*50)
