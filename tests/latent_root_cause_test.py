import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.core import IncidentEnv
from env.tasks import get_task
from models.schemas import Action, ActionType, ServiceStatus

def test_latent_root_cause_mechanism():
    print("Testing Hidden Root Cause + Delayed Consequence mechanism...")
    task = get_task("hard-latent-root-cause")
    if not task:
        print("Error: Task hard-latent-root-cause not found")
        return

    # Scenario 1: Fix the wrong target (surface symptom)
    print("\n--- Scenario 1: Fix surface symptom (auth) ---")
    env = IncidentEnv(task, seed=42)
    state = env.get_state()
    print(f"Initial State: auth status={state.services[0].status}, payments status={state.services[1].status}")
    
    # Step 1: Restart auth (wrong target, surface symptom)
    action = Action(action_type=ActionType.RESTART_SERVICE, target="auth")
    result = env.step(action)
    state = result["state"]
    print(f"Step 1: Restart auth. Info type: {result['info']['type']}")
    print(f"State after Step 1: auth status={state.services[0].status}, fake_recovery_timer={env.fake_recovery_timer}")
    
    # Step 2: Diagnostic action
    action = Action(action_type=ActionType.CHECK_LOGS, target="none")
    result = env.step(action)
    state = result["state"]
    print(f"Step 2: Check logs. Info type: {result['info']['type']}")
    print(f"State after Step 2: auth status={state.services[0].status}, fake_recovery_timer={env.fake_recovery_timer}")
    
    # Step 3: Diagnostic action -> re-degradation should occur here
    action = Action(action_type=ActionType.CHECK_METRICS, target="none")
    result = env.step(action)
    state = result["state"]
    print(f"Step 3: Check metrics. Info type: {result['info']['type']}")
    print(f"State after Step 3: auth status={state.services[0].status}, fake_recovery_timer={env.fake_recovery_timer}")
    
    found_redegrade_log = any("service degraded again due to unresolved underlying dependency" in log for log in state.logs)
    print(f"Re-degradation log found: {found_redegrade_log}")
    
    if state.services[0].status == ServiceStatus.DEGRADED and found_redegrade_log:
        print("SUCCESS: Delayed re-degradation occurred as expected")
    else:
        print("FAILURE: Delayed re-degradation did not occur as expected")

    # Scenario 2: Fix the true root cause directly
    print("\n--- Scenario 2: Fix true root cause (payments) ---")
    env = IncidentEnv(task, seed=42)
    
    # Step 1: Restart payments (true root cause)
    action = Action(action_type=ActionType.RESTART_SERVICE, target="payments")
    result = env.step(action)
    state = result["state"]
    print(f"Step 1: Restart payments. Info type: {result['info']['type']}")
    print(f"State after Step 1: auth status={state.services[0].status}, payments status={state.services[1].status}, fake_recovery_timer={env.fake_recovery_timer}")
    
    if state.services[0].status == ServiceStatus.UP and state.services[1].status == ServiceStatus.UP and env.fake_recovery_timer is None:
        print("SUCCESS: True root cause fix worked correctly")
    else:
        print("FAILURE: True root cause fix did not work as expected")

    # Scenario 3: Fix symptom, then fix root cause before timer expires
    print("\n--- Scenario 3: Fix symptom, then fix root cause before timer expires ---")
    env = IncidentEnv(task, seed=42)
    
    # Step 1: Restart auth (wrong target, surface symptom)
    action = Action(action_type=ActionType.RESTART_SERVICE, target="auth")
    result = env.step(action)
    print(f"Step 1: Restart auth. fake_recovery_timer={env.fake_recovery_timer}")
    
    # Step 2: Restart payments (true root cause)
    action = Action(action_type=ActionType.RESTART_SERVICE, target="payments")
    result = env.step(action)
    print(f"Step 2: Restart payments. fake_recovery_timer={env.fake_recovery_timer}")
    
    if env.fake_recovery_timer is None:
        print("SUCCESS: True root cause fix cancelled the fake recovery timer")
    else:
        print("FAILURE: True root cause fix did NOT cancel the fake recovery timer")

if __name__ == "__main__":
    test_latent_root_cause_mechanism()
