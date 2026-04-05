import subprocess
import time
import requests
import sys
import os

BASE_URL = "http://localhost:7860"
TASK_IDS = [
    "easy-auth-down",
    "medium-payments-degraded",
    "hard-cascading-failure",
    "hard-bad-deployment",
    "hard-cascading-ambiguous",
    "hard-latent-root-cause"
]

def print_report(name, result, expected=None, actual=None):
    status = "PASS" if result else "FAIL"
    report = f"[{status}] {name}"
    if expected is not None or actual is not None:
        report += f" (Expected: {expected}, Actual: {actual})"
    print(report)
    return result

def main():
    print("Starting server...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Wait for server to boot
    time.sleep(3)
    
    all_passed = True
    
    try:
        # 1. TEST /health endpoint
        try:
            resp = requests.get(f"{BASE_URL}/health")
            health_ok = resp.status_code == 200 and resp.json().get("status") == "healthy"
            all_passed &= print_report("Health Endpoint", health_ok, "status=healthy", resp.json().get("status"))
        except Exception as e:
            all_passed &= print_report("Health Endpoint", False, actual=str(e))

        # 2. TEST /tasks endpoint
        try:
            resp = requests.get(f"{BASE_URL}/tasks")
            tasks_data = resp.json()
            available_ids = tasks_data.get("available_ids", [])
            tasks_ok = all(tid in available_ids for tid in TASK_IDS)
            all_passed &= print_report("Tasks Endpoint", tasks_ok, f"Contains {TASK_IDS}", available_ids)
        except Exception as e:
            all_passed &= print_report("Tasks Endpoint", False, actual=str(e))

        # 3. TEST ENDPOINTS for each task ID
        for task_id in TASK_IDS:
            print(f"\nTesting Task: {task_id}")
            try:
                # a. POST /reset/{task_id}
                reset_resp = requests.post(f"{BASE_URL}/reset/{task_id}")
                reset_data = reset_resp.json()
                reset_ok = reset_resp.status_code == 200 and all(k in reset_data for k in ["services", "logs", "alerts", "time_step"])
                all_passed &= print_report(f"Reset {task_id}", reset_ok)

                # b. POST /step/{task_id}
                step_payload = {"action_type": "check_logs", "target": "auth"}
                step_resp = requests.post(f"{BASE_URL}/step/{task_id}", json=step_payload)
                step_data = step_resp.json()
                step_ok = step_resp.status_code == 200 and all(k in step_data for k in ["state", "reward", "done"])
                all_passed &= print_report(f"Step {task_id}", step_ok)

                # c. POST /grade/{task_id}
                grade_resp = requests.post(f"{BASE_URL}/grade/{task_id}")
                grade_data = grade_resp.json()
                final_score = grade_data.get("final_score", -1.0)
                grade_ok = grade_resp.status_code == 200 and 0.0 <= final_score <= 1.0
                all_passed &= print_report(f"Grade {task_id}", grade_ok, "0.0 <= score <= 1.0", final_score)

                # d. GET /state/{task_id}
                state_resp = requests.get(f"{BASE_URL}/state/{task_id}")
                all_passed &= print_report(f"State {task_id}", state_resp.status_code == 200)

            except Exception as e:
                print_report(f"Endpoints {task_id}", False, actual=str(e))
                all_passed = False

        # 4. TEST GRADER DIVERSITY - easy-auth-down
        print("\nTesting Grader Diversity (easy-auth-down)")
        try:
            # a. Reset then immediately grade (Score A)
            requests.post(f"{BASE_URL}/reset/easy-auth-down")
            score_a = requests.post(f"{BASE_URL}/grade/easy-auth-down").json().get("final_score")
            
            # b. Reset, do correct action, then grade (Score B)
            requests.post(f"{BASE_URL}/reset/easy-auth-down")
            requests.post(f"{BASE_URL}/step/easy-auth-down", json={"action_type": "restart_service", "target": "auth"})
            score_b = requests.post(f"{BASE_URL}/grade/easy-auth-down").json().get("final_score")
            
            diversity_ok = score_b > score_a
            all_passed &= print_report("Grader Diversity", diversity_ok, f"Score B ({score_b}) > Score A ({score_a})", f"{score_b} > {score_a}")
        except Exception as e:
            all_passed &= print_report("Grader Diversity", False, actual=str(e))

        # 5. TEST REWARD SIGNAL - easy-auth-down
        print("\nTesting Reward Signal (easy-auth-down)")
        try:
            # a. Wrong action
            requests.post(f"{BASE_URL}/reset/easy-auth-down")
            wrong_reward = requests.post(f"{BASE_URL}/step/easy-auth-down", json={"action_type": "restart_service", "target": "payments"}).json().get("reward")
            wrong_ok = wrong_reward <= 0
            all_passed &= print_report("Negative Reward (Wrong Action)", wrong_ok, "<= 0", wrong_reward)

            # b. Correct action
            requests.post(f"{BASE_URL}/reset/easy-auth-down")
            correct_reward = requests.post(f"{BASE_URL}/step/easy-auth-down", json={"action_type": "restart_service", "target": "auth"}).json().get("reward")
            correct_ok = correct_reward > 0
            all_passed &= print_report("Positive Reward (Correct Action)", correct_ok, "> 0", correct_reward)
        except Exception as e:
            all_passed &= print_report("Reward Signal", False, actual=str(e))

        # 6. TEST EPISODE TERMINATION
        print("\nTesting Episode Termination (easy-auth-down)")
        try:
            requests.post(f"{BASE_URL}/reset/easy-auth-down")
            steps = 0
            done = False
            while not done and steps < 10:
                resp = requests.post(f"{BASE_URL}/step/easy-auth-down", json={"action_type": "restart_service", "target": "auth"})
                done = resp.json().get("done")
                steps += 1
            
            term_ok = done and steps <= 5
            all_passed &= print_report("Episode Termination", term_ok, "done=True within 5 steps", f"done={done}, steps={steps}")
        except Exception as e:
            all_passed &= print_report("Episode Termination", False, actual=str(e))

    finally:
        print("\nKilling server process...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()

    if all_passed:
        print("\nALL TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
