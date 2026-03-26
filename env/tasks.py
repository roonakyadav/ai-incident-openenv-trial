from models.schemas import Task, TaskDifficulty, Service, ServiceStatus

TASKS = [
    Task(
        id="easy-auth-down",
        difficulty=TaskDifficulty.EASY,
        description="The authentication service is down. Restart it to restore service.",
        goal="Restore all services to UP state",
        success_conditions={
            "all_services_up": True,
            "max_error_rate": 0.05
        },
        failure_conditions={
            "bad_actions_limit": 2,
            "max_steps_exceeded": True
        },
        initial_services=[
            Service(name="auth", status=ServiceStatus.DOWN),
            Service(name="payments", status=ServiceStatus.UP),
            Service(name="frontend", status=ServiceStatus.UP)
        ],
        initial_logs=[
            "Auth service: CRITICAL - service failed to start",
            "Auth service: Database connection timeout"
        ],
        initial_alerts=[
            "CRITICAL: Auth service is down"
        ],
        max_steps=5
    ),
    Task(
        id="medium-payments-degraded",
        difficulty=TaskDifficulty.MEDIUM,
        description="The payments service is degraded due to high load. Scale it up to restore performance.",
        goal="Stabilize system with minimal cost",
        success_conditions={
            "no_service_down": True,
            "latency_threshold": 200,
            "cost_limit": 15
        },
        failure_conditions={
            "bad_actions_limit": 3,
            "system_stability_below": 0.4
        },
        initial_services=[
            Service(name="auth", status=ServiceStatus.UP),
            Service(name="payments", status=ServiceStatus.DEGRADED),
            Service(name="frontend", status=ServiceStatus.UP)
        ],
        initial_logs=[
            "Payments service: WARNING - high CPU usage detected",
            "Payments service: Latency exceeded threshold"
        ],
        initial_alerts=[
            "WARNING: Payments service performance degraded"
        ],
        max_steps=8
    ),
    Task(
        id="hard-cascading-failure",
        difficulty=TaskDifficulty.HARD,
        description="A critical dependency is experiencing high latency, causing a cascading failure. Multi-step reasoning is required.",
        goal="Resolve cascading failures by fixing root cause (DB)",
        success_conditions={
            "stability_score": 0.9,
            "root_cause_fixed": True,
            "strain_limit": 0.3
        },
        failure_conditions={
            "bad_actions_limit": 3,
            "system_stability_below": 0.3
        },
        initial_services=[
            Service(name="db", status=ServiceStatus.DEGRADED),
            Service(name="auth", status=ServiceStatus.DOWN),
            Service(name="payments", status=ServiceStatus.DEGRADED),
            Service(name="frontend", status=ServiceStatus.UP)
        ],
        initial_logs=[
            "WARN: High CPU on auth service instance a-123f.",
            "ERROR: Upstream dependency timeout for service: auth",
            "INFO: User login failed for user: testuser",
            "WARN: Payments service experiencing intermittent timeouts.",
            "INFO: Deployment pipeline #125 succeeded.",
            "WARN: Query latency for a critical dependency has increased by 300%.",
            "ERROR: Payments service crashed due to overload",
            "WARN: Frontend latency spike detected",
            "INFO: Restart recommended for payments service"
        ],
        initial_alerts=[
            "CRITICAL: Auth service is down",
            "WARN: Payments service latency high",
            "INFO: High CPU on auth service (misleading)",
            "CRITICAL: Critical dependency latency spike detected"
        ],
        max_steps=10
    )
]

def get_task(task_id: str) -> Task:
    for task in TASKS:
        if task.id == task_id:
            return task
    return None

def test_hard_task_state():
    """Prints the initial state of the hard task for verification."""
    task = get_task("hard-cascading-failure")
    if task:
        print("--- Initial State of HARD Task ---")
        print(f"Description: {task.description}")
        print("Services:")
        for service in task.initial_services:
            print(f"  - {service.name}: {service.status}")
        print("Logs:")
        for log in task.initial_logs:
            print(f"  - {log}")
        print("Alerts:")
        for alert in task.initial_alerts:
            print(f"  - {alert}")
        print("---------------------------------")
    else:
        print("HARD task not found!")
