---
tags:
- openenv
---

# AI Incident Response — OpenEnv Environment

## What This Is
This is an OpenEnv-compatible environment that simulates distributed system incident response. Agents must diagnose and recover multi-service failures by reasoning over logs, alerts, and metrics. The environment features complex failure modes including a re-degradation mechanic for incorrect fix sequences and ambiguous root cause designs that challenge causal reasoning.

## Why It Matters
In real SRE/DevOps contexts, incidents are often non-obvious and cascading. Agents that succeed here must reason about causality and dependency chains, rather than just performing simple pattern-matching. Existing benchmarks often fail to test this depth of reasoning, which is critical for autonomous operations in production environments.

## Environment Details

### Services
| Service | Role | Dependencies |
| :--- | :--- | :--- |
| auth | Authentication | payments-db (task-specific) |
| payments | Transaction processing | none |
| frontend | User interface | auth, payments |

### Observation Space
| Field | Description |
| :--- | :--- |
| services | List of service state objects (status, latency, error_rate) |
| logs | Last 10 log lines from the system |
| alerts | Active alerts currently triggering in the environment |
| system_health | Proportional health of services (0.0-1.0) |
| system_stability | Overall system stability score (0.0-1.0) |
| system_strain | Current strain level on the infrastructure (0.0-1.0) |
| time_step | Current step number in the episode |

### Action Space
| Action | Target | Effect |
| :--- | :--- | :--- |
| restart_service | service_name | Restarts a service that is currently DOWN. |
| rollback_service | service_name | Rolls back a service to its last stable version. |
| isolate_service | service_name | Isolates a service from the dependency graph temporarily. |
| check_logs | none | Returns the last 10 log entries for diagnosis. |
| check_metrics | none | Returns detailed latency and error_rate breakdown per service. |
| scale_service | service_name | Scales a DEGRADED service to increase capacity. |
| drain_traffic | service_name | Redirects traffic away from a degraded service. |
| restore_traffic | service_name | Reverses drain_traffic and restores normal flow. |
| optimize_db | db | Optimizes database queries (only works on 'db' target). |
| escalate | none | Ends the incident and provides a partial score. |

### Reward Structure
The final score is calculated using the following formula:
`final_score = 0.40 * success + 0.15 * efficiency + 0.20 * cost + 0.05 * stability + 0.20 * sequence`

The **sequence score** is critical: it rewards early correct fixes and penalizes actions taken in the wrong order (e.g., treating symptoms before root causes).

## Tasks

### easy-auth-down
- **Difficulty**: Easy
- **Scenario**: Single auth service failure.
- **Success**: auth UP, error_rate < 0.1
- **Key challenge**: Identify the correct service to restart from basic logs.

### hard-bad-deployment
- **Difficulty**: Medium-Hard
- **Scenario**: Auth failure caused by a bad deployment, despite misleading DB error logs.
- **Success**: All services UP, error_rate < 0.1
- **Key challenge**: Distinguish deployment failure from DB failure. `restart_service` provides only partial recovery; `rollback_service` is required for a full fix.

### hard-cascading-ambiguous
- **Difficulty**: Hard
- **Scenario**: Both auth and payments degraded, frontend down. A shared DB dependency means fixing auth first causes it to re-degrade.
- **Success**: All services UP, error_rate < 0.1
- **Key challenge**: Correct fix ORDER matters. A wrong sequence triggers the **re-degradation mechanic**, requiring the agent to identify `payments` as the true root cause.
- **Max steps**: 15

## Baseline Scores
| Task | Baseline Score | Model |
| :--- | :--- | :--- |
| easy-auth-down | 0.82 | llama-3.1-8b-instant |
| hard-bad-deployment | 0.54 | llama-3.1-8b-instant |
| hard-cascading-ambiguous | 0.31 | llama-3.1-8b-instant |

*Note: Run `inference.py` to reproduce.*

## Setup

### Local
```bash
git clone https://github.com/roonakyadav/meta-hackathon
cd meta-hackathon
pip install -r requirements.txt
python3 -m api.main
```

### Docker
```bash
docker build -t ai-incident-openenv .
docker run -p 7860:7860 ai-incident-openenv
```

### Inference
```bash
API_BASE_URL=https://roonakyadav-ai-incident-openenv-trial.hf.space \
MODEL_NAME=llama-3.1-8b-instant \
HF_TOKEN=your_key \
python3 inference.py
```

## API Reference
- `POST /reset` — Initialize episode. Body: `{"task_id": "...", "seed": 42}`
- `POST /step` — Take action. Body: `{"action_type": "...", "target": "..."}`
- `GET /state` — Read current state (no side effects)
