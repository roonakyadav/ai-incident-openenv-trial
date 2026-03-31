---
title: AI Incident OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

---

title: AI Incident OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
-------------

# 🚨 AI Incident Response Environment (OpenEnv)

A **deterministic benchmark environment** for evaluating how well AI agents handle **cascading failures, root-cause diagnosis, and sequential decision-making** in distributed systems.

---

## 🚨 Problem

Modern systems don’t fail in isolation.

A single issue (e.g., database degradation) can trigger:

* Authentication failures
* Payment disruptions
* Frontend degradation

Yet most AI benchmarks:

* use **toy environments**
* ignore **dependencies**
* lack **real-world failure dynamics**
* reward **surface-level actions instead of reasoning**

---

## 💡 Why This Matters

In real-world incident response:

* Fixing the wrong component can **worsen the system**
* Acting inefficiently increases **cost and recovery time**
* Misinterpreting logs leads to **incorrect decisions**
* Ignoring cascading effects causes **system-wide failure**

This environment evaluates whether an AI agent can:

* Identify the **true root cause**
* Handle **misleading signals**
* Execute **correct action sequences**
* Optimize for **stability + cost**

---

## ⚙️ Core Features

### 🔗 Cascading Failures

Failures propagate through dependencies:

```
DB → Auth → Payments → Frontend
```

Fixing downstream services without solving the root cause leads to **penalties**.

---

### ⏳ Delayed Recovery

Recovery is **not instant**.

* Fixing root cause ≠ immediate system recovery
* Services stabilize gradually
* Correct sequencing is required

---

### ⚠️ Misleading Signals (Hard Mode)

Logs and alerts may point to **wrong components**.

Agents must:

* avoid reacting blindly
* reason about dependencies
* ignore deceptive signals

---

### 📉 System Strain & Bad Actions

Incorrect actions increase:

* system strain
* latency
* instability

Repeated bad decisions are **heavily penalized**.

---

### 🧠 Sequential Decision Making

This is not a one-step problem.

Agents must:

* plan actions
* act step-by-step
* observe outcomes
* adapt strategy

---

## 🎯 Task Design

| Task   | Goal                                  | Key Challenge    |
| ------ | ------------------------------------- | ---------------- |
| Easy   | Restore all services                  | Basic recovery   |
| Medium | Stabilize system efficiently          | Action selection |
| Hard   | Fix cascading failure (DB root cause) | Causal reasoning |

---

## 🧪 Evaluation

### Reward System

The environment provides **dense feedback**:

* ✅ Correct fix → positive reward
* ❌ Wrong fix → penalty
* 🚫 Useless action → penalty
* ⏱️ Time penalty per step
* 📈 Stability improvement → bonus

---

### Grading Criteria

Final score is based on:

* Task success conditions
* Efficiency (steps taken)
* Cost of actions
* System stability
* Failure penalties

---

## 📊 Example Results

| Agent Type  | Behavior                 | Score |
| ----------- | ------------------------ | ----- |
| Bad Agent   | Random actions           | 0.14  |
| Lazy Agent  | Frequent `ignore()`      | 0.07  |
| Smart Agent | Correct causal reasoning | 1.00  |

### 🔍 Insight

* Wrong actions degrade system rapidly
* Passive agents fail consistently
* Correct reasoning is strongly rewarded

👉 The benchmark **separates weak vs strong agents clearly**

---

## 🧱 Environment Design

### Services

* `db` — root dependency
* `auth` — depends on DB
* `payments` — depends on auth
* `frontend` — depends on payments

---

### State Representation

Each step returns:

* service status (up / degraded / down)
* latency + error rate
* logs + alerts
* system stability
* system strain
* action history

---

### Actions

* `restart_service`
* `scale`
* `optimize_db`
* `check_logs`
* `ignore`

---

## 🔌 API Endpoints

| Endpoint             | Description            |
| -------------------- | ---------------------- |
| `/reset/{task}`      | Initialize environment |
| `/step/{task}`       | Execute action         |
| `/state/{task}`      | Get current state      |
| `/grade/{task}`      | Compute final score    |
| `/baseline`          | Run baseline agent     |
| `/trajectory/{task}` | Full action history    |

---

## 🖥️ UI Features

* Interactive control panel
* Real-time system state
* Logs + alerts visualization
* Evaluation panel showing:

  * reward
  * status (running / completed)
  * action feedback

---

## 🧠 What This Benchmark Tests

This is NOT just task completion.

It evaluates:

* causal reasoning
* root-cause analysis
* decision sequencing
* robustness to misleading data
* cost-aware optimization

---

## 🚀 Deployment

* Backend: FastAPI
* Containerized with Docker
* Hosted on Hugging Face Spaces

### Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/roonakyadav/meta-hackathon.git
    cd meta-hackathon
    ```

2.  **Build and run the Docker container:**
    ```bash
    docker build -t ai-incident-env .
    docker run -p 7860:7860 ai-incident-env
    ```

3.  **Run the baseline agent:**
    ```bash
    # In a new terminal
    source venv/bin/activate
    python baseline/run_agent.py
    ```

### Baseline Results

| Task                      | Rule-Based Reward | LLM (Groq) Reward |
| ------------------------- | ----------------- | ----------------- |
| `easy-auth-down`          | 0.75              | 0.85              |
| `medium-payments-degraded`| 0.75              | 0.75              |
| `hard-cascading-failure`  | 1.20              | 1.20              |

---

## ⚠️ Limitations

* Single-agent environment
* Fixed service topology
* Deterministic (no randomness)

---

## 🔮 Future Work

* Multi-agent coordination
* Dynamic scenario generation
* Partial observability
* Larger service graphs

---

## ✅ Summary

This project introduces a **realistic, deterministic benchmark** for evaluating AI agents in incident response scenarios.

Unlike traditional benchmarks, it focuses on:

* **causal reasoning over pattern matching**
* **decision quality over action execution**
* **system thinking over isolated fixes**

👉 The result is a benchmark that actually measures **how an agent thinks**, not just what it outputs.
