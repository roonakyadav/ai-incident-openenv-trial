---
title: AI Incident OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🚨 AI Incident Response Environment (OpenEnv)

A realistic, deterministic environment to evaluate how well AI agents handle **cascading system failures** in production-like conditions.

---

## 🚨 Problem

Modern systems don’t fail in isolation.

A single issue (e.g., database slowdown) can trigger:
- Authentication failures  
- Payment disruptions  
- Frontend degradation  

Yet most AI benchmarks:
- Use **toy problems**
- Ignore **dependencies and delays**
- Fail to test **real decision-making under pressure**

---

## 💡 Why This Matters

In real-world incident response:

- Fixing the wrong component can **worsen the system**
- Acting too early or too late affects **cost and stability**
- Ignoring issues allows failures to **cascade**

This environment tests whether an AI agent can:
- Identify the **true root cause**
- Handle **misleading signals**
- Make **efficient, sequential decisions**

## ⚙️ Why This Environment is Hard

This is not a simple fix-the-error task.

The environment introduces real-world complexity:

### 🔗 Cascading Failures
Failures propagate across services:
- DB issues → Auth failures → Payment disruption → Frontend degradation  
Agents must identify the **true root cause**, not just symptoms.

### ⏳ Delayed Recovery
Fixing a service does not instantly restore the system.
Recovery happens **gradually**, requiring patience and correct sequencing.

### ⚠️ Misleading Signals
Multiple services may appear broken, but:
- fixing downstream services too early leads to **penalties**
- incorrect actions increase **system strain**

### 🧠 Long-Term Consequences
Bad decisions accumulate:
- increased latency
- slower recovery
- reduced final score

### 🚫 No Passive Success
Doing nothing (`ignore`) is penalized unless the system is already stabilizing.

---

👉 The agent must:
- diagnose correctly  
- act in the right order  
- avoid unnecessary actions  
- manage long-term system health  

This makes the task closer to **real incident response**, not a toy benchmark.

## 📊 Evaluation Results

The environment clearly separates agent performance:

| Agent Type | Behavior | Score |
|----------|--------|------|
| Bad Agent | Random / incorrect fixes | 0.14 |
| Lazy Agent | Frequent ignore() | 0.07 |
| Smart Agent | Correct sequence + stabilization | 1.00 |

### 🔍 Insight

- Wrong actions significantly degrade performance  
- Passive behavior is penalized  
- Optimal strategies are strongly rewarded  

This demonstrates that the environment **accurately evaluates decision-making quality**, not just task completion.