---
title: AI Incident OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# AI Incident Response Environment

This is a FastAPI-based OpenEnv environment for evaluating AI agents on incident response tasks.

## Endpoints

- `/` → health check  
- `/tasks` → list tasks  
- `/reset/{task_id}` → start session  
- `/step/{task_id}` → take action  
- `/state/{task_id}` → get state  
- `/grade/{task_id}` → evaluate  

## Example

Go to `/docs` to interact with the API.