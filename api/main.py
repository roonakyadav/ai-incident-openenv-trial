from fastapi import FastAPI, HTTPException
from models.schemas import Action, StepResult, State, EpisodeResult
from env.core import IncidentEnv
from env.tasks import get_task, TASKS
from env.grader import IncidentGrader
from typing import Dict
from baseline.baseline_agent import BaselineAgent
from fastapi.staticfiles import StaticFiles

# ✅ DEFINE APP FIRST (CRITICAL)
app = FastAPI(title="AI Operations Incident Response Environment")

# Task aliases
TASK_ALIASES = {
    "easy": "easy-auth-down",
    "medium": "medium-payments-degraded",
    "hard": "hard-cascading-failure"
}

# In-memory session storage
sessions: Dict[str, IncidentEnv] = {}
graders: Dict[str, IncidentGrader] = {}

# List tasks
@app.get("/tasks")
async def list_tasks():
    return {
        "available_ids": [t.id for t in TASKS],
        "aliases": TASK_ALIASES,
        "tasks": TASKS
    }

# Reset environment
@app.post("/reset/{task_id}")
async def reset(task_id: str):
    actual_id = TASK_ALIASES.get(task_id.lower(), task_id)
    task = get_task(actual_id)

    if not task:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "Task not found",
                "invalid_id": task_id,
                "available_tasks": [t.id for t in TASKS],
                "aliases": list(TASK_ALIASES.keys())
            }
        )

    env = IncidentEnv(task)
    sessions[task_id] = env
    graders[task_id] = IncidentGrader()

    return env.get_state()

# Step
@app.post("/step/{task_id}", response_model=StepResult)
async def step(task_id: str, action: Action):
    if task_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Call reset first.")

    env = sessions[task_id]
    result = env.step(action)

    return StepResult(
        state=result["state"],
        reward=result["reward"],
        done=result["done"],
        info=result["info"]
    )

# Get state
@app.get("/state/{task_id}")
async def get_state(task_id: str):
    if task_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = sessions[task_id].get_state()
    
    # Transform to required structured format
    return {
        "services": {
            s.name: {
                "status": s.status.value,
                "latency": s.latency,
                "error_rate": s.error_rate
            } for s in state.services
        },
        "alerts": state.alerts,
        "logs": state.logs,
        "cost": state.total_cost,
        "stability": state.system_stability,
        "step": state.time_step,
        "dependencies": state.dependencies
    }

# Grade
@app.post("/grade/{task_id}", response_model=EpisodeResult)
async def grade(task_id: str):
    if task_id not in sessions or task_id not in graders:
        raise HTTPException(status_code=404, detail="Session not found. Call reset first.")

    env = sessions[task_id]
    grader = graders[task_id]

    final_score = grader.grade_episode(env.get_state(), env.task)
    return final_score

# Baseline
@app.get("/baseline")
async def run_baseline():
    agent = BaselineAgent()
    results = {}
    total_score = 0.0
    
    # Define tasks to run (using full IDs)
    task_ids = ["easy-auth-down", "medium-payments-degraded", "hard-cascading-failure"]
    
    for full_id in task_ids:
        task = get_task(full_id)
        if not task:
            continue
            
        env = IncidentEnv(task)
        grader = IncidentGrader()
        
        # Run episode
        done = False
        while not done and env.time_step < env.max_steps:
            state = env.get_state()
            action = agent.decide_action(state)
            result = env.step(action)
            done = result["done"]
            
        # Grade
        episode_result = grader.grade_episode(env.get_state(), env.task)
        score = episode_result.final_score
        
        # Map back to simple names for output
        short_name = full_id.split("-")[0]
        results[short_name] = round(score, 3)
        total_score += score
        
    avg_score = total_score / len(results) if results else 0.0
    
    return {
        "tasks": results,
        "average_score": round(avg_score, 3)
    }

# Serve Static UI (at the end)
app.mount("/", StaticFiles(directory="static", html=True), name="static")