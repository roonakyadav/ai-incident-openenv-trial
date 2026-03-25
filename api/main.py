from fastapi import FastAPI, HTTPException
from models.schemas import Action, StepResult, Task, State, EpisodeResult
from env.core import IncidentEnv
from env.tasks import get_task, TASKS
from env.grader import IncidentGrader
from typing import Dict, List

@app.get("/")
def root():
    return {"status": "ok", "message": "AI Incident Response Environment running"}
    
app = FastAPI(title="AI Operations Incident Response Environment")

TASK_ALIASES = {
    "easy": "easy-auth-down",
    "medium": "medium-payments-degraded",
    "hard": "hard-cascading-failure"
}

sessions: Dict[str, IncidentEnv] = {}
graders: Dict[str, IncidentGrader] = {}

@app.get("/tasks")
async def list_tasks():
    return {
        "available_ids": [t.id for t in TASKS],
        "aliases": TASK_ALIASES,
        "tasks": TASKS
    }

@app.post("/reset/{task_id}")
async def reset(task_id: str):
    # Map alias to full ID if applicable
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
    # Store session using the provided task_id (supports alias)
    sessions[task_id] = env
    graders[task_id] = IncidentGrader()
    return env.get_state()

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

@app.get("/state/{task_id}", response_model=State)
async def get_state(task_id: str):
    if task_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[task_id].get_state()

@app.post("/grade/{task_id}", response_model=EpisodeResult)
async def grade(task_id: str):
    if task_id not in sessions or task_id not in graders:
        raise HTTPException(status_code=404, detail="Session not found. Call reset first.")

    env = sessions[task_id]
    grader = graders[task_id]
    
    final_score = grader.grade_episode(env.get_state(), env.task)
    return final_score

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
