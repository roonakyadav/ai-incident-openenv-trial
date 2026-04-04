from fastapi import FastAPI, HTTPException
from models.schemas import Action, StepResult, State, EpisodeResult
from env.core import IncidentEnv
from env.tasks import get_task, TASKS
from env.grader import IncidentGrader
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
from baseline.baseline_agent import BaselineAgent
from fastapi.staticfiles import StaticFiles

# ✅ DEFINE APP FIRST (CRITICAL)
app = FastAPI(title="AI Operations Incident Response Environment")

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"
    seed: Optional[int] = 42

class StepRequest(BaseModel):
    action_type: str
    target: Optional[str] = None
    task_id: Optional[str] = "easy"

# Task aliases
TASK_ALIASES = {
    "easy": "easy-auth-down",
    "medium": "medium-payments-degraded",
    "hard": "hard-cascading-failure",
    "deployment": "hard-bad-deployment"
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
@app.post("/reset")
async def reset_generic(request: Optional[ResetRequest] = None):
    # Default to easy task if none provided
    task_id = request.task_id if request and request.task_id else "easy"
    seed = request.seed if request and request.seed is not None else 42
    
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

    env = IncidentEnv(task, seed=seed)
    sessions[task_id] = env
    graders[task_id] = IncidentGrader()

    return env.get_state()

@app.post("/reset/{task_id}")
async def reset(task_id: str, seed: Optional[int] = 42):
    return await reset_generic(ResetRequest(task_id=task_id, seed=seed))

# Step
@app.post("/step", response_model=StepResult)
async def step_generic(request: StepRequest):
    # Default to easy task if none provided
    task_id = request.task_id if request.task_id else "easy"
    
    if task_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Call reset first.")

    env = sessions[task_id]
    
    # Prevent further actions after episode has terminated
    if env.is_done:
        raise HTTPException(
            status_code=400,
            detail="Episode already terminated. Please reset."
        )
    
    # Map StepRequest to Action
    from models.schemas import ActionType
    try:
        action_type = ActionType(request.action_type)
    except ValueError:
        action_type = ActionType.UNKNOWN
        
    action = Action(
        action_type=action_type,
        target=request.target if request.target else "none"
    )
    
    try:
        result = env.step(action)
    except Exception as e:
        # Extreme fallback to prevent 500
        print(f"CRITICAL ERROR in env.step: {e}")
        return StepResult(
            state=env.get_state(),
            reward=-0.5,
            done=False,
            info={"type": "error", "message": str(e)}
        )
    
    # If episode just terminated, add termination message
    if result["done"]:
        return StepResult(
            state=result["state"],
            reward=result["reward"],
            done=True,
            info={
                **result["info"],
                "message": "Episode terminated"
            }
        )

    return StepResult(
        state=result["state"],
        reward=result["reward"],
        done=result["done"],
        info=result["info"]
    )

@app.post("/step/{task_id}", response_model=StepResult)
async def step(task_id: str, action: Action):
    # Map Action to StepRequest
    request = StepRequest(
        action_type=action.action_type,
        target=action.target,
        task_id=task_id
    )
    return await step_generic(request)

# Get current state (Pure read)
@app.get("/state")
async def get_current_state(task_id: Optional[str] = "easy"):
    if task_id not in sessions:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call /reset first."
        )
    
    env = sessions[task_id]
    return env.state().model_dump()

# Get state by task_id
@app.get("/state/{task_id}")
async def get_state(task_id: str):
    if task_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = sessions[task_id].get_state()
    
    # Return full state consistent with State schema
    return state.model_dump()

# Grade
@app.post("/grade/{task_id}", response_model=EpisodeResult)
async def grade(task_id: str):
    if task_id not in sessions or task_id not in graders:
        raise HTTPException(status_code=404, detail="Session not found. Call reset first.")

    env = sessions[task_id]
    grader = graders[task_id]

    final_score = grader.grade_episode(env.get_state(), env.task)
    return final_score

# Health check for Hugging Face Spaces
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Get trajectory
@app.get("/trajectory/{task_id}")
async def get_trajectory(task_id: str):
    if task_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    env = sessions[task_id]

    # Convert Pydantic models to dicts for JSON serialization
    return {
        "steps": [
            {
                "action": step["action"].model_dump(),
                "state_before": step["state_before"].model_dump(),
                "state_after": step["state_after"].model_dump(),
                "reward_info": step["reward_info"]
            }
            for step in env.state_history
        ],
        "total_steps": len(env.state_history)
    }

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
import os as _os
if _os.path.isdir("static"):
    app.mount("/", StaticFiles(directory="static", html=True), name="static")

@app.on_event("startup")
async def startup_event():
    print("========================================")
    print("🚀 APP STARTED SUCCESSFULLY")
    print("📡 SERVER RUNNING ON PORT 7860")
    print("========================================")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")