from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum

class ServiceStatus(str, Enum):
    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"

class Service(BaseModel):
    name: str
    status: ServiceStatus
    latency: float = 0.0
    error_rate: float = 0.0

class ActionType(str, Enum):
    RESTART_SERVICE = "restart_service"
    SCALE = "scale"
    OPTIMIZE_DB = "optimize_db"
    CHECK_LOGS = "check_logs"
    IGNORE = "ignore"
    UNKNOWN = "unknown"

class Action(BaseModel):
    action_type: ActionType
    target: str

class State(BaseModel):
    services: List[Service]
    logs: List[str]
    alerts: List[str]
    time_step: int
    bad_actions: int
    history: List[Action]
    system_health: float
    total_cost: float
    system_stability: float
    risky_actions_count: int
    dependencies: Dict[str, List[str]] = {}
    system_strain: float = 0.0

class StepResult(BaseModel):
    state: State
    reward: float
    done: bool
    info: Optional[Dict[str, Any]] = None

class EpisodeResult(BaseModel):
    final_score: float
    root_cause_score: float
    efficiency_score: float
    damage_score: float
    cost_efficiency_score: float
    stability_score: float
    steps_used: int
    bad_actions: int
    system_health_penalty: float
    total_cost: float
    system_strain: float = 0.0
    optimal_bonus: float = 0.0

class TaskDifficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class Task(BaseModel):
    id: str
    difficulty: TaskDifficulty
    description: str
    initial_services: List[Service]
    initial_logs: List[str]
    initial_alerts: List[str]
    max_steps: int
    goal: str = ""
    success_conditions: Dict[str, Any] = {}
    failure_conditions: Dict[str, Any] = {}
