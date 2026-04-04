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
    ROLLBACK_SERVICE = "rollback_service"
    ISOLATE_SERVICE = "isolate_service"
    CHECK_LOGS = "check_logs"
    CHECK_METRICS = "check_metrics"
    DRAIN_TRAFFIC = "drain_traffic"
    RESTORE_TRAFFIC = "restore_traffic"
    ESCALATE = "escalate"
    IGNORE = "ignore"
    UNKNOWN = "unknown"

class Action(BaseModel):
    action_type: ActionType
    target: str

class Reward(BaseModel):
    value: float
    reason: str = ""
    partial: bool = False

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
    symptom_fix_count: int = 0
    root_cause_step: Optional[int] = None
    delayed_failure_count: int = 0
    wasted_action_count: int = 0
    failure_type: Optional[str] = None
    diagnosed_targets: List[str] = []
    hidden_risk: float = 0.0
    side_effect_triggered: bool = False

class StepResult(BaseModel):
    state: State
    reward: float
    reward_info: Optional[Reward] = None
    done: bool
    info: Optional[Dict[str, Any]] = None

class EpisodeResult(BaseModel):
    final_score: float
    root_cause_score: float
    efficiency_score: float
    damage_score: float
    cost_efficiency_score: float
    stability_score: float
    sequence_score: float = 0.0
    sequence_bonus_applied: bool = False
    wrong_order_penalty_applied: bool = False
    steps_used: int
    bad_actions: int
    system_health_penalty: float
    total_cost: float
    system_strain: float = 0.0
    optimal_bonus: float = 0.0
    symptom_fix_count: int = 0
    root_cause_step: Optional[int] = None
    delayed_failure_count: int = 0
    wasted_action_count: int = 0
    failure_type: Optional[str] = None
    diagnosed_targets: List[str] = []

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
    true_root_cause: Optional[str] = None
    surface_symptom_target: Optional[str] = None
