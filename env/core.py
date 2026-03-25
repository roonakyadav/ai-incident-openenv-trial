from typing import List, Dict, Any
from models.schemas import State, Service, ServiceStatus, Action, ActionType, Task, TaskDifficulty
from env.grader import IncidentGrader

class IncidentEnv:
    ACTION_COSTS = {
        ActionType.RESTART_SERVICE: 1.0,
        ActionType.SCALE: 3.0,
        ActionType.OPTIMIZE_DB: 5.0,
        ActionType.CHECK_LOGS: 0.1,
        ActionType.IGNORE: 0.0
    }

    def __init__(self, task: Task):
        self.task = task
        self.services = [Service(**s.model_dump()) for s in task.initial_services]
        self.logs = list(task.initial_logs)
        self.alerts = list(task.initial_alerts)
        self.time_step = 0
        self.max_steps = 5  # Set max_steps to 5
        self.history = []
        self.bad_actions = 0
        self.total_cost = 0.0
        self.system_stability = 1.0
        self.risky_actions_count = 0
        self.grader = IncidentGrader()

    def get_state(self) -> State:
        return State(
            services=self.services,
            logs=self.logs,
            alerts=self.alerts,
            time_step=self.time_step,
            bad_actions=self.bad_actions,
            history=self.history,
            system_health=self._calculate_system_health(),
            total_cost=self.total_cost,
            system_stability=self.system_stability,
            risky_actions_count=self.risky_actions_count
        )

    def step(self, action: Action) -> Dict[str, Any]:
        self.time_step += 1
        self.history.append(action)
        
        # Add cost
        self.total_cost += self.ACTION_COSTS.get(action.action_type, 0.0)
        
        reward_info = self._apply_action(action)
        
        if reward_info["type"] in ["wrong_fix", "useless_action"]:
            self.bad_actions += 1
            # Instability penalty for wrong/useless actions
            self.system_stability = max(0.0, self.system_stability - 0.1)
        
        # Deterministic risk for OPTIMIZE_DB
        if action.action_type == ActionType.OPTIMIZE_DB:
            self.risky_actions_count += 1
            # Penalty if stability is already low OR if too many risky actions taken
            if self.system_stability < 0.7 or self.risky_actions_count > 1:
                penalty = 0.1 * self.risky_actions_count
                self.system_stability = max(0.0, self.system_stability - penalty)
        
        self._evolve_env()
        
        done = (
            self.time_step >= self.max_steps or 
            self.bad_actions > 2 or 
            self.system_stability < 0.3 or 
            self._all_services_up()
        )
        
        reward = 0.0 

        return {
            "state": self.get_state(),
            "reward": reward,
            "done": done,
            "info": reward_info
        }


    def _apply_action(self, action: Action) -> Dict[str, Any]:
        target_service = next((s for s in self.services if s.name == action.target), None)
        db_service = next((s for s in self.services if s.name == "db"), None)

        if action.action_type == ActionType.RESTART_SERVICE:
            # Penalize premature auth restart
            if self.task.difficulty == TaskDifficulty.HARD and action.target == "auth" and db_service.status == ServiceStatus.DEGRADED:
                return {"type": "wrong_fix", "target": action.target}
            if target_service and target_service.status == ServiceStatus.DOWN:
                target_service.status = ServiceStatus.UP
                return {"type": "correct_fix", "target": action.target}
            else:
                return {"type": "useless_action", "target": action.target}
        
        elif action.action_type == ActionType.SCALE:
            if target_service and target_service.status == ServiceStatus.DEGRADED and action.target != "db":
                target_service.status = ServiceStatus.UP
                return {"type": "correct_fix", "target": action.target}
            else:
                return {"type": "useless_action", "target": action.target}

        elif action.action_type == ActionType.OPTIMIZE_DB:
            if target_service and target_service.name == "db" and target_service.status == ServiceStatus.DEGRADED:
                target_service.status = ServiceStatus.UP
                return {"type": "correct_fix", "target": action.target}
            else:
                return {"type": "useless_action", "target": action.target}
        
        elif action.action_type == ActionType.CHECK_LOGS:
            return {"type": "diagnosis", "target": action.target}
            
        elif action.action_type == ActionType.IGNORE:
            return {"type": "ignore", "target": action.target}

        return {"type": "unknown", "target": action.target}

    def _evolve_env(self):
        if self.task.difficulty == TaskDifficulty.HARD:
            db_service = next((s for s in self.services if s.name == "db"), None)
            auth_service = next((s for s in self.services if s.name == "auth"), None)
            payments_service = next((s for s in self.services if s.name == "payments"), None)

            # Sequential recovery with 1-step delay
            if db_service and db_service.status == ServiceStatus.UP:
                if auth_service and auth_service.status == ServiceStatus.DOWN:
                    # After 1 step, auth recovers
                    if len(self.history) >= 2 and self.history[-2].action_type == ActionType.OPTIMIZE_DB:
                        auth_service.status = ServiceStatus.UP
            
            if auth_service and auth_service.status == ServiceStatus.UP:
                if payments_service and payments_service.status == ServiceStatus.DEGRADED:
                    # After 1 step, payments recovers
                    if len(self.history) >= 2 and self.history[-2].action_type != ActionType.SCALE:
                        payments_service.status = ServiceStatus.UP
        
        # Deterministic evolution logic for medium difficulty
        elif self.task.difficulty == TaskDifficulty.MEDIUM:
            # Add log every 3 steps
            if self.time_step % 3 == 0:
                self.logs.append(f"Spurious log entry at step {self.time_step}")
            
            auth_service = next((s for s in self.services if s.name == "auth"), None)
            payment_service = next((s for s in self.services if s.name == "payments"), None)
            
            if auth_service and auth_service.status == ServiceStatus.DOWN:
                # If auth is down for more than 2 steps, payments degrades
                down_steps = sum(1 for a in self.history if a.action_type == ActionType.IGNORE or a.action_type == ActionType.CHECK_LOGS)
                if down_steps >= 2:
                    if payment_service and payment_service.status == ServiceStatus.UP:
                        payment_service.status = ServiceStatus.DEGRADED
                        self.alerts.append("Payments service performance degrading due to auth failure")

    def _all_services_up(self) -> bool:
        return all(s.status == ServiceStatus.UP for s in self.services)

    def _calculate_system_health(self) -> float:
        up_services = sum(1 for s in self.services if s.status == ServiceStatus.UP)
        return up_services / len(self.services) if self.services else 0.0
