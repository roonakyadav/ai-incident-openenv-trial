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
        self.max_steps = task.max_steps
        self.history = []
        self.bad_actions = 0
        self.total_cost = 0.0
        # self.system_stability = 1.0 # This is now calculated
        self.risky_actions_count = 0
        self.system_strain = 0.0
        self.dependencies = self._initialize_dependencies()
        self.grader = IncidentGrader()
        self._update_metrics()
        self._update_alerts() # Initial alerts based on status
        self.system_stability = self._calculate_stability() # Correctly calculate initial stability

    def _initialize_dependencies(self) -> Dict[str, List[str]]:
        if self.task.id == "hard-cascading-failure":
            return {
                "db": ["auth"],
                "auth": ["payments"],
                "payments": ["frontend"]
            }
        elif self.task.id == "medium-payments-degraded":
            return {
                "auth": ["payments"],
                "payments": ["frontend"]
            }
        else:
            return {
                "auth": ["frontend"],
                "payments": ["frontend"]
            }

    def _update_metrics(self):
        for service in self.services:
            # Latency increases with system strain
            strain_latency_penalty = self.system_strain * 100.0
            
            if service.status == ServiceStatus.UP:
                service.latency = 20.0 + strain_latency_penalty
                service.error_rate = 0.01 + (self.system_strain * 0.05)
            elif service.status == ServiceStatus.DEGRADED:
                service.latency = 500.0 + (strain_latency_penalty * 2)
                service.error_rate = 0.15 + (self.system_strain * 0.1)
            elif service.status == ServiceStatus.DOWN:
                service.latency = 0.0
                service.error_rate = 1.0

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
            risky_actions_count=self.risky_actions_count,
            dependencies=self.dependencies,
            system_strain=self.system_strain
        )

    def step(self, action: Action) -> Dict[str, Any]:
        self.time_step += 1
        self.history.append(action)
        
        # Add cost
        self.total_cost += self.ACTION_COSTS.get(action.action_type, 0.0)
        
        reward_info = self._apply_action(action)
        
        if reward_info["type"] in ["wrong_fix", "useless_action"]:
            self.bad_actions += 1
            # Initial instability penalty for wrong/useless actions
            self.system_stability = max(0.0, self.system_stability - 0.1)
            # Increase system strain
            self.system_strain += 0.2
        
        # System strain decay
        self.system_strain = max(0.0, self.system_strain - 0.05)
        
        # Deterministic risk for OPTIMIZE_DB
        if action.action_type == ActionType.OPTIMIZE_DB:
            self.risky_actions_count += 1
            # Penalty if stability is already low OR if too many risky actions taken
            if self.system_stability < 0.7 or self.risky_actions_count > 1:
                penalty = 0.1 * self.risky_actions_count
                self.system_stability = max(0.0, self.system_stability - penalty)
                self.system_strain += 0.1
        
        # Apply logic updates
        self._apply_cascading_failures()
        self._evolve_env()
        self._update_metrics()
        self._update_alerts() # Recompute alerts based on current state
        self.system_stability = self._calculate_stability()
        self._update_logs(action, reward_info)
        
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

    def _calculate_stability(self) -> float:
        # up -> +1, degraded -> +0.5, down -> 0
        service_scores = []
        for s in self.services:
            if s.status == ServiceStatus.UP:
                service_scores.append(1.0)
            elif s.status == ServiceStatus.DEGRADED:
                service_scores.append(0.5)
            else:
                service_scores.append(0.0)
        
        base_stability = sum(service_scores) / len(self.services) if self.services else 1.0
        
        # Subtract penalty for alerts (-0.1 per critical alert)
        critical_alerts = sum(1 for a in self.alerts if "CRITICAL" in a or "ERROR" in a)
        stability = base_stability - (0.1 * critical_alerts)
        
        # Incorporate history-based penalties (bad actions and risky actions)
        # Strain penalty: reduces stability linearly
        stability -= (0.1 * self.system_strain)
        
        # Apply diminishing returns: harder to reach 1.0 from 0.7 than 0.4 from 0.2
        # If stability > 0.7, apply a damping factor to the excess
        if stability > 0.7:
            stability = 0.7 + (stability - 0.7) * 0.5
            
        return max(0.0, min(1.0, stability))

    def _apply_cascading_failures(self):
        # Deterministic cascading failures based on dependencies
        for root, dependents in self.dependencies.items():
            root_service = next((s for s in self.services if s.name == root), None)
            if not root_service:
                continue
                
            for dep_name in dependents:
                dep_service = next((s for s in self.services if s.name == dep_name), None)
                if not dep_service:
                    continue
                
                if root_service.status == ServiceStatus.DOWN:
                    if root == "db":
                        if dep_name == "auth":
                            dep_service.status = ServiceStatus.DOWN
                        elif dep_name == "payments":
                            dep_service.status = ServiceStatus.DEGRADED
                    elif root == "auth":
                        if dep_name == "frontend":
                            dep_service.status = ServiceStatus.DEGRADED
                elif root_service.status == ServiceStatus.DEGRADED:
                    if dep_service.status == ServiceStatus.UP:
                        dep_service.status = ServiceStatus.DEGRADED
                elif root_service.status == ServiceStatus.UP:
                    # Natural recovery: if root is UP and dependent is DEGRADED, it recovers
                    if dep_service.status == ServiceStatus.DEGRADED:
                        dep_service.status = ServiceStatus.UP

    def _update_logs(self, action: Action, reward_info: Dict[str, Any]):
        new_logs = []
        
        # Log action result
        if reward_info["type"] == "correct_fix":
            new_logs.append(f"{action.target.capitalize()} service recovered successfully")
        elif reward_info["type"] == "wrong_fix":
            new_logs.append(f"Failed to fix {action.target.capitalize()}: dependency unresolved")
        elif reward_info["type"] == "useless_action":
            new_logs.append(f"Action {action.action_type} on {action.target} had no impact")
            
        # Log cascading effects
        for s in self.services:
            if s.status == ServiceStatus.DEGRADED:
                for root, dependents in self.dependencies.items():
                    if s.name in dependents:
                        root_service = next((s for s in self.services if s.name == root), None)
                        if root_service and root_service.status != ServiceStatus.UP:
                            new_logs.append(f"{s.name.capitalize()} degraded due to {root} failure")
                            break
            elif s.status == ServiceStatus.DOWN:
                for root, dependents in self.dependencies.items():
                    if s.name in dependents:
                        root_service = next((s for s in self.services if s.name == root), None)
                        if root_service and root_service.status == ServiceStatus.DOWN:
                            new_logs.append(f"{s.name.capitalize()} offline due to {root} outage")
                            break
                            
        # Generic state logs
        if self.system_stability < 0.5:
            new_logs.append("System instability reaching critical levels")
            
        # Add to logs and limit to last 10
        self.logs.extend(new_logs)
        self.logs = self.logs[-10:]


    def _apply_action(self, action: Action) -> Dict[str, Any]:
        target_service = next((s for s in self.services if s.name == action.target), None)
        
        # Identify root cause for the current task
        root_cause_service = None
        if self.task.id == "hard-cascading-failure":
            root_cause_service = next((s for s in self.services if s.name == "db"), None)
        elif self.task.id == "medium-payments-degraded":
            root_cause_service = next((s for s in self.services if s.name == "payments"), None)

        if action.action_type == ActionType.RESTART_SERVICE:
            if not target_service:
                return {"type": "unknown", "target": action.target}
                
            # If target has an unresolved upstream dependency that is DOWN/DEGRADED
            for root, dependents in self.dependencies.items():
                if action.target in dependents:
                    upstream = next((s for s in self.services if s.name == root), None)
                    if upstream and upstream.status != ServiceStatus.UP:
                        return {"type": "wrong_fix", "target": action.target}

            if target_service.status == ServiceStatus.DOWN:
                target_service.status = ServiceStatus.UP
                return {"type": "correct_fix", "target": action.target}
            else:
                return {"type": "useless_action", "target": action.target}
        
        elif action.action_type == ActionType.SCALE:
            if not target_service:
                return {"type": "unknown", "target": action.target}

            # Scaling while root cause exists is expensive but might work partially
            if root_cause_service and root_cause_service.status != ServiceStatus.UP and action.target != root_cause_service.name:
                self.total_cost += 2.0 # Extra cost for scaling symptom instead of root cause
                
            if target_service.status == ServiceStatus.DEGRADED:
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
        last_action = self.history[-1] if self.history else None
        
        # Refined ignore() logic: stability stagnates or decreases if no action is taken
        if last_action and last_action.action_type == ActionType.IGNORE:
            if self.system_stability < 0.8:
                # Add log about persist instability
                self.logs.append("No action taken, system instability persists")
                # Small penalty for inaction when system is unhealthy
                self.system_strain += 0.05

        if self.task.difficulty == TaskDifficulty.HARD:
            db_service = next((s for s in self.services if s.name == "db"), None)
            auth_service = next((s for s in self.services if s.name == "auth"), None)
            payments_service = next((s for s in self.services if s.name == "payments"), None)

            # Sequential recovery with 1-step delay per level
            # Recovery is slower if strain is high
            recovery_threshold = 2 if self.system_strain < 0.5 else 3
            
            if db_service and db_service.status == ServiceStatus.UP:
                if auth_service and auth_service.status == ServiceStatus.DOWN:
                    if len(self.history) >= recovery_threshold and self.history[-recovery_threshold].action_type == ActionType.OPTIMIZE_DB:
                        auth_service.status = ServiceStatus.UP
                
                elif auth_service and auth_service.status == ServiceStatus.UP:
                    if payments_service and payments_service.status == ServiceStatus.DEGRADED:
                        # Ensure at least 1-2 steps have passed since auth became UP
                        if len(self.history) >= (recovery_threshold + 1):
                            payments_service.status = ServiceStatus.UP
            
            # If root cause is fixed, clear root cause alerts
            # Handled by _update_alerts
            pass
        
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

    def _all_services_up(self) -> bool:
        return all(s.status == ServiceStatus.UP for s in self.services)

    def _calculate_system_health(self) -> float:
        up_services = sum(1 for s in self.services if s.status == ServiceStatus.UP)
        return up_services / len(self.services) if self.services else 0.0

    def _update_alerts(self):
        # Recompute alerts based on current service states
        new_alerts = []
        for service in self.services:
            if service.status == ServiceStatus.DOWN:
                new_alerts.append(f"CRITICAL: {service.name} is down")
            elif service.status == ServiceStatus.DEGRADED:
                new_alerts.append(f"WARNING: {service.name} degraded")
        
        # Add special alerts based on system stability if needed
        # (Though requirements specify service states, we can keep it clean)
        
        self.alerts = new_alerts
