import random
from typing import List, Dict, Any
from models.schemas import State, Service, ServiceStatus, Action, ActionType, Task, TaskDifficulty
from env.grader import IncidentGrader

class IncidentEnv:
    ACTION_COSTS = {
        ActionType.RESTART_SERVICE: 1.0,
        ActionType.SCALE: 3.0,
        ActionType.OPTIMIZE_DB: 5.0,
        ActionType.ROLLBACK_SERVICE: 4.0,
        ActionType.ISOLATE_SERVICE: 2.0,
        ActionType.CHECK_LOGS: 0.1,
        ActionType.CHECK_METRICS: 0.2,
        ActionType.DRAIN_TRAFFIC: 1.5,
        ActionType.RESTORE_TRAFFIC: 0.5,
        ActionType.ESCALATE: 0.0,
        ActionType.IGNORE: 0.0
    }

    def __init__(self, task: Task, seed: int = 42):
        self.task = task
        self.random = random.Random(seed)
        self.services = [Service(**s.model_dump()) for s in task.initial_services]
        self.logs = list(task.initial_logs)
        self.alerts = list(task.initial_alerts)
        self.time_step = 0
        self.max_steps = task.max_steps
        self.history = []
        self.state_history = []  # Store trajectory for debugging and evaluation
        self.bad_actions = 0
        self.total_cost = 0.0
        # self.system_stability = 1.0 # This is now calculated
        self.risky_actions_count = 0
        self.system_strain = 0.0
        self.previous_strain = 0.0  # Initialize previous strain for reward calculation
        self.isolated_services = []  # List of names of isolated services
        self.drained_services = []   # List of names of drained services
        self.partial_fixes = []      # List of names of services with partial fixes
        self.redegrade_timers = {}   # Map service_name to steps_until_redegrade
        
        # Diagnostic Metrics
        self.symptom_fix_count = 0
        self.root_cause_step = None
        self.delayed_failure_count = 0
        self.wasted_action_count = 0
        self.diagnosis_streak = 0
        self.diagnosed_targets = set() # Track services that have been diagnosed
        self.hidden_risk = 0.0
        self.side_effect_triggered = False
        self.RISK_THRESHOLD = 0.5 # Lowered
        
        # Hidden Root Cause + Delayed Consequence mechanism
        self.true_root_cause = getattr(task, "true_root_cause", None)
        self.surface_symptom_target = getattr(task, "surface_symptom_target", None)
        self.fake_recovery_timer = None
        self.root_cause_fixed = False
        
        self.dependencies = self._initialize_dependencies()
        self.grader = IncidentGrader()
        self._update_metrics()
        self._update_alerts() # Initial alerts based on status
        self.system_stability = self._calculate_stability() # Correctly calculate initial stability
        self.previous_stability = self.system_stability  # Store previous stability for reward calculation
        self.is_done = False  # Track episode termination state

    def _initialize_dependencies(self) -> Dict[str, List[str]]:
        if self.task.id == "hard-cascading-failure":
            return {
                "db": ["auth"],
                "auth": ["payments"],
                "payments": ["frontend"]
            }
        elif self.task.id == "hard-cascading-ambiguous":
            return {
                "payments": ["auth", "frontend"],
                "auth": ["frontend"]
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
            if service.status == ServiceStatus.UP:
                service.latency = 20 + (10 * self.system_strain)
                service.error_rate = 0.01 + (0.05 * self.system_strain)
            elif service.status == ServiceStatus.DEGRADED:
                service.latency = 200 + (100 * self.system_strain)
                service.error_rate = 0.15 + (0.1 * self.system_strain)
            else:
                service.latency = 1000.0
                service.error_rate = 1.0
            
            # Apply effect of isolate_service (latency increase on dependents)
            for isolated_name in self.isolated_services:
                if isolated_name in self.dependencies and service.name in self.dependencies[isolated_name]:
                    service.latency *= 1.5  # Significant latency increase for dependents of isolated services
            
            # Apply effect of drain_traffic (reduce error_rate on frontend)
            if service.name == "frontend" and self.drained_services:
                # If any service is drained, frontend error rate drops temporarily
                service.error_rate *= 0.5
            
            # Apply effect of partial fixes (e.g., restart on bad_deployment)
            if service.name in self.partial_fixes:
                service.error_rate = 0.5

            # Add controlled noise (±5%)
            noise_latency = self.random.uniform(-0.05, 0.05)
            noise_error = self.random.uniform(-0.05, 0.05)
            
            service.latency *= (1 + noise_latency)
            service.error_rate *= (1 + noise_error)

            # Apply Hidden Risk Consequence: Subtle latency increase as risk builds
            if self.hidden_risk > 0:
                # Up to 20% latency increase as risk approaches threshold
                risk_multiplier = 1.0 + (min(self.hidden_risk, self.RISK_THRESHOLD) * 0.2)
                service.latency *= risk_multiplier

            # Ensure bounds
            service.latency = max(1.0, service.latency)
            service.error_rate = max(0.0, min(1.0, service.error_rate))

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
            system_strain=self.system_strain,
            symptom_fix_count=self.symptom_fix_count,
            root_cause_step=self.root_cause_step,
            delayed_failure_count=self.delayed_failure_count,
            wasted_action_count=self.wasted_action_count,
            diagnosed_targets=list(self.diagnosed_targets),
            hidden_risk=self.hidden_risk,
            side_effect_triggered=self.side_effect_triggered
        )

    def state(self) -> State:
        """Pure read of the current state without side effects."""
        return self.get_state()

    def step(self, action: Action) -> Dict[str, Any]:
        try:
            # Check if episode has already terminated
            if hasattr(self, "is_done") and self.is_done:
                return {
                    "state": self.get_state(),
                    "reward": 0.0,
                    "done": True,
                    "info": {
                        "message": "Episode already completed. Please reset.",
                        "type": "terminated"
                    }
                }
            
            self.time_step += 1
            
            # Wasted Actions tracking
            if len(self.history) > 0:
                prev_action = self.history[-1]
                if action.action_type == prev_action.action_type and action.target == prev_action.target:
                    self.wasted_action_count += 1
            
            if action.action_type in [ActionType.CHECK_LOGS, ActionType.CHECK_METRICS]:
                self.diagnosis_streak += 1
                if self.diagnosis_streak > 3:
                    self.wasted_action_count += 1
            else:
                self.diagnosis_streak = 0
            
            # Root Cause Detection Step tracking
            if self.root_cause_step is None and self.true_root_cause and action.target == self.true_root_cause:
                self.root_cause_step = self.time_step
            
            # Symptom Fix tracking
            fix_actions = [ActionType.RESTART_SERVICE, ActionType.SCALE, ActionType.ROLLBACK_SERVICE, ActionType.OPTIMIZE_DB]
            if action.action_type in fix_actions and self.true_root_cause and action.target != self.true_root_cause:
                self.symptom_fix_count += 1
            
            self.history.append(action)
            
            # Store previous state before applying action
            prev_state = self.get_state()
            
            # Store previous stability for reward calculation
            self.previous_stability = self.system_stability
            
            # Add cost
            self.total_cost += self.ACTION_COSTS.get(action.action_type, 0.0)
            
            # Apply action and get result
            reward_info = self._apply_action(action)
            
            # Safely extract action type and increment bad_actions if needed
            action_type = reward_info.get("type", "unknown")
            
            if action_type in ["wrong_fix", "useless_action", "unknown"]:
                self.bad_actions += 1
                # Log the bad action for debugging
                self.logs.append(f"Bad action detected: {action_type}")
                # Initial instability penalty for wrong/useless actions
                self.system_stability = max(0.0, self.system_stability - 0.1)
                # Increase system strain
                self.system_strain += 0.2
            
            # Decrement re-degradation timers
            for service_name in list(self.redegrade_timers.keys()):
                self.redegrade_timers[service_name] -= 1
                if self.redegrade_timers[service_name] <= 0:
                    target_service = self._get_service(service_name)
                    if target_service and target_service.status == ServiceStatus.UP:
                        # Only re-degrade if root cause (payments) is still not fixed
                        payments_service = self._get_service("payments")
                        if payments_service and payments_service.status != ServiceStatus.UP:
                            target_service.status = ServiceStatus.DEGRADED
                            target_service.error_rate = 0.6
                            self.logs.append(f"WARN: {service_name.capitalize()} degraded again due to unresolved shared dependency")
                    del self.redegrade_timers[service_name]
            
            # Handle Hidden Root Cause + Delayed Consequence mechanism
            if self.fake_recovery_timer is not None:
                self.fake_recovery_timer -= 1
                if self.fake_recovery_timer <= 0:
                    self.delayed_failure_count += 1
                    # Timer expired, system degrades again
                    if self.surface_symptom_target:
                        symptom_service = self._get_service(self.surface_symptom_target)
                        if symptom_service and symptom_service.status == ServiceStatus.UP:
                            symptom_service.status = ServiceStatus.DEGRADED
                            symptom_service.error_rate = 0.7
                            self.logs.append("WARN: service degraded again due to unresolved underlying dependency")
                            self.logs.append("WARN: recurring failure detected — root cause unresolved")
                            self.system_strain += 0.3
                    self.fake_recovery_timer = None
            
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
            
            # Update episode termination state
            self.is_done = done
            
            # Counterfactual Consequence System logic
            if self.hidden_risk >= self.RISK_THRESHOLD and not self.side_effect_triggered:
                self.side_effect_triggered = True
                # Trigger a delayed side effect on a random healthy service
                healthy_services = [s for s in self.services if s.status == ServiceStatus.UP]
                if healthy_services:
                    target = self.random.choice(healthy_services)
                    target.status = ServiceStatus.DEGRADED
                    target.error_rate = 0.5
                    self.logs.append(f"CRITICAL: Unexpected degradation in {target.name} due to system-wide instability")
                    self.logs.append("INFO: Root cause analysis indicates cascading instability from previous risky actions")

            # Subtle signals based on hidden_risk
            if self.hidden_risk > (self.RISK_THRESHOLD * 0.4) and self.time_step % 2 == 0:
                instability_logs = [
                    "WARN: System entropy increasing beyond normal operating parameters",
                    "INFO: Minor synchronization delays detected in secondary cluster",
                    "WARN: Resource contention observed in shared infrastructure"
                ]
                self.logs.append(self.random.choice(instability_logs))

            # Add recovery log if all services are UP
            if self._all_services_up() and done:
                # Check if the final fix was a lucky guess
                if reward_info.get("is_lucky_guess"):
                    self.logs.append("System recovered, but cause was not diagnosed. Partial success.")
                    # If it was a lucky guess, we don't end the episode immediately if we want to force validation
                    # But the requirement says "Do NOT mark task fully successful immediately" 
                    # "Require one more validation step OR apply penalty"
                    # I'll make it NOT done if it's a lucky guess and all services are up.
                    done = False
                    self.is_done = False
                else:
                    self.logs.append("System fully recovered. Episode complete.")
            
            # Calculate meaningful reward signal
            reward = self._calculate_reward(reward_info, action)
            
            # Get state after applying action
            curr_state = self.get_state()
            
            # Add score breakdown if episode is done
            if done:
                score_result = self.grader.grade_episode(curr_state, self.task)
                reward_info["score_breakdown"] = {
                    "success_score": score_result.root_cause_score,
                    "efficiency_score": score_result.efficiency_score,
                    "cost_efficiency_score": score_result.cost_efficiency_score,
                    "stability_score": score_result.stability_score,
                    "sequence_score": score_result.sequence_score,
                    "final_score": score_result.final_score,
                    "sequence_bonus_applied": score_result.sequence_bonus_applied,
                    "wrong_order_penalty_applied": score_result.wrong_order_penalty_applied
                }
            else:
                score_result = None
                reward_info["score_breakdown"] = None
            
            # Store trajectory step
            self.state_history.append({
                "action": action,
                "state_before": prev_state,
                "state_after": curr_state,
                "reward": reward,
                "reward_info": reward_info
            })

            # Add diagnostic metrics to info
            reward_info.update({
                "symptom_fixes": self.symptom_fix_count,
                "root_cause_step": self.root_cause_step,
                "delayed_failures": self.delayed_failure_count,
                "wasted_actions": self.wasted_action_count,
                "failure_type": score_result.failure_type if score_result else None
            })

            return {
                "state": curr_state,
                "reward": reward,
                "done": done,
                "info": reward_info
            }
        except Exception as e:
            # Catch-all to prevent crash
            print(f"Error in IncidentEnv.step: {e}")
            return {
                "state": self.get_state(),
                "reward": -0.5,
                "done": False,
                "info": {"type": "error", "message": str(e)}
            }

    def _calculate_reward(self, reward_info: Dict[str, Any], action: Action) -> float:
        """Calculate meaningful reward signal based on action outcomes and system state."""
        # Use the grader to calculate base step reward
        reward = self.grader.calculate_step_reward(action, reward_info, self)
        
        # Apply lucky guess penalty (50% reduction)
        if reward_info.get("is_lucky_guess"):
            reward *= 0.5
            self.logs.append(f"Lucky guess detected on {action.target}! Reward reduced.")
        
        # Risk build-up from other actions
        if reward_info.get("type") == "wrong_fix":
            self.hidden_risk += 0.3
        elif action.action_type == ActionType.SCALE and reward_info.get("type") != "correct_fix":
            # Excessive scaling on wrong targets increases infrastructure strain
            self.hidden_risk += 0.25
        elif action.action_type == ActionType.OPTIMIZE_DB and reward_info.get("type") == "useless_action":
            # Running heavy DB optimization unnecessarily increases risk
            self.hidden_risk += 0.2
        
        # Lucky guesses increase hidden risk (guessing is unstable)
        if reward_info.get("is_lucky_guess"):
            self.hidden_risk += 0.3
        
        # Successful fixes SLIGHTLY reduce risk (stabilization)
        if reward_info.get("type") == "correct_fix" and not reward_info.get("is_lucky_guess"):
            self.hidden_risk = max(0.0, self.hidden_risk - 0.1)

        # Time penalty per step
        reward -= 0.05
        
        # Bonus for stability improvement
        stability_change = self.system_stability - self.previous_stability
        if stability_change > 0:
            reward += 0.2
        
        # Bonus for all services UP
        if self._all_services_up() and not reward_info.get("is_lucky_guess"):
            reward += 0.3
        
        # Penalty for significant system strain increase
        strain_change = self.system_strain - self.previous_strain
        if strain_change > 0.2:
            reward -= 0.2
        
        # Update previous_strain for next step
        self.previous_strain = self.system_strain
        
        # Penalty for excessive bad actions
        if self.bad_actions > 2:
            reward -= 0.3
        
        # Clip reward between [-1, 1]
        reward = max(-1.0, min(1.0, reward))
        
        return reward

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
        
        # Bonus for fixing root cause early 
        if any( 
            s.name == "db" and s.status == ServiceStatus.UP 
            for s in self.services 
        ): 
            stability += 0.1 
        
        # Incorporate history-based penalties (bad actions and risky actions)
        # Strain penalty: reduces stability linearly
        stability -= (0.1 * self.system_strain)
        
        # Apply diminishing returns: harder to reach 1.0 from 0.7 than 0.4 from 0.2
        # If stability > 0.7, apply a damping factor to the excess
        if stability > 0.7:
            stability = 0.7 + (stability - 0.7) * 0.5
            
        return max(0.0, min(1.0, stability))

    def _get_service(self, name: str) -> Service:
        return next((s for s in self.services if s.name == name), None)

    def _apply_cascading_failures(self):
        # Deterministic cascading failures based on dependencies
        for root, dependents in self.dependencies.items():
            # If service is isolated, it doesn't cause cascading failures downstream
            if root in self.isolated_services:
                continue
                
            root_service = self._get_service(root)
            if not root_service:
                continue
                
            for dep_name in dependents:
                dep_service = self._get_service(dep_name)
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
                    # Cascading degradation (85% probability for more realism)
                    if dep_service.status == ServiceStatus.UP and self.random.random() < 0.85:
                        dep_service.status = ServiceStatus.DEGRADED
                elif root_service.status == ServiceStatus.UP:
                    # Natural recovery: if root is UP and dependent is DEGRADED, it recovers
                    # ONLY if the dependent was originally degraded due to this root cause
                    # and not due to its own issues (requires correct action to fix)
                    if dep_service.status == ServiceStatus.DEGRADED:
                        # Check if this degradation was solely due to this root cause
                        # For hard task: payments degradation is from auth being down, which is from db
                        # So payments needs auth to be up first, then it can recover
                        # This prevents premature recovery when root cause isn't fully resolved
                        pass  # Recovery handled by _evolve_env with proper sequencing

        # Recovery for downstream dependencies
        for service in self.services:
            if service.status == ServiceStatus.DEGRADED:
                # Find all services that this service depends on
                # (We need to invert the self.dependencies map which is root -> dependents)
                upstreams = [root for root, deps in self.dependencies.items() if service.name in deps]
                
                if upstreams:
                    dependencies_healthy = all(
                        self._get_service(dep).status == ServiceStatus.UP
                        for dep in upstreams
                    )

                    if dependencies_healthy:
                        # Prevent automatic recovery for latent root cause task symptom
                        if self.task.id == "hard-latent-root-cause" and service.name == self.surface_symptom_target:
                            if not self.root_cause_fixed:
                                continue
                                
                        service.status = ServiceStatus.UP
                        service.latency = 20
                        service.error_rate = 0.01
                        self.logs.append(f"{service.name.capitalize()} recovered as dependencies stabilized")

    def _update_logs(self, action: Action, reward_info: Dict[str, Any]):
        new_logs = []
        
        # Log action result
        if reward_info["type"] == "correct_fix":
            new_logs.append(f"{action.target.capitalize()} service recovered successfully")
        elif reward_info["type"] == "wrong_fix":
            new_logs.append(f"Failed to fix {action.target.capitalize()}: dependency unresolved")
        elif reward_info["type"] == "useless_action":
            new_logs.append(f"Action {action.action_type} on {action.target} had no impact")
            
        # Add a random neutral log (25% chance)
        if self.random.random() < 0.25:
            neutral_logs = [
                "INFO: Background job completed successfully",
                "INFO: System health check passed",
                "INFO: Routine log rotation completed",
                "INFO: Minor latency variation observed in secondary cluster"
            ]
            new_logs.append(self.random.choice(neutral_logs))

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
        
        # Add misleading logs for hard task (seeded randomness)
        if self.task.id == "hard-cascading-failure":
            # Every 2 steps, add a misleading log that blames downstream services
            if self.time_step % 2 == 0:
                misleading_logs = [
                    "WARN: Payments service instability detected (possible root cause)",
                    "ERROR: Frontend experiencing cascading failures from payments",
                    "WARN: Auth service showing signs of failure - investigate immediately",
                    "INFO: Recommendation: Focus on stabilizing payments service first"
                ]
                # Use deterministic randomness based on seed
                new_logs.append(self.random.choice(misleading_logs))
            
            # At step 3, add another strong misdirection
            if self.time_step == 3:
                new_logs.append("CRITICAL: Multiple services failing - prioritize immediate recovery over root cause analysis")
        
        # Add hints and signals for hard-latent-root-cause task
        if self.task.id == "hard-latent-root-cause":
            # Subtle hint about hidden dependency early on
            if self.time_step == 1:
                new_logs.append("INFO: cross-service dependency check: auth -> (unresolved_upstream)")
            
            # Subtle hint about the true root cause
            if self.time_step == 3:
                new_logs.append("WARN: payments service internal queue depth increasing subtly")
            
            # Another hint if things are still bad
            if self.time_step == 6:
                new_logs.append("INFO: deployment logs show payments-v3.5.0 has shared resources with auth mesh")

        # Add to logs and limit to last 10
        self.logs.extend(new_logs)
        self.logs = self.logs[-10:]


    def _apply_action(self, action: Action) -> Dict[str, Any]:
        target_service = next((s for s in self.services if s.name == action.target), None)
        
        # Hidden Root Cause mechanism
        if self.true_root_cause and action.target == self.true_root_cause and action.action_type in [ActionType.RESTART_SERVICE, ActionType.ROLLBACK_SERVICE, ActionType.SCALE]:
            # Fixing the true root cause
            self.fake_recovery_timer = None
            self.root_cause_fixed = True
            if target_service:
                target_service.status = ServiceStatus.UP
                target_service.error_rate = 0.01
            # Dependent services recover
            if self.surface_symptom_target:
                symptom_service = self._get_service(self.surface_symptom_target)
                if symptom_service:
                    symptom_service.status = ServiceStatus.UP
                symptom_service.error_rate = 0.01
            
            is_lucky_guess = action.target not in self.diagnosed_targets
            return {"type": "correct_fix", "target": action.target, "root_cause_fixed": True, "is_lucky_guess": is_lucky_guess}
        
        if self.surface_symptom_target and action.target == self.surface_symptom_target and action.action_type in [ActionType.RESTART_SERVICE, ActionType.ROLLBACK_SERVICE, ActionType.SCALE]:
            # Fixing the surface symptom (wrong target)
            if target_service and target_service.status != ServiceStatus.UP:
                target_service.status = ServiceStatus.UP
                target_service.error_rate = 0.05 # Lower error rate, temporary improvement
                self.fake_recovery_timer = 2
                return {"type": "temporary_fix", "target": action.target, "surface_symptom_fixed": True}
        
        # Identify root cause for the current task
        root_cause_service = None
        if self.task.id == "hard-cascading-failure":
            root_cause_service = next((s for s in self.services if s.name == "db"), None)
        elif self.task.id == "medium-payments-degraded":
            root_cause_service = next((s for s in self.services if s.name == "payments"), None)
        elif self.task.id == "hard-bad-deployment":
            root_cause_service = next((s for s in self.services if s.name == "auth"), None) # Let's assume auth is the target for bad deployment in the new task
        elif self.task.id == "hard-cascading-ambiguous":
            root_cause_service = next((s for s in self.services if s.name == "payments"), None)
        
        # Check if agent is repeatedly targeting wrong services (for hard task)
        if (self.task.id == "hard-cascading-failure" or self.task.id == "hard-bad-deployment" or self.task.id == "hard-cascading-ambiguous") and root_cause_service and root_cause_service.status != ServiceStatus.UP:
            # Count how many times agent targeted non-root-cause services
            wrong_target_count = 0
            for past_action in self.history:
                if past_action.target != root_cause_service.name and past_action.action_type not in [ActionType.CHECK_LOGS, ActionType.CHECK_METRICS, ActionType.IGNORE]:
                    wrong_target_count += 1
            
            # If this is another wrong action and we already have multiple wrong actions
            if wrong_target_count >= 2 and action.target != root_cause_service.name and action.action_type not in [ActionType.CHECK_LOGS, ActionType.CHECK_METRICS, ActionType.IGNORE]:
                # Increase system strain more aggressively
                self.system_strain += 0.15
                # Add penalty log
                self.logs.append("Repeated incorrect mitigation detected - focusing on symptoms instead of root cause")

        if action.action_type == ActionType.RESTART_SERVICE:
            if not target_service:
                return {"type": "unknown", "target": action.target}
            
            # Special case for bad_deployment: restart gives partial fix
            if self.task.id == "hard-bad-deployment" and target_service.name == "auth":
                if target_service.status != ServiceStatus.UP:
                    target_service.status = ServiceStatus.DEGRADED
                    if target_service.name not in self.partial_fixes:
                        self.partial_fixes.append(target_service.name)
                    return {"type": "partial_fix", "target": action.target}
                
            # If target has an unresolved upstream dependency that is DOWN/DEGRADED
            for root, dependents in self.dependencies.items():
                if action.target in dependents:
                    upstream = next((s for s in self.services if s.name == root), None)
                    if upstream and upstream.status != ServiceStatus.UP:
                        return {"type": "wrong_fix", "target": action.target}

            if target_service.status == ServiceStatus.DOWN:
                target_service.status = ServiceStatus.UP
                is_lucky_guess = action.target not in self.diagnosed_targets
                return {"type": "correct_fix", "target": action.target, "is_lucky_guess": is_lucky_guess}
            elif target_service.status == ServiceStatus.DEGRADED:
                # Special case for hard-cascading-ambiguous: restart auth while payments is down
                if self.task.id == "hard-cascading-ambiguous" and target_service.name == "auth":
                    payments_service = self._get_service("payments")
                    if payments_service and payments_service.status != ServiceStatus.UP:
                        target_service.status = ServiceStatus.UP
                        self.redegrade_timers["auth"] = 2
                        return {"type": "temporary_fix", "target": action.target}
                
                # Restart does NOT fix degraded state - requires correct action (e.g., SCALE, OPTIMIZE_DB)
                return {"type": "wrong_fix", "target": action.target}
            else:
                return {"type": "useless_action", "target": action.target}
        
        elif action.action_type == ActionType.ROLLBACK_SERVICE:
            if not target_service:
                return {"type": "unknown", "target": action.target}
            
            if self.task.id == "hard-bad-deployment" and target_service.name == "auth":
                target_service.status = ServiceStatus.UP
                target_service.error_rate = 0.01
                if target_service.name in self.partial_fixes:
                    self.partial_fixes.remove(target_service.name)
                is_lucky_guess = action.target not in self.diagnosed_targets
                return {"type": "correct_fix", "target": action.target, "is_lucky_guess": is_lucky_guess}
            elif self.task.id == "hard-cascading-ambiguous" and target_service.name == "payments":
                target_service.status = ServiceStatus.UP
                target_service.error_rate = 0.01
                # If payments is rolled back, auth should also recover (it was only degraded due to payments)
                auth_service = self._get_service("auth")
                if auth_service and auth_service.status != ServiceStatus.DOWN:
                    auth_service.status = ServiceStatus.UP
                    auth_service.error_rate = 0.01
                    # Clear any pending redegrade timer
                    if "auth" in self.redegrade_timers:
                        del self.redegrade_timers["auth"]
                is_lucky_guess = action.target not in self.diagnosed_targets
                return {"type": "correct_fix", "target": action.target, "is_lucky_guess": is_lucky_guess}
            else:
                # Penalty on wrong target
                self.system_stability = max(0.0, self.system_stability - 0.1)
                return {"type": "wrong_fix", "target": action.target}

        elif action.action_type == ActionType.ISOLATE_SERVICE:
            if not target_service:
                return {"type": "unknown", "target": action.target}
            
            if target_service.name not in self.isolated_services:
                self.isolated_services.append(target_service.name)
                return {"type": "tactical_move", "target": action.target}
            else:
                return {"type": "useless_action", "target": action.target}

        elif action.action_type == ActionType.DRAIN_TRAFFIC:
            if not target_service:
                return {"type": "unknown", "target": action.target}
            
            if target_service.name not in self.drained_services:
                self.drained_services.append(target_service.name)
                return {"type": "tactical_move", "target": action.target}
            else:
                return {"type": "useless_action", "target": action.target}

        elif action.action_type == ActionType.RESTORE_TRAFFIC:
            if not target_service:
                return {"type": "unknown", "target": action.target}
            
            if target_service.name in self.drained_services:
                self.drained_services.remove(target_service.name)
                return {"type": "tactical_move", "target": action.target}
            else:
                return {"type": "wrong_fix", "target": action.target}

        elif action.action_type == ActionType.ESCALATE:
            self.is_done = True
            return {"type": "escalate", "target": "all"}

        elif action.action_type == ActionType.SCALE:
            if not target_service:
                return {"type": "unknown", "target": action.target}

            # Scaling while root cause exists is expensive but might work partially
            if root_cause_service and root_cause_service.status != ServiceStatus.UP and action.target != root_cause_service.name:
                self.total_cost += 2.0 # Extra cost for scaling symptom instead of root cause
                # Additional strain for treating symptoms
                if self.task.id == "hard-cascading-failure" or self.task.id == "hard-bad-deployment":
                    self.system_strain += 0.1
                
            if target_service.status == ServiceStatus.DEGRADED:
                target_service.status = ServiceStatus.UP
                is_lucky_guess = action.target not in self.diagnosed_targets
                return {"type": "correct_fix", "target": action.target, "is_lucky_guess": is_lucky_guess}
            else:
                return {"type": "useless_action", "target": action.target}

        elif action.action_type == ActionType.OPTIMIZE_DB:
            if target_service and target_service.name == "db" and target_service.status == ServiceStatus.DEGRADED:
                target_service.status = ServiceStatus.UP
                is_lucky_guess = action.target not in self.diagnosed_targets
                return {"type": "correct_fix", "target": action.target, "is_lucky_guess": is_lucky_guess}
            else:
                return {"type": "useless_action", "target": action.target}
        
        elif action.action_type == ActionType.CHECK_LOGS or action.action_type == ActionType.CHECK_METRICS:
            self.diagnosed_targets.add(action.target)
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

            # Sequential recovery with 1-step delay per level (±1 step variation)
            # Recovery is slower if strain is high
            base_threshold = 2 if self.system_strain < 0.5 else 3
            recovery_threshold = base_threshold + self.random.randint(-1, 1)
            recovery_threshold = max(1, recovery_threshold) # Ensure at least 1 step
            
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
        if self.fake_recovery_timer is not None:
            return False
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
