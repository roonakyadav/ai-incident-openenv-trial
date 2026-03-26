from models.schemas import EpisodeResult, State, Task, ActionType, ServiceStatus

class IncidentGrader:
    def grade_episode(self, state: State, task: Task) -> EpisodeResult:
        # Evaluate success based on task conditions
        success_score = self._evaluate_success_conditions(state, task)
        efficiency_score = self._calculate_efficiency_score(state, task)
        cost_efficiency_score = self._calculate_cost_efficiency_score(state)
        stability_score = state.system_stability
        
        # Apply failure condition penalties
        final_score = (
            0.5 * success_score +
            0.2 * efficiency_score +
            0.2 * cost_efficiency_score +
            0.1 * stability_score
        )
        
        # Apply failure condition penalties
        final_score = self._apply_failure_conditions(state, task, final_score)
        
        # Keep existing damage score calculation for backward compatibility
        damage_score, health_penalty = self._calculate_damage_and_health_penalty(state, task)
        
        # Additional penalty for system strain
        final_score -= (0.1 * state.system_strain)
        
        return EpisodeResult(
            final_score=max(0.0, min(1.0, final_score)),
            root_cause_score=success_score,  # Repurposed as success score
            efficiency_score=efficiency_score,
            damage_score=damage_score,
            cost_efficiency_score=cost_efficiency_score,
            stability_score=stability_score,
            steps_used=state.time_step,
            bad_actions=state.bad_actions,
            system_health_penalty=health_penalty,
            total_cost=state.total_cost,
            system_strain=state.system_strain,
            optimal_bonus=0.0  # Removed optimal bonus as it's now part of success conditions
        )

    def _evaluate_success_conditions(self, state: State, task: Task) -> float:
        """Evaluate success based on task.success_conditions with equal weight per condition."""
        if not task.success_conditions:
            # Fallback: use default success criteria if no conditions defined
            return self._default_success_evaluation(state, task)
        
        num_conditions = len(task.success_conditions)
        if num_conditions == 0:
            return 0.0
            
        weight_per_condition = 1.0 / num_conditions
        score = 0.0
        
        for condition_name, condition_value in task.success_conditions.items():
            condition_score = self._evaluate_single_condition(state, task, condition_name, condition_value)
            score += weight_per_condition * condition_score
        
        return score
    
    def _evaluate_single_condition(self, state: State, task: Task, condition_name: str, condition_value: any) -> float:
        """Evaluate a single success condition and return score between 0.0 and 1.0."""
        
        if condition_name == "all_services_up":
            # Check if all services are UP
            all_up = all(s.status == ServiceStatus.UP for s in state.services)
            return 1.0 if all_up else 0.0
        
        elif condition_name == "no_service_down":
            # Ensure no service is DOWN
            none_down = all(s.status != ServiceStatus.DOWN for s in state.services)
            return 1.0 if none_down else 0.0
        
        elif condition_name == "max_error_rate":
            # Check if all error rates are below threshold
            threshold = condition_value
            all_below = all(s.error_rate < threshold for s in state.services)
            return 1.0 if all_below else 0.0
        
        elif condition_name == "latency_threshold":
            # Check if all latencies are below threshold
            threshold = condition_value
            all_below = all(s.latency < threshold for s in state.services)
            return 1.0 if all_below else 0.0
        
        elif condition_name == "cost_limit":
            # Check if total cost is within limit
            cost_limit = condition_value
            return 1.0 if state.total_cost <= cost_limit else 0.0
        
        elif condition_name == "stability_score":
            # Check if system stability meets threshold
            threshold = condition_value
            return 1.0 if state.system_stability >= threshold else 0.0
        
        elif condition_name == "strain_limit":
            # Check if system strain is within limit
            strain_limit = condition_value
            return 1.0 if state.system_strain <= strain_limit else 0.0
        
        elif condition_name == "root_cause_fixed":
            # Check if correct root cause action exists in history
            if task.id == "hard-cascading-failure":
                has_db_fix = any(
                    a.action_type == ActionType.OPTIMIZE_DB and a.target == "db" 
                    for a in state.history
                )
                return 1.0 if has_db_fix else 0.0
            elif task.id == "medium-payments-degraded":
                has_payments_fix = any(
                    (a.action_type == ActionType.SCALE or a.action_type == ActionType.RESTART_SERVICE) 
                    and a.target == "payments" 
                    for a in state.history
                )
                return 1.0 if has_payments_fix else 0.0
            else:
                # For easy tasks, check if the down service was restarted
                for service in task.initial_services:
                    if service.status == ServiceStatus.DOWN:
                        has_restart = any(
                            a.action_type == ActionType.RESTART_SERVICE and a.target == service.name
                            for a in state.history
                        )
                        return 1.0 if has_restart else 0.0
            return 0.0
        
        # Unknown condition - return 0
        return 0.0
    
    def _default_success_evaluation(self, state: State, task: Task) -> float:
        """Default success evaluation when no success_conditions are defined."""
        # All services up is the primary goal
        all_up = all(s.status == ServiceStatus.UP for s in state.services)
        if all_up:
            return 1.0
        
        # Partial credit based on how many services are up
        up_count = sum(1 for s in state.services if s.status == ServiceStatus.UP)
        return up_count / len(state.services) if state.services else 0.0
    
    def _apply_failure_conditions(self, state: State, task: Task, score: float) -> float:
        """Apply penalties based on task.failure_conditions."""
        if not task.failure_conditions:
            # Apply default failure conditions
            return self._default_failure_application(state, score)
        
        capped_score = score
        
        for condition_name, condition_value in task.failure_conditions.items():
            if condition_name == "bad_actions_limit":
                limit = condition_value
                if state.bad_actions > limit:
                    # Cap score when bad actions exceed limit
                    capped_score = min(capped_score, 0.3)
            
            elif condition_name == "system_stability_below":
                threshold = condition_value
                if state.system_stability < threshold:
                    # Reduce score significantly if stability is too low
                    capped_score = min(capped_score, 0.25)
            
            elif condition_name == "max_steps_exceeded":
                if condition_value and state.time_step >= task.max_steps:
                    # Penalty for using all steps without success
                    capped_score = min(capped_score, 0.5)
        
        return capped_score
    
    def _default_failure_application(self, state: State, score: float) -> float:
        """Default failure condition application when no failure_conditions are defined."""
        capped_score = score
        
        # Default bad actions limit
        if state.bad_actions >= 2:
            capped_score = min(capped_score, 0.25)
        
        # Default stability threshold
        if state.system_stability < 0.3:
            capped_score = min(capped_score, 0.2)
        
        return capped_score

    def _calculate_root_cause_score(self, state: State, task: Task) -> float:
        if task.difficulty == "hard":
            if any(action.action_type == ActionType.OPTIMIZE_DB and action.target == "db" for action in state.history):
                return 1.0
            return 0.0
        else:
            if all(s.status == ServiceStatus.UP for s in state.services):
                 return 1.0
            return 0.0

    def _calculate_efficiency_score(self, state: State, task: Task) -> float:
        if task.max_steps == 0:
            return 0.0
        return max(0.0, 1.0 - (state.time_step - 1) / task.max_steps) 

    def _calculate_cost_efficiency_score(self, state: State) -> float:
        max_reasonable_cost = 15.0 
        return max(0.0, 1.0 - state.total_cost / max_reasonable_cost)

    def _calculate_damage_and_health_penalty(self, state: State, task: Task) -> (float, float):
        damage_score = max(0.0, 1.0 - (state.bad_actions * 0.4)) 
        
        initial_down_services = sum(1 for s in task.initial_services if s.status != ServiceStatus.UP)
        current_down_services = sum(1 for s in state.services if s.status != ServiceStatus.UP)
        
        health_penalty = 0.0
        if current_down_services > initial_down_services:
            health_penalty = 0.2 * (current_down_services - initial_down_services)
            
        return damage_score, health_penalty

    def _calculate_ignore_penalty(self, state: State, root_cause_score: float) -> float:
        if root_cause_score > 0:
            return 0.0
            
        ignore_count = sum(1 for a in state.history if a.action_type == ActionType.IGNORE)
        return 0.1 * ignore_count 

    def _calculate_optimal_bonus(self, state: State, task: Task) -> float:
        if task.difficulty == "hard":
            has_early_fix = any(a.action_type == ActionType.OPTIMIZE_DB and a.target == "db" for a in state.history[:2])
            if has_early_fix and state.bad_actions == 0 and state.system_stability > 0.7:
                return 0.2 
        elif state.bad_actions == 0 and state.system_stability > 0.9:
            return 0.1
        return 0.0
