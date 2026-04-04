from models.schemas import EpisodeResult, State, Task, ActionType, ServiceStatus

class IncidentGrader:
    def grade_episode(self, state: State, task: Task) -> EpisodeResult:
        # Diagnostic classification logic
        failure_type = "Unknown"
        
        # Check if root cause fix was performed after diagnosis
        root_cause_diagnosed_before_fix = False
        if task.true_root_cause:
            # Find diagnosis index for root cause
            diagnosis_index = -1
            for i, action in enumerate(state.history):
                if action.target == task.true_root_cause and action.action_type in [ActionType.CHECK_LOGS, ActionType.CHECK_METRICS]:
                    diagnosis_index = i
                    break
            
            # Find fix index for root cause
            fix_index = -1
            for i, action in enumerate(state.history):
                if action.target == task.true_root_cause and action.action_type in [ActionType.RESTART_SERVICE, ActionType.ROLLBACK_SERVICE, ActionType.SCALE, ActionType.OPTIMIZE_DB]:
                    fix_index = i
                    break
            
            if diagnosis_index != -1 and fix_index != -1 and diagnosis_index < fix_index:
                root_cause_diagnosed_before_fix = True

        if state.symptom_fix_count == 0 and state.delayed_failure_count == 0 and root_cause_diagnosed_before_fix and state.root_cause_step is not None and state.root_cause_step <= 2:
            failure_type = "Efficient Reasoner"
        elif state.symptom_fix_count > 0 and state.delayed_failure_count > 0:
            failure_type = "Symptom Chaser"
        elif not root_cause_diagnosed_before_fix and state.root_cause_step is not None:
            failure_type = "Lucky Guesser"
        elif state.wasted_action_count >= 3 and state.root_cause_step is None:
            failure_type = "Stuck in Observation Loop"
        elif state.root_cause_step is not None and state.root_cause_step > 5:
            failure_type = "Late Corrector"

        # Evaluate success based on task conditions
        success_score = self._evaluate_success_conditions(state, task)
        efficiency_score = self._calculate_efficiency_score(state, task)
        cost_efficiency_score = self._calculate_cost_efficiency_score(state)
        stability_score = state.system_stability
        sequence_score = self._calculate_sequence_score(state, task)
        
        # Track bonus/penalty application for hard-cascading-ambiguous and hard-latent-root-cause
        sequence_bonus_applied = False
        wrong_order_penalty_applied = False
        
        # Apply special task-specific scoring for hard-latent-root-cause
        if task.id == "hard-latent-root-cause":
            # 1. Early correct fix bonus
            correct_fix_index = -1
            for i, action in enumerate(state.history):
                if action.target == getattr(task, "true_root_cause", "payments") and action.action_type in [ActionType.RESTART_SERVICE, ActionType.ROLLBACK_SERVICE]:
                    correct_fix_index = i
                    break
            
            if 0 <= correct_fix_index < 4:
                sequence_score += 0.35
                sequence_bonus_applied = True
            
            # 2. Penalty for fixing symptom before root cause
            symptom_fix_index = -1
            for i, action in enumerate(state.history):
                if action.target == getattr(task, "surface_symptom_target", "auth") and action.action_type in [ActionType.RESTART_SERVICE, ActionType.ROLLBACK_SERVICE]:
                    symptom_fix_index = i
                    break
            
            if symptom_fix_index != -1 and (correct_fix_index == -1 or symptom_fix_index < correct_fix_index):
                sequence_score -= 0.20
                wrong_order_penalty_applied = True

        # Apply special task-specific scoring for hard-cascading-ambiguous
        if task.id == "hard-cascading-ambiguous":
            # 1. Sequence bonus for early rollback on payments
            rollback_index = -1
            for i, action in enumerate(state.history):
                if action.action_type == ActionType.ROLLBACK_SERVICE and action.target == "payments":
                    rollback_index = i
                    break
            
            if 0 <= rollback_index < 5:
                sequence_score += 0.30
                sequence_bonus_applied = True
            
            # 2. Penalty for fixing auth before payments
            auth_restart_index = -1
            for i, action in enumerate(state.history):
                if action.action_type == ActionType.RESTART_SERVICE and action.target == "auth":
                    auth_restart_index = i
                    break
            
            if auth_restart_index != -1 and (rollback_index == -1 or auth_restart_index < rollback_index):
                sequence_score -= 0.15
                wrong_order_penalty_applied = True
            
            # 3. Productive diagnosis after payments fix if auth is still degraded (due to re-degrade)
            # Find when payments was fixed
            if rollback_index != -1:
                productive_diagnosis_count = 0
                for action in state.history[rollback_index+1:]:
                    if action.action_type in [ActionType.CHECK_METRICS, ActionType.CHECK_LOGS]:
                        productive_diagnosis_count += 1
                        if productive_diagnosis_count <= 2:
                            sequence_score += 0.05
        
        # Clamp sequence_score to [0.0, 1.0]
        sequence_score = max(0.0, min(1.0, sequence_score))
        
        # Apply failure condition penalties with sequence score integration
        final_score = (
            0.40 * success_score +
            0.15 * efficiency_score +
            0.20 * cost_efficiency_score +
            0.05 * stability_score +
            0.20 * sequence_score
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
            sequence_score=sequence_score,
            sequence_bonus_applied=sequence_bonus_applied,
            wrong_order_penalty_applied=wrong_order_penalty_applied,
            steps_used=state.time_step,
            bad_actions=state.bad_actions,
            system_health_penalty=health_penalty,
            total_cost=state.total_cost,
            system_strain=state.system_strain,
            optimal_bonus=sequence_score,  # Using sequence_score for optimal_bonus for visibility
            symptom_fix_count=state.symptom_fix_count,
            root_cause_step=state.root_cause_step,
            delayed_failure_count=state.delayed_failure_count,
            wasted_action_count=state.wasted_action_count,
            failure_type=failure_type,
            diagnosed_targets=state.diagnosed_targets
        )

    def _calculate_sequence_score(self, state: State, task: Task) -> float:
        """Calculate score based on action sequence and timing."""
        if not state.history or task.max_steps <= 0:
            return 0.0
            
        score = 0.0
        
        # a) Early root cause reward: find first occurrence of correct root cause action
        root_cause_index = -1
        for i, action in enumerate(state.history):
            is_correct = False
            if task.id == "hard-cascading-failure":
                if action.action_type == ActionType.OPTIMIZE_DB and action.target == "db":
                    is_correct = True
            elif task.id == "hard-bad-deployment":
                if action.action_type == ActionType.ROLLBACK_SERVICE and action.target == "auth":
                    is_correct = True
            elif task.id == "medium-payments-degraded":
                if action.target == "payments" and action.action_type in [ActionType.SCALE, ActionType.RESTART_SERVICE]:
                    is_correct = True
            else:
                # Easy tasks: find initial down service and check for restart
                for service in task.initial_services:
                    if service.status == ServiceStatus.DOWN:
                        if action.action_type == ActionType.RESTART_SERVICE and action.target == service.name:
                            is_correct = True
                            break
            
            if is_correct:
                root_cause_index = i
                break
        
        if root_cause_index != -1:
            score += max(0.0, 1.0 - (root_cause_index / task.max_steps))
            
        # b) Penalize delayed recovery: If all services become UP
        # In this env, the episode ends when all services are UP, so state.time_step is the index
        all_up = all(s.status == ServiceStatus.UP for s in state.services)
        if all_up:
            recovery_index = state.time_step
            score += max(0.0, 1.0 - (recovery_index / task.max_steps))
            
        # c) Penalize excessive useless/wrong actions early
        # Since we don't have per-step state, we approximate "bad actions" 
        # as any action that's not diagnosis, ignore, or correct fix
        bad_actions_early = 0
        half_max_steps = task.max_steps / 2
        for i, action in enumerate(state.history):
            if i >= half_max_steps:
                break
                
            # Check if this was a correct fix (reusing logic from part a)
            is_correct = False
            if task.id == "hard-cascading-failure":
                if action.action_type == ActionType.OPTIMIZE_DB and action.target == "db":
                    is_correct = True
            elif task.id == "hard-bad-deployment":
                if action.action_type == ActionType.ROLLBACK_SERVICE and action.target == "auth":
                    is_correct = True
            elif task.id == "medium-payments-degraded":
                if action.target == "payments" and action.action_type in [ActionType.SCALE, ActionType.RESTART_SERVICE]:
                    is_correct = True
            else:
                for service in task.initial_services:
                    if service.status == ServiceStatus.DOWN:
                        if action.action_type == ActionType.RESTART_SERVICE and action.target == service.name:
                            is_correct = True
                            break
            
            # If not correct fix and not diagnosis/ignore, it's potentially bad
            if not is_correct and action.action_type not in [ActionType.CHECK_LOGS, ActionType.IGNORE]:
                bad_actions_early += 1
        
        if half_max_steps > 0:
            penalty = bad_actions_early / half_max_steps
            score -= penalty * 0.5
            
        return max(0.0, min(1.0, score))

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
            # Replace with proportional score: (# services UP) / (total services)
            if not state.services:
                return 1.0
            up_count = sum(1 for s in state.services if s.status == ServiceStatus.UP)
            return up_count / len(state.services)
        
        elif condition_name == "no_service_down":
            # Replace with proportional score: (# services NOT DOWN) / (total services)
            if not state.services:
                return 1.0
            not_down_count = sum(1 for s in state.services if s.status != ServiceStatus.DOWN)
            return not_down_count / len(state.services)
        
        elif condition_name == "max_error_rate":
            # Score inversely proportional to error rate: threshold / actual_error_rate
            # Average across services
            if not state.services:
                return 1.0
            threshold = float(condition_value)
            scores = []
            for s in state.services:
                if s.error_rate <= threshold:
                    scores.append(1.0)
                elif s.error_rate == 0: # Should not happen if it's > threshold
                    scores.append(1.0)
                else:
                    scores.append(max(0.0, min(1.0, threshold / s.error_rate)))
            return sum(scores) / len(scores)
        
        elif condition_name == "latency_threshold":
            # Score inversely proportional to latency: threshold / actual_latency
            # Average across services
            if not state.services:
                return 1.0
            threshold = float(condition_value)
            scores = []
            for s in state.services:
                if s.latency <= threshold:
                    scores.append(1.0)
                elif s.latency == 0:
                    scores.append(1.0)
                else:
                    scores.append(max(0.0, min(1.0, threshold / s.latency)))
            return sum(scores) / len(scores)
        
        elif condition_name == "cost_limit":
            # Score = max(0.0, 1.0 - (total_cost / cost_limit))
            cost_limit = float(condition_value)
            if cost_limit <= 0:
                return 1.0 if state.total_cost <= 0 else 0.0
            return max(0.0, min(1.0, 1.0 - (state.total_cost / cost_limit)))
        
        elif condition_name == "stability_score":
            # Score = min(1.0, state.system_stability / threshold)
            threshold = float(condition_value)
            if threshold <= 0:
                return 1.0
            return max(0.0, min(1.0, state.system_stability / threshold))
        
        elif condition_name == "strain_limit":
            # Score = max(0.0, 1.0 - (state.system_strain / strain_limit))
            strain_limit = float(condition_value)
            if strain_limit <= 0:
                return 1.0 if state.system_strain <= 0 else 0.0
            return max(0.0, min(1.0, 1.0 - (state.system_strain / strain_limit)))
        
        elif condition_name == "root_cause_fixed":
            # Check if correct root cause action exists in history
            if task.id == "hard-cascading-failure":
                has_db_fix = any(
                    a.action_type == ActionType.OPTIMIZE_DB and a.target == "db" 
                    for a in state.history
                )
                return 1.0 if has_db_fix else 0.0
            elif task.id == "hard-bad-deployment":
                has_rollback = any(
                    a.action_type == ActionType.ROLLBACK_SERVICE and a.target == "auth"
                    for a in state.history
                )
                return 1.0 if has_rollback else 0.0
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

    def calculate_step_reward(self, action, reward_info, env) -> float:
        """Calculate step-wise reward for new and existing actions."""
        reward = 0.0
        
        # New actions reward logic
        if action.action_type == ActionType.CHECK_METRICS:
            reward += 0.05
        
        elif action.action_type == ActionType.DRAIN_TRAFFIC:
            target = env._get_service(action.target)
            if target and target.status != ServiceStatus.UP:
                reward += 0.10
            else:
                reward -= 0.10
        
        elif action.action_type == ActionType.RESTORE_TRAFFIC:
            # Check if it was actually drained (reward_info["type"] would be "tactical_move" if successful)
            if reward_info.get("type") == "tactical_move":
                reward += 0.05
            else:
                reward -= 0.05
        
        elif action.action_type == ActionType.ROLLBACK_SERVICE:
            if reward_info.get("type") == "correct_fix":
                reward += 0.75
            else:
                reward -= 0.20
        
        elif action.action_type == ActionType.ISOLATE_SERVICE:
            target = env._get_service(action.target)
            if target and target.status != ServiceStatus.UP:
                reward += 0.05
            else:
                reward -= 0.10
        
        elif action.action_type == ActionType.ESCALATE:
            # Escalate returns a partial score based on system health
            health = env._calculate_system_health()
            reward += health * 0.5
            
        # Existing actions base rewards (from core.py)
        if reward_info["type"] == "correct_fix" and action.action_type != ActionType.ROLLBACK_SERVICE:
            reward += 0.3
        elif reward_info["type"] == "temporary_fix":
            reward += 0.1
        elif reward_info["type"] == "diagnosis" and action.action_type != ActionType.CHECK_METRICS:
            reward += 0.1
        elif reward_info["type"] == "useless_action":
            reward -= 0.2
        elif reward_info["type"] == "wrong_fix" and action.action_type != ActionType.ROLLBACK_SERVICE:
            reward -= 0.4
        elif reward_info["type"] == "partial_fix":
            reward += 0.15

        return reward
