from models.schemas import EpisodeResult, State, Task, ActionType, ServiceStatus

class IncidentGrader:
    def grade_episode(self, state: State, task: Task) -> EpisodeResult:
        root_cause_score = self._calculate_root_cause_score(state, task)
        efficiency_score = self._calculate_efficiency_score(state, task)
        cost_efficiency_score = self._calculate_cost_efficiency_score(state)
        stability_score = state.system_stability
        damage_score, health_penalty = self._calculate_damage_and_health_penalty(state, task)
        
        # New: Passive behavior penalty (ignore() when root cause NOT fixed)
        ignore_penalty = self._calculate_ignore_penalty(state, root_cause_score)
        
        # New: Optimal execution bonus
        optimal_bonus = self._calculate_optimal_bonus(state, task)

        # Refined weights for > 0.85 potential:
        # Final Stability: 40%
        # Efficiency (steps): 15%
        # Cost Efficiency: 15%
        # Root Cause Quality: 30%
        final_score = (
            0.30 * root_cause_score +
            0.15 * efficiency_score +
            0.15 * cost_efficiency_score +
            0.40 * stability_score
        ) - health_penalty - (0.1 * state.system_strain) - ignore_penalty + optimal_bonus

        # Strict separation:
        if root_cause_score == 0:
            final_score = min(final_score, 0.3)
        
        if state.bad_actions >= 2:
            final_score = min(final_score, 0.25)

        return EpisodeResult(
            final_score=max(0.0, min(1.0, final_score)),
            root_cause_score=root_cause_score,
            efficiency_score=efficiency_score,
            damage_score=damage_score,
            cost_efficiency_score=cost_efficiency_score,
            stability_score=stability_score,
            steps_used=state.time_step,
            bad_actions=state.bad_actions,
            system_health_penalty=health_penalty,
            total_cost=state.total_cost,
            system_strain=state.system_strain,
            optimal_bonus=optimal_bonus
        )

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
