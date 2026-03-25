from models.schemas import EpisodeResult, State, Task, ActionType, ServiceStatus

class IncidentGrader:
    def grade_episode(self, state: State, task: Task) -> EpisodeResult:
        root_cause_score = self._calculate_root_cause_score(state, task)
        efficiency_score = self._calculate_efficiency_score(state, task)
        cost_efficiency_score = self._calculate_cost_efficiency_score(state)
        stability_score = state.system_stability
        damage_score, health_penalty = self._calculate_damage_and_health_penalty(state, task)

        final_score = (
            0.3 * root_cause_score +
            0.3 * efficiency_score +
            0.2 * cost_efficiency_score +
            0.2 * stability_score
        ) - health_penalty

        if root_cause_score == 0:
            final_score = min(final_score, 0.6)

        return EpisodeResult(
            final_score=max(0, final_score),
            root_cause_score=root_cause_score,
            efficiency_score=efficiency_score,
            damage_score=damage_score,
            cost_efficiency_score=cost_efficiency_score,
            stability_score=stability_score,
            steps_used=state.time_step,
            bad_actions=state.bad_actions,
            system_health_penalty=health_penalty,
            total_cost=state.total_cost
        )

    def _calculate_root_cause_score(self, state: State, task: Task) -> float:
        if task.difficulty == "hard":
            # Stricter: only the correct action counts
            if any(action.action_type == ActionType.OPTIMIZE_DB and action.target == "db" for action in state.history):
                return 1.0
            return 0.0
        else:
            # For easy/medium, full recovery is expected
            if all(s.status == ServiceStatus.UP for s in state.services):
                 return 1.0
            return 0.0

    def _calculate_efficiency_score(self, state: State, task: Task) -> float:
        if task.max_steps == 0:
            return 0.0
        return max(0, 1 - state.time_step / task.max_steps)

    def _calculate_cost_efficiency_score(self, state: State) -> float:
        # Assuming max reasonable cost is around 15.0 for a task
        max_reasonable_cost = 15.0
        return max(0.0, 1.0 - state.total_cost / max_reasonable_cost)

    def _calculate_damage_and_health_penalty(self, state: State, task: Task) -> (float, float):
        damage_score = max(0, 1 - state.bad_actions / 3)
        
        initial_down_services = sum(1 for s in task.initial_services if s.status != ServiceStatus.UP)
        current_down_services = sum(1 for s in state.services if s.status != ServiceStatus.UP)
        
        health_penalty = 0.0
        if current_down_services > initial_down_services:
            health_penalty = 0.1 * (current_down_services - initial_down_services)
            
        return damage_score, health_penalty
