from models.schemas import State, Action, ActionType, ServiceStatus
from typing import Dict, List, Optional

class BaselineAgent:
    def decide_action(self, state: State) -> Action:
        services_by_name = {s.name: s for s in state.services}
        
        # 1. Identify root cause first based on dependencies
        # Build inverse dependency map (who depends on what)
        # dependencies: { root: [dependents] }
        # depends_on: { dependent: root }
        depends_on = {}
        for root, dependents in state.dependencies.items():
            for dep in dependents:
                depends_on[dep] = root

        # 2. Find the "highest" unhealthy service in the dependency chain
        # i.e., an unhealthy service whose upstream is healthy (or has no upstream)
        unhealthy_services = [s for s in state.services if s.status != ServiceStatus.UP]
        
        if not unhealthy_services:
            # System is healthy, but stability might still be low (waiting for stabilization)
            return Action(action_type=ActionType.IGNORE, target="none")

        # Sort unhealthy services by "depth" in dependency chain or just find one with no unhealthy upstream
        target_service = None
        for s in unhealthy_services:
            upstream_name = depends_on.get(s.name)
            if upstream_name:
                upstream = services_by_name.get(upstream_name)
                if upstream and upstream.status != ServiceStatus.UP:
                    # This service has an unhealthy upstream, so it's not the root cause we can fix now
                    continue
            
            # This service is either root or its upstream is healthy
            target_service = s
            break
        
        if not target_service:
            # Fallback: if we couldn't find a clean root, just take the first unhealthy one
            target_service = unhealthy_services[0]

        # 3. Choose action based on status
        if target_service.name == "db" and target_service.status == ServiceStatus.DEGRADED:
            return Action(action_type=ActionType.OPTIMIZE_DB, target=target_service.name)
        elif target_service.status == ServiceStatus.DOWN:
            return Action(action_type=ActionType.RESTART_SERVICE, target=target_service.name)
        elif target_service.status == ServiceStatus.DEGRADED:
            return Action(action_type=ActionType.SCALE, target=target_service.name)
        
        return Action(action_type=ActionType.IGNORE, target="none")
