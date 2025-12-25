"""
Main SAGE Controller for online architecture adaptation.

Implements the constrained contextual bandit algorithm with
Lagrangian constraint enforcement.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import json
import time

from .graph import AgentGraph
from .actions import GraphAction, ActionType, ActionSpace
from .monitors import SignalMonitor, CompositeMonitor
from .constraints import ConstraintSet, HardConstraint, SoftConstraint


@dataclass
class SAGEConfig:
    """Configuration for SAGE controller."""
    # Learning parameters
    bandit_alpha: float = 0.2           # UCB exploration parameter
    lagrange_lr: float = 0.1            # Lagrange multiplier learning rate
    reward_discount: float = 0.99       # Temporal discounting
    
    # Constraint parameters
    risk_constraint: float = 0.1        # Default risk threshold (kappa)
    cost_budget: float = 1.0            # Per-step cost budget
    
    # Adaptation parameters
    adaptation_interval: int = 10       # Steps between adaptations
    min_samples_before_adapt: int = 20  # Minimum samples before first adaptation
    
    # Memory parameters
    history_window: int = 100           # Sliding window for statistics
    belief_momentum: float = 0.7        # Momentum for regime belief updates


@dataclass
class ActionStats:
    """Statistics for a single action."""
    reward_sum: float = 0.0
    risk_sum: float = 0.0
    cost_sum: float = 0.0
    count: int = 0
    
    def update(self, reward: float, risk: float, cost: float):
        self.reward_sum += reward
        self.risk_sum += risk
        self.cost_sum += cost
        self.count += 1
        
    def get_estimates(self) -> Tuple[float, float, float]:
        """Returns (reward_est, risk_est, cost_est)."""
        if self.count == 0:
            return 0.0, 0.5, 0.5  # Pessimistic priors
        return (
            self.reward_sum / self.count,
            self.risk_sum / self.count,
            self.cost_sum / self.count
        )


class SAGEController:
    """
    Main controller for SAGE online adaptation.
    
    Implements Safe-LinUCB-Lag: a constrained contextual bandit that
    selects graph modification actions to maximize utility while
    satisfying risk constraints.
    
    Key features:
    - Bayesian-style regime inference from observations
    - Pessimistic risk estimation (upper confidence bound)
    - Lagrangian dual updates for constraint satisfaction
    - Hard constraint projection (filter invalid actions)
    """
    
    def __init__(self,
                 graph: AgentGraph,
                 config: Optional[SAGEConfig] = None,
                 constraints: Optional[ConstraintSet] = None,
                 templates: Optional[List[Dict]] = None):
        """
        Initialize SAGE controller.
        
        Args:
            graph: The agent graph to adapt
            config: Configuration parameters
            constraints: Safety constraint set
            templates: Pre-defined architecture templates
        """
        self.graph = graph
        self.config = config or SAGEConfig()
        self.constraints = constraints or ConstraintSet()
        self.templates = templates or []
        
        # Action space
        self.action_space = ActionSpace(
            graph=graph,
            templates=templates
        )
        
        # Per-action statistics
        self.action_stats: Dict[str, ActionStats] = {}
        
        # Lagrange multipliers for soft constraints
        self.lagrange_multipliers: Dict[str, float] = {}
        for c in self.constraints.soft_constraints:
            self.lagrange_multipliers[c.name] = 1.0
        
        # Runtime state
        self.cumulative_risk = 0.0
        self.cumulative_cost = 0.0
        self.total_steps = 0
        self.steps_since_adapt = 0
        
        # Observation history
        self.observation_history: deque = deque(maxlen=self.config.history_window)
        
        # Current selected action
        self.current_action: Optional[GraphAction] = None
        
        # Metrics logging
        self.metrics_log: List[Dict] = []
        
    def select_action(self, observation: Dict[str, float]) -> GraphAction:
        """
        Select the best action given current observation.
        
        Args:
            observation: Dict of signal values from monitors
            
        Returns:
            GraphAction to apply
        """
        self.observation_history.append(observation)
        
        # Get available actions (filtered by hard constraints)
        available_actions = self.action_space.get_available_actions()
        valid_actions = self.action_space.filter_by_constraints(
            available_actions, 
            self.constraints.hard_constraints
        )
        
        if not valid_actions:
            # Fallback to no-op if no valid actions
            return GraphAction(ActionType.NO_OP)
        
        # Check if we should adapt
        if self.steps_since_adapt < self.config.adaptation_interval:
            if self.current_action is not None:
                return self.current_action
        
        # Compute current risk/cost status
        risk_slack = self._get_risk_slack()
        
        # Select best action using Lagrangian UCB
        best_action = valid_actions[0]
        best_value = float('-inf')
        
        for action in valid_actions:
            value = self._compute_action_value(action, observation, risk_slack)
            if value > best_value:
                best_value = value
                best_action = action
        
        self.current_action = best_action
        self.steps_since_adapt = 0
        
        return best_action
    
    def _compute_action_value(self, action: GraphAction, 
                               observation: Dict[str, float],
                               risk_slack: float) -> float:
        """Compute Lagrangian UCB value for an action."""
        action_key = self._action_key(action)
        
        # Get or create stats
        if action_key not in self.action_stats:
            self.action_stats[action_key] = ActionStats()
        stats = self.action_stats[action_key]
        
        reward_est, risk_est, cost_est = stats.get_estimates()
        n = max(stats.count, 1)
        
        # UCB exploration bonus
        exploration = self.config.bandit_alpha * np.sqrt(
            np.log(self.total_steps + 2) / n
        )
        
        # Optimistic reward, pessimistic risk
        reward_ucb = reward_est + exploration
        
        # Adaptive pessimism based on risk slack
        if risk_slack > 0.02:
            risk_ucb = risk_est + 0.3 * exploration
        elif risk_slack > 0.01:
            risk_ucb = risk_est + 0.5 * exploration
        else:
            risk_ucb = risk_est + exploration  # Very pessimistic
        
        # Lagrangian objective
        lambda_risk = self.lagrange_multipliers.get("risk_constraint", 1.0)
        value = reward_ucb - lambda_risk * risk_ucb
        
        # Add other soft constraint penalties
        for c in self.constraints.soft_constraints:
            if c.name != "risk_constraint":
                lam = self.lagrange_multipliers.get(c.name, 1.0)
                value -= lam * cost_est
        
        return value
    
    def _get_risk_slack(self) -> float:
        """Get current slack in risk constraint."""
        if self.total_steps == 0:
            return self.config.risk_constraint
        avg_risk = self.cumulative_risk / self.total_steps
        return self.config.risk_constraint - avg_risk
    
    def apply_action(self, action: GraphAction) -> bool:
        """Apply an action to the graph."""
        # Validate action
        error = action.validate(self.graph)
        if error:
            print(f"Action validation failed: {error}")
            return False
        
        # Check hard constraints on resulting graph
        test_graph = self.graph.copy()
        action.apply(test_graph)
        violations = test_graph.validate_constraints(self.constraints.hard_constraints)
        
        if violations:
            print(f"Action would violate constraints: {violations}")
            return False
        
        # Apply to real graph
        return action.apply(self.graph)
    
    def update(self, 
               action: GraphAction,
               utility: float,
               cost: float,
               risk: float,
               observation: Optional[Dict[str, float]] = None):
        """
        Update controller state after observing outcomes.
        
        Args:
            action: The action that was taken
            utility: Observed utility (e.g., coverage delta)
            cost: Observed cost (e.g., API cost)
            risk: Observed risk (e.g., injection alert rate)
            observation: Current observation dict
        """
        # Compute reward
        reward = utility - self.config.cost_budget * cost - risk
        
        # Update action statistics
        action_key = self._action_key(action)
        if action_key not in self.action_stats:
            self.action_stats[action_key] = ActionStats()
        self.action_stats[action_key].update(reward, risk, cost)
        
        # Update cumulative tracking
        self.cumulative_risk += risk
        self.cumulative_cost += cost
        self.total_steps += 1
        self.steps_since_adapt += 1
        
        # Update Lagrange multipliers (dual ascent)
        self._update_lagrange_multipliers()
        
        # Log metrics
        self.metrics_log.append({
            "step": self.total_steps,
            "action": action_key,
            "utility": utility,
            "cost": cost,
            "risk": risk,
            "reward": reward,
            "avg_risk": self.cumulative_risk / self.total_steps,
            "lambda_risk": self.lagrange_multipliers.get("risk_constraint", 1.0),
            "timestamp": time.time(),
        })
    
    def _update_lagrange_multipliers(self):
        """Update Lagrange multipliers using dual ascent."""
        if self.total_steps == 0:
            return
        
        # Risk constraint
        avg_risk = self.cumulative_risk / self.total_steps
        risk_violation = avg_risk - self.config.risk_constraint
        
        # Asymmetric learning rate
        if risk_violation > 0:
            lr = self.config.lagrange_lr * 1.5
        else:
            lr = self.config.lagrange_lr * 0.3
        
        current_lambda = self.lagrange_multipliers.get("risk_constraint", 1.0)
        new_lambda = np.clip(current_lambda + lr * risk_violation, 0.1, 10.0)
        self.lagrange_multipliers["risk_constraint"] = new_lambda
        
        # Cost constraint
        avg_cost = self.cumulative_cost / self.total_steps
        cost_violation = avg_cost - self.config.cost_budget
        
        current_lambda = self.lagrange_multipliers.get("cost_budget", 1.0)
        new_lambda = np.clip(current_lambda + self.config.lagrange_lr * cost_violation, 0.1, 10.0)
        self.lagrange_multipliers["cost_budget"] = new_lambda
    
    def _action_key(self, action: GraphAction) -> str:
        """Get a unique string key for an action."""
        return f"{action.action_type.value}:{json.dumps(action.params, sort_keys=True)}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current controller statistics."""
        return {
            "total_steps": self.total_steps,
            "avg_risk": self.cumulative_risk / max(self.total_steps, 1),
            "avg_cost": self.cumulative_cost / max(self.total_steps, 1),
            "risk_slack": self._get_risk_slack(),
            "lagrange_multipliers": dict(self.lagrange_multipliers),
            "num_actions_explored": len(self.action_stats),
        }
    
    def reset(self, keep_learned: bool = True):
        """
        Reset controller state.
        
        Args:
            keep_learned: If True, keep learned action statistics
        """
        self.cumulative_risk = 0.0
        self.cumulative_cost = 0.0
        self.total_steps = 0
        self.steps_since_adapt = 0
        
        for c in self.constraints.soft_constraints:
            self.lagrange_multipliers[c.name] = 1.0
        
        self.observation_history.clear()
        self.current_action = None
        
        if not keep_learned:
            self.action_stats.clear()
            self.metrics_log.clear()
    
    def save_state(self, path: str):
        """Save controller state to file."""
        state = {
            "config": {
                "bandit_alpha": self.config.bandit_alpha,
                "lagrange_lr": self.config.lagrange_lr,
                "risk_constraint": self.config.risk_constraint,
                "cost_budget": self.config.cost_budget,
            },
            "action_stats": {
                k: {
                    "reward_sum": v.reward_sum,
                    "risk_sum": v.risk_sum,
                    "cost_sum": v.cost_sum,
                    "count": v.count,
                }
                for k, v in self.action_stats.items()
            },
            "lagrange_multipliers": self.lagrange_multipliers,
            "cumulative_risk": self.cumulative_risk,
            "cumulative_cost": self.cumulative_cost,
            "total_steps": self.total_steps,
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, path: str):
        """Load controller state from file."""
        with open(path, 'r') as f:
            state = json.load(f)
        
        for k, v in state.get("action_stats", {}).items():
            stats = ActionStats(
                reward_sum=v["reward_sum"],
                risk_sum=v["risk_sum"],
                cost_sum=v["cost_sum"],
                count=v["count"],
            )
            self.action_stats[k] = stats
        
        self.lagrange_multipliers = state.get("lagrange_multipliers", {})
        self.cumulative_risk = state.get("cumulative_risk", 0.0)
        self.cumulative_cost = state.get("cumulative_cost", 0.0)
        self.total_steps = state.get("total_steps", 0)


class TemplateController(SAGEController):
    """
    Simplified controller that only switches between templates.
    
    Useful when the action space is just selecting from predefined
    architecture configurations.
    """
    
    def __init__(self,
                 graph: AgentGraph,
                 templates: List[Dict],
                 config: Optional[SAGEConfig] = None,
                 constraints: Optional[ConstraintSet] = None):
        super().__init__(graph, config, constraints, templates)
        
        # Statistics per template
        self.template_stats: Dict[int, ActionStats] = {
            i: ActionStats() for i in range(len(templates))
        }
        
        self.current_template: int = 0
        
    def select_template(self, observation: Dict[str, float]) -> int:
        """Select best template index."""
        risk_slack = self._get_risk_slack()
        
        best_template = 0
        best_value = float('-inf')
        
        for i, template in enumerate(self.templates):
            stats = self.template_stats[i]
            reward_est, risk_est, cost_est = stats.get_estimates()
            n = max(stats.count, 1)
            
            exploration = self.config.bandit_alpha * np.sqrt(
                np.log(self.total_steps + 2) / n
            )
            
            reward_ucb = reward_est + exploration
            risk_ucb = risk_est + 0.5 * exploration
            
            lambda_risk = self.lagrange_multipliers.get("risk_constraint", 1.0)
            value = reward_ucb - lambda_risk * risk_ucb
            
            if value > best_value:
                best_value = value
                best_template = i
        
        self.current_template = best_template
        return best_template
    
    def update_template(self, template_idx: int,
                        utility: float, cost: float, risk: float):
        """Update statistics for a template."""
        reward = utility - self.config.cost_budget * cost - risk
        self.template_stats[template_idx].update(reward, risk, cost)
        
        self.cumulative_risk += risk
        self.cumulative_cost += cost
        self.total_steps += 1
        
        self._update_lagrange_multipliers()

