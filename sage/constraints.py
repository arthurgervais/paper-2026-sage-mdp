"""
Safety constraints for SAGE.

Defines hard and soft constraints that guide architecture adaptation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Set
from abc import ABC, abstractmethod

from .graph import AgentGraph, Node, NodeType


class SafetyConstraint(ABC):
    """Base class for safety constraints."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        
    @abstractmethod
    def check_graph(self, graph: AgentGraph) -> Optional[str]:
        """
        Check if a graph satisfies this constraint.
        Returns error message if violated, None if satisfied.
        """
        pass


class HardConstraint(SafetyConstraint):
    """
    Hard constraint that must always be satisfied.
    
    Actions that would violate hard constraints are filtered out
    before being considered by the policy.
    """
    pass


class SoftConstraint(SafetyConstraint):
    """
    Soft constraint enforced via Lagrangian optimization.
    
    Soft constraints are satisfied on average over time,
    not necessarily at every step.
    """
    
    def __init__(self, name: str, threshold: float, description: str = ""):
        super().__init__(name, description)
        self.threshold = threshold
        
    @abstractmethod
    def get_value(self, graph: AgentGraph, signals: Dict[str, float]) -> float:
        """Get current constraint value."""
        pass
    
    def is_violated(self, graph: AgentGraph, signals: Dict[str, float]) -> bool:
        """Check if constraint is currently violated."""
        return self.get_value(graph, signals) > self.threshold


# =============================================================================
# Hard Constraints
# =============================================================================

class GuardRequiredConstraint(HardConstraint):
    """
    Requires a guard node on all paths to sensitive nodes.
    
    This ensures that prompts always pass through safety filtering
    before reaching sensitive components.
    """
    
    def __init__(self, sensitive_nodes: Set[str]):
        super().__init__(
            name="guard_required",
            description="Guard required on paths to sensitive nodes"
        )
        self.sensitive_nodes = sensitive_nodes
        
    def check_graph(self, graph: AgentGraph) -> Optional[str]:
        """Check that all paths to sensitive nodes pass through a guard."""
        # Find all entry points (nodes with no predecessors)
        entry_points = [
            n.id for n in graph.get_active_nodes()
            if not graph.get_predecessors(n.id)
        ]
        
        # Get all guard nodes
        guards = {n.id for n in graph.get_nodes_by_type(NodeType.GUARD) if n.enabled}
        
        for sensitive in self.sensitive_nodes:
            if sensitive not in graph.nodes:
                continue
            if not graph.nodes[sensitive].enabled:
                continue
                
            # Check each entry point
            for entry in entry_points:
                if self._has_unguarded_path(graph, entry, sensitive, guards):
                    return f"Unguarded path from {entry} to {sensitive}"
        
        return None
    
    def _has_unguarded_path(self, graph: AgentGraph, start: str, 
                            end: str, guards: Set[str]) -> bool:
        """Check if there's a path from start to end that doesn't pass through a guard."""
        visited = set()
        queue = [(start, False)]  # (node, passed_through_guard)
        
        while queue:
            current, guarded = queue.pop(0)
            
            if current == end:
                if not guarded:
                    return True  # Found unguarded path
                continue
            
            if current in visited:
                continue
            visited.add(current)
            
            # Check if this node is a guard
            is_guard = current in guards
            new_guarded = guarded or is_guard
            
            # Explore successors
            for next_node in graph.get_successors(current):
                edge = graph.get_edge(current, next_node)
                if edge and edge.enabled and graph.nodes[next_node].enabled:
                    queue.append((next_node, new_guarded))
        
        return False


class ConnectivityConstraint(HardConstraint):
    """
    Ensures the graph remains connected.
    
    All active nodes must be reachable from entry points.
    """
    
    def __init__(self):
        super().__init__(
            name="connectivity",
            description="Graph must remain connected"
        )
        
    def check_graph(self, graph: AgentGraph) -> Optional[str]:
        # Find entry points
        entry_points = [
            n.id for n in graph.get_active_nodes()
            if not graph.get_predecessors(n.id)
        ]
        
        if not entry_points:
            return "No entry points in graph"
        
        # BFS from entry points
        reachable = set()
        queue = list(entry_points)
        
        while queue:
            current = queue.pop(0)
            if current in reachable:
                continue
            reachable.add(current)
            
            for next_node in graph.get_successors(current):
                edge = graph.get_edge(current, next_node)
                if edge and edge.enabled and graph.nodes[next_node].enabled:
                    queue.append(next_node)
        
        # Check all active nodes are reachable
        for node in graph.get_active_nodes():
            if node.id not in reachable:
                return f"Node {node.id} is unreachable"
        
        return None


class MinNodesConstraint(HardConstraint):
    """Requires minimum number of certain node types."""
    
    def __init__(self, node_type: NodeType, min_count: int):
        super().__init__(
            name=f"min_{node_type.value}",
            description=f"At least {min_count} {node_type.value} nodes required"
        )
        self.node_type = node_type
        self.min_count = min_count
        
    def check_graph(self, graph: AgentGraph) -> Optional[str]:
        active_of_type = [
            n for n in graph.get_nodes_by_type(self.node_type)
            if n.enabled
        ]
        if len(active_of_type) < self.min_count:
            return f"Only {len(active_of_type)} active {self.node_type.value} nodes, need {self.min_count}"
        return None


class CycleConstraint(HardConstraint):
    """Prevents cycles in the graph (optional - some architectures allow cycles)."""
    
    def __init__(self):
        super().__init__(
            name="no_cycles",
            description="Graph must be acyclic"
        )
        
    def check_graph(self, graph: AgentGraph) -> Optional[str]:
        # Topological sort to detect cycles
        in_degree = {n.id: 0 for n in graph.get_active_nodes()}
        
        for edge in graph.get_active_edges():
            if edge.target in in_degree:
                in_degree[edge.target] += 1
        
        queue = [n for n, d in in_degree.items() if d == 0]
        visited = 0
        
        while queue:
            current = queue.pop(0)
            visited += 1
            
            for next_node in graph.get_successors(current):
                edge = graph.get_edge(current, next_node)
                if edge and edge.enabled and next_node in in_degree:
                    in_degree[next_node] -= 1
                    if in_degree[next_node] == 0:
                        queue.append(next_node)
        
        if visited < len(in_degree):
            return "Graph contains a cycle"
        return None


# =============================================================================
# Soft Constraints
# =============================================================================

class RiskConstraint(SoftConstraint):
    """
    Constrains average risk below a threshold.
    
    The main safety constraint in SAGE - ensures prompt injection
    and other risks stay below acceptable levels on average.
    """
    
    def __init__(self, threshold: float = 0.1):
        super().__init__(
            name="risk_constraint",
            threshold=threshold,
            description=f"Average risk must stay below {threshold}"
        )
        
    def check_graph(self, graph: AgentGraph) -> Optional[str]:
        # Soft constraint - doesn't block actions
        return None
    
    def get_value(self, graph: AgentGraph, signals: Dict[str, float]) -> float:
        return signals.get("risk_mean", 0.0)


class CostBudgetConstraint(SoftConstraint):
    """
    Constrains average cost below a budget.
    """
    
    def __init__(self, budget: float = 1.0):
        super().__init__(
            name="cost_budget",
            threshold=budget,
            description=f"Average cost must stay below {budget}"
        )
        
    def check_graph(self, graph: AgentGraph) -> Optional[str]:
        return None
    
    def get_value(self, graph: AgentGraph, signals: Dict[str, float]) -> float:
        return signals.get("cost_mean", 0.0)


class LatencyConstraint(SoftConstraint):
    """
    Constrains average latency below a threshold.
    """
    
    def __init__(self, max_latency: float = 5.0):
        super().__init__(
            name="latency_constraint",
            threshold=max_latency,
            description=f"Average latency must stay below {max_latency}s"
        )
        
    def check_graph(self, graph: AgentGraph) -> Optional[str]:
        return None
    
    def get_value(self, graph: AgentGraph, signals: Dict[str, float]) -> float:
        return signals.get("latency_mean", 0.0)


# =============================================================================
# Constraint Sets
# =============================================================================

@dataclass
class ConstraintSet:
    """Collection of constraints for a SAGE deployment."""
    hard_constraints: List[HardConstraint] = field(default_factory=list)
    soft_constraints: List[SoftConstraint] = field(default_factory=list)
    
    def add_hard(self, constraint: HardConstraint):
        self.hard_constraints.append(constraint)
        
    def add_soft(self, constraint: SoftConstraint):
        self.soft_constraints.append(constraint)
        
    def check_hard(self, graph: AgentGraph) -> List[str]:
        """Check all hard constraints. Returns list of violations."""
        violations = []
        for c in self.hard_constraints:
            violation = c.check_graph(graph)
            if violation:
                violations.append(f"{c.name}: {violation}")
        return violations
    
    def get_soft_values(self, graph: AgentGraph, 
                        signals: Dict[str, float]) -> Dict[str, float]:
        """Get current values of all soft constraints."""
        return {c.name: c.get_value(graph, signals) for c in self.soft_constraints}
    
    def get_soft_violations(self, graph: AgentGraph,
                            signals: Dict[str, float]) -> Dict[str, float]:
        """Get constraint violation amounts (positive = violated)."""
        violations = {}
        for c in self.soft_constraints:
            value = c.get_value(graph, signals)
            violations[c.name] = value - c.threshold
        return violations


def default_security_constraints(
    sensitive_nodes: Set[str],
    risk_threshold: float = 0.1,
    cost_budget: float = 1.0
) -> ConstraintSet:
    """Create default constraint set for security applications."""
    constraints = ConstraintSet()
    
    # Hard constraints
    constraints.add_hard(GuardRequiredConstraint(sensitive_nodes))
    constraints.add_hard(ConnectivityConstraint())
    constraints.add_hard(MinNodesConstraint(NodeType.GUARD, min_count=1))
    
    # Soft constraints
    constraints.add_soft(RiskConstraint(threshold=risk_threshold))
    constraints.add_soft(CostBudgetConstraint(budget=cost_budget))
    
    return constraints

