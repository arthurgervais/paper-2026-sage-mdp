"""
SAGE: Safe Adaptive Graph Editor
================================

Online adaptation of agentic security architectures under safety constraints.

Components:
- AgentGraph: Represents multi-agent system topology
- SAGEController: Main controller for online adaptation
- Monitors: Track utility, cost, and risk signals
- Constraints: Define safety requirements

Example usage:
    from sage import AgentGraph, SAGEController
    
    graph = AgentGraph()
    graph.add_node("llm", type="model", model="gpt-4")
    graph.add_node("fuzzer", type="tool", tool="afl")
    graph.add_edge("llm", "fuzzer")
    
    controller = SAGEController(
        graph=graph,
        risk_constraint=0.1,
    )
    
    # During engagement
    for observation in engagement_loop():
        action = controller.select_action(observation)
        controller.apply_action(action)
        result = run_step()
        controller.update(result)
"""

from .graph import AgentGraph, Node, Edge
from .controller import SAGEController
from .actions import GraphAction, ActionType
from .monitors import SignalMonitor, UtilitySignal, CostSignal, RiskSignal
from .constraints import SafetyConstraint, HardConstraint, SoftConstraint

__version__ = "0.1.0"
__all__ = [
    "AgentGraph",
    "Node", 
    "Edge",
    "SAGEController",
    "GraphAction",
    "ActionType",
    "SignalMonitor",
    "UtilitySignal",
    "CostSignal", 
    "RiskSignal",
    "SafetyConstraint",
    "HardConstraint",
    "SoftConstraint",
]

