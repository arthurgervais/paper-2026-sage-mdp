"""
Graph modification actions for SAGE.

Defines the action space for online architecture adaptation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum

from .graph import AgentGraph, Node, Edge, NodeType


class ActionType(Enum):
    """Types of graph modification actions."""
    # Node operations
    ADD_NODE = "add_node"
    REMOVE_NODE = "remove_node"
    ENABLE_NODE = "enable_node"
    DISABLE_NODE = "disable_node"
    UPDATE_NODE_PARAMS = "update_node_params"
    SWITCH_MODEL = "switch_model"
    
    # Edge operations
    ADD_EDGE = "add_edge"
    REMOVE_EDGE = "remove_edge"
    ENABLE_EDGE = "enable_edge"
    DISABLE_EDGE = "disable_edge"
    UPDATE_EDGE_WEIGHT = "update_edge_weight"
    REWIRE_EDGE = "rewire_edge"
    
    # Composite operations
    SWITCH_TEMPLATE = "switch_template"
    RUN_DIAGNOSTIC = "run_diagnostic"
    NO_OP = "no_op"


@dataclass
class GraphAction:
    """
    A single graph modification action.
    
    Actions are the primitives that SAGE uses to adapt the agent graph.
    Each action has a type and parameters specific to that type.
    """
    action_type: ActionType
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    description: str = ""
    estimated_cost: float = 0.0
    estimated_risk: float = 0.0
    
    def apply(self, graph: AgentGraph) -> bool:
        """
        Apply this action to a graph.
        Returns True if successful, False otherwise.
        """
        try:
            if self.action_type == ActionType.ADD_NODE:
                graph.add_node(
                    node_id=self.params["node_id"],
                    node_type=NodeType(self.params["node_type"]),
                    **self.params.get("config", {})
                )
                
            elif self.action_type == ActionType.REMOVE_NODE:
                graph.remove_node(self.params["node_id"])
                
            elif self.action_type == ActionType.ENABLE_NODE:
                node = graph.get_node(self.params["node_id"])
                if node:
                    node.enabled = True
                    
            elif self.action_type == ActionType.DISABLE_NODE:
                node = graph.get_node(self.params["node_id"])
                if node:
                    node.enabled = False
                    
            elif self.action_type == ActionType.UPDATE_NODE_PARAMS:
                node = graph.get_node(self.params["node_id"])
                if node:
                    node.parameters.update(self.params.get("parameters", {}))
                    
            elif self.action_type == ActionType.SWITCH_MODEL:
                node = graph.get_node(self.params["node_id"])
                if node and node.type == NodeType.MODEL:
                    node.config["model"] = self.params["new_model"]
                    
            elif self.action_type == ActionType.ADD_EDGE:
                graph.add_edge(
                    source=self.params["source"],
                    target=self.params["target"],
                    **self.params.get("config", {})
                )
                
            elif self.action_type == ActionType.REMOVE_EDGE:
                graph.remove_edge(
                    source=self.params["source"],
                    target=self.params["target"]
                )
                
            elif self.action_type == ActionType.ENABLE_EDGE:
                edge = graph.get_edge(self.params["source"], self.params["target"])
                if edge:
                    edge.enabled = True
                    
            elif self.action_type == ActionType.DISABLE_EDGE:
                edge = graph.get_edge(self.params["source"], self.params["target"])
                if edge:
                    edge.enabled = False
                    
            elif self.action_type == ActionType.UPDATE_EDGE_WEIGHT:
                edge = graph.get_edge(self.params["source"], self.params["target"])
                if edge:
                    edge.weight = self.params["weight"]
                    
            elif self.action_type == ActionType.REWIRE_EDGE:
                # Remove old edge, add new one
                old_source = self.params["old_source"]
                old_target = self.params["old_target"]
                new_source = self.params.get("new_source", old_source)
                new_target = self.params.get("new_target", old_target)
                
                old_edge = graph.remove_edge(old_source, old_target)
                if old_edge:
                    graph.add_edge(
                        source=new_source,
                        target=new_target,
                        **old_edge.config
                    )
                    
            elif self.action_type == ActionType.SWITCH_TEMPLATE:
                # Apply a predefined template configuration
                template = self.params.get("template")
                if template:
                    _apply_template(graph, template)
                    
            elif self.action_type == ActionType.RUN_DIAGNOSTIC:
                # Diagnostic probe - doesn't modify graph
                pass
                
            elif self.action_type == ActionType.NO_OP:
                pass
                
            else:
                return False
                
            return True
            
        except Exception as e:
            print(f"Action failed: {e}")
            return False
    
    def validate(self, graph: AgentGraph) -> Optional[str]:
        """
        Check if this action can be applied to the graph.
        Returns error message if invalid, None if valid.
        """
        if self.action_type == ActionType.ADD_NODE:
            if self.params.get("node_id") in graph.nodes:
                return f"Node {self.params['node_id']} already exists"
                
        elif self.action_type == ActionType.REMOVE_NODE:
            if self.params.get("node_id") not in graph.nodes:
                return f"Node {self.params['node_id']} does not exist"
                
        elif self.action_type in [ActionType.ENABLE_NODE, ActionType.DISABLE_NODE, 
                                   ActionType.UPDATE_NODE_PARAMS, ActionType.SWITCH_MODEL]:
            if self.params.get("node_id") not in graph.nodes:
                return f"Node {self.params['node_id']} does not exist"
                
        elif self.action_type == ActionType.ADD_EDGE:
            source = self.params.get("source")
            target = self.params.get("target")
            if source not in graph.nodes:
                return f"Source node {source} does not exist"
            if target not in graph.nodes:
                return f"Target node {target} does not exist"
            if (source, target) in graph.edges:
                return f"Edge {source} -> {target} already exists"
                
        elif self.action_type in [ActionType.REMOVE_EDGE, ActionType.ENABLE_EDGE,
                                   ActionType.DISABLE_EDGE, ActionType.UPDATE_EDGE_WEIGHT]:
            source = self.params.get("source")
            target = self.params.get("target")
            if (source, target) not in graph.edges:
                return f"Edge {source} -> {target} does not exist"
                
        return None
    
    def to_dict(self) -> Dict:
        return {
            "action_type": self.action_type.value,
            "params": self.params,
            "description": self.description,
            "estimated_cost": self.estimated_cost,
            "estimated_risk": self.estimated_risk,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "GraphAction":
        return cls(
            action_type=ActionType(d["action_type"]),
            params=d.get("params", {}),
            description=d.get("description", ""),
            estimated_cost=d.get("estimated_cost", 0.0),
            estimated_risk=d.get("estimated_risk", 0.0),
        )


def _apply_template(graph: AgentGraph, template: Dict):
    """Apply a template configuration to a graph."""
    # Disable all nodes first
    for node in graph.nodes.values():
        node.enabled = False
    
    # Enable nodes specified in template
    for node_id in template.get("enabled_nodes", []):
        if node_id in graph.nodes:
            graph.nodes[node_id].enabled = True
    
    # Update parameters
    for node_id, params in template.get("node_params", {}).items():
        if node_id in graph.nodes:
            graph.nodes[node_id].parameters.update(params)
    
    # Enable edges
    for edge in graph.edges.values():
        edge.enabled = False
    for source, target in template.get("enabled_edges", []):
        if (source, target) in graph.edges:
            graph.edges[(source, target)].enabled = True


class ActionSpace:
    """
    Defines the available actions for a given graph configuration.
    
    This class generates the set of valid actions that SAGE can take,
    filtered by hard constraints.
    """
    
    def __init__(self, 
                 graph: AgentGraph,
                 templates: Optional[List[Dict]] = None,
                 allowed_model_switches: Optional[Dict[str, List[str]]] = None):
        """
        Args:
            graph: The agent graph
            templates: Pre-defined architecture templates
            allowed_model_switches: Dict mapping node_id to list of allowed models
        """
        self.graph = graph
        self.templates = templates or []
        self.allowed_model_switches = allowed_model_switches or {}
        
    def get_available_actions(self) -> List[GraphAction]:
        """Get all valid actions for the current graph state."""
        actions = [GraphAction(ActionType.NO_OP, description="Do nothing")]
        
        # Node enable/disable
        for node in self.graph.nodes.values():
            if node.enabled:
                actions.append(GraphAction(
                    ActionType.DISABLE_NODE,
                    params={"node_id": node.id},
                    description=f"Disable {node.id}"
                ))
            else:
                actions.append(GraphAction(
                    ActionType.ENABLE_NODE,
                    params={"node_id": node.id},
                    description=f"Enable {node.id}"
                ))
        
        # Model switches
        for node_id, models in self.allowed_model_switches.items():
            node = self.graph.get_node(node_id)
            if node and node.type == NodeType.MODEL:
                current_model = node.config.get("model", "")
                for model in models:
                    if model != current_model:
                        actions.append(GraphAction(
                            ActionType.SWITCH_MODEL,
                            params={"node_id": node_id, "new_model": model},
                            description=f"Switch {node_id} to {model}"
                        ))
        
        # Edge enable/disable
        for edge in self.graph.edges.values():
            if edge.enabled:
                actions.append(GraphAction(
                    ActionType.DISABLE_EDGE,
                    params={"source": edge.source, "target": edge.target},
                    description=f"Disable {edge.source} -> {edge.target}"
                ))
            else:
                actions.append(GraphAction(
                    ActionType.ENABLE_EDGE,
                    params={"source": edge.source, "target": edge.target},
                    description=f"Enable {edge.source} -> {edge.target}"
                ))
        
        # Template switches
        for i, template in enumerate(self.templates):
            actions.append(GraphAction(
                ActionType.SWITCH_TEMPLATE,
                params={"template": template, "template_id": i},
                description=f"Switch to template {i}"
            ))
        
        return actions
    
    def filter_by_constraints(self, 
                              actions: List[GraphAction],
                              constraints: List["SafetyConstraint"]) -> List[GraphAction]:
        """Filter actions that would violate hard constraints."""
        valid_actions = []
        
        for action in actions:
            # Test action on a copy of the graph
            test_graph = self.graph.copy()
            action.apply(test_graph)
            
            # Check constraints
            violations = test_graph.validate_constraints(constraints)
            if not violations:
                valid_actions.append(action)
        
        return valid_actions

