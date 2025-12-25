"""
Agent Graph representation for SAGE.

Represents the topology of a multi-agent security system as a directed graph
where nodes are components (LLMs, tools, guards) and edges are data flows.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum
import json


class NodeType(Enum):
    """Types of nodes in an agent graph."""
    MODEL = "model"           # LLM or ML model
    TOOL = "tool"             # External tool (fuzzer, analyzer, etc.)
    GUARD = "guard"           # Safety guardrail
    AGGREGATOR = "aggregator" # Combines outputs from multiple nodes
    ROUTER = "router"         # Routes inputs to different nodes
    MEMORY = "memory"         # Persistent storage (RAG, vector DB)
    HUMAN = "human"           # Human-in-the-loop


@dataclass
class Node:
    """A node in the agent graph."""
    id: str
    type: NodeType
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime state
    enabled: bool = True
    parameters: Dict[str, float] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "config": self.config,
            "enabled": self.enabled,
            "parameters": self.parameters,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "Node":
        return cls(
            id=d["id"],
            type=NodeType(d["type"]),
            config=d.get("config", {}),
            enabled=d.get("enabled", True),
            parameters=d.get("parameters", {}),
        )


@dataclass
class Edge:
    """A directed edge in the agent graph."""
    source: str
    target: str
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime state
    enabled: bool = True
    weight: float = 1.0
    
    def __hash__(self):
        return hash((self.source, self.target))
    
    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "target": self.target,
            "config": self.config,
            "enabled": self.enabled,
            "weight": self.weight,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "Edge":
        return cls(
            source=d["source"],
            target=d["target"],
            config=d.get("config", {}),
            enabled=d.get("enabled", True),
            weight=d.get("weight", 1.0),
        )


class AgentGraph:
    """
    Represents the topology of a multi-agent security system.
    
    The graph is mutable and can be modified during execution through
    SAGE's adaptation actions.
    """
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[tuple, Edge] = {}  # (source, target) -> Edge
        self._adjacency: Dict[str, Set[str]] = {}  # source -> set of targets
        self._reverse_adjacency: Dict[str, Set[str]] = {}  # target -> set of sources
        
    def add_node(self, node_id: str, node_type: NodeType, **config) -> Node:
        """Add a node to the graph."""
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists")
        
        node = Node(id=node_id, type=node_type, config=config)
        self.nodes[node_id] = node
        self._adjacency[node_id] = set()
        self._reverse_adjacency[node_id] = set()
        return node
    
    def remove_node(self, node_id: str) -> Optional[Node]:
        """Remove a node and all connected edges."""
        if node_id not in self.nodes:
            return None
        
        node = self.nodes.pop(node_id)
        
        # Remove outgoing edges
        for target in list(self._adjacency.get(node_id, [])):
            self.remove_edge(node_id, target)
        
        # Remove incoming edges
        for source in list(self._reverse_adjacency.get(node_id, [])):
            self.remove_edge(source, node_id)
        
        del self._adjacency[node_id]
        del self._reverse_adjacency[node_id]
        
        return node
    
    def add_edge(self, source: str, target: str, **config) -> Edge:
        """Add a directed edge between nodes."""
        if source not in self.nodes:
            raise ValueError(f"Source node {source} does not exist")
        if target not in self.nodes:
            raise ValueError(f"Target node {target} does not exist")
        
        key = (source, target)
        if key in self.edges:
            raise ValueError(f"Edge {source} -> {target} already exists")
        
        edge = Edge(source=source, target=target, config=config)
        self.edges[key] = edge
        self._adjacency[source].add(target)
        self._reverse_adjacency[target].add(source)
        return edge
    
    def remove_edge(self, source: str, target: str) -> Optional[Edge]:
        """Remove an edge from the graph."""
        key = (source, target)
        if key not in self.edges:
            return None
        
        edge = self.edges.pop(key)
        self._adjacency[source].discard(target)
        self._reverse_adjacency[target].discard(source)
        return edge
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def get_edge(self, source: str, target: str) -> Optional[Edge]:
        """Get an edge by source and target."""
        return self.edges.get((source, target))
    
    def get_successors(self, node_id: str) -> List[str]:
        """Get all nodes that this node connects to."""
        return list(self._adjacency.get(node_id, []))
    
    def get_predecessors(self, node_id: str) -> List[str]:
        """Get all nodes that connect to this node."""
        return list(self._reverse_adjacency.get(node_id, []))
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """Get all nodes of a specific type."""
        return [n for n in self.nodes.values() if n.type == node_type]
    
    def get_active_nodes(self) -> List[Node]:
        """Get all enabled nodes."""
        return [n for n in self.nodes.values() if n.enabled]
    
    def get_active_edges(self) -> List[Edge]:
        """Get all enabled edges between enabled nodes."""
        return [
            e for e in self.edges.values()
            if e.enabled 
            and self.nodes[e.source].enabled 
            and self.nodes[e.target].enabled
        ]
    
    def has_path(self, source: str, target: str) -> bool:
        """Check if there's a path from source to target."""
        visited = set()
        queue = [source]
        
        while queue:
            current = queue.pop(0)
            if current == target:
                return True
            if current in visited:
                continue
            visited.add(current)
            
            for next_node in self._adjacency.get(current, []):
                edge = self.edges.get((current, next_node))
                if edge and edge.enabled and self.nodes[next_node].enabled:
                    queue.append(next_node)
        
        return False
    
    def validate_constraints(self, constraints: List["SafetyConstraint"]) -> List[str]:
        """
        Check if graph satisfies all hard constraints.
        Returns list of violation messages (empty if valid).
        """
        violations = []
        for constraint in constraints:
            if hasattr(constraint, 'check_graph'):
                violation = constraint.check_graph(self)
                if violation:
                    violations.append(violation)
        return violations
    
    def copy(self) -> "AgentGraph":
        """Create a deep copy of the graph."""
        new_graph = AgentGraph()
        for node in self.nodes.values():
            new_node = Node(
                id=node.id,
                type=node.type,
                config=node.config.copy(),
                enabled=node.enabled,
                parameters=node.parameters.copy(),
            )
            new_graph.nodes[node.id] = new_node
            new_graph._adjacency[node.id] = set()
            new_graph._reverse_adjacency[node.id] = set()
        
        for edge in self.edges.values():
            new_edge = Edge(
                source=edge.source,
                target=edge.target,
                config=edge.config.copy(),
                enabled=edge.enabled,
                weight=edge.weight,
            )
            new_graph.edges[(edge.source, edge.target)] = new_edge
            new_graph._adjacency[edge.source].add(edge.target)
            new_graph._reverse_adjacency[edge.target].add(edge.source)
        
        return new_graph
    
    def to_dict(self) -> Dict:
        """Serialize graph to dictionary."""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "AgentGraph":
        """Deserialize graph from dictionary."""
        graph = cls()
        for node_data in d.get("nodes", []):
            node = Node.from_dict(node_data)
            graph.nodes[node.id] = node
            graph._adjacency[node.id] = set()
            graph._reverse_adjacency[node.id] = set()
        
        for edge_data in d.get("edges", []):
            edge = Edge.from_dict(edge_data)
            graph.edges[(edge.source, edge.target)] = edge
            graph._adjacency[edge.source].add(edge.target)
            graph._reverse_adjacency[edge.target].add(edge.source)
        
        return graph
    
    def to_json(self) -> str:
        """Serialize graph to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, s: str) -> "AgentGraph":
        """Deserialize graph from JSON string."""
        return cls.from_dict(json.loads(s))
    
    def __repr__(self):
        return f"AgentGraph(nodes={len(self.nodes)}, edges={len(self.edges)})"

