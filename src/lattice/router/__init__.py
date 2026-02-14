"""Router — topology-enforced message dispatch for the Lattice runtime."""

from lattice.router.router import (
    Agent,
    RouteNotAllowedError,
    Router,
    UnknownAgentError,
)
from lattice.router.topology import (
    CustomTopology,
    HubTopology,
    MeshTopology,
    PipelineTopology,
    TopologyChecker,
    create_topology,
)

__all__ = [
    "Agent",
    "CustomTopology",
    "HubTopology",
    "MeshTopology",
    "PipelineTopology",
    "RouteNotAllowedError",
    "Router",
    "TopologyChecker",
    "UnknownAgentError",
    "create_topology",
]
