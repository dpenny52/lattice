"""Topology rule checkers for the Lattice router."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from lattice.config.models import TopologyConfig


@runtime_checkable
class TopologyChecker(Protocol):
    """Protocol for topology rule enforcement."""

    def is_allowed(self, from_agent: str, to_agent: str) -> bool:
        """Return True if *from_agent* is allowed to send to *to_agent*."""
        ...


class MeshTopology:
    """Any agent can message any other agent."""

    def is_allowed(self, from_agent: str, to_agent: str) -> bool:
        return True


class PipelineTopology:
    """Only sequential neighbors in the flow list may communicate.

    Communication is bidirectional between adjacent agents.
    """

    def __init__(self, flow: list[str]) -> None:
        self._neighbors: set[tuple[str, str]] = set()
        for i in range(len(flow) - 1):
            self._neighbors.add((flow[i], flow[i + 1]))
            self._neighbors.add((flow[i + 1], flow[i]))

    def is_allowed(self, from_agent: str, to_agent: str) -> bool:
        return (from_agent, to_agent) in self._neighbors


class HubTopology:
    """Workers can only communicate with the coordinator, not each other."""

    def __init__(self, coordinator: str, workers: list[str]) -> None:
        self._coordinator = coordinator
        self._workers = set(workers)

    def is_allowed(self, from_agent: str, to_agent: str) -> bool:
        return (
            from_agent == self._coordinator and to_agent in self._workers
        ) or (from_agent in self._workers and to_agent == self._coordinator)


class CustomTopology:
    """Only explicitly declared directed edges are allowed."""

    def __init__(self, edges: dict[str, list[str]]) -> None:
        self._edges: set[tuple[str, str]] = set()
        for src, dsts in edges.items():
            for dst in dsts:
                self._edges.add((src, dst))

    def is_allowed(self, from_agent: str, to_agent: str) -> bool:
        return (from_agent, to_agent) in self._edges


def create_topology(config: TopologyConfig) -> TopologyChecker:
    """Factory: build the appropriate topology checker from config."""
    match config.type:
        case "mesh":
            return MeshTopology()
        case "pipeline":
            if config.flow is None:
                msg = "Pipeline topology requires 'flow'"
                raise ValueError(msg)
            return PipelineTopology(config.flow)
        case "hub":
            if config.coordinator is None or config.workers is None:
                msg = "Hub topology requires 'coordinator' and 'workers'"
                raise ValueError(msg)
            return HubTopology(config.coordinator, config.workers)
        case "custom":
            if config.edges is None:
                msg = "Custom topology requires 'edges'"
                raise ValueError(msg)
            return CustomTopology(config.edges)
        case _:  # pragma: no cover
            msg = f"Unknown topology type: {config.type!r}"
            raise ValueError(msg)
