"""Pydantic v2 models for lattice.yaml configuration."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_MODEL_RE = re.compile(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$")


class AgentConfig(BaseModel):
    """Configuration for a single agent in the team."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["llm", "cli", "script", "remote", "human"] = Field(
        default="llm",
        description="Agent type — defaults to 'llm' when model is present",
    )
    model: str | None = Field(
        default=None,
        description="LLM model identifier, e.g. 'anthropic/claude-sonnet-4-5-20250929'",
    )
    role: str | None = Field(
        default=None,
        description="System prompt text or path to a prompt file",
    )
    tools: list[str | dict[str, object]] = Field(
        default_factory=list,
        description="Tools available to the agent",
    )
    cli: str | None = Field(
        default=None,
        description="CLI tool name for cli-type agents (e.g. 'claude', 'codex')",
    )
    command: str | None = Field(
        default=None,
        description="Shell command for script-type agents or custom cli",
    )
    url: str | None = Field(
        default=None,
        description="Endpoint URL for remote-type agents",
    )

    @model_validator(mode="before")
    @classmethod
    def _infer_type(cls, values: dict[str, object]) -> dict[str, object]:
        if not isinstance(values, dict):
            return values
        has_model = values.get("model") is not None
        has_type = values.get("type") is not None
        if has_model and not has_type:
            values["type"] = "llm"
        if not has_model and not has_type:
            msg = (
                "Agent must specify either 'type' or 'model' (model implies type='llm')"
            )
            raise ValueError(msg)
        return values

    @model_validator(mode="after")
    def _validate_type_requirements(self) -> AgentConfig:
        match self.type:
            case "llm":
                if not self.model:
                    msg = self._field_error("model")
                    raise ValueError(msg)
                if not _MODEL_RE.match(self.model):
                    msg = (
                        f"Invalid model format '{self.model}' — "
                        "expected 'provider/model-name'"
                    )
                    raise ValueError(msg)
                if not self.role:
                    msg = self._field_error("role")
                    raise ValueError(msg)
            case "cli":
                if not self.role:
                    msg = self._field_error("role")
                    raise ValueError(msg)
                if not self.cli and not self.command:
                    msg = (
                        "Agent type 'cli' requires 'cli' or 'command', "
                        "but neither was provided"
                    )
                    raise ValueError(msg)
            case "script":
                if not self.command:
                    msg = self._field_error("command")
                    raise ValueError(msg)
            case "remote":
                if not self.url:
                    msg = self._field_error("url")
                    raise ValueError(msg)
        return self

    def _field_error(self, field: str) -> str:
        return f"Agent type '{self.type}' requires '{field}'"


class TopologyConfig(BaseModel):
    """Defines how agents are wired together."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["mesh", "pipeline", "hub", "custom"] = Field(
        default="mesh",
        description="Topology pattern",
    )
    flow: list[str] | None = Field(
        default=None,
        description="Ordered agent names for pipeline topology",
    )
    coordinator: str | None = Field(
        default=None,
        description="Hub coordinator agent name",
    )
    workers: list[str] | None = Field(
        default=None,
        description="Worker agent names for hub topology",
    )
    edges: dict[str, list[str]] | None = Field(
        default=None,
        description="Explicit edges for custom topology",
    )

    @model_validator(mode="after")
    def _validate_topology_fields(self) -> TopologyConfig:
        match self.type:
            case "pipeline":
                if not self.flow:
                    msg = "Topology 'pipeline' requires 'flow'"
                    raise ValueError(msg)
            case "hub":
                missing = [
                    f for f in ("coordinator", "workers") if getattr(self, f) is None
                ]
                if missing:
                    joined = " and ".join(f"'{f}'" for f in missing)
                    msg = f"Topology 'hub' requires {joined}"
                    raise ValueError(msg)
            case "custom":
                if not self.edges:
                    msg = "Topology 'custom' requires 'edges'"
                    raise ValueError(msg)
        return self


class CommunicationConfig(BaseModel):
    """Settings for inter-agent communication."""

    model_config = ConfigDict(extra="forbid")

    protocol: str = Field(
        default="a2a",
        description="Communication protocol",
    )
    record: bool = Field(
        default=True,
        description="Whether to record message transcripts",
    )
    heartbeat: int = Field(
        default=20,
        ge=0,
        description="Heartbeat interval in seconds (0 to disable)",
    )


class LatticeConfig(BaseModel):
    """Top-level lattice.yaml configuration."""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(description="Config schema version")
    team: str = Field(description="Team name")
    description: str | None = Field(
        default=None,
        description="Human-readable team description",
    )
    output: str | None = Field(
        default=None,
        description="Default output directory",
    )
    entry: str | None = Field(
        default=None,
        description="Entry-point agent name (defaults to first agent)",
    )
    credentials: dict[str, str] | None = Field(
        default=None,
        description="Credential env-var mappings",
    )
    agents: dict[str, AgentConfig] = Field(
        description="Agent definitions (at least one required)",
    )
    topology: TopologyConfig = Field(
        default_factory=TopologyConfig,
        description="Agent topology configuration",
    )
    communication: CommunicationConfig = Field(
        default_factory=CommunicationConfig,
        description="Communication settings",
    )

    @model_validator(mode="after")
    def _validate_agents_and_entry(self) -> LatticeConfig:
        if not self.agents:
            msg = "At least one agent must be defined"
            raise ValueError(msg)
        if self.entry is None:
            self.entry = next(iter(self.agents))
        if self.entry not in self.agents:
            available = ", ".join(f"'{a}'" for a in self.agents)
            msg = (
                f"Entry agent '{self.entry}' not found — available agents: {available}"
            )
            raise ValueError(msg)
        self._validate_topology_references()
        return self

    def _validate_topology_references(self) -> None:
        agent_names = set(self.agents)
        topo = self.topology
        refs: list[str] = []
        if topo.flow:
            refs.extend(topo.flow)
        if topo.coordinator:
            refs.append(topo.coordinator)
        if topo.workers:
            refs.extend(topo.workers)
        if topo.edges:
            for src, dsts in topo.edges.items():
                refs.append(src)
                refs.extend(dsts)
        unknown = sorted(set(refs) - agent_names)
        if unknown:
            joined = ", ".join(f"'{n}'" for n in unknown)
            msg = f"Topology references unknown agents: {joined}"
            raise ValueError(msg)
