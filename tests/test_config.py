"""Tests for lattice config models and parser."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from lattice.config.models import (
    AgentConfig,
    CommunicationConfig,
    LatticeConfig,
    TopologyConfig,
)
from lattice.config.parser import ConfigError, load_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_raw(
    **overrides: Any,
) -> dict[str, Any]:
    """Return the smallest valid config dict, with optional overrides."""
    base: dict[str, Any] = {
        "version": "1",
        "team": "test-team",
        "agents": {
            "agent1": {"model": "anthropic/claude-sonnet-4-5", "role": "do stuff"},
        },
    }
    base.update(overrides)
    return base


def _write_yaml(path: Path, data: dict[str, Any]) -> Path:
    """Write *data* as YAML and return the file path."""
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


# ===================================================================
# Model validation tests
# ===================================================================


class TestMinimalConfig:
    def test_parses_successfully(self) -> None:
        cfg = LatticeConfig.model_validate(_minimal_raw())
        assert cfg.version == "1"
        assert cfg.team == "test-team"
        assert "agent1" in cfg.agents

    def test_defaults_applied(self) -> None:
        cfg = LatticeConfig.model_validate(_minimal_raw())
        assert cfg.entry == "agent1"
        assert cfg.topology.type == "mesh"
        assert cfg.communication.protocol == "a2a"
        assert cfg.communication.record is True


class TestFullConfig:
    def test_all_fields(self) -> None:
        raw = _minimal_raw(
            description="A full team",
            output="./out.md",
            entry="agent1",
            credentials={"anthropic": "MY_KEY"},
            topology={"type": "mesh"},
            communication={"protocol": "a2a", "record": False, "heartbeat": 10},
        )
        raw["agents"]["cli_agent"] = {
            "type": "cli",
            "cli": "claude",
            "role": "do cli things",
        }
        raw["agents"]["script_agent"] = {
            "type": "script",
            "command": "python run.py",
        }
        cfg = LatticeConfig.model_validate(raw)
        assert cfg.description == "A full team"
        assert len(cfg.agents) == 3
        assert cfg.communication.heartbeat == 10


# ===================================================================
# Agent validation
# ===================================================================


class TestAgentDefaults:
    def test_model_implies_llm_type(self) -> None:
        agent = AgentConfig.model_validate(
            {"model": "openai/gpt-4o", "role": "analyst"}
        )
        assert agent.type == "llm"

    def test_tools_default_empty(self) -> None:
        agent = AgentConfig.model_validate(
            {"model": "openai/gpt-4o", "role": "analyst"}
        )
        assert agent.tools == []


class TestAgentLLM:
    def test_requires_model(self) -> None:
        with pytest.raises(ValidationError, match="model"):
            AgentConfig.model_validate({"type": "llm", "role": "stuff"})

    def test_requires_role(self) -> None:
        with pytest.raises(ValidationError, match="role"):
            AgentConfig.model_validate(
                {"type": "llm", "model": "anthropic/claude-sonnet-4-5"}
            )


class TestAgentCLI:
    def test_requires_role(self) -> None:
        with pytest.raises(ValidationError, match="role"):
            AgentConfig.model_validate({"type": "cli", "cli": "claude"})

    def test_requires_cli_or_command(self) -> None:
        with pytest.raises(ValidationError, match="cli.*command"):
            AgentConfig.model_validate({"type": "cli", "role": "do stuff"})

    def test_cli_field_valid(self) -> None:
        agent = AgentConfig.model_validate(
            {"type": "cli", "cli": "claude", "role": "do stuff"}
        )
        assert agent.cli == "claude"

    def test_command_field_valid(self) -> None:
        agent = AgentConfig.model_validate(
            {"type": "cli", "command": "./my-agent.sh", "role": "do stuff"}
        )
        assert agent.command == "./my-agent.sh"


class TestAgentScript:
    def test_requires_command(self) -> None:
        with pytest.raises(ValidationError, match="command"):
            AgentConfig.model_validate({"type": "script"})

    def test_valid(self) -> None:
        agent = AgentConfig.model_validate(
            {"type": "script", "command": "python run.py"}
        )
        assert agent.command == "python run.py"


class TestAgentRemote:
    def test_requires_url(self) -> None:
        with pytest.raises(ValidationError, match="url"):
            AgentConfig.model_validate({"type": "remote"})

    def test_valid(self) -> None:
        agent = AgentConfig.model_validate(
            {"type": "remote", "url": "https://api.example.com"}
        )
        assert agent.url == "https://api.example.com"


class TestAgentHuman:
    def test_valid(self) -> None:
        agent = AgentConfig.model_validate({"type": "human"})
        assert agent.type == "human"


class TestAgentNoTypeNoModel:
    def test_raises(self) -> None:
        with pytest.raises(ValidationError, match="type.*model"):
            AgentConfig.model_validate({"role": "stuff"})


# ===================================================================
# Model format validation
# ===================================================================


class TestModelFormat:
    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-sonnet-4-5",
            "openai/gpt-4o",
            "google/gemini-2.0-flash",
            "ollama/llama3.3",
        ],
    )
    def test_valid_formats(self, model: str) -> None:
        agent = AgentConfig.model_validate({"model": model, "role": "stuff"})
        assert agent.model == model

    @pytest.mark.parametrize(
        "model",
        [
            "no-slash",
            "/leading-slash",
            "trailing-slash/",
            "",
        ],
    )
    def test_invalid_formats(self, model: str) -> None:
        with pytest.raises(ValidationError, match="model format|model"):
            AgentConfig.model_validate({"model": model, "role": "stuff"})


# ===================================================================
# Topology validation
# ===================================================================


class TestTopology:
    def test_defaults_to_mesh(self) -> None:
        cfg = LatticeConfig.model_validate(_minimal_raw())
        assert cfg.topology.type == "mesh"

    def test_pipeline_requires_flow(self) -> None:
        with pytest.raises(ValidationError, match="flow"):
            TopologyConfig.model_validate({"type": "pipeline"})

    def test_pipeline_valid(self) -> None:
        t = TopologyConfig.model_validate({"type": "pipeline", "flow": ["a", "b", "c"]})
        assert t.flow == ["a", "b", "c"]

    def test_hub_requires_coordinator_and_workers(self) -> None:
        with pytest.raises(ValidationError, match="coordinator.*workers"):
            TopologyConfig.model_validate({"type": "hub"})

    def test_hub_valid(self) -> None:
        t = TopologyConfig.model_validate(
            {"type": "hub", "coordinator": "coord", "workers": ["w1", "w2"]}
        )
        assert t.coordinator == "coord"

    def test_custom_requires_edges(self) -> None:
        with pytest.raises(ValidationError, match="edges"):
            TopologyConfig.model_validate({"type": "custom"})

    def test_custom_valid(self) -> None:
        t = TopologyConfig.model_validate(
            {"type": "custom", "edges": {"a": ["b"], "b": ["a"]}}
        )
        assert t.edges == {"a": ["b"], "b": ["a"]}


# ===================================================================
# Communication defaults
# ===================================================================


class TestCommunication:
    def test_defaults(self) -> None:
        c = CommunicationConfig.model_validate({})
        assert c.protocol == "a2a"
        assert c.record is True
        assert c.heartbeat == 20


# ===================================================================
# LatticeConfig-level validation
# ===================================================================


class TestLatticeConfigEntry:
    def test_defaults_to_first_agent(self) -> None:
        cfg = LatticeConfig.model_validate(_minimal_raw())
        assert cfg.entry == "agent1"

    def test_invalid_entry(self) -> None:
        with pytest.raises(ValidationError, match="not found"):
            LatticeConfig.model_validate(_minimal_raw(entry="nonexistent"))


class TestLatticeConfigRequired:
    def test_missing_version(self) -> None:
        raw = _minimal_raw()
        del raw["version"]
        with pytest.raises(ValidationError, match="version"):
            LatticeConfig.model_validate(raw)

    def test_missing_team(self) -> None:
        raw = _minimal_raw()
        del raw["team"]
        with pytest.raises(ValidationError, match="team"):
            LatticeConfig.model_validate(raw)

    def test_empty_agents(self) -> None:
        with pytest.raises(ValidationError, match="(?i)at least one"):
            LatticeConfig.model_validate(_minimal_raw(agents={}))


class TestExtraFieldsForbidden:
    def test_top_level(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            LatticeConfig.model_validate(_minimal_raw(bogus="nope"))

    def test_agent_level(self) -> None:
        raw = _minimal_raw()
        raw["agents"]["agent1"]["bogus"] = "nope"
        with pytest.raises(ValidationError, match="extra"):
            LatticeConfig.model_validate(raw)


# ===================================================================
# Parser tests
# ===================================================================


class TestLoadConfig:
    def test_from_file(self, tmp_path: Path) -> None:
        cfg_file = _write_yaml(tmp_path / "lattice.yaml", _minimal_raw())
        cfg = load_config(cfg_file)
        assert cfg.team == "test-team"

    def test_default_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        _write_yaml(tmp_path / "lattice.yaml", _minimal_raw())
        cfg = load_config()
        assert cfg.team == "test-team"

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="not found"):
            load_config(tmp_path / "nope.yaml")

    def test_missing_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ConfigError, match="lattice init"):
            load_config()

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("{{{{not yaml", encoding="utf-8")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            load_config(bad)

    def test_yaml_not_mapping(self, tmp_path: Path) -> None:
        bad = tmp_path / "list.yaml"
        bad.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="mapping"):
            load_config(bad)

    def test_validation_error_messages(self, tmp_path: Path) -> None:
        cfg_file = _write_yaml(tmp_path / "bad.yaml", {"version": "1"})
        with pytest.raises(ConfigError, match="validation failed"):
            load_config(cfg_file)


class TestRoleFileReference:
    def test_resolves_relative_path(self, tmp_path: Path) -> None:
        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "researcher.md").write_text(
            "You are a researcher.", encoding="utf-8"
        )
        raw = _minimal_raw()
        raw["agents"]["agent1"]["role"] = "./prompts/researcher.md"
        cfg_file = _write_yaml(tmp_path / "lattice.yaml", raw)
        cfg = load_config(cfg_file)
        assert cfg.agents["agent1"].role == "You are a researcher."

    def test_missing_role_file(self, tmp_path: Path) -> None:
        raw = _minimal_raw()
        raw["agents"]["agent1"]["role"] = "./prompts/missing.md"
        cfg_file = _write_yaml(tmp_path / "lattice.yaml", raw)
        with pytest.raises(ConfigError, match="Role file not found"):
            load_config(cfg_file)

    def test_path_traversal_blocked(self, tmp_path: Path) -> None:
        raw = _minimal_raw()
        raw["agents"]["agent1"]["role"] = "./../../etc/passwd"
        cfg_file = _write_yaml(tmp_path / "lattice.yaml", raw)
        with pytest.raises(ConfigError, match="escapes project"):
            load_config(cfg_file)


class TestTopologyCrossValidation:
    def test_pipeline_unknown_agent(self) -> None:
        raw = _minimal_raw(topology={"type": "pipeline", "flow": ["agent1", "ghost"]})
        with pytest.raises(ValidationError, match="unknown agents.*ghost"):
            LatticeConfig.model_validate(raw)

    def test_hub_unknown_worker(self) -> None:
        raw = _minimal_raw(
            topology={
                "type": "hub",
                "coordinator": "agent1",
                "workers": ["ghost"],
            }
        )
        with pytest.raises(ValidationError, match="unknown agents.*ghost"):
            LatticeConfig.model_validate(raw)

    def test_valid_topology_refs(self) -> None:
        raw = _minimal_raw()
        raw["agents"]["agent2"] = {
            "model": "openai/gpt-4o",
            "role": "helper",
        }
        raw["topology"] = {
            "type": "pipeline",
            "flow": ["agent1", "agent2"],
        }
        cfg = LatticeConfig.model_validate(raw)
        assert cfg.topology.flow == ["agent1", "agent2"]


class TestHeartbeatBounds:
    def test_zero_disables(self) -> None:
        c = CommunicationConfig.model_validate({"heartbeat": 0})
        assert c.heartbeat == 0

    def test_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CommunicationConfig.model_validate({"heartbeat": -1})


class TestEnvLoading:
    def test_loads_env_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LATTICE_TEST_KEY", raising=False)
        (tmp_path / ".env").write_text("LATTICE_TEST_KEY=secret123\n", encoding="utf-8")
        _write_yaml(tmp_path / "lattice.yaml", _minimal_raw())
        load_config(tmp_path / "lattice.yaml")
        import os

        assert os.environ.get("LATTICE_TEST_KEY") == "secret123"

    def test_no_env_is_fine(self, tmp_path: Path) -> None:
        cfg_file = _write_yaml(tmp_path / "lattice.yaml", _minimal_raw())
        cfg = load_config(cfg_file)
        assert cfg.team == "test-team"
