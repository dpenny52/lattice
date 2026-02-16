"""Tests for error handling and user-facing error messages."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.agent.providers import (
    AnthropicProvider,
    GoogleProvider,
    OpenAIProvider,
)
from lattice.config.models import TopologyConfig
from lattice.config.parser import ConfigError, load_config


class TestProviderAPIKeyErrors:
    """Test that providers fail with clear messages when API keys are missing."""

    def test_anthropic_missing_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AnthropicProvider should raise ValueError with helpful message."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY not found"):
            AnthropicProvider(api_key=None)

    def test_openai_missing_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OpenAIProvider should raise ValueError with helpful message."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OPENAI_API_KEY not found"):
            OpenAIProvider(api_key=None)

    def test_google_missing_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """GoogleProvider should raise ValueError when key is missing at runtime."""
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        provider = GoogleProvider(api_key=None)
        # Google provider delays the check until chat() is called
        with pytest.raises(ValueError, match="GOOGLE_API_KEY not found"):
            import asyncio
            asyncio.run(provider.chat([], [], "gemini-2.0-flash"))

    def test_anthropic_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AnthropicProvider should succeed with valid API key."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key-123")
        provider = AnthropicProvider(api_key=None)
        assert provider is not None

    def test_openai_with_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """OpenAIProvider should succeed with valid API key."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-123")
        provider = OpenAIProvider(api_key=None)
        assert provider is not None


class TestConfigErrorMessages:
    """Test that config validation errors are user-friendly."""

    def test_missing_config_file(self, tmp_path: Path) -> None:
        """Missing lattice.yaml should suggest running lattice init."""
        os.chdir(tmp_path)
        with pytest.raises(
            ConfigError,
            match="No lattice.yaml found. Run `lattice init` to create one.",
        ):
            load_config(None)

    def test_invalid_yaml_syntax(self, tmp_path: Path) -> None:
        """Invalid YAML should show line/column information."""
        config = tmp_path / "lattice.yaml"
        config.write_text("team: my-team\ninvalid: [unclosed bracket\n")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            load_config(config)

    def test_validation_field_errors(self, tmp_path: Path) -> None:
        """Validation errors should show field paths clearly."""
        config = tmp_path / "lattice.yaml"
        config.write_text(
            """
version: "1"
team: test
agents:
  worker:
    type: llm
    # missing required 'model' field
    role: test
"""
        )
        with pytest.raises(ConfigError, match="requires 'model'"):
            load_config(config)


class TestSubprocessErrorReporting:
    """Test that subprocess crashes show stderr clearly."""

    async def test_script_nonzero_exit_shows_stderr(self, tmp_path: Path) -> None:
        """Script exit errors should show last 5 lines of stderr."""
        from lattice.agent.script_bridge import ScriptBridge
        from lattice.router.router import Router
        from lattice.session.recorder import SessionRecorder

        recorder = SessionRecorder(team="test", config_hash="abc123")
        router = Router(
            topology=TopologyConfig(type="mesh"),
            recorder=recorder,
        )

        script = tmp_path / "fail.sh"
        script.write_text("#!/bin/sh\necho 'Error line 1' >&2\nexit 1\n")
        script.chmod(0o755)

        bridge = ScriptBridge(
            name="failer",
            role="test",
            command=str(script),
            router=router,
            recorder=recorder,
        )

        # Capture click.echo calls
        with patch("click.echo") as mock_echo:
            await bridge.handle_message("user", "test")

        # Verify error message was shown with exact format
        error_calls = [
            call for call in mock_echo.call_args_list
            if len(call[0]) > 0 and "exited with code" in str(call[0][0])
        ]
        assert len(error_calls) > 0
        error_msg = str(error_calls[0][0][0])
        assert "exited with code 1." in error_msg
        assert "Stderr:" in error_msg
        assert "Error line 1" in error_msg

    async def test_cli_bridge_shows_claude_not_found(self, tmp_path: Path) -> None:
        """CLI bridge should show helpful message when Claude CLI is missing."""
        from lattice.agent.cli_bridge import CLIBridge
        from lattice.router.router import Router
        from lattice.session.recorder import SessionRecorder

        recorder = SessionRecorder(team="test", config_hash="abc123")
        router = Router(
            topology=TopologyConfig(type="mesh"),
            recorder=recorder,
        )

        bridge = CLIBridge(
            name="worker",
            role="test",
            router=router,
            recorder=recorder,
            team_name="test",
            peer_names=["user"],
            cli_type="claude",
        )

        # Mock subprocess to raise FileNotFoundError
        with (
            patch("lattice.memory_monitor.get_available_mb", return_value=8192.0),
            patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError()),
            patch("click.echo") as mock_echo,
        ):
            await bridge.handle_message("user", "test")

        # Verify helpful error message was shown
        error_calls = [call for call in mock_echo.call_args_list if len(call[0]) > 0]
        assert any("Claude CLI not found" in str(call[0][0]) for call in error_calls)
        assert any("npm install" in str(call[0][0]) for call in error_calls)


class TestRateLimitErrors:
    """Test that rate limit errors show clear messages."""

    async def test_rate_limit_shows_user_message(self, tmp_path: Path) -> None:
        """Rate limit errors should show user-friendly message with retry info."""
        from lattice.agent.llm_agent import LLMAgent, RateLimitGate
        from lattice.agent.providers import LLMProvider
        from lattice.router.router import Router
        from lattice.session.recorder import SessionRecorder

        recorder = SessionRecorder(team="test", config_hash="abc123")
        router = Router(
            topology=TopologyConfig(type="mesh"),
            recorder=recorder,
        )
        router.register("user", MagicMock())

        # Create mock provider that raises a 429 error
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.chat = AsyncMock(
            side_effect=Exception("429 rate_limit exceeded")
        )

        gate = RateLimitGate(pause_seconds=1.0)  # Short pause for testing

        agent = LLMAgent(
            name="test_agent",
            model_string="anthropic/claude-sonnet-4-5",
            role="test",
            router=router,
            recorder=recorder,
            team_name="test",
            peer_names=["user"],
            provider=mock_provider,
            rate_gate=gate,
        )

        with patch("click.echo") as mock_echo:
            await agent.handle_message("user", "test message")

        # Verify rate limit message was shown with exact format
        error_calls = [call for call in mock_echo.call_args_list if len(call[0]) > 0]
        assert any(
            "got a 429 from anthropic (rate limited)"
            in str(call).lower()
            for call in error_calls
        )

    async def test_auth_error_shows_user_message(self, tmp_path: Path) -> None:
        """401 auth errors should show user-friendly message."""
        from lattice.agent.llm_agent import LLMAgent
        from lattice.agent.providers import LLMProvider
        from lattice.router.router import Router
        from lattice.session.recorder import SessionRecorder

        recorder = SessionRecorder(team="test", config_hash="abc123")
        router = Router(
            topology=TopologyConfig(type="mesh"),
            recorder=recorder,
        )
        router.register("user", MagicMock())

        # Create mock provider that raises a 401 error
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.chat = AsyncMock(
            side_effect=Exception("401 Unauthorized")
        )

        agent = LLMAgent(
            name="test_agent",
            model_string="openai/gpt-4",
            role="test",
            router=router,
            recorder=recorder,
            team_name="test",
            peer_names=["user"],
            provider=mock_provider,
        )

        with patch("click.echo") as mock_echo:
            await agent.handle_message("user", "test message")

        # Verify auth error message was shown
        error_calls = [call for call in mock_echo.call_args_list if len(call[0]) > 0]
        assert any(
            "got a 401 from openai (authentication failed)"
            in str(call).lower()
            for call in error_calls
        )

    async def test_server_error_shows_user_message(self, tmp_path: Path) -> None:
        """500 server errors should show user-friendly message with retry info."""
        from lattice.agent.llm_agent import LLMAgent
        from lattice.agent.providers import LLMProvider
        from lattice.router.router import Router
        from lattice.session.recorder import SessionRecorder

        recorder = SessionRecorder(team="test", config_hash="abc123")
        router = Router(
            topology=TopologyConfig(type="mesh"),
            recorder=recorder,
        )
        router.register("user", MagicMock())

        # Create mock provider that raises a 500 error
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.chat = AsyncMock(
            side_effect=Exception("500 Internal Server Error")
        )

        agent = LLMAgent(
            name="test_agent",
            model_string="google/gemini-2.0-flash",
            role="test",
            router=router,
            recorder=recorder,
            team_name="test",
            peer_names=["user"],
            provider=mock_provider,
        )

        with patch("click.echo") as mock_echo:
            await agent.handle_message("user", "test message")

        # Verify server error message was shown
        error_calls = [call for call in mock_echo.call_args_list if len(call[0]) > 0]
        assert any(
            "got a 500 from google (server error)"
            in str(call).lower()
            for call in error_calls
        )


class TestNetworkErrors:
    """Test that network errors show clear messages."""

    async def test_connection_error_shows_user_message(self, tmp_path: Path) -> None:
        """Connection errors should show user-friendly message with retry info."""
        from lattice.agent.llm_agent import LLMAgent
        from lattice.agent.providers import LLMProvider
        from lattice.router.router import Router
        from lattice.session.recorder import SessionRecorder

        recorder = SessionRecorder(team="test", config_hash="abc123")
        router = Router(
            topology=TopologyConfig(type="mesh"),
            recorder=recorder,
        )
        router.register("user", MagicMock())

        # Create mock provider that raises a connection error
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.chat = AsyncMock(
            side_effect=Exception("Connection refused to api.anthropic.com")
        )

        agent = LLMAgent(
            name="test_agent",
            model_string="anthropic/claude-sonnet-4-5",
            role="test",
            router=router,
            recorder=recorder,
            team_name="test",
            peer_names=["user"],
            provider=mock_provider,
        )

        with patch("click.echo") as mock_echo:
            await agent.handle_message("user", "test message")

        # Verify network error message was shown
        error_calls = [call for call in mock_echo.call_args_list if len(call[0]) > 0]
        assert any("network error" in str(call).lower() for call in error_calls)


class TestErrorRecording:
    """Test that all errors are recorded in session JSONL."""

    async def test_llm_errors_recorded(self, tmp_path: Path) -> None:
        """LLM errors should be recorded as ErrorEvent."""
        from lattice.agent.llm_agent import LLMAgent
        from lattice.agent.providers import LLMProvider
        from lattice.router.router import Router
        from lattice.session.recorder import SessionRecorder

        recorder = SessionRecorder(team="test", config_hash="abc123")
        router = Router(
            topology=TopologyConfig(type="mesh"),
            recorder=recorder,
        )
        router.register("user", MagicMock())

        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.chat = AsyncMock(side_effect=Exception("Test error"))

        agent = LLMAgent(
            name="test_agent",
            model_string="anthropic/claude-sonnet-4-5",
            role="test",
            router=router,
            recorder=recorder,
            team_name="test",
            peer_names=["user"],
            provider=mock_provider,
        )

        with patch("click.echo"):
            await agent.handle_message("user", "test message")

        # Check that error events were recorded
        lines = recorder.session_file.read_text().strip().splitlines()
        events = [json.loads(line) for line in lines]
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) > 0
        assert error_events[0]["agent"] == "test_agent"
        assert "Test error" in error_events[0]["error"]
        assert error_events[0]["context"] == "api_call"

    async def test_script_errors_recorded(self, tmp_path: Path) -> None:
        """Script errors should be recorded as ErrorEvent."""
        from lattice.agent.script_bridge import ScriptBridge
        from lattice.router.router import Router
        from lattice.session.recorder import SessionRecorder

        recorder = SessionRecorder(team="test", config_hash="abc123")
        router = Router(
            topology=TopologyConfig(type="mesh"),
            recorder=recorder,
        )

        script = tmp_path / "fail.sh"
        script.write_text("#!/bin/sh\nexit 1\n")
        script.chmod(0o755)

        bridge = ScriptBridge(
            name="failer",
            role="test",
            command=str(script),
            router=router,
            recorder=recorder,
        )

        with patch("click.echo"):
            await bridge.handle_message("user", "test")

        # Check that error events were recorded
        lines = recorder.session_file.read_text().strip().splitlines()
        events = [json.loads(line) for line in lines]
        error_events = [e for e in events if e.get("type") == "error"]
        assert len(error_events) > 0
        assert error_events[0]["agent"] == "failer"
        assert error_events[0]["context"] == "subprocess"
