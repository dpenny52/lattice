"""Tests for the ``lattice up`` REPL — command parsing, routing, and lifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from lattice.agent.llm_agent import LLMAgent
from lattice.agent.providers import LLMResponse, TokenUsage
from lattice.commands.up import (
    UserAgent,
    _handle_command,
    _make_response_callback,
    _repl_loop,
    _run_session,
)
from lattice.config.models import (
    AgentConfig,
    LatticeConfig,
    TopologyConfig,
)
from lattice.router.router import Router
from lattice.session.recorder import SessionRecorder

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


class MockProvider:
    """Fake LLM provider that returns preconfigured responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
    ) -> LLMResponse:
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp


def _make_recorder(tmp_path: Path) -> SessionRecorder:
    return SessionRecorder("test-team", "abc123", sessions_dir=tmp_path / "sessions")


def _make_router(tmp_path: Path) -> tuple[Router, SessionRecorder]:
    recorder = _make_recorder(tmp_path)
    router = Router(topology=TopologyConfig(type="mesh"), recorder=recorder)
    return router, recorder


def _make_test_agent(
    router: Router,
    recorder: SessionRecorder,
    name: str = "agent-a",
    peer_names: list[str] | None = None,
    on_response: Any = None,
) -> LLMAgent:
    """Create an LLMAgent with a mock provider that returns a simple response."""
    provider = MockProvider(
        [LLMResponse(content="Hello from agent!", usage=TokenUsage(10, 5))]
    )
    return LLMAgent(
        name=name,
        model_string="mock/test-model",
        role="You are helpful.",
        router=router,
        recorder=recorder,
        team_name="test-team",
        peer_names=peer_names or ["user"],
        provider=provider,
        model_override="test-model",
        on_response=on_response,
    )


def _make_config(
    agents: dict[str, AgentConfig] | None = None,
    entry: str | None = None,
) -> LatticeConfig:
    """Build a minimal LatticeConfig for testing."""
    if agents is None:
        agents = {
            "researcher": AgentConfig(
                type="llm",
                model="anthropic/claude-sonnet-4-5-20250929",
                role="You are a researcher.",
            ),
            "writer": AgentConfig(
                type="llm",
                model="openai/gpt-4o",
                role="You are a writer.",
            ),
        }
    return LatticeConfig(
        version="0.1",
        team="test-team",
        agents=agents,
        entry=entry,
    )


# ================================================================== #
# UserAgent
# ================================================================== #


class TestUserAgent:
    async def test_handle_message_prints(self, capsys: Any) -> None:
        agent = UserAgent()
        await agent.handle_message("researcher", "here are my findings")
        captured = capsys.readouterr()
        assert "[researcher] here are my findings" in captured.out

    async def test_handle_message_different_agents(self, capsys: Any) -> None:
        agent = UserAgent()
        await agent.handle_message("agent-a", "first")
        await agent.handle_message("agent-b", "second")
        captured = capsys.readouterr()
        assert "[agent-a] first" in captured.out
        assert "[agent-b] second" in captured.out


# ================================================================== #
# Response callback
# ================================================================== #


class TestResponseCallback:
    def test_make_response_callback(self, capsys: Any) -> None:
        cb = _make_response_callback("writer")
        cb("The article is done.")
        captured = capsys.readouterr()
        assert "[writer] The article is done." in captured.out

    def test_callback_captures_agent_name(self, capsys: Any) -> None:
        """Each callback is bound to its own agent name."""
        cb_a = _make_response_callback("agent-a")
        cb_b = _make_response_callback("agent-b")
        cb_a("from a")
        cb_b("from b")
        captured = capsys.readouterr()
        assert "[agent-a] from a" in captured.out
        assert "[agent-b] from b" in captured.out


# ================================================================== #
# on_response callback integration with LLMAgent
# ================================================================== #


class TestOnResponseCallback:
    async def test_callback_fires_on_text_response(
        self, tmp_path: Path
    ) -> None:
        """on_response fires when the agent produces a plain-text reply."""
        router, recorder = _make_router(tmp_path)
        captured: list[str] = []

        agent = _make_test_agent(
            router,
            recorder,
            on_response=lambda content: captured.append(content),
        )
        router.register("agent-a", agent)

        await agent.handle_message("user", "hello")

        assert captured == ["Hello from agent!"]
        recorder.close()

    async def test_callback_not_called_on_empty_response(
        self, tmp_path: Path
    ) -> None:
        """on_response does NOT fire when the LLM returns empty content."""
        router, recorder = _make_router(tmp_path)
        captured: list[str] = []

        provider = MockProvider(
            [LLMResponse(content=None, tool_calls=[], usage=TokenUsage(5, 2))]
        )
        agent = LLMAgent(
            name="agent-a",
            model_string="mock/test",
            role="test",
            router=router,
            recorder=recorder,
            team_name="test-team",
            peer_names=["user"],
            provider=provider,
            model_override="test",
            on_response=lambda content: captured.append(content),
        )
        router.register("agent-a", agent)

        await agent.handle_message("user", "hi")

        assert captured == []
        recorder.close()

    async def test_no_callback_no_crash(self, tmp_path: Path) -> None:
        """Agent works fine without an on_response callback."""
        router, recorder = _make_router(tmp_path)
        agent = _make_test_agent(router, recorder, on_response=None)
        router.register("agent-a", agent)

        # Should not raise
        await agent.handle_message("user", "hello")

        thread = agent._threads["user"]
        assert thread[-1]["content"] == "Hello from agent!"
        recorder.close()


# ================================================================== #
# Slash commands
# ================================================================== #


class TestHandleCommand:
    def test_done_returns_true(self) -> None:
        assert _handle_command("/done", {}) is True

    def test_status_returns_false(self, capsys: Any) -> None:
        assert _handle_command("/status", {}) is False
        captured = capsys.readouterr()
        assert "Status:" in captured.out

    def test_agents_lists_agents(self, capsys: Any) -> None:
        agents = {"researcher": MagicMock(), "writer": MagicMock()}
        assert _handle_command("/agents", agents) is False
        captured = capsys.readouterr()
        assert "researcher (llm)" in captured.out
        assert "writer (llm)" in captured.out

    def test_unknown_command(self, capsys: Any) -> None:
        assert _handle_command("/foobar", {}) is False
        captured = capsys.readouterr()
        assert "Unknown command: /foobar" in captured.out

    def test_command_case_insensitive(self) -> None:
        assert _handle_command("/Done", {}) is True
        assert _handle_command("/DONE", {}) is True

    def test_command_with_trailing_text(self) -> None:
        """Only the first word matters for commands."""
        assert _handle_command("/done extra stuff", {}) is True


# ================================================================== #
# REPL loop — input parsing and routing
# ================================================================== #


class TestReplLoop:
    async def test_plain_text_routes_to_entry(self, tmp_path: Path) -> None:
        """Undecorated input goes to the entry agent."""
        router, recorder = _make_router(tmp_path)
        mock_agent = MagicMock()
        mock_agent.handle_message = AsyncMock()
        router.register("entry", mock_agent)

        agents = {"entry": MagicMock(spec=LLMAgent)}
        shutdown = asyncio.Event()

        # Simulate: user types "hello", then "/done"
        inputs = iter(["hello", "/done"])
        with patch("lattice.commands.up._read_input", side_effect=inputs):
            await _repl_loop(router, "entry", agents, shutdown)

        # Wait for async dispatch
        await asyncio.sleep(0.05)
        mock_agent.handle_message.assert_called_once_with("user", "hello")
        recorder.close()

    async def test_at_agent_routes_directly(self, tmp_path: Path) -> None:
        """@agent syntax routes to the named agent."""
        router, recorder = _make_router(tmp_path)
        writer_mock = MagicMock()
        writer_mock.handle_message = AsyncMock()
        router.register("writer", writer_mock)
        router.register("researcher", MagicMock())

        agents = {
            "researcher": MagicMock(spec=LLMAgent),
            "writer": MagicMock(spec=LLMAgent),
        }
        shutdown = asyncio.Event()

        inputs = iter(["@writer draft the intro", "/done"])
        with patch("lattice.commands.up._read_input", side_effect=inputs):
            await _repl_loop(router, "researcher", agents, shutdown)

        await asyncio.sleep(0.05)
        writer_mock.handle_message.assert_called_once_with(
            "user", "draft the intro"
        )
        recorder.close()

    async def test_at_unknown_agent(self, tmp_path: Path, capsys: Any) -> None:
        """@nonexistent prints an error, doesn't crash."""
        router, recorder = _make_router(tmp_path)
        router.register("entry", MagicMock())

        agents = {"entry": MagicMock(spec=LLMAgent)}
        shutdown = asyncio.Event()

        inputs = iter(["@ghost do something", "/done"])
        with patch("lattice.commands.up._read_input", side_effect=inputs):
            await _repl_loop(router, "entry", agents, shutdown)

        captured = capsys.readouterr()
        assert "Unknown agent: ghost" in captured.out
        recorder.close()

    async def test_at_agent_no_message(self, tmp_path: Path, capsys: Any) -> None:
        """@agent with no message body prints an error."""
        router, recorder = _make_router(tmp_path)
        router.register("entry", MagicMock())

        agents = {"entry": MagicMock(spec=LLMAgent)}
        shutdown = asyncio.Event()

        inputs = iter(["@entry", "/done"])
        with patch("lattice.commands.up._read_input", side_effect=inputs):
            await _repl_loop(router, "entry", agents, shutdown)

        captured = capsys.readouterr()
        assert "No message provided for @entry" in captured.out
        recorder.close()

    async def test_empty_input_ignored(self, tmp_path: Path) -> None:
        """Blank lines are silently skipped."""
        router, recorder = _make_router(tmp_path)
        entry_mock = MagicMock()
        entry_mock.handle_message = AsyncMock()
        router.register("entry", entry_mock)

        agents = {"entry": MagicMock(spec=LLMAgent)}
        shutdown = asyncio.Event()

        inputs = iter(["", "  ", "\t", "/done"])
        with patch("lattice.commands.up._read_input", side_effect=inputs):
            await _repl_loop(router, "entry", agents, shutdown)

        # No messages should have been sent
        entry_mock.handle_message.assert_not_called()
        recorder.close()

    async def test_eof_exits_cleanly(self, tmp_path: Path) -> None:
        """EOFError (Ctrl+D) exits the loop without crashing."""
        router, recorder = _make_router(tmp_path)
        router.register("entry", MagicMock())

        agents = {"entry": MagicMock(spec=LLMAgent)}
        shutdown = asyncio.Event()

        with patch("lattice.commands.up._read_input", side_effect=EOFError):
            await _repl_loop(router, "entry", agents, shutdown)

        # If we get here, it exited cleanly
        recorder.close()

    async def test_shutdown_event_exits_loop(self, tmp_path: Path) -> None:
        """Setting the shutdown event stops the REPL."""
        router, recorder = _make_router(tmp_path)
        router.register("entry", MagicMock())

        agents = {"entry": MagicMock(spec=LLMAgent)}
        shutdown = asyncio.Event()
        shutdown.set()

        # Should return immediately because shutdown is already set
        with patch("lattice.commands.up._read_input") as mock_input:
            await _repl_loop(router, "entry", agents, shutdown)
            mock_input.assert_not_called()

        recorder.close()

    async def test_done_command_exits(self, tmp_path: Path) -> None:
        """/done exits the REPL."""
        router, recorder = _make_router(tmp_path)
        router.register("entry", MagicMock())

        agents = {"entry": MagicMock(spec=LLMAgent)}
        shutdown = asyncio.Event()

        inputs = iter(["/done"])
        with patch("lattice.commands.up._read_input", side_effect=inputs):
            await _repl_loop(router, "entry", agents, shutdown)

        recorder.close()


# ================================================================== #
# Full session lifecycle
# ================================================================== #


class TestRunSession:
    async def test_startup_banner(self, tmp_path: Path, capsys: Any) -> None:
        """_run_session prints the startup banner with team info."""
        config = _make_config()

        # Mock LLMAgent creation to avoid real provider setup
        mock_agent = MagicMock(spec=LLMAgent)
        mock_agent.handle_message = AsyncMock()
        mock_agent.name = "researcher"

        with (
            patch("lattice.commands.up.LLMAgent", return_value=mock_agent),
            patch("lattice.commands.up._repl_loop", new_callable=AsyncMock),
            patch(
                "lattice.commands.up.SessionRecorder",
            ) as mock_recorder_cls,
        ):
            mock_recorder = MagicMock()
            mock_recorder.session_id = "abc123def456"
            mock_recorder.session_file = tmp_path / "test.jsonl"
            mock_recorder._pending_tasks = set()
            mock_recorder_cls.return_value = mock_recorder

            # Patch Router too so we control _pending_tasks
            with patch("lattice.commands.up.Router") as mock_router_cls:
                mock_router = MagicMock()
                mock_router._pending_tasks = set()
                mock_router_cls.return_value = mock_router

                await _run_session(config, verbose=False)

        captured = capsys.readouterr()
        assert "Lattice -- test-team" in captured.out
        assert "Agents: 2" in captured.out
        assert "abc123def456" in captured.out

    async def test_no_llm_agents_aborts(self, tmp_path: Path, capsys: Any) -> None:
        """If no LLM agents can be created, session aborts with error."""
        # Create config with only a non-LLM agent
        config = LatticeConfig(
            version="0.1",
            team="test-team",
            agents={
                "runner": AgentConfig(
                    type="script",
                    command="echo hello",
                ),
            },
        )

        with patch(
            "lattice.commands.up.SessionRecorder",
        ) as mock_recorder_cls:
            mock_recorder = MagicMock()
            mock_recorder.session_id = "abc123"
            mock_recorder.session_file = tmp_path / "test.jsonl"
            mock_recorder_cls.return_value = mock_recorder

            with patch("lattice.commands.up.Router") as mock_router_cls:
                mock_router = MagicMock()
                mock_router._pending_tasks = set()
                mock_router_cls.return_value = mock_router

                await _run_session(config, verbose=False)

        captured = capsys.readouterr()
        assert "No LLM agents configured" in captured.err
        mock_recorder.end.assert_called_once_with("error")


# ================================================================== #
# Edge cases and integration
# ================================================================== #


class TestEdgeCases:
    async def test_multiple_commands_in_sequence(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """Multiple commands work in sequence."""
        router, recorder = _make_router(tmp_path)
        entry_mock = MagicMock()
        entry_mock.handle_message = AsyncMock()
        router.register("entry", entry_mock)

        agents = {"entry": MagicMock(spec=LLMAgent)}
        shutdown = asyncio.Event()

        inputs = iter(["/status", "/agents", "/done"])
        with patch("lattice.commands.up._read_input", side_effect=inputs):
            await _repl_loop(router, "entry", agents, shutdown)

        captured = capsys.readouterr()
        assert "Status:" in captured.out
        assert "entry (llm)" in captured.out
        recorder.close()

    async def test_user_agent_satisfies_protocol(self) -> None:
        """UserAgent satisfies the Router Agent protocol."""
        from lattice.router.router import Agent

        assert isinstance(UserAgent(), Agent)
