"""Tests for loop mode (``lattice up --loop``)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from lattice.agent.llm_agent import LLMAgent
from lattice.agent.providers import LLMResponse, TokenUsage
from lattice.commands.up import _loop_mode, _run_session
from lattice.config.models import AgentConfig, LatticeConfig, TopologyConfig
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
        resp = self._responses[self._call_count % len(self._responses)]
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
    responses: list[LLMResponse] | None = None,
) -> LLMAgent:
    """Create an LLMAgent with a mock provider."""
    if responses is None:
        responses = [LLMResponse(content="Done!", usage=TokenUsage(10, 5))]

    provider = MockProvider(responses)
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
    )


def _make_config(entry: str | None = None) -> LatticeConfig:
    """Build a minimal LatticeConfig for testing."""
    return LatticeConfig(
        version="0.1",
        team="test-team",
        agents={
            "coordinator": AgentConfig(
                type="llm",
                model="anthropic/claude-sonnet-4-5-20250929",
                role="You are a coordinator.",
            ),
        },
        entry=entry or "coordinator",
    )


# ================================================================== #
# Loop mode basic functionality
# ================================================================== #


class TestLoopModeBasic:
    async def test_loop_completes_normally(self, tmp_path: Path) -> None:
        """Loop runs until agent signals done via heartbeat."""
        router, recorder = _make_router(tmp_path)
        agent = _make_test_agent(
            router,
            recorder,
            responses=[
                LLMResponse(content="Working...", usage=TokenUsage(10, 5)),
            ],
        )
        router.register("agent-a", agent)

        agents = {"agent-a": agent}
        all_agents = {"agent-a": agent}
        shutdown_event = asyncio.Event()

        # Mock heartbeat that signals done after first iteration
        mock_heartbeat = MagicMock()
        mock_heartbeat.done_flag = False
        mock_heartbeat.start = AsyncMock()

        async def set_done_after_first_iteration() -> None:
            # Wait for first iteration to start
            await asyncio.sleep(0.15)
            mock_heartbeat.done_flag = True

        # Simulate user input
        with patch("lattice.commands.up._read_input", return_value="test prompt"):
            asyncio.create_task(set_done_after_first_iteration())

            reason, count = await _loop_mode(
                router,
                "agent-a",
                all_agents,
                agents,
                shutdown_event,
                mock_heartbeat,
                recorder,
                -1,  # infinite loop
            )

        assert reason == "complete"
        # Should complete after a small number of iterations (timing dependent)
        assert 1 <= count <= 5
        recorder.close()

    async def test_max_iterations_respected(self, tmp_path: Path) -> None:
        """Loop stops after max iterations."""
        router, recorder = _make_router(tmp_path)
        agent = _make_test_agent(router, recorder)
        router.register("agent-a", agent)

        agents = {"agent-a": agent}
        all_agents = {"agent-a": agent}
        shutdown_event = asyncio.Event()

        with patch("lattice.commands.up._read_input", return_value="test prompt"):
            reason, count = await _loop_mode(
                router,
                "agent-a",
                all_agents,
                agents,
                shutdown_event,
                None,  # no heartbeat
                recorder,
                3,  # max 3 iterations
            )

        assert reason == "complete"
        assert count == 3
        recorder.close()

    async def test_ctrl_c_stops_loop(self, tmp_path: Path) -> None:
        """Setting shutdown_event stops the loop."""
        router, recorder = _make_router(tmp_path)
        agent = _make_test_agent(router, recorder)
        router.register("agent-a", agent)

        agents = {"agent-a": agent}
        all_agents = {"agent-a": agent}
        shutdown_event = asyncio.Event()

        async def trigger_shutdown() -> None:
            await asyncio.sleep(0.2)
            shutdown_event.set()

        with patch("lattice.commands.up._read_input", return_value="test prompt"):
            task = asyncio.create_task(
                _loop_mode(
                    router,
                    "agent-a",
                    all_agents,
                    agents,
                    shutdown_event,
                    None,
                    recorder,
                    -1,  # infinite
                )
            )
            asyncio.create_task(trigger_shutdown())
            reason, count = await task

        assert reason == "ctrl_c"
        assert count >= 0  # at least started
        recorder.close()

    async def test_eof_on_prompt_input(self, tmp_path: Path) -> None:
        """EOFError when reading prompt exits cleanly."""
        router, recorder = _make_router(tmp_path)
        agent = _make_test_agent(router, recorder)
        router.register("agent-a", agent)

        agents = {"agent-a": agent}
        all_agents = {"agent-a": agent}
        shutdown_event = asyncio.Event()

        with patch("lattice.commands.up._read_input", side_effect=EOFError):
            reason, count = await _loop_mode(
                router,
                "agent-a",
                all_agents,
                agents,
                shutdown_event,
                None,
                recorder,
                -1,
            )

        assert reason == "ctrl_c"
        assert count == 0
        recorder.close()

    async def test_empty_prompt_exits(self, tmp_path: Path) -> None:
        """Empty prompt exits loop mode."""
        router, recorder = _make_router(tmp_path)
        agent = _make_test_agent(router, recorder)
        router.register("agent-a", agent)

        agents = {"agent-a": agent}
        all_agents = {"agent-a": agent}
        shutdown_event = asyncio.Event()

        with patch("lattice.commands.up._read_input", return_value="   "):
            reason, count = await _loop_mode(
                router,
                "agent-a",
                all_agents,
                agents,
                shutdown_event,
                None,
                recorder,
                -1,
            )

        assert reason == "user_shutdown"
        assert count == 0
        recorder.close()


# ================================================================== #
# Loop boundary events
# ================================================================== #


class TestLoopBoundaryEvents:
    async def test_loop_boundary_events_recorded(self, tmp_path: Path) -> None:
        """Each loop iteration logs start/end boundary events."""
        router, recorder = _make_router(tmp_path)
        agent = _make_test_agent(router, recorder)
        router.register("agent-a", agent)

        agents = {"agent-a": agent}
        all_agents = {"agent-a": agent}
        shutdown_event = asyncio.Event()

        with patch("lattice.commands.up._read_input", return_value="test"):
            await _loop_mode(
                router, "agent-a", all_agents, agents, shutdown_event, None, recorder, 2
            )

        # Read the JSONL file and check for loop boundary events
        events = []
        with recorder.session_file.open() as f:
            import json

            for line in f:
                event = json.loads(line)
                if event.get("type") == "loop_boundary":
                    events.append(event)

        # Should have 2 loops × 2 boundaries (start, end) = 4 events
        assert len(events) == 4

        # Check structure
        assert events[0]["boundary"] == "start"
        assert events[0]["iteration"] == 1
        assert events[1]["boundary"] == "end"
        assert events[1]["iteration"] == 1
        assert events[2]["boundary"] == "start"
        assert events[2]["iteration"] == 2
        assert events[3]["boundary"] == "end"
        assert events[3]["iteration"] == 2

        recorder.close()


# ================================================================== #
# Context reset
# ================================================================== #


class TestContextReset:
    async def test_context_reset_between_loops(self, tmp_path: Path) -> None:
        """Agent conversation threads are cleared between iterations."""
        router, recorder = _make_router(tmp_path)
        agent = _make_test_agent(router, recorder)
        router.register("agent-a", agent)

        agents = {"agent-a": agent}
        all_agents = {"agent-a": agent}
        shutdown_event = asyncio.Event()

        # Send a message to populate the agent's thread
        await agent.handle_message("user", "initial message")
        assert len(agent._thread) > 0

        # Run one loop iteration
        with patch("lattice.commands.up._read_input", return_value="test"):
            await _loop_mode(
                router, "agent-a", all_agents, agents, shutdown_event, None, recorder, 1
            )

        # After reset, the thread should have been cleared and repopulated
        # but should not contain the "initial message" from before the loop
        initial_msg_found = any(
            msg.get("content") == "[from user]: initial message"
            for msg in agent._thread
        )
        assert not initial_msg_found

        recorder.close()


# ================================================================== #
# Integration with _run_session
# ================================================================== #


class TestRunSessionIntegration:
    async def test_loop_count_in_summary(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """Loop count appears in the shutdown summary."""
        config = _make_config()

        # Mock LLMAgent creation
        mock_agent = MagicMock(spec=LLMAgent)
        mock_agent.handle_message = AsyncMock()
        mock_agent.name = "coordinator"
        mock_agent.reset_context = MagicMock()
        mock_agent._threads = {}

        with (
            patch("lattice.commands.up.LLMAgent", return_value=mock_agent),
            patch("lattice.commands.up._read_input", return_value="test prompt"),
            patch("lattice.commands.up._loop_mode") as mock_loop_mode,
            patch("lattice.commands.up.SessionRecorder") as mock_recorder_cls,
        ):
            # Mock loop mode to return after 3 loops
            mock_loop_mode.return_value = ("complete", 3)

            mock_recorder = MagicMock()
            mock_recorder.session_id = "abc123"
            mock_recorder.session_file = tmp_path / "test.jsonl"
            mock_recorder.event_count = 42
            mock_recorder_cls.return_value = mock_recorder

            with patch("lattice.commands.up.Router") as mock_router_cls:
                mock_router = MagicMock()
                mock_router._pending_tasks = set()
                mock_router.pending_tasks = set()
                mock_router_cls.return_value = mock_router

                await _run_session(config, verbose=False, loop_iterations=3)

        captured = capsys.readouterr()
        assert "3 loop(s)" in captured.out

    async def test_no_loop_count_without_loop_mode(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """Loop count does not appear in summary when not in loop mode."""
        config = _make_config()

        mock_agent = MagicMock(spec=LLMAgent)
        mock_agent.handle_message = AsyncMock()
        mock_agent.name = "coordinator"

        with (
            patch("lattice.commands.up.LLMAgent", return_value=mock_agent),
            patch("lattice.commands.up._repl_loop") as mock_repl,
            patch("lattice.commands.up.SessionRecorder") as mock_recorder_cls,
        ):
            mock_repl.return_value = "user_shutdown"

            mock_recorder = MagicMock()
            mock_recorder.session_id = "abc123"
            mock_recorder.session_file = tmp_path / "test.jsonl"
            mock_recorder.event_count = 42
            mock_recorder_cls.return_value = mock_recorder

            with patch("lattice.commands.up.Router") as mock_router_cls:
                mock_router = MagicMock()
                mock_router._pending_tasks = set()
                mock_router.pending_tasks = set()
                mock_router_cls.return_value = mock_router

                await _run_session(config, verbose=False, loop_iterations=None)

        captured = capsys.readouterr()
        assert "loop(s)" not in captured.out
