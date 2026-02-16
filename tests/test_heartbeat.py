"""Tests for the heartbeat mechanism."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from lattice.agent.llm_agent import LLMAgent
from lattice.agent.providers import LLMResponse, TokenUsage
from lattice.commands.up import (
    _handle_command,
    _install_heartbeat_hook,
    _repl_loop,
)
from lattice.config.models import TopologyConfig
from lattice.heartbeat import SYSTEM_SENDER, Heartbeat
from lattice.router.router import Router
from lattice.session.models import StatusEvent
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
        **kwargs: Any,
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


def _make_heartbeat(
    tmp_path: Path,
    interval: int = 20,
    entry_agent: str = "agent-a",
) -> tuple[Heartbeat, Router, SessionRecorder, asyncio.Event]:
    router, recorder = _make_router(tmp_path)
    shutdown_event = asyncio.Event()

    # Register a mock agent so the router can dispatch to it
    mock_agent = MagicMock()
    mock_agent.handle_message = AsyncMock()
    router.register(entry_agent, mock_agent)

    heartbeat = Heartbeat(
        interval=interval,
        router=router,
        entry_agent=entry_agent,
        recorder=recorder,
        shutdown_event=shutdown_event,
    )
    return heartbeat, router, recorder, shutdown_event


# ================================================================== #
# Heartbeat class unit tests
# ================================================================== #


class TestHeartbeatInit:
    def test_disabled_when_interval_zero(self, tmp_path: Path) -> None:
        """Heartbeat with interval=0 should not start a task."""
        heartbeat, _, recorder, _ = _make_heartbeat(tmp_path, interval=0)
        assert heartbeat._interval == 0
        recorder.close()

    async def test_start_noop_when_disabled(self, tmp_path: Path) -> None:
        """start() is a no-op when interval is 0."""
        heartbeat, _, recorder, _ = _make_heartbeat(tmp_path, interval=0)
        await heartbeat.start()
        assert heartbeat._task is None
        recorder.close()

    async def test_start_creates_task(self, tmp_path: Path) -> None:
        """start() creates an asyncio task when interval > 0."""
        heartbeat, _, recorder, shutdown = _make_heartbeat(tmp_path, interval=5)
        await heartbeat.start()
        assert heartbeat._task is not None
        assert not heartbeat._task.done()
        shutdown.set()
        await heartbeat.stop()
        recorder.close()


class TestHeartbeatFire:
    async def test_fire_sends_message_via_router(self, tmp_path: Path) -> None:
        """fire() sends a heartbeat message through the router."""
        heartbeat, router, recorder, _ = _make_heartbeat(tmp_path)
        agent = router._agents["agent-a"]

        await heartbeat.fire()

        # The router should have dispatched to the agent
        await asyncio.sleep(0.05)  # let the dispatched task run
        agent.handle_message.assert_called_once()
        call_args = agent.handle_message.call_args
        assert call_args[0][0] == SYSTEM_SENDER
        recorder.close()

    async def test_fire_records_status_event(self, tmp_path: Path) -> None:
        """fire() records a StatusEvent."""
        heartbeat, _, recorder, _ = _make_heartbeat(tmp_path)

        with patch.object(recorder, "record") as mock_record:
            await heartbeat.fire()

        # Check that a StatusEvent was recorded
        status_calls = [
            c
            for c in mock_record.call_args_list
            if isinstance(c[0][0], StatusEvent) and c[0][0].status == "heartbeat_sent"
        ]
        assert len(status_calls) == 1
        recorder.close()

    async def test_fire_uses_system_sender(self, tmp_path: Path) -> None:
        """Heartbeat messages come from __system__."""
        heartbeat, router, recorder, _ = _make_heartbeat(tmp_path)

        with patch.object(router, "send", new_callable=AsyncMock) as mock_send:
            await heartbeat.fire()

        mock_send.assert_called_once()
        assert mock_send.call_args[0][0] == "__system__"
        assert mock_send.call_args[0][1] == "agent-a"
        recorder.close()


class TestHeartbeatLoop:
    async def test_loop_fires_after_interval(self, tmp_path: Path) -> None:
        """The heartbeat loop fires after the configured interval."""
        heartbeat, router, recorder, shutdown = _make_heartbeat(
            tmp_path,
            interval=1,
        )
        agent = router._agents["agent-a"]

        await heartbeat.start()
        # Wait slightly longer than the interval
        await asyncio.sleep(1.5)

        shutdown.set()
        await heartbeat.stop()

        assert agent.handle_message.call_count >= 1
        recorder.close()

    async def test_loop_paused_skips_fire(self, tmp_path: Path) -> None:
        """The heartbeat loop skips firing when paused."""
        heartbeat, router, recorder, shutdown = _make_heartbeat(
            tmp_path,
            interval=1,
        )
        agent = router._agents["agent-a"]

        heartbeat.pause()
        await heartbeat.start()
        await asyncio.sleep(1.5)

        shutdown.set()
        await heartbeat.stop()

        agent.handle_message.assert_not_called()
        recorder.close()

    async def test_loop_resumes_after_unpause(self, tmp_path: Path) -> None:
        """The heartbeat loop fires after resuming from pause."""
        heartbeat, router, recorder, shutdown = _make_heartbeat(
            tmp_path,
            interval=1,
        )
        agent = router._agents["agent-a"]

        heartbeat.pause()
        await heartbeat.start()
        await asyncio.sleep(1.2)

        # Should not have fired yet
        agent.handle_message.assert_not_called()

        heartbeat.resume()
        await asyncio.sleep(1.5)

        shutdown.set()
        await heartbeat.stop()

        assert agent.handle_message.call_count >= 1
        recorder.close()

    async def test_stop_cancels_loop(self, tmp_path: Path) -> None:
        """stop() cancels the background task."""
        heartbeat, _, recorder, _ = _make_heartbeat(tmp_path, interval=5)
        await heartbeat.start()
        task = heartbeat._task
        assert task is not None

        await heartbeat.stop()
        assert task.done()
        assert heartbeat._task is None
        recorder.close()

    async def test_shutdown_event_stops_loop(self, tmp_path: Path) -> None:
        """Setting the shutdown event stops the heartbeat loop."""
        heartbeat, _, recorder, shutdown = _make_heartbeat(
            tmp_path,
            interval=1,
        )
        await heartbeat.start()
        shutdown.set()
        await asyncio.sleep(0.1)

        # Task should finish on its own
        await heartbeat.stop()
        recorder.close()


class TestHeartbeatResponseDetection:
    def test_done_marker_sets_flag(self, tmp_path: Path) -> None:
        """check_response detects [HEARTBEAT:DONE] marker."""
        heartbeat, _, recorder, _ = _make_heartbeat(tmp_path)
        assert not heartbeat.done_flag

        heartbeat.check_response("All tasks complete. [HEARTBEAT:DONE]")
        assert heartbeat.done_flag
        recorder.close()

    def test_stuck_marker_records_event(self, tmp_path: Path) -> None:
        """check_response detects [HEARTBEAT:STUCK] marker."""
        heartbeat, _, recorder, _ = _make_heartbeat(tmp_path)

        with patch.object(recorder, "record") as mock_record:
            heartbeat.check_response("Waiting for input. [HEARTBEAT:STUCK]")

        status_calls = [
            c
            for c in mock_record.call_args_list
            if isinstance(c[0][0], StatusEvent) and c[0][0].status == "heartbeat_stuck"
        ]
        assert len(status_calls) == 1
        assert not heartbeat.done_flag
        recorder.close()

    def test_no_marker_no_action(self, tmp_path: Path) -> None:
        """check_response does nothing without markers."""
        heartbeat, _, recorder, _ = _make_heartbeat(tmp_path)

        with patch.object(recorder, "record") as mock_record:
            heartbeat.check_response("Working on the analysis now.")

        mock_record.assert_not_called()
        assert not heartbeat.done_flag
        recorder.close()

    def test_done_marker_case_insensitive(self, tmp_path: Path) -> None:
        """[heartbeat:done] should be detected regardless of case."""
        heartbeat, _, recorder, _ = _make_heartbeat(tmp_path)
        heartbeat.check_response("[heartbeat:done] all finished")
        assert heartbeat.done_flag
        recorder.close()


# ================================================================== #
# Router __system__ bypass
# ================================================================== #


class TestSystemSenderBypass:
    async def test_system_bypasses_topology(self, tmp_path: Path) -> None:
        """__system__ sender should bypass topology checks."""
        recorder = _make_recorder(tmp_path)
        # Use a restrictive topology (pipeline with only one flow)
        router = Router(
            topology=TopologyConfig(type="pipeline", flow=["a", "b"]),
            recorder=recorder,
        )

        mock_a = MagicMock()
        mock_a.handle_message = AsyncMock()
        mock_b = MagicMock()
        mock_b.handle_message = AsyncMock()
        router.register("a", mock_a)
        router.register("b", mock_b)

        # __system__ should be able to send to "a" even though
        # it's not in the topology
        await router.send("__system__", "a", "heartbeat check")
        await asyncio.sleep(0.05)

        mock_a.handle_message.assert_called_once_with("__system__", "heartbeat check")
        recorder.close()


# ================================================================== #
# Heartbeat agent thread isolation
# ================================================================== #


class TestHeartbeatThreadIsolation:
    async def test_heartbeat_uses_system_thread(self, tmp_path: Path) -> None:
        """Heartbeat messages use the __system__ thread, not a peer thread."""
        router, recorder = _make_router(tmp_path)
        provider = MockProvider(
            [LLMResponse(content="Status update!", usage=TokenUsage(10, 5))]
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
        )
        router.register("agent-a", agent)

        # Send a user message first
        await agent.handle_message("user", "hello")
        assert len(agent._thread) == 2  # user msg + assistant reply

        # Now send a heartbeat message
        provider._responses = [
            LLMResponse(content="Working on it!", usage=TokenUsage(10, 5))
        ]
        provider._call_count = 0
        await agent.handle_message("__system__", "status?")

        # Single thread should contain all messages (user + system)
        assert len(agent._thread) == 4  # 2 from user + 2 from system
        recorder.close()


# ================================================================== #
# Integration with REPL
# ================================================================== #


class TestHeartbeatReplIntegration:
    async def test_status_command_fires_heartbeat(self, tmp_path: Path) -> None:
        """/status fires an immediate heartbeat when heartbeat is available."""
        heartbeat, _, recorder, _ = _make_heartbeat(tmp_path)

        with patch.object(heartbeat, "fire", new_callable=AsyncMock) as mock_fire:
            result = await _handle_command("/status", {}, heartbeat=heartbeat)

        assert result is False
        mock_fire.assert_called_once()
        recorder.close()

    async def test_status_without_heartbeat_prints_idle(
        self,
        tmp_path: Path,
        capsys: Any,
    ) -> None:
        """/status without heartbeat falls back to static message."""
        result = await _handle_command("/status", {}, heartbeat=None)
        assert result is False
        captured = capsys.readouterr()
        assert "Status: all agents idle" in captured.out

    async def test_done_flag_triggers_shutdown(self, tmp_path: Path) -> None:
        """When heartbeat.done_flag is set, the REPL exits."""
        heartbeat, router, recorder, shutdown = _make_heartbeat(tmp_path)
        heartbeat._done_flag = True

        agents = {"agent-a": MagicMock(spec=LLMAgent)}

        with patch("lattice.commands.up._read_input") as mock_input:
            await _repl_loop(router, "agent-a", agents, shutdown, heartbeat)
            # Should NOT have read any input since done_flag was already set
            mock_input.assert_not_called()

        assert shutdown.is_set()
        recorder.close()


# ================================================================== #
# Heartbeat hook installation
# ================================================================== #


class TestHeartbeatHook:
    async def test_install_hook_wraps_callback(self, tmp_path: Path) -> None:
        """_install_heartbeat_hook wraps the agent's on_response."""
        router, recorder = _make_router(tmp_path)
        provider = MockProvider([LLMResponse(content="hi", usage=TokenUsage(10, 5))])
        captured: list[str] = []

        async def _capture(content: str) -> None:
            captured.append(content)

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
            on_response=_capture,
        )

        heartbeat, _, _, _ = _make_heartbeat(tmp_path)
        _install_heartbeat_hook(agent, heartbeat)

        # Call the hooked callback (now async)
        await agent._on_response("All done! [HEARTBEAT:DONE]")  # type: ignore[misc]

        # Original callback should still fire
        assert captured == ["All done! [HEARTBEAT:DONE]"]
        # Heartbeat should detect done marker
        assert heartbeat.done_flag
        recorder.close()

    async def test_install_hook_without_original_callback(
        self,
        tmp_path: Path,
    ) -> None:
        """Hook works even if agent has no original callback."""
        router, recorder = _make_router(tmp_path)
        provider = MockProvider([LLMResponse(content="hi", usage=TokenUsage(10, 5))])
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
            on_response=None,
        )

        heartbeat, _, _, _ = _make_heartbeat(tmp_path)
        _install_heartbeat_hook(agent, heartbeat)

        # Should not crash (now async)
        await agent._on_response("[HEARTBEAT:STUCK]")  # type: ignore[misc]
        assert not heartbeat.done_flag
        recorder.close()
