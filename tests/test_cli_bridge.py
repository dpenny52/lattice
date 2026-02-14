"""Tests for the CLI Bridge agent."""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from lattice.agent.cli_bridge import CLIBridge
from lattice.config.models import AgentConfig, LatticeConfig, TopologyConfig
from lattice.router.router import Agent, Router
from lattice.session.models import (
    AgentDoneEvent,
    AgentStartEvent,
    ErrorEvent,
    StatusEvent,
)
from lattice.session.recorder import SessionRecorder

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_recorder(tmp_path: Path) -> SessionRecorder:
    return SessionRecorder("test-team", "abc123", sessions_dir=tmp_path / "sessions")


def _make_router(tmp_path: Path) -> tuple[Router, SessionRecorder]:
    recorder = _make_recorder(tmp_path)
    router = Router(topology=TopologyConfig(type="mesh"), recorder=recorder)
    return router, recorder


def _make_bridge(
    router: Router,
    recorder: SessionRecorder,
    name: str = "cli-agent",
    cli_type: str | None = None,
    command: str | None = None,
    on_response: Any = None,
) -> CLIBridge:
    return CLIBridge(
        name=name,
        role="You are a test CLI agent.",
        router=router,
        recorder=recorder,
        team_name="test-team",
        peer_names=["user", "agent-b"],
        cli_type=cli_type,
        command=command,
        on_response=on_response,
    )


class MockAsyncStdout:
    """Async-aware mock stdout that yields lines on demand.

    Lines can be added at any time via ``feed()``.  ``readline()``
    blocks until a line is available or ``close()`` is called.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[bytes] = asyncio.Queue()

    def feed(self, data: bytes) -> None:
        self._queue.put_nowait(data)

    def close(self) -> None:
        """Signal EOF."""
        self._queue.put_nowait(b"")

    async def readline(self) -> bytes:
        return await self._queue.get()


def _make_mock_process(
    stdout: MockAsyncStdout | None = None,
) -> MagicMock:
    """Create a mock subprocess with async-aware stdin/stdout."""
    proc = MagicMock()
    proc.returncode = None

    stdin = MagicMock()
    stdin.write = MagicMock()
    stdin.drain = AsyncMock()
    proc.stdin = stdin

    if stdout is None:
        stdout = MockAsyncStdout()
    proc.stdout = stdout
    proc.stderr = MagicMock()
    proc.wait = AsyncMock(return_value=0)
    proc.terminate = MagicMock()
    proc.kill = MagicMock()

    return proc


def _capture_events(recorder: SessionRecorder) -> list[Any]:
    """Monkey-patch recorder to capture all events."""
    events: list[Any] = []
    original = recorder.record

    def capture(event: Any) -> None:
        events.append(event)
        original(event)

    recorder.record = capture  # type: ignore[assignment]
    return events


# ------------------------------------------------------------------ #
# Protocol compliance
# ------------------------------------------------------------------ #


class TestCLIBridgeProtocol:
    """Verify CLIBridge satisfies the Agent protocol."""

    def test_satisfies_agent_protocol(self, tmp_path: Path) -> None:
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, cli_type="claude")
        assert isinstance(bridge, Agent)
        recorder.close()

    def test_has_handle_message(self, tmp_path: Path) -> None:
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, cli_type="claude")
        assert hasattr(bridge, "handle_message")
        assert inspect.iscoroutinefunction(bridge.handle_message)
        recorder.close()


# ------------------------------------------------------------------ #
# Custom CLI — long-running subprocess
# ------------------------------------------------------------------ #


class TestCustomCLI:
    """Tests for custom CLI subprocess communication."""

    async def test_send_task_to_subprocess(self, tmp_path: Path) -> None:
        """Task message is written to subprocess stdin as JSONL."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, command="cat")

        stdout = MockAsyncStdout()
        mock_proc = _make_mock_process(stdout)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.start()
            await asyncio.sleep(0.01)

            # Run handle_message in background -- it will block on the future.
            task = asyncio.create_task(
                bridge.handle_message("user", "do something")
            )
            await asyncio.sleep(0.01)

            # Now feed the response so the future resolves.
            stdout.feed(
                json.dumps({
                    "type": "result",
                    "task_id": "t_001",
                    "content": "done!",
                }).encode() + b"\n"
            )

            await asyncio.wait_for(task, timeout=2.0)

        # Verify task was written to stdin.
        calls = mock_proc.stdin.write.call_args_list
        assert len(calls) >= 1
        written = calls[0][0][0].decode()
        task_msg = json.loads(written)
        assert task_msg["type"] == "task"
        assert task_msg["id"] == "t_001"
        assert task_msg["from"] == "user"
        assert task_msg["content"] == "do something"

        stdout.close()
        recorder.close()

    async def test_subprocess_result_callback(self, tmp_path: Path) -> None:
        """on_response callback fires when subprocess returns a result."""
        router, recorder = _make_router(tmp_path)
        captured: list[str] = []
        bridge = _make_bridge(
            router, recorder,
            command="cat",
            on_response=lambda c: captured.append(c),
        )

        stdout = MockAsyncStdout()
        mock_proc = _make_mock_process(stdout)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.start()
            await asyncio.sleep(0.01)

            task = asyncio.create_task(
                bridge.handle_message("user", "work")
            )
            await asyncio.sleep(0.01)

            stdout.feed(
                json.dumps({
                    "type": "result",
                    "task_id": "t_001",
                    "content": "task complete",
                }).encode() + b"\n"
            )

            await asyncio.wait_for(task, timeout=2.0)

        assert captured == ["task complete"]
        stdout.close()
        recorder.close()


# ------------------------------------------------------------------ #
# Claude adapter
# ------------------------------------------------------------------ #


class TestClaudeAdapter:
    """Tests for the Claude CLI adapter mode."""

    async def test_claude_subprocess_invocation(self, tmp_path: Path) -> None:
        """Claude adapter runs claude -p with correct args."""
        router, recorder = _make_router(tmp_path)
        captured: list[str] = []
        bridge = _make_bridge(
            router, recorder,
            cli_type="claude",
            on_response=lambda c: captured.append(c),
        )

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        output = json.dumps({"result": "Analysis complete"})
        mock_proc.communicate = AsyncMock(return_value=(output.encode(), b""))

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            await bridge.handle_message("user", "analyze this")

        mock_exec.assert_called_once()
        call_args = mock_exec.call_args
        args = call_args[0]
        assert args[0] == "claude"
        assert "-p" in args
        assert "--output-format" in args
        assert "json" in args
        assert "--dangerously-skip-permissions" in args

        # Verify prompt includes role and message.
        prompt_idx = args.index("-p") + 1
        prompt = args[prompt_idx]
        assert "You are a test CLI agent." in prompt
        assert "Task from user: analyze this" in prompt

        assert captured == ["Analysis complete"]
        recorder.close()

    async def test_claude_not_found(self, tmp_path: Path) -> None:
        """Handles missing claude binary gracefully."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, cli_type="claude")
        events = _capture_events(recorder)

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("claude not found"),
        ):
            await bridge.handle_message("user", "do thing")

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert any("not found" in e.error for e in error_events)
        recorder.close()

    async def test_claude_nonzero_exit(self, tmp_path: Path) -> None:
        """Handles non-zero exit from claude CLI."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, cli_type="claude")
        events = _capture_events(recorder)

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"API error"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.handle_message("user", "do thing")

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert any("exited with code 1" in e.error for e in error_events)
        recorder.close()

    async def test_claude_start_is_noop(self, tmp_path: Path) -> None:
        """start() is a no-op for Claude adapter."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, cli_type="claude")

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            await bridge.start()
            mock_exec.assert_not_called()

        assert bridge._started is True
        recorder.close()


# ------------------------------------------------------------------ #
# Task tracking
# ------------------------------------------------------------------ #


class TestTaskTracking:
    """Verify task IDs are generated and tracked correctly."""

    async def test_task_ids_increment(self, tmp_path: Path) -> None:
        """Each task gets an incrementing ID."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, command="cat")

        stdout = MockAsyncStdout()
        mock_proc = _make_mock_process(stdout)
        written_messages: list[bytes] = []

        def capture_write(data: bytes) -> None:
            written_messages.append(data)

        mock_proc.stdin.write = capture_write

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.start()
            await asyncio.sleep(0.01)

            for i in range(1, 4):
                task = asyncio.create_task(
                    bridge.handle_message("user", f"task {i}")
                )
                await asyncio.sleep(0.01)

                tid = f"t_{i:03d}"
                payload = json.dumps({
                    "type": "result",
                    "task_id": tid,
                    "content": f"r{i}",
                })
                stdout.feed(payload.encode() + b"\n")
                await asyncio.wait_for(task, timeout=2.0)

        task_ids = []
        for msg_bytes in written_messages:
            msg = json.loads(msg_bytes.decode())
            if msg.get("type") == "task":
                task_ids.append(msg["id"])

        assert task_ids == ["t_001", "t_002", "t_003"]
        stdout.close()
        recorder.close()


# ------------------------------------------------------------------ #
# Read loop
# ------------------------------------------------------------------ #


class TestReadLoop:
    """Verify background read loop processes messages correctly."""

    async def test_read_loop_handles_non_json(self, tmp_path: Path) -> None:
        """Non-JSON output from subprocess is logged but doesn't crash."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, command="echo")

        stdout = MockAsyncStdout()
        mock_proc = _make_mock_process(stdout)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.start()
            await asyncio.sleep(0.01)

            stdout.feed(b"not json\n")
            stdout.feed(b"also not json\n")
            await asyncio.sleep(0.05)

        # No crash.
        stdout.close()
        recorder.close()

    async def test_read_loop_handles_empty_lines(self, tmp_path: Path) -> None:
        """Empty lines from subprocess are skipped."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, command="echo")

        stdout = MockAsyncStdout()
        mock_proc = _make_mock_process(stdout)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.start()
            await asyncio.sleep(0.01)

            stdout.feed(b"\n")
            stdout.feed(b"  \n")
            await asyncio.sleep(0.05)

        stdout.close()
        recorder.close()

    async def test_read_loop_unknown_type(self, tmp_path: Path) -> None:
        """Unknown message types are logged but don't crash."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, command="echo")

        stdout = MockAsyncStdout()
        mock_proc = _make_mock_process(stdout)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.start()
            await asyncio.sleep(0.01)

            stdout.feed(
                json.dumps({"type": "unknown_thing", "data": "xyz"}).encode() + b"\n"
            )
            await asyncio.sleep(0.05)

        stdout.close()
        recorder.close()


# ------------------------------------------------------------------ #
# Message routing from subprocess
# ------------------------------------------------------------------ #


class TestMessageRouting:
    """Verify outbound messages from CLI go through the router."""

    async def test_message_routed_to_peer(self, tmp_path: Path) -> None:
        """A 'message' from subprocess is dispatched via the router."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, command="echo", name="cli-agent")
        router.register("cli-agent", bridge)

        mock_peer = MagicMock()
        mock_peer.handle_message = AsyncMock()
        router.register("agent-b", mock_peer)

        stdout = MockAsyncStdout()
        mock_proc = _make_mock_process(stdout)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.start()
            await asyncio.sleep(0.01)

            task = asyncio.create_task(
                bridge.handle_message("user", "send message to agent-b")
            )
            await asyncio.sleep(0.01)

            # Subprocess sends a message to agent-b.
            stdout.feed(
                json.dumps({
                    "type": "message",
                    "to": "agent-b",
                    "content": "hello from cli",
                }).encode() + b"\n"
            )
            await asyncio.sleep(0.05)

            # Then return the result for the task.
            stdout.feed(
                json.dumps({
                    "type": "result",
                    "task_id": "t_001",
                    "content": "done",
                }).encode() + b"\n"
            )

            await asyncio.wait_for(task, timeout=2.0)

        await asyncio.sleep(0.1)
        mock_peer.handle_message.assert_called_once_with("cli-agent", "hello from cli")
        stdout.close()
        recorder.close()

    async def test_message_to_non_peer_blocked(self, tmp_path: Path) -> None:
        """A subprocess cannot message agents outside its peer list."""
        router, recorder = _make_router(tmp_path)
        # peer_names defaults to ["user", "agent-b"] via _make_bridge
        bridge = _make_bridge(router, recorder, command="echo", name="cli-agent")
        router.register("cli-agent", bridge)

        secret_agent = MagicMock()
        secret_agent.handle_message = AsyncMock()
        router.register("secret-agent", secret_agent)

        stdout = MockAsyncStdout()
        mock_proc = _make_mock_process(stdout)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.start()
            await asyncio.sleep(0.01)

            task = asyncio.create_task(
                bridge.handle_message("user", "try to reach secret-agent")
            )
            await asyncio.sleep(0.01)

            # Subprocess tries to message a non-peer.
            stdout.feed(
                json.dumps({
                    "type": "message",
                    "to": "secret-agent",
                    "content": "sneaky message",
                }).encode() + b"\n"
            )
            await asyncio.sleep(0.05)

            # Finish the task.
            stdout.feed(
                json.dumps({
                    "type": "result",
                    "task_id": "t_001",
                    "content": "done",
                }).encode() + b"\n"
            )
            await asyncio.wait_for(task, timeout=2.0)

        # The message to non-peer should NOT have been delivered.
        secret_agent.handle_message.assert_not_called()
        stdout.close()
        recorder.close()


# ------------------------------------------------------------------ #
# Status events
# ------------------------------------------------------------------ #


class TestStatusEvents:
    """Verify status messages from CLI are recorded."""

    async def test_status_recorded(self, tmp_path: Path) -> None:
        """Status messages from subprocess are recorded as StatusEvent."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, command="echo")
        events = _capture_events(recorder)

        stdout = MockAsyncStdout()
        mock_proc = _make_mock_process(stdout)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.start()
            await asyncio.sleep(0.01)

            task = asyncio.create_task(
                bridge.handle_message("user", "search")
            )
            await asyncio.sleep(0.01)

            stdout.feed(
                json.dumps({
                    "type": "status",
                    "status": "searching arxiv...",
                }).encode() + b"\n"
            )
            await asyncio.sleep(0.05)

            stdout.feed(
                json.dumps({
                    "type": "result",
                    "task_id": "t_001",
                    "content": "done",
                }).encode() + b"\n"
            )

            await asyncio.wait_for(task, timeout=2.0)

        status_events = [e for e in events if isinstance(e, StatusEvent)]
        assert any(e.status == "searching arxiv..." for e in status_events)
        stdout.close()
        recorder.close()


# ------------------------------------------------------------------ #
# Subprocess crash
# ------------------------------------------------------------------ #


class TestSubprocessCrash:
    """Verify graceful handling when subprocess dies."""

    async def test_crash_records_error(self, tmp_path: Path) -> None:
        """Subprocess crash records an ErrorEvent."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, command="false")
        events = _capture_events(recorder)

        stdout = MockAsyncStdout()
        mock_proc = _make_mock_process(stdout)
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.start()
            await asyncio.sleep(0.01)

            # Subprocess exits immediately.
            stdout.close()
            await asyncio.sleep(0.1)

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert any("exited with code 1" in e.error for e in error_events)
        recorder.close()

    async def test_crash_fails_pending_futures(self, tmp_path: Path) -> None:
        """Pending tasks get failed when subprocess crashes."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, command="false")

        stdout = MockAsyncStdout()
        mock_proc = _make_mock_process(stdout)
        mock_proc.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.start()
            await asyncio.sleep(0.01)

            task = asyncio.create_task(
                bridge.handle_message("user", "do work")
            )
            await asyncio.sleep(0.01)

            # Subprocess crashes (EOF).
            stdout.close()

            # handle_message should complete (with logged error) rather than hang.
            await asyncio.wait_for(task, timeout=2.0)

        recorder.close()


# ------------------------------------------------------------------ #
# Shutdown sequence
# ------------------------------------------------------------------ #


class TestShutdown:
    """Verify shutdown sequence."""

    async def test_graceful_shutdown(self, tmp_path: Path) -> None:
        """Shutdown sends shutdown message and waits."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, command="cat")

        stdout = MockAsyncStdout()
        mock_proc = _make_mock_process(stdout)

        async def graceful_wait() -> int:
            mock_proc.returncode = 0
            return 0

        mock_proc.wait = graceful_wait

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.start()
            await asyncio.sleep(0.01)

            # Close stdout so read_loop finishes.
            stdout.close()
            await asyncio.sleep(0.05)

            await bridge.shutdown()

        # Verify shutdown message was sent.
        calls = mock_proc.stdin.write.call_args_list
        assert len(calls) >= 1
        shutdown_written = calls[-1][0][0].decode()
        shutdown_msg = json.loads(shutdown_written)
        assert shutdown_msg["type"] == "shutdown"
        assert bridge._started is False
        recorder.close()

    async def test_shutdown_noop_when_not_started(self, tmp_path: Path) -> None:
        """Shutdown is a no-op when the bridge hasn't started."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, command="cat")
        await bridge.shutdown()
        assert bridge._started is False
        recorder.close()

    async def test_claude_shutdown_is_noop(self, tmp_path: Path) -> None:
        """Claude adapter shutdown doesn't try to kill a process."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, cli_type="claude")
        await bridge.start()
        await bridge.shutdown()
        assert bridge._started is False
        recorder.close()

    async def test_shutdown_escalates_to_sigterm(self, tmp_path: Path) -> None:
        """If process doesn't exit after shutdown message, SIGTERM is sent."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, command="cat")

        stdout = MockAsyncStdout()
        mock_proc = _make_mock_process(stdout)

        wait_count = 0

        async def slow_then_fast_wait() -> int:
            nonlocal wait_count
            wait_count += 1
            if wait_count == 1:
                await asyncio.sleep(100)
            mock_proc.returncode = 0
            return 0

        mock_proc.wait = slow_then_fast_wait

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch("lattice.agent.cli_bridge._SHUTDOWN_WAIT", 0.05),
            patch("lattice.agent.cli_bridge._SIGTERM_WAIT", 0.05),
        ):
            await bridge.start()
            await asyncio.sleep(0.01)
            stdout.close()
            await asyncio.sleep(0.05)
            await bridge.shutdown()

        mock_proc.terminate.assert_called_once()
        recorder.close()


# ------------------------------------------------------------------ #
# Event recording
# ------------------------------------------------------------------ #


class TestEventRecording:
    """Verify AgentStart/AgentDone events are recorded."""

    async def test_agent_start_and_done_events(self, tmp_path: Path) -> None:
        """handle_message records AgentStartEvent and AgentDoneEvent."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, cli_type="claude")
        events = _capture_events(recorder)

        mock_proc = MagicMock()
        mock_proc.returncode = 0
        output = json.dumps({"result": "ok"})
        mock_proc.communicate = AsyncMock(return_value=(output.encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.handle_message("user", "test")

        event_types = [type(e) for e in events]
        assert AgentStartEvent in event_types
        assert AgentDoneEvent in event_types

        start_events = [e for e in events if isinstance(e, AgentStartEvent)]
        assert start_events[0].agent == "cli-agent"
        assert start_events[0].agent_type == "cli"
        recorder.close()


# ------------------------------------------------------------------ #
# Integration with up.py
# ------------------------------------------------------------------ #


class TestIntegrationWithUp:
    """Verify CLI agents are created and started in _run_session."""

    async def test_cli_agents_created(self, tmp_path: Path) -> None:
        """CLI agents in config are created as CLIBridge instances."""
        from lattice.commands.up import _run_session

        config = LatticeConfig(
            version="0.1",
            team="test-team",
            agents={
                "coder": AgentConfig(
                    type="cli",
                    cli="claude",
                    role="You are a coder.",
                ),
            },
        )

        mock_bridge = MagicMock(spec=CLIBridge)
        mock_bridge.handle_message = AsyncMock()
        mock_bridge.start = AsyncMock()
        mock_bridge.shutdown = AsyncMock()

        with (
            patch(
                "lattice.commands.up.CLIBridge",
                return_value=mock_bridge,
            ) as mock_cls,
            patch("lattice.commands.up._repl_loop", new_callable=AsyncMock),
            patch("lattice.commands.up.SessionRecorder") as mock_recorder_cls,
            patch("lattice.commands.up.Router") as mock_router_cls,
        ):
            mock_recorder = MagicMock()
            mock_recorder.session_id = "abc123"
            mock_recorder.session_file = tmp_path / "test.jsonl"
            mock_recorder_cls.return_value = mock_recorder

            mock_router = MagicMock()
            mock_router._pending_tasks = set()
            mock_router_cls.return_value = mock_router

            await _run_session(config, verbose=False)

        mock_cls.assert_called_once()
        call_kwargs = mock_cls.call_args
        assert call_kwargs[1]["name"] == "coder"
        assert call_kwargs[1]["cli_type"] == "claude"

        mock_bridge.start.assert_called_once()
        mock_bridge.shutdown.assert_called_once()

    async def test_mixed_agents_counted_correctly(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        """Startup banner counts both LLM and CLI agents."""
        from lattice.commands.up import _run_session

        config = LatticeConfig(
            version="0.1",
            team="test-team",
            agents={
                "researcher": AgentConfig(
                    type="llm",
                    model="anthropic/claude-sonnet-4-5-20250929",
                    role="You research.",
                ),
                "coder": AgentConfig(
                    type="cli",
                    cli="claude",
                    role="You code.",
                ),
            },
        )

        mock_llm = MagicMock()
        mock_llm.handle_message = AsyncMock()

        mock_bridge = MagicMock(spec=CLIBridge)
        mock_bridge.handle_message = AsyncMock()
        mock_bridge.start = AsyncMock()
        mock_bridge.shutdown = AsyncMock()

        with (
            patch("lattice.commands.up.LLMAgent", return_value=mock_llm),
            patch("lattice.commands.up.CLIBridge", return_value=mock_bridge),
            patch("lattice.commands.up._repl_loop", new_callable=AsyncMock),
            patch("lattice.commands.up.SessionRecorder") as mock_recorder_cls,
            patch("lattice.commands.up.Router") as mock_router_cls,
        ):
            mock_recorder = MagicMock()
            mock_recorder.session_id = "abc123"
            mock_recorder.session_file = tmp_path / "test.jsonl"
            mock_recorder_cls.return_value = mock_recorder

            mock_router = MagicMock()
            mock_router._pending_tasks = set()
            mock_router_cls.return_value = mock_router

            await _run_session(config, verbose=False)

        captured = capsys.readouterr()
        assert "Agents: 2" in captured.out

    async def test_agents_command_shows_types(self, capsys: Any) -> None:
        """The /agents command shows (cli) for CLI agents."""
        from lattice.commands.up import _handle_command

        mock_llm = MagicMock(spec=object)
        mock_bridge = MagicMock(spec=CLIBridge)

        agents: dict[str, Any] = {
            "researcher": mock_llm,
            "coder": mock_bridge,
        }

        await _handle_command("/agents", agents)  # type: ignore[arg-type]

        captured = capsys.readouterr()
        assert "researcher (llm)" in captured.out
        assert "coder (cli)" in captured.out

    async def test_only_cli_agents_can_run(self, tmp_path: Path, capsys: Any) -> None:
        """A team with only CLI agents can still run."""
        from lattice.commands.up import _run_session

        config = LatticeConfig(
            version="0.1",
            team="test-team",
            agents={
                "coder": AgentConfig(
                    type="cli",
                    cli="claude",
                    role="You code.",
                ),
            },
        )

        mock_bridge = MagicMock(spec=CLIBridge)
        mock_bridge.handle_message = AsyncMock()
        mock_bridge.start = AsyncMock()
        mock_bridge.shutdown = AsyncMock()

        with (
            patch("lattice.commands.up.CLIBridge", return_value=mock_bridge),
            patch("lattice.commands.up._repl_loop", new_callable=AsyncMock),
            patch("lattice.commands.up.SessionRecorder") as mock_recorder_cls,
            patch("lattice.commands.up.Router") as mock_router_cls,
        ):
            mock_recorder = MagicMock()
            mock_recorder.session_id = "abc123"
            mock_recorder.session_file = tmp_path / "test.jsonl"
            mock_recorder_cls.return_value = mock_recorder

            mock_router = MagicMock()
            mock_router._pending_tasks = set()
            mock_router_cls.return_value = mock_router

            await _run_session(config, verbose=False)

        captured = capsys.readouterr()
        assert "No agents configured" not in captured.err
        assert "Agents: 1" in captured.out
