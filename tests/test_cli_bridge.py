"""Tests for the CLI Bridge agent."""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.agent.cli_bridge import CLIBridge
from lattice.config.models import AgentConfig, LatticeConfig, TopologyConfig
from lattice.router.router import Agent, Router
from lattice.session.models import (
    AgentDoneEvent,
    AgentStartEvent,
    CLIProgressEvent,
    CLITextChunkEvent,
    CLIThinkingEvent,
    CLIToolCallEvent,
    ErrorEvent,
    StatusEvent,
)
from lattice.session.recorder import SessionRecorder

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture(autouse=True)
def _mock_available_memory():
    """Ensure the pre-flight memory gate never blocks tests.

    Individual tests that need to exercise the gate (e.g. TestClaudeMemoryGate)
    apply their own, more specific patch which takes precedence.
    """
    with patch("lattice.memory_monitor.get_available_mb", return_value=8192.0):
        yield


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _async_capture(captured: list[str]) -> Any:
    """Return an async callback that appends to *captured*."""
    async def _cb(content: str) -> None:
        captured.append(content)
    return _cb


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
    proc.stderr.read = AsyncMock(return_value=b"")
    proc.wait = AsyncMock(return_value=0)
    proc.terminate = MagicMock()
    proc.kill = MagicMock()

    return proc


def _make_mock_claude_process(
    text_response: str = "",
    returncode: int = 0,
) -> tuple[MagicMock, MockAsyncStdout]:
    """Create a mock Claude CLI process that streams a text response.

    Returns (proc, stdout) so caller can feed additional events if needed.
    """
    stdout = MockAsyncStdout()

    proc = MagicMock()
    proc.returncode = None
    proc.stdout = stdout
    proc.stderr = MagicMock()
    proc.stderr.read = AsyncMock(return_value=b"")
    proc.wait = AsyncMock(return_value=returncode)

    # Pre-feed the text response if provided.
    if text_response:
        stdout.feed(json.dumps({
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": text_response}]},
        }).encode() + b"\n")
        # Claude CLI always emits a final "result" event with the aggregated text.
        stdout.feed(json.dumps({
            "type": "result",
            "result": text_response,
        }).encode() + b"\n")

    return proc, stdout


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

            # Extract the task_id that was written to stdin.
            calls = mock_proc.stdin.write.call_args_list
            assert len(calls) >= 1
            written = calls[0][0][0].decode()
            task_msg = json.loads(written)
            actual_task_id = task_msg["id"]

            # Now feed the response with the actual task ID so the future resolves.
            stdout.feed(
                json.dumps({
                    "type": "result",
                    "task_id": actual_task_id,
                    "content": "done!",
                }).encode() + b"\n"
            )

            await asyncio.wait_for(task, timeout=2.0)

        # Verify task was written correctly.
        assert task_msg["type"] == "task"
        assert actual_task_id.startswith("t_")  # UUIDs now
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
            on_response=_async_capture(captured),
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

            # Extract the task_id from stdin.
            calls = mock_proc.stdin.write.call_args_list
            assert len(calls) >= 1
            written = calls[0][0][0].decode()
            task_msg = json.loads(written)
            actual_task_id = task_msg["id"]

            stdout.feed(
                json.dumps({
                    "type": "result",
                    "task_id": actual_task_id,
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
            on_response=_async_capture(captured),
        )

        # Mock streaming stdout.
        stdout = MockAsyncStdout()

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdout = stdout
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"")
        mock_proc.wait = AsyncMock(return_value=0)

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ) as mock_exec:
            task = asyncio.create_task(bridge.handle_message("user", "analyze this"))
            await asyncio.sleep(0.01)

            # Feed streaming events (real Claude CLI format).
            stdout.feed(json.dumps({
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Analysis complete"}]},
            }).encode() + b"\n")
            # Claude CLI emits a final "result" event with the aggregated text.
            stdout.feed(json.dumps({
                "type": "result",
                "result": "Analysis complete",
            }).encode() + b"\n")
            stdout.close()

            await asyncio.wait_for(task, timeout=2.0)

        mock_exec.assert_called_once()
        call_args = mock_exec.call_args
        args = call_args[0]
        assert args[0] == "claude"
        assert "-p" in args
        assert "--output-format" in args
        assert "stream-json" in args
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

        stdout = MockAsyncStdout()

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdout = stdout
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"API error")
        mock_proc.wait = AsyncMock(return_value=1)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            task = asyncio.create_task(bridge.handle_message("user", "do thing"))
            await asyncio.sleep(0.01)
            stdout.close()
            await asyncio.wait_for(task, timeout=2.0)

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
        """Each task gets a unique ID (UUIDs)."""
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

                # Extract the actual task_id from the written message.
                msg = json.loads(written_messages[-1].decode())
                tid = msg["id"]
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

        # Verify we got 3 unique task IDs with the expected prefix.
        assert len(task_ids) == 3
        assert all(tid.startswith("t_") for tid in task_ids)
        assert len(set(task_ids)) == 3  # All unique
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

            # Extract the task_id from stdin.
            calls = mock_proc.stdin.write.call_args_list
            assert len(calls) >= 1
            written = calls[0][0][0].decode()
            task_msg = json.loads(written)
            actual_task_id = task_msg["id"]

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
                    "task_id": actual_task_id,
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

            # Extract the task_id from stdin.
            calls = mock_proc.stdin.write.call_args_list
            assert len(calls) >= 1
            written = calls[0][0][0].decode()
            task_msg = json.loads(written)
            actual_task_id = task_msg["id"]

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
                    "task_id": actual_task_id,
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

            # Extract the task_id from stdin.
            calls = mock_proc.stdin.write.call_args_list
            assert len(calls) >= 1
            written = calls[0][0][0].decode()
            task_msg = json.loads(written)
            actual_task_id = task_msg["id"]

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
                    "task_id": actual_task_id,
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

        stdout = MockAsyncStdout()

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdout = stdout
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"")
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            task = asyncio.create_task(bridge.handle_message("user", "test"))
            await asyncio.sleep(0.01)
            stdout.feed(json.dumps({
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "ok"}]},
            }).encode() + b"\n")
            stdout.close()
            await asyncio.wait_for(task, timeout=2.0)

        event_types = [type(e) for e in events]
        assert AgentStartEvent in event_types
        assert AgentDoneEvent in event_types

        start_events = [e for e in events if isinstance(e, AgentStartEvent)]
        assert start_events[0].agent == "cli-agent"
        assert start_events[0].agent_type == "cli"
        recorder.close()


# ------------------------------------------------------------------ #
# Message queuing (Story 3.3)
# ------------------------------------------------------------------ #


class TestMessageQueuing:
    """Verify message queuing and conversation continuity for Claude adapter."""

    async def test_message_queued_when_busy(self, tmp_path: Path) -> None:
        """Messages queue when Claude agent is busy."""
        router, recorder = _make_router(tmp_path)
        captured: list[str] = []
        bridge = _make_bridge(
            router, recorder,
            cli_type="claude",
            on_response=_async_capture(captured),
        )

        first_task_started = asyncio.Event()

        # First subprocess blocks on stdout until we feed data and close it.
        stdout_1 = MockAsyncStdout()
        mock_proc_1 = MagicMock()
        mock_proc_1.returncode = None
        mock_proc_1.stdout = stdout_1
        mock_proc_1.stderr = MagicMock()
        mock_proc_1.stderr.read = AsyncMock(return_value=b"")
        mock_proc_1.wait = AsyncMock(return_value=0)

        # Second subprocess responds normally.
        mock_proc_2, stdout_2 = _make_mock_claude_process(text_response="second response")

        call_count = 0

        async def mock_create_subprocess(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                first_task_started.set()
                return mock_proc_1
            return mock_proc_2

        with patch("asyncio.create_subprocess_exec", side_effect=mock_create_subprocess):
            # Start first message.
            task_1 = asyncio.create_task(
                bridge.handle_message("user", "first task")
            )

            # Wait for the first subprocess to spawn.  By the time this event
            # fires, _claude_busy is already True and task_1 is blocked on
            # stdout readline — no sleep needed.
            await asyncio.wait_for(first_task_started.wait(), timeout=2.0)

            # Second message should queue (agent is busy) and return immediately.
            task_2 = asyncio.create_task(
                bridge.handle_message("user", "second task")
            )
            await asyncio.wait_for(task_2, timeout=2.0)

            assert "[cli-agent is busy, message queued]" in captured

            # Now let first task complete by feeding data and closing stdout.
            # Also pre-close stdout_2 since _process_message_queue will
            # process the queued message inline before task_1 returns.
            stdout_1.feed(json.dumps({
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "first response"}]},
            }).encode() + b"\n")
            stdout_1.feed(json.dumps({
                "type": "result",
                "result": "first response",
            }).encode() + b"\n")
            stdout_1.close()
            stdout_2.close()

            # task_1 processes the queue inline before returning.
            await asyncio.wait_for(task_1, timeout=2.0)

        assert "first response" in captured
        assert "second response" in captured
        recorder.close()

    async def test_queued_messages_delivered_in_order(self, tmp_path: Path) -> None:
        """Multiple queued messages are delivered in FIFO order."""
        router, recorder = _make_router(tmp_path)
        captured: list[str] = []
        bridge = _make_bridge(
            router, recorder,
            cli_type="claude",
            on_response=_async_capture(captured),
        )

        call_count = 0

        async def mock_create_subprocess(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            proc, stdout = _make_mock_claude_process(text_response=f"response {call_count}")
            # Close stdout immediately so readline() hits EOF after the pre-fed line.
            stdout.close()
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_create_subprocess):
            # Send 4 messages rapidly.
            tasks = [
                asyncio.create_task(bridge.handle_message("user", f"task {i}"))
                for i in range(1, 5)
            ]

            # Wait for all to complete.
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)

        # Responses should appear in order.
        response_texts = [c for c in captured if c.startswith("response")]
        assert response_texts == ["response 1", "response 2", "response 3", "response 4"]
        recorder.close()

    async def test_continue_flag_used_for_followups(self, tmp_path: Path) -> None:
        """Queued messages use --continue flag to preserve conversation context."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, cli_type="claude")

        first_task_started = asyncio.Event()

        # First subprocess with conversation_id — blocks on stdout read.
        stdout_1 = MockAsyncStdout()
        mock_proc_1 = MagicMock()
        mock_proc_1.returncode = None
        mock_proc_1.stdout = stdout_1
        mock_proc_1.stderr = MagicMock()
        mock_proc_1.stderr.read = AsyncMock(return_value=b"")
        mock_proc_1.wait = AsyncMock(return_value=0)

        # Second subprocess (follow-up).
        mock_proc_2, stdout_2 = _make_mock_claude_process(text_response="second")

        call_count = 0
        recorded_args: list[tuple[Any, ...]] = []

        async def mock_create_subprocess(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            recorded_args.append(args)
            if call_count == 1:
                first_task_started.set()
                return mock_proc_1
            return mock_proc_2

        with patch("asyncio.create_subprocess_exec", side_effect=mock_create_subprocess):
            # First message.
            task_1 = asyncio.create_task(bridge.handle_message("user", "first"))
            await asyncio.wait_for(first_task_started.wait(), timeout=2.0)

            # Second message (queues — agent is busy).
            task_2 = asyncio.create_task(bridge.handle_message("user", "second"))
            await asyncio.wait_for(task_2, timeout=2.0)

            # Let first task complete with session_id (from system init).
            # Pre-close stdout_2 since _process_message_queue runs inline.
            stdout_1.feed(json.dumps({
                "type": "system", "subtype": "init", "session_id": "conv-abc123",
            }).encode() + b"\n")
            stdout_1.feed(json.dumps({
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "first"}]},
            }).encode() + b"\n")
            stdout_1.close()
            stdout_2.close()

            await asyncio.wait_for(task_1, timeout=2.0)

        # Verify first call doesn't use --continue.
        first_call_args = recorded_args[0]
        assert "--continue" not in first_call_args

        # Verify second call uses --continue with conversation_id.
        second_call_args = recorded_args[1]
        assert "--continue" in second_call_args
        continue_idx = second_call_args.index("--continue")
        assert second_call_args[continue_idx + 1] == "conv-abc123"
        recorder.close()

    async def test_immediate_dispatch_when_idle(self, tmp_path: Path) -> None:
        """Messages sent to idle Claude agent dispatch immediately (no queue)."""
        router, recorder = _make_router(tmp_path)
        captured: list[str] = []
        bridge = _make_bridge(
            router, recorder,
            cli_type="claude",
            on_response=_async_capture(captured),
        )

        mock_proc, stdout = _make_mock_claude_process(text_response="immediate response")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            task = asyncio.create_task(bridge.handle_message("user", "task"))
            await asyncio.sleep(0.01)
            stdout.close()
            await asyncio.wait_for(task, timeout=2.0)

        # No queue message should appear.
        assert "[cli-agent is busy, message queued]" not in captured
        assert "immediate response" in captured
        recorder.close()

    async def test_queue_feedback_shows_agent_busy(self, tmp_path: Path) -> None:
        """Console feedback shows when agent is busy and message is queued."""
        router, recorder = _make_router(tmp_path)
        captured: list[str] = []
        bridge = _make_bridge(
            router, recorder,
            cli_type="claude",
            on_response=_async_capture(captured),
        )

        first_task_started = asyncio.Event()

        # First subprocess blocks on stdout read.
        stdout_1 = MockAsyncStdout()
        mock_proc_1 = MagicMock()
        mock_proc_1.returncode = None
        mock_proc_1.stdout = stdout_1
        mock_proc_1.stderr = MagicMock()
        mock_proc_1.stderr.read = AsyncMock(return_value=b"")
        mock_proc_1.wait = AsyncMock(return_value=0)

        # Second subprocess responds immediately.
        mock_proc_2, stdout_2 = _make_mock_claude_process(text_response="r2")

        call_count = 0

        async def mock_create_subprocess(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                first_task_started.set()
                return mock_proc_1
            return mock_proc_2

        with patch("asyncio.create_subprocess_exec", side_effect=mock_create_subprocess):
            task_1 = asyncio.create_task(bridge.handle_message("user", "t1"))
            await asyncio.wait_for(first_task_started.wait(), timeout=2.0)

            task_2 = asyncio.create_task(bridge.handle_message("user", "t2"))
            await asyncio.wait_for(task_2, timeout=2.0)

            # Verify busy feedback before first task completes.
            assert "[cli-agent is busy, message queued]" in captured

            # Pre-close stdout_2 since _process_message_queue runs inline.
            stdout_1.feed(json.dumps({
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "r1"}]},
            }).encode() + b"\n")
            stdout_1.close()
            stdout_2.close()

            await asyncio.wait_for(task_1, timeout=2.0)

        recorder.close()

    async def test_error_unblocks_queue(self, tmp_path: Path) -> None:
        """If Claude task fails, the agent is unblocked and can process queue."""
        router, recorder = _make_router(tmp_path)
        captured: list[str] = []
        bridge = _make_bridge(
            router, recorder,
            cli_type="claude",
            on_response=_async_capture(captured),
        )

        # First task fails.
        mock_proc_1, stdout_1 = _make_mock_claude_process(text_response="", returncode=1)
        mock_proc_1.stderr.read = AsyncMock(return_value=b"error")

        # Second task succeeds.
        mock_proc_2, stdout_2 = _make_mock_claude_process(text_response="success")

        call_count = 0

        async def mock_create_subprocess(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_proc_1
            return mock_proc_2

        with patch("asyncio.create_subprocess_exec", side_effect=mock_create_subprocess):
            task_1 = asyncio.create_task(bridge.handle_message("user", "fail"))
            await asyncio.sleep(0.01)
            stdout_1.close()

            task_2 = asyncio.create_task(bridge.handle_message("user", "succeed"))
            await asyncio.sleep(0.01)
            stdout_2.close()

            await asyncio.gather(task_1, task_2)

        # Second task should have processed despite first task failure.
        assert "success" in captured
        recorder.close()

    async def test_queue_persists_across_multiple_tasks(self, tmp_path: Path) -> None:
        """Queue can accumulate multiple messages and deliver all in order."""
        router, recorder = _make_router(tmp_path)
        captured: list[str] = []
        bridge = _make_bridge(
            router, recorder,
            cli_type="claude",
            on_response=_async_capture(captured),
        )

        num_tasks = 10
        call_count = 0

        async def mock_create_subprocess(*args: Any, **kwargs: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            proc, stdout = _make_mock_claude_process(text_response=f"task-{call_count}")
            # Close immediately so readline() hits EOF after the pre-fed line.
            stdout.close()
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=mock_create_subprocess):
            tasks = [
                asyncio.create_task(bridge.handle_message("user", f"msg-{i+1}"))
                for i in range(num_tasks)
            ]

            await asyncio.wait_for(asyncio.gather(*tasks), timeout=5.0)

        # All tasks should have been processed in order.
        task_responses = [c for c in captured if c.startswith("task-")]
        assert len(task_responses) == num_tasks
        for i in range(num_tasks):
            assert task_responses[i] == f"task-{i+1}"
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


# ------------------------------------------------------------------ #
# Claude streaming (Story 5.2)
# ------------------------------------------------------------------ #


class TestClaudeStreaming:
    """Tests for streaming Claude CLI output."""

    async def test_stream_text_chunks(self, tmp_path: Path) -> None:
        """Text chunks from Claude are recorded as CLITextChunkEvent."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, cli_type="claude")
        events = _capture_events(recorder)

        # Mock streaming stdout.
        stdout = MockAsyncStdout()

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdout = stdout
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"")
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            task = asyncio.create_task(bridge.handle_message("user", "test"))
            await asyncio.sleep(0.01)

            # Feed streaming text events (real Claude CLI format).
            stdout.feed(json.dumps({
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "Hello "}]},
            }).encode() + b"\n")
            stdout.feed(json.dumps({
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "world!"}]},
            }).encode() + b"\n")
            stdout.close()

            await asyncio.wait_for(task, timeout=2.0)

        text_events = [e for e in events if isinstance(e, CLITextChunkEvent)]
        assert len(text_events) == 2
        assert text_events[0].text == "Hello "
        assert text_events[1].text == "world!"
        recorder.close()

    async def test_stream_tool_calls(self, tmp_path: Path) -> None:
        """Tool use events from Claude are recorded as CLIToolCallEvent."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, cli_type="claude")
        events = _capture_events(recorder)

        stdout = MockAsyncStdout()

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdout = stdout
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"")
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            task = asyncio.create_task(bridge.handle_message("user", "use a tool"))
            await asyncio.sleep(0.01)

            # Feed tool use event (real Claude CLI format).
            stdout.feed(
                json.dumps({
                    "type": "assistant",
                    "message": {"content": [{
                        "type": "tool_use",
                        "name": "file-read",
                        "input": {"path": "/tmp/test.txt"},
                    }]},
                }).encode() + b"\n"
            )
            stdout.close()

            await asyncio.wait_for(task, timeout=2.0)

        tool_events = [e for e in events if isinstance(e, CLIToolCallEvent)]
        assert len(tool_events) == 1
        assert tool_events[0].tool == "file-read"
        assert tool_events[0].args == {"path": "/tmp/test.txt"}
        recorder.close()

    async def test_stream_thinking(self, tmp_path: Path) -> None:
        """Thinking events from Claude are recorded as CLIThinkingEvent."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, cli_type="claude")
        events = _capture_events(recorder)

        stdout = MockAsyncStdout()

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdout = stdout
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"")
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            task = asyncio.create_task(bridge.handle_message("user", "think"))
            await asyncio.sleep(0.01)

            # Feed thinking event (real Claude CLI format).
            stdout.feed(
                json.dumps({
                    "type": "assistant",
                    "message": {"content": [{
                        "type": "thinking",
                        "thinking": "I need to analyze the problem first...",
                    }]},
                }).encode() + b"\n"
            )
            stdout.close()

            await asyncio.wait_for(task, timeout=2.0)

        thinking_events = [e for e in events if isinstance(e, CLIThinkingEvent)]
        assert len(thinking_events) == 1
        assert thinking_events[0].content == "I need to analyze the problem first..."
        recorder.close()

    async def test_stream_progress(self, tmp_path: Path) -> None:
        """Status events from Claude are recorded as CLIProgressEvent."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, cli_type="claude")
        events = _capture_events(recorder)

        stdout = MockAsyncStdout()

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdout = stdout
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"")
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            task = asyncio.create_task(bridge.handle_message("user", "work"))
            await asyncio.sleep(0.01)

            # Feed system init event (real Claude CLI format — status comes from system init).
            stdout.feed(
                json.dumps({"type": "system", "subtype": "init", "session_id": "test-123"}).encode() + b"\n"
            )
            stdout.close()

            await asyncio.wait_for(task, timeout=2.0)

        progress_events = [e for e in events if isinstance(e, CLIProgressEvent)]
        assert len(progress_events) == 1
        assert progress_events[0].status == "initialized"
        recorder.close()

    async def test_stream_malformed_json(self, tmp_path: Path) -> None:
        """Malformed JSON in stream doesn't crash the agent."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, cli_type="claude")
        events = _capture_events(recorder)

        stdout = MockAsyncStdout()

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdout = stdout
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"")
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            task = asyncio.create_task(bridge.handle_message("user", "test"))
            await asyncio.sleep(0.01)

            # Feed malformed JSON.
            stdout.feed(b"not json at all\n")
            # Then valid JSON (real Claude CLI format).
            stdout.feed(json.dumps({
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "recovered"}]},
            }).encode() + b"\n")
            stdout.close()

            await asyncio.wait_for(task, timeout=2.0)

        # Should have recovered and processed the valid event.
        text_events = [e for e in events if isinstance(e, CLITextChunkEvent)]
        assert len(text_events) == 1
        assert text_events[0].text == "recovered"
        recorder.close()

    async def test_stream_conversation_id(self, tmp_path: Path) -> None:
        """Conversation ID from stream is stored for --continue."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, cli_type="claude")

        stdout = MockAsyncStdout()

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdout = stdout
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"")
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            task = asyncio.create_task(bridge.handle_message("user", "first"))
            await asyncio.sleep(0.01)

            # System init event carries the session_id (used for --continue).
            stdout.feed(
                json.dumps({"type": "system", "subtype": "init", "session_id": "conv-xyz789"}).encode() + b"\n"
            )
            stdout.close()

            await asyncio.wait_for(task, timeout=2.0)

        assert bridge._claude_conversation_id == "conv-xyz789"
        recorder.close()

    async def test_stream_events_bracketed_by_agent_start_done(self, tmp_path: Path) -> None:
        """Streaming events are bracketed by AgentStartEvent and AgentDoneEvent."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, cli_type="claude")
        events = _capture_events(recorder)

        stdout = MockAsyncStdout()

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdout = stdout
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"")
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            task = asyncio.create_task(bridge.handle_message("user", "test"))
            await asyncio.sleep(0.01)

            stdout.feed(json.dumps({
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "test"}]},
            }).encode() + b"\n")
            stdout.close()

            await asyncio.wait_for(task, timeout=2.0)

        # Find the bracketing events.
        start_idx = next(i for i, e in enumerate(events) if isinstance(e, AgentStartEvent))
        done_idx = next(i for i, e in enumerate(events) if isinstance(e, AgentDoneEvent))
        text_idx = next(i for i, e in enumerate(events) if isinstance(e, CLITextChunkEvent))

        # Text event should be between start and done.
        assert start_idx < text_idx < done_idx
        recorder.close()

    async def test_stream_uses_stream_json_format(self, tmp_path: Path) -> None:
        """Claude adapter uses --output-format stream-json."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, cli_type="claude")

        stdout = MockAsyncStdout()

        mock_proc = MagicMock()
        mock_proc.returncode = None
        mock_proc.stdout = stdout
        mock_proc.stderr = MagicMock()
        mock_proc.stderr.read = AsyncMock(return_value=b"")
        mock_proc.wait = AsyncMock(return_value=0)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            task = asyncio.create_task(bridge.handle_message("user", "test"))
            await asyncio.sleep(0.01)
            stdout.close()
            await asyncio.wait_for(task, timeout=2.0)

        # Verify command includes --output-format stream-json.
        call_args = mock_exec.call_args[0]
        assert "--output-format" in call_args
        format_idx = call_args.index("--output-format")
        assert call_args[format_idx + 1] == "stream-json"
        recorder.close()

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


# ------------------------------------------------------------------ #
# Pre-flight memory gate
# ------------------------------------------------------------------ #


class TestClaudeMemoryGate:
    """Tests for the pre-flight memory check in _handle_claude_task."""

    async def test_low_memory_blocks_spawn(self, tmp_path: Path) -> None:
        """Agent should refuse to spawn and record an error when memory is low."""
        router, recorder = _make_router(tmp_path)
        captured: list[str] = []
        bridge = _make_bridge(router, recorder, cli_type="claude", on_response=_async_capture(captured))
        bridge._started = True
        events = _capture_events(recorder)

        with patch("lattice.memory_monitor.get_available_mb", return_value=500.0):
            await bridge._handle_claude_task("user", "do stuff")

        # Should NOT have spawned a subprocess.
        assert bridge._current_claude_pid is None
        assert not bridge._claude_busy

        # Should have recorded an ErrorEvent.
        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) == 1
        assert "Insufficient memory" in error_events[0].error

        # Should have notified via on_response.
        assert any("insufficient memory" in msg for msg in captured)

    async def test_none_memory_allows_spawn(self, tmp_path: Path) -> None:
        """When get_available_mb returns None (unsupported), spawn should proceed."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, cli_type="claude")
        bridge._started = True

        proc, stdout = _make_mock_claude_process("done")
        stdout.close()

        with (
            patch("lattice.memory_monitor.get_available_mb", return_value=None),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc),
        ):
            await bridge._handle_claude_task("user", "do stuff")

        # Should have attempted spawn (proc.wait was called).
        proc.wait.assert_awaited()

    async def test_sufficient_memory_allows_spawn(self, tmp_path: Path) -> None:
        """When memory is above the threshold, spawn should proceed normally."""
        router, recorder = _make_router(tmp_path)
        bridge = _make_bridge(router, recorder, cli_type="claude")
        bridge._started = True

        proc, stdout = _make_mock_claude_process("all good")
        stdout.close()

        with (
            patch("lattice.memory_monitor.get_available_mb", return_value=2048.0),
            patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=proc),
        ):
            await bridge._handle_claude_task("user", "do stuff")

        proc.wait.assert_awaited()
