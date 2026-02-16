"""Tests for the Script Bridge agent."""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from lattice.agent.script_bridge import ScriptBridge
from lattice.config.models import AgentConfig, LatticeConfig, TopologyConfig
from lattice.router.router import Agent, Router
from lattice.session.models import AgentDoneEvent, AgentStartEvent, ErrorEvent
from lattice.session.recorder import SessionRecorder

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


def _make_script_bridge(
    router: Router,
    recorder: SessionRecorder,
    name: str = "script-agent",
    command: str = "cat",
    timeout: float = 30.0,
    on_response: Any = None,
) -> ScriptBridge:
    return ScriptBridge(
        name=name,
        role="Test script agent.",
        command=command,
        router=router,
        recorder=recorder,
        timeout=timeout,
        on_response=on_response,
    )


def _capture_events(recorder: SessionRecorder) -> list[Any]:
    """Monkey-patch recorder to capture all events."""
    events: list[Any] = []
    original = recorder.record

    def capture(event: Any) -> None:
        events.append(event)
        original(event)

    recorder.record = capture  # type: ignore[assignment]
    return events


def _make_mock_process(
    stdout: bytes = b"",
    stderr: bytes = b"",
    returncode: int = 0,
) -> MagicMock:
    """Create a mock subprocess for asyncio.create_subprocess_exec."""
    proc = MagicMock()
    proc.returncode = returncode
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.wait = AsyncMock(return_value=returncode)
    proc.kill = MagicMock()
    return proc


# ------------------------------------------------------------------ #
# Protocol compliance
# ------------------------------------------------------------------ #


class TestScriptBridgeProtocol:
    """Verify ScriptBridge satisfies the Agent protocol."""

    def test_satisfies_agent_protocol(self, tmp_path: Path) -> None:
        router, recorder = _make_router(tmp_path)
        bridge = _make_script_bridge(router, recorder)
        assert isinstance(bridge, Agent)
        recorder.close()

    def test_has_handle_message(self, tmp_path: Path) -> None:
        router, recorder = _make_router(tmp_path)
        bridge = _make_script_bridge(router, recorder)
        assert hasattr(bridge, "handle_message")
        assert inspect.iscoroutinefunction(bridge.handle_message)
        recorder.close()


# ------------------------------------------------------------------ #
# Basic execution
# ------------------------------------------------------------------ #


class TestBasicExecution:
    """Test basic stdin -> stdout subprocess execution."""

    async def test_sends_content_to_stdin_gets_stdout(self, tmp_path: Path) -> None:
        """Content is piped to stdin, stdout is returned as result."""
        router, recorder = _make_router(tmp_path)

        # Register a mock agent to receive the response.
        mock_sender = MagicMock()
        mock_sender.handle_message = AsyncMock()
        router.register("user", mock_sender)

        bridge = _make_script_bridge(router, recorder)
        router.register("script-agent", bridge)

        mock_proc = _make_mock_process(stdout=b"processed output\n")

        with patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            await bridge.handle_message("user", "input data")

        # Verify content was passed to communicate().
        mock_proc.communicate.assert_called_once_with(input=b"input data")
        recorder.close()

    async def test_stdout_stripped(self, tmp_path: Path) -> None:
        """Trailing whitespace/newlines in stdout are stripped."""
        router, recorder = _make_router(tmp_path)
        mock_sender = MagicMock()
        mock_sender.handle_message = AsyncMock()
        router.register("user", mock_sender)

        bridge = _make_script_bridge(router, recorder)
        router.register("script-agent", bridge)

        mock_proc = _make_mock_process(stdout=b"  result  \n\n")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.handle_message("user", "test")

        # The router.send should be called with stripped text.
        await asyncio.sleep(0.05)
        mock_sender.handle_message.assert_called_once_with(
            "script-agent",
            "result",
        )
        recorder.close()


# ------------------------------------------------------------------ #
# Result routing
# ------------------------------------------------------------------ #


class TestResultRouting:
    """Result is sent back to the sender via router.send()."""

    async def test_result_routed_back_to_sender(self, tmp_path: Path) -> None:
        """stdout is sent back to from_agent via the router."""
        router, recorder = _make_router(tmp_path)

        mock_sender = MagicMock()
        mock_sender.handle_message = AsyncMock()
        router.register("agent-a", mock_sender)

        bridge = _make_script_bridge(router, recorder)
        router.register("script-agent", bridge)

        mock_proc = _make_mock_process(stdout=b"hello back")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.handle_message("agent-a", "do work")

        await asyncio.sleep(0.05)
        mock_sender.handle_message.assert_called_once_with(
            "script-agent",
            "hello back",
        )
        recorder.close()


# ------------------------------------------------------------------ #
# on_response callback
# ------------------------------------------------------------------ #


class TestOnResponseCallback:
    """Verify on_response fires with stdout on success."""

    async def test_callback_fires_on_success(self, tmp_path: Path) -> None:
        router, recorder = _make_router(tmp_path)
        captured: list[str] = []

        mock_sender = MagicMock()
        mock_sender.handle_message = AsyncMock()
        router.register("user", mock_sender)

        bridge = _make_script_bridge(
            router,
            recorder,
            on_response=_async_capture(captured),
        )
        router.register("script-agent", bridge)

        mock_proc = _make_mock_process(stdout=b"callback result")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.handle_message("user", "test")

        assert captured == ["callback result"]
        recorder.close()

    async def test_callback_not_fired_on_empty_stdout(self, tmp_path: Path) -> None:
        router, recorder = _make_router(tmp_path)
        captured: list[str] = []

        bridge = _make_script_bridge(
            router,
            recorder,
            on_response=_async_capture(captured),
        )

        mock_proc = _make_mock_process(stdout=b"")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.handle_message("user", "test")

        assert captured == []
        recorder.close()

    async def test_callback_not_fired_on_error(self, tmp_path: Path) -> None:
        router, recorder = _make_router(tmp_path)
        captured: list[str] = []

        bridge = _make_script_bridge(
            router,
            recorder,
            on_response=_async_capture(captured),
        )

        mock_proc = _make_mock_process(returncode=1, stderr=b"fail")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.handle_message("user", "test")

        assert captured == []
        recorder.close()


# ------------------------------------------------------------------ #
# Non-zero exit code
# ------------------------------------------------------------------ #


class TestNonZeroExit:
    """Verify error handling for non-zero exit codes."""

    async def test_nonzero_exit_records_error(self, tmp_path: Path) -> None:
        router, recorder = _make_router(tmp_path)
        events = _capture_events(recorder)

        bridge = _make_script_bridge(router, recorder)

        mock_proc = _make_mock_process(
            returncode=1,
            stderr=b"something went wrong",
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.handle_message("user", "fail")

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert any("exited with code 1" in e.error for e in error_events)
        assert any("something went wrong" in e.error for e in error_events)
        recorder.close()

    async def test_stderr_truncated_to_500_chars(self, tmp_path: Path) -> None:
        router, recorder = _make_router(tmp_path)
        events = _capture_events(recorder)

        bridge = _make_script_bridge(router, recorder)

        long_stderr = b"A" * 1000
        mock_proc = _make_mock_process(returncode=1, stderr=long_stderr)

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.handle_message("user", "fail")

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) >= 1
        # The error message should contain at most 500 A's from stderr.
        error_msg = error_events[0].error
        a_count = error_msg.count("A")
        assert a_count == 500
        recorder.close()


# ------------------------------------------------------------------ #
# Timeout handling
# ------------------------------------------------------------------ #


class TestTimeout:
    """Verify timeout kills the process and records an error."""

    async def test_timeout_kills_process(self, tmp_path: Path) -> None:
        router, recorder = _make_router(tmp_path)
        events = _capture_events(recorder)

        bridge = _make_script_bridge(router, recorder, timeout=0.1)

        mock_proc = MagicMock()
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock(return_value=137)

        async def slow_communicate(input: bytes | None = None) -> tuple[bytes, bytes]:  # noqa: A002
            await asyncio.sleep(10)
            return b"", b""

        mock_proc.communicate = slow_communicate

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.handle_message("user", "slow task")

        mock_proc.kill.assert_called_once()

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert any("timed out" in e.error for e in error_events)
        recorder.close()


# ------------------------------------------------------------------ #
# Empty stdout
# ------------------------------------------------------------------ #


class TestEmptyStdout:
    """Verify empty stdout is handled gracefully."""

    async def test_empty_stdout_no_response_sent(self, tmp_path: Path) -> None:
        router, recorder = _make_router(tmp_path)
        events = _capture_events(recorder)

        bridge = _make_script_bridge(router, recorder)

        mock_proc = _make_mock_process(stdout=b"")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.handle_message("user", "test")

        # No router.send should be called — verify no MessageEvent for
        # script-agent -> user.
        from lattice.session.models import MessageEvent

        msg_events = [
            e
            for e in events
            if isinstance(e, MessageEvent) and e.from_agent == "script-agent"
        ]
        assert msg_events == []
        recorder.close()

    async def test_whitespace_only_stdout_treated_as_empty(
        self,
        tmp_path: Path,
    ) -> None:
        router, recorder = _make_router(tmp_path)

        bridge = _make_script_bridge(router, recorder)

        mock_proc = _make_mock_process(stdout=b"   \n\n  ")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.handle_message("user", "test")

        # Should complete without errors — AgentDoneEvent still recorded.
        recorder.close()


# ------------------------------------------------------------------ #
# Subprocess spawn failure
# ------------------------------------------------------------------ #


class TestSpawnFailure:
    """Verify graceful handling of subprocess spawn errors."""

    async def test_file_not_found(self, tmp_path: Path) -> None:
        router, recorder = _make_router(tmp_path)
        events = _capture_events(recorder)

        bridge = _make_script_bridge(router, recorder, command="nonexistent_cmd")

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("not found"),
        ):
            await bridge.handle_message("user", "test")

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert any("not found" in e.error.lower() for e in error_events)
        recorder.close()

    async def test_os_error(self, tmp_path: Path) -> None:
        router, recorder = _make_router(tmp_path)
        events = _capture_events(recorder)

        bridge = _make_script_bridge(router, recorder)

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=OSError("permission denied"),
        ):
            await bridge.handle_message("user", "test")

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert any("permission denied" in e.error.lower() for e in error_events)
        recorder.close()


# ------------------------------------------------------------------ #
# Event recording
# ------------------------------------------------------------------ #


class TestEventRecording:
    """Verify AgentStart and AgentDone events are recorded."""

    async def test_start_and_done_events_on_success(self, tmp_path: Path) -> None:
        router, recorder = _make_router(tmp_path)
        events = _capture_events(recorder)

        mock_sender = MagicMock()
        mock_sender.handle_message = AsyncMock()
        router.register("user", mock_sender)

        bridge = _make_script_bridge(router, recorder)
        router.register("script-agent", bridge)

        mock_proc = _make_mock_process(stdout=b"ok")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.handle_message("user", "test")

        event_types = [type(e) for e in events]
        assert AgentStartEvent in event_types
        assert AgentDoneEvent in event_types

        start_events = [e for e in events if isinstance(e, AgentStartEvent)]
        assert start_events[0].agent == "script-agent"
        assert start_events[0].agent_type == "script"

        done_events = [e for e in events if isinstance(e, AgentDoneEvent)]
        assert done_events[0].agent == "script-agent"
        assert done_events[0].reason == "completed"
        recorder.close()

    async def test_start_event_recorded_on_error(self, tmp_path: Path) -> None:
        """AgentStartEvent is recorded even when the script fails."""
        router, recorder = _make_router(tmp_path)
        events = _capture_events(recorder)

        bridge = _make_script_bridge(router, recorder)

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=FileNotFoundError("nope"),
        ):
            await bridge.handle_message("user", "test")

        start_events = [e for e in events if isinstance(e, AgentStartEvent)]
        assert len(start_events) == 1
        assert start_events[0].agent_type == "script"
        recorder.close()

    async def test_done_event_on_empty_stdout(self, tmp_path: Path) -> None:
        """AgentDoneEvent is still recorded when stdout is empty."""
        router, recorder = _make_router(tmp_path)
        events = _capture_events(recorder)

        bridge = _make_script_bridge(router, recorder)

        mock_proc = _make_mock_process(stdout=b"")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.handle_message("user", "test")

        done_events = [e for e in events if isinstance(e, AgentDoneEvent)]
        assert len(done_events) == 1
        recorder.close()


# ------------------------------------------------------------------ #
# Multiple sequential messages (stateless)
# ------------------------------------------------------------------ #


class TestStatelessExecution:
    """Each message spawns a fresh subprocess."""

    async def test_multiple_messages_spawn_separate_processes(
        self,
        tmp_path: Path,
    ) -> None:
        router, recorder = _make_router(tmp_path)

        mock_sender = MagicMock()
        mock_sender.handle_message = AsyncMock()
        router.register("user", mock_sender)

        bridge = _make_script_bridge(router, recorder)
        router.register("script-agent", bridge)

        mock_proc_1 = _make_mock_process(stdout=b"result-1")
        mock_proc_2 = _make_mock_process(stdout=b"result-2")
        mock_proc_3 = _make_mock_process(stdout=b"result-3")

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=[mock_proc_1, mock_proc_2, mock_proc_3],
        ) as mock_exec:
            await bridge.handle_message("user", "msg-1")
            await bridge.handle_message("user", "msg-2")
            await bridge.handle_message("user", "msg-3")

        # Three separate subprocess spawns.
        assert mock_exec.call_count == 3

        # Each got the right input.
        mock_proc_1.communicate.assert_called_once_with(input=b"msg-1")
        mock_proc_2.communicate.assert_called_once_with(input=b"msg-2")
        mock_proc_3.communicate.assert_called_once_with(input=b"msg-3")
        recorder.close()


# ------------------------------------------------------------------ #
# stdin content verification
# ------------------------------------------------------------------ #


class TestStdinContent:
    """Verify exact content is written to stdin."""

    async def test_multiline_content(self, tmp_path: Path) -> None:
        router, recorder = _make_router(tmp_path)

        mock_sender = MagicMock()
        mock_sender.handle_message = AsyncMock()
        router.register("user", mock_sender)

        bridge = _make_script_bridge(router, recorder)
        router.register("script-agent", bridge)

        content = "line 1\nline 2\nline 3"
        mock_proc = _make_mock_process(stdout=b"ok")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.handle_message("user", content)

        mock_proc.communicate.assert_called_once_with(
            input=b"line 1\nline 2\nline 3",
        )
        recorder.close()

    async def test_unicode_content(self, tmp_path: Path) -> None:
        router, recorder = _make_router(tmp_path)

        mock_sender = MagicMock()
        mock_sender.handle_message = AsyncMock()
        router.register("user", mock_sender)

        bridge = _make_script_bridge(router, recorder)
        router.register("script-agent", bridge)

        content = "hello unicode: \u00e9\u00e8\u00ea \u2603"
        mock_proc = _make_mock_process(stdout=b"ok")

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await bridge.handle_message("user", content)

        mock_proc.communicate.assert_called_once_with(
            input=content.encode(),
        )
        recorder.close()


# ------------------------------------------------------------------ #
# Integration with up.py
# ------------------------------------------------------------------ #


class TestIntegrationWithUp:
    """Verify script agents are created in _run_session."""

    async def test_script_agents_created(self, tmp_path: Path) -> None:
        """Script agents in config are created as ScriptBridge instances."""
        from lattice.commands.up import _run_session

        config = LatticeConfig(
            version="0.1",
            team="test-team",
            agents={
                "formatter": AgentConfig(
                    type="script",
                    command="python format.py",
                ),
            },
        )

        mock_script = MagicMock(spec=ScriptBridge)
        mock_script.handle_message = AsyncMock()

        with (
            patch(
                "lattice.commands.up.ScriptBridge",
                return_value=mock_script,
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
        assert call_kwargs[1]["name"] == "formatter"
        assert call_kwargs[1]["command"] == "python format.py"

    async def test_agents_command_shows_script_type(
        self,
        capsys: Any,
    ) -> None:
        """The /agents command shows (script) for script agents."""
        from lattice.commands.up import _handle_command

        mock_llm = MagicMock(spec=object)
        mock_script = MagicMock(spec=ScriptBridge)

        agents: dict[str, Any] = {
            "researcher": mock_llm,
            "formatter": mock_script,
        }

        await _handle_command("/agents", agents)  # type: ignore[arg-type]

        captured = capsys.readouterr()
        assert "researcher (llm)" in captured.out
        assert "formatter (script)" in captured.out

    async def test_mixed_agents_counted_correctly(
        self,
        tmp_path: Path,
        capsys: Any,
    ) -> None:
        """Startup banner counts LLM, CLI, and script agents."""
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
                "formatter": AgentConfig(
                    type="script",
                    command="jq .",
                ),
            },
        )

        mock_llm = MagicMock()
        mock_llm.handle_message = AsyncMock()

        mock_script = MagicMock(spec=ScriptBridge)
        mock_script.handle_message = AsyncMock()

        with (
            patch("lattice.commands.up.LLMAgent", return_value=mock_llm),
            patch("lattice.commands.up.ScriptBridge", return_value=mock_script),
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
