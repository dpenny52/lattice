"""Tests for graceful shutdown — ShutdownManager, pidfile, ``lattice down``."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from click.testing import CliRunner

from lattice.commands.down import down
from lattice.commands.up import _repl_loop
from lattice.config.models import TopologyConfig
from lattice.pidfile import (
    PIDFILE_NAME,
    is_process_running,
    read_pidfile,
    remove_pidfile,
    write_pidfile,
)
from lattice.router.router import Router
from lattice.session.recorder import SessionRecorder
from lattice.shutdown import ShutdownManager, _format_duration

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_recorder(tmp_path: Path) -> SessionRecorder:
    return SessionRecorder("test-team", "abc123", sessions_dir=tmp_path / "sessions")


def _make_router(tmp_path: Path) -> tuple[Router, SessionRecorder]:
    recorder = _make_recorder(tmp_path)
    router = Router(topology=TopologyConfig(type="mesh"), recorder=recorder)
    return router, recorder


def _make_shutdown_manager(
    tmp_path: Path,
    *,
    pending_tasks: set[asyncio.Task[None]] | None = None,
    cli_bridges: dict[str, Any] | None = None,
    all_agents: dict[str, Any] | None = None,
    heartbeat: Any = None,
) -> tuple[ShutdownManager, Router, SessionRecorder]:
    router, recorder = _make_router(tmp_path)
    if pending_tasks is not None:
        router._pending_tasks = pending_tasks
    shutdown_event = asyncio.Event()
    mgr = ShutdownManager(
        router=router,
        recorder=recorder,
        heartbeat=heartbeat,
        cli_bridges=cli_bridges or {},
        all_agents=all_agents or {},
        shutdown_event=shutdown_event,
    )
    return mgr, router, recorder


# ================================================================== #
# _format_duration
# ================================================================== #


class TestFormatDuration:
    def test_seconds_under_60(self) -> None:
        assert _format_duration(34.2) == "34.2s"

    def test_seconds_zero(self) -> None:
        assert _format_duration(0.0) == "0.0s"

    def test_seconds_at_60(self) -> None:
        assert _format_duration(60.0) == "1m 00s"

    def test_seconds_over_60(self) -> None:
        assert _format_duration(82.0) == "1m 22s"

    def test_seconds_large(self) -> None:
        assert _format_duration(3661.0) == "61m 01s"

    def test_fractional_under_60(self) -> None:
        assert _format_duration(5.7) == "5.7s"


# ================================================================== #
# ShutdownManager — Step 1: SIGNAL
# ================================================================== #


class TestShutdownSignal:
    async def test_signal_sets_shutdown_event(self, tmp_path: Path) -> None:
        mgr, _, recorder = _make_shutdown_manager(tmp_path)
        assert not mgr._shutdown_event.is_set()
        await mgr._signal()
        assert mgr._shutdown_event.is_set()
        recorder.close()

    async def test_signal_stops_heartbeat(self, tmp_path: Path) -> None:
        heartbeat = MagicMock()
        heartbeat.stop = AsyncMock()
        mgr, _, recorder = _make_shutdown_manager(tmp_path, heartbeat=heartbeat)

        await mgr._signal()

        heartbeat.stop.assert_awaited_once()
        recorder.close()

    async def test_signal_no_heartbeat(self, tmp_path: Path) -> None:
        """_signal() doesn't crash when heartbeat is None."""
        mgr, _, recorder = _make_shutdown_manager(tmp_path, heartbeat=None)
        await mgr._signal()
        assert mgr._shutdown_event.is_set()
        recorder.close()


# ================================================================== #
# ShutdownManager — Step 2: DRAIN
# ================================================================== #


class TestShutdownDrain:
    async def test_drain_no_pending_returns_true(self, tmp_path: Path) -> None:
        mgr, _, recorder = _make_shutdown_manager(tmp_path)
        result = await mgr._drain()
        assert result is True
        recorder.close()

    async def test_drain_waits_for_completed_tasks(self, tmp_path: Path) -> None:
        """drain returns True when all pending tasks complete in time."""
        completed = asyncio.create_task(asyncio.sleep(0.01))
        mgr, _, recorder = _make_shutdown_manager(
            tmp_path, pending_tasks={completed},
        )
        result = await mgr._drain()
        assert result is True
        recorder.close()

    async def test_drain_timeout_returns_false(self, tmp_path: Path) -> None:
        """drain returns False when tasks don't finish within the timeout."""
        # Create a task that sleeps forever
        never_done = asyncio.create_task(asyncio.sleep(9999))
        mgr, _, recorder = _make_shutdown_manager(
            tmp_path, pending_tasks={never_done},
        )
        # Use a very short timeout to keep the test fast
        mgr.DRAIN_TIMEOUT = 0.05

        result = await mgr._drain()
        assert result is False

        # Cleanup
        never_done.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await never_done
        recorder.close()


# ================================================================== #
# ShutdownManager — Step 3: KILL
# ================================================================== #


class TestShutdownKill:
    async def test_kill_cancels_remaining_tasks(self, tmp_path: Path) -> None:
        task = asyncio.create_task(asyncio.sleep(9999))
        mgr, _, recorder = _make_shutdown_manager(
            tmp_path, pending_tasks={task},
        )

        await mgr._kill()

        assert task.cancelled()
        recorder.close()

    async def test_kill_shuts_down_cli_bridges(self, tmp_path: Path) -> None:
        bridge = MagicMock()
        bridge.shutdown = AsyncMock()
        mgr, _, recorder = _make_shutdown_manager(
            tmp_path,
            cli_bridges={"my-bridge": bridge},
        )

        await mgr._kill()

        bridge.shutdown.assert_awaited_once()
        recorder.close()

    async def test_kill_handles_bridge_error(self, tmp_path: Path) -> None:
        """_kill() logs but doesn't crash on bridge shutdown errors."""
        bridge = MagicMock()
        bridge.shutdown = AsyncMock(side_effect=RuntimeError("boom"))
        mgr, _, recorder = _make_shutdown_manager(
            tmp_path,
            cli_bridges={"bad-bridge": bridge},
        )

        # Should not raise
        await mgr._kill()
        recorder.close()

    async def test_kill_no_tasks_no_bridges(self, tmp_path: Path) -> None:
        """_kill() is a no-op when nothing is running."""
        mgr, _, recorder = _make_shutdown_manager(tmp_path)
        await mgr._kill()
        recorder.close()


# ================================================================== #
# ShutdownManager — Step 4: CLOSE
# ================================================================== #


class TestShutdownClose:
    async def test_close_shuts_down_all_agents(self, tmp_path: Path) -> None:
        agent_a = MagicMock()
        agent_a.shutdown = AsyncMock()
        agent_b = MagicMock()
        agent_b.shutdown = AsyncMock()
        mgr, _, recorder = _make_shutdown_manager(
            tmp_path,
            all_agents={"a": agent_a, "b": agent_b},
        )

        await mgr._close("user_shutdown")

        agent_a.shutdown.assert_awaited_once()
        agent_b.shutdown.assert_awaited_once()

    async def test_close_calls_recorder_end(self, tmp_path: Path) -> None:
        mgr, _, recorder = _make_shutdown_manager(tmp_path)

        with patch.object(recorder, "end") as mock_end:
            await mgr._close("ctrl_c")

        mock_end.assert_called_once_with("ctrl_c")

    async def test_close_prints_summary(
        self, tmp_path: Path, capsys: Any,
    ) -> None:
        agent = MagicMock()
        agent.shutdown = AsyncMock()
        mgr, _, recorder = _make_shutdown_manager(
            tmp_path, all_agents={"agent-a": agent},
        )

        await mgr._close("complete")

        captured = capsys.readouterr()
        assert "Session ended (complete)" in captured.out
        assert "1 agent(s)" in captured.out
        assert "events" in captured.out

    async def test_close_handles_agent_shutdown_error(
        self, tmp_path: Path,
    ) -> None:
        """_close() logs but doesn't crash on agent shutdown errors."""
        bad_agent = MagicMock()
        bad_agent.shutdown = AsyncMock(side_effect=RuntimeError("oops"))
        mgr, _, recorder = _make_shutdown_manager(
            tmp_path, all_agents={"bad": bad_agent},
        )

        # Should not raise
        await mgr._close("error")


# ================================================================== #
# ShutdownManager — Full execute()
# ================================================================== #


class TestShutdownExecute:
    async def test_execute_runs_all_steps(self, tmp_path: Path) -> None:
        """execute() runs signal -> drain -> close (skip kill when drain succeeds)."""
        mgr, _, recorder = _make_shutdown_manager(tmp_path)

        call_order: list[str] = []

        async def sig() -> None:
            call_order.append("signal")

        async def drain() -> bool:
            call_order.append("drain")
            return True

        async def kill() -> None:
            call_order.append("kill")

        async def close(reason: str) -> None:
            call_order.append("close")

        mgr._signal = sig  # type: ignore[assignment]
        mgr._drain = drain  # type: ignore[assignment]
        mgr._kill = kill  # type: ignore[assignment]
        mgr._close = close  # type: ignore[assignment]

        await mgr.execute("user_shutdown")

        assert call_order == ["signal", "drain", "close"]
        recorder.close()

    async def test_execute_kills_on_drain_timeout(
        self, tmp_path: Path,
    ) -> None:
        """execute() calls _kill() when _drain() returns False."""
        mgr, _, recorder = _make_shutdown_manager(tmp_path)

        call_order: list[str] = []

        async def sig() -> None:
            call_order.append("signal")

        async def drain() -> bool:
            call_order.append("drain")
            return False

        async def kill() -> None:
            call_order.append("kill")

        async def close(reason: str) -> None:
            call_order.append("close")

        mgr._signal = sig  # type: ignore[assignment]
        mgr._drain = drain  # type: ignore[assignment]
        mgr._kill = kill  # type: ignore[assignment]
        mgr._close = close  # type: ignore[assignment]

        await mgr.execute("ctrl_c")

        assert call_order == ["signal", "drain", "kill", "close"]
        recorder.close()

    async def test_execute_idempotent(self, tmp_path: Path) -> None:
        """Calling execute() twice doesn't crash."""
        agent = MagicMock()
        agent.shutdown = AsyncMock()
        mgr, _, recorder = _make_shutdown_manager(
            tmp_path, all_agents={"a": agent},
        )

        await mgr.execute("user_shutdown")
        # Second call -- recorder is already closed, should not crash
        await mgr.execute("user_shutdown")


# ================================================================== #
# Pidfile
# ================================================================== #


class TestPidfile:
    def test_write_creates_file(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr("lattice.pidfile.PIDFILE_DIR", tmp_path)
        path = write_pidfile("sess-123", "my-team")
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["pid"] == os.getpid()
        assert data["session_id"] == "sess-123"
        assert data["team"] == "my-team"

    def test_read_returns_data(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr("lattice.pidfile.PIDFILE_DIR", tmp_path)
        write_pidfile("sess-456", "team-x")
        data = read_pidfile()
        assert data is not None
        assert data["session_id"] == "sess-456"
        assert data["team"] == "team-x"

    def test_read_returns_none_when_missing(
        self, tmp_path: Path, monkeypatch: Any,
    ) -> None:
        monkeypatch.setattr("lattice.pidfile.PIDFILE_DIR", tmp_path)
        assert read_pidfile() is None

    def test_remove_deletes_file(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr("lattice.pidfile.PIDFILE_DIR", tmp_path)
        write_pidfile("sess-789", "team-y")
        pidfile = tmp_path / PIDFILE_NAME
        assert pidfile.exists()
        remove_pidfile()
        assert not pidfile.exists()

    def test_remove_noop_when_missing(
        self, tmp_path: Path, monkeypatch: Any,
    ) -> None:
        monkeypatch.setattr("lattice.pidfile.PIDFILE_DIR", tmp_path)
        # Should not raise
        remove_pidfile()

    def test_is_process_running_current_pid(self) -> None:
        assert is_process_running(os.getpid()) is True

    def test_is_process_running_nonexistent(self) -> None:
        # PID 99999999 is very unlikely to exist
        assert is_process_running(99999999) is False

    def test_read_returns_none_on_invalid_json(
        self, tmp_path: Path, monkeypatch: Any,
    ) -> None:
        monkeypatch.setattr("lattice.pidfile.PIDFILE_DIR", tmp_path)
        pidfile = tmp_path / PIDFILE_NAME
        pidfile.write_text("not valid json!!!")
        assert read_pidfile() is None

    def test_write_creates_directory(self, tmp_path: Path, monkeypatch: Any) -> None:
        nested = tmp_path / "deeply" / "nested"
        monkeypatch.setattr("lattice.pidfile.PIDFILE_DIR", nested)
        path = write_pidfile("sess-abc", "team-z")
        assert path.exists()
        assert nested.is_dir()


# ================================================================== #
# ``lattice down`` command
# ================================================================== #


class TestDownCommand:
    def test_no_pidfile(self) -> None:
        runner = CliRunner()
        with patch("lattice.commands.down.read_pidfile", return_value=None):
            result = runner.invoke(down)
        assert result.exit_code == 1
        assert "No running session" in result.output

    def test_invalid_pidfile_format(self) -> None:
        runner = CliRunner()
        with (
            patch(
                "lattice.commands.down.read_pidfile",
                return_value={"pid": "not-an-int", "session_id": "abc"},
            ),
            patch("lattice.commands.down.remove_pidfile") as mock_remove,
        ):
            result = runner.invoke(down)
        assert result.exit_code == 1
        assert "Invalid pidfile" in result.output
        mock_remove.assert_called_once()

    def test_invalid_pid_out_of_range(self) -> None:
        """PIDs <= 1 or > PID_MAX should be rejected."""
        runner = CliRunner()
        with (
            patch(
                "lattice.commands.down.read_pidfile",
                return_value={"pid": 0, "session_id": "abc"},
            ),
            patch("lattice.commands.down.remove_pidfile"),
        ):
            result = runner.invoke(down)
        assert result.exit_code == 1
        assert "Invalid pidfile" in result.output

    def test_stale_pidfile(self) -> None:
        runner = CliRunner()
        with (
            patch(
                "lattice.commands.down.read_pidfile",
                return_value={"pid": 12345, "session_id": "stale", "team": "t"},
            ),
            patch("lattice.commands.down.is_process_running", return_value=False),
            patch("lattice.commands.down.remove_pidfile") as mock_remove,
        ):
            result = runner.invoke(down)
        assert result.exit_code == 0
        assert "no longer running" in result.output
        mock_remove.assert_called_once()

    def test_running_process_sigterm(self) -> None:
        """Sends SIGTERM and waits for process to exit."""
        runner = CliRunner()
        with (
            patch(
                "lattice.commands.down.read_pidfile",
                return_value={"pid": 12345, "session_id": "live", "team": "t"},
            ),
            patch(
                "lattice.commands.down.is_process_running",
                side_effect=[True, False],
            ),
            patch("lattice.commands.down.remove_pidfile"),
            patch("lattice.commands.down.os.kill") as mock_kill,
            patch("lattice.commands.down.time.sleep"),
        ):
            result = runner.invoke(down)
        assert result.exit_code == 0
        assert "Shutting down session live" in result.output
        import signal as sig
        mock_kill.assert_called_once_with(12345, sig.SIGTERM)

    def test_permission_error(self) -> None:
        runner = CliRunner()
        with (
            patch(
                "lattice.commands.down.read_pidfile",
                return_value={"pid": 12345, "session_id": "s", "team": "t"},
            ),
            patch("lattice.commands.down.is_process_running", return_value=True),
            patch("lattice.commands.down.os.kill", side_effect=PermissionError),
        ):
            result = runner.invoke(down)
        assert result.exit_code == 1
        assert "Permission denied" in result.output

    def test_sigkill_escalation(self) -> None:
        """Process doesn't exit after SIGTERM — escalates to SIGKILL."""
        runner = CliRunner()
        # is_process_running returns True for all 100 poll checks + initial
        with (
            patch(
                "lattice.commands.down.read_pidfile",
                return_value={"pid": 12345, "session_id": "s", "team": "t"},
            ),
            patch(
                "lattice.commands.down.is_process_running",
                side_effect=[True] + [True] * 100,
            ),
            patch("lattice.commands.down.remove_pidfile"),
            patch("lattice.commands.down.os.kill") as mock_kill,
            patch("lattice.commands.down.time.sleep"),
        ):
            result = runner.invoke(down)
        assert result.exit_code == 0
        assert "SIGKILL" in result.output
        # Should have sent both SIGTERM and SIGKILL
        import signal as sig
        calls = mock_kill.call_args_list
        assert calls[0] == ((12345, sig.SIGTERM),)
        assert calls[1] == ((12345, sig.SIGKILL),)


# ================================================================== #
# Integration: shutdown writes proper JSONL
# ================================================================== #


class TestShutdownIntegration:
    async def test_session_jsonl_integrity(self, tmp_path: Path) -> None:
        """Full integration: router + recorder + shutdown produce valid JSONL."""
        recorder = SessionRecorder(
            "test-team", "hash123", sessions_dir=tmp_path / "sessions",
        )
        router = Router(topology=TopologyConfig(type="mesh"), recorder=recorder)

        # Register a mock agent
        mock_agent = MagicMock()
        mock_agent.handle_message = AsyncMock()
        mock_agent.shutdown = AsyncMock()
        router.register("agent-a", mock_agent)

        # Send a message through the router
        await router.send("user", "agent-a", "hello world")
        await asyncio.sleep(0.05)  # let dispatch run

        # Create and run shutdown
        shutdown_event = asyncio.Event()
        mgr = ShutdownManager(
            router=router,
            recorder=recorder,
            heartbeat=None,
            cli_bridges={},
            all_agents={"agent-a": mock_agent},
            shutdown_event=shutdown_event,
        )
        await mgr.execute("user_shutdown")

        # Read the JSONL file
        lines = recorder.session_file.read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]

        # Verify session_start
        assert events[0]["type"] == "session_start"
        assert events[0]["session_id"] == recorder.session_id

        # Verify session_end
        end_event = events[-1]
        assert end_event["type"] == "session_end"
        assert end_event["reason"] == "user_shutdown"
        assert end_event["duration_ms"] > 0

        # Verify monotonic sequence
        seqs = [e["seq"] for e in events]
        assert seqs == list(range(len(events)))

        # Verify there's a message event in between
        message_events = [e for e in events if e["type"] == "message"]
        assert len(message_events) >= 1
        assert message_events[0]["to"] == "agent-a"
        assert message_events[0]["content"] == "hello world"


# ================================================================== #
# Shutdown reason tracking
# ================================================================== #


class TestShutdownReasons:
    async def test_done_returns_user_shutdown(self, tmp_path: Path) -> None:
        """/done causes _repl_loop to return 'user_shutdown'."""
        router, recorder = _make_router(tmp_path)
        router.register("entry", MagicMock())
        agents: dict[str, Any] = {"entry": MagicMock()}
        shutdown = asyncio.Event()

        inputs = iter(["/done"])
        with patch("lattice.commands.up._read_input", side_effect=inputs):
            reason = await _repl_loop(router, "entry", agents, shutdown)

        assert reason == "user_shutdown"
        recorder.close()

    async def test_heartbeat_done_returns_complete(self, tmp_path: Path) -> None:
        """When heartbeat.done_flag is set, _repl_loop returns 'complete'."""
        from lattice.heartbeat import Heartbeat

        router, recorder = _make_router(tmp_path)
        mock_agent = MagicMock()
        mock_agent.handle_message = AsyncMock()
        router.register("entry", mock_agent)

        shutdown = asyncio.Event()
        heartbeat = Heartbeat(
            interval=0,
            router=router,
            entry_agent="entry",
            recorder=recorder,
            shutdown_event=shutdown,
        )
        heartbeat._done_flag = True

        agents: dict[str, Any] = {"entry": MagicMock()}

        with patch("lattice.commands.up._read_input") as mock_input:
            reason = await _repl_loop(router, "entry", agents, shutdown, heartbeat)
            mock_input.assert_not_called()

        assert reason == "complete"
        recorder.close()

    async def test_shutdown_event_returns_ctrl_c(self, tmp_path: Path) -> None:
        """When shutdown_event is set externally, _repl_loop returns 'ctrl_c'."""
        router, recorder = _make_router(tmp_path)
        router.register("entry", MagicMock())
        agents: dict[str, Any] = {"entry": MagicMock()}
        shutdown = asyncio.Event()
        shutdown.set()

        with patch("lattice.commands.up._read_input") as mock_input:
            reason = await _repl_loop(router, "entry", agents, shutdown)
            mock_input.assert_not_called()

        assert reason == "ctrl_c"
        recorder.close()
