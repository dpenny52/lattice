"""Tests for per-agent memory profiling."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import TypeAdapter

from lattice.agent.memory_profile import (
    AgentMemoryLogger,
    AgentMemoryProfiler,
    estimate_thread_size_kb,
)
from lattice.memory_monitor import get_pid_rss_mb
from lattice.session.models import MemorySnapshotEvent, SessionEvent
from lattice.session.recorder import SessionRecorder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EVENT_ADAPTER: TypeAdapter[SessionEvent] = TypeAdapter(SessionEvent)


def _base(seq: int = 0) -> dict[str, Any]:
    return {"ts": "2026-02-15T12:00:00.000Z", "seq": seq}


def _make_mock_llm_agent(thread: list[dict[str, Any]] | None = None) -> MagicMock:
    agent = MagicMock()
    agent._thread = thread or []
    return agent


def _make_mock_cli_bridge(
    subprocess_pid: int | None = None,
    queue_depth: int = 0,
) -> MagicMock:
    agent = MagicMock()
    agent.current_subprocess_pid = subprocess_pid
    queue = asyncio.Queue(maxsize=100)
    for _ in range(queue_depth):
        queue.put_nowait(("sender", "msg"))
    agent._message_queue = queue
    return agent


def _make_mock_script_bridge(last_peak_rss_mb: float | None = None) -> MagicMock:
    agent = MagicMock()
    agent._last_peak_rss_mb = last_peak_rss_mb
    return agent


# ===================================================================
# MemorySnapshotEvent model tests
# ===================================================================


class TestMemorySnapshotEvent:
    def test_round_trip(self) -> None:
        evt = MemorySnapshotEvent(
            **_base(),
            agent="lead",
            agent_type="llm",
            process_rss_mb=189.5,
            thread_messages=47,
            thread_size_kb=312.8,
            system_available_mb=3456.2,
        )
        data = json.loads(evt.model_dump_json(by_alias=True))
        assert data["type"] == "memory_snapshot"
        assert data["agent"] == "lead"
        assert data["agent_type"] == "llm"
        assert data["process_rss_mb"] == 189.5
        assert data["thread_messages"] == 47
        assert data["thread_size_kb"] == 312.8
        assert data["system_available_mb"] == 3456.2

    def test_discriminated_union(self) -> None:
        raw = {
            **_base(),
            "type": "memory_snapshot",
            "agent": "dev",
            "agent_type": "cli",
        }
        evt = _EVENT_ADAPTER.validate_python(raw)
        assert isinstance(evt, MemorySnapshotEvent)

    def test_optional_fields_default_none(self) -> None:
        evt = MemorySnapshotEvent(
            **_base(),
            agent="worker",
            agent_type="script",
        )
        assert evt.subprocess_rss_mb is None
        assert evt.subprocess_pid is None
        assert evt.thread_messages is None
        assert evt.queue_depth is None


# ===================================================================
# estimate_thread_size_kb tests
# ===================================================================


class TestEstimateThreadSizeKb:
    def test_empty_thread(self) -> None:
        assert estimate_thread_size_kb([]) == pytest.approx(len("[]") / 1024, abs=0.01)

    def test_known_data(self) -> None:
        thread = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there! How can I help you?"},
        ]
        expected_kb = len(json.dumps(thread)) / 1024
        assert estimate_thread_size_kb(thread) == pytest.approx(expected_kb, abs=0.01)

    def test_scales_with_content(self) -> None:
        small = [{"role": "user", "content": "x"}]
        large = [{"role": "user", "content": "x" * 10000}]
        assert estimate_thread_size_kb(large) > estimate_thread_size_kb(small) * 5


# ===================================================================
# get_pid_rss_mb tests
# ===================================================================


class TestGetPidRssMb:
    def test_current_process(self) -> None:
        """Should return a positive value for our own PID."""
        rss = get_pid_rss_mb(os.getpid())
        # Should work on macOS and Linux.
        assert rss is not None
        assert rss > 0

    def test_nonexistent_pid(self) -> None:
        """Should return None for a PID that doesn't exist."""
        rss = get_pid_rss_mb(99999999)
        assert rss is None


# ===================================================================
# AgentMemoryLogger tests
# ===================================================================


class TestAgentMemoryLogger:
    def test_creates_file(self, tmp_path: Path) -> None:
        log_path = tmp_path / "test.memory.jsonl"
        logger = AgentMemoryLogger(log_path)
        assert log_path.exists()
        logger.close()

    def test_writes_valid_jsonl(self, tmp_path: Path) -> None:
        log_path = tmp_path / "test.memory.jsonl"
        agent_logger = AgentMemoryLogger(log_path)
        agent_logger.write({"agent": "lead", "type": "llm", "process_rss_mb": 100.5})
        agent_logger.write({"agent": "lead", "type": "llm", "process_rss_mb": 105.2})
        agent_logger.close()

        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "ts" in data
            assert data["ts"].endswith("Z")
            assert data["agent"] == "lead"

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        log_path = tmp_path / "test.memory.jsonl"
        agent_logger = AgentMemoryLogger(log_path)
        agent_logger.close()
        agent_logger.close()  # should not raise

    def test_write_after_close_is_safe(self, tmp_path: Path) -> None:
        log_path = tmp_path / "test.memory.jsonl"
        agent_logger = AgentMemoryLogger(log_path)
        agent_logger.close()
        # Should not raise.
        agent_logger.write({"agent": "x"})
        assert log_path.read_text().strip() == ""


# ===================================================================
# AgentMemoryProfiler tests
# ===================================================================


class TestAgentMemoryProfiler:
    def test_register_creates_sidecar(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        shutdown = asyncio.Event()
        profiler = AgentMemoryProfiler(
            recorder=rec,
            shutdown_event=shutdown,
            sessions_dir=tmp_path,
            session_base_name="test_session",
        )
        agent = _make_mock_llm_agent()
        profiler.register("lead", "llm", agent)

        expected = tmp_path / "test_session.lead.memory.jsonl"
        assert expected.exists()
        rec.close()

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        shutdown = asyncio.Event()
        profiler = AgentMemoryProfiler(
            recorder=rec,
            shutdown_event=shutdown,
            sessions_dir=tmp_path,
            session_base_name="test_session",
            interval=1,
        )
        agent = _make_mock_llm_agent()
        profiler.register("lead", "llm", agent)

        await profiler.start()
        assert profiler._task is not None
        assert not profiler._task.done()

        await profiler.stop()
        assert profiler._task is None
        rec.close()

    @pytest.mark.asyncio
    async def test_stop_without_start(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        shutdown = asyncio.Event()
        profiler = AgentMemoryProfiler(
            recorder=rec,
            shutdown_event=shutdown,
            sessions_dir=tmp_path,
            session_base_name="test_session",
        )
        # Should not raise.
        await profiler.stop()
        rec.close()

    @pytest.mark.asyncio
    async def test_no_agents_skips_start(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        shutdown = asyncio.Event()
        profiler = AgentMemoryProfiler(
            recorder=rec,
            shutdown_event=shutdown,
            sessions_dir=tmp_path,
            session_base_name="test_session",
        )
        await profiler.start()
        assert profiler._task is None
        rec.close()

    def test_snapshot_llm_agent(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        shutdown = asyncio.Event()
        profiler = AgentMemoryProfiler(
            recorder=rec,
            shutdown_event=shutdown,
            sessions_dir=tmp_path,
            session_base_name="test",
        )
        thread = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        agent = _make_mock_llm_agent(thread)
        profiler.register("lead", "llm", agent)

        # Trigger a manual snapshot.
        profiler._snapshot_all()

        # Check per-agent sidecar file.
        sidecar = tmp_path / "test.lead.memory.jsonl"
        lines = sidecar.read_text().strip().splitlines()
        assert len(lines) == 1

        data = json.loads(lines[0])
        assert data["agent"] == "lead"
        assert data["agent_type"] == "llm"
        assert data["thread_messages"] == 2
        assert data["thread_size_kb"] > 0
        rec.close()

    def test_snapshot_cli_agent_no_subprocess(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        shutdown = asyncio.Event()
        profiler = AgentMemoryProfiler(
            recorder=rec,
            shutdown_event=shutdown,
            sessions_dir=tmp_path,
            session_base_name="test",
        )
        agent = _make_mock_cli_bridge(subprocess_pid=None, queue_depth=3)
        profiler.register("dev", "cli", agent)

        profiler._snapshot_all()

        sidecar = tmp_path / "test.dev.memory.jsonl"
        data = json.loads(sidecar.read_text().strip())
        assert data["agent"] == "dev"
        assert data["agent_type"] == "cli"
        assert data["queue_depth"] == 3
        assert data.get("subprocess_pid") is None
        rec.close()

    def test_snapshot_cli_agent_with_subprocess(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        shutdown = asyncio.Event()
        profiler = AgentMemoryProfiler(
            recorder=rec,
            shutdown_event=shutdown,
            sessions_dir=tmp_path,
            session_base_name="test",
        )
        # Use our own PID so get_pid_rss_mb returns something.
        agent = _make_mock_cli_bridge(subprocess_pid=os.getpid(), queue_depth=0)
        profiler.register("dev", "cli", agent)

        profiler._snapshot_all()

        sidecar = tmp_path / "test.dev.memory.jsonl"
        data = json.loads(sidecar.read_text().strip())
        assert data["subprocess_pid"] == os.getpid()
        assert data["subprocess_rss_mb"] is not None
        assert data["subprocess_rss_mb"] > 0
        rec.close()

    def test_snapshot_script_agent(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        shutdown = asyncio.Event()
        profiler = AgentMemoryProfiler(
            recorder=rec,
            shutdown_event=shutdown,
            sessions_dir=tmp_path,
            session_base_name="test",
        )
        agent = _make_mock_script_bridge(last_peak_rss_mb=42.5)
        profiler.register("tester", "script", agent)

        profiler._snapshot_all()

        sidecar = tmp_path / "test.tester.memory.jsonl"
        data = json.loads(sidecar.read_text().strip())
        assert data["agent"] == "tester"
        assert data["agent_type"] == "script"
        assert data["subprocess_rss_mb"] == 42.5
        rec.close()

    def test_snapshot_script_clears_peak_rss(self, tmp_path: Path) -> None:
        """After reading _last_peak_rss_mb the profiler should reset it to None."""
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        shutdown = asyncio.Event()
        profiler = AgentMemoryProfiler(
            recorder=rec,
            shutdown_event=shutdown,
            sessions_dir=tmp_path,
            session_base_name="test",
        )
        agent = _make_mock_script_bridge(last_peak_rss_mb=42.5)
        profiler.register("tester", "script", agent)

        profiler._snapshot_all()

        # Value should have been consumed.
        assert agent._last_peak_rss_mb is None

        # Second snapshot should not repeat the stale value.
        profiler._snapshot_all()
        sidecar = tmp_path / "test.tester.memory.jsonl"
        lines = sidecar.read_text().strip().splitlines()
        assert len(lines) == 2
        second = json.loads(lines[1])
        assert second.get("subprocess_rss_mb") is None
        rec.close()

    def test_snapshot_records_to_main_session(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        shutdown = asyncio.Event()
        profiler = AgentMemoryProfiler(
            recorder=rec,
            shutdown_event=shutdown,
            sessions_dir=tmp_path,
            session_base_name="test",
        )
        agent = _make_mock_llm_agent()
        profiler.register("lead", "llm", agent)

        profiler._snapshot_all()

        # Check that a memory_snapshot event was written to the main session log.
        lines = rec.session_file.read_text().strip().splitlines()
        types = [json.loads(line)["type"] for line in lines]
        assert "memory_snapshot" in types
        rec.close()

    @pytest.mark.asyncio
    async def test_profiler_takes_snapshots_on_interval(self, tmp_path: Path) -> None:
        """Profiler should take at least one snapshot within its interval."""
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        shutdown = asyncio.Event()
        profiler = AgentMemoryProfiler(
            recorder=rec,
            shutdown_event=shutdown,
            sessions_dir=tmp_path,
            session_base_name="test",
            interval=1,  # 1 second for fast test
        )
        agent = _make_mock_llm_agent([{"role": "user", "content": "test"}])
        profiler.register("lead", "llm", agent)

        await profiler.start()
        # Wait enough time for at least one snapshot.
        await asyncio.sleep(1.5)
        await profiler.stop()

        sidecar = tmp_path / "test.lead.memory.jsonl"
        lines = sidecar.read_text().strip().splitlines()
        assert len(lines) >= 1
        rec.close()


# ===================================================================
# CLIBridge.current_subprocess_pid property tests
# ===================================================================


class TestCLIBridgeSubprocessPid:
    def test_initial_pid_is_none(self) -> None:
        from lattice.agent.cli_bridge import CLIBridge

        bridge = CLIBridge(
            name="dev",
            role="developer",
            router=MagicMock(),
            recorder=MagicMock(),
            team_name="test",
            peer_names=[],
            cli_type="claude",
        )
        assert bridge.current_subprocess_pid is None

    def test_custom_cli_returns_process_pid(self) -> None:
        from lattice.agent.cli_bridge import CLIBridge

        bridge = CLIBridge(
            name="dev",
            role="developer",
            router=MagicMock(),
            recorder=MagicMock(),
            team_name="test",
            peer_names=[],
            command="echo hello",
        )
        # Simulate a running process.
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = None  # still running
        bridge._process = mock_proc

        assert bridge.current_subprocess_pid == 12345

    def test_custom_cli_returns_none_when_exited(self) -> None:
        from lattice.agent.cli_bridge import CLIBridge

        bridge = CLIBridge(
            name="dev",
            role="developer",
            router=MagicMock(),
            recorder=MagicMock(),
            team_name="test",
            peer_names=[],
            command="echo hello",
        )
        # Simulate an exited process.
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = 0  # exited
        bridge._process = mock_proc

        assert bridge.current_subprocess_pid is None

    def test_claude_adapter_returns_current_pid(self) -> None:
        from lattice.agent.cli_bridge import CLIBridge

        bridge = CLIBridge(
            name="dev",
            role="developer",
            router=MagicMock(),
            recorder=MagicMock(),
            team_name="test",
            peer_names=[],
            cli_type="claude",
        )
        bridge._current_claude_pid = 54321
        assert bridge.current_subprocess_pid == 54321
