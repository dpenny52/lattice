"""Per-agent memory profiler — periodic snapshots to JSONL sidecar files."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lattice.background_loop import BackgroundLoop
from lattice.memory_monitor import get_available_mb, get_pid_rss_mb
from lattice.session.models import MemorySnapshotEvent
from lattice.session.recorder import SessionRecorder

logger = logging.getLogger(__name__)

#: Default profiling interval (seconds). 5x slower than the system monitor.
_DEFAULT_INTERVAL = 10


def estimate_thread_size_kb(thread: list[dict[str, Any]]) -> float:
    """Estimate memory footprint of a conversation thread in KB.

    Uses JSON serialization as a fast proxy — correlates well with actual
    in-memory size without the overhead of ``tracemalloc``.
    """
    return len(json.dumps(thread, default=str)) / 1024


class AgentMemoryLogger:
    """Per-agent JSONL file writer for memory snapshots.

    Lightweight wrapper: timestamp + snapshot dict, one per line.
    Thread-safe via a lock (same pattern as ``SessionRecorder``).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._fh = path.open("a", encoding="utf-8")

    def write(self, snapshot: dict[str, Any]) -> None:
        """Append a timestamped snapshot line to the JSONL file."""
        snapshot["ts"] = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        with self._lock:
            if self._fh.closed:
                return
            self._fh.write(json.dumps(snapshot, default=str) + "\n")
            self._fh.flush()

    def close(self) -> None:
        """Close the underlying file handle."""
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


class AgentMemoryProfiler(BackgroundLoop):
    """Periodically measures per-agent memory and writes to sidecar logs.

    Maintains a registry of agents and runs an async background loop
    that snapshots each agent every ``interval`` seconds.
    """

    def __init__(
        self,
        recorder: SessionRecorder,
        shutdown_event: asyncio.Event,
        sessions_dir: Path,
        session_base_name: str,
        interval: int = _DEFAULT_INTERVAL,
    ) -> None:
        super().__init__(shutdown_event=shutdown_event, interval=interval)
        self._recorder = recorder
        self._sessions_dir = sessions_dir
        self._session_base_name = session_base_name

        # Registry: name -> (agent_type, agent_object).
        self._agents: dict[str, tuple[str, Any]] = {}
        # Per-agent JSONL loggers.
        self._loggers: dict[str, AgentMemoryLogger] = {}

    def register(self, name: str, agent_type: str, agent: Any) -> None:
        """Register an agent for periodic profiling."""
        self._agents[name] = (agent_type, agent)
        log_path = self._sessions_dir / f"{self._session_base_name}.{name}.memory.jsonl"
        self._loggers[name] = AgentMemoryLogger(log_path)
        logger.debug("Memory profiler: registered %s (%s)", name, agent_type)

    def _should_start(self) -> bool:
        if not self._agents:
            logger.debug("Memory profiler: no agents registered, not starting")
            return False
        logger.info(
            "Memory profiler started: %d agents, %ds interval",
            len(self._agents),
            self._interval,
        )
        return True

    async def stop(self) -> None:
        """Cancel the profiling loop and close all file handles."""
        await super().stop()
        for agent_logger in self._loggers.values():
            agent_logger.close()

    async def _tick(self) -> None:
        self._snapshot_all()

    def _snapshot_all(self) -> None:
        """Take a memory snapshot for every registered agent."""
        system_available = get_available_mb()
        host_rss = get_pid_rss_mb(os.getpid())

        for name, (agent_type, agent) in self._agents.items():
            try:
                snapshot = self._snapshot_agent(
                    name, agent_type, agent, system_available, host_rss,
                )
                # Write to main session JSONL first (before sidecar mutates dict).
                self._recorder.record(
                    MemorySnapshotEvent(ts="", seq=0, **snapshot)
                )

                # Write to per-agent sidecar JSONL.
                agent_logger = self._loggers.get(name)
                if agent_logger is not None:
                    agent_logger.write(snapshot)
            except Exception:
                logger.exception("Memory profiler: error snapshotting %s", name)

    def _snapshot_agent(
        self,
        name: str,
        agent_type: str,
        agent: Any,
        system_available: float | None,
        host_rss: float | None,
    ) -> dict[str, Any]:
        """Build a snapshot dict for a single agent."""
        snapshot: dict[str, Any] = {
            "agent": name,
            "agent_type": agent_type,
            "system_available_mb": system_available,
        }

        if agent_type == "llm":
            snapshot["process_rss_mb"] = host_rss
            thread = getattr(agent, "_thread", None)
            if thread is not None:
                snapshot["thread_messages"] = len(thread)
                snapshot["thread_size_kb"] = round(estimate_thread_size_kb(thread), 1)

        elif agent_type == "cli":
            snapshot["process_rss_mb"] = host_rss
            # Get subprocess PID and RSS.
            sub_pid = getattr(agent, "current_subprocess_pid", None)
            if sub_pid is not None:
                snapshot["subprocess_pid"] = sub_pid
                snapshot["subprocess_rss_mb"] = get_pid_rss_mb(sub_pid)
            # Queue depth.
            queue = getattr(agent, "_message_queue", None)
            if queue is not None:
                snapshot["queue_depth"] = queue.qsize()

        elif agent_type == "script":
            snapshot["process_rss_mb"] = host_rss
            # Read-and-clear: _last_peak_rss_mb is a high-water mark from
            # RUSAGE_CHILDREN so it only updates when a new peak is set.
            # Clearing after read prevents stale values from repeating
            # across snapshots when subsequent executions stay below the peak.
            peak_rss = getattr(agent, "_last_peak_rss_mb", None)
            if peak_rss is not None:
                snapshot["subprocess_rss_mb"] = peak_rss
                agent._last_peak_rss_mb = None

        return snapshot
