"""Memory monitor — periodic process tree memory checks with warnings."""

from __future__ import annotations

import asyncio
import logging
import os
from collections import defaultdict

from lattice.session.models import StatusEvent
from lattice.session.recorder import SessionRecorder

logger = logging.getLogger(__name__)

#: Default interval between memory checks (seconds).
_DEFAULT_INTERVAL = 5

#: Memory thresholds as fractions of total system RAM.
_WARN_THRESHOLD = 0.70
_CRITICAL_THRESHOLD = 0.85


def _get_total_system_mb() -> float:
    """Return total physical memory in MB."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except (ValueError, OSError):
        return 0.0


async def _get_tree_rss_mb() -> float | None:
    """Return total RSS (in MB) for this process and all descendants."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ps", "ax", "-o", "pid=,ppid=,rss=",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
    except Exception:
        return None

    pid = os.getpid()
    children_map: dict[int, list[int]] = defaultdict(list)
    rss_map: dict[int, int] = {}

    for line in stdout.decode(errors="replace").strip().split("\n"):
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            p, pp, rss = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        children_map[pp].append(p)
        rss_map[p] = rss

    # Walk the tree from our PID.
    total_kb = 0
    stack = [pid]
    while stack:
        current = stack.pop()
        total_kb += rss_map.get(current, 0)
        stack.extend(children_map.get(current, []))

    return total_kb / 1024


class MemoryMonitor:
    """Periodically checks process tree memory and logs warnings.

    Runs as a background asyncio task alongside the heartbeat.
    Warnings are printed to stderr and recorded as session events.
    """

    def __init__(
        self,
        recorder: SessionRecorder,
        shutdown_event: asyncio.Event,
        interval: int = _DEFAULT_INTERVAL,
    ) -> None:
        self._recorder = recorder
        self._shutdown_event = shutdown_event
        self._interval = interval
        self._task: asyncio.Task[None] | None = None

        self._total_mb = _get_total_system_mb()
        self._warn_mb = self._total_mb * _WARN_THRESHOLD
        self._critical_mb = self._total_mb * _CRITICAL_THRESHOLD
        self._last_level: str = "ok"

    async def start(self) -> None:
        """Start the background monitoring loop."""
        if self._total_mb <= 0:
            logger.warning("Could not determine system memory — memory monitor disabled")
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Cancel the monitoring loop."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        """Sleep-and-check loop that runs until shutdown or cancellation."""
        try:
            while not self._shutdown_event.is_set():
                elapsed = 0.0
                while elapsed < self._interval:
                    if self._shutdown_event.is_set():
                        return
                    await asyncio.sleep(min(1.0, self._interval - elapsed))
                    elapsed += 1.0

                await self._check()
        except asyncio.CancelledError:
            return

    async def _check(self) -> None:
        """Run a single memory check."""
        tree_mb = await _get_tree_rss_mb()
        if tree_mb is None:
            return

        pct = tree_mb / self._total_mb * 100

        if tree_mb >= self._critical_mb:
            level = "critical"
        elif tree_mb >= self._warn_mb:
            level = "warn"
        else:
            level = "ok"

        # Only log on transitions or repeated critical.
        if level == "ok":
            if self._last_level != "ok":
                logger.info("Memory returned to normal: %.0f MB (%.0f%%)", tree_mb, pct)
            self._last_level = level
            return

        if level != self._last_level or level == "critical":
            import click

            tag = "⚠️  HIGH MEMORY" if level == "warn" else "🔴 CRITICAL MEMORY"
            msg = (
                f"{tag}: process tree using {tree_mb:.0f} MB "
                f"({pct:.0f}% of {self._total_mb:.0f} MB system RAM)"
            )
            click.echo(msg, err=True)
            logger.warning(msg)

            self._recorder.record(
                StatusEvent(
                    ts="",
                    seq=0,
                    agent="__system__",
                    status=f"memory_{level}: {tree_mb:.0f}MB ({pct:.0f}%)",
                )
            )

        self._last_level = level
