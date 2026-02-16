"""BackgroundLoop — base class for periodic async background tasks.

Provides a shutdown-aware sleep loop and managed task lifecycle
(start / stop) used by Heartbeat, MemoryMonitor, and AgentMemoryProfiler.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

logger = logging.getLogger(__name__)


class BackgroundLoop:
    """Base class for async background loops with graceful shutdown.

    Subclasses override :meth:`_should_start` (optional guard) and
    :meth:`_tick` (the work to do each interval).
    """

    def __init__(
        self,
        shutdown_event: asyncio.Event,
        interval: int | float,
    ) -> None:
        self._shutdown_event = shutdown_event
        self._interval = interval
        self._task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """Start the background loop if :meth:`_should_start` allows."""
        if not self._should_start():
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Cancel the background task and wait for cleanup."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    # ------------------------------------------------------------------ #
    # Override points
    # ------------------------------------------------------------------ #

    def _should_start(self) -> bool:
        """Return ``False`` to skip starting.  Override in subclasses."""
        return True

    async def _tick(self) -> None:
        """Work to perform each interval.  Must be overridden."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    async def _shutdown_aware_sleep(self, duration: float) -> bool:
        """Sleep in 1-second chunks, returning ``True`` if shutdown was signalled."""
        elapsed = 0.0
        while elapsed < duration:
            if self._shutdown_event.is_set():
                return True
            await asyncio.sleep(min(1.0, duration - elapsed))
            elapsed += 1.0
        return False

    async def _loop(self) -> None:
        """Sleep-and-tick loop that runs until shutdown or cancellation."""
        try:
            while not self._shutdown_event.is_set():
                if await self._shutdown_aware_sleep(self._interval):
                    return
                await self._tick()
        except asyncio.CancelledError:
            return
