"""Heartbeat — periodic progress checks for the entry agent."""

from __future__ import annotations

import asyncio
import logging
import re

from lattice.router.router import Router
from lattice.session.models import StatusEvent
from lattice.session.recorder import SessionRecorder

logger = logging.getLogger(__name__)

#: Heartbeat prompt sent to the entry agent.
_HEARTBEAT_PROMPT = (
    "Briefly update the user on current progress. "
    "If all tasks are complete, include the marker [HEARTBEAT:DONE]. "
    "If you are stuck or waiting for input, include the marker [HEARTBEAT:STUCK]."
)

#: Pattern to detect done/stuck markers in heartbeat responses.
_DONE_RE = re.compile(r"\[HEARTBEAT:DONE\]", re.IGNORECASE)
_STUCK_RE = re.compile(r"\[HEARTBEAT:STUCK\]", re.IGNORECASE)

#: Special sender name for heartbeat messages (bypasses topology).
SYSTEM_SENDER = "__system__"


class Heartbeat:
    """Sends periodic heartbeat messages to the entry agent.

    The heartbeat fires every *interval* seconds, sending a system-level
    message through the router.  The entry agent handles it in a dedicated
    ``"__system__"`` conversation thread so heartbeat context never
    pollutes peer threads.
    """

    def __init__(
        self,
        interval: int,
        router: Router,
        entry_agent: str,
        recorder: SessionRecorder,
        shutdown_event: asyncio.Event,
    ) -> None:
        self._interval = interval
        self._router = router
        self._entry_agent = entry_agent
        self._recorder = recorder
        self._shutdown_event = shutdown_event
        self._task: asyncio.Task[None] | None = None
        self._paused = False
        self._done_flag = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """Start the heartbeat timer loop as a background task."""
        if self._interval <= 0:
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Cancel the heartbeat timer and wait for cleanup."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def fire(self) -> None:
        """Send an immediate heartbeat (used by ``/status``)."""
        await self._send_heartbeat()

    def pause(self) -> None:
        """Pause heartbeat (during user input)."""
        self._paused = True

    def resume(self) -> None:
        """Resume heartbeat (after user input)."""
        self._paused = False

    @property
    def done_flag(self) -> bool:
        """True if the entry agent signalled completion."""
        return self._done_flag

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    async def _loop(self) -> None:
        """Sleep-and-fire loop that runs until shutdown or cancellation."""
        try:
            while not self._shutdown_event.is_set():
                # Sleep in small increments so we can respond to shutdown
                elapsed = 0.0
                while elapsed < self._interval:
                    if self._shutdown_event.is_set():
                        return
                    await asyncio.sleep(min(1.0, self._interval - elapsed))
                    elapsed += 1.0

                if self._paused:
                    continue

                await self._send_heartbeat()
        except asyncio.CancelledError:
            return

    async def _send_heartbeat(self) -> None:
        """Send a single heartbeat message and record a status event."""
        self._recorder.record(
            StatusEvent(
                ts="",
                seq=0,
                agent=self._entry_agent,
                status="heartbeat_sent",
            )
        )

        try:
            await self._router.send(
                SYSTEM_SENDER, self._entry_agent, _HEARTBEAT_PROMPT,
            )
        except Exception:
            logger.exception("Failed to send heartbeat to %s", self._entry_agent)

    def check_response(self, content: str) -> None:
        """Inspect a heartbeat response for done/stuck markers.

        Called by the response callback wrapper installed in ``up.py``.
        """
        if _DONE_RE.search(content):
            self._done_flag = True
            self._recorder.record(
                StatusEvent(
                    ts="",
                    seq=0,
                    agent=self._entry_agent,
                    status="heartbeat_done",
                )
            )
        elif _STUCK_RE.search(content):
            self._recorder.record(
                StatusEvent(
                    ts="",
                    seq=0,
                    agent=self._entry_agent,
                    status="heartbeat_stuck",
                )
            )
