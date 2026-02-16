"""Heartbeat — periodic progress checks for the entry agent."""

from __future__ import annotations

import asyncio
import logging
import re

from lattice.background_loop import BackgroundLoop
from lattice.constants import SYSTEM_SENDER
from lattice.router.router import Router
from lattice.session.models import StatusEvent
from lattice.session.recorder import SessionRecorder

logger = logging.getLogger(__name__)

#: Heartbeat prompt sent to the entry agent.
_HEARTBEAT_PROMPT = (
    "Briefly update the user on current progress. "
    "If all tasks are complete, include the marker [HEARTBEAT:DONE]. "
    "If you are stuck and unable to make progress (e.g. hit an error, "
    "need human input), include [HEARTBEAT:STUCK]. "
    "Note: waiting for other agents to finish their work is normal — "
    "do NOT report STUCK just because you sent messages and are "
    "waiting for replies."
)

#: Pattern to detect done/stuck markers in heartbeat responses.
_DONE_RE = re.compile(r"\[HEARTBEAT:DONE\]", re.IGNORECASE)
_STUCK_RE = re.compile(r"\[HEARTBEAT:STUCK\]", re.IGNORECASE)


class Heartbeat(BackgroundLoop):
    """Sends periodic heartbeat messages to the entry agent.

    The heartbeat fires every *interval* seconds, sending a system-level
    message through the router.  The entry agent handles it in its
    unified conversation thread alongside all other messages.
    """

    def __init__(
        self,
        interval: int,
        router: Router,
        entry_agent: str,
        recorder: SessionRecorder,
        shutdown_event: asyncio.Event,
    ) -> None:
        super().__init__(shutdown_event=shutdown_event, interval=interval)
        self._router = router
        self._entry_agent = entry_agent
        self._recorder = recorder
        self._paused = False
        self._done_flag = False
        self._pending = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def _should_start(self) -> bool:
        return self._interval > 0

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

    def consume_pending(self) -> bool:
        """Return ``True`` (once) if a heartbeat response is expected.

        Atomically clears the pending flag so only the first call after
        a heartbeat send returns ``True``.
        """
        if self._pending:
            self._pending = False
            return True
        return False

    @staticmethod
    def strip_markers(content: str) -> str:
        """Remove ``[HEARTBEAT:DONE]`` / ``[HEARTBEAT:STUCK]`` markers."""
        result = _DONE_RE.sub("", content)
        result = _STUCK_RE.sub("", result)
        return result.strip()

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    async def _tick(self) -> None:
        """Send a heartbeat unless paused."""
        if self._paused:
            return
        await self._send_heartbeat()

    async def _send_heartbeat(self) -> None:
        """Send a single heartbeat message and record a status event."""
        self._pending = True
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
