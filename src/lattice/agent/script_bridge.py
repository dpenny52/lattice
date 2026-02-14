"""Script Bridge agent — stateless subprocess execution per message."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shlex
from collections.abc import Callable

from lattice.router.router import Router
from lattice.session.models import AgentDoneEvent, AgentStartEvent, ErrorEvent
from lattice.session.recorder import SessionRecorder

logger = logging.getLogger(__name__)

#: Maximum stderr characters to include in error messages.
_MAX_STDERR_CHARS = 500


class ScriptBridge:
    """Agent that runs a shell command as a fresh subprocess per message.

    Satisfies the ``Router.Agent`` protocol via ``handle_message``.

    Unlike ``CLIBridge``, the script bridge is **stateless** — each incoming
    message spawns a new subprocess, pipes the message content to stdin,
    and returns stdout as the result.  Scripts cannot initiate messages
    to other agents; they only respond to the sender.
    """

    def __init__(
        self,
        name: str,
        role: str,
        command: str,
        router: Router,
        recorder: SessionRecorder,
        timeout: float = 30.0,
        on_response: Callable[[str], None] | None = None,
    ) -> None:
        self.name = name
        self._role = role
        self._command = command
        self._router = router
        self._recorder = recorder
        self._timeout = timeout
        self._on_response = on_response

    async def handle_message(self, from_agent: str, content: str) -> None:
        """Run the script with *content* on stdin and route stdout back."""
        self._recorder.record(
            AgentStartEvent(ts="", seq=0, agent=self.name, agent_type="script")
        )

        try:
            await self._execute(from_agent, content)
        finally:
            self._recorder.record(
                AgentDoneEvent(ts="", seq=0, agent=self.name, reason="completed")
            )

    async def _execute(self, from_agent: str, content: str) -> None:
        """Run the subprocess and handle the result."""
        try:
            args = shlex.split(self._command)
        except ValueError as exc:
            self._record_error(f"Invalid command: {exc}")
            return

        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            self._record_error(f"Command not found: {args[0]}")
            return
        except OSError as exc:
            self._record_error(f"Failed to spawn subprocess: {exc}")
            return

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=content.encode()),
                timeout=self._timeout,
            )
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            self._record_error(f"Script timed out after {self._timeout}s")
            return

        if proc.returncode != 0:
            stderr_text = stderr_bytes.decode(errors="replace").strip()[
                :_MAX_STDERR_CHARS
            ]
            self._record_error(
                f"Script exited with code {proc.returncode}: {stderr_text}"
            )
            return

        stdout_text = stdout_bytes.decode(errors="replace").strip()
        if not stdout_text:
            return

        await self._router.send(self.name, from_agent, stdout_text)

        if self._on_response:
            self._on_response(stdout_text)

    def _record_error(self, error_msg: str) -> None:
        """Log and record an error event."""
        logger.error("%s: %s", self.name, error_msg)
        self._recorder.record(
            ErrorEvent(
                ts="", seq=0, agent=self.name, error=error_msg, retrying=False,
            )
        )
