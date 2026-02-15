"""Script Bridge agent — stateless subprocess execution per message."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shlex
from collections.abc import Awaitable, Callable

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
        on_response: Callable[[str], Awaitable[None]] | None = None,
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
            import click
            click.echo(f"Agent '{self.name}' — invalid command: {exc}", err=True)
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
            import click
            click.echo(
                f"Agent '{self.name}' — command not found: {args[0]}\n"
                f"Make sure '{args[0]}' is installed and on your PATH.",
                err=True,
            )
            self._record_error(f"Command not found: {args[0]}")
            return
        except OSError as exc:
            import click
            click.echo(f"Agent '{self.name}' — failed to spawn subprocess: {exc}", err=True)
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
            import click
            click.echo(
                f"Agent '{self.name}' — script timed out after {self._timeout}s",
                err=True,
            )
            self._record_error(f"Script timed out after {self._timeout}s")
            return

        if proc.returncode != 0:
            stderr_text = stderr_bytes.decode(errors="replace").strip()

            # Show last 5 lines of stderr
            stderr_lines = [line for line in stderr_text.split('\n') if line.strip()]
            last_lines = stderr_lines[-5:] if len(stderr_lines) > 5 else stderr_lines
            stderr_preview = "\n  ".join(last_lines)

            import click
            error_msg = f"Agent '{self.name}' exited with code {proc.returncode}."
            if stderr_preview:
                click.echo(f"{error_msg} Stderr:\n  {stderr_preview}", err=True)
            else:
                click.echo(error_msg, err=True)

            # Record with truncated stderr in session log
            full_error = f"Script exited with code {proc.returncode}: {stderr_text[:_MAX_STDERR_CHARS]}"
            self._record_error(full_error)
            return

        stdout_text = stdout_bytes.decode(errors="replace").strip()
        if not stdout_text:
            return

        await self._router.send(self.name, from_agent, stdout_text)

        if self._on_response:
            await self._on_response(stdout_text)

    async def shutdown(self) -> None:
        """No-op shutdown -- scripts are stateless."""

    def _record_error(self, error_msg: str) -> None:
        """Log and record an error event."""
        logger.error("%s: %s", self.name, error_msg)
        self._recorder.record(
            ErrorEvent(
                ts="", seq=0, agent=self.name, error=error_msg, retrying=False, context="subprocess",
            )
        )
