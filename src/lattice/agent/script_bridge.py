"""Script Bridge agent — stateless subprocess execution per message."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import platform
import re
import resource
import shlex
from collections.abc import Awaitable, Callable

import click

from lattice.agent.helpers import format_stderr_preview, record_error
from lattice.router.router import Router
from lattice.session.models import AgentDoneEvent, AgentStartEvent
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
        timeout: float = 120.0,
        on_response: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.name = name
        self._role = role
        self._command = command
        self._router = router
        self._recorder = recorder
        self._timeout = timeout
        self._on_response = on_response

        # Peak RSS delta from last execution (MB), for memory profiling.
        self._last_peak_rss_mb: float | None = None

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

    # Shell metacharacters that require a shell interpreter.
    _SHELL_META_RE = re.compile(r"[|&;<>(){}\$`!]|2>&1")

    async def _execute(self, from_agent: str, content: str) -> None:
        """Run the subprocess and handle the result."""
        needs_shell = self._SHELL_META_RE.search(self._command) is not None

        if needs_shell:
            try:
                proc = await asyncio.create_subprocess_shell(
                    self._command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except OSError as exc:
                click.echo(
                    f"Agent '{self.name}' — failed to spawn subprocess: {exc}",
                    err=True,
                )
                self._record_error(f"Failed to spawn subprocess: {exc}")
                return
        else:
            try:
                args = shlex.split(self._command)
            except ValueError as exc:
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
                click.echo(
                    f"Agent '{self.name}' — command not found: {args[0]}\n"
                    f"Make sure '{args[0]}' is installed and on your PATH.",
                    err=True,
                )
                self._record_error(f"Command not found: {args[0]}")
                return
            except OSError as exc:
                click.echo(
                    f"Agent '{self.name}' — failed to spawn subprocess: {exc}",
                    err=True,
                )
                self._record_error(f"Failed to spawn subprocess: {exc}")
                return

        # Capture children RSS before/after for peak memory delta.
        # NOTE: ru_maxrss is a high-water mark across ALL waited-for children,
        # not per-child.  The delta is only non-zero when this child sets a new
        # peak — i.e. uses more RSS than any previous child in this process.
        # Executions that stay below the prior peak will report delta=0 and
        # _last_peak_rss_mb won't be updated.
        pre_rusage = resource.getrusage(resource.RUSAGE_CHILDREN)

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=content.encode()),
                timeout=self._timeout,
            )
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
            click.echo(
                f"Agent '{self.name}' — script timed out after {self._timeout}s",
                err=True,
            )
            self._record_error(f"Script timed out after {self._timeout}s")
            return

        # Calculate peak RSS delta from children.
        post_rusage = resource.getrusage(resource.RUSAGE_CHILDREN)
        rss_delta = post_rusage.ru_maxrss - pre_rusage.ru_maxrss
        if rss_delta > 0:
            # macOS ru_maxrss is in bytes, Linux in KB — normalize to MB.
            if platform.system() == "Darwin":
                self._last_peak_rss_mb = rss_delta / (1024 * 1024)
            else:
                self._last_peak_rss_mb = rss_delta / 1024

        if proc.returncode != 0:
            stderr_text = stderr_bytes.decode(errors="replace").strip()

            # Show last 5 lines of stderr
            stderr_preview = format_stderr_preview(stderr_text)

            error_msg = f"Agent '{self.name}' exited with code {proc.returncode}."
            if stderr_preview:
                click.echo(f"{error_msg} Stderr:\n  {stderr_preview}", err=True)
            else:
                click.echo(error_msg, err=True)

            # Record with truncated stderr in session log
            stderr_snippet = stderr_text[:_MAX_STDERR_CHARS]
            full_error = f"Script exited with code {proc.returncode}: {stderr_snippet}"
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
        record_error(self._recorder, self.name, error_msg, logger=logger)
