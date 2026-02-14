"""CLI Bridge agent — spawns CLI tools as subprocesses with JSONL communication."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import shlex
from collections.abc import Callable

from lattice.router.router import Router
from lattice.session.models import (
    AgentDoneEvent,
    AgentStartEvent,
    ErrorEvent,
    StatusEvent,
)
from lattice.session.recorder import SessionRecorder

logger = logging.getLogger(__name__)

#: Seconds to wait for graceful shutdown before SIGTERM.
_SHUTDOWN_WAIT = 5.0

#: Seconds to wait after SIGTERM before SIGKILL.
_SIGTERM_WAIT = 3.0


class CLIBridge:
    """Agent that wraps a CLI subprocess with bidirectional JSONL.

    Satisfies the ``Router.Agent`` protocol via ``handle_message``.

    Two modes of operation:

    * **Claude adapter** (``cli_type="claude"``): spawns a new ``claude``
      subprocess per task, collects JSON output, and routes results.
    * **Custom CLI** (``command`` specified): spawns a long-running process
      and communicates over stdin/stdout using the Lattice JSONL protocol.
    """

    def __init__(
        self,
        name: str,
        role: str,
        router: Router,
        recorder: SessionRecorder,
        team_name: str,
        peer_names: list[str],
        cli_type: str | None = None,
        command: str | None = None,
        on_response: Callable[[str], None] | None = None,
    ) -> None:
        self.name = name
        self._role = role
        self._router = router
        self._recorder = recorder
        self._team_name = team_name
        self._peer_names = peer_names
        self._cli_type = cli_type
        self._command = command
        self._on_response = on_response

        # Subprocess state (custom CLI only).
        self._process: asyncio.subprocess.Process | None = None
        self._read_task: asyncio.Task[None] | None = None
        self._started = False

        # Task tracking.
        self._task_counter = 0
        self._pending_tasks: dict[str, asyncio.Future[str]] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def start(self) -> None:
        """Start the subprocess.

        For custom CLIs, spawns the long-running process and begins the
        background read loop.  For the Claude adapter, this is a no-op
        (each task spawns its own subprocess).
        """
        if self._started:
            return

        if self._cli_type == "claude":
            # Claude adapter doesn't use a long-running process.
            self._started = True
            return

        if not self._command:
            msg = f"CLI agent '{self.name}' has no command configured"
            raise ValueError(msg)

        args = shlex.split(self._command)
        self._process = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._started = True
        self._read_task = asyncio.create_task(self._read_loop())

    async def shutdown(self) -> None:
        """Graceful shutdown: signal -> wait -> SIGTERM -> SIGKILL."""
        if self._cli_type == "claude":
            self._started = False
            return

        proc = self._process
        if proc is None or proc.returncode is not None:
            self._started = False
            return

        # 1. Send shutdown message via stdin.
        try:
            if proc.stdin is not None:
                shutdown_msg = json.dumps({"type": "shutdown"}) + "\n"
                proc.stdin.write(shutdown_msg.encode())
                await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

        # 2. Wait for graceful exit.
        try:
            await asyncio.wait_for(proc.wait(), timeout=_SHUTDOWN_WAIT)
        except TimeoutError:
            # 3. SIGTERM.
            with contextlib.suppress(ProcessLookupError):
                proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=_SIGTERM_WAIT)
            except TimeoutError:
                # 4. SIGKILL.
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                await proc.wait()

        # Cancel the read loop.
        if self._read_task is not None and not self._read_task.done():
            self._read_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._read_task

        self._started = False

    # ------------------------------------------------------------------ #
    # Agent protocol
    # ------------------------------------------------------------------ #

    async def handle_message(self, from_agent: str, content: str) -> None:
        """Handle an incoming message -- dispatch to the subprocess."""
        # Lazy start on first message.
        if not self._started:
            await self.start()

        self._recorder.record(
            AgentStartEvent(ts="", seq=0, agent=self.name, agent_type="cli")
        )

        if self._cli_type == "claude":
            await self._handle_claude_task(from_agent, content)
        else:
            await self._handle_custom_task(from_agent, content)

        self._recorder.record(
            AgentDoneEvent(ts="", seq=0, agent=self.name, reason="completed")
        )

    # ------------------------------------------------------------------ #
    # Claude adapter
    # ------------------------------------------------------------------ #

    async def _handle_claude_task(self, from_agent: str, content: str) -> None:
        """Run a single Claude CLI invocation for this task."""
        prompt = f"{self._role}\n\nTask from {from_agent}: {content}"

        try:
            proc = await asyncio.create_subprocess_exec(
                "claude",
                "-p",
                prompt,
                "--output-format",
                "json",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await proc.communicate()
        except FileNotFoundError:
            error_msg = "Claude CLI not found -- is 'claude' installed and on PATH?"
            logger.error("%s: %s", self.name, error_msg)
            self._recorder.record(
                ErrorEvent(
                    ts="", seq=0, agent=self.name, error=error_msg, retrying=False,
                )
            )
            return
        except OSError as exc:
            error_msg = f"Failed to spawn Claude CLI: {exc}"
            logger.error("%s: %s", self.name, error_msg)
            self._recorder.record(
                ErrorEvent(
                    ts="", seq=0, agent=self.name, error=error_msg, retrying=False,
                )
            )
            return

        if proc.returncode != 0:
            stderr_text = stderr_bytes.decode(errors="replace").strip()
            error_msg = f"Claude CLI exited with code {proc.returncode}: {stderr_text}"
            logger.error("%s: %s", self.name, error_msg)
            self._recorder.record(
                ErrorEvent(
                    ts="", seq=0, agent=self.name, error=error_msg, retrying=False,
                )
            )
            return

        # Parse the JSON output.
        stdout_text = stdout_bytes.decode(errors="replace").strip()
        if not stdout_text:
            return

        try:
            result = json.loads(stdout_text)
        except json.JSONDecodeError:
            # Treat raw text as the result.
            result = {"result": stdout_text}

        # Extract the response content.
        if isinstance(result, dict):
            result_text = result.get("result", "")
        else:
            result_text = str(result)

        if result_text and self._on_response:
            self._on_response(result_text)

    # ------------------------------------------------------------------ #
    # Custom CLI (long-running JSONL process)
    # ------------------------------------------------------------------ #

    async def _handle_custom_task(self, from_agent: str, content: str) -> None:
        """Send a task to the long-running subprocess and wait for completion."""
        proc = self._process
        if proc is None or proc.stdin is None:
            error_msg = "Subprocess not running"
            logger.error("%s: %s", self.name, error_msg)
            self._recorder.record(
                ErrorEvent(
                    ts="", seq=0, agent=self.name, error=error_msg, retrying=False,
                )
            )
            return

        # Generate task ID.
        self._task_counter += 1
        task_id = f"t_{self._task_counter:03d}"

        # Create a future for task completion.
        loop = asyncio.get_event_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_tasks[task_id] = future

        # Send task to subprocess stdin.
        task_msg = json.dumps({
            "type": "task",
            "id": task_id,
            "from": from_agent,
            "content": content,
        }) + "\n"

        try:
            proc.stdin.write(task_msg.encode())
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            error_msg = f"Failed to write to subprocess: {exc}"
            logger.error("%s: %s", self.name, error_msg)
            self._recorder.record(
                ErrorEvent(
                    ts="", seq=0, agent=self.name, error=error_msg, retrying=False,
                )
            )
            self._pending_tasks.pop(task_id, None)
            if not future.done():
                future.set_exception(RuntimeError(error_msg))
            return

        # Wait for the task to complete (result comes via _read_loop).
        try:
            result = await future
        except Exception as exc:
            logger.error("%s: task %s failed: %s", self.name, task_id, exc)
        else:
            if result and self._on_response:
                self._on_response(result)

    # ------------------------------------------------------------------ #
    # Background read loop
    # ------------------------------------------------------------------ #

    async def _read_loop(self) -> None:
        """Continuously read JSONL from subprocess stdout and dispatch."""
        proc = self._process
        if proc is None or proc.stdout is None:
            return

        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    # EOF -- subprocess has exited.
                    break

                line_str = line.decode(errors="replace").strip()
                if not line_str:
                    continue

                try:
                    msg = json.loads(line_str)
                except json.JSONDecodeError:
                    logger.warning(
                        "%s: non-JSON output from subprocess: %s",
                        self.name,
                        line_str[:200],
                    )
                    continue

                await self._dispatch_message(msg)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("%s: read loop error: %s", self.name, exc)

        # If we got here, the subprocess exited (or we hit an error).
        returncode = proc.returncode
        if returncode is not None and returncode != 0:
            error_msg = f"Subprocess exited with code {returncode}"
            logger.error("%s: %s", self.name, error_msg)
            self._recorder.record(
                ErrorEvent(
                    ts="", seq=0, agent=self.name, error=error_msg, retrying=False,
                )
            )

        # Fail any pending futures.
        for task_id, future in list(self._pending_tasks.items()):
            if not future.done():
                future.set_exception(
                    RuntimeError(f"Subprocess exited before task {task_id} completed")
                )
            self._pending_tasks.pop(task_id, None)

    async def _dispatch_message(self, msg: dict[str, object]) -> None:
        """Route a parsed JSONL message from the subprocess."""
        msg_type = msg.get("type")

        if msg_type == "status":
            status_text = str(msg.get("status", ""))
            self._recorder.record(
                StatusEvent(ts="", seq=0, agent=self.name, status=status_text)
            )

        elif msg_type == "message":
            to = str(msg.get("to", ""))
            content = str(msg.get("content", ""))
            if to and content:
                try:
                    await self._router.send(self.name, to, content)
                except Exception as exc:
                    logger.error(
                        "%s: failed to route message to '%s': %s",
                        self.name,
                        to,
                        exc,
                    )

        elif msg_type == "result":
            task_id = str(msg.get("task_id", ""))
            content = str(msg.get("content", ""))
            future = self._pending_tasks.get(task_id)
            if future is not None and not future.done():
                future.set_result(content)

        elif msg_type == "done":
            task_id = str(msg.get("task_id", ""))
            future = self._pending_tasks.pop(task_id, None)
            if future is not None and not future.done():
                future.set_result("")

        else:
            logger.warning(
                "%s: unknown message type from subprocess: %s",
                self.name,
                msg_type,
            )
