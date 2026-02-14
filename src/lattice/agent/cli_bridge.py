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
    CLIProgressEvent,
    CLITextChunkEvent,
    CLIThinkingEvent,
    CLIToolCallEvent,
    ErrorEvent,
    StatusEvent,
)
from lattice.session.recorder import SessionRecorder

logger = logging.getLogger(__name__)

#: Seconds to wait for graceful shutdown before SIGTERM.
_SHUTDOWN_WAIT = 5.0

#: Seconds to wait after SIGTERM before SIGKILL.
_SIGTERM_WAIT = 3.0

#: Maximum bytes per JSONL line from subprocess stdout (1 MB).
_MAX_LINE_BYTES = 1_048_576


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

        # Claude adapter: message queue & conversation state.
        self._message_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue(maxsize=100)
        self._claude_busy = False
        self._claude_busy_lock = asyncio.Lock()
        self._claude_conversation_id: str | None = None

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
            stderr=asyncio.subprocess.DEVNULL,
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

        if self._cli_type == "claude":
            # Atomically check busy state to prevent race conditions.
            async with self._claude_busy_lock:
                if self._claude_busy:
                    # Check if queue has space before adding.
                    if self._message_queue.full():
                        error_msg = f"Message queue full (100 messages) — rejecting message from {from_agent}"
                        logger.error("%s: %s", self.name, error_msg)
                        self._recorder.record(
                            ErrorEvent(
                                ts="", seq=0, agent=self.name, error=error_msg, retrying=False,
                            )
                        )
                        if self._on_response:
                            self._on_response(f"[{self.name} queue full, message rejected]")
                        return

                    # Queue the message and show feedback.
                    await self._message_queue.put((from_agent, content))
                    if self._on_response:
                        self._on_response(f"[{self.name} is busy, message queued]")
                    logger.info(
                        "%s: queued message from %s (queue size: %d)",
                        self.name,
                        from_agent,
                        self._message_queue.qsize(),
                    )
                    return

            # Not busy — process immediately and handle queue afterwards.
            self._recorder.record(
                AgentStartEvent(ts="", seq=0, agent=self.name, agent_type="cli")
            )
            await self._handle_claude_task(from_agent, content)
            self._recorder.record(
                AgentDoneEvent(ts="", seq=0, agent=self.name, reason="completed")
            )

            # Process queued messages in order.
            await self._process_message_queue()
        else:
            # Custom CLI — no queue needed.
            self._recorder.record(
                AgentStartEvent(ts="", seq=0, agent=self.name, agent_type="cli")
            )
            await self._handle_custom_task(from_agent, content)
            self._recorder.record(
                AgentDoneEvent(ts="", seq=0, agent=self.name, reason="completed")
            )

    # ------------------------------------------------------------------ #
    # Claude adapter
    # ------------------------------------------------------------------ #

    async def _handle_claude_task(
        self, from_agent: str, content: str, *, is_followup: bool = False
    ) -> None:
        """Run a single Claude CLI invocation for this task.

        Args:
            from_agent: Name of the agent sending the message.
            content: Message content.
            is_followup: If True, use --continue to preserve conversation context.
        """
        self._claude_busy = True

        # Include role only in the first message; follow-ups use the preserved conversation.
        if is_followup:
            prompt = f"Task from {from_agent}: {content}"
        else:
            prompt = f"{self._role}\n\nTask from {from_agent}: {content}"

        try:
            cmd_args = ["claude", "-p", prompt, "--output-format", "stream-json"]

            # Use --continue flag for follow-up messages to preserve context.
            if is_followup and self._claude_conversation_id is not None:
                cmd_args.extend(["--continue", self._claude_conversation_id])

            cmd_args.append("--dangerously-skip-permissions")

            proc = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            error_msg = "Claude CLI not found -- is 'claude' installed and on PATH?"
            logger.error("%s: %s", self.name, error_msg)
            self._recorder.record(
                ErrorEvent(
                    ts="", seq=0, agent=self.name, error=error_msg, retrying=False,
                )
            )
            self._claude_busy = False
            return
        except OSError as exc:
            error_msg = f"Failed to spawn Claude CLI: {exc}"
            logger.error("%s: %s", self.name, error_msg)
            self._recorder.record(
                ErrorEvent(
                    ts="", seq=0, agent=self.name, error=error_msg, retrying=False,
                )
            )
            self._claude_busy = False
            return

        # Stream stdout and parse events as they arrive.
        result_text = await self._stream_claude_output(proc)

        # Wait for process to complete.
        returncode = await proc.wait()

        if returncode != 0:
            # Read stderr for error details.
            stderr_bytes = await proc.stderr.read() if proc.stderr else b""
            stderr_text = stderr_bytes[:2048].decode(errors="replace").strip()[:500]
            error_msg = f"Claude CLI exited with code {returncode}: {stderr_text}"
            logger.error("%s: %s", self.name, error_msg)
            self._recorder.record(
                ErrorEvent(
                    ts="", seq=0, agent=self.name, error=error_msg, retrying=False,
                )
            )
            self._claude_busy = False
            return

        if result_text:
            if self._on_response:
                self._on_response(result_text)
            # Route the result back to the sender so the conversation continues.
            # Skip routing back to "user" — that's handled by on_response.
            if from_agent != "user":
                await self._router.send(self.name, from_agent, result_text)

        self._claude_busy = False

    async def _stream_claude_output(self, proc: asyncio.subprocess.Process) -> str:
        """Stream and parse JSON events from Claude CLI stdout.

        Returns the final result text.
        """
        if proc.stdout is None:
            return ""

        result_text = ""
        partial_line = ""

        try:
            while True:
                # Read one line at a time.
                line_bytes = await proc.stdout.readline()
                if not line_bytes:
                    # EOF
                    break

                if len(line_bytes) > _MAX_LINE_BYTES:
                    logger.warning(
                        "%s: stdout line exceeds %d bytes, skipping",
                        self.name,
                        _MAX_LINE_BYTES,
                    )
                    continue

                decoded = line_bytes.decode(errors="replace")
                line_str = (partial_line + decoded).strip()
                partial_line = ""

                if not line_str:
                    continue

                # Try to parse as JSON.
                try:
                    event = json.loads(line_str)
                except json.JSONDecodeError as exc:
                    # Check if this looks like incomplete JSON (doesn't end with newline from original).
                    # If the original line ended with \n, it's complete but malformed.
                    if decoded.endswith("\n"):
                        # Complete line but invalid JSON — log and skip.
                        logger.warning(
                            "%s: malformed JSON from Claude stdout: %s",
                            self.name,
                            line_str[:200],
                        )
                        continue
                    else:
                        # No newline yet — might be incomplete, save for next iteration.
                        partial_line = line_str
                        continue

                # Dispatch the parsed event.
                result_text = await self._dispatch_claude_event(event, result_text)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("%s: error reading Claude stdout: %s", self.name, exc)

        return result_text

    async def _dispatch_claude_event(self, event: dict[str, object], current_result: str) -> str:
        """Process a single streaming event from Claude CLI.

        Returns updated result text.
        """
        event_type = event.get("type")

        if event_type == "text":
            # Text chunk from Claude's response.
            text = str(event.get("text", ""))
            if text:
                self._recorder.record(
                    CLITextChunkEvent(ts="", seq=0, agent=self.name, text=text)
                )
                current_result += text

        elif event_type == "tool_use":
            # Claude is calling a tool.
            tool_name = str(event.get("name", ""))
            tool_args = event.get("input", {})
            if not isinstance(tool_args, dict):
                tool_args = {}
            self._recorder.record(
                CLIToolCallEvent(
                    ts="", seq=0, agent=self.name, tool=tool_name, args=tool_args
                )
            )

        elif event_type == "thinking":
            # Claude's internal reasoning.
            content = str(event.get("content", ""))
            if content:
                self._recorder.record(
                    CLIThinkingEvent(ts="", seq=0, agent=self.name, content=content)
                )

        elif event_type == "status":
            # Progress update.
            status = str(event.get("status", ""))
            if status:
                self._recorder.record(
                    CLIProgressEvent(ts="", seq=0, agent=self.name, status=status)
                )

        elif event_type == "conversation_id":
            # Store conversation ID for --continue.
            conv_id = str(event.get("id", ""))
            if conv_id and conv_id.replace("-", "").replace("_", "").isalnum():
                self._claude_conversation_id = conv_id
            else:
                logger.warning(
                    "%s: invalid conversation_id format '%s', ignoring",
                    self.name,
                    conv_id[:100],
                )

        elif event_type == "result":
            # Final result — might be redundant with accumulated text.
            result = str(event.get("content", ""))
            if result:
                current_result = result

        # Ignore other event types.
        return current_result

    async def _process_message_queue(self) -> None:
        """Process queued messages sequentially as follow-ups."""
        while not self._message_queue.empty():
            from_agent, content = await self._message_queue.get()

            logger.info(
                "%s: processing queued message from %s (queue size: %d)",
                self.name,
                from_agent,
                self._message_queue.qsize(),
            )

            # Show user feedback when starting a queued message.
            if self._on_response:
                self._on_response(f"[{self.name} processing queued message from {from_agent}]")

            # Record start/done events for each queued task.
            self._recorder.record(
                AgentStartEvent(ts="", seq=0, agent=self.name, agent_type="cli")
            )
            await self._handle_claude_task(from_agent, content, is_followup=True)
            self._recorder.record(
                AgentDoneEvent(ts="", seq=0, agent=self.name, reason="completed")
            )

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
        loop = asyncio.get_running_loop()
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

                if len(line) > _MAX_LINE_BYTES:
                    logger.warning(
                        "%s: subprocess line exceeds %d bytes, skipping",
                        self.name,
                        _MAX_LINE_BYTES,
                    )
                    continue

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
                if to not in self._peer_names:
                    logger.warning(
                        "%s: subprocess tried to message non-peer '%s' -- blocked",
                        self.name,
                        to,
                    )
                    return
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
            future = self._pending_tasks.pop(task_id, None)
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
