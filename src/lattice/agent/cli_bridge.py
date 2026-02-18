"""CLI Bridge agent — spawns CLI tools as subprocesses with JSONL communication."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import shlex
from collections.abc import Awaitable, Callable

import click

from lattice.agent.helpers import format_stderr_preview, record_error
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

#: Env vars stripped from CLI subprocesses so they use subscription auth.
_STRIPPED_ENV_KEYS = {"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY"}

#: Max V8 heap size (MB) for Node.js CLI subprocesses (e.g. Claude CLI).
#: Prevents a single agent from OOM-killing the entire process tree.
_NODE_HEAP_LIMIT_MB = 2048

#: Minimum available system memory (MB) required to spawn a CLI subprocess.
#: If below this, the agent refuses to start and reports an error.
_MIN_AVAILABLE_MB = 1024


class CLIBridge:
    """Agent that wraps a CLI subprocess with bidirectional JSONL.

    Satisfies the ``Router.Agent`` protocol via ``handle_message``.

    Three modes of operation:

    * **Claude adapter** (``cli_type="claude"``): spawns a new ``claude``
      subprocess per task, collects JSON output, and routes results.
    * **Codex adapter** (``cli_type="codex"``): spawns a new ``codex``
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
        on_response: Callable[[str], Awaitable[None]] | None = None,
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

        # Codex adapter: thread state for resume.
        self._codex_thread_id: str | None = None

        # Per-agent memory profiling: current CLI subprocess PID.
        self._current_claude_pid: int | None = None

    @property
    def current_subprocess_pid(self) -> int | None:
        """PID of the currently running subprocess, if any.

        For Claude/Codex adapter: the PID of the active CLI subprocess.
        For custom CLI: the PID of the long-running process.
        Returns ``None`` when no subprocess is running.
        """
        # Custom CLI — long-running process.
        if self._process is not None and self._process.returncode is None:
            return self._process.pid
        # Per-task CLI adapter (Claude/Codex).
        return self._current_claude_pid

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

        if self._cli_type in ("claude", "codex"):
            # Per-task adapters don't use a long-running process.
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
            start_new_session=True,
        )
        self._started = True
        self._read_task = asyncio.create_task(self._read_loop())

    async def shutdown(self) -> None:
        """Graceful shutdown: signal -> wait -> SIGTERM -> SIGKILL."""
        if self._cli_type in ("claude", "codex"):
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

        if self._cli_type in ("claude", "codex"):
            # Atomically check busy state to prevent race conditions.
            async with self._claude_busy_lock:
                if self._claude_busy:
                    # Check if queue has space before adding.
                    if self._message_queue.full():
                        error_msg = (
                            "Message queue full (100 messages)"
                            f" — rejecting message from {from_agent}"
                        )
                        record_error(
                            self._recorder,
                            self.name,
                            error_msg,
                            logger=logger,
                        )
                        if self._on_response:
                            msg = f"[{self.name} queue full, message rejected]"
                            await self._on_response(msg)
                        return

                    # Queue the message and show feedback.
                    await self._message_queue.put((from_agent, content))
                    if self._on_response:
                        msg = f"[{self.name} is busy, message queued]"
                        await self._on_response(msg)
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
            if self._cli_type == "claude":
                await self._handle_claude_task(from_agent, content)
            else:
                await self._handle_codex_task(from_agent, content)
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
        try:
            await self._run_claude_task(from_agent, content, is_followup=is_followup)
        finally:
            self._current_claude_pid = None
            self._claude_busy = False

    async def _run_claude_task(
        self, from_agent: str, content: str, *, is_followup: bool = False
    ) -> None:
        """Inner implementation of a Claude CLI task (called by _handle_claude_task)."""
        # Pre-flight memory check — refuse to spawn if system is too low.
        from lattice.memory_monitor import get_available_mb

        available = get_available_mb()
        if available is not None and available < _MIN_AVAILABLE_MB:
            error_msg = (
                f"Agent '{self.name}' — not enough memory to spawn CLI subprocess. "
                f"Available: {available:.0f} MB, required: {_MIN_AVAILABLE_MB} MB. "
                f"Close other applications to free memory."
            )
            click.echo(error_msg, err=True)
            logger.error("%s: %s", self.name, error_msg)
            self._recorder.record(
                ErrorEvent(
                    ts="",
                    seq=0,
                    agent=self.name,
                    error=f"Insufficient memory: {available:.0f}MB available",
                    retrying=False,
                    context="subprocess",
                )
            )
            if self._on_response:
                msg = (
                    f"[{self.name}: insufficient memory ({available:.0f} MB available)]"
                )
                await self._on_response(msg)
            return

        # Include role only in the first message; follow-ups use
        # the preserved conversation.
        if is_followup:
            prompt = f"Task from {from_agent}: {content}"
        else:
            prompt = f"{self._role}\n\nTask from {from_agent}: {content}"

        try:
            cmd_args = [
                "claude",
                "-p",
                prompt,
                "--output-format",
                "stream-json",
                "--verbose",
            ]

            # Use --continue flag for follow-up messages to preserve context.
            if is_followup and self._claude_conversation_id is not None:
                cmd_args.extend(["--continue", self._claude_conversation_id])

            cmd_args.append("--dangerously-skip-permissions")

            # Strip LLM API keys so the CLI uses subscription auth
            # instead of accidentally hitting the API on the user's key.
            # Cap Node.js V8 heap to prevent OOM-killing the process tree.
            cli_env = {
                k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV_KEYS
            }
            node_opts = cli_env.get("NODE_OPTIONS", "")
            if "--max-old-space-size" not in node_opts:
                separator = " " if node_opts else ""
                heap_flag = f"--max-old-space-size={_NODE_HEAP_LIMIT_MB}"
                cli_env["NODE_OPTIONS"] = f"{node_opts}{separator}{heap_flag}"

            proc = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_MAX_LINE_BYTES,
                env=cli_env,
                start_new_session=True,
            )
            self._current_claude_pid = proc.pid
        except FileNotFoundError:
            error_msg = (
                f"Agent '{self.name}' — Claude CLI not found.\n"
                "Make sure 'claude' is installed and on your PATH.\n"
                "Install: npm install -g @anthropic-ai/claude-code"
            )
            click.echo(error_msg, err=True)
            record_error(
                self._recorder,
                self.name,
                "Claude CLI not found",
                logger=logger,
            )
            return
        except OSError as exc:
            error_msg = f"Agent '{self.name}' — failed to spawn Claude CLI: {exc}"
            click.echo(error_msg, err=True)
            logger.error("%s: failed to spawn Claude CLI: %s", self.name, exc)
            self._recorder.record(
                ErrorEvent(
                    ts="",
                    seq=0,
                    agent=self.name,
                    error=f"Failed to spawn Claude CLI: {exc}",
                    retrying=False,
                    context="subprocess",
                )
            )
            return

        # Stream stdout and parse events as they arrive.
        result_text = await self._stream_claude_output(proc)

        # Drain any remaining stdout/stderr so the subprocess can exit.
        # Without this, the process can deadlock if it's blocked writing
        # to a full pipe buffer (e.g. after a buffer-overflow skip).
        try:
            if proc.stdout and not proc.stdout.at_eof():
                await proc.stdout.read()
        except Exception:
            pass
        stderr_bytes = b""
        try:
            if proc.stderr:
                stderr_bytes = await proc.stderr.read()
        except Exception:
            pass

        # Wait for process to complete.
        returncode = await proc.wait()

        if returncode != 0:
            stderr_text = stderr_bytes.decode(errors="replace").strip()

            # Show last 5 lines of stderr for user-friendly output
            stderr_preview = format_stderr_preview(stderr_text)

            error_msg = f"Agent '{self.name}' exited with code {returncode}."
            if stderr_preview:
                error_msg += f" Stderr:\n  {stderr_preview}"

            click.echo(error_msg, err=True)

            # Record full stderr in session log
            full_error = (
                f"Claude CLI exited with code {returncode}: {stderr_text[:2048]}"
            )
            record_error(self._recorder, self.name, full_error, logger=logger)
            return

        if result_text:
            if self._on_response:
                await self._on_response(result_text)
            # Route the result back to the sender so the conversation continues.
            # Skip routing back to "user" — that's handled by on_response.
            if from_agent != "user":
                await self._router.send(self.name, from_agent, result_text)

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
                try:
                    line_bytes = await proc.stdout.readline()
                except ValueError:
                    # Line exceeded StreamReader buffer limit — skip and keep reading.
                    logger.warning(
                        "%s: stdout line exceeded buffer limit, skipping",
                        self.name,
                    )
                    partial_line = ""
                    continue

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
                except json.JSONDecodeError:
                    # Check if this looks like incomplete JSON
                    # (doesn't end with newline from original).
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

    async def _dispatch_claude_event(
        self,
        event: dict[str, object],
        current_result: str,
    ) -> str:
        """Process a single streaming event from Claude CLI.

        Claude CLI ``--output-format stream-json --verbose`` emits these
        top-level event types:

        * ``system``   — init event with session_id, tools, etc.
        * ``assistant`` — wraps an API message; content blocks are nested
          inside ``message.content[]`` as ``text``, ``tool_use``, or
          ``thinking`` blocks.
        * ``user``     — tool results fed back to the model.
        * ``result``   — final aggregated result with ``result`` field.

        Returns updated result text.
        """
        event_type = event.get("type")

        if event_type == "system":
            # Init event — extract session_id for --continue.
            session_id = event.get("session_id")
            if isinstance(session_id, str) and session_id:
                self._claude_conversation_id = session_id
            self._recorder.record(
                CLIProgressEvent(ts="", seq=0, agent=self.name, status="initialized")
            )

        elif event_type == "assistant":
            # Assistant message — extract content blocks.
            # We only record events here (for the watch TUI / session log).
            # The actual result text comes from the final "result" event,
            # which is Claude's clean summary — NOT the full transcript of
            # every text block (which can be enormous for long sessions).
            message = event.get("message")
            if isinstance(message, dict):
                content_blocks = message.get("content")
                if isinstance(content_blocks, list):
                    for block in content_blocks:
                        if not isinstance(block, dict):
                            continue
                        self._dispatch_content_block(block)

        elif event_type == "result":
            # Final result — use the aggregated result text.
            result = event.get("result")
            if isinstance(result, str) and result:
                current_result = result

        # Ignore "user" (tool result echoes) and unknown types.
        return current_result

    def _dispatch_content_block(self, block: dict[str, object]) -> None:
        """Process a single content block from an assistant message.

        Records events for the session log / watch TUI only.
        Does NOT accumulate text — the final result comes from the
        ``result`` event emitted by Claude CLI at the end of the session.
        """
        block_type = block.get("type")

        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                self._recorder.record(
                    CLITextChunkEvent(ts="", seq=0, agent=self.name, text=text)
                )

        elif block_type == "tool_use":
            tool_name = str(block.get("name", ""))
            tool_args = block.get("input", {})
            if not isinstance(tool_args, dict):
                tool_args = {}
            self._recorder.record(
                CLIToolCallEvent(
                    ts="", seq=0, agent=self.name, tool=tool_name, args=tool_args
                )
            )

        elif block_type == "thinking":
            content = block.get("thinking")
            if isinstance(content, str) and content:
                self._recorder.record(
                    CLIThinkingEvent(ts="", seq=0, agent=self.name, content=content)
                )

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
                msg = f"[{self.name} processing queued message from {from_agent}]"
                await self._on_response(msg)

            # Record start/done events for each queued task.
            self._recorder.record(
                AgentStartEvent(ts="", seq=0, agent=self.name, agent_type="cli")
            )
            if self._cli_type == "claude":
                await self._handle_claude_task(from_agent, content, is_followup=True)
            elif self._cli_type == "codex":
                await self._handle_codex_task(from_agent, content, is_followup=True)
            self._recorder.record(
                AgentDoneEvent(ts="", seq=0, agent=self.name, reason="completed")
            )

    # ------------------------------------------------------------------ #
    # Codex adapter
    # ------------------------------------------------------------------ #

    async def _handle_codex_task(
        self, from_agent: str, content: str, *, is_followup: bool = False
    ) -> None:
        """Run a single Codex CLI invocation for this task.

        Args:
            from_agent: Name of the agent sending the message.
            content: Message content.
            is_followup: If True, use resume to preserve conversation context.
        """
        self._claude_busy = True
        try:
            await self._run_codex_task(from_agent, content, is_followup=is_followup)
        finally:
            self._current_claude_pid = None
            self._claude_busy = False

    async def _run_codex_task(
        self, from_agent: str, content: str, *, is_followup: bool = False
    ) -> None:
        """Inner implementation of a Codex CLI task (called by _handle_codex_task)."""
        # Pre-flight memory check — refuse to spawn if system is too low.
        from lattice.memory_monitor import get_available_mb

        available = get_available_mb()
        if available is not None and available < _MIN_AVAILABLE_MB:
            error_msg = (
                f"Agent '{self.name}' — not enough memory to spawn CLI subprocess. "
                f"Available: {available:.0f} MB, required: {_MIN_AVAILABLE_MB} MB. "
                f"Close other applications to free memory."
            )
            click.echo(error_msg, err=True)
            logger.error("%s: %s", self.name, error_msg)
            self._recorder.record(
                ErrorEvent(
                    ts="",
                    seq=0,
                    agent=self.name,
                    error=f"Insufficient memory: {available:.0f}MB available",
                    retrying=False,
                    context="subprocess",
                )
            )
            if self._on_response:
                msg = (
                    f"[{self.name}: insufficient memory ({available:.0f} MB available)]"
                )
                await self._on_response(msg)
            return

        # Include role only in the first message; follow-ups use
        # the preserved conversation.
        if is_followup:
            prompt = f"Task from {from_agent}: {content}"
        else:
            prompt = f"{self._role}\n\nTask from {from_agent}: {content}"

        try:
            if is_followup and self._codex_thread_id is not None:
                cmd_args = [
                    "codex",
                    "exec",
                    "resume",
                    self._codex_thread_id,
                    prompt,
                    "--json",
                    "--dangerously-bypass-approvals-and-sandbox",
                ]
            else:
                cmd_args = [
                    "codex",
                    "exec",
                    prompt,
                    "--json",
                    "--dangerously-bypass-approvals-and-sandbox",
                ]

            # Strip LLM API keys so the CLI uses subscription auth.
            # Codex is a Rust binary — no Node.js heap cap needed.
            cli_env = {
                k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV_KEYS
            }

            proc = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=_MAX_LINE_BYTES,
                env=cli_env,
                start_new_session=True,
            )
            self._current_claude_pid = proc.pid
        except FileNotFoundError:
            error_msg = (
                f"Agent '{self.name}' — Codex CLI not found.\n"
                "Make sure 'codex' is installed and on your PATH.\n"
                "Install: npm install -g @openai/codex"
            )
            click.echo(error_msg, err=True)
            record_error(
                self._recorder,
                self.name,
                "Codex CLI not found",
                logger=logger,
            )
            return
        except OSError as exc:
            error_msg = f"Agent '{self.name}' — failed to spawn Codex CLI: {exc}"
            click.echo(error_msg, err=True)
            logger.error("%s: failed to spawn Codex CLI: %s", self.name, exc)
            self._recorder.record(
                ErrorEvent(
                    ts="",
                    seq=0,
                    agent=self.name,
                    error=f"Failed to spawn Codex CLI: {exc}",
                    retrying=False,
                    context="subprocess",
                )
            )
            return

        # Stream stdout and parse events as they arrive.
        result_text = await self._stream_codex_output(proc)

        # Drain any remaining stdout/stderr so the subprocess can exit.
        try:
            if proc.stdout and not proc.stdout.at_eof():
                await proc.stdout.read()
        except Exception:
            pass
        stderr_bytes = b""
        try:
            if proc.stderr:
                stderr_bytes = await proc.stderr.read()
        except Exception:
            pass

        # Wait for process to complete.
        returncode = await proc.wait()

        if returncode != 0:
            stderr_text = stderr_bytes.decode(errors="replace").strip()

            # Show last 5 lines of stderr for user-friendly output
            stderr_preview = format_stderr_preview(stderr_text)

            error_msg = f"Agent '{self.name}' exited with code {returncode}."
            if stderr_preview:
                error_msg += f" Stderr:\n  {stderr_preview}"

            click.echo(error_msg, err=True)

            # Record full stderr in session log
            full_error = (
                f"Codex CLI exited with code {returncode}: {stderr_text[:2048]}"
            )
            record_error(self._recorder, self.name, full_error, logger=logger)
            return

        if result_text:
            if self._on_response:
                await self._on_response(result_text)
            # Route the result back to the sender so the conversation continues.
            # Skip routing back to "user" — that's handled by on_response.
            if from_agent != "user":
                await self._router.send(self.name, from_agent, result_text)

    async def _stream_codex_output(self, proc: asyncio.subprocess.Process) -> str:
        """Stream and parse JSON events from Codex CLI stdout.

        Returns the final result text (last agent_message).
        """
        if proc.stdout is None:
            return ""

        result_text = ""
        partial_line = ""

        try:
            while True:
                try:
                    line_bytes = await proc.stdout.readline()
                except ValueError:
                    # Line exceeded StreamReader buffer limit — skip and keep reading.
                    logger.warning(
                        "%s: stdout line exceeded buffer limit, skipping",
                        self.name,
                    )
                    partial_line = ""
                    continue

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
                except json.JSONDecodeError:
                    if decoded.endswith("\n"):
                        logger.warning(
                            "%s: malformed JSON from Codex stdout: %s",
                            self.name,
                            line_str[:200],
                        )
                        continue
                    else:
                        partial_line = line_str
                        continue

                # Dispatch the parsed event.
                result_text = await self._dispatch_codex_event(event, result_text)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("%s: error reading Codex stdout: %s", self.name, exc)

        return result_text

    async def _dispatch_codex_event(
        self,
        event: dict[str, object],
        current_result: str,
    ) -> str:
        """Process a single streaming event from Codex CLI.

        Codex CLI ``--json`` emits these top-level event types:

        * ``thread.started`` — init event with thread_id.
        * ``item.started``   — item begins (no-op).
        * ``item.completed`` — item finished; dispatched to
          ``_dispatch_codex_item``.
        * ``turn.completed`` — turn finished (no-op, usage stats).
        * ``turn.failed``    — turn failed with error.
        * ``error``          — top-level error event.

        Returns updated result text.
        """
        event_type = event.get("type")

        if event_type == "thread.started":
            # Init event — extract thread_id for resume.
            thread_id = event.get("thread_id")
            if isinstance(thread_id, str) and thread_id:
                self._codex_thread_id = thread_id
            self._recorder.record(
                CLIProgressEvent(ts="", seq=0, agent=self.name, status="initialized")
            )

        elif event_type == "item.completed":
            # Completed item — dispatch to item handler.
            item = event.get("item")
            if isinstance(item, dict):
                current_result = self._dispatch_codex_item(item, current_result)

        elif event_type == "turn.failed":
            error = event.get("error")
            error_msg = ""
            if isinstance(error, dict):
                error_msg = str(error.get("message", ""))
            elif isinstance(error, str):
                error_msg = error
            if error_msg:
                logger.error("%s: Codex turn failed: %s", self.name, error_msg)
                self._recorder.record(
                    ErrorEvent(
                        ts="",
                        seq=0,
                        agent=self.name,
                        error=f"Codex turn failed: {error_msg}",
                        retrying=False,
                        context="subprocess",
                    )
                )

        elif event_type == "error":
            error_msg = str(event.get("message", ""))
            if error_msg:
                logger.error("%s: Codex error: %s", self.name, error_msg)
                self._recorder.record(
                    ErrorEvent(
                        ts="",
                        seq=0,
                        agent=self.name,
                        error=f"Codex error: {error_msg}",
                        retrying=False,
                        context="subprocess",
                    )
                )

        # Ignore item.started, turn.completed, and unknown types.
        return current_result

    def _dispatch_codex_item(
        self,
        item: dict[str, object],
        current_result: str,
    ) -> str:
        """Process a single completed item from Codex CLI.

        Maps Codex item types to session events:

        * ``agent_message``      → ``CLITextChunkEvent`` (last one = final result).
        * ``reasoning``          → ``CLIThinkingEvent``.
        * ``command_execution``  → ``CLIToolCallEvent`` (tool="bash").
        * ``file_change``        → ``CLIToolCallEvent`` (tool="file_change").
        * ``mcp_tool_call``      → ``CLIToolCallEvent`` (tool from item).
        * ``error``              → ``ErrorEvent``.

        Returns updated result text.
        """
        item_type = item.get("type")

        if item_type == "agent_message":
            text = item.get("text")
            if isinstance(text, str) and text:
                self._recorder.record(
                    CLITextChunkEvent(ts="", seq=0, agent=self.name, text=text)
                )
                current_result = text

        elif item_type == "reasoning":
            text = item.get("text")
            if isinstance(text, str) and text:
                self._recorder.record(
                    CLIThinkingEvent(ts="", seq=0, agent=self.name, content=text)
                )

        elif item_type == "command_execution":
            command = str(item.get("command", ""))
            self._recorder.record(
                CLIToolCallEvent(
                    ts="",
                    seq=0,
                    agent=self.name,
                    tool="bash",
                    args={"command": command},
                )
            )

        elif item_type == "file_change":
            changes = item.get("changes", [])
            if not isinstance(changes, list):
                changes = []
            self._recorder.record(
                CLIToolCallEvent(
                    ts="",
                    seq=0,
                    agent=self.name,
                    tool="file_change",
                    args={"changes": changes},
                )
            )

        elif item_type == "mcp_tool_call":
            tool = str(item.get("tool", "mcp"))
            arguments = item.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}
            self._recorder.record(
                CLIToolCallEvent(
                    ts="",
                    seq=0,
                    agent=self.name,
                    tool=tool,
                    args=arguments,
                )
            )

        elif item_type == "error":
            error_msg = str(item.get("text", ""))
            if error_msg:
                self._recorder.record(
                    ErrorEvent(
                        ts="",
                        seq=0,
                        agent=self.name,
                        error=f"Codex item error: {error_msg}",
                        retrying=False,
                        context="subprocess",
                    )
                )

        return current_result

    # ------------------------------------------------------------------ #
    # Custom CLI (long-running JSONL process)
    # ------------------------------------------------------------------ #

    async def _handle_custom_task(self, from_agent: str, content: str) -> None:
        """Send a task to the long-running subprocess and wait for completion."""
        proc = self._process
        if proc is None or proc.stdin is None:
            error_msg = "Subprocess not running"
            record_error(self._recorder, self.name, error_msg, logger=logger)
            return

        # Generate task ID.
        self._task_counter += 1
        task_id = f"t_{self._task_counter:03d}"

        # Create a future for task completion.
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._pending_tasks[task_id] = future

        # Send task to subprocess stdin.
        task_msg = (
            json.dumps(
                {
                    "type": "task",
                    "id": task_id,
                    "from": from_agent,
                    "content": content,
                }
            )
            + "\n"
        )

        try:
            proc.stdin.write(task_msg.encode())
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            error_msg = f"Failed to write to subprocess: {exc}"
            record_error(self._recorder, self.name, error_msg, logger=logger)
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
                await self._on_response(result)

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
            error_msg = f"Agent '{self.name}' exited with code {returncode}."
            click.echo(error_msg, err=True)
            logger.error("%s: subprocess exited with code %d", self.name, returncode)
            self._recorder.record(
                ErrorEvent(
                    ts="",
                    seq=0,
                    agent=self.name,
                    error=error_msg,
                    retrying=False,
                    context="subprocess",
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
