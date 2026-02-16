"""Core LLM agent -- the runtime for ``type: llm`` agents."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import click

from lattice.agent.providers import LLMProvider, LLMResponse, create_provider
from lattice.agent.tools import ToolRegistry
from lattice.router.router import Router
from lattice.session.models import (
    AgentDoneEvent,
    AgentResponseEvent,
    AgentStartEvent,
    ErrorEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
)
from lattice.session.recorder import SessionRecorder

logger = logging.getLogger(__name__)

#: Maximum retries for transient LLM API errors.
_MAX_RETRIES = 3

#: Base delay (seconds) for exponential backoff.
_BASE_DELAY = 1.0

#: Maximum tool-call loop iterations before the agent bails out.
_MAX_LOOP_ITERATIONS = 200

#: Default pause duration (seconds) when a 429 rate limit is hit.
_RATE_LIMIT_PAUSE = 60.0

#: Tool results above this size (chars) are compacted after the LLM processes them.
_TOOL_RESULT_COMPACT_THRESHOLD = 1000


class RateLimitGate:
    """Shared gate that pauses all LLM calls after any agent hits a 429.

    Create one instance and pass it to every ``LLMAgent``.  When any agent
    encounters a rate-limit error, it calls ``pause()`` which causes all
    agents to wait before their next LLM call.
    """

    def __init__(self, pause_seconds: float = _RATE_LIMIT_PAUSE) -> None:
        self._pause_seconds = pause_seconds
        self._resume_at: float = 0.0

    async def wait_if_paused(self) -> None:
        """Block until the rate-limit cooldown expires (if active)."""
        now = time.monotonic()
        if now < self._resume_at:
            delay = self._resume_at - now
            logger.info("Rate limit gate: all LLM calls paused for %.1fs", delay)
            await asyncio.sleep(delay)

    def pause(self) -> None:
        """Activate the gate — all agents will wait before their next call."""
        self._resume_at = max(
            self._resume_at,
            time.monotonic() + self._pause_seconds,
        )
        pause_secs = int(self._pause_seconds)
        click.echo(
            f"⚠️  Rate limit hit — pausing all LLM calls for {pause_secs}s...",
            err=True,
        )
        logger.warning(
            "Rate limit hit — pausing all LLM calls for %.0fs",
            self._pause_seconds,
        )


#: Pattern matching common API key formats to redact from error messages.
_API_KEY_RE = re.compile(
    r"(sk-[a-zA-Z0-9]{20,}|key-[a-zA-Z0-9]{20,}|AIza[a-zA-Z0-9_-]{30,})",
)


def _sanitize_error(exc: BaseException) -> str:
    """Return a string representation of *exc* with API keys redacted."""
    return _API_KEY_RE.sub("[REDACTED]", str(exc))


class LLMAgent:
    """An async agent that wraps an LLM provider.

    Satisfies the ``Router.Agent`` protocol via ``handle_message``.

    All messages (from any peer) share a single conversation thread.
    Messages are prefixed with ``[from <agent>]`` so the LLM knows who
    is talking.
    """

    def __init__(
        self,
        name: str,
        model_string: str,
        role: str,
        router: Router,
        recorder: SessionRecorder,
        team_name: str,
        peer_names: list[str],
        credentials: dict[str, str] | None = None,
        provider: LLMProvider | None = None,
        model_override: str | None = None,
        configured_tools: list[str | dict[str, object]] | None = None,
        allowed_paths: list[str] | None = None,
        on_response: Callable[[str], Awaitable[None]] | None = None,
        rate_gate: RateLimitGate | None = None,
    ) -> None:
        self.name = name
        self._role = role
        self._router = router
        self._recorder = recorder
        self._team_name = team_name
        self._peer_names = peer_names
        self._on_response = on_response
        self._rate_gate = rate_gate

        # Single unified conversation thread.
        self._thread: list[dict[str, Any]] = []

        # Provider + model.
        if provider is not None:
            self._provider = provider
            self._model = model_override or model_string
        else:
            self._provider, self._model = create_provider(model_string, credentials)

        # Agent lifecycle state — used by _compute_mood().
        self._state: str = "idle"

        # Tool registry.
        resolved_paths = [Path(p).expanduser() for p in (allowed_paths or [])]
        self._tools = ToolRegistry(
            name,
            router,
            recorder,
            configured_tools=configured_tools,
            allowed_paths=resolved_paths or None,
        )

    # ------------------------------------------------------------------ #
    # Agent protocol
    # ------------------------------------------------------------------ #

    async def handle_message(self, from_agent: str, content: str) -> None:
        """Handle an incoming message from *from_agent*.

        This is the main agent loop:
        1. Append the message to the unified thread (prefixed with sender).
        2. Call the LLM.
        3. Process tool calls or return on plain text.
        """
        self._state = "thinking"
        self._recorder.record(
            AgentStartEvent(ts="", seq=0, agent=self.name, agent_type="llm")
        )

        self._thread.append(
            {
                "role": "user",
                "content": f"[from {from_agent}]: {content}",
            }
        )

        await self._run_loop(self._thread)

        self._recorder.record(
            AgentDoneEvent(ts="", seq=0, agent=self.name, reason="completed")
        )
        # Only go idle if _run_loop didn't already set error state.
        if self._state != "error":
            self._state = "idle"

    async def shutdown(self) -> None:
        """No-op shutdown for protocol consistency."""

    def reset_context(self) -> None:
        """Clear the conversation thread."""
        self._thread.clear()

    def _compute_mood(self) -> str:
        """Return an emoji representing the agent's current mood.

        Mood is derived from ``self._state``:
        - ``"idle"``      → 😴 (sleeping)
        - ``"thinking"``  → 🤔 (thinking)
        - ``"error"``     → 😡 (frustrated)
        - ``"success"``   → 😊 (happy)
        """
        return {
            "idle": "\U0001f634",  # 😴
            "thinking": "\U0001f914",  # 🤔
            "error": "\U0001f621",  # 😡
            "success": "\U0001f60a",  # 😊
        }.get(self._state, "\U0001f634")

    # ------------------------------------------------------------------ #
    # Internal loop
    # ------------------------------------------------------------------ #

    async def _run_loop(self, thread: list[dict[str, Any]]) -> None:
        """Call the LLM in a loop until it returns plain text (no tool calls).

        Enforces a hard cap of ``_MAX_LOOP_ITERATIONS`` to prevent runaway
        tool-call loops (e.g. an LLM that never stops calling tools).
        """
        for _iteration in range(_MAX_LOOP_ITERATIONS):
            # Everything before this index will have been seen after the call.
            pre_call_len = len(thread)

            response = await self._call_llm(thread)
            if response is None:
                # All retries exhausted.
                self._state = "error"
                return

            # The LLM has now processed everything up to pre_call_len.
            # Compact large tool results it has already seen to free memory.
            self._compact_tool_results(thread, up_to=pre_call_len)

            if not response.tool_calls:
                # Plain text response -- agent is done for this turn.
                self._state = "success"
                if response.content:
                    thread.append({"role": "assistant", "content": response.content})
                    self._recorder.record(
                        AgentResponseEvent(
                            ts="",
                            seq=0,
                            agent=self.name,
                            content=response.content,
                        )
                    )
                    if self._on_response is not None:
                        try:
                            await self._on_response(response.content)
                        except Exception:
                            logger.exception(
                                "%s: on_response callback failed",
                                self.name,
                            )
                return

            # Build assistant message with tool calls for the thread.
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": response.content or "",
            }
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in response.tool_calls
            ]
            thread.append(assistant_msg)

            # Execute each tool call and feed results back.
            for tc in response.tool_calls:
                result = await self._tools.execute(tc.name, tc.arguments)
                thread.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": result,
                    }
                )
        else:
            self._state = "error"
            logger.warning(
                "Agent %r hit max loop iterations (%d) -- forcing stop",
                self.name,
                _MAX_LOOP_ITERATIONS,
            )
            self._recorder.record(
                ErrorEvent(
                    ts="",
                    seq=0,
                    agent=self.name,
                    error=f"Max loop iterations ({_MAX_LOOP_ITERATIONS}) exceeded",
                    retrying=False,
                    context="api_call",
                )
            )

    @staticmethod
    def _is_rate_limit_error(exc: BaseException) -> bool:
        """Return True if *exc* looks like a 429 rate-limit response."""
        exc_str = str(exc).lower()
        return "429" in exc_str or "rate_limit" in exc_str or "rate limit" in exc_str

    async def _call_llm(self, thread: list[dict[str, Any]]) -> LLMResponse | None:
        """Call the LLM with retries and exponential backoff."""
        messages = self._build_messages(thread)
        tools = self._tools.definitions

        for attempt in range(1, _MAX_RETRIES + 1):
            # Wait if another agent triggered the rate-limit gate.
            if self._rate_gate is not None:
                await self._rate_gate.wait_if_paused()

            self._recorder.record(
                LLMCallStartEvent(
                    ts="",
                    seq=0,
                    agent=self.name,
                    model=self._model,
                    messages_count=len(messages),
                )
            )

            start = time.monotonic()
            try:
                response = await self._provider.chat(messages, tools, self._model)
            except Exception as exc:
                elapsed_ms = int((time.monotonic() - start) * 1000)
                retrying = attempt < _MAX_RETRIES
                safe_error = _sanitize_error(exc)

                # Generate user-friendly error message
                provider = (
                    self._model.split("/")[0] if "/" in self._model else "provider"
                )
                if self._is_rate_limit_error(exc):
                    pause = int(_RATE_LIMIT_PAUSE)
                    if retrying:
                        user_msg = (
                            f"Agent '{self.name}' got a 429"
                            f" from {provider} (rate limited)."
                            f" Retrying in {pause}s..."
                        )
                    else:
                        user_msg = (
                            f"Agent '{self.name}' got a 429"
                            f" from {provider} (rate limited)."
                        )
                    click.echo(user_msg, err=True)
                elif (
                    "401" in safe_error
                    or "unauthorized" in safe_error.lower()
                    or "api key" in safe_error.lower()
                    or "authentication" in safe_error.lower()
                ):
                    msg_401 = (
                        f"Agent '{self.name}' got a 401"
                        f" from {provider} (authentication"
                        " failed). Check your API key."
                    )
                    click.echo(msg_401, err=True)
                elif (
                    "500" in safe_error
                    or "internal server" in safe_error.lower()
                    or "service unavailable" in safe_error.lower()
                ):
                    if retrying:
                        delay = _BASE_DELAY * (2 ** (attempt - 1))
                        user_msg = (
                            f"Agent '{self.name}' got a 500"
                            f" from {provider} (server error)."
                            f" Retrying in {delay:.0f}s..."
                        )
                    else:
                        user_msg = (
                            f"Agent '{self.name}' got a 500"
                            f" from {provider} (server error)."
                        )
                    click.echo(user_msg, err=True)
                elif (
                    "connection" in safe_error.lower()
                    or "network" in safe_error.lower()
                    or "timeout" in safe_error.lower()
                    or "dns" in safe_error.lower()
                ):
                    if retrying:
                        delay = _BASE_DELAY * (2 ** (attempt - 1))
                        user_msg = (
                            f"Agent '{self.name}'"
                            f" — network error: {safe_error}."
                            f" Retrying in {delay:.0f}s..."
                        )
                    else:
                        user_msg = f"Agent '{self.name}' — network error: {safe_error}"
                    click.echo(user_msg, err=True)
                else:
                    user_msg = f"Agent '{self.name}' — LLM call failed: {safe_error}"
                    if retrying:
                        delay = _BASE_DELAY * (2 ** (attempt - 1))
                        user_msg += f". Retrying in {delay:.0f}s..."
                    click.echo(user_msg, err=True)

                logger.warning(
                    "LLM call failed (attempt %d/%d): %s",
                    attempt,
                    _MAX_RETRIES,
                    safe_error,
                )
                self._recorder.record(
                    ErrorEvent(
                        ts="",
                        seq=0,
                        agent=self.name,
                        error=safe_error,
                        retrying=retrying,
                        context="api_call",
                    )
                )
                self._recorder.record(
                    LLMCallEndEvent(
                        ts="",
                        seq=0,
                        agent=self.name,
                        model=self._model,
                        tokens={"input": 0, "output": 0},
                        duration_ms=elapsed_ms,
                    )
                )
                if retrying:
                    if self._rate_gate is not None and self._is_rate_limit_error(exc):
                        self._rate_gate.pause()
                        await self._rate_gate.wait_if_paused()
                    else:
                        delay = _BASE_DELAY * (2 ** (attempt - 1))
                        await asyncio.sleep(delay)
                    continue
                return None

            elapsed_ms = int((time.monotonic() - start) * 1000)
            self._recorder.record(
                LLMCallEndEvent(
                    ts="",
                    seq=0,
                    agent=self.name,
                    model=self._model,
                    tokens={
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens,
                    },
                    duration_ms=elapsed_ms,
                )
            )
            return response

        return None  # Should not be reached, but satisfies type checker.

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compact_tool_results(
        thread: list[dict[str, Any]],
        *,
        up_to: int,
    ) -> None:
        """Replace large tool results with compact stubs in-place.

        Only touches entries before index *up_to* — those have already been
        processed by the LLM and no longer need their full content.
        """
        for i in range(up_to):
            entry = thread[i]
            if entry.get("role") != "tool":
                continue
            content = entry.get("content", "")
            if len(content) <= _TOOL_RESULT_COMPACT_THRESHOLD:
                continue
            name = entry.get("name", "unknown")
            orig_len = len(content)
            entry["content"] = (
                f"[{name} result: {orig_len:,} chars — compacted after processing]"
            )

    def _build_messages(self, thread: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Prepend the system prompt to the conversation thread."""
        system_prompt = self._build_system_prompt()
        return [{"role": "system", "content": system_prompt}, *thread]

    def _build_system_prompt(self) -> str:
        """Combine the agent's role with team context."""
        parts = [self._role]
        if self._peer_names:
            peer_list = ", ".join(self._peer_names)
            parts.append(
                f'\nYou are part of a team called "{self._team_name}".\n'
                f"You can communicate with: {peer_list}\n"
                "Use the send_message tool to talk to them.\n"
                "send_message is fire-and-forget — it dispatches your message "
                "and returns immediately. Replies from other agents will arrive "
                "later as new messages prefixed with [from agent_name]. "
                "This is normal; do not consider yourself stuck while waiting."
            )
        return "\n".join(parts)
