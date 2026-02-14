"""Core LLM agent -- the runtime for ``type: llm`` agents."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from typing import Any

from lattice.agent.providers import LLMProvider, LLMResponse, create_provider
from lattice.agent.tools import ToolRegistry
from lattice.router.router import Router
from lattice.session.models import (
    AgentDoneEvent,
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
_MAX_LOOP_ITERATIONS = 20

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

    Each peer gets its own conversation thread -- messages from different
    peers never pollute each other's context.
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
        on_response: Callable[[str], None] | None = None,
    ) -> None:
        self.name = name
        self._role = role
        self._router = router
        self._recorder = recorder
        self._team_name = team_name
        self._peer_names = peer_names
        self._on_response = on_response

        # Per-peer conversation threads.
        self._threads: dict[str, list[dict[str, Any]]] = {}

        # Provider + model.
        if provider is not None:
            self._provider = provider
            self._model = model_override or model_string
        else:
            self._provider, self._model = create_provider(model_string, credentials)

        # Tool registry.
        self._tools = ToolRegistry(
            name, router, recorder, configured_tools=configured_tools,
        )

    # ------------------------------------------------------------------ #
    # Agent protocol
    # ------------------------------------------------------------------ #

    async def handle_message(self, from_agent: str, content: str) -> None:
        """Handle an incoming message from *from_agent*.

        This is the main agent loop:
        1. Append the message to the per-peer thread.
        2. Call the LLM.
        3. Process tool calls or return on plain text.
        """
        self._recorder.record(
            AgentStartEvent(ts="", seq=0, agent=self.name, agent_type="llm")
        )

        thread = self._get_thread(from_agent)
        thread.append({"role": "user", "content": content})

        await self._run_loop(thread)

        self._recorder.record(
            AgentDoneEvent(ts="", seq=0, agent=self.name, reason="completed")
        )

    async def shutdown(self) -> None:
        """No-op shutdown for protocol consistency."""

    # ------------------------------------------------------------------ #
    # Internal loop
    # ------------------------------------------------------------------ #

    async def _run_loop(self, thread: list[dict[str, Any]]) -> None:
        """Call the LLM in a loop until it returns plain text (no tool calls).

        Enforces a hard cap of ``_MAX_LOOP_ITERATIONS`` to prevent runaway
        tool-call loops (e.g. an LLM that never stops calling tools).
        """
        for _iteration in range(_MAX_LOOP_ITERATIONS):
            response = await self._call_llm(thread)
            if response is None:
                # All retries exhausted.
                return

            if not response.tool_calls:
                # Plain text response -- agent is done for this turn.
                if response.content:
                    thread.append({"role": "assistant", "content": response.content})
                    if self._on_response is not None:
                        self._on_response(response.content)
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
                )
            )

    async def _call_llm(self, thread: list[dict[str, Any]]) -> LLMResponse | None:
        """Call the LLM with retries and exponential backoff."""
        messages = self._build_messages(thread)
        tools = self._tools.definitions

        for attempt in range(1, _MAX_RETRIES + 1):
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
                "Use the send_message tool to talk to them."
            )
        return "\n".join(parts)

    def _get_thread(self, peer: str) -> list[dict[str, Any]]:
        """Return or create the conversation thread for *peer*."""
        if peer not in self._threads:
            self._threads[peer] = []
        return self._threads[peer]
