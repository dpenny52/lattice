"""Tool definitions and registry for LLM agents."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lattice.agent.builtin_tools import (
    BUILTIN_TOOL_SCHEMAS,
    handle_code_exec,
    handle_file_read,
    handle_file_write,
    handle_web_search,
)
from lattice.session.models import StatusEvent, ToolCallEvent, ToolResultEvent

if TYPE_CHECKING:
    from lattice.router.router import Router
    from lattice.session.recorder import SessionRecorder

logger = logging.getLogger(__name__)

#: Timeout (seconds) for blocking send_message with wait_for_reply=true.
_SEND_MESSAGE_TIMEOUT = 300.0


SEND_MESSAGE_TOOL: dict[str, Any] = {
    "name": "send_message",
    "description": (
        "Send a message to another agent on your team. "
        "By default, waits for the target agent to reply and returns their "
        "response. Set wait_for_reply to false for fire-and-forget."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "Name of the agent to message",
            },
            "content": {
                "type": "string",
                "description": "The message content",
            },
            "wait_for_reply": {
                "type": "boolean",
                "description": (
                    "If true (default), block until the target agent replies "
                    "and return their response. If false, send and return immediately."
                ),
                "default": True,
            },
        },
        "required": ["to", "content"],
    },
}


def _is_mcp_reference(tool: str | dict[str, object]) -> bool:
    """Check if a tool entry is an MCP server reference."""
    if isinstance(tool, dict):
        return "mcp" in tool
    return tool.startswith("mcp:")


class ToolRegistry:
    """Holds tool definitions and executes tool calls.

    The ``send_message`` tool is always registered automatically. Additional
    built-in tools are registered based on the agent's ``tools`` config list.
    MCP references are recognised but deferred (logged and skipped).
    """

    def __init__(
        self,
        agent_name: str,
        router: Router,
        recorder: SessionRecorder,
        configured_tools: list[str | dict[str, object]] | None = None,
        working_dir: Path | None = None,
        allowed_paths: list[Path] | None = None,
    ) -> None:
        self._agent_name = agent_name
        self._router = router
        self._recorder = recorder
        self._working_dir = working_dir or Path.cwd()
        self._allowed_paths = allowed_paths or []

        # Always include send_message.
        self._tools: dict[str, dict[str, Any]] = {"send_message": SEND_MESSAGE_TOOL}

        # Register configured built-in tools.
        for tool in configured_tools or []:
            if _is_mcp_reference(tool):
                ref = tool if isinstance(tool, str) else tool.get("mcp", tool)
                logger.warning(
                    "MCP tool reference %r recognised but deferred (not yet supported)",
                    ref,
                )
                continue

            name = tool if isinstance(tool, str) else str(tool)
            if name in BUILTIN_TOOL_SCHEMAS:
                self._tools[name] = BUILTIN_TOOL_SCHEMAS[name]
            else:
                logger.warning("Unknown tool %r requested -- skipping", name)

    @property
    def definitions(self) -> list[dict[str, Any]]:
        """Return all tool schemas for the LLM tool-use interface."""
        return list(self._tools.values())

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name and return the result as a string.

        Records ``tool_call`` and ``tool_result`` events.
        """
        self._recorder.record(
            ToolCallEvent(
                ts="",
                seq=0,
                agent=self._agent_name,
                tool=name,
                args=arguments,
            )
        )

        start = time.monotonic()
        try:
            result = await self._dispatch(name, arguments)
        except Exception as exc:
            result = f"Error: {exc}"

        elapsed_ms = int((time.monotonic() - start) * 1000)
        self._recorder.record(
            ToolResultEvent(
                ts="",
                seq=0,
                agent=self._agent_name,
                tool=name,
                duration_ms=elapsed_ms,
                result_size=len(result.encode()),
            )
        )

        return result

    async def _dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """Route to the correct handler."""
        if name == "send_message":
            return await self._handle_send_message(arguments)

        if name == "web-search":
            return await handle_web_search(arguments)

        if name == "file-read":
            return await handle_file_read(
                arguments, self._working_dir, self._allowed_paths,
            )

        if name == "file-write":
            return await handle_file_write(
                arguments, self._working_dir, self._allowed_paths,
            )

        if name == "code-exec":
            return await handle_code_exec(arguments)

        msg = f"Unknown tool: {name}"
        raise ValueError(msg)

    async def _handle_send_message(self, arguments: dict[str, Any]) -> str:
        """Dispatch a message through the router.

        When ``wait_for_reply`` is true (default), registers a response
        channel and blocks until the target agent replies.
        """
        to = arguments.get("to", "")
        content = arguments.get("content", "")
        wait = arguments.get("wait_for_reply", True)

        if not wait:
            # Fire-and-forget: original behavior.
            await self._router.send(self._agent_name, to, content)
            return json.dumps({"status": "sent", "to": to})

        # Blocking mode: register a response channel before sending.
        future = self._router.expect_response(
            from_agent=to, to_agent=self._agent_name
        )
        try:
            await self._router.send(self._agent_name, to, content)
        except Exception:
            self._router.cancel_response(from_agent=to, to_agent=self._agent_name)
            raise

        # Record a status event so watch TUI shows the waiting state.
        self._recorder.record(
            StatusEvent(
                ts="", seq=0, agent=self._agent_name, status=f"waiting for {to}"
            )
        )

        try:
            response = await asyncio.wait_for(future, timeout=_SEND_MESSAGE_TIMEOUT)
        except TimeoutError:
            self._router.cancel_response(from_agent=to, to_agent=self._agent_name)
            return json.dumps({
                "status": "error",
                "error": f"Timed out waiting for reply from '{to}' "
                         f"after {_SEND_MESSAGE_TIMEOUT:.0f}s",
            })

        return json.dumps({
            "status": "reply_received",
            "from": to,
            "content": response,
        })
