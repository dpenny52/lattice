"""Tool definitions and registry for LLM agents."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lattice.agent.builtin_tools import (
    BUILTIN_TOOL_SCHEMAS,
    handle_code_exec,
    handle_file_read,
    handle_file_write,
    handle_web_search,
)
from lattice.session.models import ToolCallEvent, ToolResultEvent

if TYPE_CHECKING:
    from lattice.router.router import Router
    from lattice.session.recorder import SessionRecorder

logger = logging.getLogger(__name__)

#: Type alias for async tool handler functions.
ToolHandler = Callable[[dict[str, Any]], Awaitable[str]]

SEND_MESSAGE_TOOL: dict[str, Any] = {
    "name": "send_message",
    "description": (
        "Send a message to another agent on your team (fire-and-forget). "
        "The message is dispatched asynchronously; you will not receive "
        "the reply inline. Replies arrive as new messages in your conversation."
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

        # Handler dispatch table — maps tool name → async handler function.
        self._handlers: dict[str, ToolHandler] = {
            "send_message": self._handle_send_message,
            "web-search": lambda args: handle_web_search(args),
            "file-read": lambda args: handle_file_read(
                args, self._working_dir, self._allowed_paths
            ),
            "file-write": lambda args: handle_file_write(
                args, self._working_dir, self._allowed_paths
            ),
            "code-exec": lambda args: handle_code_exec(args),
        }

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
        """Route to the correct handler via dispatch table."""
        handler = self._handlers.get(name)
        if handler is None:
            msg = f"Unknown tool: {name}"
            raise ValueError(msg)
        return await handler(arguments)

    async def _handle_send_message(self, arguments: dict[str, Any]) -> str:
        """Dispatch a message through the router (fire-and-forget)."""
        to = arguments.get("to", "")
        content = arguments.get("content", "")
        await self._router.send(self._agent_name, to, content)
        return json.dumps({"status": "sent", "to": to})
