"""Tool definitions and registry for LLM agents."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from lattice.session.models import ToolCallEvent, ToolResultEvent

if TYPE_CHECKING:
    from lattice.router.router import Router
    from lattice.session.recorder import SessionRecorder


SEND_MESSAGE_TOOL: dict[str, Any] = {
    "name": "send_message",
    "description": "Send a message to another agent on your team",
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


class ToolRegistry:
    """Holds tool definitions and executes tool calls.

    The ``send_message`` tool is always registered automatically. Additional
    built-in tools (web-search, file-read, etc.) will be added in Story 2.4.
    """

    def __init__(
        self,
        agent_name: str,
        router: Router,
        recorder: SessionRecorder,
    ) -> None:
        self._agent_name = agent_name
        self._router = router
        self._recorder = recorder

    @property
    def definitions(self) -> list[dict[str, Any]]:
        """Return all tool schemas for the LLM tool-use interface."""
        return [SEND_MESSAGE_TOOL]

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

        msg = f"Unknown tool: {name}"
        raise ValueError(msg)

    async def _handle_send_message(self, arguments: dict[str, Any]) -> str:
        """Dispatch a message through the router."""
        to = arguments.get("to", "")
        content = arguments.get("content", "")
        await self._router.send(self._agent_name, to, content)
        return json.dumps({"status": "sent", "to": to})
