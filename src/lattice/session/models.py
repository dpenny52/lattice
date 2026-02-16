"""Pydantic v2 models for session recording events."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag


class _EventBase(BaseModel):
    """Common envelope fields shared by every session event."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    ts: str = Field(description="ISO 8601 timestamp with milliseconds")
    seq: int = Field(ge=0, description="Monotonic sequence number")


class SessionStartEvent(_EventBase):
    """Emitted once at the start of a session."""

    type: Literal["session_start"] = "session_start"
    session_id: str = Field(description="Unique session identifier")
    team: str = Field(description="Team name")
    config_hash: str = Field(description="Hash of the resolved config")


class SessionEndEvent(_EventBase):
    """Emitted once when a session ends."""

    type: Literal["session_end"] = "session_end"
    reason: Literal["complete", "user_shutdown", "ctrl_c", "error"] = Field(
        description="Why the session ended",
    )
    duration_ms: int = Field(description="Total session duration in milliseconds")
    total_tokens: dict[str, int] = Field(
        description="Aggregate token usage with 'input' and 'output' keys",
    )


class MessageEvent(_EventBase):
    """An inter-agent message."""

    type: Literal["message"] = "message"
    from_agent: str = Field(alias="from", description="Sending agent name")
    to: str = Field(description="Receiving agent name")
    content: str = Field(description="Message body")


class LLMCallStartEvent(_EventBase):
    """Emitted when an LLM call begins."""

    type: Literal["llm_call_start"] = "llm_call_start"
    agent: str = Field(description="Agent making the call")
    model: str = Field(description="LLM model identifier")
    messages_count: int = Field(description="Number of messages in the prompt")


class LLMCallEndEvent(_EventBase):
    """Emitted when an LLM call completes."""

    type: Literal["llm_call_end"] = "llm_call_end"
    agent: str = Field(description="Agent that made the call")
    model: str = Field(description="LLM model identifier")
    tokens: dict[str, int] = Field(
        description="Token usage with 'input' and 'output' keys",
    )
    duration_ms: int = Field(description="Call duration in milliseconds")


class ToolCallEvent(_EventBase):
    """Emitted when a tool is invoked."""

    type: Literal["tool_call"] = "tool_call"
    agent: str = Field(description="Agent invoking the tool")
    tool: str = Field(description="Tool name")
    args: dict[str, Any] = Field(description="Tool arguments")


class ToolResultEvent(_EventBase):
    """Emitted when a tool call returns."""

    type: Literal["tool_result"] = "tool_result"
    agent: str = Field(description="Agent that invoked the tool")
    tool: str = Field(description="Tool name")
    duration_ms: int = Field(description="Tool execution duration in milliseconds")
    result_size: int = Field(description="Size of the serialized result in bytes")


class StatusEvent(_EventBase):
    """Free-form agent status update."""

    type: Literal["status"] = "status"
    agent: str = Field(description="Agent reporting status")
    status: str = Field(description="Status message")


class ErrorEvent(_EventBase):
    """An error encountered during the session."""

    type: Literal["error"] = "error"
    agent: str | None = Field(
        default=None,
        description="Agent that hit the error (null for system-level errors)",
    )
    error: str = Field(description="Error description")
    retrying: bool = Field(description="Whether the operation will be retried")
    context: str | None = Field(
        default=None,
        description="Error context: api_call, config_validation, subprocess, etc.",
    )


class AgentStartEvent(_EventBase):
    """Emitted when an agent begins execution."""

    type: Literal["agent_start"] = "agent_start"
    agent: str = Field(description="Agent name")
    agent_type: str = Field(description="Agent type (llm, cli, script, ...)")


class AgentDoneEvent(_EventBase):
    """Emitted when an agent finishes execution."""

    type: Literal["agent_done"] = "agent_done"
    agent: str = Field(description="Agent name")
    reason: str = Field(description="Why the agent finished")


class LoopBoundaryEvent(_EventBase):
    """Emitted when a loop iteration starts or ends."""

    type: Literal["loop_boundary"] = "loop_boundary"
    boundary: Literal["start", "end"] = Field(
        description="Whether this is a loop start or end"
    )
    iteration: int = Field(ge=1, description="Loop iteration number (1-indexed)")


class AgentResponseEvent(_EventBase):
    """Emitted when an LLM agent produces a text response."""

    type: Literal["agent_response"] = "agent_response"
    agent: str = Field(description="Agent name")
    content: str = Field(description="Response text")


class CLITextChunkEvent(_EventBase):
    """Emitted when a CLI agent outputs a text chunk (streaming)."""

    type: Literal["cli_text_chunk"] = "cli_text_chunk"
    agent: str = Field(description="CLI agent name")
    text: str = Field(description="Text chunk content")


class CLIToolCallEvent(_EventBase):
    """Emitted when a CLI agent calls a tool (streaming)."""

    type: Literal["cli_tool_call"] = "cli_tool_call"
    agent: str = Field(description="CLI agent name")
    tool: str = Field(description="Tool name")
    args: dict[str, Any] = Field(description="Tool arguments")


class CLIThinkingEvent(_EventBase):
    """Emitted when a CLI agent shows internal thinking (streaming)."""

    type: Literal["cli_thinking"] = "cli_thinking"
    agent: str = Field(description="CLI agent name")
    content: str = Field(description="Thinking content")


class CLIProgressEvent(_EventBase):
    """Emitted when a CLI agent reports progress (streaming)."""

    type: Literal["cli_progress"] = "cli_progress"
    agent: str = Field(description="CLI agent name")
    status: str = Field(description="Progress status message")


class MemorySnapshotEvent(_EventBase):
    """Periodic per-agent memory snapshot for profiling."""

    type: Literal["memory_snapshot"] = "memory_snapshot"
    agent: str = Field(description="Agent name")
    agent_type: str = Field(description="Agent type: llm, cli, or script")
    process_rss_mb: float | None = Field(
        default=None, description="Host process RSS in MB"
    )
    subprocess_rss_mb: float | None = Field(
        default=None, description="Subprocess RSS in MB (CLI agents)"
    )
    subprocess_pid: int | None = Field(
        default=None, description="Subprocess PID (CLI agents)"
    )
    thread_messages: int | None = Field(
        default=None, description="Message count in conversation thread (LLM agents)"
    )
    thread_size_kb: float | None = Field(
        default=None, description="Estimated thread size in KB (LLM agents)"
    )
    queue_depth: int | None = Field(
        default=None, description="Message queue depth (CLI agents)"
    )
    system_available_mb: float | None = Field(
        default=None, description="System available memory in MB"
    )


def _event_discriminator(v: Any) -> str:
    """Extract the discriminator value from raw data or a model instance."""
    if isinstance(v, dict):
        return str(v.get("type", ""))
    return str(getattr(v, "type", ""))


SessionEvent = Annotated[
    Annotated[SessionStartEvent, Tag("session_start")]
    | Annotated[SessionEndEvent, Tag("session_end")]
    | Annotated[MessageEvent, Tag("message")]
    | Annotated[LLMCallStartEvent, Tag("llm_call_start")]
    | Annotated[LLMCallEndEvent, Tag("llm_call_end")]
    | Annotated[ToolCallEvent, Tag("tool_call")]
    | Annotated[ToolResultEvent, Tag("tool_result")]
    | Annotated[StatusEvent, Tag("status")]
    | Annotated[ErrorEvent, Tag("error")]
    | Annotated[AgentStartEvent, Tag("agent_start")]
    | Annotated[AgentDoneEvent, Tag("agent_done")]
    | Annotated[LoopBoundaryEvent, Tag("loop_boundary")]
    | Annotated[AgentResponseEvent, Tag("agent_response")]
    | Annotated[CLITextChunkEvent, Tag("cli_text_chunk")]
    | Annotated[CLIToolCallEvent, Tag("cli_tool_call")]
    | Annotated[CLIThinkingEvent, Tag("cli_thinking")]
    | Annotated[CLIProgressEvent, Tag("cli_progress")]
    | Annotated[MemorySnapshotEvent, Tag("memory_snapshot")],
    Discriminator(_event_discriminator),
]
"""Discriminated union of all session event types."""
