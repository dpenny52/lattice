"""Tests for the LLM agent runtime."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from lattice.agent.llm_agent import LLMAgent
from lattice.agent.providers import LLMResponse, TokenUsage, ToolCall
from lattice.config.models import TopologyConfig
from lattice.router.router import Router
from lattice.session.models import (
    AgentDoneEvent,
    AgentStartEvent,
    ErrorEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from lattice.session.recorder import SessionRecorder

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


class MockProvider:
    """A fake LLM provider that returns preconfigured responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools, "model": model})
        resp = self._responses[self._call_count]
        self._call_count += 1
        return resp


class FailThenSucceedProvider:
    """Fails N times then succeeds."""

    def __init__(self, fail_count: int, success_response: LLMResponse) -> None:
        self._fail_count = fail_count
        self._success = success_response
        self._attempt = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        self._attempt += 1
        if self._attempt <= self._fail_count:
            msg = f"API error on attempt {self._attempt}"
            raise RuntimeError(msg)
        return self._success


class AlwaysFailProvider:
    """Always raises an error."""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        msg = "Permanent API failure"
        raise RuntimeError(msg)


@pytest.fixture
def recorder(tmp_path: Any) -> SessionRecorder:
    return SessionRecorder(
        team="test-team",
        config_hash="abc123",
        sessions_dir=tmp_path / "sessions",
    )


@pytest.fixture
def router(recorder: SessionRecorder) -> Router:
    return Router(topology=TopologyConfig(type="mesh"), recorder=recorder)


def _make_agent(
    router: Router,
    recorder: SessionRecorder,
    provider: Any,
    name: str = "agent-a",
    peer_names: list[str] | None = None,
) -> LLMAgent:
    """Create an LLMAgent with a mock provider."""
    return LLMAgent(
        name=name,
        model_string="mock/test-model",
        role="You are a helpful assistant.",
        router=router,
        recorder=recorder,
        team_name="test-team",
        peer_names=peer_names or ["agent-b"],
        provider=provider,
        model_override="test-model",
    )


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


class TestLLMAgentBasic:
    """Basic agent behavior tests."""

    async def test_simple_text_response(
        self, router: Router, recorder: SessionRecorder
    ) -> None:
        """Agent receives message, calls LLM, gets text -- done."""
        provider = MockProvider(
            [LLMResponse(content="Hello!", usage=TokenUsage(10, 5))]
        )
        agent = _make_agent(router, recorder, provider)
        router.register("agent-a", agent)

        await agent.handle_message("user", "hi there")

        assert provider._call_count == 1
        # The thread should have the user message + assistant response.
        assert len(agent._thread) == 2
        assert agent._thread[0]["content"] == "[from user]: hi there"
        assert agent._thread[1]["content"] == "Hello!"

    async def test_empty_response_no_crash(
        self, router: Router, recorder: SessionRecorder
    ) -> None:
        """LLM returns empty content and no tool calls -- loop exits cleanly."""
        provider = MockProvider(
            [LLMResponse(content=None, tool_calls=[], usage=TokenUsage(5, 2))]
        )
        agent = _make_agent(router, recorder, provider)
        router.register("agent-a", agent)

        await agent.handle_message("user", "hi")

        assert provider._call_count == 1
        # Thread should only have the user message (no assistant entry).
        assert len(agent._thread) == 1
        assert agent._thread[0]["content"] == "[from user]: hi"

    async def test_unknown_tool_returns_error(
        self, router: Router, recorder: SessionRecorder
    ) -> None:
        """Unknown tool name returns an error string, loop continues."""
        responses = [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="nonexistent_tool",
                        arguments={"foo": "bar"},
                    )
                ],
                usage=TokenUsage(10, 5),
            ),
            LLMResponse(content="Ok, that failed.", usage=TokenUsage(5, 3)),
        ]
        provider = MockProvider(responses)
        agent = _make_agent(router, recorder, provider)
        router.register("agent-a", agent)

        await agent.handle_message("user", "use some tool")

        assert provider._call_count == 2
        # The tool result in the thread should contain the error.
        tool_results = [m for m in agent._thread if m.get("role") == "tool"]
        assert len(tool_results) == 1
        assert "Error" in tool_results[0]["content"]
        assert "Unknown tool" in tool_results[0]["content"]

    async def test_send_message_tool_call(
        self, router: Router, recorder: SessionRecorder
    ) -> None:
        """Agent calls send_message tool, message dispatched via router."""
        # First call: LLM returns a tool call.
        # Second call: LLM returns text (loop ends).
        responses = [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="send_message",
                        arguments={
                            "to": "agent-b",
                            "content": "hey b!",
                        },
                    )
                ],
                usage=TokenUsage(20, 10),
            ),
            LLMResponse(content="Done.", usage=TokenUsage(15, 5)),
        ]
        provider = MockProvider(responses)
        agent_a = _make_agent(router, recorder, provider, name="agent-a")
        router.register("agent-a", agent_a)

        # Register a mock agent-b to receive the message.
        agent_b = MagicMock()
        agent_b.handle_message = AsyncMock()
        router.register("agent-b", agent_b)

        await agent_a.handle_message("user", "tell agent-b hey")

        assert provider._call_count == 2
        # agent-b should have received the dispatched message.
        agent_b.handle_message.assert_called_once_with("agent-a", "hey b!")

    async def test_multiple_tool_calls_then_text(
        self, router: Router, recorder: SessionRecorder
    ) -> None:
        """LLM returns tool call, then another, then text."""
        responses = [
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="send_message",
                        arguments={
                            "to": "agent-b",
                            "content": "msg1",
                        },
                    )
                ],
                usage=TokenUsage(10, 5),
            ),
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c2",
                        name="send_message",
                        arguments={
                            "to": "agent-b",
                            "content": "msg2",
                        },
                    )
                ],
                usage=TokenUsage(10, 5),
            ),
            LLMResponse(content="All done.", usage=TokenUsage(10, 5)),
        ]

        provider = MockProvider(responses)
        agent = _make_agent(router, recorder, provider)
        router.register("agent-a", agent)

        agent_b = MagicMock()
        agent_b.handle_message = AsyncMock()
        router.register("agent-b", agent_b)

        await agent.handle_message("user", "send two messages")

        assert provider._call_count == 3
        assert agent_b.handle_message.call_count == 2


class TestTeamContext:
    """Team context injection into system prompt."""

    async def test_team_context_in_system_prompt(
        self, router: Router, recorder: SessionRecorder
    ) -> None:
        provider = MockProvider([LLMResponse(content="ok", usage=TokenUsage(5, 2))])
        agent = _make_agent(router, recorder, provider, peer_names=["bob", "carol"])
        router.register("agent-a", agent)

        await agent.handle_message("user", "hi")

        # Check that the system prompt contains team context.
        system_msg = provider.calls[0]["messages"][0]
        assert system_msg["role"] == "system"
        assert 'team called "test-team"' in system_msg["content"]
        assert "bob, carol" in system_msg["content"]
        assert "send_message" in system_msg["content"]

    async def test_send_message_tool_always_present(
        self, router: Router, recorder: SessionRecorder
    ) -> None:
        provider = MockProvider([LLMResponse(content="ok", usage=TokenUsage(5, 2))])
        agent = _make_agent(router, recorder, provider)
        router.register("agent-a", agent)

        await agent.handle_message("user", "hi")

        tools = provider.calls[0]["tools"]
        tool_names = [t["name"] for t in tools]
        assert "send_message" in tool_names


class TestEventRecording:
    """Event recording for LLM calls and tool calls."""

    async def test_llm_call_events(
        self, router: Router, recorder: SessionRecorder
    ) -> None:
        """llm_call_start and llm_call_end events are recorded."""
        provider = MockProvider([LLMResponse(content="ok", usage=TokenUsage(10, 5))])
        agent = _make_agent(router, recorder, provider)
        router.register("agent-a", agent)

        events: list[Any] = []
        original_record = recorder.record

        def capture_record(event: Any) -> None:
            events.append(event)
            original_record(event)

        recorder.record = capture_record  # type: ignore[assignment]

        await agent.handle_message("user", "hi")

        event_types = [type(e) for e in events]
        assert AgentStartEvent in event_types
        assert LLMCallStartEvent in event_types
        assert LLMCallEndEvent in event_types
        assert AgentDoneEvent in event_types

        # Check token counts in llm_call_end.
        end_events = [e for e in events if isinstance(e, LLMCallEndEvent)]
        assert len(end_events) == 1
        assert end_events[0].tokens == {"input": 10, "output": 5}

    async def test_tool_call_events(
        self, router: Router, recorder: SessionRecorder
    ) -> None:
        """tool_call and tool_result events are recorded for tool executions."""
        provider = MockProvider(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="send_message",
                            arguments={
                                "to": "agent-b",
                                "content": "hi",
                            },
                        )
                    ],
                    usage=TokenUsage(10, 5),
                ),
                LLMResponse(content="Done", usage=TokenUsage(5, 3)),
            ]
        )
        agent = _make_agent(router, recorder, provider)
        router.register("agent-a", agent)

        agent_b = MagicMock()
        agent_b.handle_message = AsyncMock()
        router.register("agent-b", agent_b)

        events: list[Any] = []
        original_record = recorder.record

        def capture_record(event: Any) -> None:
            events.append(event)
            original_record(event)

        recorder.record = capture_record  # type: ignore[assignment]

        await agent.handle_message("user", "talk to b")

        event_types = [type(e) for e in events]
        assert ToolCallEvent in event_types
        assert ToolResultEvent in event_types

        tool_call_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert tool_call_events[0].tool == "send_message"


class TestRetryBehavior:
    """API error retry with exponential backoff."""

    async def test_retry_then_succeed(
        self, router: Router, recorder: SessionRecorder
    ) -> None:
        """Agent retries on API error and eventually succeeds."""
        success = LLMResponse(content="recovered!", usage=TokenUsage(5, 3))
        provider = FailThenSucceedProvider(fail_count=2, success_response=success)
        agent = _make_agent(router, recorder, provider)
        router.register("agent-a", agent)

        # Patch sleep to avoid real delays.
        import lattice.agent.llm_agent as llm_mod

        original_delay = llm_mod._BASE_DELAY
        llm_mod._BASE_DELAY = 0.0
        try:
            await agent.handle_message("user", "hi")
        finally:
            llm_mod._BASE_DELAY = original_delay

        assert agent._thread[-1]["content"] == "recovered!"

    async def test_retries_exhausted(
        self, router: Router, recorder: SessionRecorder
    ) -> None:
        """All retries fail -- error event recorded, no crash."""
        provider = AlwaysFailProvider()
        agent = _make_agent(router, recorder, provider)
        router.register("agent-a", agent)

        events: list[Any] = []
        original_record = recorder.record

        def capture_record(event: Any) -> None:
            events.append(event)
            original_record(event)

        recorder.record = capture_record  # type: ignore[assignment]

        import lattice.agent.llm_agent as llm_mod

        original_delay = llm_mod._BASE_DELAY
        llm_mod._BASE_DELAY = 0.0
        try:
            await agent.handle_message("user", "hi")
        finally:
            llm_mod._BASE_DELAY = original_delay

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(error_events) == 3  # 3 retries

        # Last error should have retrying=False.
        assert error_events[-1].retrying is False
        # First two should have retrying=True.
        assert error_events[0].retrying is True
        assert error_events[1].retrying is True
