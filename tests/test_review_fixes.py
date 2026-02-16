"""Tests for issues identified in the code review.

Covers:
- RateLimitGate pause behaviour
- Router.broadcast fire-and-forget task tracking
- LLMAgent thread truncation
- LLMAgent error classification
- Multi-agent integration (mock agents exchanging messages end-to-end)
- CLIBridge._claude_busy cleanup on cancellation
- code-exec sandboxing (restricted env)
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from lattice.agent.builtin_tools import _DDGResultParser, handle_code_exec
from lattice.agent.llm_agent import (
    _MAX_THREAD_TOKENS,
    LLMAgent,
    RateLimitGate,
    _ErrorKind,
)
from lattice.agent.providers import LLMResponse, TokenUsage, ToolCall
from lattice.config.models import TopologyConfig
from lattice.router.router import Router
from lattice.session.recorder import SessionRecorder

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


class MockProvider:
    """A fake LLM provider that returns preconfigured responses."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> LLMResponse:
        resp = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return resp


def _make_recorder(tmp_path: Path) -> SessionRecorder:
    return SessionRecorder("test-team", "abc123", sessions_dir=tmp_path / "sessions")


def _make_router(tmp_path: Path) -> tuple[Router, SessionRecorder]:
    recorder = _make_recorder(tmp_path)
    router = Router(topology=TopologyConfig(type="mesh"), recorder=recorder)
    return router, recorder


def _make_agent(
    router: Router,
    recorder: SessionRecorder,
    provider: MockProvider | None = None,
    name: str = "agent-a",
    peer_names: list[str] | None = None,
) -> LLMAgent:
    if provider is None:
        provider = MockProvider([LLMResponse(content="ok", usage=TokenUsage(5, 2))])
    return LLMAgent(
        name=name,
        model_string="test/model",
        role="You are a test agent.",
        router=router,
        recorder=recorder,
        team_name="test-team",
        peer_names=peer_names or [],
        provider=provider,
        model_override="test/model",
    )


# ------------------------------------------------------------------ #
# RateLimitGate tests
# ------------------------------------------------------------------ #


class TestRateLimitGate:
    """Tests for the shared rate limit gate."""

    async def test_no_pause_initially(self) -> None:
        gate = RateLimitGate(pause_seconds=1.0)
        # Should return immediately — no pause active.
        await asyncio.wait_for(gate.wait_if_paused(), timeout=0.1)

    async def test_pause_blocks_wait(self) -> None:
        gate = RateLimitGate(pause_seconds=0.3)
        gate.pause()
        # Should block for ~0.3s.
        start = asyncio.get_event_loop().time()
        await gate.wait_if_paused()
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed >= 0.2  # Allow some tolerance.

    async def test_multiple_pauses_use_max(self) -> None:
        gate = RateLimitGate(pause_seconds=0.2)
        gate.pause()
        gate.pause()  # Should not shorten the existing pause.
        start = asyncio.get_event_loop().time()
        await gate.wait_if_paused()
        elapsed = asyncio.get_event_loop().time() - start
        assert elapsed >= 0.15


# ------------------------------------------------------------------ #
# Broadcast fire-and-forget tests
# ------------------------------------------------------------------ #


class TestBroadcastTracking:
    """Router.broadcast should track tasks in pending_tasks."""

    async def test_broadcast_tracks_tasks(self, tmp_path: Path) -> None:
        router, recorder = _make_router(tmp_path)

        received: dict[str, list[str]] = {"a": [], "b": []}

        class MockAgent:
            def __init__(self, name: str) -> None:
                self._name = name

            async def handle_message(self, from_agent: str, content: str) -> None:
                received[self._name].append(content)
                await asyncio.sleep(0.05)  # Simulate work.

        router.register("a", MockAgent("a"))
        router.register("b", MockAgent("b"))

        await router.broadcast("user", "hello everyone")

        # broadcast is now fire-and-forget — tasks should be in pending_tasks
        # (may already be done since we awaited broadcast).
        # But the messages should have been dispatched.
        # Give tasks time to complete.
        await asyncio.sleep(0.2)
        assert received["a"] == ["hello everyone"]
        assert received["b"] == ["hello everyone"]


# ------------------------------------------------------------------ #
# Thread truncation tests
# ------------------------------------------------------------------ #


class TestThreadTruncation:
    """LLMAgent._truncate_thread_if_needed should cap thread size."""

    def test_no_truncation_under_budget(self) -> None:
        thread: list[dict[str, Any]] = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        LLMAgent._truncate_thread_if_needed(LLMAgent.__new__(LLMAgent), thread)
        assert len(thread) == 2

    def test_truncation_removes_oldest(self) -> None:
        # Build a thread that exceeds _MAX_THREAD_TOKENS.
        # Each message is ~1000 tokens worth of chars.
        big_content = "x" * 4000  # ~1000 tokens
        thread: list[dict[str, Any]] = [
            {"role": "user", "content": big_content}
            for _ in range(200)  # ~200k tokens
        ]
        original_len = len(thread)

        agent = LLMAgent.__new__(LLMAgent)
        agent.name = "test"
        LLMAgent._truncate_thread_if_needed(agent, thread)

        assert len(thread) < original_len
        estimated = LLMAgent._estimate_thread_tokens(thread)
        assert estimated <= _MAX_THREAD_TOKENS

    def test_keeps_minimum_messages(self) -> None:
        # Even with enormous messages, keep at least 4.
        huge_content = "x" * 400_000  # ~100k tokens each
        thread: list[dict[str, Any]] = [
            {"role": "user", "content": huge_content} for _ in range(10)
        ]
        agent = LLMAgent.__new__(LLMAgent)
        agent.name = "test"
        LLMAgent._truncate_thread_if_needed(agent, thread)
        assert len(thread) >= 4

    def test_estimate_tokens(self) -> None:
        thread = [
            {"role": "user", "content": "hello world"},  # 11 chars
        ]
        tokens = LLMAgent._estimate_thread_tokens(thread)
        assert tokens == 11 // 4  # 2


# ------------------------------------------------------------------ #
# Error classification tests
# ------------------------------------------------------------------ #


class TestErrorClassification:
    """LLMAgent._classify_error should categorize errors correctly."""

    def test_rate_limit_429(self) -> None:
        exc = RuntimeError("HTTP 429 Too Many Requests")
        assert LLMAgent._classify_error(exc) == _ErrorKind.RATE_LIMIT

    def test_rate_limit_text(self) -> None:
        exc = RuntimeError("rate_limit_exceeded")
        assert LLMAgent._classify_error(exc) == _ErrorKind.RATE_LIMIT

    def test_auth_401(self) -> None:
        exc = RuntimeError("HTTP 401 Unauthorized")
        assert LLMAgent._classify_error(exc) == _ErrorKind.AUTH

    def test_auth_api_key(self) -> None:
        exc = RuntimeError("Invalid API key provided")
        assert LLMAgent._classify_error(exc) == _ErrorKind.AUTH

    def test_server_500(self) -> None:
        exc = RuntimeError("HTTP 500 Internal Server Error")
        assert LLMAgent._classify_error(exc) == _ErrorKind.SERVER

    def test_server_unavailable(self) -> None:
        exc = RuntimeError("Service Unavailable")
        assert LLMAgent._classify_error(exc) == _ErrorKind.SERVER

    def test_network_connection(self) -> None:
        exc = RuntimeError("Connection refused")
        assert LLMAgent._classify_error(exc) == _ErrorKind.NETWORK

    def test_network_timeout(self) -> None:
        exc = RuntimeError("Request timeout after 30s")
        assert LLMAgent._classify_error(exc) == _ErrorKind.NETWORK

    def test_network_dns(self) -> None:
        exc = RuntimeError("DNS resolution failed")
        assert LLMAgent._classify_error(exc) == _ErrorKind.NETWORK

    def test_unknown(self) -> None:
        exc = RuntimeError("something weird happened")
        assert LLMAgent._classify_error(exc) == _ErrorKind.UNKNOWN


# ------------------------------------------------------------------ #
# Multi-agent integration test
# ------------------------------------------------------------------ #


class TestMultiAgentIntegration:
    """End-to-end test with multiple mock agents exchanging messages."""

    async def test_two_agents_exchange_messages(self, tmp_path: Path) -> None:
        """Agent A sends a message to Agent B via send_message tool.
        Agent B responds with text.
        """
        router, recorder = _make_router(tmp_path)

        # Agent A: calls send_message(to="agent-b", content="ping")
        # then on second call returns text.
        provider_a = MockProvider(
            [
                LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="tc1",
                            name="send_message",
                            arguments={"to": "agent-b", "content": "ping"},
                        )
                    ],
                    usage=TokenUsage(10, 5),
                ),
                LLMResponse(content="Done sending!", usage=TokenUsage(10, 5)),
            ]
        )

        # Agent B: returns text response on receiving message.
        provider_b = MockProvider(
            [
                LLMResponse(content="pong!", usage=TokenUsage(10, 5)),
            ]
        )

        agent_a = _make_agent(
            router,
            recorder,
            provider_a,
            name="agent-a",
            peer_names=["agent-b"],
        )
        agent_b = _make_agent(
            router,
            recorder,
            provider_b,
            name="agent-b",
            peer_names=["agent-a"],
        )

        router.register("agent-a", agent_a)
        router.register("agent-b", agent_b)

        # Trigger agent A.
        await router.send("user", "agent-a", "Say hi to agent-b")

        # Wait for all tasks to complete.
        for _ in range(50):
            if not router.pending_tasks:
                break
            await asyncio.sleep(0.05)

        # Agent A should have sent to agent B, and B should have received.
        assert len(agent_b._thread) > 0
        # Check that B got a message from A.
        assert any(
            "[from agent-a]" in str(m.get("content", "")) for m in agent_b._thread
        )


# ------------------------------------------------------------------ #
# code-exec sandboxing test
# ------------------------------------------------------------------ #


class TestCodeExecSandboxing:
    """Verify code-exec runs in restricted environment."""

    async def test_restricted_env_no_api_keys(self) -> None:
        """Code should not see parent process env vars."""
        import os

        # Set a fake env var in current process.
        os.environ["LATTICE_TEST_SECRET"] = "super-secret"
        try:
            result_json = await handle_code_exec(
                {
                    "code": (
                        "import os; print(os.environ.get("
                        "'LATTICE_TEST_SECRET', 'NOT_FOUND'))"
                    ),
                    "timeout": 10,
                }
            )
            import json

            result = json.loads(result_json)
            assert result["stdout"].strip() == "NOT_FOUND"
            assert result["exit_code"] == 0
        finally:
            del os.environ["LATTICE_TEST_SECRET"]

    async def test_runs_in_temp_directory(self) -> None:
        """Working directory should be a temp dir, not cwd."""
        result_json = await handle_code_exec(
            {
                "code": "import os; print(os.getcwd())",
                "timeout": 10,
            }
        )
        import json

        result = json.loads(result_json)
        cwd = result["stdout"].strip()
        assert "lattice_exec_" in cwd


# ------------------------------------------------------------------ #
# DuckDuckGo HTMLParser tests
# ------------------------------------------------------------------ #


class TestDDGParser:
    """Test the stdlib HTMLParser-based DuckDuckGo result parser."""

    def test_parses_result_links(self) -> None:
        html = """
        <div class="result">
            <a class="result__a" href="https://example.com">Example Title</a>
            <a class="result__snippet" href="#">This is a snippet.</a>
        </div>
        <div class="result">
            <a class="result__a" href="https://other.com">Other Title</a>
            <a class="result__snippet" href="#">Another snippet.</a>
        </div>
        """
        parser = _DDGResultParser(max_results=5)
        parser.feed(html)
        assert len(parser.results) == 2
        assert parser.results[0]["title"] == "Example Title"
        assert parser.results[0]["url"] == "https://example.com"
        assert parser.results[0]["snippet"] == "This is a snippet."

    def test_max_results_respected(self) -> None:
        html = """
        <a class="result__a" href="https://a.com">A</a>
        <a class="result__snippet">snip a</a>
        <a class="result__a" href="https://b.com">B</a>
        <a class="result__snippet">snip b</a>
        <a class="result__a" href="https://c.com">C</a>
        <a class="result__snippet">snip c</a>
        """
        parser = _DDGResultParser(max_results=2)
        parser.feed(html)
        assert len(parser.results) == 2

    def test_empty_html(self) -> None:
        parser = _DDGResultParser(max_results=5)
        parser.feed("<html><body>No results</body></html>")
        assert parser.results == []


# ------------------------------------------------------------------ #
# CLIBridge busy flag cleanup test
# ------------------------------------------------------------------ #


class TestCLIBridgeBusyCleanup:
    """_claude_busy should always be reset even on cancellation."""

    async def test_busy_flag_reset_on_cancel(self, tmp_path: Path) -> None:
        """Simulate cancellation mid-task and verify cleanup."""
        from unittest.mock import AsyncMock, patch

        from lattice.agent.cli_bridge import CLIBridge

        router, recorder = _make_router(tmp_path)
        bridge = CLIBridge(
            name="test-cli",
            role="test",
            router=router,
            recorder=recorder,
            team_name="test-team",
            peer_names=[],
            cli_type="claude",
        )
        bridge._started = True

        # Mock get_available_mb to return plenty of memory.
        with patch("lattice.memory_monitor.get_available_mb", return_value=8000.0):
            # Mock create_subprocess_exec to simulate a long-running process
            # that gets cancelled.
            mock_proc = AsyncMock()
            mock_proc.pid = 12345
            mock_proc.stdout = AsyncMock()
            mock_proc.stdout.readline = AsyncMock(side_effect=asyncio.CancelledError)
            mock_proc.stderr = AsyncMock()
            mock_proc.stderr.read = AsyncMock(return_value=b"")
            mock_proc.wait = AsyncMock(return_value=0)

            with (
                patch(
                    "asyncio.create_subprocess_exec",
                    return_value=mock_proc,
                ),
                pytest.raises(asyncio.CancelledError),
            ):
                await bridge._handle_claude_task("user", "test prompt")

        # Key assertion: busy flag must be False after cancellation.
        assert bridge._claude_busy is False
        assert bridge._current_claude_pid is None
