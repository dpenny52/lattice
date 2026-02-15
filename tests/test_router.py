"""Tests for the Router module — topology, dispatch, recording, and error handling."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from lattice.config.models import TopologyConfig
from lattice.router.router import (
    Agent,
    RouteNotAllowedError,
    Router,
    UnknownAgentError,
)
from lattice.router.topology import (
    CustomTopology,
    HubTopology,
    MeshTopology,
    PipelineTopology,
    TopologyChecker,
    create_topology,
)
from lattice.session.recorder import SessionRecorder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockAgent:
    """Test double that records received messages."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    async def handle_message(self, from_agent: str, content: str) -> None:
        self.messages.append((from_agent, content))


class FailingAgent:
    """Agent that always raises on handle_message."""

    async def handle_message(self, from_agent: str, content: str) -> None:
        msg = "boom"
        raise RuntimeError(msg)


class SlowAgent:
    """Agent that sleeps before recording, useful for concurrency tests."""

    def __init__(self, delay: float = 0.05) -> None:
        self.messages: list[tuple[str, str]] = []
        self._delay = delay

    async def handle_message(self, from_agent: str, content: str) -> None:
        await asyncio.sleep(self._delay)
        self.messages.append((from_agent, content))


class RespondingAgent:
    """Agent that replies via router.send() when it receives a message."""

    def __init__(
        self, router: Router, name: str, reply: str, delay: float = 0.0,
    ) -> None:
        self._router = router
        self._name = name
        self._reply = reply
        self._delay = delay
        self.messages: list[tuple[str, str]] = []

    async def handle_message(self, from_agent: str, content: str) -> None:
        self.messages.append((from_agent, content))
        if self._delay:
            await asyncio.sleep(self._delay)
        await self._router.send(self._name, from_agent, self._reply)


def _make_recorder(tmp_path: Path) -> SessionRecorder:
    return SessionRecorder("test-team", "abc123", sessions_dir=tmp_path)


def _read_events(recorder: SessionRecorder) -> list[dict[str, Any]]:
    """Read all events from the recorder's JSONL file."""
    lines = recorder.session_file.read_text().strip().splitlines()
    return [json.loads(line) for line in lines]


# ===================================================================
# Topology checker unit tests
# ===================================================================


class TestMeshTopology:
    def test_any_to_any(self) -> None:
        topo = MeshTopology()
        assert topo.is_allowed("a", "b")
        assert topo.is_allowed("b", "a")
        assert topo.is_allowed("x", "y")

    def test_self_send(self) -> None:
        topo = MeshTopology()
        assert topo.is_allowed("a", "a")

    def test_implements_protocol(self) -> None:
        assert isinstance(MeshTopology(), TopologyChecker)


class TestPipelineTopology:
    def test_sequential_neighbors_allowed(self) -> None:
        topo = PipelineTopology(["a", "b", "c"])
        assert topo.is_allowed("a", "b")
        assert topo.is_allowed("b", "c")

    def test_reverse_direction_allowed(self) -> None:
        topo = PipelineTopology(["a", "b", "c"])
        assert topo.is_allowed("b", "a")
        assert topo.is_allowed("c", "b")

    def test_non_neighbors_blocked(self) -> None:
        topo = PipelineTopology(["a", "b", "c"])
        assert not topo.is_allowed("a", "c")
        assert not topo.is_allowed("c", "a")

    def test_self_send_blocked(self) -> None:
        topo = PipelineTopology(["a", "b", "c"])
        assert not topo.is_allowed("a", "a")

    def test_two_element_pipeline(self) -> None:
        topo = PipelineTopology(["x", "y"])
        assert topo.is_allowed("x", "y")
        assert topo.is_allowed("y", "x")


class TestHubTopology:
    def test_worker_to_coordinator(self) -> None:
        topo = HubTopology("coord", ["w1", "w2"])
        assert topo.is_allowed("w1", "coord")
        assert topo.is_allowed("w2", "coord")

    def test_coordinator_to_worker(self) -> None:
        topo = HubTopology("coord", ["w1", "w2"])
        assert topo.is_allowed("coord", "w1")
        assert topo.is_allowed("coord", "w2")

    def test_worker_to_worker_blocked(self) -> None:
        topo = HubTopology("coord", ["w1", "w2"])
        assert not topo.is_allowed("w1", "w2")
        assert not topo.is_allowed("w2", "w1")

    def test_coordinator_self_send_blocked(self) -> None:
        topo = HubTopology("coord", ["w1"])
        assert not topo.is_allowed("coord", "coord")

    def test_unknown_agent_blocked(self) -> None:
        topo = HubTopology("coord", ["w1"])
        assert not topo.is_allowed("stranger", "coord")
        assert not topo.is_allowed("coord", "stranger")


class TestCustomTopology:
    def test_explicit_edges_allowed(self) -> None:
        topo = CustomTopology({"a": ["b", "c"], "b": ["a"]})
        assert topo.is_allowed("a", "b")
        assert topo.is_allowed("a", "c")
        assert topo.is_allowed("b", "a")

    def test_missing_edges_blocked(self) -> None:
        topo = CustomTopology({"a": ["b"]})
        assert not topo.is_allowed("b", "a")
        assert not topo.is_allowed("a", "c")

    def test_directed_nature(self) -> None:
        topo = CustomTopology({"a": ["b"]})
        assert topo.is_allowed("a", "b")
        assert not topo.is_allowed("b", "a")

    def test_self_loop_if_declared(self) -> None:
        topo = CustomTopology({"a": ["a"]})
        assert topo.is_allowed("a", "a")


class TestCreateTopologyFactory:
    def test_mesh(self) -> None:
        config = TopologyConfig(type="mesh")
        topo = create_topology(config)
        assert isinstance(topo, MeshTopology)

    def test_pipeline(self) -> None:
        config = TopologyConfig(type="pipeline", flow=["a", "b", "c"])
        topo = create_topology(config)
        assert isinstance(topo, PipelineTopology)
        assert topo.is_allowed("a", "b")
        assert not topo.is_allowed("a", "c")

    def test_hub(self) -> None:
        config = TopologyConfig(type="hub", coordinator="c", workers=["w1", "w2"])
        topo = create_topology(config)
        assert isinstance(topo, HubTopology)
        assert topo.is_allowed("w1", "c")
        assert not topo.is_allowed("w1", "w2")

    def test_custom(self) -> None:
        config = TopologyConfig(type="custom", edges={"a": ["b"]})
        topo = create_topology(config)
        assert isinstance(topo, CustomTopology)
        assert topo.is_allowed("a", "b")
        assert not topo.is_allowed("b", "a")


# ===================================================================
# Router tests
# ===================================================================


class TestRouterRegistry:
    async def test_register_and_send(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        agent = MockAgent()
        router.register("agent-a", agent)
        await router.send("user", "agent-a", "hello")
        await asyncio.sleep(0.05)
        assert agent.messages == [("user", "hello")]
        rec.close()

    async def test_unknown_agent_raises(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        with pytest.raises(UnknownAgentError, match="Unknown agent 'ghost'"):
            await router.send("user", "ghost", "hello")
        rec.close()


class TestMeshRouting:
    async def test_any_agent_can_reach_any_other(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        a = MockAgent()
        b = MockAgent()
        router.register("a", a)
        router.register("b", b)

        await router.send("a", "b", "hi from a")
        await router.send("b", "a", "hi from b")
        await asyncio.sleep(0.05)

        assert b.messages == [("a", "hi from a")]
        assert a.messages == [("b", "hi from b")]
        rec.close()


class TestPipelineRouting:
    async def test_neighbors_allowed(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        topo = TopologyConfig(type="pipeline", flow=["a", "b", "c"])
        router = Router(topo, rec)
        a, b, c = MockAgent(), MockAgent(), MockAgent()
        router.register("a", a)
        router.register("b", b)
        router.register("c", c)

        await router.send("a", "b", "step1")
        await router.send("b", "c", "step2")
        await asyncio.sleep(0.05)

        assert b.messages == [("a", "step1")]
        assert c.messages == [("b", "step2")]
        rec.close()

    async def test_non_neighbors_blocked(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        topo = TopologyConfig(type="pipeline", flow=["a", "b", "c"])
        router = Router(topo, rec)
        for name in ("a", "b", "c"):
            router.register(name, MockAgent())

        with pytest.raises(RouteNotAllowedError):
            await router.send("a", "c", "skip")
        rec.close()


class TestHubRouting:
    async def test_worker_to_coordinator(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        topo = TopologyConfig(type="hub", coordinator="coord", workers=["w1", "w2"])
        router = Router(topo, rec)
        coord = MockAgent()
        router.register("coord", coord)
        router.register("w1", MockAgent())
        router.register("w2", MockAgent())

        await router.send("w1", "coord", "result")
        await asyncio.sleep(0.05)
        assert coord.messages == [("w1", "result")]
        rec.close()

    async def test_coordinator_to_worker(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        topo = TopologyConfig(type="hub", coordinator="coord", workers=["w1", "w2"])
        router = Router(topo, rec)
        w1 = MockAgent()
        router.register("coord", MockAgent())
        router.register("w1", w1)
        router.register("w2", MockAgent())

        await router.send("coord", "w1", "task")
        await asyncio.sleep(0.05)
        assert w1.messages == [("coord", "task")]
        rec.close()

    async def test_worker_to_worker_blocked(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        topo = TopologyConfig(type="hub", coordinator="coord", workers=["w1", "w2"])
        router = Router(topo, rec)
        for name in ("coord", "w1", "w2"):
            router.register(name, MockAgent())

        with pytest.raises(RouteNotAllowedError):
            await router.send("w1", "w2", "hey")
        rec.close()


class TestCustomRouting:
    async def test_declared_edges_work(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        topo = TopologyConfig(type="custom", edges={"a": ["b"]})
        router = Router(topo, rec)
        b = MockAgent()
        router.register("a", MockAgent())
        router.register("b", b)

        await router.send("a", "b", "directed")
        await asyncio.sleep(0.05)
        assert b.messages == [("a", "directed")]
        rec.close()

    async def test_undeclared_edges_blocked(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        topo = TopologyConfig(type="custom", edges={"a": ["b"]})
        router = Router(topo, rec)
        router.register("a", MockAgent())
        router.register("b", MockAgent())

        with pytest.raises(RouteNotAllowedError):
            await router.send("b", "a", "nope")
        rec.close()


class TestUserBypass:
    async def test_user_bypasses_pipeline(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        topo = TopologyConfig(type="pipeline", flow=["a", "b", "c"])
        router = Router(topo, rec)
        c = MockAgent()
        router.register("a", MockAgent())
        router.register("b", MockAgent())
        router.register("c", c)

        # user can skip directly to c even though a→c is not allowed
        await router.send("user", "c", "god mode")
        await asyncio.sleep(0.05)
        assert c.messages == [("user", "god mode")]
        rec.close()

    async def test_user_bypasses_hub(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        topo = TopologyConfig(type="hub", coordinator="coord", workers=["w1", "w2"])
        router = Router(topo, rec)
        w2 = MockAgent()
        router.register("coord", MockAgent())
        router.register("w1", MockAgent())
        router.register("w2", w2)

        await router.send("user", "w2", "direct")
        await asyncio.sleep(0.05)
        assert w2.messages == [("user", "direct")]
        rec.close()

    async def test_user_bypasses_custom(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        topo = TopologyConfig(type="custom", edges={"a": ["b"]})
        router = Router(topo, rec)
        a = MockAgent()
        router.register("a", a)
        router.register("b", MockAgent())

        # b→a is not an edge, but user can reach a
        await router.send("user", "a", "override")
        await asyncio.sleep(0.05)
        assert a.messages == [("user", "override")]
        rec.close()


class TestSessionRecording:
    async def test_message_recorded_before_dispatch(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        router.register("a", MockAgent())
        router.register("b", MockAgent())

        await router.send("a", "b", "hello")
        await asyncio.sleep(0.05)

        events = _read_events(rec)
        message_events = [e for e in events if e["type"] == "message"]
        assert len(message_events) == 1
        msg = message_events[0]
        assert msg["from"] == "a"
        assert msg["to"] == "b"
        assert msg["content"] == "hello"
        rec.close()

    async def test_multiple_sends_recorded(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        router.register("a", MockAgent())
        router.register("b", MockAgent())

        await router.send("a", "b", "msg1")
        await router.send("b", "a", "msg2")
        await asyncio.sleep(0.05)

        events = _read_events(rec)
        message_events = [e for e in events if e["type"] == "message"]
        assert len(message_events) == 2
        rec.close()

    async def test_failed_route_not_recorded(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        topo = TopologyConfig(type="pipeline", flow=["a", "b", "c"])
        router = Router(topo, rec)
        for name in ("a", "b", "c"):
            router.register(name, MockAgent())

        with pytest.raises(RouteNotAllowedError):
            await router.send("a", "c", "blocked")

        events = _read_events(rec)
        message_events = [e for e in events if e["type"] == "message"]
        assert len(message_events) == 0
        rec.close()


class TestAsyncDispatch:
    async def test_send_returns_immediately(self, tmp_path: Path) -> None:
        """send() should return before the agent finishes handling."""
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        slow = SlowAgent(delay=0.1)
        router.register("a", MockAgent())
        router.register("b", slow)

        await router.send("a", "b", "fast return")
        # Agent hasn't processed yet
        assert len(slow.messages) == 0
        # Wait for it to finish
        await asyncio.sleep(0.2)
        assert slow.messages == [("a", "fast return")]
        rec.close()


class TestBroadcast:
    async def test_broadcast_to_all(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        a = MockAgent()
        b = MockAgent()
        c = MockAgent()
        router.register("a", a)
        router.register("b", b)
        router.register("c", c)

        await router.broadcast("a", "hey everyone")
        await asyncio.sleep(0.05)

        # a should not receive its own broadcast
        assert len(a.messages) == 0
        assert b.messages == [("a", "hey everyone")]
        assert c.messages == [("a", "hey everyone")]
        rec.close()

    async def test_broadcast_to_specific_targets(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        a = MockAgent()
        b = MockAgent()
        c = MockAgent()
        router.register("a", a)
        router.register("b", b)
        router.register("c", c)

        await router.broadcast("a", "just b", targets=["b"])
        await asyncio.sleep(0.05)

        assert b.messages == [("a", "just b")]
        assert len(c.messages) == 0
        rec.close()

    async def test_broadcast_concurrent(self, tmp_path: Path) -> None:
        """Broadcast dispatches concurrently — both slow agents should finish
        in roughly the same wall-clock time as one."""
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        s1 = SlowAgent(delay=0.1)
        s2 = SlowAgent(delay=0.1)
        router.register("sender", MockAgent())
        router.register("s1", s1)
        router.register("s2", s2)

        await router.broadcast("sender", "parallel")
        # Both tasks are running concurrently via gather, wait a bit
        await asyncio.sleep(0.2)
        assert len(s1.messages) == 1
        assert len(s2.messages) == 1
        rec.close()

    async def test_broadcast_records_messages(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        router.register("a", MockAgent())
        router.register("b", MockAgent())
        router.register("c", MockAgent())

        await router.broadcast("a", "recorded")
        await asyncio.sleep(0.05)

        events = _read_events(rec)
        message_events = [e for e in events if e["type"] == "message"]
        assert len(message_events) == 2  # b and c
        rec.close()

    async def test_broadcast_skips_disallowed_routes(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        topo = TopologyConfig(type="hub", coordinator="coord", workers=["w1", "w2"])
        router = Router(topo, rec)
        w1 = MockAgent()
        w2 = MockAgent()
        coord = MockAgent()
        router.register("coord", coord)
        router.register("w1", w1)
        router.register("w2", w2)

        # w1 broadcasting: only coord is reachable, not w2
        await router.broadcast("w1", "broadcast from worker")
        await asyncio.sleep(0.05)

        assert coord.messages == [("w1", "broadcast from worker")]
        assert len(w2.messages) == 0
        rec.close()


class TestErrorIsolation:
    async def test_failing_agent_doesnt_block_others(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        good = MockAgent()
        bad = FailingAgent()
        router.register("sender", MockAgent())
        router.register("good", good)
        router.register("bad", bad)

        # Broadcast to both — bad should fail, good should still receive
        await router.broadcast("sender", "test isolation")
        await asyncio.sleep(0.1)

        assert good.messages == [("sender", "test isolation")]
        rec.close()


class TestEdgeCases:
    async def test_empty_content(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        a = MockAgent()
        router.register("a", a)
        router.register("b", MockAgent())

        await router.send("b", "a", "")
        await asyncio.sleep(0.05)
        assert a.messages == [("b", "")]
        rec.close()

    async def test_self_send_mesh(self, tmp_path: Path) -> None:
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        a = MockAgent()
        router.register("a", a)

        await router.send("a", "a", "self-talk")
        await asyncio.sleep(0.05)
        assert a.messages == [("a", "self-talk")]
        rec.close()

    async def test_broadcast_no_targets(self, tmp_path: Path) -> None:
        """Broadcast with empty targets list does nothing."""
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        router.register("a", MockAgent())

        await router.broadcast("a", "nobody", targets=[])
        events = _read_events(rec)
        message_events = [e for e in events if e["type"] == "message"]
        assert len(message_events) == 0
        rec.close()

    async def test_agent_protocol_check(self) -> None:
        """MockAgent should satisfy the Agent protocol."""
        assert isinstance(MockAgent(), Agent)


# ===================================================================
# Response channel tests
# ===================================================================


class TestResponseChannels:
    async def test_expect_response_creates_channel(self, tmp_path: Path) -> None:
        """expect_response should create a pending future."""
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        router.register("a", MockAgent())
        router.register("b", MockAgent())

        future = router.expect_response(from_agent="b", to_agent="a")
        assert not future.done()
        # Clean up
        router.cancel_response(from_agent="b", to_agent="a")
        rec.close()

    async def test_send_resolves_response_channel(self, tmp_path: Path) -> None:
        """Matching send() resolves future, skips handle_message."""
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        a = MockAgent()
        b = MockAgent()
        router.register("a", a)
        router.register("b", b)

        # Agent "a" is waiting for "b" to respond.
        future = router.expect_response(from_agent="b", to_agent="a")

        # "b" sends a reply to "a" — this should resolve the future.
        await router.send("b", "a", "here's the reply")
        await asyncio.sleep(0.05)

        assert future.done()
        assert future.result() == "here's the reply"
        # handle_message should NOT have been called on "a".
        assert len(a.messages) == 0
        rec.close()

    async def test_send_without_channel_dispatches_normally(
        self, tmp_path: Path,
    ) -> None:
        """No channel = normal handle_message dispatch."""
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        a = MockAgent()
        router.register("a", a)
        router.register("b", MockAgent())

        await router.send("b", "a", "normal message")
        await asyncio.sleep(0.05)

        assert a.messages == [("b", "normal message")]
        rec.close()

    async def test_duplicate_channel_raises(self, tmp_path: Path) -> None:
        """Double expect_response for the same pair should raise RuntimeError."""
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        router.register("a", MockAgent())
        router.register("b", MockAgent())

        router.expect_response(from_agent="b", to_agent="a")
        with pytest.raises(RuntimeError, match="Duplicate response channel"):
            router.expect_response(from_agent="b", to_agent="a")
        # Clean up
        router.cancel_response(from_agent="b", to_agent="a")
        rec.close()

    async def test_cancel_response_cleans_up(self, tmp_path: Path) -> None:
        """cancel_response should remove the channel and cancel the future."""
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        router.register("a", MockAgent())
        router.register("b", MockAgent())

        future = router.expect_response(from_agent="b", to_agent="a")
        router.cancel_response(from_agent="b", to_agent="a")

        assert future.cancelled()
        # Should be able to register again after cancel.
        future2 = router.expect_response(from_agent="b", to_agent="a")
        assert not future2.done()
        router.cancel_response(from_agent="b", to_agent="a")
        rec.close()

    async def test_cancel_all_responses(self, tmp_path: Path) -> None:
        """cancel_all_responses should cancel all pending futures."""
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        router.register("a", MockAgent())
        router.register("b", MockAgent())
        router.register("c", MockAgent())

        f1 = router.expect_response(from_agent="b", to_agent="a")
        f2 = router.expect_response(from_agent="c", to_agent="a")

        router.cancel_all_responses()

        assert f1.cancelled()
        assert f2.cancelled()
        rec.close()

    async def test_failed_task_resolves_waiting_channel(
        self, tmp_path: Path,
    ) -> None:
        """Failed dispatch resolves waiting channels with error."""
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        bad = FailingAgent()
        waiter = MockAgent()
        router.register("bad", bad)
        router.register("waiter", waiter)

        # waiter is expecting a response from bad.
        future = router.expect_response(from_agent="bad", to_agent="waiter")

        # Send a message TO bad — bad will raise in handle_message.
        await router.send("waiter", "bad", "do something")
        # Wait for the task callback to fire.
        await asyncio.sleep(0.1)

        assert future.done()
        result = future.result()
        assert "Error" in result
        assert "bad" in result
        rec.close()

    async def test_response_channel_message_still_recorded(
        self, tmp_path: Path,
    ) -> None:
        """Intercepted messages are still recorded to JSONL."""
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        router.register("a", MockAgent())
        router.register("b", MockAgent())

        router.expect_response(from_agent="b", to_agent="a")
        await router.send("b", "a", "intercepted reply")
        await asyncio.sleep(0.05)

        events = _read_events(rec)
        message_events = [e for e in events if e["type"] == "message"]
        assert len(message_events) == 1
        assert message_events[0]["from"] == "b"
        assert message_events[0]["to"] == "a"
        assert message_events[0]["content"] == "intercepted reply"
        rec.close()

    async def test_responding_agent_round_trip(self, tmp_path: Path) -> None:
        """End-to-end: expect_response + send + agent replies = future resolved."""
        rec = _make_recorder(tmp_path)
        router = Router(TopologyConfig(type="mesh"), rec)
        lead = MockAgent()
        router.register("lead", lead)

        # dev will reply "done" when it receives a message.
        dev = RespondingAgent(router, "dev", reply="done", delay=0.02)
        router.register("dev", dev)

        # lead expects a response from dev.
        future = router.expect_response(from_agent="dev", to_agent="lead")

        # lead sends a task to dev (this dispatches normally since there's
        # no channel for (lead, dev)).
        await router.send("lead", "dev", "implement feature X")

        # Wait for dev to process and reply.
        result = await asyncio.wait_for(future, timeout=1.0)

        assert result == "done"
        assert dev.messages == [("lead", "implement feature X")]
        # lead's handle_message not called (channel intercepted).
        assert len(lead.messages) == 0
        rec.close()
