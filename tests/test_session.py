"""Tests for session event models and the JSONL recorder."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from pydantic import TypeAdapter

from lattice.session.models import (
    AgentDoneEvent,
    AgentStartEvent,
    ErrorEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
    MessageEvent,
    SessionEndEvent,
    SessionEvent,
    SessionStartEvent,
    StatusEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from lattice.session.recorder import SessionRecorder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EVENT_ADAPTER: TypeAdapter[SessionEvent] = TypeAdapter(SessionEvent)


def _base(seq: int = 0) -> dict[str, Any]:
    """Return the common envelope fields."""
    return {"ts": "2026-02-14T12:00:00.000Z", "seq": seq}


# ===================================================================
# Event serialization tests
# ===================================================================


class TestSessionStartEvent:
    def test_round_trip(self) -> None:
        evt = SessionStartEvent(
            **_base(),
            session_id="abc123",
            team="my-team",
            config_hash="deadbeef",
        )
        data = json.loads(evt.model_dump_json(by_alias=True))
        assert data["type"] == "session_start"
        assert data["session_id"] == "abc123"
        assert data["team"] == "my-team"
        assert data["config_hash"] == "deadbeef"

    def test_discriminated_union(self) -> None:
        raw = {
            **_base(),
            "type": "session_start",
            "session_id": "abc",
            "team": "t",
            "config_hash": "h",
        }
        evt = _EVENT_ADAPTER.validate_python(raw)
        assert isinstance(evt, SessionStartEvent)


class TestSessionEndEvent:
    def test_round_trip(self) -> None:
        evt = SessionEndEvent(
            **_base(),
            reason="complete",
            duration_ms=5000,
            total_tokens={"input": 100, "output": 200},
        )
        data = json.loads(evt.model_dump_json(by_alias=True))
        assert data["type"] == "session_end"
        assert data["reason"] == "complete"
        assert data["duration_ms"] == 5000
        assert data["total_tokens"] == {"input": 100, "output": 200}


class TestMessageEvent:
    def test_from_alias(self) -> None:
        """``from_agent`` in Python serializes to ``from`` in JSON."""
        evt = MessageEvent(
            **_base(),
            from_agent="agent-a",
            to="agent-b",
            content="hello",
        )
        data = json.loads(evt.model_dump_json(by_alias=True))
        assert data["from"] == "agent-a"
        assert "from_agent" not in data

    def test_construct_by_alias(self) -> None:
        """Can construct via the alias key ``from``."""
        raw: dict[str, Any] = {
            **_base(),
            "type": "message",
            "from": "agent-a",
            "to": "agent-b",
            "content": "hi",
        }
        evt = _EVENT_ADAPTER.validate_python(raw)
        assert isinstance(evt, MessageEvent)
        assert evt.from_agent == "agent-a"

    def test_construct_by_field_name(self) -> None:
        """Can also construct with the Python field name."""
        evt = MessageEvent(
            **_base(),
            from_agent="agent-a",
            to="agent-b",
            content="hi",
        )
        assert evt.from_agent == "agent-a"


class TestLLMCallStartEvent:
    def test_round_trip(self) -> None:
        evt = LLMCallStartEvent(
            **_base(), agent="a", model="anthropic/claude-sonnet-4-5", messages_count=5
        )
        data = json.loads(evt.model_dump_json(by_alias=True))
        assert data["type"] == "llm_call_start"
        assert data["messages_count"] == 5


class TestLLMCallEndEvent:
    def test_round_trip(self) -> None:
        evt = LLMCallEndEvent(
            **_base(),
            agent="a",
            model="anthropic/claude-sonnet-4-5",
            tokens={"input": 10, "output": 20},
            duration_ms=300,
        )
        data = json.loads(evt.model_dump_json(by_alias=True))
        assert data["type"] == "llm_call_end"
        assert data["tokens"] == {"input": 10, "output": 20}


class TestToolCallEvent:
    def test_round_trip(self) -> None:
        evt = ToolCallEvent(
            **_base(), agent="a", tool="read_file", args={"path": "/tmp/x"}
        )
        data = json.loads(evt.model_dump_json(by_alias=True))
        assert data["type"] == "tool_call"
        assert data["args"] == {"path": "/tmp/x"}


class TestToolResultEvent:
    def test_round_trip(self) -> None:
        evt = ToolResultEvent(
            **_base(), agent="a", tool="read_file", duration_ms=42, result_size=1024
        )
        data = json.loads(evt.model_dump_json(by_alias=True))
        assert data["type"] == "tool_result"
        assert data["result_size"] == 1024


class TestStatusEvent:
    def test_round_trip(self) -> None:
        evt = StatusEvent(**_base(), agent="a", status="thinking")
        data = json.loads(evt.model_dump_json(by_alias=True))
        assert data["type"] == "status"
        assert data["status"] == "thinking"

    def test_mood_default(self) -> None:
        evt = StatusEvent(**_base(), agent="a", status="idle")
        assert evt.mood == "🤔"
        data = json.loads(evt.model_dump_json(by_alias=True))
        assert data["mood"] == "🤔"

    def test_mood_custom(self) -> None:
        evt = StatusEvent(**_base(), agent="a", status="working", mood="🔥")
        assert evt.mood == "🔥"
        data = json.loads(evt.model_dump_json(by_alias=True))
        assert data["mood"] == "🔥"


class TestErrorEvent:
    def test_round_trip(self) -> None:
        evt = ErrorEvent(**_base(), agent="a", error="timeout", retrying=True)
        data = json.loads(evt.model_dump_json(by_alias=True))
        assert data["type"] == "error"
        assert data["retrying"] is True


class TestAgentStartEvent:
    def test_round_trip(self) -> None:
        evt = AgentStartEvent(**_base(), agent="researcher", agent_type="llm")
        data = json.loads(evt.model_dump_json(by_alias=True))
        assert data["type"] == "agent_start"
        assert data["agent_type"] == "llm"
        # The envelope `type` and the payload `agent_type` must not collide
        assert data["type"] != data["agent_type"]


class TestAgentDoneEvent:
    def test_round_trip(self) -> None:
        evt = AgentDoneEvent(**_base(), agent="researcher", reason="task_complete")
        data = json.loads(evt.model_dump_json(by_alias=True))
        assert data["type"] == "agent_done"
        assert data["reason"] == "task_complete"


class TestDiscriminatedUnion:
    """Verify the union type can deserialize every event variant."""

    @pytest.mark.parametrize(
        ("raw", "expected_cls"),
        [
            pytest.param(
                {
                    **_base(),
                    "type": "session_start",
                    "session_id": "x",
                    "team": "t",
                    "config_hash": "h",
                },
                SessionStartEvent,
                id="session_start",
            ),
            pytest.param(
                {
                    **_base(),
                    "type": "session_end",
                    "reason": "complete",
                    "duration_ms": 1,
                    "total_tokens": {"input": 0, "output": 0},
                },
                SessionEndEvent,
                id="session_end",
            ),
            pytest.param(
                {
                    **_base(),
                    "type": "message",
                    "from": "a",
                    "to": "b",
                    "content": "hi",
                },
                MessageEvent,
                id="message",
            ),
            pytest.param(
                {
                    **_base(),
                    "type": "llm_call_start",
                    "agent": "a",
                    "model": "m",
                    "messages_count": 1,
                },
                LLMCallStartEvent,
                id="llm_call_start",
            ),
            pytest.param(
                {
                    **_base(),
                    "type": "llm_call_end",
                    "agent": "a",
                    "model": "m",
                    "tokens": {"input": 0, "output": 0},
                    "duration_ms": 1,
                },
                LLMCallEndEvent,
                id="llm_call_end",
            ),
            pytest.param(
                {
                    **_base(),
                    "type": "tool_call",
                    "agent": "a",
                    "tool": "t",
                    "args": {},
                },
                ToolCallEvent,
                id="tool_call",
            ),
            pytest.param(
                {
                    **_base(),
                    "type": "tool_result",
                    "agent": "a",
                    "tool": "t",
                    "duration_ms": 1,
                    "result_size": 0,
                },
                ToolResultEvent,
                id="tool_result",
            ),
            pytest.param(
                {
                    **_base(),
                    "type": "status",
                    "agent": "a",
                    "status": "ok",
                },
                StatusEvent,
                id="status",
            ),
            pytest.param(
                {
                    **_base(),
                    "type": "error",
                    "agent": "a",
                    "error": "oops",
                    "retrying": False,
                },
                ErrorEvent,
                id="error",
            ),
            pytest.param(
                {**_base(), "type": "agent_start", "agent": "a", "agent_type": "llm"},
                AgentStartEvent,
                id="agent_start",
            ),
            pytest.param(
                {**_base(), "type": "agent_done", "agent": "a", "reason": "done"},
                AgentDoneEvent,
                id="agent_done",
            ),
        ],
    )
    def test_union_resolves(self, raw: dict[str, Any], expected_cls: type[Any]) -> None:
        evt = _EVENT_ADAPTER.validate_python(raw)
        assert isinstance(evt, expected_cls)


class TestEnvelopeFields:
    """Every event must carry ``ts``, ``seq``, and ``type``."""

    def test_fields_present_in_json(self) -> None:
        evt = StatusEvent(**_base(seq=7), agent="a", status="ok")
        data = json.loads(evt.model_dump_json(by_alias=True))
        assert data["ts"] == "2026-02-14T12:00:00.000Z"
        assert data["seq"] == 7
        assert "type" in data


# ===================================================================
# SessionRecorder tests
# ===================================================================


class TestRecorderCreation:
    def test_creates_sessions_dir(self, tmp_path: Path) -> None:
        sdir = tmp_path / "sess"
        assert not sdir.exists()
        rec = SessionRecorder("team1", "abc", sessions_dir=sdir)
        assert sdir.is_dir()
        rec.close()

    def test_unique_session_ids(self, tmp_path: Path) -> None:
        r1 = SessionRecorder("t", "h", sessions_dir=tmp_path)
        r2 = SessionRecorder("t", "h", sessions_dir=tmp_path)
        assert r1.session_id != r2.session_id
        r1.close()
        r2.close()

    def test_session_id_length(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        assert len(rec.session_id) == 12
        rec.close()

    def test_file_naming(self, tmp_path: Path) -> None:
        rec = SessionRecorder("my-team", "hash", sessions_dir=tmp_path)
        name = rec.session_file.name
        # Pattern: YYYY-MM-DD_my-team_<12hex>.jsonl
        assert name.endswith(".jsonl")
        assert "_my-team_" in name
        assert len(rec.session_id) == 12
        rec.close()


class TestRecorderSessionStart:
    def test_first_event_is_session_start(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        lines = rec.session_file.read_text().strip().splitlines()
        assert len(lines) == 1
        first = json.loads(lines[0])
        assert first["type"] == "session_start"
        assert first["seq"] == 0
        assert first["session_id"] == rec.session_id
        rec.close()


class TestRecorderSequencing:
    def test_monotonic_seq(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        rec.record(StatusEvent(ts="", seq=0, agent="a", status="one"))
        rec.record(StatusEvent(ts="", seq=0, agent="a", status="two"))
        rec.close()

        lines = rec.session_file.read_text().strip().splitlines()
        seqs = [json.loads(line)["seq"] for line in lines]
        assert seqs == [0, 1, 2]

    def test_iso_timestamps(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        rec.record(StatusEvent(ts="", seq=0, agent="a", status="x"))
        rec.close()

        lines = rec.session_file.read_text().strip().splitlines()
        for line in lines:
            ts = json.loads(line)["ts"]
            # ISO 8601 basic shape: YYYY-MM-DDTHH:MM:SS.mmmZ
            assert "T" in ts
            assert ts.endswith("Z")


class TestRecorderJSONL:
    def test_each_line_is_valid_json(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        rec.record(StatusEvent(ts="", seq=0, agent="a", status="x"))
        rec.record(ErrorEvent(ts="", seq=0, agent="a", error="boom", retrying=False))
        rec.close()

        for line in rec.session_file.read_text().strip().splitlines():
            data = json.loads(line)
            assert isinstance(data, dict)

    def test_flush_mid_session(self, tmp_path: Path) -> None:
        """File should be readable while the session is still active."""
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        rec.record(StatusEvent(ts="", seq=0, agent="a", status="mid"))
        # Read the file while recorder is still open
        lines = rec.session_file.read_text().strip().splitlines()
        assert len(lines) == 2  # session_start + status
        rec.close()


class TestRecorderEnd:
    def test_end_writes_session_end(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        rec.end("complete", total_tokens={"input": 100, "output": 200})

        lines = rec.session_file.read_text().strip().splitlines()
        last = json.loads(lines[-1])
        assert last["type"] == "session_end"
        assert last["reason"] == "complete"
        assert last["total_tokens"] == {"input": 100, "output": 200}

    def test_end_has_duration_ms(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        time.sleep(0.01)  # small delay so duration > 0
        rec.end("complete")

        lines = rec.session_file.read_text().strip().splitlines()
        last = json.loads(lines[-1])
        assert last["duration_ms"] >= 0

    def test_end_default_tokens(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        rec.end("user_shutdown")

        lines = rec.session_file.read_text().strip().splitlines()
        last = json.loads(lines[-1])
        assert last["total_tokens"] == {"input": 0, "output": 0}


class TestRecorderCloseWithoutEnd:
    def test_no_session_end_event(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        rec.record(StatusEvent(ts="", seq=0, agent="a", status="x"))
        rec.close()

        lines = rec.session_file.read_text().strip().splitlines()
        types = [json.loads(line)["type"] for line in lines]
        assert "session_end" not in types


class TestRecorderVerbose:
    def test_verbose_creates_sidecar(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path, verbose=True)
        verbose_files = list(tmp_path.glob("*.verbose.jsonl"))
        assert len(verbose_files) == 1
        rec.close()

    def test_verbose_entries(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path, verbose=True)
        rec.record_verbose(0, {"full": "payload"})
        rec.close()

        verbose_files = list(tmp_path.glob("*.verbose.jsonl"))
        content = verbose_files[0].read_text().strip()
        entry = json.loads(content)
        assert entry["seq"] == 0
        assert entry["full_result"] == {"full": "payload"}

    def test_non_verbose_no_sidecar(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path, verbose=False)
        rec.record_verbose(0, {"ignored": True})
        rec.close()

        verbose_files = list(tmp_path.glob("*.verbose.jsonl"))
        assert len(verbose_files) == 0


class TestRecorderIdempotent:
    def test_end_twice_is_safe(self, tmp_path: Path) -> None:
        """Calling end() twice should not raise or write a second session_end."""
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        rec.end("complete")
        rec.end("complete")  # second call is a no-op

        lines = rec.session_file.read_text().strip().splitlines()
        end_events = [line for line in lines if '"session_end"' in line]
        assert len(end_events) == 1

    def test_record_after_close_is_safe(self, tmp_path: Path) -> None:
        """Recording after close should silently drop the event."""
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        rec.close()
        # This should not raise
        rec.record(StatusEvent(ts="", seq=0, agent="a", status="dropped"))

        lines = rec.session_file.read_text().strip().splitlines()
        types = [json.loads(line)["type"] for line in lines]
        assert "status" not in types

    def test_record_after_end_is_safe(self, tmp_path: Path) -> None:
        """Recording after end should silently drop the event."""
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        rec.end("complete")
        rec.record(StatusEvent(ts="", seq=0, agent="a", status="dropped"))

        lines = rec.session_file.read_text().strip().splitlines()
        last = json.loads(lines[-1])
        assert last["type"] == "session_end"


class TestRecorderPathSafety:
    def test_rejects_path_traversal_team_name(self, tmp_path: Path) -> None:
        """Team names with path traversal characters are rejected."""
        with pytest.raises(ValueError, match="Invalid team name"):
            SessionRecorder("../../etc/evil", "h", sessions_dir=tmp_path)

    def test_rejects_slash_in_team_name(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Invalid team name"):
            SessionRecorder("foo/bar", "h", sessions_dir=tmp_path)

    def test_allows_valid_team_names(self, tmp_path: Path) -> None:
        for name in ["my-team", "team_1", "ResearchSquad", "a"]:
            rec = SessionRecorder(name, "h", sessions_dir=tmp_path)
            rec.close()


class TestRecorderThreadSafety:
    def test_concurrent_writes(self, tmp_path: Path) -> None:
        rec = SessionRecorder("t", "h", sessions_dir=tmp_path)
        n_threads = 8
        events_per_thread = 50
        barrier = threading.Barrier(n_threads)

        def writer() -> None:
            barrier.wait()
            for _ in range(events_per_thread):
                rec.record(StatusEvent(ts="", seq=0, agent="a", status="go"))

        threads = [threading.Thread(target=writer) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        rec.close()

        lines = rec.session_file.read_text().strip().splitlines()
        # 1 session_start + n_threads * events_per_thread
        expected = 1 + n_threads * events_per_thread
        assert len(lines) == expected

        # Every line must be valid JSON
        for line in lines:
            json.loads(line)

        # Sequence numbers should be a complete range with no gaps
        seqs = sorted(json.loads(line)["seq"] for line in lines)
        assert seqs == list(range(expected))
