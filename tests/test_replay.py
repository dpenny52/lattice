"""Tests for lattice replay command."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from lattice.commands.replay import (
    SessionData,
    SessionMetadata,
    _extract_metadata,
    _list_sessions,
    _parse_event,
    _parse_session_file,
    replay,
)
from lattice.session.models import (
    AgentStartEvent,
    LLMCallEndEvent,
    MessageEvent,
    SessionEndEvent,
    SessionStartEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_session_file(
    path: Path,
    events: list[dict],
) -> None:
    """Write a session JSONL file with the given events."""
    with path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event) + "\n")


def _minimal_session(session_id: str, team: str) -> list[dict]:
    """Return a minimal valid session with start and end events."""
    return [
        {
            "ts": "2026-02-14T12:00:00.000Z",
            "seq": 0,
            "type": "session_start",
            "session_id": session_id,
            "team": team,
            "config_hash": "abc123",
        },
        {
            "ts": "2026-02-14T12:01:00.000Z",
            "seq": 1,
            "type": "session_end",
            "reason": "complete",
            "duration_ms": 60000,
            "total_tokens": {"input": 100, "output": 50},
        },
    ]


# ===================================================================
# Event parsing tests
# ===================================================================


class TestParseEvent:
    def test_parse_session_start(self) -> None:
        raw = {
            "ts": "2026-02-14T12:00:00.000Z",
            "seq": 0,
            "type": "session_start",
            "session_id": "abc123",
            "team": "my-team",
            "config_hash": "deadbeef",
        }
        event = _parse_event(raw)
        assert isinstance(event, SessionStartEvent)
        assert event.session_id == "abc123"
        assert event.team == "my-team"

    def test_parse_session_end(self) -> None:
        raw = {
            "ts": "2026-02-14T12:00:00.000Z",
            "seq": 1,
            "type": "session_end",
            "reason": "complete",
            "duration_ms": 5000,
            "total_tokens": {"input": 100, "output": 200},
        }
        event = _parse_event(raw)
        assert isinstance(event, SessionEndEvent)
        assert event.reason == "complete"
        assert event.duration_ms == 5000

    def test_parse_message(self) -> None:
        raw = {
            "ts": "2026-02-14T12:00:00.000Z",
            "seq": 2,
            "type": "message",
            "from": "agent-a",
            "to": "agent-b",
            "content": "hello",
        }
        event = _parse_event(raw)
        assert isinstance(event, MessageEvent)
        assert event.from_agent == "agent-a"
        assert event.to == "agent-b"
        assert event.content == "hello"

    def test_parse_llm_call_end(self) -> None:
        raw = {
            "ts": "2026-02-14T12:00:00.000Z",
            "seq": 3,
            "type": "llm_call_end",
            "agent": "agent-a",
            "model": "claude-3-5-sonnet-20241022",
            "tokens": {"input": 1000, "output": 500},
            "duration_ms": 2000,
        }
        event = _parse_event(raw)
        assert isinstance(event, LLMCallEndEvent)
        assert event.agent == "agent-a"
        assert event.tokens == {"input": 1000, "output": 500}


# ===================================================================
# Metadata extraction tests
# ===================================================================


class TestExtractMetadata:
    def test_basic_session(self, tmp_path: Path) -> None:
        events = [
            SessionStartEvent(
                ts="2026-02-14T12:00:00.000Z",
                seq=0,
                session_id="abc123",
                team="my-team",
                config_hash="deadbeef",
            ),
            SessionEndEvent(
                ts="2026-02-14T12:01:00.000Z",
                seq=1,
                reason="complete",
                duration_ms=60000,
                total_tokens={"input": 100, "output": 50},
            ),
        ]
        file_path = tmp_path / "session.jsonl"
        meta = _extract_metadata(file_path, events)

        assert meta.session_id == "abc123"
        assert meta.team == "my-team"
        assert meta.duration_ms == 60000
        assert meta.total_tokens == {"input": 100, "output": 50}
        assert meta.is_complete is True
        assert meta.event_count == 2
        assert meta.message_count == 0

    def test_in_progress_session(self, tmp_path: Path) -> None:
        events = [
            SessionStartEvent(
                ts="2026-02-14T12:00:00.000Z",
                seq=0,
                session_id="xyz789",
                team="test-team",
                config_hash="abc123",
            ),
        ]
        file_path = tmp_path / "session.jsonl"
        meta = _extract_metadata(file_path, events)

        assert meta.session_id == "xyz789"
        assert meta.is_complete is False
        assert meta.duration_ms is None
        assert meta.end_ts is None

    def test_message_counting(self, tmp_path: Path) -> None:
        events = [
            SessionStartEvent(
                ts="2026-02-14T12:00:00.000Z",
                seq=0,
                session_id="abc",
                team="t",
                config_hash="h",
            ),
            MessageEvent(
                ts="2026-02-14T12:00:01.000Z",
                seq=1,
                from_agent="a",
                to="b",
                content="msg1",
            ),
            MessageEvent(
                ts="2026-02-14T12:00:02.000Z",
                seq=2,
                from_agent="b",
                to="a",
                content="msg2",
            ),
        ]
        file_path = tmp_path / "session.jsonl"
        meta = _extract_metadata(file_path, events)

        assert meta.message_count == 2

    def test_token_accumulation(self, tmp_path: Path) -> None:
        """Token counts should be accumulated from LLM call events."""
        events = [
            SessionStartEvent(
                ts="2026-02-14T12:00:00.000Z",
                seq=0,
                session_id="abc",
                team="t",
                config_hash="h",
            ),
            LLMCallEndEvent(
                ts="2026-02-14T12:00:01.000Z",
                seq=1,
                agent="agent-a",
                model="claude-3-5-sonnet-20241022",
                tokens={"input": 100, "output": 50},
                duration_ms=1000,
            ),
            LLMCallEndEvent(
                ts="2026-02-14T12:00:02.000Z",
                seq=2,
                agent="agent-a",
                model="claude-3-5-sonnet-20241022",
                tokens={"input": 200, "output": 75},
                duration_ms=1500,
            ),
        ]
        file_path = tmp_path / "session.jsonl"
        meta = _extract_metadata(file_path, events)

        # Should accumulate tokens from both LLM calls
        assert meta.total_tokens == {"input": 300, "output": 125}

    def test_session_end_tokens_override(self, tmp_path: Path) -> None:
        """If session_end has non-zero tokens, prefer those."""
        events = [
            SessionStartEvent(
                ts="2026-02-14T12:00:00.000Z",
                seq=0,
                session_id="abc",
                team="t",
                config_hash="h",
            ),
            LLMCallEndEvent(
                ts="2026-02-14T12:00:01.000Z",
                seq=1,
                agent="agent-a",
                model="claude-3-5-sonnet-20241022",
                tokens={"input": 100, "output": 50},
                duration_ms=1000,
            ),
            SessionEndEvent(
                ts="2026-02-14T12:01:00.000Z",
                seq=2,
                reason="complete",
                duration_ms=60000,
                total_tokens={"input": 500, "output": 250},  # Non-zero, should win
            ),
        ]
        file_path = tmp_path / "session.jsonl"
        meta = _extract_metadata(file_path, events)

        # Should use session_end tokens since they're non-zero
        assert meta.total_tokens == {"input": 500, "output": 250}

    def test_session_end_zero_tokens_fallback(self, tmp_path: Path) -> None:
        """If session_end has zero tokens, use accumulated."""
        events = [
            SessionStartEvent(
                ts="2026-02-14T12:00:00.000Z",
                seq=0,
                session_id="abc",
                team="t",
                config_hash="h",
            ),
            LLMCallEndEvent(
                ts="2026-02-14T12:00:01.000Z",
                seq=1,
                agent="agent-a",
                model="claude-3-5-sonnet-20241022",
                tokens={"input": 100, "output": 50},
                duration_ms=1000,
            ),
            SessionEndEvent(
                ts="2026-02-14T12:01:00.000Z",
                seq=2,
                reason="ctrl_c",
                duration_ms=60000,
                total_tokens={"input": 0, "output": 0},  # Zero, should fallback
            ),
        ]
        file_path = tmp_path / "session.jsonl"
        meta = _extract_metadata(file_path, events)

        # Should use accumulated tokens since session_end is zero
        assert meta.total_tokens == {"input": 100, "output": 50}

    def test_agent_discovery(self, tmp_path: Path) -> None:
        """Should collect unique agent names from agent_start events."""
        events = [
            SessionStartEvent(
                ts="2026-02-14T12:00:00.000Z",
                seq=0,
                session_id="abc",
                team="t",
                config_hash="h",
            ),
            AgentStartEvent(
                ts="2026-02-14T12:00:01.000Z",
                seq=1,
                agent="agent-a",
                agent_type="llm",
            ),
            AgentStartEvent(
                ts="2026-02-14T12:00:02.000Z",
                seq=2,
                agent="agent-b",
                agent_type="cli",
            ),
            AgentStartEvent(
                ts="2026-02-14T12:00:03.000Z",
                seq=3,
                agent="agent-a",  # Duplicate, should be deduped
                agent_type="llm",
            ),
        ]
        file_path = tmp_path / "session.jsonl"
        meta = _extract_metadata(file_path, events)

        assert meta.agents == {"agent-a", "agent-b"}


# ===================================================================
# File parsing tests
# ===================================================================


class TestParseSessionFile:
    def test_valid_file(self, tmp_path: Path) -> None:
        session_file = tmp_path / "session.jsonl"
        events = _minimal_session("abc123", "my-team")
        _write_session_file(session_file, events)

        data = _parse_session_file(session_file)

        assert len(data.events) == 2
        assert data.metadata.session_id == "abc123"
        assert data.metadata.team == "my-team"
        assert len(data.parse_warnings) == 0

    def test_malformed_json_line(self, tmp_path: Path) -> None:
        session_file = tmp_path / "session.jsonl"
        with session_file.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(_minimal_session("abc", "t")[0]) + "\n")
            fh.write("{ invalid json }\n")  # Malformed line
            fh.write(json.dumps(_minimal_session("abc", "t")[1]) + "\n")

        data = _parse_session_file(session_file)

        # Should have 2 valid events and 1 warning
        assert len(data.events) == 2
        assert len(data.parse_warnings) == 1
        assert "Invalid JSON" in data.parse_warnings[0]

    def test_empty_lines_ignored(self, tmp_path: Path) -> None:
        session_file = tmp_path / "session.jsonl"
        with session_file.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(_minimal_session("abc", "t")[0]) + "\n")
            fh.write("\n")  # Empty line
            fh.write("  \n")  # Whitespace line
            fh.write(json.dumps(_minimal_session("abc", "t")[1]) + "\n")

        data = _parse_session_file(session_file)

        # Should have 2 events, empty lines ignored
        assert len(data.events) == 2
        assert len(data.parse_warnings) == 0


# ===================================================================
# Session listing tests
# ===================================================================


class TestListSessions:
    def test_empty_directory(self, tmp_path: Path) -> None:
        sessions = _list_sessions(tmp_path)
        assert len(sessions) == 0

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        sessions = _list_sessions(tmp_path / "nonexistent")
        assert len(sessions) == 0

    def test_list_multiple_sessions(self, tmp_path: Path) -> None:
        # Create 3 session files
        for i, sid in enumerate(["aaa", "bbb", "ccc"]):
            session_file = tmp_path / f"2026-02-14_team_{sid}.jsonl"
            events = _minimal_session(sid, f"team-{i}")
            _write_session_file(session_file, events)

        sessions = _list_sessions(tmp_path)

        assert len(sessions) == 3
        assert {s.session_id for s in sessions} == {"aaa", "bbb", "ccc"}

    def test_skip_verbose_files(self, tmp_path: Path) -> None:
        # Create regular session file
        session_file = tmp_path / "2026-02-14_team_abc.jsonl"
        _write_session_file(session_file, _minimal_session("abc", "team"))

        # Create verbose sidecar (should be ignored)
        verbose_file = tmp_path / "2026-02-14_team_abc.verbose.jsonl"
        _write_session_file(verbose_file, _minimal_session("abc", "team"))

        sessions = _list_sessions(tmp_path)

        # Should only find the main session file, not the verbose one
        assert len(sessions) == 1
        assert sessions[0].session_id == "abc"

    def test_sort_by_most_recent(self, tmp_path: Path) -> None:
        # Create sessions with different timestamps
        events_old = [
            {
                "ts": "2026-02-14T10:00:00.000Z",
                "seq": 0,
                "type": "session_start",
                "session_id": "old",
                "team": "t",
                "config_hash": "h",
            }
        ]
        events_new = [
            {
                "ts": "2026-02-14T12:00:00.000Z",
                "seq": 0,
                "type": "session_start",
                "session_id": "new",
                "team": "t",
                "config_hash": "h",
            }
        ]

        _write_session_file(tmp_path / "old.jsonl", events_old)
        _write_session_file(tmp_path / "new.jsonl", events_new)

        sessions = _list_sessions(tmp_path)

        # Should be sorted by start_ts, most recent first
        assert len(sessions) == 2
        assert sessions[0].session_id == "new"
        assert sessions[1].session_id == "old"


# ===================================================================
# CLI integration tests
# ===================================================================


class TestReplayCLI:
    def test_list_sessions(self, tmp_path: Path) -> None:
        # Create a test session
        session_file = tmp_path / "2026-02-14_team_abc123.jsonl"
        _write_session_file(session_file, _minimal_session("abc123", "my-team"))

        runner = CliRunner()
        result = runner.invoke(replay, ["--sessions-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "abc123" in result.output
        assert "my-team" in result.output

    def test_list_empty_directory(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(replay, ["--sessions-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "No sessions found" in result.output

    def test_load_specific_session(self, tmp_path: Path) -> None:
        # Create a test session
        session_file = tmp_path / "2026-02-14_team_xyz789.jsonl"
        _write_session_file(session_file, _minimal_session("xyz789", "test-team"))

        runner = CliRunner()
        result = runner.invoke(replay, ["xyz789", "--sessions-dir", str(tmp_path)])

        assert result.exit_code == 0
        assert "xyz789" in result.output
        assert "test-team" in result.output
        assert "Complete ✓" in result.output

    def test_session_not_found(self, tmp_path: Path) -> None:
        runner = CliRunner()
        result = runner.invoke(replay, ["nonexistent", "--sessions-dir", str(tmp_path)])

        assert result.exit_code == 1
        assert "Session not found" in result.output

    def test_ambiguous_session_id(self, tmp_path: Path) -> None:
        # Create two sessions with overlapping IDs
        _write_session_file(tmp_path / "2026-02-14_team_abc123.jsonl", _minimal_session("abc123", "t"))
        _write_session_file(tmp_path / "2026-02-14_team_abc456.jsonl", _minimal_session("abc456", "t"))

        runner = CliRunner()
        result = runner.invoke(replay, ["abc", "--sessions-dir", str(tmp_path)])

        # Should error because "abc" matches both sessions
        assert result.exit_code == 1
        assert "Ambiguous" in result.output

    def test_verbose_mode(self, tmp_path: Path) -> None:
        session_file = tmp_path / "2026-02-14_team_test123.jsonl"
        _write_session_file(session_file, _minimal_session("test123", "test-team"))

        runner = CliRunner()
        result = runner.invoke(replay, ["test123", "--sessions-dir", str(tmp_path), "-v"])

        assert result.exit_code == 0
        assert "Events" in result.output
        assert "session_start" in result.output
        assert "session_end" in result.output


# ===================================================================
# Duration formatting tests
# ===================================================================


class TestDurationFormatting:
    def test_seconds(self, tmp_path: Path) -> None:
        meta = SessionMetadata(
            session_id="abc",
            team="t",
            start_ts=pytest.approx,  # type: ignore[arg-type]
            end_ts=None,
            duration_ms=5000,  # 5 seconds
            message_count=0,
            total_tokens={"input": 0, "output": 0},
            agents=set(),
            event_count=0,
            file_path=tmp_path,
            is_complete=True,
        )
        assert "5.0s" in meta.duration_str

    def test_minutes(self, tmp_path: Path) -> None:
        meta = SessionMetadata(
            session_id="abc",
            team="t",
            start_ts=pytest.approx,  # type: ignore[arg-type]
            end_ts=None,
            duration_ms=90000,  # 90 seconds = 1.5 minutes
            message_count=0,
            total_tokens={"input": 0, "output": 0},
            agents=set(),
            event_count=0,
            file_path=tmp_path,
            is_complete=True,
        )
        assert "1.5m" in meta.duration_str

    def test_hours(self, tmp_path: Path) -> None:
        meta = SessionMetadata(
            session_id="abc",
            team="t",
            start_ts=pytest.approx,  # type: ignore[arg-type]
            end_ts=None,
            duration_ms=7200000,  # 2 hours
            message_count=0,
            total_tokens={"input": 0, "output": 0},
            agents=set(),
            event_count=0,
            file_path=tmp_path,
            is_complete=True,
        )
        assert "2.0h" in meta.duration_str

    def test_in_progress(self, tmp_path: Path) -> None:
        meta = SessionMetadata(
            session_id="abc",
            team="t",
            start_ts=pytest.approx,  # type: ignore[arg-type]
            end_ts=None,
            duration_ms=None,
            message_count=0,
            total_tokens={"input": 0, "output": 0},
            agents=set(),
            event_count=0,
            file_path=tmp_path,
            is_complete=False,
        )
        assert meta.duration_str == "in progress"
