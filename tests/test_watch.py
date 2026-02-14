"""Tests for the watch command and TUI."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lattice.commands.watch import (
    AgentState,
    MessageLink,
    SessionStats,
    WatchApp,
    _find_latest_session,
)


def test_agent_state_initialization():
    """Test AgentState dataclass initialization."""
    agent = AgentState(name="test-agent", agent_type="llm")
    assert agent.name == "test-agent"
    assert agent.agent_type == "llm"
    assert agent.active is False
    assert agent.current_activity == "standby"


def test_message_link_initialization():
    """Test MessageLink dataclass initialization."""
    msg = MessageLink(
        from_agent="agent1",
        to_agent="agent2",
        content="Hello",
    )
    assert msg.from_agent == "agent1"
    assert msg.to_agent == "agent2"
    assert msg.content == "Hello"
    assert msg.completed is False


def test_session_stats_initialization():
    """Test SessionStats dataclass initialization."""
    stats = SessionStats()
    assert stats.start_time is None
    assert stats.message_count == 0
    assert stats.total_tokens == {"input": 0, "output": 0}
    assert stats.loop_iteration == 0
    assert stats.session_ended is False


def test_watch_app_initialization(tmp_path: Path):
    """Test WatchApp initialization."""
    session_file = tmp_path / "test_session.jsonl"
    session_file.write_text('{"type": "session_start", "ts": "2026-01-01T00:00:00.000Z", "seq": 0, "session_id": "test", "team": "test", "config_hash": "abc123"}\\n')

    app = WatchApp(session_file=session_file, enable_input=False)
    assert app.session_file == session_file
    assert app.enable_input is False
    assert isinstance(app.agents, dict)
    assert isinstance(app.messages, list)
    assert isinstance(app.stats, SessionStats)


def test_find_latest_session_no_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test _find_latest_session with no session files."""
    # Change to tmp directory with no sessions
    monkeypatch.chdir(tmp_path)
    result = _find_latest_session()
    assert result is None


def test_find_latest_session_with_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test _find_latest_session returns most recent session."""
    monkeypatch.chdir(tmp_path)

    # Create sessions directory
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir()

    # Create test session files with different timestamps
    older_file = sessions_dir / "2026-01-01_team_abc123.jsonl"
    newer_file = sessions_dir / "2026-01-02_team_def456.jsonl"
    verbose_file = sessions_dir / "2026-01-02_team_def456.verbose.jsonl"

    older_file.write_text("")
    newer_file.write_text("")
    verbose_file.write_text("")

    result = _find_latest_session()

    # Should return newer_file, not verbose_file
    assert result is not None
    assert result.name == "2026-01-02_team_def456.jsonl"


def test_process_event_session_start(tmp_path: Path):
    """Test processing a session_start event."""
    session_file = tmp_path / "test.jsonl"
    session_file.write_text("")

    app = WatchApp(session_file=session_file)
    event = {
        "type": "session_start",
        "ts": "2026-02-14T12:00:00.000Z",
        "seq": 0,
        "session_id": "test123",
        "team": "test-team",
        "config_hash": "abc",
    }

    app._process_event(event)

    assert app.stats.start_time is not None


def test_process_event_agent_start(tmp_path: Path):
    """Test processing an agent_start event."""
    session_file = tmp_path / "test.jsonl"
    session_file.write_text("")

    app = WatchApp(session_file=session_file)
    event = {
        "type": "agent_start",
        "ts": "2026-02-14T12:00:00.000Z",
        "seq": 1,
        "agent": "test-agent",
        "agent_type": "llm",
    }

    app._process_event(event)

    assert "test-agent" in app.agents
    assert app.agents["test-agent"].active is True
    assert app.agents["test-agent"].agent_type == "llm"


def test_process_event_message(tmp_path: Path):
    """Test processing a message event."""
    session_file = tmp_path / "test.jsonl"
    session_file.write_text("")

    app = WatchApp(session_file=session_file)
    event = {
        "type": "message",
        "ts": "2026-02-14T12:00:00.000Z",
        "seq": 2,
        "from": "agent1",
        "to": "agent2",
        "content": "Hello there",
    }

    app._process_event(event)

    assert len(app.messages) == 1
    assert app.messages[0].from_agent == "agent1"
    assert app.messages[0].to_agent == "agent2"
    assert app.stats.message_count == 1


def test_process_event_llm_call_end(tmp_path: Path):
    """Test processing an llm_call_end event."""
    session_file = tmp_path / "test.jsonl"
    session_file.write_text("")

    app = WatchApp(session_file=session_file)
    event = {
        "type": "llm_call_end",
        "ts": "2026-02-14T12:00:00.000Z",
        "seq": 3,
        "agent": "test-agent",
        "model": "claude-sonnet-4-5",
        "tokens": {"input": 100, "output": 50},
        "duration_ms": 1500,
    }

    app._process_event(event)

    assert app.stats.total_tokens["input"] == 100
    assert app.stats.total_tokens["output"] == 50


def test_process_event_loop_boundary(tmp_path: Path):
    """Test processing loop boundary events."""
    session_file = tmp_path / "test.jsonl"
    session_file.write_text("")

    app = WatchApp(session_file=session_file)

    # Loop start
    event_start = {
        "type": "loop_boundary",
        "ts": "2026-02-14T12:00:00.000Z",
        "seq": 4,
        "boundary": "start",
        "iteration": 2,
    }
    app._process_event(event_start)
    assert app.stats.loop_iteration == 2

    # Loop end
    event_end = {
        "type": "loop_boundary",
        "ts": "2026-02-14T12:01:00.000Z",
        "seq": 5,
        "boundary": "end",
        "iteration": 2,
    }
    app._process_event(event_end)
    assert app.stats.loop_iteration == 2  # Should stay at 2
