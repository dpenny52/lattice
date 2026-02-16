"""Pidfile management for cross-process shutdown communication."""

from __future__ import annotations

import json
import os
from pathlib import Path

#: Pidfile directory — relative to CWD so it's project-scoped.
#: Both ``lattice up`` and ``lattice down`` must be run from the same directory.
PIDFILE_DIR = Path(".lattice")
PIDFILE_NAME = "session.pid"


def write_pidfile(session_id: str, team: str) -> Path:
    """Write pidfile with current PID, session_id, and team name.
    Returns the pidfile path."""
    PIDFILE_DIR.mkdir(parents=True, exist_ok=True)
    pidfile = PIDFILE_DIR / PIDFILE_NAME
    data = {
        "pid": os.getpid(),
        "session_id": session_id,
        "team": team,
    }
    pidfile.write_text(json.dumps(data))
    return pidfile


def read_pidfile() -> dict[str, object] | None:
    """Read and return pidfile contents, or None if not found."""
    pidfile = PIDFILE_DIR / PIDFILE_NAME
    if not pidfile.exists():
        return None
    try:
        result: dict[str, object] = json.loads(pidfile.read_text())
        return result
    except (json.JSONDecodeError, OSError):
        return None


def remove_pidfile() -> None:
    """Remove the pidfile if it exists."""
    pidfile = PIDFILE_DIR / PIDFILE_NAME
    pidfile.unlink(missing_ok=True)


def is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)  # Signal 0 = check existence
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # Process exists but we can't signal it
