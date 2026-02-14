"""Session recorder — append-only JSONL writer for session events."""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Literal

from lattice.session.models import SessionEndEvent, SessionEvent, SessionStartEvent

#: Valid team name pattern — alphanumeric, hyphens, underscores only.
_SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")

EndReason = Literal["complete", "user_shutdown", "ctrl_c", "error"]


class SessionRecorder:
    """Records session events to an append-only JSONL file.

    Thread-safe: all writes are serialized through a ``threading.Lock``.
    Crash-safe: the file is flushed after every event.
    """

    def __init__(
        self,
        team: str,
        config_hash: str,
        sessions_dir: Path | None = None,
        verbose: bool = False,
    ) -> None:
        if not _SAFE_NAME_RE.match(team):
            msg = (
                f"Invalid team name {team!r}: must contain only "
                "alphanumeric characters, hyphens, and underscores."
            )
            raise ValueError(msg)

        self._team = team
        self._config_hash = config_hash
        self._verbose = verbose
        self._lock = threading.Lock()
        self._seq = 0
        self._closed = False
        self._start_ns = time.monotonic_ns()

        # Session identity
        self._session_id = uuid.uuid4().hex[:12]

        # File setup
        if sessions_dir is None:
            sessions_dir = Path("sessions")
        sessions_dir.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
        base = f"{date_str}_{team}_{self._session_id}"
        self._session_file = sessions_dir / f"{base}.jsonl"

        self._fh: IO[str] | None = None
        self._verbose_fh: IO[str] | None = None
        try:
            self._fh = self._session_file.open("a", encoding="utf-8")
            if verbose:
                verbose_path = sessions_dir / f"{base}.verbose.jsonl"
                self._verbose_fh = verbose_path.open("a", encoding="utf-8")
            # Write the opening event immediately
            self.record(
                SessionStartEvent(
                    ts="",  # placeholder — record() overwrites
                    seq=0,  # placeholder — record() overwrites
                    session_id=self._session_id,
                    team=team,
                    config_hash=config_hash,
                )
            )
        except Exception:
            self._close_handles()
            raise

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        """Unique session identifier (12-char hex)."""
        return self._session_id

    @property
    def session_file(self) -> Path:
        """Path to the primary JSONL file."""
        return self._session_file

    @property
    def event_count(self) -> int:
        """Number of events recorded so far."""
        return self._seq

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, event: SessionEvent) -> None:
        """Write *event* to the JSONL file.

        Automatically stamps ``ts`` and ``seq`` on every event, then
        flushes to disk so no data is lost on a crash.

        Silently drops events after the recorder has been closed.
        """
        with self._lock:
            if self._closed or self._fh is None:
                return
            event.seq = self._seq
            event.ts = _iso_now()
            self._seq += 1
            line = event.model_dump_json(by_alias=True)
            self._fh.write(line + "\n")
            self._fh.flush()

    def record_verbose(self, seq: int, full_result: Any) -> None:
        """Write a verbose sidecar entry keyed by *seq*.

        Only writes when verbose mode was enabled at init time.
        """
        if self._verbose_fh is None:
            return
        with self._lock:
            if self._closed:
                return
            payload = json.dumps({"seq": seq, "full_result": full_result}, default=str)
            self._verbose_fh.write(payload + "\n")
            self._verbose_fh.flush()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def end(
        self,
        reason: EndReason,
        total_tokens: dict[str, int] | None = None,
    ) -> None:
        """Write a ``session_end`` event and close all file handles.

        Idempotent — calling ``end()`` on an already-closed recorder is a
        no-op.
        """
        if self._closed:
            return

        elapsed_ns = time.monotonic_ns() - self._start_ns
        duration_ms = int(elapsed_ns / 1_000_000)
        tokens = total_tokens if total_tokens is not None else {"input": 0, "output": 0}

        self.record(
            SessionEndEvent(
                ts="",
                seq=0,
                reason=reason,
                duration_ms=duration_ms,
                total_tokens=tokens,
            )
        )
        self.close()

    def close(self) -> None:
        """Close file handles **without** writing a ``session_end`` event."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._close_handles()

    def _close_handles(self) -> None:
        """Close underlying file handles (caller must hold lock or be in init)."""
        if self._fh is not None and not self._fh.closed:
            self._fh.close()
        if self._verbose_fh is not None and not self._verbose_fh.closed:
            self._verbose_fh.close()


def _iso_now() -> str:
    """Return the current UTC time as ISO 8601 with milliseconds."""
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
