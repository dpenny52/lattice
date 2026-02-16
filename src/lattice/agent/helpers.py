"""Shared helper functions for agent implementations."""

from __future__ import annotations

import logging

from lattice.session.models import ErrorEvent
from lattice.session.recorder import SessionRecorder


def format_stderr_preview(stderr_text: str, max_lines: int = 5) -> str:
    """Extract and format the last N non-empty lines from stderr output."""
    lines = [line for line in stderr_text.split("\n") if line.strip()]
    last = lines[-max_lines:] if len(lines) > max_lines else lines
    return "\n  ".join(last)


def record_error(
    recorder: SessionRecorder,
    agent_name: str,
    error_msg: str,
    context: str = "subprocess",
    retrying: bool = False,
    logger: logging.Logger | None = None,
) -> None:
    """Log and record an error event in one call."""
    if logger:
        logger.error("%s: %s", agent_name, error_msg)
    recorder.record(
        ErrorEvent(
            ts="",
            seq=0,
            agent=agent_name,
            error=error_msg,
            retrying=retrying,
            context=context,
        )
    )
