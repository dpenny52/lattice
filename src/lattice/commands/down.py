"""lattice down — signal a running session to shut down gracefully."""

from __future__ import annotations

import contextlib
import os
import signal
import time

import click

from lattice.pidfile import is_process_running, read_pidfile, remove_pidfile

#: Maximum valid PID on most systems (Linux default PID_MAX).
_PID_MAX = 4_194_304


@click.command()
def down() -> None:
    """Signal a running lattice session to shut down gracefully."""
    data = read_pidfile()

    if data is None:
        click.echo("No running session found (no pidfile at .lattice/session.pid)")
        raise SystemExit(1)

    pid = data.get("pid")
    session_id = data.get("session_id", "unknown")
    team = data.get("team", "unknown")

    if not isinstance(pid, int) or pid <= 1 or pid > _PID_MAX:
        click.echo("Invalid pidfile: PID missing or out of range")
        remove_pidfile()
        raise SystemExit(1)

    if not is_process_running(pid):
        click.echo(
            f"Session {session_id} (PID {pid}) is no longer running. "
            "Cleaning up stale pidfile."
        )
        remove_pidfile()
        raise SystemExit(0)

    click.echo(f"Shutting down session {session_id} ({team}, PID {pid})...")

    try:
        os.kill(pid, signal.SIGTERM)
    except PermissionError:
        click.echo(f"Permission denied: cannot signal PID {pid}")
        raise SystemExit(1) from None
    except ProcessLookupError:
        click.echo("Process already exited.")
        remove_pidfile()
        raise SystemExit(0) from None

    # Wait for the process to exit
    for _ in range(100):  # 10 seconds, 100ms intervals
        time.sleep(0.1)
        if not is_process_running(pid):
            click.echo("Session ended.")
            # Pidfile should be cleaned up by the exiting process,
            # but clean up just in case
            remove_pidfile()
            return

    click.echo("Session didn't exit within 10s. Sending SIGKILL...")
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.kill(pid, signal.SIGKILL)

    remove_pidfile()
    click.echo("Session killed.")
