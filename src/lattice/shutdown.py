"""ShutdownManager — orchestrates the 4-step graceful shutdown sequence."""

from __future__ import annotations

import asyncio
import logging
import time

import click

from lattice.agent.cli_bridge import CLIBridge
from lattice.agent.llm_agent import LLMAgent
from lattice.agent.script_bridge import ScriptBridge
from lattice.heartbeat import Heartbeat
from lattice.router.router import Router
from lattice.session.recorder import SessionRecorder

logger = logging.getLogger(__name__)


def _format_duration(seconds: float) -> str:
    """Format a duration as '1m 22s' or '34.2s'."""
    if seconds >= 60:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs:02d}s"
    return f"{seconds:.1f}s"


class ShutdownManager:
    """Orchestrates the 4-step graceful shutdown sequence.

    Steps:
        1. SIGNAL  -- set shutdown flag, stop heartbeat
        2. DRAIN   -- wait for in-flight tasks with timeout
        3. KILL    -- force-cancel anything still running
        4. CLOSE   -- write session_end, print summary
    """

    DRAIN_TIMEOUT = 10.0  # seconds

    def __init__(
        self,
        router: Router,
        recorder: SessionRecorder,
        heartbeat: Heartbeat | None,
        cli_bridges: dict[str, CLIBridge],
        all_agents: dict[str, LLMAgent | CLIBridge | ScriptBridge],
        shutdown_event: asyncio.Event,
        loop_count: int = 0,
    ) -> None:
        self._router = router
        self._recorder = recorder
        self._heartbeat = heartbeat
        self._cli_bridges = cli_bridges
        self._all_agents = all_agents
        self._shutdown_event = shutdown_event
        self._start_time = time.monotonic()
        self._loop_count = loop_count

    async def execute(self, reason: str) -> None:
        """Run the full shutdown sequence."""
        await self._signal()
        drained = await self._drain()
        if not drained:
            await self._kill()
        await self._close(reason)

    # ------------------------------------------------------------------ #
    # Step 1: SIGNAL
    # ------------------------------------------------------------------ #

    async def _signal(self) -> None:
        """Stop accepting input, cancel response channels, and stop heartbeat."""
        self._shutdown_event.set()

        # Unblock any agents waiting for responses before stopping heartbeat.
        self._router.cancel_all_responses()

        if self._heartbeat is not None:
            await self._heartbeat.stop()

    # ------------------------------------------------------------------ #
    # Step 2: DRAIN
    # ------------------------------------------------------------------ #

    async def _drain(self) -> bool:
        """Wait for in-flight router tasks with timeout.

        Returns True if all tasks completed, False if timeout hit.
        """
        pending = list(self._router.pending_tasks)
        if not pending:
            return True

        click.echo("\nDraining in-flight tasks...")
        try:
            await asyncio.wait_for(
                asyncio.gather(*pending, return_exceptions=True),
                timeout=self.DRAIN_TIMEOUT,
            )
            return True
        except TimeoutError:
            remaining = len([t for t in pending if not t.done()])
            logger.warning("Drain timeout: %d tasks still running", remaining)
            return False

    # ------------------------------------------------------------------ #
    # Step 3: KILL
    # ------------------------------------------------------------------ #

    async def _kill(self) -> None:
        """Force-cancel remaining async tasks and shutdown CLI bridges."""
        # Cancel remaining router tasks.
        remaining = list(self._router.pending_tasks)
        interrupted_count = 0
        for task in remaining:
            if not task.done():
                task.cancel()
                interrupted_count += 1

        if interrupted_count:
            click.echo(f"Cancelled {interrupted_count} remaining task(s).")

        # Wait briefly for cancellations to propagate.
        if remaining:
            await asyncio.gather(*remaining, return_exceptions=True)

        # Shutdown CLI bridges (SIGTERM -> SIGKILL).
        for name, bridge in self._cli_bridges.items():
            try:
                await bridge.shutdown()
            except Exception:
                logger.exception("Error shutting down CLI bridge '%s'", name)

    # ------------------------------------------------------------------ #
    # Step 4: CLOSE
    # ------------------------------------------------------------------ #

    async def _close(self, reason: str) -> None:
        """Shutdown agents, write session_end, print summary."""
        # Shutdown all agents that weren't already handled in _kill.
        for name, agent in self._all_agents.items():
            # CLI bridges may have been shut down in _kill; calling again is safe
            # because CLIBridge.shutdown() is idempotent.
            try:
                await agent.shutdown()
            except Exception:
                logger.exception("Error shutting down agent '%s'", name)

        # Write session_end event.
        self._recorder.end(reason)

        # Print summary.
        elapsed = time.monotonic() - self._start_time
        duration_str = _format_duration(elapsed)
        msg_count = self._recorder.event_count
        agent_count = len(self._all_agents)

        summary_parts = [
            f"\nSession ended ({reason})",
            duration_str,
            f"{msg_count} events",
            f"{agent_count} agent(s)",
        ]
        if self._loop_count > 0:
            summary_parts.append(f"{self._loop_count} loop(s)")

        click.echo(" | ".join(summary_parts))
        click.echo(f"Log: {self._recorder.session_file}")
