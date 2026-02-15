"""Router — central message dispatcher for the Lattice runtime."""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

import click

from lattice.config.models import TopologyConfig
from lattice.router.topology import create_topology
from lattice.session.models import MessageEvent
from lattice.session.recorder import SessionRecorder

logger = logging.getLogger(__name__)

#: Max characters of message content to show in the console preview.
_PREVIEW_LEN = 120


class RouteNotAllowedError(Exception):
    """Raised when a message violates topology rules."""


class UnknownAgentError(Exception):
    """Raised when a message targets an unregistered agent."""


@runtime_checkable
class Agent(Protocol):
    """Minimal protocol that agents must satisfy for the router."""

    async def handle_message(self, from_agent: str, content: str) -> None:
        """Handle an incoming message from another agent."""
        ...


class Router:
    """Central message dispatcher that enforces topology and records events."""

    def __init__(
        self,
        topology: TopologyConfig,
        recorder: SessionRecorder,
    ) -> None:
        self._topology = create_topology(topology)
        self._recorder = recorder
        self._agents: dict[str, Agent] = {}
        self._pending_tasks: set[asyncio.Task[None]] = set()

    @property
    def pending_tasks(self) -> set[asyncio.Task[None]]:
        """Currently in-flight dispatch tasks."""
        return self._pending_tasks

    def register(self, name: str, agent: Agent) -> None:
        """Register an agent by name."""
        self._agents[name] = agent

    async def send(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
    ) -> None:
        """Send a message from one agent to another.

        Records a ``MessageEvent`` to the session JSONL, then dispatches
        to the target agent via ``asyncio.create_task()``.

        The special sender ``"user"`` bypasses topology checks.
        """
        if to_agent not in self._agents:
            msg = f"Unknown agent '{to_agent}'"
            raise UnknownAgentError(msg)

        # "user" and "__system__" bypass topology — god mode
        bypassed = from_agent in ("user", "__system__")
        if not bypassed and not self._topology.is_allowed(from_agent, to_agent):
            msg = f"Route from '{from_agent}' to '{to_agent}' is not allowed"
            raise RouteNotAllowedError(msg)

        # Record before dispatch
        self._recorder.record(
            MessageEvent(
                ts="",
                seq=0,
                from_agent=from_agent,
                to=to_agent,
                content=content,
            )
        )

        # Print agent-to-agent messages to console (user messages are
        # already shown by UserAgent / the REPL input line).
        if from_agent != "user" and to_agent != "user":
            preview = content.replace("\n", " ")
            if len(preview) > _PREVIEW_LEN:
                preview = preview[:_PREVIEW_LEN] + "…"
            click.echo(
                click.style(f"  {from_agent} → {to_agent}: ", fg="cyan")
                + preview
            )

        # Dispatch asynchronously — task is tracked to prevent GC
        agent = self._agents[to_agent]
        task = asyncio.create_task(agent.handle_message(from_agent, content))
        self._pending_tasks.add(task)
        task.add_done_callback(self._task_done)

    async def broadcast(
        self,
        from_agent: str,
        content: str,
        targets: list[str] | None = None,
    ) -> None:
        """Send a message to multiple agents concurrently.

        If *targets* is ``None``, sends to all registered agents except
        the sender. Errors in one dispatch don't block others.
        """
        if targets is None:
            targets = [name for name in self._agents if name != from_agent]

        tasks: list[asyncio.Task[None]] = []
        for target in targets:
            try:
                if target not in self._agents:
                    logger.warning("Broadcast skip: unknown agent '%s'", target)
                    continue

                bypassed = from_agent in ("user", "__system__")
                if not bypassed and not self._topology.is_allowed(
                    from_agent, target
                ):
                    logger.warning(
                        "Broadcast skip: route '%s' -> '%s' not allowed",
                        from_agent,
                        target,
                    )
                    continue

                self._recorder.record(
                    MessageEvent(
                        ts="",
                        seq=0,
                        from_agent=from_agent,
                        to=target,
                        content=content,
                    )
                )

                if from_agent != "user" and target != "user":
                    preview = content.replace("\n", " ")
                    if len(preview) > _PREVIEW_LEN:
                        preview = preview[:_PREVIEW_LEN] + "…"
                    click.echo(
                        click.style(f"  {from_agent} → {target}: ", fg="cyan")
                        + preview
                    )

                agent = self._agents[target]
                task = asyncio.create_task(
                    agent.handle_message(from_agent, content)
                )
                tasks.append(task)
            except Exception:
                logger.exception(
                    "Broadcast error setting up dispatch to '%s'", target
                )

        # Await all tasks, isolating errors
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                logger.error("Broadcast dispatch error: %s", result)

    def _task_done(self, task: asyncio.Task[None]) -> None:
        """Callback for fire-and-forget tasks — log errors, remove from set."""
        self._pending_tasks.discard(task)

        if not task.cancelled():
            exc = task.exception()
            if exc is not None:
                logger.error("Dispatch error: %s", exc)
