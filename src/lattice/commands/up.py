"""lattice up — start the agent team and interactive REPL."""

from __future__ import annotations

import asyncio
import hashlib
import signal
from collections.abc import Callable
from pathlib import Path

import click

from lattice.agent.cli_bridge import CLIBridge
from lattice.agent.llm_agent import LLMAgent
from lattice.agent.script_bridge import ScriptBridge
from lattice.config.models import LatticeConfig
from lattice.config.parser import ConfigError, load_config
from lattice.heartbeat import Heartbeat
from lattice.pidfile import remove_pidfile, write_pidfile
from lattice.router.router import Router
from lattice.session.recorder import SessionRecorder
from lattice.shutdown import ShutdownManager

# ------------------------------------------------------------------ #
# UserAgent — pseudo-agent that prints messages to the terminal
# ------------------------------------------------------------------ #


class UserAgent:
    """Pseudo-agent registered as ``"user"`` in the router.

    When an LLM agent calls ``send_message(to="user", ...)``, the
    router dispatches to this agent, which simply prints the content.
    """

    async def handle_message(self, from_agent: str, content: str) -> None:
        """Print the incoming message in ``[agent] text`` format."""
        click.echo(f"[{from_agent}] {content}")


# ------------------------------------------------------------------ #
# Click command
# ------------------------------------------------------------------ #


@click.command()
@click.option(
    "-f", "--file", "config_file", type=click.Path(), help="Config file path."
)
@click.option("--watch", "enable_watch", is_flag=True, help="Re-run on file changes.")
@click.option("--loop", is_flag=True, help="Keep running in a loop.")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output.")
def up(
    config_file: str | None,
    enable_watch: bool,
    loop: bool,
    verbose: bool,
) -> None:
    """Start the agent team and enter the interactive REPL."""
    if enable_watch:
        click.echo("Warning: --watch is not yet implemented.", err=True)
    if loop:
        click.echo("Warning: --loop is not yet implemented.", err=True)

    try:
        config = load_config(Path(config_file) if config_file else None)
    except ConfigError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc

    asyncio.run(_run_session(config, verbose))


# ------------------------------------------------------------------ #
# Session runner
# ------------------------------------------------------------------ #


async def _run_session(config: LatticeConfig, verbose: bool) -> None:
    """Wire up all components and run the REPL until shutdown."""
    # 1. Create recorder
    config_hash = hashlib.sha256(
        config.model_dump_json().encode()
    ).hexdigest()[:16]
    recorder = SessionRecorder(
        team=config.team, config_hash=config_hash, verbose=verbose,
    )

    # 2. Create router
    router = Router(topology=config.topology, recorder=recorder)

    # 3. Register the "user" pseudo-agent so LLM agents can send_message to "user"
    user_agent = UserAgent()
    router.register("user", user_agent)

    # 4. Create and register agents
    agent_names = list(config.agents.keys())
    agents: dict[str, LLMAgent] = {}
    cli_bridges: dict[str, CLIBridge] = {}
    script_bridges: dict[str, ScriptBridge] = {}

    for name, agent_config in config.agents.items():
        peer_names = [n for n in agent_names if n != name] + ["user"]

        if agent_config.type == "llm":
            agent = LLMAgent(
                name=name,
                model_string=agent_config.model or "",
                role=agent_config.role or "",
                router=router,
                recorder=recorder,
                team_name=config.team,
                peer_names=peer_names,
                credentials=config.credentials,
                configured_tools=agent_config.tools,
                on_response=_make_response_callback(name),
            )
            router.register(name, agent)
            agents[name] = agent

        elif agent_config.type == "cli":
            bridge = CLIBridge(
                name=name,
                role=agent_config.role or "",
                router=router,
                recorder=recorder,
                team_name=config.team,
                peer_names=peer_names,
                cli_type=agent_config.cli,
                command=agent_config.command,
                on_response=_make_response_callback(name),
            )
            router.register(name, bridge)
            cli_bridges[name] = bridge

        elif agent_config.type == "script":
            script = ScriptBridge(
                name=name,
                role=agent_config.role or "",
                command=agent_config.command or "",
                router=router,
                recorder=recorder,
                on_response=_make_response_callback(name),
            )
            router.register(name, script)
            script_bridges[name] = script

        else:
            click.echo(
                f"Warning: Agent '{name}' has type '{agent_config.type}' "
                "-- not yet supported. Skipping."
            )

    all_agents: dict[str, LLMAgent | CLIBridge | ScriptBridge] = {
        **agents, **cli_bridges, **script_bridges,
    }

    if not all_agents:
        click.echo("Error: No agents configured. Nothing to run.", err=True)
        recorder.end("error")
        return

    # Start CLI bridges.
    for bridge in cli_bridges.values():
        await bridge.start()

    # 5. Print startup banner
    entry = config.entry or next(iter(all_agents))
    click.echo(f"\n  Lattice -- {config.team}")
    click.echo(f"  Agents: {len(all_agents)} | Session: {recorder.session_id}")
    click.echo(f"  Entry:  {entry} | Topology: {config.topology.type}")
    click.echo(f"  Log:    {recorder.session_file}")
    click.echo()

    # Write pidfile for `lattice down` communication
    write_pidfile(recorder.session_id, config.team)

    # 6. Enter REPL
    shutdown_event = asyncio.Event()

    # Handle Ctrl+C / SIGTERM gracefully
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, shutdown_event.set)

    # 7. Create heartbeat
    heartbeat = Heartbeat(
        interval=config.communication.heartbeat,
        router=router,
        entry_agent=entry,
        recorder=recorder,
        shutdown_event=shutdown_event,
    )

    # Wire heartbeat response checking into the entry agent's callback
    if entry in agents:
        _install_heartbeat_hook(agents[entry], heartbeat)

    # 8. Create shutdown manager
    shutdown_mgr = ShutdownManager(
        router=router,
        recorder=recorder,
        heartbeat=heartbeat,
        cli_bridges=cli_bridges,
        all_agents=all_agents,
        shutdown_event=shutdown_event,
    )

    # Track why we're shutting down — updated by REPL exit path
    shutdown_reason = "user_shutdown"

    try:
        shutdown_reason = await _repl_loop(
            router, entry, all_agents, shutdown_event, heartbeat,
        )
    finally:
        await shutdown_mgr.execute(shutdown_reason)
        remove_pidfile()


# ------------------------------------------------------------------ #
# REPL loop
# ------------------------------------------------------------------ #


async def _repl_loop(
    router: Router,
    entry_agent: str,
    agents: dict[str, LLMAgent | CLIBridge | ScriptBridge],
    shutdown_event: asyncio.Event,
    heartbeat: Heartbeat | None = None,
) -> str:
    """Read user input in a loop, dispatch to agents, handle commands.

    Returns the shutdown reason string.
    """
    if heartbeat is not None:
        await heartbeat.start()

    reason = "user_shutdown"

    while not shutdown_event.is_set():
        # Check if heartbeat signalled completion
        if heartbeat is not None and heartbeat.done_flag:
            reason = "complete"
            shutdown_event.set()
            break

        try:
            if heartbeat is not None:
                heartbeat.pause()
            line = await asyncio.get_event_loop().run_in_executor(
                None, _read_input,
            )
        except EOFError:
            break
        finally:
            if heartbeat is not None:
                heartbeat.resume()

        line = line.strip()
        if not line:
            continue

        # -- Slash commands ------------------------------------------------
        if line.startswith("/"):
            should_break = await _handle_command(
                line, agents, heartbeat,
            )
            if should_break:
                break
            continue

        # -- @agent routing ------------------------------------------------
        if line.startswith("@"):
            parts = line.split(None, 1)
            target = parts[0][1:]  # strip the @
            content = parts[1] if len(parts) > 1 else ""

            if target not in agents:
                click.echo(f"Unknown agent: {target}")
                continue
            if not content:
                click.echo(f"No message provided for @{target}")
                continue

            await router.send("user", target, content)
            continue

        # -- Plain text -> entry agent --------------------------------------
        await router.send("user", entry_agent, line)

    # If we got here because shutdown_event was set externally (Ctrl+C / SIGTERM)
    # rather than via /done or heartbeat completion, mark as ctrl_c.
    if shutdown_event.is_set() and reason == "user_shutdown":
        reason = "ctrl_c"

    return reason


def _read_input() -> str:
    """Blocking input() wrapper for use with ``run_in_executor``."""
    return input("> ")


async def _handle_command(
    line: str,
    agents: dict[str, LLMAgent | CLIBridge | ScriptBridge],
    heartbeat: Heartbeat | None = None,
) -> bool:
    """Process a slash command. Returns ``True`` if the REPL should exit."""
    cmd = line.split()[0].lower()

    if cmd == "/done":
        return True

    if cmd == "/status":
        if heartbeat is not None:
            await heartbeat.fire()
        else:
            click.echo("Status: all agents idle")
        return False

    if cmd == "/agents":
        for name, agent in agents.items():
            if isinstance(agent, CLIBridge):
                agent_type = "cli"
            elif isinstance(agent, ScriptBridge):
                agent_type = "script"
            else:
                agent_type = "llm"
            click.echo(f"  {name} ({agent_type})")
        return False

    click.echo(f"Unknown command: {cmd}")
    return False


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_response_callback(agent_name: str) -> Callable[[str], None]:
    """Return a closure that prints ``[agent_name] text`` to the terminal.

    Used as the ``on_response`` callback for ``LLMAgent`` -- fires when
    the agent produces a plain-text response (end of its tool-call loop).
    """

    def _callback(content: str) -> None:
        click.echo(f"[{agent_name}] {content}")

    return _callback


def _install_heartbeat_hook(agent: LLMAgent, heartbeat: Heartbeat) -> None:
    """Wrap the entry agent's ``on_response`` to check heartbeat markers."""
    if not hasattr(agent, "_on_response"):
        return

    original_callback = agent._on_response

    def _hooked(content: str) -> None:
        if original_callback is not None:
            original_callback(content)
        heartbeat.check_response(content)

    agent._on_response = _hooked
