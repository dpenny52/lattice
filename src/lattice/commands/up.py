"""lattice up — start the agent team and interactive REPL."""

from __future__ import annotations

import asyncio
import hashlib
import signal
from collections.abc import Callable
from pathlib import Path

import click

from lattice.agent.cli_bridge import CLIBridge
from lattice.agent.llm_agent import LLMAgent, RateLimitGate
from lattice.agent.script_bridge import ScriptBridge
from lattice.config.models import LatticeConfig
from lattice.config.parser import ConfigError, load_config
from lattice.heartbeat import Heartbeat
from lattice.pidfile import remove_pidfile, write_pidfile
from lattice.router.router import Router
from lattice.session.models import LoopBoundaryEvent
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
@click.option("--watch", "enable_watch", is_flag=True, help="Enable live TUI with input bar.")
@click.option(
    "--loop",
    "loop_iterations",
    type=int,
    default=None,
    is_flag=False,
    flag_value=-1,
    help="Re-run prompt in a loop. Optional: specify max iterations (omit for infinite).",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output.")
def up(
    config_file: str | None,
    enable_watch: bool,
    loop_iterations: int | None,
    verbose: bool,
) -> None:
    """Start the agent team and enter the interactive REPL."""
    try:
        config = load_config(Path(config_file) if config_file else None)
    except ConfigError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(1) from exc

    asyncio.run(_run_session(config, verbose, loop_iterations, enable_watch))


# ------------------------------------------------------------------ #
# Session runner
# ------------------------------------------------------------------ #


async def _run_session(
    config: LatticeConfig, verbose: bool, loop_iterations: int | None = None, enable_watch: bool = False
) -> None:
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
    rate_gate = RateLimitGate()

    for name, agent_config in config.agents.items():
        peer_names = [n for n in agent_names if n != name] + ["user"]

        try:
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
                    allowed_paths=config.allowed_paths or None,
                    on_response=_make_response_callback(name),
                    rate_gate=rate_gate,
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
        except ValueError as exc:
            # API key or provider errors
            click.echo(f"Error initializing agent '{name}': {exc}", err=True)
            recorder.end("error")
            raise SystemExit(1) from exc
        except Exception as exc:
            # Unexpected errors
            click.echo(f"Error initializing agent '{name}': {exc}", err=True)
            recorder.end("error")
            raise SystemExit(1) from exc

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
    # In watch mode, let Textual handle Ctrl+C via its quit action
    if not enable_watch:
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

    # Track why we're shutting down and loop count
    shutdown_reason = "user_shutdown"
    loop_count = 0

    try:
        # Determine if we're running in loop mode
        if loop_iterations is not None:
            shutdown_reason, loop_count = await _loop_mode(
                router,
                entry,
                all_agents,
                agents,
                shutdown_event,
                heartbeat,
                recorder,
                loop_iterations,
            )
        elif enable_watch:
            # Combined mode: run TUI with input bar
            shutdown_reason = await _watch_mode(
                router, entry, all_agents, shutdown_event, heartbeat, recorder,
            )
        else:
            shutdown_reason = await _repl_loop(
                router, entry, all_agents, shutdown_event, heartbeat,
            )
    finally:
        # Create shutdown manager with loop count
        shutdown_mgr = ShutdownManager(
            router=router,
            recorder=recorder,
            heartbeat=heartbeat,
            cli_bridges=cli_bridges,
            all_agents=all_agents,
            shutdown_event=shutdown_event,
            loop_count=loop_count,
        )
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


async def _watch_mode(
    router: Router,
    entry_agent: str,
    agents: dict[str, LLMAgent | CLIBridge | ScriptBridge],
    shutdown_event: asyncio.Event,
    heartbeat: Heartbeat | None,
    recorder: SessionRecorder,
) -> str:
    """Run the TUI with input bar instead of the REPL.

    Returns the shutdown reason string.
    """
    from lattice.commands.watch import WatchApp

    if heartbeat is not None:
        await heartbeat.start()

    # Create and run the TUI app
    app = WatchApp(
        session_file=recorder.session_file,
        enable_input=True,
        router=router,
        entry_agent=entry_agent,
        all_agents=agents,
        heartbeat=heartbeat,
        shutdown_event=shutdown_event,
    )

    # Run the app in a separate task so we can monitor heartbeat
    app_task = asyncio.create_task(_run_tui_app(app))

    reason = "user_shutdown"

    # Monitor for completion or external shutdown
    while not shutdown_event.is_set():
        # Check if heartbeat signalled completion
        if heartbeat is not None and heartbeat.done_flag:
            reason = "complete"
            shutdown_event.set()
            break

        # Check if TUI app exited
        if app_task.done():
            break

        await asyncio.sleep(0.1)

    # If TUI is still running, exit it gracefully
    if not app_task.done():
        app.exit()

    # Wait for app to finish
    try:
        await app_task
    except Exception:
        pass

    # If we got here because shutdown_event was set externally (Ctrl+C / SIGTERM)
    # rather than via /done or heartbeat completion, mark as ctrl_c.
    if shutdown_event.is_set() and reason == "user_shutdown":
        reason = "ctrl_c"

    return reason


async def _run_tui_app(app: "WatchApp") -> None:  # type: ignore[name-defined]
    """Run the Textual app in async context."""
    await app.run_async()


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


# ------------------------------------------------------------------ #
# Loop mode
# ------------------------------------------------------------------ #


async def _loop_mode(
    router: Router,
    entry_agent: str,
    all_agents: dict[str, LLMAgent | CLIBridge | ScriptBridge],
    agents: dict[str, LLMAgent],
    shutdown_event: asyncio.Event,
    heartbeat: Heartbeat | None,
    recorder: SessionRecorder,
    loop_iterations: int,
) -> tuple[str, int]:
    """Run the prompt in a loop with fresh context each iteration.

    Args:
        router: Message router
        entry_agent: Name of the entry agent
        all_agents: All agents (LLM + CLI + Script)
        agents: LLM agents only (for context reset)
        shutdown_event: Event to signal shutdown
        heartbeat: Optional heartbeat monitor
        recorder: Session recorder
        loop_iterations: Max iterations (-1 for infinite loop)

    Returns:
        Tuple of (shutdown reason string, loop count)
    """
    # Get initial prompt from user
    click.echo("Enter prompt for loop (this prompt will be re-run each iteration):")
    try:
        initial_prompt = await asyncio.get_event_loop().run_in_executor(
            None, _read_input,
        )
    except EOFError:
        return ("ctrl_c", 0)

    initial_prompt = initial_prompt.strip()
    if not initial_prompt:
        click.echo("No prompt provided. Exiting loop mode.")
        return ("user_shutdown", 0)

    if heartbeat is not None:
        await heartbeat.start()

    reason = "complete"
    iteration = 1

    while not shutdown_event.is_set():
        # Check max iterations
        if loop_iterations > 0 and iteration > loop_iterations:
            reason = "complete"
            break

        # Log loop boundary start
        recorder.record(
            LoopBoundaryEvent(
                ts="", seq=0, boundary="start", iteration=iteration
            )
        )

        # Print visual separator
        click.echo(f"\n── Loop {iteration} ──\n")

        # Reset context for all LLM agents
        for agent in agents.values():
            agent.reset_context()

        # Send the initial prompt to the entry agent
        await router.send("user", entry_agent, initial_prompt)

        # Wait for the agent to complete this iteration
        # Two exit conditions:
        # 1. Heartbeat detects DONE marker -> exit entire loop
        # 2. All router tasks complete -> continue to next iteration
        while not shutdown_event.is_set():
            # Give the agent time to start processing
            await asyncio.sleep(0.05)

            # Check if heartbeat detected "done" marker for entire loop
            if heartbeat is not None and heartbeat.done_flag:
                # Agent declared completion for the entire loop
                reason = "complete"
                # Wait for any remaining tasks to complete
                if router.pending_tasks:
                    await asyncio.gather(
                        *list(router.pending_tasks), return_exceptions=True
                    )
                # Log loop boundary end
                recorder.record(
                    LoopBoundaryEvent(
                        ts="", seq=0, boundary="end", iteration=iteration
                    )
                )
                return (reason, iteration)

            # Check if all tasks are done for this iteration
            pending = list(router.pending_tasks)
            if not pending:
                # All tasks complete for this iteration
                break

            # Wait for tasks to complete, checking periodically
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=0.5,
                )
            except TimeoutError:
                # Timeout just means we'll check again
                pass

        # Log loop boundary end
        recorder.record(
            LoopBoundaryEvent(
                ts="", seq=0, boundary="end", iteration=iteration
            )
        )

        # If we're shutting down externally (Ctrl+C), break
        if shutdown_event.is_set():
            reason = "ctrl_c"
            break

        iteration += 1

    return (reason, iteration - 1)
