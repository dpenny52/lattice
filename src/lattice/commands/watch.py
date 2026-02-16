"""lattice watch — live TUI for agent team activity."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widgets import Header, Input, Label, Static

from lattice.constants import SYSTEM_SENDER

# ------------------------------------------------------------------ #
# Data models
# ------------------------------------------------------------------ #


@dataclass
class AgentState:
    """Tracks the current state of an agent."""

    name: str
    agent_type: str = "unknown"
    active: bool = False
    current_activity: str = "standby"


@dataclass
class MessageLink:
    """Represents a message sent from one agent to another."""

    from_agent: str
    to_agent: str
    content: str
    completed: bool = False


@dataclass
class SessionStats:
    """Aggregated session statistics."""

    start_time: datetime | None = None
    message_count: int = 0
    total_tokens: dict[str, int] = field(
        default_factory=lambda: {"input": 0, "output": 0}
    )
    loop_iteration: int = 0
    session_ended: bool = False


# ------------------------------------------------------------------ #
# Textual widgets
# ------------------------------------------------------------------ #


class AgentsPanel(Static):
    """Left panel showing agent list with status indicators."""

    agents: reactive[dict[str, AgentState]] = reactive(dict, recompose=True)

    def compose(self) -> ComposeResult:
        """Render agent list."""
        yield Label("[bold]Agents[/bold]", classes="panel-title")
        if not self.agents:
            yield Label("No agents yet...", classes="empty-state")
        else:
            for agent in self.agents.values():
                indicator = "●" if agent.active else "○"
                status_class = "active" if agent.active else "standby"
                yield Label(
                    f"{indicator} [bold]{agent.name}[/bold] ({agent.agent_type})\n"
                    f"  {agent.current_activity}",
                    classes=f"agent-item {status_class}",
                )


class MessageFlowPanel(Static):
    """Right panel showing message communication graph."""

    messages: reactive[list[MessageLink]] = reactive(list, recompose=True)

    def compose(self) -> ComposeResult:
        """Render message flow graph."""
        yield Label("[bold]Message Flow[/bold]", classes="panel-title")
        if not self.messages:
            yield Label("No messages yet...", classes="empty-state")
        else:
            # Group messages by from -> to pairs
            pairs: dict[tuple[str, str], list[MessageLink]] = defaultdict(list)
            for msg in self.messages:
                pairs[(msg.from_agent, msg.to_agent)].append(msg)

            for (from_agent, to_agent), msgs in pairs.items():
                completed = all(m.completed for m in msgs)
                check = "✓" if completed else "◦"
                yield Label(
                    f"{check} {from_agent} → {to_agent} ({len(msgs)})",
                    classes=f"message-flow {'completed' if completed else 'pending'}",
                )


class LatestPanel(VerticalScroll):
    """Bottom panel showing scrollable event tail."""

    events: reactive[deque[str]] = reactive(deque, recompose=True)
    max_events: int = 100

    def compose(self) -> ComposeResult:
        """Render latest events."""
        yield Label("[bold]Latest Events[/bold]", classes="panel-title")
        if not self.events:
            yield Label("Waiting for events...", classes="empty-state")
        else:
            for event_line in self.events:
                yield Label(event_line, classes="event-line", markup=True)

    def watch_events(self) -> None:
        """Auto-scroll to bottom when events change."""
        # Double call_after_refresh: first waits for the recompose triggered
        # by the reactive change, second waits for the new children to be laid out.
        self.call_after_refresh(
            lambda: self.call_after_refresh(self.scroll_end, animate=False)
        )

    def add_event(self, event_str: str) -> None:
        """Add an event to the tail, maintaining max size."""
        events_copy = deque(self.events)
        events_copy.append(event_str)
        if len(events_copy) > self.max_events:
            events_copy.popleft()
        self.events = events_copy


class StatsFooter(Static):
    """Footer showing live statistics."""

    stats: reactive[SessionStats] = reactive(SessionStats)

    def render(self) -> str:
        """Render stats line."""
        if not self.stats.start_time:
            return "Session not started"

        # Handle timezone-aware datetime
        now = datetime.now(self.stats.start_time.tzinfo)
        duration = now - self.stats.start_time
        duration_str = str(duration).split(".")[0]  # Remove microseconds

        tokens = self.stats.total_tokens
        token_str = f"{tokens['input']:,}i / {tokens['output']:,}o"

        parts = [
            f"Duration: {duration_str}",
            f"Messages: {self.stats.message_count}",
            f"Tokens: {token_str}",
        ]

        if self.stats.loop_iteration > 0:
            parts.append(f"Loop: {self.stats.loop_iteration}")

        if self.stats.session_ended:
            parts.append("[red]SESSION ENDED[/red]")

        return " | ".join(parts)


# ------------------------------------------------------------------ #
# Main Textual app
# ------------------------------------------------------------------ #


class WatchApp(App[None]):
    """Live TUI for watching agent team activity."""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 2;
        grid-rows: auto 1fr;
    }

    #agents-panel {
        column-span: 1;
        row-span: 1;
        border: solid $primary;
        padding: 1;
        max-height: 12;
    }

    #message-flow-panel {
        column-span: 1;
        row-span: 1;
        border: solid $primary;
        padding: 1;
        max-height: 12;
    }

    #latest-panel {
        column-span: 2;
        row-span: 1;
        border: solid $primary;
        padding: 1;
        height: 100%;
    }

    #input-bar {
        dock: bottom;
        height: 3;
        background: $panel;
        border-top: solid $primary;
        padding: 0 1;
    }

    #stats-footer {
        column-span: 2;
        dock: bottom;
        height: 3;
        background: $panel;
        border-top: solid $primary;
        padding: 1;
    }

    .panel-title {
        text-style: bold;
        margin-bottom: 1;
    }

    .empty-state {
        color: $text-muted;
        text-style: italic;
    }

    .agent-item {
        margin-bottom: 1;
    }

    .agent-item.active {
        color: $success;
    }

    .agent-item.standby {
        color: $text-muted;
    }

    .message-flow.completed {
        color: $success;
    }

    .message-flow.pending {
        color: $warning;
    }

    .event-line {
        margin: 0 0 1 0;
        width: 1fr;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("t", "toggle_tools", "Toggle tools"),
    ]

    def __init__(
        self,
        session_file: Path,
        enable_input: bool = False,
        router: Any = None,
        entry_agent: str | None = None,
        all_agents: dict[str, Any] | None = None,
        heartbeat: Any = None,
        shutdown_event: asyncio.Event | None = None,
        show_tools: bool = False,
    ) -> None:
        """Initialize the watch app.

        Args:
            session_file: Path to the session JSONL file to watch
            enable_input: Whether to show the input bar (for combined mode)
            router: Router instance (for combined mode)
            entry_agent: Entry agent name (for combined mode)
            all_agents: Dict of all agents (for combined mode commands)
            heartbeat: Heartbeat monitor (for /status command)
            shutdown_event: Shutdown event to signal graceful shutdown
            show_tools: Whether to show tool call events in the event feed
        """
        super().__init__()
        self.session_file = session_file
        self.enable_input = enable_input
        self.router = router
        self.entry_agent = entry_agent
        self.all_agents = all_agents or {}
        self.heartbeat = heartbeat
        self.shutdown_event = shutdown_event
        self.show_tools = show_tools

        # State
        self.agents: dict[str, AgentState] = {}
        self.messages: list[MessageLink] = []
        self.stats = SessionStats()
        self.last_position = 0
        self.event_lines: deque[str] = deque(maxlen=100)

        # Debouncing
        self.pending_update = False
        self.update_scheduled = False

        # Shutdown flag
        self._stop_watching = False

    def compose(self) -> ComposeResult:
        """Compose the TUI layout."""
        yield Header(show_clock=True)

        yield AgentsPanel(id="agents-panel")
        yield MessageFlowPanel(id="message-flow-panel")
        yield LatestPanel(id="latest-panel")
        yield StatsFooter(id="stats-footer")

        # Add input bar if enabled (combined mode)
        if self.enable_input:
            yield Input(placeholder="Type a message...", id="input-bar")

    def on_mount(self) -> None:
        """Start watching the session file."""
        self.watch_session_file()

        # Focus input bar in combined mode
        if self.enable_input:
            self.set_focus(self.query_one("#input-bar", Input))

    def on_unmount(self) -> None:
        """Handle app shutdown."""
        self._stop_watching = True

    def action_toggle_tools(self) -> None:
        """Toggle tool call visibility in the event feed."""
        self.show_tools = not self.show_tools
        state = "shown" if self.show_tools else "hidden"
        self.notify(f"Tool calls {state} (press t to toggle)", timeout=3)

    async def action_quit(self) -> None:
        """Override quit action to trigger graceful shutdown in combined mode."""
        if self.enable_input and self.shutdown_event is not None:
            # Signal graceful shutdown instead of immediate exit
            self.shutdown_event.set()
        self.exit()

    @work(exclusive=True)
    async def watch_session_file(self) -> None:
        """Watch the session JSONL file and update state."""
        while not self._stop_watching:
            try:
                await self._read_new_events()
            except Exception as e:
                self.notify(f"Error reading session file: {e}", severity="error")

            await asyncio.sleep(0.2)  # Poll every 200ms

    async def _read_new_events(self) -> None:
        """Read new events from the session file since last position."""
        if not self.session_file.exists():
            return

        with self.session_file.open("r", encoding="utf-8") as f:
            # Seek to last read position
            f.seek(self.last_position)

            # Read new lines
            new_events = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event_dict = json.loads(line)
                    new_events.append(event_dict)
                except json.JSONDecodeError:
                    continue

            # Update position
            self.last_position = f.tell()

        # Process events
        if new_events:
            for event_dict in new_events:
                self._process_event(event_dict)

            # Schedule debounced UI update
            if not self.update_scheduled:
                self.update_scheduled = True
                self.set_timer(0.15, self._update_ui)

    def _process_event(self, event_dict: dict[str, Any]) -> None:
        """Process a single event and update state."""
        event_type = event_dict.get("type")

        if event_type == "session_start":
            self.stats.start_time = datetime.fromisoformat(
                event_dict["ts"].replace("Z", "+00:00")
            )
            self._add_event_line(
                f"[cyan]Session started[/cyan]: {event_dict['session_id']}"
            )

        elif event_type == "session_end":
            self.stats.session_ended = True
            self._add_event_line(
                f"[red]Session ended[/red]: {event_dict['reason']}"
            )

        elif event_type == "agent_start":
            agent_name = event_dict["agent"]
            if agent_name not in self.agents:
                self.agents[agent_name] = AgentState(
                    name=agent_name,
                    agent_type=event_dict["agent_type"],
                )
            self.agents[agent_name].active = True
            self.agents[agent_name].current_activity = "starting"
            self._add_event_line(f"[green]Agent started[/green]: {agent_name}")

        elif event_type == "agent_done":
            agent_name = event_dict["agent"]
            if agent_name in self.agents:
                self.agents[agent_name].active = False
                self.agents[agent_name].current_activity = "done"
            self._add_event_line(
                f"[yellow]Agent done[/yellow]: {agent_name} ({event_dict['reason']})"
            )

        elif event_type == "message":
            from_agent = event_dict["from"]
            to_agent = event_dict["to"]
            content = event_dict["content"]

            msg = MessageLink(
                from_agent=from_agent,
                to_agent=to_agent,
                content=content,
                completed=False,
            )
            self.messages.append(msg)
            self.stats.message_count += 1

            # Heartbeat responses routed as __system__ → user are already
            # shown as agent_response events — skip to avoid duplicates.
            if from_agent == SYSTEM_SENDER and to_agent == "user":
                return

            display_content = content.replace("\n", " ")
            self._add_event_line(
                f"[blue]Message[/blue]: {from_agent} → {to_agent}: {display_content}"
            )

        elif event_type == "llm_call_start":
            agent_name = event_dict["agent"]
            if agent_name in self.agents:
                self.agents[agent_name].current_activity = (
                    f"calling {event_dict['model']}"
                )

        elif event_type == "llm_call_end":
            agent_name = event_dict["agent"]
            tokens = event_dict["tokens"]
            self.stats.total_tokens["input"] += tokens.get("input", 0)
            self.stats.total_tokens["output"] += tokens.get("output", 0)

            if agent_name in self.agents:
                self.agents[agent_name].current_activity = "processing"

            duration_ms = event_dict["duration_ms"]
            self._add_event_line(
                f"[magenta]LLM call[/magenta]: {agent_name}"
                f" ({duration_ms}ms,"
                f" {tokens['input']}i/{tokens['output']}o)"
            )

        elif event_type == "tool_call":
            agent_name = event_dict["agent"]
            tool_name = event_dict["tool"]
            args = event_dict["args"]

            if agent_name in self.agents:
                self.agents[agent_name].current_activity = f"calling {tool_name}"

            if self.show_tools:
                args_str = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:2])
                if len(args) > 2:
                    args_str += ", ..."
                self._add_event_line(
                    f"[cyan]Tool call[/cyan]: {agent_name}.{tool_name}({args_str})"
                )

        elif event_type == "tool_result":
            agent_name = event_dict["agent"]
            tool_name = event_dict["tool"]
            duration_ms = event_dict["duration_ms"]

            if agent_name in self.agents:
                self.agents[agent_name].current_activity = "processing"

            if self.show_tools:
                self._add_event_line(
                    f"[cyan]Tool result[/cyan]: "
                    f"{agent_name}.{tool_name} ({duration_ms}ms)"
                )

        elif event_type == "status":
            agent_name = event_dict["agent"]
            status = event_dict["status"]

            if agent_name in self.agents:
                self.agents[agent_name].current_activity = status

            self._add_event_line(f"[yellow]Status[/yellow]: {agent_name}: {status}")

        elif event_type == "error":
            agent_name = event_dict["agent"]
            error = event_dict["error"]
            retrying = event_dict["retrying"]

            retry_str = " (retrying)" if retrying else ""
            self._add_event_line(
                f"[red]Error[/red]: {agent_name}: {error}{retry_str}"
            )

        elif event_type == "agent_response":
            agent_name = event_dict["agent"]
            content = event_dict["content"]
            if agent_name in self.agents:
                self.agents[agent_name].current_activity = "responded"

            display_content = content.replace("\n", " ")
            self._add_event_line(
                f"[green]{agent_name}[/green]: {display_content}"
            )

        elif event_type == "cli_text_chunk":
            agent_name = event_dict["agent"]
            text = event_dict["text"]
            if agent_name in self.agents:
                self.agents[agent_name].current_activity = "responding"

            display_text = text.replace("\n", " ")
            self._add_event_line(
                f"[green]{agent_name}[/green]: {display_text}"
            )

        elif event_type == "cli_tool_call":
            agent_name = event_dict["agent"]
            tool_name = event_dict["tool"]
            args = event_dict.get("args", {})

            if agent_name in self.agents:
                self.agents[agent_name].current_activity = f"calling {tool_name}"

            if self.show_tools:
                args_str = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:2])
                if len(args) > 2:
                    args_str += ", ..."
                self._add_event_line(
                    f"[cyan]{agent_name}[/cyan]: {tool_name}({args_str})"
                )

        elif event_type == "cli_thinking":
            agent_name = event_dict["agent"]
            if agent_name in self.agents:
                self.agents[agent_name].current_activity = "thinking"

            self._add_event_line(f"[dim]CLI thinking[/dim]: {agent_name}")

        elif event_type == "cli_progress":
            agent_name = event_dict["agent"]
            status = event_dict["status"]

            if agent_name in self.agents:
                self.agents[agent_name].current_activity = status

            self._add_event_line(
                f"[yellow]CLI progress[/yellow]: {agent_name}: {status}"
            )

        elif event_type == "loop_boundary":
            boundary = event_dict["boundary"]
            iteration = event_dict["iteration"]

            if boundary == "start":
                self.stats.loop_iteration = iteration
                self._add_event_line(
                    f"[bold cyan]Loop {iteration} started[/bold cyan]"
                )
            else:
                self._add_event_line(
                    f"[bold cyan]Loop {iteration} ended[/bold cyan]"
                )

    def _add_event_line(self, event_str: str) -> None:
        """Add an event line to the latest panel."""
        self.event_lines.append(event_str)

    def _update_ui(self) -> None:
        """Update all UI panels (debounced)."""
        self.update_scheduled = False

        # Update agents panel
        agents_panel = self.query_one("#agents-panel", AgentsPanel)
        agents_panel.agents = dict(self.agents)

        # Update message flow panel
        message_flow_panel = self.query_one("#message-flow-panel", MessageFlowPanel)
        message_flow_panel.messages = list(self.messages)

        # Update latest panel
        latest_panel = self.query_one("#latest-panel", LatestPanel)
        latest_panel.events = deque(self.event_lines)

        # Update stats footer
        stats_footer = self.query_one("#stats-footer", StatsFooter)
        stats_footer.stats = self.stats

    @on(Input.Submitted)
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission from the input bar."""
        if not self.enable_input or self.router is None or self.entry_agent is None:
            return

        line = event.value.strip()
        event.input.clear()

        if not line:
            return

        # -- Slash commands ------------------------------------------------
        if line.startswith("/"):
            should_quit = await self._handle_command(line)
            if should_quit:
                if self.shutdown_event is not None:
                    self.shutdown_event.set()
                self.exit()
            return

        # -- @agent routing ------------------------------------------------
        if line.startswith("@"):
            parts = line.split(None, 1)
            target = parts[0][1:]  # strip the @
            content = parts[1] if len(parts) > 1 else ""

            if target not in self.all_agents:
                self.notify(f"Unknown agent: {target}", severity="warning")
                return
            if not content:
                self.notify(f"No message provided for @{target}", severity="warning")
                return

            await self.router.send("user", target, content)
            return

        # -- Plain text -> entry agent --------------------------------------
        await self.router.send("user", self.entry_agent, line)

    async def _handle_command(self, line: str) -> bool:
        """Process a slash command. Returns True if the app should exit."""
        cmd = line.split()[0].lower()

        if cmd == "/done":
            return True

        if cmd == "/status":
            if self.heartbeat is not None:
                await self.heartbeat.fire()
            else:
                self.notify("Status: all agents idle", severity="information")
            return False

        if cmd == "/agents":
            agent_list = []
            for name, agent in self.all_agents.items():
                # Import here to avoid circular dependency
                from lattice.agent.cli_bridge import CLIBridge
                from lattice.agent.script_bridge import ScriptBridge

                if isinstance(agent, CLIBridge):
                    agent_type = "cli"
                elif isinstance(agent, ScriptBridge):
                    agent_type = "script"
                else:
                    agent_type = "llm"
                agent_list.append(f"  {name} ({agent_type})")

            self.notify("\n".join(agent_list), severity="information", timeout=10)
            return False

        self.notify(f"Unknown command: {cmd}", severity="warning")
        return False


# ------------------------------------------------------------------ #
# Click commands
# ------------------------------------------------------------------ #


@click.command()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output.")
@click.option(
    "--session",
    "session_file",
    type=click.Path(exists=True, path_type=Path),
    help="Specific session file to watch (auto-detected if omitted).",
)
def watch(verbose: bool, session_file: Path | None) -> None:
    """Watch agent team activity in real-time with a live TUI."""
    if session_file is None:
        session_file = _find_latest_session()
        if session_file is None:
            click.echo(
                "Error: No active session found. Run `lattice up` first.",
                err=True,
            )
            raise SystemExit(1)
        click.echo(f"Watching: {session_file}")

    app = WatchApp(session_file=session_file, enable_input=False, show_tools=verbose)
    app.run()


def _find_latest_session() -> Path | None:
    """Find the most recent session JSONL file in ./sessions/."""
    sessions_dir = Path("sessions")
    if not sessions_dir.exists():
        return None

    session_files = sorted(
        sessions_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # Filter out .verbose.jsonl files
    session_files = [f for f in session_files if not f.name.endswith(".verbose.jsonl")]

    return session_files[0] if session_files else None
