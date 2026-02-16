"""lattice replay — replay a recorded session."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import click
from pydantic import ValidationError
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.reactive import reactive
from textual.widgets import Header, Input, Label, Static

from lattice.session.models import (
    AgentDoneEvent,
    AgentStartEvent,
    CLIProgressEvent,
    CLITextChunkEvent,
    CLIThinkingEvent,
    CLIToolCallEvent,
    ErrorEvent,
    LLMCallEndEvent,
    LLMCallStartEvent,
    LoopBoundaryEvent,
    MessageEvent,
    SessionEndEvent,
    SessionEvent,
    SessionStartEvent,
    StatusEvent,
    ToolCallEvent,
    ToolResultEvent,
)


@dataclass
class SessionMetadata:
    """Metadata extracted from a session file."""

    session_id: str
    team: str
    start_ts: datetime
    end_ts: datetime | None
    duration_ms: int | None
    message_count: int
    total_tokens: dict[str, int]
    agents: set[str]
    event_count: int
    file_path: Path
    is_complete: bool

    @property
    def duration_str(self) -> str:
        """Human-readable duration string."""
        if self.duration_ms is None:
            return "in progress"
        seconds = self.duration_ms / 1000
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = seconds / 60
        if minutes < 60:
            return f"{minutes:.1f}m"
        hours = minutes / 60
        return f"{hours:.1f}h"


@dataclass
class SessionData:
    """Parsed session data ready for replay."""

    metadata: SessionMetadata
    events: list[SessionEvent]
    parse_warnings: list[str]


def _parse_session_file(file_path: Path) -> SessionData:
    """Parse a session JSONL file into structured data.

    Returns parsed events and metadata, with warnings for any malformed lines.
    """
    events: list[SessionEvent] = []
    warnings: list[str] = []

    # First pass: parse all events
    with file_path.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                raw = json.loads(line)
                event = _parse_event(raw)
                events.append(event)
            except json.JSONDecodeError as e:
                warnings.append(f"Line {line_num}: Invalid JSON — {e}")
            except ValidationError as e:
                warnings.append(f"Line {line_num}: Validation error — {e}")
            except Exception as e:
                warnings.append(f"Line {line_num}: Unexpected error — {e}")

    # Extract metadata from events
    metadata = _extract_metadata(file_path, events)

    return SessionData(metadata=metadata, events=events, parse_warnings=warnings)


def _parse_event(raw: dict[str, Any]) -> SessionEvent:
    """Parse a raw JSON dict into a SessionEvent.

    Uses Pydantic's discriminated union to automatically select the right
    event type based on the "type" field.
    """
    # Import the SessionEvent type adapter
    from pydantic import TypeAdapter

    adapter: TypeAdapter[SessionEvent] = TypeAdapter(SessionEvent)
    return adapter.validate_python(raw)


def _extract_metadata(file_path: Path, events: list[SessionEvent]) -> SessionMetadata:
    """Extract session metadata from parsed events."""
    session_id = "unknown"
    team = "unknown"
    start_ts = datetime.now()
    end_ts: datetime | None = None
    duration_ms: int | None = None
    message_count = 0
    accumulated_tokens = {"input": 0, "output": 0}
    session_end_tokens: dict[str, int] | None = None
    agents: set[str] = set()
    is_complete = False

    for event in events:
        if isinstance(event, SessionStartEvent):
            session_id = event.session_id
            team = event.team
            start_ts = datetime.fromisoformat(event.ts.replace("Z", "+00:00"))
        elif isinstance(event, SessionEndEvent):
            end_ts = datetime.fromisoformat(event.ts.replace("Z", "+00:00"))
            duration_ms = event.duration_ms
            session_end_tokens = event.total_tokens
            is_complete = True
        elif isinstance(event, MessageEvent):
            message_count += 1
        elif isinstance(event, LLMCallEndEvent):
            accumulated_tokens["input"] += event.tokens.get("input", 0)
            accumulated_tokens["output"] += event.tokens.get("output", 0)
        elif isinstance(event, AgentStartEvent):
            agents.add(event.agent)

    # Prefer session_end tokens if present and non-zero, otherwise use accumulated
    total_tokens = accumulated_tokens
    if session_end_tokens is not None:
        end_total = (
            session_end_tokens.get("input", 0)
            + session_end_tokens.get("output", 0)
        )
        if end_total > 0:
            total_tokens = session_end_tokens

    return SessionMetadata(
        session_id=session_id,
        team=team,
        start_ts=start_ts,
        end_ts=end_ts,
        duration_ms=duration_ms,
        message_count=message_count,
        total_tokens=total_tokens,
        agents=agents,
        event_count=len(events),
        file_path=file_path,
        is_complete=is_complete,
    )


def _list_sessions(sessions_dir: Path) -> list[SessionMetadata]:
    """List all sessions in the sessions directory.

    Returns metadata for each session, sorted by start time (most recent first).
    """
    if not sessions_dir.exists():
        return []

    sessions: list[SessionMetadata] = []
    for file_path in sessions_dir.glob("*.jsonl"):
        # Skip verbose sidecar files
        if file_path.stem.endswith(".verbose"):
            continue

        try:
            data = _parse_session_file(file_path)
            sessions.append(data.metadata)
        except Exception as e:
            click.echo(f"⚠️  Failed to parse {file_path.name}: {e}", err=True)

    # Sort by start time, most recent first
    sessions.sort(key=lambda s: s.start_ts, reverse=True)
    return sessions


def _format_session_list(sessions: list[SessionMetadata]) -> None:
    """Pretty-print a list of sessions."""
    if not sessions:
        click.echo("No sessions found. Run `lattice up` to create one.")
        return

    click.echo(f"\n📊 Found {len(sessions)} session(s):\n")

    # Table header
    header = (
        f"{'SESSION ID':<14} {'TEAM':<20} {'START':<20} "
        f"{'DURATION':<12} {'MESSAGES':<10} {'TOKENS':<12}"
    )
    click.echo(header)
    click.echo("─" * len(header))

    # Table rows
    for session in sessions:
        start_str = session.start_ts.strftime("%Y-%m-%d %H:%M:%S")
        total = (
            session.total_tokens['input']
            + session.total_tokens['output']
        )
        tokens_str = f"{total:,}"
        status_marker = "✓" if session.is_complete else "⋯"

        row = (
            f"{session.session_id:<14} "
            f"{session.team:<20} "
            f"{start_str:<20} "
            f"{session.duration_str:<12} "
            f"{session.message_count:<10} "
            f"{tokens_str:<12} {status_marker}"
        )
        click.echo(row)

    click.echo()


def _format_session_detail(data: SessionData) -> None:
    """Pretty-print detailed session information."""
    meta = data.metadata

    click.echo(f"\n📝 Session: {meta.session_id}")
    click.echo(f"   Team: {meta.team}")
    click.echo(f"   Started: {meta.start_ts.strftime('%Y-%m-%d %H:%M:%S')}")
    if meta.end_ts:
        click.echo(f"   Ended: {meta.end_ts.strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo(f"   Duration: {meta.duration_str}")
    click.echo(f"   Status: {'Complete ✓' if meta.is_complete else 'In Progress ⋯'}")
    click.echo("\n📊 Statistics:")
    click.echo(f"   Events: {meta.event_count:,}")
    click.echo(f"   Messages: {meta.message_count:,}")
    tok_in = meta.total_tokens['input']
    tok_out = meta.total_tokens['output']
    click.echo(f"   Tokens: {tok_in:,} in / {tok_out:,} out")
    click.echo(f"\n👥 Agents ({len(meta.agents)}):")
    for agent in sorted(meta.agents):
        click.echo(f"   • {agent}")

    if data.parse_warnings:
        click.echo(f"\n⚠️  Parse warnings ({len(data.parse_warnings)}):")
        for warning in data.parse_warnings[:10]:  # Limit to first 10
            click.echo(f"   {warning}")
        if len(data.parse_warnings) > 10:
            click.echo(f"   ... and {len(data.parse_warnings) - 10} more")

    click.echo()


# ------------------------------------------------------------------ #
# Textual app for interactive replay
# ------------------------------------------------------------------ #


class SessionMetadataPanel(Static):
    """Top panel showing session metadata."""

    metadata: reactive[SessionMetadata | None] = reactive(None, recompose=True)

    def compose(self) -> ComposeResult:
        """Render session metadata."""
        if self.metadata is None:
            yield Label("No session loaded", classes="empty-state")
            return

        meta = self.metadata
        yield Label(f"[bold]Session:[/bold] {meta.session_id}", markup=True)
        team_dur = (
            f"[bold]Team:[/bold] {meta.team}"
            f" | [bold]Duration:[/bold] {meta.duration_str}"
        )
        yield Label(team_dur, markup=True)
        tok_in = meta.total_tokens['input']
        tok_out = meta.total_tokens['output']
        stats = (
            f"[bold]Events:[/bold] {meta.event_count:,}"
            f" | [bold]Messages:[/bold] {meta.message_count:,}"
            f" | [bold]Tokens:[/bold] {tok_in:,}i / {tok_out:,}o"
        )
        yield Label(stats, markup=True)
        agents_str = ", ".join(sorted(meta.agents))
        yield Label(f"[bold]Agents:[/bold] {agents_str}", markup=True)


class EventDetailPanel(VerticalScroll):
    """Main panel showing the current event detail."""

    # Strip inherited scroll bindings (up/down/pgup/pgdown/home/end) so all
    # key events bubble up to the App, where our navigation bindings live.
    # Mouse-wheel scrolling still works for long events.
    BINDINGS = []

    event: reactive[SessionEvent | None] = reactive(None, recompose=True)
    verbose_data: reactive[dict[int, Any]] = reactive(dict)

    def compose(self) -> ComposeResult:
        """Render the current event."""
        if self.event is None:
            yield Label("No event selected", classes="empty-state")
            return

        event = self.event

        # Header: seq, timestamp, type
        yield Label(
            f"[bold cyan]Event #{event.seq:04d}[/bold cyan]"
            f" | {event.ts} | [yellow]{event.type}[/yellow]",
            markup=True,
        )
        yield Label("")  # Blank line

        # Event-specific rendering
        if isinstance(event, SessionStartEvent):
            yield Label(f"[bold]Session ID:[/bold] {event.session_id}", markup=True)
            yield Label(f"[bold]Team:[/bold] {event.team}", markup=True)
            yield Label(f"[bold]Config Hash:[/bold] {event.config_hash}", markup=True)

        elif isinstance(event, SessionEndEvent):
            yield Label(f"[bold]Reason:[/bold] {event.reason}", markup=True)
            yield Label(f"[bold]Duration:[/bold] {event.duration_ms:,}ms", markup=True)
            yield Label(
                f"[bold]Total Tokens:[/bold] "
                f"{event.total_tokens['input']:,}i"
                f" / {event.total_tokens['output']:,}o",
                markup=True,
            )

        elif isinstance(event, MessageEvent):
            from_to = (
                f"[bold]From:[/bold] {event.from_agent}"
                f" → [bold]To:[/bold] {event.to}"
            )
            yield Label(from_to, markup=True)
            yield Label("[bold]Content:[/bold]", markup=True)
            # Render message content with word wrapping
            for line in event.content.split("\n"):
                yield Label(f"  {line}", markup=False)

        elif isinstance(event, LLMCallStartEvent):
            yield Label(f"[bold]Agent:[/bold] {event.agent}", markup=True)
            yield Label(f"[bold]Model:[/bold] {event.model}", markup=True)
            yield Label(
                f"[bold]Messages Count:[/bold] "
                f"{event.messages_count}",
                markup=True,
            )

        elif isinstance(event, LLMCallEndEvent):
            yield Label(f"[bold]Agent:[/bold] {event.agent}", markup=True)
            yield Label(f"[bold]Model:[/bold] {event.model}", markup=True)
            yield Label(f"[bold]Duration:[/bold] {event.duration_ms:,}ms", markup=True)
            yield Label(
                f"[bold]Tokens:[/bold] "
                f"{event.tokens['input']:,}i"
                f" / {event.tokens['output']:,}o",
                markup=True,
            )

        elif isinstance(event, ToolCallEvent):
            yield Label(f"[bold]Agent:[/bold] {event.agent}", markup=True)
            yield Label(f"[bold]Tool:[/bold] {event.tool}", markup=True)
            yield Label("[bold]Arguments:[/bold]", markup=True)
            args_json = json.dumps(event.args, indent=2, default=str)
            for line in args_json.split("\n"):
                yield Label(f"  {line}", markup=False)

        elif isinstance(event, ToolResultEvent):
            yield Label(f"[bold]Agent:[/bold] {event.agent}", markup=True)
            yield Label(f"[bold]Tool:[/bold] {event.tool}", markup=True)
            yield Label(f"[bold]Duration:[/bold] {event.duration_ms:,}ms", markup=True)
            yield Label(
                f"[bold]Result Size:[/bold] "
                f"{event.result_size:,} bytes",
                markup=True,
            )

            # If verbose mode and we have the full result, display it
            if event.seq in self.verbose_data:
                yield Label("")
                yield Label("[bold]Full Result:[/bold]", markup=True)
                result_json = json.dumps(
                    self.verbose_data[event.seq],
                    indent=2,
                    default=str,
                )
                # Limit to first 100 lines
                lines = result_json.split("\n")
                for line in lines[:100]:
                    yield Label(f"  {line}", markup=False)
                if len(lines) > 100:
                    yield Label(f"  ... ({len(lines) - 100} more lines)", markup=False)

        elif isinstance(event, StatusEvent):
            yield Label(f"[bold]Agent:[/bold] {event.agent}", markup=True)
            yield Label(f"[bold]Status:[/bold] {event.status}", markup=True)

        elif isinstance(event, ErrorEvent):
            yield Label(f"[bold]Agent:[/bold] {event.agent}", markup=True)
            yield Label(f"[bold]Retrying:[/bold] {event.retrying}", markup=True)
            yield Label("[bold red]Error:[/bold red]", markup=True)
            for line in event.error.split("\n"):
                yield Label(f"  {line}", markup=False)

        elif isinstance(event, AgentStartEvent):
            yield Label(f"[bold]Agent:[/bold] {event.agent}", markup=True)
            yield Label(f"[bold]Type:[/bold] {event.agent_type}", markup=True)

        elif isinstance(event, AgentDoneEvent):
            yield Label(f"[bold]Agent:[/bold] {event.agent}", markup=True)
            yield Label(f"[bold]Reason:[/bold] {event.reason}", markup=True)

        elif isinstance(event, LoopBoundaryEvent):
            yield Label(f"[bold]Boundary:[/bold] {event.boundary}", markup=True)
            yield Label(f"[bold]Iteration:[/bold] {event.iteration}", markup=True)

        elif isinstance(event, CLITextChunkEvent):
            yield Label(f"[bold]Agent:[/bold] {event.agent}", markup=True)
            yield Label("[bold]Text:[/bold]", markup=True)
            for line in event.text.split("\n"):
                yield Label(f"  {line}", markup=False)

        elif isinstance(event, CLIToolCallEvent):
            yield Label(f"[bold]Agent:[/bold] {event.agent}", markup=True)
            yield Label(f"[bold]Tool:[/bold] {event.tool}", markup=True)
            yield Label("[bold]Arguments:[/bold]", markup=True)
            args_json = json.dumps(event.args, indent=2, default=str)
            for line in args_json.split("\n"):
                yield Label(f"  {line}", markup=False)

        elif isinstance(event, CLIThinkingEvent):
            yield Label(f"[bold]Agent:[/bold] {event.agent}", markup=True)
            yield Label("[bold]Content:[/bold]", markup=True)
            for line in event.content.split("\n"):
                yield Label(f"  {line}", markup=False)

        elif isinstance(event, CLIProgressEvent):
            yield Label(f"[bold]Agent:[/bold] {event.agent}", markup=True)
            yield Label(f"[bold]Status:[/bold] {event.status}", markup=True)


class NavigationFooter(Static):
    """Footer showing navigation hints and current position."""

    current_index: reactive[int] = reactive(0)
    total_events: reactive[int] = reactive(0)

    def render(self) -> str:
        """Render navigation status."""
        if self.total_events == 0:
            return "No events"

        position = f"Event {self.current_index + 1} / {self.total_events}"
        hints = "j/↓: Next | k/↑: Prev | g: Jump to | /: Search | q: Quit"
        return f"{position} | {hints}"


class ReplayApp(App[None]):
    """Interactive session replay debugger."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #metadata-panel {
        height: auto;
        border: solid $primary;
        padding: 1;
        margin-bottom: 1;
    }

    #event-detail-panel {
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }

    #navigation-footer {
        dock: bottom;
        height: 3;
        background: $panel;
        border-top: solid $primary;
        padding: 1;
    }

    #input-overlay {
        display: none;
        dock: bottom;
        height: 3;
        background: $panel;
        border-top: solid $primary;
        padding: 0 1;
    }

    .empty-state {
        color: $text-muted;
        text-style: italic;
    }
    """

    BINDINGS = [
        Binding("j", "next_event", "Next"),
        Binding("k", "prev_event", "Previous"),
        Binding("down", "next_event", "Next", show=False, priority=True),
        Binding("up", "prev_event", "Previous", show=False, priority=True),
        Binding("g", "jump_to", "Jump to"),
        Binding("slash", "filter_search", "Filter"),
        Binding("a", "filter_agent", "Filter Agent"),
        Binding("t", "filter_type", "Filter Type"),
        Binding("c", "clear_filter", "Clear Filter"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(
        self,
        session_data: SessionData,
        verbose_data: dict[int, Any] | None = None,
    ) -> None:
        """Initialize the replay app.

        Args:
            session_data: Parsed session data
            verbose_data: Optional dict mapping seq -> full tool result
        """
        super().__init__()
        self.session_data = session_data
        self.verbose_data = verbose_data if verbose_data is not None else {}
        self.current_index = 0

        # Filtering state
        self.all_events = session_data.events
        self.filtered_events = list(self.all_events)
        self.active_filter: str | None = None

        # Input mode
        self.input_mode: str | None = None  # "jump", "search", etc.

    def compose(self) -> ComposeResult:
        """Compose the TUI layout."""
        yield Header(show_clock=False)
        yield SessionMetadataPanel(id="metadata-panel")
        yield EventDetailPanel(id="event-detail-panel")
        yield NavigationFooter(id="navigation-footer")
        with Container(id="input-overlay"):
            yield Input(placeholder="Enter command...", id="input-field")

    def on_mount(self) -> None:
        """Initialize state on mount."""
        # Set metadata
        metadata_panel = self.query_one("#metadata-panel", SessionMetadataPanel)
        metadata_panel.metadata = self.session_data.metadata

        # Set verbose data
        event_panel = self.query_one("#event-detail-panel", EventDetailPanel)
        event_panel.verbose_data = self.verbose_data

        # Show first event
        self._update_event_display()

        # Focus the event panel so key events bubble up to app bindings.
        event_panel.focus()

    def _update_event_display(self) -> None:
        """Update the event detail panel with the current event."""
        if not self.filtered_events:
            return

        event = self.filtered_events[self.current_index]

        event_panel = self.query_one("#event-detail-panel", EventDetailPanel)
        event_panel.event = event

        nav_footer = self.query_one("#navigation-footer", NavigationFooter)
        nav_footer.current_index = self.current_index
        nav_footer.total_events = len(self.filtered_events)

        # Scroll to top
        event_panel.scroll_home(animate=False)

    def action_next_event(self) -> None:
        """Navigate to the next event."""
        if self.current_index < len(self.filtered_events) - 1:
            self.current_index += 1
            self._update_event_display()

    def action_prev_event(self) -> None:
        """Navigate to the previous event."""
        if self.current_index > 0:
            self.current_index -= 1
            self._update_event_display()

    def _show_input_dialog(self, mode: str, placeholder: str) -> None:
        """Show the input overlay with the given mode and placeholder."""
        self.input_mode = mode
        input_overlay = self.query_one("#input-overlay")
        input_overlay.display = True
        input_field = self.query_one("#input-field", Input)
        input_field.value = ""
        input_field.placeholder = placeholder
        input_field.focus()

    def action_jump_to(self) -> None:
        """Prompt for a sequence number to jump to."""
        self._show_input_dialog("jump", "Enter sequence number (0-indexed)...")

    def action_filter_search(self) -> None:
        """Prompt for a text search filter."""
        self._show_input_dialog("search", "Search event content...")

    def action_filter_agent(self) -> None:
        """Prompt for an agent name filter."""
        self._show_input_dialog("agent", "Filter by agent name...")

    def action_filter_type(self) -> None:
        """Prompt for an event type filter."""
        self._show_input_dialog(
            "type",
            "Filter by event type (e.g., message, tool_call)...",
        )

    def action_clear_filter(self) -> None:
        """Clear any active filters."""
        self.filtered_events = list(self.all_events)
        self.active_filter = None
        self.current_index = 0
        self._update_event_display()
        self.notify("Filter cleared")

    @on(Input.Submitted)
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        value = event.value.strip()

        # Hide input overlay and return focus to event panel.
        input_overlay = self.query_one("#input-overlay")
        input_overlay.display = False
        self.query_one("#event-detail-panel", EventDetailPanel).focus()

        if self.input_mode == "jump":
            self._handle_jump(value)
        elif self.input_mode == "search":
            self._handle_search(value)
        elif self.input_mode == "agent":
            self._handle_agent_filter(value)
        elif self.input_mode == "type":
            self._handle_type_filter(value)

        self.input_mode = None

    def _handle_jump(self, value: str) -> None:
        """Jump to a specific sequence number."""
        try:
            seq = int(value)
            # Find the event with this seq in filtered_events
            for i, event in enumerate(self.filtered_events):
                if event.seq == seq:
                    self.current_index = i
                    self._update_event_display()
                    self.notify(f"Jumped to event #{seq}")
                    return
            self.notify(f"Event #{seq} not found in current view", severity="warning")
        except ValueError:
            self.notify("Invalid sequence number", severity="error")

    def _handle_search(self, query: str) -> None:
        """Filter events by text content."""
        if not query:
            self.notify("Empty search query", severity="warning")
            return

        query_lower = query.lower()
        self.filtered_events = []

        for event in self.all_events:
            # Search in different fields depending on event type
            searchable_text = ""

            if isinstance(event, MessageEvent):
                searchable_text = f"{event.from_agent} {event.to} {event.content}"
            elif isinstance(event, (ToolCallEvent, CLIToolCallEvent)):
                searchable_text = f"{event.agent} {event.tool} {json.dumps(event.args)}"
            elif isinstance(event, (
                ToolResultEvent, StatusEvent, ErrorEvent,
                AgentStartEvent, AgentDoneEvent,
            )):
                searchable_text = f"{event.agent}"
            elif isinstance(event, (LLMCallStartEvent, LLMCallEndEvent)):
                searchable_text = f"{event.agent} {event.model}"
            elif isinstance(event, (CLITextChunkEvent, CLIThinkingEvent)):
                text_val = getattr(
                    event, 'text', getattr(event, 'content', ''),
                )
                searchable_text = f"{event.agent} {text_val}"
            elif isinstance(event, CLIProgressEvent):
                searchable_text = f"{event.agent} {event.status}"

            if query_lower in searchable_text.lower():
                self.filtered_events.append(event)

        self.active_filter = f"search: {query}"
        self.current_index = 0
        self._update_event_display()
        self.notify(f"Found {len(self.filtered_events)} events matching '{query}'")

    def _handle_agent_filter(self, agent_name: str) -> None:
        """Filter events by agent name."""
        if not agent_name:
            self.notify("Empty agent name", severity="warning")
            return

        self.filtered_events = []

        for event in self.all_events:
            event_agent = getattr(event, "agent", None)
            if (
                (event_agent and event_agent == agent_name)
                or (
                    isinstance(event, MessageEvent)
                    and (
                        event.from_agent == agent_name
                        or event.to == agent_name
                    )
                )
            ):
                self.filtered_events.append(event)

        self.active_filter = f"agent: {agent_name}"
        self.current_index = 0
        self._update_event_display()
        count = len(self.filtered_events)
        self.notify(f"Found {count} events for agent '{agent_name}'")

    def _handle_type_filter(self, event_type: str) -> None:
        """Filter events by event type."""
        if not event_type:
            self.notify("Empty event type", severity="warning")
            return

        self.filtered_events = [e for e in self.all_events if e.type == event_type]

        self.active_filter = f"type: {event_type}"
        self.current_index = 0
        self._update_event_display()
        self.notify(f"Found {len(self.filtered_events)} events of type '{event_type}'")



def _load_verbose_data(session_file: Path) -> dict[int, Any]:
    """Load verbose sidecar data if it exists.

    Returns a dict mapping seq -> full_result.
    """
    verbose_file = session_file.parent / f"{session_file.stem}.verbose.jsonl"
    if not verbose_file.exists():
        return {}

    verbose_data: dict[int, Any] = {}
    try:
        with verbose_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    seq = entry.get("seq")
                    full_result = entry.get("full_result")
                    if seq is not None:
                        verbose_data[seq] = full_result
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass  # Gracefully handle missing or malformed verbose files

    return verbose_data


@click.command()
@click.argument("session_id", required=False)
@click.option(
    "-d",
    "--sessions-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default="sessions",
    help="Directory containing session files (default: ./sessions)",
)
@click.option(
    "-v", "--verbose",
    is_flag=True,
    help="Load full tool results from verbose sidecar.",
)
def replay(session_id: str | None, sessions_dir: Path, verbose: bool) -> None:
    """Replay a recorded session.

    If SESSION_ID is provided, load and display that specific session interactively.
    If SESSION_ID is omitted, list all available sessions.
    """
    # List mode: show all sessions
    if session_id is None:
        sessions = _list_sessions(sessions_dir)
        _format_session_list(sessions)
        return

    # Load mode: load a specific session
    # Try to find the session file
    matching_files = list(sessions_dir.glob(f"*{session_id}*.jsonl"))
    # Filter out verbose files
    matching_files = [f for f in matching_files if not f.stem.endswith(".verbose")]

    if not matching_files:
        click.echo(f"❌ Session not found: {session_id}", err=True)
        click.echo(
            "\nTip: Run 'lattice replay' to list all available sessions.",
            err=True,
        )
        raise SystemExit(1)

    if len(matching_files) > 1:
        click.echo(f"❌ Ambiguous session ID: {session_id}", err=True)
        click.echo("   Matches multiple files:", err=True)
        for f in matching_files:
            click.echo(f"     • {f.name}", err=True)
        raise SystemExit(1)

    # Parse the session file
    session_file = matching_files[0]
    click.echo(f"Loading session from {session_file.name}...")
    data = _parse_session_file(session_file)

    # Load verbose data if requested
    verbose_data: dict[int, Any] = {}
    if verbose:
        verbose_data = _load_verbose_data(session_file)
        if verbose_data:
            click.echo(f"Loaded {len(verbose_data)} verbose entries")
        else:
            click.echo("⚠️  No verbose sidecar found or empty", err=True)

    # Show parse warnings if any
    if data.parse_warnings:
        click.echo(f"\n⚠️  {len(data.parse_warnings)} parse warnings:")
        for warning in data.parse_warnings[:5]:
            click.echo(f"   {warning}")
        if len(data.parse_warnings) > 5:
            click.echo(f"   ... and {len(data.parse_warnings) - 5} more")
        click.echo()

    # Launch interactive TUI
    app = ReplayApp(session_data=data, verbose_data=verbose_data)
    app.run()
