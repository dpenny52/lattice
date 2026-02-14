"""lattice replay — replay a recorded session."""

import click


@click.command()
@click.argument("session_id")
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output.")
def replay(session_id: str, verbose: bool) -> None:
    """Replay a recorded session by SESSION_ID."""
    click.echo("lattice replay: not yet implemented")
