"""lattice up — bring the environment up."""

import click


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
    """Bring the environment up."""
    click.echo("lattice up: not yet implemented")
