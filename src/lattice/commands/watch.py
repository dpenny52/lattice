"""lattice watch — watch for changes and re-run."""

import click


@click.command()
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output.")
def watch(verbose: bool) -> None:
    """Watch for file changes and re-run."""
    click.echo("lattice watch: not yet implemented")
