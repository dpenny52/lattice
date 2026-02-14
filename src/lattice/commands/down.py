"""lattice down — tear the environment down."""

import click


@click.command()
@click.option(
    "-f", "--file", "config_file", type=click.Path(), help="Config file path."
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output.")
def down(config_file: str | None, verbose: bool) -> None:
    """Tear the environment down."""
    click.echo("lattice down: not yet implemented")
