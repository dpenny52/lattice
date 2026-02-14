"""Root CLI group and version flag."""

import click

from lattice import __version__
from lattice.commands.down import down
from lattice.commands.init import init
from lattice.commands.replay import replay
from lattice.commands.up import up
from lattice.commands.watch import watch


@click.group()
@click.version_option(version=__version__, prog_name="lattice")
def cli() -> None:
    """Lattice — declarative dev-environment orchestrator."""


cli.add_command(init)
cli.add_command(up)
cli.add_command(down)
cli.add_command(watch)
cli.add_command(replay)
