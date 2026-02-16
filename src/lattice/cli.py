"""Root CLI group and version flag."""

import atexit
import faulthandler
import signal
import sys

import click

# Dump tracebacks on segfaults (e.g. from ctypes memory monitoring).
faulthandler.enable()

# Ensure SIGPIPE doesn't silently kill the process (e.g. when stdout
# pipe closes while click.echo is writing).  Python normally sets
# SIG_IGN at startup, but this is defensive.
if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)

# Log an atexit message so we can distinguish "process exited normally"
# (atexit fires) from "process killed by signal" (atexit doesn't fire).
def _exit_watchdog() -> None:
    click.echo(
        f"lattice process exiting (exit code {getattr(sys, 'last_value', '?')})",
        err=True,
    )

atexit.register(_exit_watchdog)

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
