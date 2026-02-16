"""lattice init — scaffold a new Lattice project."""

from __future__ import annotations

from pathlib import Path

import click

CONFIG_FILENAME = "lattice.yaml"
ENV_EXAMPLE_FILENAME = ".env.example"

TEMPLATE_YAML = """\
# Lattice team configuration
# Docs: https://github.com/your-org/lattice-cli
version: "1"

# Team name — used in session filenames and the TUI header
team: my-team

# Which agent receives the user's initial message (default: first listed)
# entry: researcher

agents:
  # An LLM-powered researcher that can search the web
  researcher:
    model: anthropic/claude-sonnet-4-5
    role: |
      You are a research assistant. Find relevant information on the
      topic the user asks about and summarize your findings clearly.
      When you have enough information, send your summary to the writer.
    tools:
      - web-search

  # An LLM-powered writer that synthesises research into a report
  writer:
    model: anthropic/claude-sonnet-4-5
    role: |
      You are a technical writer. Take the research findings you receive
      and synthesize them into a clear, well-structured report.
    tools:
      - file-write

# Topology — who can talk to whom (default: hub)
# Options: hub (default, workers ↔ coordinator), mesh, pipeline, custom
topology:
  type: hub
  coordinator: researcher
  workers: [writer]

# Communication settings
# communication:
#   protocol: a2a      # default
#   record: true        # auto-record sessions to ./sessions/
#   heartbeat: 20       # progress updates every N seconds (0 to disable)
"""

TEMPLATE_ENV_EXAMPLE = """\
# API keys for LLM providers used by your agents.
# Copy this file to .env and fill in your keys.
#
# Lattice reads these automatically — no extra config needed
# unless your env vars are named differently (see `credentials` in lattice.yaml).

ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
"""


@click.command()
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing lattice.yaml if it exists.",
)
def init(force: bool) -> None:
    """Scaffold a new Lattice project in the current directory."""
    cwd = Path.cwd()
    config_path = cwd / CONFIG_FILENAME
    env_example_path = cwd / ENV_EXAMPLE_FILENAME

    # Guard against overwriting an existing config
    if config_path.exists() and not force:
        raise click.ClickException(
            f"{CONFIG_FILENAME} already exists. Use --force to overwrite."
        )

    # Write lattice.yaml
    try:
        config_path.write_text(TEMPLATE_YAML, encoding="utf-8")
    except OSError as exc:
        raise click.ClickException(f"Cannot write {CONFIG_FILENAME}: {exc}") from exc
    click.echo(f"  Created {CONFIG_FILENAME}")

    # Write .env.example (skip if it already exists)
    if not env_example_path.exists() or force:
        try:
            env_example_path.write_text(TEMPLATE_ENV_EXAMPLE, encoding="utf-8")
        except OSError as exc:
            raise click.ClickException(
                f"Cannot write {ENV_EXAMPLE_FILENAME}: {exc}"
            ) from exc
        click.echo(f"  Created {ENV_EXAMPLE_FILENAME}")
    else:
        click.echo(f"  Skipped {ENV_EXAMPLE_FILENAME} (already exists)")

    # Next steps
    click.echo()
    click.echo("Next steps:")
    click.echo(f"  1. Edit {CONFIG_FILENAME} to configure your agents")
    click.echo("  2. Copy .env.example to .env and add your API keys")
    click.echo("  3. Run `lattice up` to start your team")
