"""Smoke tests for the Lattice CLI."""

from click.testing import CliRunner

from lattice import __version__
from lattice.cli import cli


def test_help() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Lattice" in result.output


def test_version() -> None:
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert f"lattice, version {__version__}" in result.output


def test_init_stub() -> None:
    result = CliRunner().invoke(cli, ["init"])
    assert result.exit_code == 0
    assert "not yet implemented" in result.output


def test_up_stub() -> None:
    result = CliRunner().invoke(cli, ["up"])
    assert result.exit_code == 0
    assert "not yet implemented" in result.output


def test_up_flags() -> None:
    result = CliRunner().invoke(cli, ["up", "--help"])
    assert result.exit_code == 0
    assert "--watch" in result.output
    assert "--loop" in result.output
    assert "--verbose" in result.output


def test_down_stub() -> None:
    result = CliRunner().invoke(cli, ["down"])
    assert result.exit_code == 0
    assert "not yet implemented" in result.output


def test_watch_stub() -> None:
    result = CliRunner().invoke(cli, ["watch"])
    assert result.exit_code == 0
    assert "not yet implemented" in result.output


def test_replay_stub() -> None:
    result = CliRunner().invoke(cli, ["replay", "test-session"])
    assert result.exit_code == 0
    assert "not yet implemented" in result.output
