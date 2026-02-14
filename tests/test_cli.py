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


def test_init_runs() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["init"])
        assert result.exit_code == 0
        assert "Created lattice.yaml" in result.output


def test_up_no_config_errors() -> None:
    """lattice up without a lattice.yaml exits with error."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["up"])
        assert result.exit_code == 1
        assert "lattice.yaml" in result.output or "Error" in result.output


def test_up_flags() -> None:
    result = CliRunner().invoke(cli, ["up", "--help"])
    assert result.exit_code == 0
    assert "--watch" in result.output
    assert "--loop" in result.output
    assert "--verbose" in result.output


def test_down_no_session() -> None:
    result = CliRunner().invoke(cli, ["down"])
    assert result.exit_code == 1
    assert "No running session found" in result.output


def test_watch_stub() -> None:
    result = CliRunner().invoke(cli, ["watch"])
    assert result.exit_code == 0
    assert "not yet implemented" in result.output


def test_replay_stub() -> None:
    result = CliRunner().invoke(cli, ["replay", "test-session"])
    assert result.exit_code == 0
    assert "not yet implemented" in result.output
