"""Tests for `lattice init` command."""

from __future__ import annotations

from pathlib import Path

import yaml
from click.testing import CliRunner

from lattice.cli import cli
from lattice.commands.init import (
    CONFIG_FILENAME,
    ENV_EXAMPLE_FILENAME,
    TEMPLATE_YAML,
)
from lattice.config.models import LatticeConfig


class TestInitCreatesFiles:
    """lattice init creates the expected files."""

    def test_creates_lattice_yaml(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert Path(CONFIG_FILENAME).is_file()

    def test_creates_env_example(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert Path(ENV_EXAMPLE_FILENAME).is_file()

    def test_output_mentions_created_files(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init"])
            assert f"Created {CONFIG_FILENAME}" in result.output
            assert f"Created {ENV_EXAMPLE_FILENAME}" in result.output


class TestGeneratedConfigIsValid:
    """The generated lattice.yaml must parse and validate correctly."""

    def test_yaml_parses(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init"])
            data = yaml.safe_load(Path(CONFIG_FILENAME).read_text())
            assert isinstance(data, dict)

    def test_config_validates(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init"])
            data = yaml.safe_load(Path(CONFIG_FILENAME).read_text())
            config = LatticeConfig.model_validate(data)
            assert config.version == "1"
            assert config.team == "my-team"

    def test_has_two_example_agents(self) -> None:
        data = yaml.safe_load(TEMPLATE_YAML)
        config = LatticeConfig.model_validate(data)
        assert len(config.agents) == 2

    def test_agents_have_roles(self) -> None:
        data = yaml.safe_load(TEMPLATE_YAML)
        config = LatticeConfig.model_validate(data)
        for agent in config.agents.values():
            assert agent.role is not None
            assert len(agent.role.strip()) > 0

    def test_agents_have_models(self) -> None:
        data = yaml.safe_load(TEMPLATE_YAML)
        config = LatticeConfig.model_validate(data)
        for agent in config.agents.values():
            assert agent.model is not None

    def test_entry_defaults_to_first_agent(self) -> None:
        data = yaml.safe_load(TEMPLATE_YAML)
        config = LatticeConfig.model_validate(data)
        first_agent = next(iter(data["agents"]))
        assert config.entry == first_agent


class TestConfigHasComments:
    """The generated config file includes explanatory comments."""

    def test_has_field_comments(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init"])
            text = Path(CONFIG_FILENAME).read_text()
            # Should have comments explaining fields
            assert "# " in text
            # Should mention topology
            assert "topology" in text.lower()
            # Should mention communication
            assert "communication" in text.lower()


class TestExistingFileGuard:
    """lattice init refuses to overwrite existing config without --force."""

    def test_refuses_overwrite_without_force(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path(CONFIG_FILENAME).write_text("existing content")
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 1
            assert "already exists" in result.output

    def test_preserves_existing_content(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            original = "version: '1'\nteam: old\n"
            Path(CONFIG_FILENAME).write_text(original)
            runner.invoke(cli, ["init"])
            assert Path(CONFIG_FILENAME).read_text() == original


class TestForceFlag:
    """--force overwrites existing files."""

    def test_force_overwrites_config(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path(CONFIG_FILENAME).write_text("old content")
            result = runner.invoke(cli, ["init", "--force"])
            assert result.exit_code == 0
            content = Path(CONFIG_FILENAME).read_text()
            assert content == TEMPLATE_YAML

    def test_force_overwrites_env_example(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            Path(ENV_EXAMPLE_FILENAME).write_text("OLD_KEY=")
            result = runner.invoke(cli, ["init", "--force"])
            assert result.exit_code == 0
            content = Path(ENV_EXAMPLE_FILENAME).read_text()
            assert "ANTHROPIC_API_KEY" in content


class TestEnvExample:
    """The .env.example file contains expected API key placeholders."""

    def test_contains_anthropic_key(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init"])
            content = Path(ENV_EXAMPLE_FILENAME).read_text()
            assert "ANTHROPIC_API_KEY" in content

    def test_contains_openai_key(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init"])
            content = Path(ENV_EXAMPLE_FILENAME).read_text()
            assert "OPENAI_API_KEY" in content

    def test_contains_google_key(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(cli, ["init"])
            content = Path(ENV_EXAMPLE_FILENAME).read_text()
            assert "GOOGLE_API_KEY" in content

    def test_does_not_overwrite_existing_env_example(self, tmp_path: Path) -> None:
        """Without --force, .env.example is preserved if it exists."""
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            existing = "MY_CUSTOM_KEY=secret"
            Path(ENV_EXAMPLE_FILENAME).write_text(existing)
            result = runner.invoke(cli, ["init"])
            assert result.exit_code == 0
            assert Path(ENV_EXAMPLE_FILENAME).read_text() == existing
            assert f"Skipped {ENV_EXAMPLE_FILENAME}" in result.output
            assert f"Created {ENV_EXAMPLE_FILENAME}" not in result.output


class TestNextSteps:
    """After init, helpful next steps are printed."""

    def test_prints_next_steps(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init"])
            assert "Next steps:" in result.output
            assert "lattice up" in result.output

    def test_mentions_env_setup(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init"])
            assert ".env" in result.output

    def test_mentions_editing_config(self, tmp_path: Path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["init"])
            assert CONFIG_FILENAME in result.output


class TestInitHelp:
    """lattice init --help shows the --force flag."""

    def test_help_shows_force(self) -> None:
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--help"])
        assert result.exit_code == 0
        assert "--force" in result.output
