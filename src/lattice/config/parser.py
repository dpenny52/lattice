"""Load, validate, and resolve lattice.yaml configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import ValidationError

from lattice.config.models import LatticeConfig

DEFAULT_CONFIG_NAME = "lattice.yaml"


class ConfigError(Exception):
    """User-facing configuration error."""


def load_config(path: Path | None = None) -> LatticeConfig:
    """Load and validate a lattice.yaml file.

    Args:
        path: Explicit config file path. If None, looks for
              lattice.yaml in the current directory.

    Returns:
        A validated LatticeConfig instance.

    Raises:
        ConfigError: On missing file, bad YAML, or validation failure.
    """
    config_path = _resolve_path(path)
    raw = _read_yaml(config_path)
    _resolve_role_references(raw, config_path.parent)
    _load_env(config_path.parent)
    return _validate(raw)


def _resolve_path(path: Path | None) -> Path:
    if path is not None:
        resolved = Path(path)
        if not resolved.is_file():
            msg = f"Config file not found: {resolved}"
            raise ConfigError(msg)
        return resolved

    default = Path.cwd() / DEFAULT_CONFIG_NAME
    if not default.is_file():
        msg = (
            f"No {DEFAULT_CONFIG_NAME} found in {Path.cwd()}. "
            "Run `lattice init` to create one."
        )
        raise ConfigError(msg)
    return default


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"Cannot read config file: {exc}"
        raise ConfigError(msg) from exc

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        detail = ""
        if hasattr(exc, "problem_mark") and exc.problem_mark is not None:
            mark = exc.problem_mark
            detail = f" (line {mark.line + 1}, column {mark.column + 1})"
        msg = f"Invalid YAML in {path.name}{detail}"
        raise ConfigError(msg) from exc

    if not isinstance(data, dict):
        msg = f"Expected a YAML mapping in {path.name}, got {type(data).__name__}"
        raise ConfigError(msg)

    return data


def _resolve_role_references(raw: dict[str, Any], base_dir: Path) -> None:
    agents = raw.get("agents")
    if not isinstance(agents, dict):
        return

    for name, agent in agents.items():
        if not isinstance(agent, dict):
            continue
        role = agent.get("role")
        if not isinstance(role, str):
            continue
        if role.startswith("./") or role.startswith("/"):
            role_path = (base_dir / role).resolve()
            if not role_path.is_relative_to(base_dir.resolve()):
                msg = f"Role file for agent '{name}' escapes project directory: {role}"
                raise ConfigError(msg)
            if not role_path.is_file():
                msg = f"Role file not found for agent '{name}': {role}"
                raise ConfigError(msg)
            agent["role"] = role_path.read_text(encoding="utf-8").strip()


def _load_env(config_dir: Path) -> None:
    env_path = config_dir / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


def _validate(raw: dict[str, Any]) -> LatticeConfig:
    try:
        return LatticeConfig.model_validate(raw)
    except ValidationError as exc:
        errors = exc.errors()
        parts: list[str] = []
        for err in errors:
            loc = " → ".join(str(s) for s in err["loc"])
            msg = err["msg"]
            # Make certain error messages more user-friendly
            if "field required" in msg.lower():
                msg = "This field is required"
            elif "input should be" in msg.lower():
                msg = f"Invalid value: {msg}"
            parts.append(f"  {loc}: {msg}")
        joined = "\n".join(parts)
        msg = f"Config validation failed:\n{joined}"
        raise ConfigError(msg) from exc
