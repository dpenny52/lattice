# Lattice

Declarative dev-environment orchestrator.

## Install

```bash
uv pip install lattice-cli
```

## Usage

```bash
lattice --help
lattice init
lattice up
lattice down
lattice watch
lattice replay <session-id>
```

## Development

```bash
uv sync
uv run lattice --help
uv run pytest tests -v
uv run ruff check .
uv run mypy src
```
