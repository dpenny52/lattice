# Lattice

Declarative multi-agent orchestration CLI. Define agent teams in YAML, run them with `lattice up`, observe with `lattice watch`.

![lattice watch TUI](docs/watch-tui.png)

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
