# Lattice

Declarative multi-agent orchestration CLI. Define agent teams in YAML, run them with `lattice up`, observe with `lattice watch`, debug with `lattice replay`.

## Project Structure

```
src/lattice/
  cli.py                  # Click CLI entry point
  commands/
    up.py                 # `lattice up` — session runner, REPL, loop mode
    down.py               # `lattice down` — remote shutdown via pidfile
    init.py               # `lattice init` — scaffold a starter config
    watch.py              # `lattice watch` — live TUI (stub)
    replay.py             # `lattice replay` — session debugger (stub)
  agent/
    llm_agent.py          # LLM agent runtime (tool loop, retries, rate limiting)
    cli_bridge.py         # CLI subprocess agent (Claude CLI adapter + custom JSONL)
    script_bridge.py      # Stateless script agent (one subprocess per message)
    providers.py          # LLM provider abstraction (Anthropic, OpenAI, Google)
    tools.py              # Tool registry and execution
    builtin_tools.py      # Built-in tools: file-read, file-write, web-search, code-exec
  router/
    router.py             # Central message dispatcher, topology enforcement
    topology.py           # Topology implementations (mesh, hub, pipeline, custom)
  session/
    models.py             # Pydantic event models for JSONL recording
    recorder.py           # Append-only JSONL session writer
  config/
    models.py             # Pydantic config models (LatticeConfig, AgentConfig, etc.)
    parser.py             # YAML config loader and validator
  heartbeat.py            # Periodic progress checks to entry agent
  shutdown.py             # 4-step graceful shutdown (signal → drain → kill → close)
  pidfile.py              # PID file management for `lattice down`
```

## Tech Stack

- Python 3.12+, async/await throughout
- Click for CLI
- Pydantic v2 for all data models (config + events)
- pytest + pytest-asyncio for tests (`asyncio_mode = "auto"`)
- Ruff for linting, mypy (strict) for type checking
- Textual for TUI components
- LLM SDKs: anthropic, openai, google-genai

## Key Architecture Decisions

- **Router is the single chokepoint** for all inter-agent messages. It enforces topology, records events, and prints agent-to-agent messages to console.
- **Per-peer conversation threads** — each LLM agent maintains separate threads per peer so context doesn't bleed between conversations.
- **CLI Bridge routes results back to sender** — when a CLI subprocess (e.g. Claude) finishes, its result is sent back to the requesting agent via `router.send()`. Messages from `"user"` are print-only (no routing back).
- **Shared RateLimitGate** — all LLM agents share a single gate. When any agent hits a 429, all agents pause for 60s to avoid cascading rate limits.
- **Session JSONL is append-only** — events are flushed after every write. Verbose sidecar file (`*.verbose.jsonl`) stores full tool results matched by seq.

## Running Tests

```bash
.venv/bin/python -m pytest tests/ -v --tb=short
```

All tests use mocked LLM responses and subprocess calls. No API keys needed.

## Config Format

Configs are YAML files (default: `lattice.yaml` in cwd, or `-f path`). See `lattice-dogfood.yaml` for the project's own self-build config.

Agent types:
- `type: llm` — API-powered agent (requires `model` and `role`)
- `type: cli` — CLI subprocess agent (e.g. `cli: claude`)
- `type: script` — one-shot script (runs per message, stateless)

Topologies: `mesh`, `hub`, `pipeline`, `custom`

## Dogfood Config (`lattice-dogfood.yaml`)

Lattice building itself. The lead agent (Haiku, API-powered) reads specs from Obsidian and coordinates. All workers (dev, code-reviewer, security-reviewer) are CLI bridge agents using `claude`. Tester is a script bridge running pytest.

**Lead agent rules:**
- Only reads spec/planning files — never source code
- Describes WHAT to build, not WHERE or HOW
- Never references file paths unless the spec mentions them
- Dev, reviewers, and tester know the codebase — lead doesn't

## Development Model

Lattice is building itself. The `lattice-dogfood.yaml` config runs a team that implements features from the spec. Human intervention is for testing, config tweaks, and fixing infrastructure issues — feature dev is done by the Lattice team.

If you're working in this repo directly, you're probably debugging the orchestration layer or tweaking agent behavior, not implementing user stories.

## Current Status

Spec: `~/Documents/obsidian/general/lattice-v01-user-stories.md`

- Phases 1-4: Complete (foundation, runtime, external agents, session lifecycle)
- Phase 5: In progress (5.1 `lattice watch` TUI, 5.2 `lattice replay`)
- Phase 6: Not started (error handling polish, docs, packaging)

## Common Patterns

- Event models live in `session/models.py` — add new event types there as Pydantic models
- New tools go in `builtin_tools.py` with a JSON schema definition in `tools.py`
- Agent console output uses `on_response` callback, agent-to-agent messages print via the router
- Shutdown is managed by `ShutdownManager` with a 4-step sequence: signal → drain → kill → close
