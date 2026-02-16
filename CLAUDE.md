# Lattice

**Always run checks before committing:** `.venv/bin/ruff check src/ tests/` (lint), `.venv/bin/ruff format --check src/ tests/` (formatting), and `.venv/bin/mypy src/` (types) — fix any errors before pushing.

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

## Running `lattice up` from Claude Code

Running the dogfood config from inside a Claude Code session requires workarounds:

1. **Nested session guard** — Claude CLI refuses to launch inside another Claude Code session (`CLAUDECODE` env var is set). Bypass with: `env -u CLAUDECODE .venv/bin/lattice up ...`

2. **REPL stdin with `-p` flag** — When using `-p "prompt"` in a background process, the REPL hits `EOFError` on `input()` immediately after sending the prompt (no stdin). The session exits before CLI agents finish their work. Workaround: pipe a prompt followed by a long sleep to keep stdin open:
   ```bash
   (echo "your prompt"; sleep 600) | env -u CLAUDECODE .venv/bin/lattice up -f lattice-dogfood.yaml
   ```

3. **`--loop` mode race condition** — `_loop_mode` in `up.py` checks `router.pending_tasks` after a 50ms sleep + 500ms gather timeout. If the entry agent's task completes before spawned subtasks are tracked in `pending_tasks`, the loop exits prematurely (often in <1 second with only 7 events). The loop thinks there's no pending work because the _router_ task for the entry agent finished, even though the entry agent dispatched work to CLI agents that are still initializing.

## Resolved: Silent Process Crash (was killing itself via tests)

The lattice process was dying silently ~3 minutes into dogfood sessions. No traceback, no crash report, no OOM — just gone.

**Root cause:** `tests/test_cli.py::test_down_no_session` invoked `CliRunner().invoke(cli, ["down"])` without `isolated_filesystem()`. When lattice was running (`.lattice/session.pid` exists in CWD), the test read the real pidfile, confirmed the process was alive via `os.kill(pid, 0)`, and sent `os.kill(pid, SIGTERM)`. The crash always coincided with CLI agents running `pytest tests/` via Bash tool calls.

**Fix:** Wrapped `test_down_no_session` with `runner.isolated_filesystem()` (matching the pattern `test_watch_no_session` already used). This ensures the test sees no pidfile and correctly returns "No running session found".

**Diagnostics retained** (still useful for future debugging):
- `faulthandler.enable()` in `cli.py` — dumps traceback on segfault
- `start_new_session=True` on subprocess spawning — isolates child process groups
- `SA_SIGINFO` signal handler in `up.py` — logs sender PID on SIGTERM
- Atexit watchdog in `cli.py` — detects unclean exits
- Asyncio exception handler in `up.py` — catches unhandled task exceptions

## Memory Profile: What We Know

Each Claude CLI subprocess (`type: cli, cli: claude`) uses **300–430 MB RSS** and grows over time. With 3 concurrent CLI agents (dev + code-reviewer + security-reviewer), expect **~1.2 GB** of subprocess memory on top of the host process (~85 MB).

The pre-flight memory gate in `cli_bridge.py` blocks subprocess spawning when system available memory < 1 GB (`_MIN_AVAILABLE_MB = 1024`). On memory-constrained systems this can prevent all CLI agents from launching, leaving the session idle with only heartbeat pings.

Memory sidecar files are written to `sessions/<session>.<agent>.memory.jsonl` with snapshots every 10 seconds. Key fields:
- `process_rss_mb` — host Python process RSS (same for all agents, ~85 MB)
- `subprocess_rss_mb` — Claude CLI subprocess RSS (300-430 MB per agent)
- `system_available_mb` — free + inactive + purgeable pages
- `thread_size_kb` / `thread_messages` — LLM agent conversation thread size (grows with each exchange)
- `queue_depth` — CLI bridge message queue (>0 means messages waiting for busy subprocess)
