# Lattice

Declarative multi-agent orchestration CLI. Define agent teams in YAML, run them with `lattice up`, observe with `lattice watch`, debug with `lattice replay`.

![lattice watch TUI](docs/watch-tui.png)

## Quick Start

```bash
lattice init            # creates lattice.yaml + .env.example
# edit .env with your API keys
# edit lattice.yaml to define your team
lattice up              # start the session
```

The generated `lattice.yaml` is a working config with two example agents. Edit it to match your use case, then run `lattice up` to start an interactive REPL where you can talk to your agent team.

## Example Configs

**Minimal — two LLM agents in a hub (default):**

```yaml
version: "1"
team: my-team
agents:
  researcher:
    model: anthropic/claude-sonnet-4-5-20250929
    role: You research topics and report findings.
    tools: [web-search]
  writer:
    model: openai/gpt-4o
    role: You write clear, concise content based on research.
```

The default topology is `hub` — the first agent (`researcher`) becomes coordinator and all others become workers automatically. No topology block needed.

**Pipeline — sequential handoff:**

```yaml
version: "1"
team: content-pipeline
agents:
  drafter:
    model: anthropic/claude-sonnet-4-5-20250929
    role: Write a first draft based on the user's request.
  editor:
    model: anthropic/claude-sonnet-4-5-20250929
    role: Improve the draft for clarity and correctness.
  formatter:
    type: script
    command: python format_output.py
topology:
  type: pipeline
  flow: [drafter, editor, formatter]
```

**Hub — explicit coordinator with workers:**

```yaml
version: "1"
team: dev-team
entry: lead
agents:
  lead:
    model: anthropic/claude-haiku-4-5
    role: Coordinate the team. Read specs, delegate tasks, review results.
    tools: [file-read]
  dev:
    type: cli
    cli: claude
    role: Senior Python developer. Implement what's asked.
  reviewer:
    type: cli
    cli: claude
    role: Code reviewer. Flag real issues, skip nitpicks.
  tester:
    type: script
    command: python -m pytest tests/ -v --tb=short
topology:
  type: hub
  coordinator: lead
  workers: [dev, reviewer, tester]
communication:
  heartbeat: 30
```

Since `lead` is already the entry agent, the `coordinator` and `workers` fields could be omitted here — they'd be auto-inferred. Explicit is useful when you want a different agent as coordinator or want to exclude agents from the worker list.

## REPL

After `lattice up`, you get an interactive prompt:

```
> tell the team to build a login page        # routes to entry agent
> @writer rewrite the intro paragraph         # routes directly to writer
> first line of a \                           # multi-line with backslash
... multi-line message                        # continuation
> /status                                     # request progress update
> /agents                                     # list running agents
> /done                                       # graceful shutdown
```

## CLI Reference

### `lattice init`

Scaffold a starter `lattice.yaml` and `.env.example` in the current directory.

| Flag | Description |
|------|-------------|
| `--force` | Overwrite existing files |

### `lattice up`

Start the agent team and enter the interactive REPL.

| Flag | Description |
|------|-------------|
| `-f, --file PATH` | Config file path (default: `lattice.yaml` in cwd) |
| `--watch` | Run with the live TUI instead of the plain REPL |
| `--loop [N]` | Re-run the prompt in a loop, optionally capped at N iterations |
| `-v, --verbose` | Write full tool results to a verbose sidecar file |

### `lattice down`

Signal a running session to shut down gracefully. Reads the PID from `.lattice/session.pid` and sends SIGTERM.

### `lattice watch`

Open a live TUI that tails the active session.

| Flag | Description |
|------|-------------|
| `--session PATH` | Watch a specific session file (default: latest in `sessions/`) |
| `-v, --verbose` | Show tool call events in the event feed |

Hotkeys: `t` toggle tool calls, `q` quit.

### `lattice replay [SESSION_ID]`

Step through a recorded session event by event. Without a session ID, lists all available sessions.

| Flag | Description |
|------|-------------|
| `-d, --sessions-dir PATH` | Session directory (default: `./sessions/`) |
| `-v, --verbose` | Load full tool results from the verbose sidecar |

Hotkeys: `j`/`k` or arrows to navigate, `g` jump to seq, `/` search, `a` filter by agent, `t` filter by type, `c` clear filters, `q` quit.

## Config Reference

### Top-level fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `version` | string | *required* | Schema version (`"1"`) |
| `team` | string | *required* | Team name (used in session filenames) |
| `description` | string | — | Human-readable description |
| `entry` | string | first agent | Which agent receives user messages |
| `agents` | dict | *required* | Agent definitions (at least one) |
| `topology` | object | `{type: hub}` | Communication topology |
| `communication` | object | see below | Protocol and recording settings |
| `allowed_paths` | list | `[]` | Extra directories agents can access |

### Agent fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | `llm` | Agent type: `llm`, `cli`, or `script` |
| `model` | string | — | LLM model (`provider/model-name`) |
| `role` | string | — | System prompt (inline text or file path) |
| `tools` | list | `[]` | Available tools (`web-search`, `file-read`, `file-write`, `code-exec`) |
| `cli` | string | — | CLI tool name for cli agents (e.g. `claude`) |
| `command` | string | — | Shell command for script agents or custom CLI |

Type requirements: `llm` needs `model` + `role`. `cli` needs `role` + (`cli` or `command`). `script` needs `command`.

### Topology

| Type | Description | Extra fields |
|------|-------------|-------------|
| `hub` | Workers talk only to coordinator (default). Auto-infers entry as coordinator if omitted. | `coordinator`, `workers` |
| `mesh` | Any agent can message any other | — |
| `pipeline` | Sequential chain | `flow: [a, b, c]` |
| `custom` | Explicit directed edges | `edges: {a: [b, c]}` |

### Communication

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `protocol` | string | `a2a` | Communication protocol |
| `record` | bool | `true` | Record session to JSONL |
| `heartbeat` | int | `20` | Seconds between progress checks (0 to disable) |

## Session Recording

Every session is recorded to `sessions/{date}_{team}_{id}.jsonl`. Each line is a typed event with a monotonic sequence number and ISO 8601 timestamp:

| Event type | Description |
|------------|-------------|
| `session_start` | Session identity, team name, config hash |
| `session_end` | Reason, duration, aggregate token usage |
| `message` | Inter-agent message (from, to, content) |
| `llm_call_start` | LLM call begins (model, message count) |
| `llm_call_end` | LLM call completes (tokens, duration) |
| `tool_call` / `tool_result` | Tool invocation and return |
| `error` | Error with retry flag and context |
| `agent_start` / `agent_done` | Agent lifecycle |
| `status` | Free-form agent status |
| `loop_boundary` | Loop iteration marker |
| `cli_text_chunk` | Streaming text from a CLI agent |
| `cli_tool_call` | Tool call from a CLI agent |
| `cli_thinking` | Thinking from a CLI agent |
| `cli_progress` | Progress status from a CLI agent |

Pass `-v` to `lattice up` to write full tool results to a verbose sidecar (`*.verbose.jsonl`), matched by sequence number.

## Development

```bash
uv sync
uv run lattice --help
uv run pytest tests -v
uv run ruff check .
uv run mypy src
```
