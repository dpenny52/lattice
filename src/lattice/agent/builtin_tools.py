"""Built-in tools that agents can use out of the box."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import tempfile
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Tool schemas (matching SEND_MESSAGE_TOOL format in tools.py)
# ------------------------------------------------------------------ #

WEB_SEARCH_TOOL: dict[str, Any] = {
    "name": "web-search",
    "description": "Search the web and return results.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default 5)",
            },
        },
        "required": ["query"],
    },
}

FILE_READ_TOOL: dict[str, Any] = {
    "name": "file-read",
    "description": "Read the contents of a local file.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to read (relative to cwd)",
            },
        },
        "required": ["path"],
    },
}

FILE_WRITE_TOOL: dict[str, Any] = {
    "name": "file-write",
    "description": "Write content to a local file. Creates dirs if needed.",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to write (relative to cwd)",
            },
            "content": {
                "type": "string",
                "description": "Content to write to the file",
            },
        },
        "required": ["path", "content"],
    },
}

CODE_EXEC_TOOL: dict[str, Any] = {
    "name": "code-exec",
    "description": "Execute a Python code snippet in a subprocess.",
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum execution time in seconds (default 30)",
            },
        },
        "required": ["code"],
    },
}

#: Registry of all built-in tool schemas, keyed by name.
BUILTIN_TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "web-search": WEB_SEARCH_TOOL,
    "file-read": FILE_READ_TOOL,
    "file-write": FILE_WRITE_TOOL,
    "code-exec": CODE_EXEC_TOOL,
}

# ------------------------------------------------------------------ #
# Path validation
# ------------------------------------------------------------------ #

_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 300
_DEFAULT_MAX_RESULTS = 5
_MAX_RESULTS_CAP = 20
_MAX_READ_SIZE = 10 * 1024 * 1024  # 10 MB


def _validate_path(
    path_str: str,
    working_dir: Path,
    allowed_paths: list[Path] | None = None,
) -> Path:
    """Resolve *path_str* against *working_dir* and reject path traversal.

    Paths under *working_dir* or any entry in *allowed_paths* are permitted.
    Raises ``ValueError`` if the resolved path escapes all allowed directories.
    """
    expanded = Path(path_str).expanduser()
    if expanded.is_absolute():
        resolved = expanded.resolve()
    else:
        resolved = (working_dir / expanded).resolve()

    # Check working directory.
    working_resolved = working_dir.resolve()
    wd_prefix = str(working_resolved) + "/"
    if resolved == working_resolved or str(resolved).startswith(wd_prefix):
        return resolved

    # Check allowed paths.
    for allowed in allowed_paths or []:
        allowed_resolved = allowed.resolve()
        ap_prefix = str(allowed_resolved) + "/"
        if resolved == allowed_resolved or str(resolved).startswith(ap_prefix):
            return resolved

    msg = f"Path '{path_str}' escapes the working directory"
    raise ValueError(msg)


# ------------------------------------------------------------------ #
# Tool implementations
# ------------------------------------------------------------------ #


async def handle_web_search(arguments: dict[str, Any]) -> str:
    """Search the web via DuckDuckGo HTML and return results as JSON."""
    query = arguments.get("query", "")
    max_results = min(
        arguments.get("max_results", _DEFAULT_MAX_RESULTS),
        _MAX_RESULTS_CAP,
    )

    encoded = urllib.parse.urlencode({"q": query})
    url = f"https://html.duckduckgo.com/html/?{encoded}"

    def _fetch() -> str:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (compatible; Lattice/0.1; "
                    "+https://github.com/lattice-agent)"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw: bytes = resp.read()
            return raw.decode("utf-8", errors="replace")

    loop = asyncio.get_running_loop()
    try:
        html = await loop.run_in_executor(None, _fetch)
    except Exception as exc:
        return json.dumps({"error": str(exc), "results": []})

    # Parse result snippets from the DuckDuckGo HTML response.
    results = _parse_ddg_html(html, max_results)
    return json.dumps({"query": query, "results": results})


class _DDGResultParser(HTMLParser):
    """Extract search results from DuckDuckGo HTML using stdlib HTMLParser.

    Looks for ``<a class="result__a" href="...">Title</a>`` and
    ``<a class="result__snippet" ...>Snippet</a>`` elements.
    """

    def __init__(self, max_results: int) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._max_results = max_results
        self._in_result_link = False
        self._in_snippet = False
        self._current: dict[str, str] = {}
        self._text_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if len(self.results) >= self._max_results:
            return
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class", "") or ""

        if tag == "a" and "result__a" in cls:
            self._in_result_link = True
            self._current = {
                "url": attrs_dict.get("href", "") or "",
                "title": "",
                "snippet": "",
            }
            self._text_buf = []
        elif "result__snippet" in cls:
            self._in_snippet = True
            self._text_buf = []

    def handle_endtag(self, tag: str) -> None:
        if self._in_result_link and tag == "a":
            self._in_result_link = False
            self._current["title"] = "".join(self._text_buf).strip()
        elif self._in_snippet:
            self._in_snippet = False
            self._current["snippet"] = "".join(self._text_buf).strip()
            if self._current.get("title") or self._current.get("url"):
                self.results.append(self._current)
            self._current = {}

    def handle_data(self, data: str) -> None:
        if self._in_result_link or self._in_snippet:
            self._text_buf.append(data)


def _parse_ddg_html(html: str, max_results: int) -> list[dict[str, str]]:
    """Extract search results from DuckDuckGo HTML response."""
    parser = _DDGResultParser(max_results)
    parser.feed(html)
    return parser.results


async def handle_file_read(
    arguments: dict[str, Any],
    working_dir: Path,
    allowed_paths: list[Path] | None = None,
) -> str:
    """Read a local file and return its contents."""
    path_str = arguments.get("path", "")
    resolved = _validate_path(path_str, working_dir, allowed_paths)

    if not resolved.exists():
        msg = f"File not found: {path_str}"
        raise FileNotFoundError(msg)
    if not resolved.is_file():
        msg = f"Not a file: {path_str}"
        raise ValueError(msg)

    size = resolved.stat().st_size
    if size > _MAX_READ_SIZE:
        msg = f"File too large: {size} bytes (max {_MAX_READ_SIZE})"
        raise ValueError(msg)

    content = resolved.read_text(encoding="utf-8", errors="replace")
    return json.dumps({"path": path_str, "content": content})


async def handle_file_write(
    arguments: dict[str, Any],
    working_dir: Path,
    allowed_paths: list[Path] | None = None,
) -> str:
    """Write content to a local file."""
    path_str = arguments.get("path", "")
    content = arguments.get("content", "")
    resolved = _validate_path(path_str, working_dir, allowed_paths)

    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return json.dumps({"path": path_str, "bytes_written": len(content.encode("utf-8"))})


async def handle_code_exec(arguments: dict[str, Any]) -> str:
    """Execute Python code in a sandboxed subprocess with a timeout.

    Sandboxing measures:
    - Runs in a temporary working directory (cleaned up after execution).
    - Restricted environment: only PATH, HOME (set to tmpdir), and LANG.
    - LLM API keys and other sensitive env vars are NOT inherited.
    """
    code = arguments.get("code", "")
    timeout = min(
        arguments.get("timeout", _DEFAULT_TIMEOUT),
        _MAX_TIMEOUT,
    )

    with tempfile.TemporaryDirectory(prefix="lattice_exec_") as tmpdir:
        # Minimal environment — no inherited secrets or config.
        restricted_env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": tmpdir,
            "LANG": "en_US.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            code,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=tmpdir,
            env=restricted_env,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return json.dumps(
                {
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": "Execution timed out",
                    "timed_out": True,
                }
            )

    return json.dumps(
        {
            "exit_code": proc.returncode,
            "stdout": stdout_bytes.decode("utf-8", errors="replace"),
            "stderr": stderr_bytes.decode("utf-8", errors="replace"),
            "timed_out": False,
        }
    )
