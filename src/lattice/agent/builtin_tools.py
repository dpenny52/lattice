"""Built-in tools that agents can use out of the box."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import urllib.parse
import urllib.request
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


def _validate_path(path_str: str, working_dir: Path) -> Path:
    """Resolve *path_str* against *working_dir* and reject path traversal.

    Raises ``ValueError`` if the resolved path escapes the working directory.
    """
    resolved = (working_dir / path_str).resolve()
    working_resolved = working_dir.resolve()
    wd_prefix = str(working_resolved) + "/"
    if resolved != working_resolved and not str(resolved).startswith(wd_prefix):
        msg = f"Path '{path_str}' escapes the working directory"
        raise ValueError(msg)
    return resolved


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
            headers={"User-Agent": "Lattice/0.1"},
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


def _parse_ddg_html(html: str, max_results: int) -> list[dict[str, str]]:
    """Extract search results from DuckDuckGo HTML response.

    Uses simple string parsing to avoid adding an HTML parser dependency.
    """
    results: list[dict[str, str]] = []
    marker = 'class="result__a"'
    pos = 0

    while len(results) < max_results:
        idx = html.find(marker, pos)
        if idx == -1:
            break

        # Extract href
        href_start = html.rfind('href="', max(0, idx - 200), idx)
        href = ""
        if href_start != -1:
            href_start += len('href="')
            href_end = html.find('"', href_start)
            if href_end != -1:
                href = html[href_start:href_end]

        # Extract link text
        tag_end = html.find(">", idx)
        if tag_end == -1:
            pos = idx + len(marker)
            continue
        close_tag = html.find("</a>", tag_end)
        title = ""
        if close_tag != -1:
            title = html[tag_end + 1 : close_tag]
            title = re.sub(r"<[^>]+>", "", title).strip()

        # Extract snippet
        snippet_marker = 'class="result__snippet"'
        search_from = close_tag if close_tag != -1 else idx
        snippet_idx = html.find(snippet_marker, search_from)
        snippet = ""
        next_result = html.find(marker, idx + len(marker))
        in_bounds = next_result == -1 or snippet_idx < next_result
        if snippet_idx != -1 and in_bounds:
            stag_end = html.find(">", snippet_idx)
            if stag_end != -1:
                sclose = html.find("</", stag_end)
                if sclose != -1:
                    snippet = html[stag_end + 1 : sclose]
                    snippet = re.sub(r"<[^>]+>", "", snippet).strip()

        if title or href:
            results.append({"title": title, "url": href, "snippet": snippet})

        pos = idx + len(marker)

    return results


async def handle_file_read(
    arguments: dict[str, Any],
    working_dir: Path,
) -> str:
    """Read a local file and return its contents."""
    path_str = arguments.get("path", "")
    resolved = _validate_path(path_str, working_dir)

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
) -> str:
    """Write content to a local file."""
    path_str = arguments.get("path", "")
    content = arguments.get("content", "")
    resolved = _validate_path(path_str, working_dir)

    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    return json.dumps({"path": path_str, "bytes_written": len(content.encode("utf-8"))})


async def handle_code_exec(arguments: dict[str, Any]) -> str:
    """Execute Python code in a subprocess with a timeout."""
    code = arguments.get("code", "")
    timeout = min(
        arguments.get("timeout", _DEFAULT_TIMEOUT),
        _MAX_TIMEOUT,
    )

    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c", code,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except TimeoutError:
        proc.kill()
        await proc.wait()
        return json.dumps({
            "exit_code": -1,
            "stdout": "",
            "stderr": "Execution timed out",
            "timed_out": True,
        })

    return json.dumps({
        "exit_code": proc.returncode,
        "stdout": stdout_bytes.decode("utf-8", errors="replace"),
        "stderr": stderr_bytes.decode("utf-8", errors="replace"),
        "timed_out": False,
    })
