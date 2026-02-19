"""Tests for built-in tools and the updated ToolRegistry."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.agent.builtin_tools import (
    BUILTIN_TOOL_SCHEMAS,
    CODE_EXEC_TOOL,
    FILE_READ_TOOL,
    FILE_WRITE_TOOL,
    WEB_SEARCH_TOOL,
    _validate_path,
    handle_code_exec,
    handle_file_read,
    handle_file_write,
    handle_web_search,
)
from lattice.agent.tools import ToolRegistry, _is_mcp_reference
from lattice.config.models import TopologyConfig
from lattice.router.router import Router
from lattice.session.models import ToolCallEvent, ToolResultEvent
from lattice.session.recorder import SessionRecorder

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def recorder(tmp_path: Path) -> SessionRecorder:
    return SessionRecorder(
        team="test-team",
        config_hash="abc123",
        sessions_dir=tmp_path / "sessions",
    )


@pytest.fixture
def router(recorder: SessionRecorder) -> Router:
    return Router(topology=TopologyConfig(type="mesh"), recorder=recorder)


@pytest.fixture
def working_dir(tmp_path: Path) -> Path:
    d = tmp_path / "workdir"
    d.mkdir()
    return d


# ------------------------------------------------------------------ #
# Tool schema tests
# ------------------------------------------------------------------ #


class TestToolSchemas:
    """Each built-in tool has a well-formed JSON schema."""

    @pytest.mark.parametrize(
        "schema",
        [WEB_SEARCH_TOOL, FILE_READ_TOOL, FILE_WRITE_TOOL, CODE_EXEC_TOOL],
        ids=["web-search", "file-read", "file-write", "code-exec"],
    )
    def test_schema_has_required_keys(self, schema: dict[str, Any]) -> None:
        assert "name" in schema
        assert "description" in schema
        assert "input_schema" in schema
        assert schema["input_schema"]["type"] == "object"
        assert "properties" in schema["input_schema"]
        assert "required" in schema["input_schema"]

    def test_builtin_schemas_registry(self) -> None:
        assert set(BUILTIN_TOOL_SCHEMAS.keys()) == {
            "web-search",
            "file-read",
            "file-write",
            "list-dir",
            "code-exec",
        }


# ------------------------------------------------------------------ #
# Path validation tests
# ------------------------------------------------------------------ #


class TestPathValidation:
    """Path validation rejects traversal attempts."""

    def test_valid_relative_path(self, working_dir: Path) -> None:
        (working_dir / "hello.txt").write_text("hi")
        resolved = _validate_path("hello.txt", working_dir)
        assert resolved == (working_dir / "hello.txt").resolve()

    def test_valid_nested_path(self, working_dir: Path) -> None:
        sub = working_dir / "sub"
        sub.mkdir()
        resolved = _validate_path("sub/file.txt", working_dir)
        assert resolved == (sub / "file.txt").resolve()

    def test_rejects_parent_traversal(self, working_dir: Path) -> None:
        with pytest.raises(ValueError, match="escapes the working directory"):
            _validate_path("../outside.txt", working_dir)

    def test_rejects_absolute_path_outside(self, working_dir: Path) -> None:
        with pytest.raises(ValueError, match="escapes the working directory"):
            _validate_path("/etc/passwd", working_dir)

    def test_rejects_sneaky_traversal(self, working_dir: Path) -> None:
        with pytest.raises(ValueError, match="escapes the working directory"):
            _validate_path("sub/../../outside.txt", working_dir)

    def test_allows_current_dir(self, working_dir: Path) -> None:
        resolved = _validate_path("./file.txt", working_dir)
        assert resolved == (working_dir / "file.txt").resolve()


# ------------------------------------------------------------------ #
# file-read tests
# ------------------------------------------------------------------ #


class TestFileRead:
    """Tests for the file-read tool handler."""

    async def test_read_existing_file(self, working_dir: Path) -> None:
        (working_dir / "test.txt").write_text("hello world")
        result = await handle_file_read({"path": "test.txt"}, working_dir)
        data = json.loads(result)
        assert data["path"] == "test.txt"
        assert data["content"] == "hello world"

    async def test_read_nested_file(self, working_dir: Path) -> None:
        sub = working_dir / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("nested content")
        result = await handle_file_read({"path": "subdir/nested.txt"}, working_dir)
        data = json.loads(result)
        assert data["content"] == "nested content"

    async def test_read_nonexistent_file(self, working_dir: Path) -> None:
        with pytest.raises(FileNotFoundError, match="File not found"):
            await handle_file_read({"path": "nope.txt"}, working_dir)

    async def test_read_directory_raises(self, working_dir: Path) -> None:
        (working_dir / "adir").mkdir()
        with pytest.raises(ValueError, match="Not a file"):
            await handle_file_read({"path": "adir"}, working_dir)

    async def test_read_path_traversal_rejected(self, working_dir: Path) -> None:
        with pytest.raises(ValueError, match="escapes"):
            await handle_file_read({"path": "../../etc/passwd"}, working_dir)


# ------------------------------------------------------------------ #
# file-write tests
# ------------------------------------------------------------------ #


class TestFileWrite:
    """Tests for the file-write tool handler."""

    async def test_write_new_file(self, working_dir: Path) -> None:
        result = await handle_file_write(
            {"path": "new.txt", "content": "hello"}, working_dir
        )
        data = json.loads(result)
        assert data["path"] == "new.txt"
        assert data["bytes_written"] == 5
        assert (working_dir / "new.txt").read_text() == "hello"

    async def test_write_creates_parent_dirs(self, working_dir: Path) -> None:
        result = await handle_file_write(
            {"path": "a/b/c.txt", "content": "deep"}, working_dir
        )
        data = json.loads(result)
        assert data["bytes_written"] == 4
        assert (working_dir / "a" / "b" / "c.txt").read_text() == "deep"

    async def test_write_overwrites_existing(self, working_dir: Path) -> None:
        (working_dir / "exist.txt").write_text("old")
        await handle_file_write({"path": "exist.txt", "content": "new"}, working_dir)
        assert (working_dir / "exist.txt").read_text() == "new"

    async def test_write_path_traversal_rejected(self, working_dir: Path) -> None:
        with pytest.raises(ValueError, match="escapes"):
            await handle_file_write(
                {"path": "../outside.txt", "content": "nope"}, working_dir
            )

    async def test_write_unicode_content(self, working_dir: Path) -> None:
        content = "Hello, world! \u2603 \u2764"
        result = await handle_file_write(
            {"path": "unicode.txt", "content": content}, working_dir
        )
        data = json.loads(result)
        assert data["bytes_written"] == len(content.encode("utf-8"))
        assert (working_dir / "unicode.txt").read_text() == content


# ------------------------------------------------------------------ #
# code-exec tests
# ------------------------------------------------------------------ #


class TestCodeExec:
    """Tests for the code-exec tool handler."""

    async def test_simple_print(self) -> None:
        result = await handle_code_exec({"code": "print('hello')"})
        data = json.loads(result)
        assert data["exit_code"] == 0
        assert data["stdout"].strip() == "hello"
        assert data["timed_out"] is False

    async def test_stderr_captured(self) -> None:
        result = await handle_code_exec(
            {"code": "import sys; sys.stderr.write('err\\n')"}
        )
        data = json.loads(result)
        assert data["exit_code"] == 0
        assert "err" in data["stderr"]

    async def test_nonzero_exit_code(self) -> None:
        result = await handle_code_exec({"code": "raise SystemExit(42)"})
        data = json.loads(result)
        assert data["exit_code"] == 42

    async def test_syntax_error(self) -> None:
        result = await handle_code_exec({"code": "def bad syntax"})
        data = json.loads(result)
        assert data["exit_code"] != 0
        assert "SyntaxError" in data["stderr"]

    async def test_timeout(self) -> None:
        result = await handle_code_exec(
            {"code": "import time; time.sleep(60)", "timeout": 1}
        )
        data = json.loads(result)
        assert data["timed_out"] is True
        assert data["exit_code"] == -1

    async def test_default_timeout_respected(self) -> None:
        """Fast code completes well within default timeout."""
        result = await handle_code_exec({"code": "print(2+2)"})
        data = json.loads(result)
        assert data["exit_code"] == 0
        assert data["stdout"].strip() == "4"
        assert data["timed_out"] is False


# ------------------------------------------------------------------ #
# web-search tests
# ------------------------------------------------------------------ #


class TestWebSearch:
    """Tests for the web-search tool handler."""

    async def test_returns_json_on_network_error(self) -> None:
        """When the HTTP request fails, we get a JSON error, not an exception."""
        with patch("lattice.agent.builtin_tools.asyncio") as mock_asyncio:
            loop = MagicMock()
            mock_asyncio.get_running_loop.return_value = loop
            loop.run_in_executor = AsyncMock(
                side_effect=Exception("Connection refused")
            )
            result = await handle_web_search({"query": "test"})
        data = json.loads(result)
        assert "error" in data
        assert data["results"] == []

    async def test_returns_parsed_results(self) -> None:
        """When HTML is returned, we parse out search results."""
        fake_html = (
            '<div class="result">'
            '<a href="http://example.com" class="result__a">Example Title</a>'
            '<span class="result__snippet">A snippet here</span>'
            "</div>"
        )
        with patch("lattice.agent.builtin_tools.asyncio") as mock_asyncio:
            loop = MagicMock()
            mock_asyncio.get_running_loop.return_value = loop
            loop.run_in_executor = AsyncMock(return_value=fake_html)
            result = await handle_web_search({"query": "test"})
        data = json.loads(result)
        assert data["query"] == "test"
        assert isinstance(data["results"], list)
        if data["results"]:
            assert "title" in data["results"][0]
            assert "url" in data["results"][0]

    async def test_max_results_parameter(self) -> None:
        """max_results limits the number of returned results."""
        # Build HTML with multiple results
        entries = []
        for i in range(10):
            entries.append(
                f'<a href="http://example.com/{i}" class="result__a">Title {i}</a>'
                f'<span class="result__snippet">Snippet {i}</span>'
            )
        fake_html = "".join(entries)

        with patch("lattice.agent.builtin_tools.asyncio") as mock_asyncio:
            loop = MagicMock()
            mock_asyncio.get_running_loop.return_value = loop
            loop.run_in_executor = AsyncMock(return_value=fake_html)
            result = await handle_web_search({"query": "test", "max_results": 3})
        data = json.loads(result)
        assert len(data["results"]) <= 3


# ------------------------------------------------------------------ #
# MCP reference detection
# ------------------------------------------------------------------ #


class TestMCPReferences:
    """MCP tool references are detected and deferred."""

    def test_string_mcp_reference(self) -> None:
        assert _is_mcp_reference("mcp: ./my-server") is True
        assert _is_mcp_reference("mcp:https://server.com") is True

    def test_dict_mcp_reference(self) -> None:
        assert _is_mcp_reference({"mcp": "./path/to/server"}) is True

    def test_non_mcp_string(self) -> None:
        assert _is_mcp_reference("web-search") is False
        assert _is_mcp_reference("file-read") is False

    def test_non_mcp_dict(self) -> None:
        assert _is_mcp_reference({"name": "custom"}) is False


# ------------------------------------------------------------------ #
# ToolRegistry tests
# ------------------------------------------------------------------ #


class TestToolRegistryConfiguration:
    """ToolRegistry respects configured tool lists."""

    def test_send_message_always_present(
        self, router: Router, recorder: SessionRecorder
    ) -> None:
        reg = ToolRegistry("agent", router, recorder)
        names = [t["name"] for t in reg.definitions]
        assert "send_message" in names

    def test_no_builtin_tools_by_default(
        self, router: Router, recorder: SessionRecorder
    ) -> None:
        reg = ToolRegistry("agent", router, recorder)
        names = [t["name"] for t in reg.definitions]
        assert names == ["send_message"]

    def test_configured_tools_registered(
        self, router: Router, recorder: SessionRecorder
    ) -> None:
        reg = ToolRegistry(
            "agent",
            router,
            recorder,
            configured_tools=["web-search", "file-read"],
        )
        names = [t["name"] for t in reg.definitions]
        assert "send_message" in names
        assert "web-search" in names
        assert "file-read" in names
        assert "file-write" not in names
        assert "code-exec" not in names

    def test_all_builtin_tools(self, router: Router, recorder: SessionRecorder) -> None:
        reg = ToolRegistry(
            "agent",
            router,
            recorder,
            configured_tools=["web-search", "file-read", "file-write", "code-exec"],
        )
        names = [t["name"] for t in reg.definitions]
        assert set(names) == {
            "send_message",
            "web-search",
            "file-read",
            "file-write",
            "code-exec",
        }

    def test_unknown_tool_skipped_with_warning(
        self, router: Router, recorder: SessionRecorder, caplog: Any
    ) -> None:
        with caplog.at_level(logging.WARNING):
            reg = ToolRegistry(
                "agent",
                router,
                recorder,
                configured_tools=["nonexistent-tool"],
            )
        names = [t["name"] for t in reg.definitions]
        assert "nonexistent-tool" not in names
        assert "Unknown tool" in caplog.text

    def test_mcp_reference_deferred_with_warning(
        self, router: Router, recorder: SessionRecorder, caplog: Any
    ) -> None:
        with caplog.at_level(logging.WARNING):
            reg = ToolRegistry(
                "agent",
                router,
                recorder,
                configured_tools=["mcp: ./server", "file-read"],
            )
        names = [t["name"] for t in reg.definitions]
        assert "file-read" in names
        assert "deferred" in caplog.text

    def test_mcp_dict_reference_deferred(
        self, router: Router, recorder: SessionRecorder, caplog: Any
    ) -> None:
        with caplog.at_level(logging.WARNING):
            ToolRegistry(
                "agent",
                router,
                recorder,
                configured_tools=[{"mcp": "https://server.com"}],
            )
        assert "deferred" in caplog.text


# ------------------------------------------------------------------ #
# ToolRegistry dispatch and event recording
# ------------------------------------------------------------------ #


class TestToolRegistryDispatch:
    """ToolRegistry dispatches to the correct handlers."""

    async def test_dispatch_file_read(
        self, router: Router, recorder: SessionRecorder, working_dir: Path
    ) -> None:
        (working_dir / "test.txt").write_text("contents")
        reg = ToolRegistry(
            "agent",
            router,
            recorder,
            configured_tools=["file-read"],
            working_dir=working_dir,
        )
        result = await reg.execute("file-read", {"path": "test.txt"})
        data = json.loads(result)
        assert data["content"] == "contents"

    async def test_dispatch_file_write(
        self, router: Router, recorder: SessionRecorder, working_dir: Path
    ) -> None:
        reg = ToolRegistry(
            "agent",
            router,
            recorder,
            configured_tools=["file-write"],
            working_dir=working_dir,
        )
        result = await reg.execute("file-write", {"path": "out.txt", "content": "hi"})
        data = json.loads(result)
        assert data["bytes_written"] == 2
        assert (working_dir / "out.txt").read_text() == "hi"

    async def test_dispatch_code_exec(
        self, router: Router, recorder: SessionRecorder
    ) -> None:
        reg = ToolRegistry(
            "agent",
            router,
            recorder,
            configured_tools=["code-exec"],
        )
        result = await reg.execute("code-exec", {"code": "print(42)"})
        data = json.loads(result)
        assert data["exit_code"] == 0
        assert "42" in data["stdout"]

    async def test_dispatch_unknown_tool_returns_error(
        self, router: Router, recorder: SessionRecorder
    ) -> None:
        reg = ToolRegistry("agent", router, recorder)
        result = await reg.execute("nonexistent", {"foo": "bar"})
        assert "Error" in result
        assert "Unknown tool" in result

    async def test_event_recording_on_tool_call(
        self, router: Router, recorder: SessionRecorder, working_dir: Path
    ) -> None:
        """tool_call and tool_result events are recorded."""
        (working_dir / "data.txt").write_text("data")
        reg = ToolRegistry(
            "agent",
            router,
            recorder,
            configured_tools=["file-read"],
            working_dir=working_dir,
        )

        events: list[Any] = []
        original_record = recorder.record

        def capture(event: Any) -> None:
            events.append(event)
            original_record(event)

        recorder.record = capture  # type: ignore[assignment]

        await reg.execute("file-read", {"path": "data.txt"})

        event_types = [type(e) for e in events]
        assert ToolCallEvent in event_types
        assert ToolResultEvent in event_types

        tool_call = next(e for e in events if isinstance(e, ToolCallEvent))
        assert tool_call.tool == "file-read"
        assert tool_call.agent == "agent"

        tool_result = next(e for e in events if isinstance(e, ToolResultEvent))
        assert tool_result.tool == "file-read"
        assert tool_result.result_size > 0


# ------------------------------------------------------------------ #
# Backward compatibility
# ------------------------------------------------------------------ #


class TestToolRegistryBackwardCompat:
    """Existing code that creates ToolRegistry without configured_tools still works."""

    def test_old_constructor_signature(
        self, router: Router, recorder: SessionRecorder
    ) -> None:
        """ToolRegistry(name, router, recorder) still works."""
        reg = ToolRegistry("agent", router, recorder)
        names = [t["name"] for t in reg.definitions]
        assert names == ["send_message"]

    async def test_send_message_still_dispatches(
        self, router: Router, recorder: SessionRecorder
    ) -> None:
        agent_b = MagicMock()
        agent_b.handle_message = AsyncMock()
        router.register("agent-b", agent_b)

        reg = ToolRegistry("agent-a", router, recorder)
        result = await reg.execute(
            "send_message",
            {"to": "agent-b", "content": "hi"},
        )
        data = json.loads(result)
        assert data["status"] == "sent"
        agent_b.handle_message.assert_called_once_with("agent-a", "hi")
