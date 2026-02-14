"""Tests for LLM provider implementations."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lattice.agent.providers import (
    AnthropicProvider,
    GoogleProvider,
    OpenAIProvider,
    create_provider,
)

# ------------------------------------------------------------------ #
# Anthropic provider
# ------------------------------------------------------------------ #


class TestAnthropicProvider:
    """Tests for AnthropicProvider message translation."""

    @pytest.fixture
    def provider(self) -> AnthropicProvider:
        init_path = "lattice.agent.providers.AnthropicProvider.__init__"
        with patch(init_path, return_value=None):
            p = AnthropicProvider.__new__(AnthropicProvider)
            p._client = AsyncMock()
            return p

    async def test_text_response(self, provider: AnthropicProvider) -> None:
        """Text response is extracted correctly."""
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Hello world"

        usage = MagicMock()
        usage.input_tokens = 10
        usage.output_tokens = 5

        response = MagicMock()
        response.content = [text_block]
        response.usage = usage

        provider._client.messages.create = AsyncMock(return_value=response)

        result = await provider.chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            model="claude-sonnet-4-5-20250929",
        )

        assert result.content == "Hello world"
        assert result.tool_calls == []
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5

    async def test_tool_use_response(self, provider: AnthropicProvider) -> None:
        """Tool use blocks are extracted as ToolCall objects."""
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "call_123"
        tool_block.name = "send_message"
        tool_block.input = {"to": "bob", "content": "hey"}

        usage = MagicMock()
        usage.input_tokens = 20
        usage.output_tokens = 15

        response = MagicMock()
        response.content = [tool_block]
        response.usage = usage

        provider._client.messages.create = AsyncMock(return_value=response)

        result = await provider.chat(
            messages=[{"role": "user", "content": "talk to bob"}],
            tools=[
                {
                    "name": "send_message",
                    "description": "Send a message",
                    "input_schema": {},
                }
            ],
            model="claude-sonnet-4-5-20250929",
        )

        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].id == "call_123"
        assert result.tool_calls[0].name == "send_message"
        assert result.tool_calls[0].arguments == {"to": "bob", "content": "hey"}

    async def test_system_prompt_separated(self, provider: AnthropicProvider) -> None:
        """System messages are separated from user messages for Anthropic."""
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "ok"

        usage = MagicMock()
        usage.input_tokens = 5
        usage.output_tokens = 2

        response = MagicMock()
        response.content = [text_block]
        response.usage = usage

        provider._client.messages.create = AsyncMock(return_value=response)

        await provider.chat(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "hi"},
            ],
            tools=[],
            model="claude-sonnet-4-5-20250929",
        )

        call_kwargs = provider._client.messages.create.call_args
        assert call_kwargs.kwargs["system"] == "You are helpful"
        # System message should NOT be in the messages list.
        api_messages = call_kwargs.kwargs["messages"]
        assert all(m["role"] != "system" for m in api_messages)


# ------------------------------------------------------------------ #
# OpenAI provider
# ------------------------------------------------------------------ #


class TestOpenAIProvider:
    """Tests for OpenAIProvider message translation."""

    @pytest.fixture
    def provider(self) -> OpenAIProvider:
        init_path = "lattice.agent.providers.OpenAIProvider.__init__"
        with patch(init_path, return_value=None):
            p = OpenAIProvider.__new__(OpenAIProvider)
            p._client = AsyncMock()
            return p

    async def test_text_response(self, provider: OpenAIProvider) -> None:
        """Text content is extracted from the first choice."""
        msg = MagicMock()
        msg.content = "Hello from GPT"
        msg.tool_calls = None

        choice = MagicMock()
        choice.message = msg

        usage = MagicMock()
        usage.prompt_tokens = 12
        usage.completion_tokens = 8

        response = MagicMock()
        response.choices = [choice]
        response.usage = usage

        provider._client.chat.completions.create = AsyncMock(return_value=response)

        result = await provider.chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            model="gpt-4o",
        )

        assert result.content == "Hello from GPT"
        assert result.tool_calls == []
        assert result.usage.input_tokens == 12
        assert result.usage.output_tokens == 8

    async def test_tool_call_response(self, provider: OpenAIProvider) -> None:
        """OpenAI tool calls are converted to our ToolCall format."""
        tc = MagicMock()
        tc.id = "call_abc"
        tc.function.name = "send_message"
        tc.function.arguments = '{"to": "alice", "content": "hi"}'

        msg = MagicMock()
        msg.content = None
        msg.tool_calls = [tc]

        choice = MagicMock()
        choice.message = msg

        usage = MagicMock()
        usage.prompt_tokens = 15
        usage.completion_tokens = 10

        response = MagicMock()
        response.choices = [choice]
        response.usage = usage

        provider._client.chat.completions.create = AsyncMock(return_value=response)

        result = await provider.chat(
            messages=[{"role": "user", "content": "talk to alice"}],
            tools=[{"name": "send_message", "description": "Send", "input_schema": {}}],
            model="gpt-4o",
        )

        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "send_message"
        assert result.tool_calls[0].arguments == {"to": "alice", "content": "hi"}

    async def test_tools_formatted_as_functions(self, provider: OpenAIProvider) -> None:
        """Tools are wrapped in OpenAI's function calling format."""
        msg = MagicMock()
        msg.content = "ok"
        msg.tool_calls = None

        choice = MagicMock()
        choice.message = msg

        response = MagicMock()
        response.choices = [choice]
        response.usage = MagicMock(prompt_tokens=5, completion_tokens=2)

        provider._client.chat.completions.create = AsyncMock(return_value=response)

        tool = {
            "name": "send_message",
            "description": "Send a message",
            "input_schema": {"type": "object", "properties": {}},
        }

        await provider.chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=[tool],
            model="gpt-4o",
        )

        call_kwargs = provider._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["tools"][0]["type"] == "function"
        assert call_kwargs["tools"][0]["function"]["name"] == "send_message"


# ------------------------------------------------------------------ #
# Google provider
# ------------------------------------------------------------------ #


class TestGoogleProvider:
    """Tests for GoogleProvider message translation."""

    @pytest.fixture
    def provider(self) -> GoogleProvider:
        init_path = "lattice.agent.providers.GoogleProvider.__init__"
        with patch(init_path, return_value=None):
            p = GoogleProvider.__new__(GoogleProvider)
            p._client = MagicMock()
            return p

    async def test_text_response(self, provider: GoogleProvider) -> None:
        """Text parts are extracted from candidates."""
        part = MagicMock()
        part.text = "Hello from Gemini"
        part.function_call = None

        candidate = MagicMock()
        candidate.content.parts = [part]

        usage_meta = MagicMock()
        usage_meta.prompt_token_count = 8
        usage_meta.candidates_token_count = 6

        response = MagicMock()
        response.candidates = [candidate]
        response.usage_metadata = usage_meta

        provider._client.aio.models.generate_content = AsyncMock(return_value=response)

        result = await provider.chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=[],
            model="gemini-2.0-flash",
        )

        assert result.content == "Hello from Gemini"
        assert result.tool_calls == []
        assert result.usage.input_tokens == 8
        assert result.usage.output_tokens == 6

    async def test_function_call_response(self, provider: GoogleProvider) -> None:
        """Function calls from Gemini are converted to ToolCall."""
        fc = MagicMock()
        fc.id = None
        fc.name = "send_message"
        fc.args = {"to": "carol", "content": "hello"}

        part = MagicMock()
        part.text = None
        part.function_call = fc

        candidate = MagicMock()
        candidate.content.parts = [part]

        usage_meta = MagicMock()
        usage_meta.prompt_token_count = 10
        usage_meta.candidates_token_count = 5

        response = MagicMock()
        response.candidates = [candidate]
        response.usage_metadata = usage_meta

        provider._client.aio.models.generate_content = AsyncMock(return_value=response)

        result = await provider.chat(
            messages=[{"role": "user", "content": "talk to carol"}],
            tools=[{"name": "send_message", "description": "Send", "input_schema": {}}],
            model="gemini-2.0-flash",
        )

        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "send_message"
        assert result.tool_calls[0].arguments == {"to": "carol", "content": "hello"}
        # When id is None, should generate a fallback id.
        assert result.tool_calls[0].id == "call_send_message"


# ------------------------------------------------------------------ #
# Provider factory
# ------------------------------------------------------------------ #


class TestCreateProvider:
    """Tests for the create_provider factory function."""

    def test_anthropic(self) -> None:
        provider, model = create_provider("anthropic/claude-sonnet-4-5-20250929")
        assert isinstance(provider, AnthropicProvider)
        assert model == "claude-sonnet-4-5-20250929"

    def test_openai(self) -> None:
        provider, model = create_provider("openai/gpt-4o")
        assert isinstance(provider, OpenAIProvider)
        assert model == "gpt-4o"

    def test_google(self) -> None:
        provider, model = create_provider("google/gemini-2.0-flash")
        assert isinstance(provider, GoogleProvider)
        assert model == "gemini-2.0-flash"

    def test_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider("cohere/command")

    def test_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="Invalid model string"):
            create_provider("no-slash-here")
