"""LLM provider abstraction and vendor-specific implementations."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from os import environ
from typing import Any, Protocol


@dataclass
class ToolCall:
    """A single tool invocation returned by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class TokenUsage:
    """Token counts for a single LLM call."""

    input_tokens: int
    output_tokens: int


@dataclass
class LLMResponse:
    """Normalised response from any LLM provider."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: TokenUsage = field(default_factory=lambda: TokenUsage(0, 0))


class LLMProvider(Protocol):
    """Protocol every provider must implement."""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
    ) -> LLMResponse: ...


# --------------------------------------------------------------------------- #
# Anthropic
# --------------------------------------------------------------------------- #


class AnthropicProvider:
    """Anthropic (Claude) provider using the async SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        from anthropic import AsyncAnthropic

        key = api_key or environ.get("ANTHROPIC_API_KEY", "")
        self._client = AsyncAnthropic(api_key=key)

    @staticmethod
    def _translate_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Translate OpenAI-style messages to Anthropic content-block format.

        Returns ``(system_text, api_messages)``.
        """
        system_text = ""
        api_messages: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")

            if role == "system":
                system_text = str(msg.get("content", ""))

            elif role == "assistant" and msg.get("tool_calls"):
                # Convert tool_calls list to Anthropic content blocks.
                content_blocks: list[dict[str, Any]] = []
                text = msg.get("content")
                if text:
                    content_blocks.append({"type": "text", "text": str(text)})
                for tc in msg["tool_calls"]:
                    arguments = tc.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = {}
                    content_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["name"],
                            "input": arguments,
                        }
                    )
                api_messages.append({"role": "assistant", "content": content_blocks})

            elif role == "tool":
                # Convert to user message with tool_result block.
                # Group with previous user message if it already has tool_result blocks.
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": msg["tool_call_id"],
                    "content": str(msg.get("content", "")),
                }
                if (
                    api_messages
                    and api_messages[-1].get("role") == "user"
                    and isinstance(api_messages[-1].get("content"), list)
                    and api_messages[-1]["content"]
                    and api_messages[-1]["content"][0].get("type") == "tool_result"
                ):
                    # Merge into the previous user message.
                    api_messages[-1]["content"].append(result_block)
                else:
                    api_messages.append(
                        {"role": "user", "content": [result_block]}
                    )

            else:
                # Plain user or assistant messages pass through.
                api_messages.append(msg)

        return system_text, api_messages

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
    ) -> LLMResponse:
        system_text, api_messages = self._translate_messages(messages)

        # Translate tool schemas to Anthropic format.
        anthropic_tools = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("input_schema", t.get("parameters", {})),
            }
            for t in tools
        ]

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "messages": api_messages,
        }
        if system_text:
            kwargs["system"] = system_text
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools

        response = await self._client.messages.create(**kwargs)

        # Extract content text and tool calls from content blocks.
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input)
                        if isinstance(block.input, dict)
                        else {},
                    )
                )

        content = "\n".join(text_parts) if text_parts else None
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        return LLMResponse(content=content, tool_calls=tool_calls, usage=usage)


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #


class OpenAIProvider:
    """OpenAI provider using the async SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        from openai import AsyncOpenAI

        key = api_key or environ.get("OPENAI_API_KEY", "")
        self._client = AsyncOpenAI(api_key=key)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
    ) -> LLMResponse:
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", t.get("parameters", {})),
                },
            }
            for t in tools
        ]

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools

        response = await self._client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        msg = choice.message

        content = msg.content
        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                        if tc.function.arguments
                        else {},
                    )
                )

        usage = TokenUsage(0, 0)
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens or 0,
            )

        return LLMResponse(content=content, tool_calls=tool_calls, usage=usage)


# --------------------------------------------------------------------------- #
# Google
# --------------------------------------------------------------------------- #


class GoogleProvider:
    """Google Gemini provider using the google-genai async SDK."""

    def __init__(self, api_key: str | None = None) -> None:
        from google import genai

        key = api_key or environ.get("GOOGLE_API_KEY") or None
        self._client = genai.Client(api_key=key) if key else None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        model: str,
    ) -> LLMResponse:
        from google.genai import types

        if self._client is None:
            msg = (
                "Google API key not set. "
                "Set GOOGLE_API_KEY or pass credentials."
            )
            raise ValueError(msg)

        # Build system instruction and contents.
        # TODO(story-2.4): translate tool_calls / tool-result messages into
        #  Google function-call / function-response parts for multi-turn
        #  tool use.
        system_text: str | None = None
        contents: list[types.Content] = []

        for m in messages:
            role = m.get("role", "user")
            text = str(m.get("content", ""))
            if role == "system":
                system_text = text
                continue
            # Google uses "user" and "model" roles.
            g_role = "model" if role == "assistant" else "user"
            contents.append(
                types.Content(
                    role=g_role,
                    parts=[types.Part.from_text(text=text)],
                )
            )

        # Build tool declarations.
        function_declarations: list[types.FunctionDeclaration] = []
        for t in tools:
            schema = t.get("input_schema", t.get("parameters", {}))
            function_declarations.append(
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=schema,
                )
            )

        config_kwargs: dict[str, Any] = {}
        if system_text:
            config_kwargs["system_instruction"] = system_text
        if function_declarations:
            config_kwargs["tools"] = [
                types.Tool(function_declarations=function_declarations)
            ]

        config = types.GenerateContentConfig(**config_kwargs)

        response = await self._client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

        # Extract text and tool calls.
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        if response.candidates:
            content_obj = response.candidates[0].content
            parts = content_obj.parts if content_obj else None
            for part in parts or []:
                if part.text:
                    text_parts.append(part.text)
                if part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        ToolCall(
                            id=fc.id if fc.id else f"call_{fc.name}",
                            name=fc.name or "",
                            arguments=dict(fc.args) if fc.args else {},
                        )
                    )

        content = "\n".join(text_parts) if text_parts else None
        usage = TokenUsage(0, 0)
        if response.usage_metadata:
            usage = TokenUsage(
                input_tokens=response.usage_metadata.prompt_token_count or 0,
                output_tokens=response.usage_metadata.candidates_token_count or 0,
            )

        return LLMResponse(content=content, tool_calls=tool_calls, usage=usage)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

_PROVIDER_MAP: dict[str, type[AnthropicProvider | OpenAIProvider | GoogleProvider]] = {
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "google": GoogleProvider,
}


def create_provider(
    model_string: str,
    credentials: dict[str, str] | None = None,
) -> tuple[LLMProvider, str]:
    """Parse ``provider/model-name`` and return ``(provider_instance, model_name)``.

    Raises ``ValueError`` for unknown providers.
    """
    if "/" not in model_string:
        msg = (
            f"Invalid model string {model_string!r} -- "
            "expected format 'provider/model-name'"
        )
        raise ValueError(msg)

    provider_name, model_name = model_string.split("/", 1)

    provider_cls = _PROVIDER_MAP.get(provider_name)
    if provider_cls is None:
        known = ", ".join(sorted(_PROVIDER_MAP))
        msg = f"Unknown provider {provider_name!r} -- supported providers: {known}"
        raise ValueError(msg)

    # Resolve API key from credentials override or environment.
    api_key: str | None = None
    if credentials:
        key_env_var = credentials.get(provider_name)
        if key_env_var:
            api_key = environ.get(key_env_var)

    return provider_cls(api_key=api_key), model_name
