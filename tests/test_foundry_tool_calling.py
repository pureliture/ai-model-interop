"""Tests for Foundry OpenAI, Anthropic, and xAI protocol structured tool calling (Milestones 2, 3 & 4 / Slices 2, 3 & 4).

Verifies:
- OpenAI protocol: non-stream & stream tool calling, multi-turn history, edge cases
- Anthropic protocol: non-stream & stream tool_use, tool_choice mapping, multi-turn history with consecutive tool result merging, edge cases
- xAI Responses protocol: non-stream & stream function_call, tool_choice mapping, multi-turn history with function_call_output items, edge cases
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import openai_compatible_bridge.providers.vertex as vertex
from openai_compatible_bridge.main import create_app
from openai_compatible_bridge.providers.foundry import FoundryChatClient


class _DummyProvider:
    async def close(self) -> None:
        pass


WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

STOCK_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_stock_quote",
        "description": "Fetch stock quote for a ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
            },
            "required": ["ticker"],
        },
    },
}


def _register_foundry_openai_alias(
    model_alias: str = "foundry:gpt-6-astra",
    provider_model: str = "gpt-6-astra",
) -> dict[str, dict]:
    old_registry = vertex.MODEL_REGISTRY.copy()
    vertex.MODEL_REGISTRY[model_alias] = {
        "provider": "foundry",
        "kind": "chat",
        "provider_model": provider_model,
        "protocol": "openai_chat_completions",
    }
    return old_registry


def _restore_registry(old_registry: dict[str, dict]) -> None:
    vertex.MODEL_REGISTRY.clear()
    vertex.MODEL_REGISTRY.update(old_registry)


def test_foundry_openai_tool_calling_non_stream():
    """Verify single and parallel tool_calls, tool_choice, and finish_reason='tool_calls'."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)

        # First request: single tool call
        if len(captured_requests) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_w100",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city": "Tokyo", "unit": "celsius"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 15, "completion_tokens": 20, "total_tokens": 35},
                },
            )

        # Second request: parallel multiple tool calls
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_s1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_stock_quote",
                                        "arguments": '{"ticker": "NVDA"}',
                                    },
                                },
                                {
                                    "id": "call_s2",
                                    "type": "function",
                                    "function": {
                                        "name": "get_stock_quote",
                                        "arguments": '{"ticker": "AAPL"}',
                                    },
                                },
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 40, "total_tokens": 70},
            },
        )

    # 1. Unit test with direct FoundryChatClient
    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    async def run_unit_test():
        res1 = await client.generate(
            model="gpt-6-astra",
            messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            parallel_tool_calls=False,
            max_tokens=256,
            temperature=1.0,
        )
        assert res1["text"] is None
        assert res1["finish_reason"] == "tool_calls"
        assert res1["tool_calls"] is not None
        assert len(res1["tool_calls"]) == 1
        assert res1["tool_calls"][0]["id"] == "call_w100"
        assert res1["tool_calls"][0]["function"]["name"] == "get_weather"
        assert res1["usage"]["total_tokens"] == 35

        # Check captured request 1
        req1 = captured_requests[0]
        assert req1["model"] == "gpt-6-astra"
        assert req1["tools"] == [WEATHER_TOOL]
        assert req1["tool_choice"] == "auto"
        assert req1["parallel_tool_calls"] is False
        assert req1["max_completion_tokens"] == 256
        assert req1["temperature"] == 1.0

        res2 = await client.generate(
            model="gpt-6-astra",
            messages=[{"role": "user", "content": "Fetch quotes for NVDA and AAPL"}],
            tools=[WEATHER_TOOL, STOCK_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_stock_quote"}},
            parallel_tool_calls=True,
        )
        assert res2["text"] is None
        assert res2["finish_reason"] == "tool_calls"
        assert len(res2["tool_calls"]) == 2
        assert res2["tool_calls"][0]["id"] == "call_s1"
        assert res2["tool_calls"][1]["id"] == "call_s2"
        assert res2["usage"]["total_tokens"] == 70

        # Check captured request 2
        req2 = captured_requests[1]
        assert req2["parallel_tool_calls"] is True
        assert req2["tool_choice"] == {"type": "function", "function": {"name": "get_stock_quote"}}
        assert len(req2["tools"]) == 2

    asyncio.run(run_unit_test())
    asyncio.run(client.close())

    # 2. E2E test with TestClient + create_app
    captured_requests.clear()
    client2 = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client2.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_openai_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client2,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            # Single tool call via public API
            resp1 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
                    "tools": [WEATHER_TOOL],
                    "tool_choice": "auto",
                },
            )
            assert resp1.status_code == 200
            data1 = resp1.json()
            assert data1["choices"][0]["finish_reason"] == "tool_calls"
            assert data1["choices"][0]["message"]["role"] == "assistant"
            assert data1["choices"][0]["message"]["content"] is None
            assert data1["choices"][0]["message"]["tool_calls"][0]["id"] == "call_w100"
            assert data1["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"

            # Parallel tool calls via public API
            resp2 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "Fetch quotes for NVDA and AAPL"}],
                    "tools": [WEATHER_TOOL, STOCK_TOOL],
                    "parallel_tool_calls": True,
                },
            )
            assert resp2.status_code == 200
            data2 = resp2.json()
            assert data2["choices"][0]["finish_reason"] == "tool_calls"
            assert len(data2["choices"][0]["message"]["tool_calls"]) == 2
            assert data2["choices"][0]["message"]["tool_calls"][0]["id"] == "call_s1"
            assert data2["choices"][0]["message"]["tool_calls"][1]["id"] == "call_s2"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client2.close())


def test_foundry_openai_tool_calling_stream():
    """Verify SSE streaming delta_tool_calls, terminal finish_reason='tool_calls', and [DONE]."""
    stream_payload = (
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_tokyo_01","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\": "}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"Tokyo\\"}"}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":18,"completion_tokens":22,"total_tokens":40}}\n\n'
        b"data: [DONE]\n\n"
    )

    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    # 1. Unit test with direct FoundryChatClient.stream_chat
    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    async def run_unit_stream():
        events = []
        async for event in client.stream_chat(
            model="gpt-6-astra",
            messages=[{"role": "user", "content": "Weather in Tokyo?"}],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
        ):
            events.append(event)
        return events

    events = asyncio.run(run_unit_stream())
    asyncio.run(client.close())

    assert len(events) >= 3
    # First chunk contains function definition
    first_tool_call = events[0]["delta_tool_calls"][0]
    assert first_tool_call["index"] == 0
    assert first_tool_call["id"] == "call_tokyo_01"
    assert first_tool_call["function"]["name"] == "get_weather"

    # Subsequent chunks contain arguments
    arg_chunks = [e["delta_tool_calls"][0]["function"]["arguments"] for e in events if e.get("delta_tool_calls") and e["delta_tool_calls"][0]["function"].get("arguments")]
    accumulated_args = "".join(arg_chunks)
    assert json.loads(accumulated_args) == {"city": "Tokyo"}

    # Last chunk has finish_reason="tool_calls" and usage
    last_event = events[-1]
    assert last_event["finish_reason"] == "tool_calls"
    assert last_event["usage"]["total_tokens"] == 40

    # 2. E2E test with TestClient + create_app streaming
    captured_requests.clear()
    client2 = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client2.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_openai_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client2,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Weather in Tokyo?"}],
                    "tools": [WEATHER_TOOL],
                },
            ) as response:
                lines = list(response.iter_lines())

            assert response.status_code == 200
            assert len(captured_requests) == 1
            assert captured_requests[0]["stream"] is True
            assert captured_requests[0]["tools"] == [WEATHER_TOOL]

            parsed_events = []
            for line in lines:
                if line.startswith("data: ") and line.strip() != "data: [DONE]":
                    parsed_events.append(json.loads(line[len("data: ") :]))

            # First SSE event has role: assistant and first tool_call
            assert parsed_events[0]["choices"][0]["delta"]["role"] == "assistant"
            assert parsed_events[0]["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_tokyo_01"
            assert parsed_events[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "get_weather"

            # Reconstruct arguments across deltas
            received_args = "".join(
                ev["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                for ev in parsed_events
                if "tool_calls" in ev["choices"][0]["delta"] and "arguments" in ev["choices"][0]["delta"]["tool_calls"][0]["function"]
            )
            assert json.loads(received_args) == {"city": "Tokyo"}

            # Terminal finish_reason is "tool_calls"
            assert parsed_events[-1]["choices"][0]["finish_reason"] == "tool_calls"
            # Final line is [DONE]
            assert [l for l in lines if l.strip()][-1] == "data: [DONE]"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client2.close())


def test_foundry_openai_multi_turn_history():
    """Verify user -> assistant(tool_calls) -> tool(result) -> assistant(final text) multi-turn cycle."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)

        # Turn 1: user query -> assistant emits tool call
        if len(captured_requests) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_nvda_999",
                                        "type": "function",
                                        "function": {
                                            "name": "get_stock_quote",
                                            "arguments": '{"ticker": "NVDA"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 16, "total_tokens": 28},
                },
            )

        # Turn 2: tool result provided -> assistant produces final answer
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "NVIDIA (NVDA) is currently trading at $128.50 USD.",
                            "tool_calls": None,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 45, "completion_tokens": 20, "total_tokens": 65},
            },
        )

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_openai_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            # Turn 1: Initial query
            r1 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "What is NVDA stock price?"}],
                    "tools": [STOCK_TOOL],
                },
            )
            assert r1.status_code == 200
            d1 = r1.json()
            tool_call = d1["choices"][0]["message"]["tool_calls"][0]
            assert tool_call["id"] == "call_nvda_999"

            # Turn 2: Follow up with tool result
            messages_turn2 = [
                {"role": "user", "content": "What is NVDA stock price?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_nvda_999",
                    "name": "get_stock_quote",
                    "content": '{"price": 128.50, "currency": "USD"}',
                },
            ]

            r2 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": messages_turn2,
                    "tools": [STOCK_TOOL],
                },
            )
            assert r2.status_code == 200
            d2 = r2.json()
            assert d2["choices"][0]["finish_reason"] == "stop"
            assert d2["choices"][0]["message"]["role"] == "assistant"
            assert "128.50" in d2["choices"][0]["message"]["content"]
            assert d2["choices"][0]["message"].get("tool_calls") is None

            # Verify that upstream received full preserved multi-turn history
            req2 = captured_requests[1]
            upstream_msgs = req2["messages"]
            assert len(upstream_msgs) == 3
            assert upstream_msgs[0]["role"] == "user"
            assert upstream_msgs[0]["content"] == "What is NVDA stock price?"

            # Assistant turn preserved
            assert upstream_msgs[1]["role"] == "assistant"
            assert upstream_msgs[1]["content"] is None
            assert upstream_msgs[1]["tool_calls"][0]["id"] == "call_nvda_999"
            assert upstream_msgs[1]["tool_calls"][0]["function"]["name"] == "get_stock_quote"

            # Tool turn preserved
            assert upstream_msgs[2]["role"] == "tool"
            assert upstream_msgs[2]["tool_call_id"] == "call_nvda_999"
            assert upstream_msgs[2]["name"] == "get_stock_quote"
            assert '128.50' in upstream_msgs[2]["content"]
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_foundry_openai_tool_calling_with_assistant_text_and_empty_arguments():
    """Verify tool call with both text thought and empty argument string (e.g. get_time)."""
    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Let me check the time for you.",
                            "tool_calls": [
                                {
                                    "id": "call_time_01",
                                    "type": "function",
                                    "function": {"name": "get_current_time", "arguments": ""},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
            },
        )

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    async def run():
        res = await client.generate(
            model="gpt-6-astra",
            messages=[{"role": "user", "content": "What time is it?"}],
        )
        assert res["text"] == "Let me check the time for you."
        assert res["finish_reason"] == "tool_calls"
        assert res["tool_calls"][0]["id"] == "call_time_01"
        assert res["tool_calls"][0]["function"]["arguments"] == ""

    asyncio.run(run())
    asyncio.run(client.close())


def test_foundry_openai_stream_fallback_finish_reason():
    """Verify stream finishes cleanly when upstream omits explicit finish_reason on tool call chunks."""
    stream_payload = (
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_01","type":"function","function":{"name":"ping","arguments":"{}"}}]},"finish_reason":null}]}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    async def run():
        events = []
        async for event in client.stream_chat(
            model="gpt-6-astra",
            messages=[{"role": "user", "content": "ping"}],
        ):
            events.append(event)
        return events

    events = asyncio.run(run())
    asyncio.run(client.close())

    # Fallback chunk with finish_reason="tool_calls" should be generated because saw_tool_calls was True
    assert any(e.get("finish_reason") == "tool_calls" for e in events)


def _register_foundry_anthropic_alias(
    model_alias: str = "foundry:claude-sonnet-5",
    provider_model: str = "claude-sonnet-5",
) -> dict[str, dict]:
    old_registry = vertex.MODEL_REGISTRY.copy()
    vertex.MODEL_REGISTRY[model_alias] = {
        "provider": "foundry",
        "kind": "chat",
        "provider_model": provider_model,
        "protocol": "anthropic_messages",
    }
    return old_registry


def test_foundry_anthropic_tool_calling_non_stream():
    """Verify Anthropic protocol single and parallel tool_use, tool_choice mapping, and finish_reason='tool_calls'."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/anthropic/v1/messages")
        assert request.headers.get("anthropic-version") == "2023-06-01"
        assert request.headers.get("authorization") == "Bearer test-token"
        body = json.loads(request.content)
        captured_requests.append(body)

        # First request: single tool call
        if len(captured_requests) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "msg_01",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_weather_100",
                            "name": "get_weather",
                            "input": {"city": "Tokyo", "unit": "celsius"},
                        }
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 15, "output_tokens": 20},
                },
            )

        # Second request: parallel multiple tool calls
        return httpx.Response(
            200,
            json={
                "id": "msg_02",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_s1",
                        "name": "get_stock_quote",
                        "input": {"ticker": "NVDA"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_s2",
                        "name": "get_stock_quote",
                        "input": {"ticker": "AAPL"},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 30, "output_tokens": 40},
            },
        )

    # 1. Unit test with direct FoundryChatClient
    client = FoundryChatClient(
        base_url="https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions",
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    async def run_unit_test():
        res1 = await client.generate(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=256,
            temperature=0.7,
            resolved_config={"protocol": "anthropic_messages"},
        )
        assert res1["text"] is None
        assert res1["finish_reason"] == "tool_calls"
        assert res1["tool_calls"] is not None
        assert len(res1["tool_calls"]) == 1
        assert res1["tool_calls"][0]["id"] == "toolu_weather_100"
        assert res1["tool_calls"][0]["type"] == "function"
        assert res1["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(res1["tool_calls"][0]["function"]["arguments"]) == {"city": "Tokyo", "unit": "celsius"}
        assert res1["usage"]["total_tokens"] == 35

        # Check captured request 1
        req1 = captured_requests[0]
        assert req1["model"] == "claude-sonnet-5"
        assert "temperature" not in req1  # Non-default sampling fields omitted
        assert req1["tools"] == [
            {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "input_schema": WEATHER_TOOL["function"]["parameters"],
            }
        ]
        assert req1["tool_choice"] == {"type": "auto"}
        assert req1["max_tokens"] == 256

        res2 = await client.generate(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "Fetch quotes for NVDA and AAPL"}],
            tools=[WEATHER_TOOL, STOCK_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_stock_quote"}},
            resolved_config={"protocol": "anthropic_messages"},
        )
        assert res2["text"] is None
        assert res2["finish_reason"] == "tool_calls"
        assert len(res2["tool_calls"]) == 2
        assert res2["tool_calls"][0]["id"] == "toolu_s1"
        assert res2["tool_calls"][1]["id"] == "toolu_s2"
        assert res2["usage"]["total_tokens"] == 70

        # Check captured request 2
        req2 = captured_requests[1]
        assert req2["tool_choice"] == {"type": "tool", "name": "get_stock_quote"}
        assert len(req2["tools"]) == 2

    asyncio.run(run_unit_test())
    asyncio.run(client.close())

    # 2. E2E test with TestClient + create_app
    captured_requests.clear()
    client2 = FoundryChatClient(
        base_url="https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions",
        token="test-token",
    )
    client2.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_anthropic_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client2,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            # Single tool call via public API
            resp1 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:claude-sonnet-5",
                    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
                    "tools": [WEATHER_TOOL],
                    "tool_choice": "auto",
                },
            )
            assert resp1.status_code == 200
            data1 = resp1.json()
            assert data1["choices"][0]["finish_reason"] == "tool_calls"
            assert data1["choices"][0]["message"]["role"] == "assistant"
            assert data1["choices"][0]["message"]["content"] is None
            assert data1["choices"][0]["message"]["tool_calls"][0]["id"] == "toolu_weather_100"
            assert data1["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"

            # Parallel tool calls via public API
            resp2 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:claude-sonnet-5",
                    "messages": [{"role": "user", "content": "Fetch quotes for NVDA and AAPL"}],
                    "tools": [WEATHER_TOOL, STOCK_TOOL],
                    "tool_choice": {"type": "function", "function": {"name": "get_stock_quote"}},
                },
            )
            assert resp2.status_code == 200
            data2 = resp2.json()
            assert data2["choices"][0]["finish_reason"] == "tool_calls"
            assert len(data2["choices"][0]["message"]["tool_calls"]) == 2
            assert data2["choices"][0]["message"]["tool_calls"][0]["id"] == "toolu_s1"
            assert data2["choices"][0]["message"]["tool_calls"][1]["id"] == "toolu_s2"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client2.close())


def test_foundry_anthropic_tool_calling_stream():
    """Verify Anthropic SSE content_block_* streaming, delta_tool_calls accumulation, finish_reason='tool_calls', and [DONE]."""
    stream_payload = (
        b'data: {"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","content":[],"model":"claude-sonnet-5","usage":{"input_tokens":18,"output_tokens":0}}}\n\n'
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_tokyo_01","name":"get_weather","input":{}}}\n\n'
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"city\\": "}}\n\n'
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\\"Tokyo\\"}"}}\n\n'
        b'data: {"type":"content_block_stop","index":0}\n\n'
        b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":22}}\n\n'
        b'data: {"type":"message_stop"}\n\n'
        b"data: [DONE]\n\n"
    )

    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/anthropic/v1/messages")
        assert request.headers.get("anthropic-version") == "2023-06-01"
        body = json.loads(request.content)
        captured_requests.append(body)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    # 1. Unit test with direct FoundryChatClient.stream_chat
    client = FoundryChatClient(
        base_url="https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions",
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    async def run_unit_stream():
        events = []
        async for event in client.stream_chat(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "Weather in Tokyo?"}],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            resolved_config={"protocol": "anthropic_messages"},
        ):
            events.append(event)
        return events

    events = asyncio.run(run_unit_stream())
    asyncio.run(client.close())

    assert len(events) >= 3
    # First chunk contains tool call initialization
    first_tool_call = events[0]["delta_tool_calls"][0]
    assert first_tool_call["index"] == 0
    assert first_tool_call["id"] == "toolu_tokyo_01"
    assert first_tool_call["function"]["name"] == "get_weather"

    # Subsequent chunks contain partial JSON arguments
    arg_chunks = [
        e["delta_tool_calls"][0]["function"]["arguments"]
        for e in events
        if e.get("delta_tool_calls") and e["delta_tool_calls"][0]["function"].get("arguments")
    ]
    accumulated_args = "".join(arg_chunks)
    assert json.loads(accumulated_args) == {"city": "Tokyo"}

    # Last event has finish_reason="tool_calls" and normalized usage
    last_event = events[-1]
    assert last_event["finish_reason"] == "tool_calls"
    assert last_event["usage"]["total_tokens"] == 40

    # 2. E2E test with TestClient + create_app streaming
    captured_requests.clear()
    client2 = FoundryChatClient(
        base_url="https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions",
        token="test-token",
    )
    client2.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_anthropic_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client2,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:claude-sonnet-5",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Weather in Tokyo?"}],
                    "tools": [WEATHER_TOOL],
                },
            ) as response:
                lines = list(response.iter_lines())

            assert response.status_code == 200
            assert len(captured_requests) == 1
            assert captured_requests[0]["stream"] is True
            assert len(captured_requests[0]["tools"]) == 1

            parsed_events = []
            for line in lines:
                if line.startswith("data: ") and line.strip() != "data: [DONE]":
                    parsed_events.append(json.loads(line[len("data: ") :]))

            # First SSE event has role: assistant and first tool_call
            assert parsed_events[0]["choices"][0]["delta"]["role"] == "assistant"
            assert parsed_events[0]["choices"][0]["delta"]["tool_calls"][0]["id"] == "toolu_tokyo_01"
            assert parsed_events[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "get_weather"

            # Reconstruct arguments across deltas
            received_args = "".join(
                ev["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                for ev in parsed_events
                if "tool_calls" in ev["choices"][0]["delta"]
                and "arguments" in ev["choices"][0]["delta"]["tool_calls"][0]["function"]
            )
            assert json.loads(received_args) == {"city": "Tokyo"}

            # Terminal finish_reason is "tool_calls"
            assert parsed_events[-1]["choices"][0]["finish_reason"] == "tool_calls"
            # Final line is [DONE]
            assert [l for l in lines if l.strip()][-1] == "data: [DONE]"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client2.close())


def test_foundry_anthropic_multi_turn_history():
    """Verify User -> Assistant(tool_use) -> multiple Tool results -> single merged User turn -> Assistant final text."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)

        # Turn 1: user query -> assistant emits 2 parallel tool calls
        if len(captured_requests) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "msg_01",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_w1",
                            "name": "get_weather",
                            "input": {"city": "Tokyo"},
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_s1",
                            "name": "get_stock_quote",
                            "input": {"ticker": "NVDA"},
                        },
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 15, "output_tokens": 25},
                },
            )

        # Turn 2: tool results provided -> assistant produces final answer
        return httpx.Response(
            200,
            json={
                "id": "msg_02",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Tokyo is currently 22C and clear. NVIDIA (NVDA) is trading at $128.50 USD.",
                    }
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 60, "output_tokens": 30},
            },
        )

    client = FoundryChatClient(
        base_url="https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions",
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_anthropic_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            # Turn 1: Initial query requesting both weather and stock
            r1 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:claude-sonnet-5",
                    "messages": [{"role": "user", "content": "What is Tokyo weather and NVDA stock price?"}],
                    "tools": [WEATHER_TOOL, STOCK_TOOL],
                },
            )
            assert r1.status_code == 200
            d1 = r1.json()
            tool_calls = d1["choices"][0]["message"]["tool_calls"]
            assert len(tool_calls) == 2
            assert tool_calls[0]["id"] == "toolu_w1"
            assert tool_calls[1]["id"] == "toolu_s1"

            # Turn 2: Follow up with two consecutive role: "tool" messages
            messages_turn2 = [
                {"role": "user", "content": "What is Tokyo weather and NVDA stock price?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
                {
                    "role": "tool",
                    "tool_call_id": "toolu_w1",
                    "name": "get_weather",
                    "content": '{"city": "Tokyo", "temp": 22, "condition": "Clear"}',
                },
                {
                    "role": "tool",
                    "tool_call_id": "toolu_s1",
                    "name": "get_stock_quote",
                    "content": '{"ticker": "NVDA", "price": 128.50, "currency": "USD"}',
                },
            ]

            r2 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:claude-sonnet-5",
                    "messages": messages_turn2,
                    "tools": [WEATHER_TOOL, STOCK_TOOL],
                },
            )
            assert r2.status_code == 400
            error = r2.json()["error"]
            assert error["type"] == error["code"] == "invalid_request_error"
            assert error["param"] == "messages[2].name"
            assert "anthropic_messages" in error["message"]
            assert len(captured_requests) == 1
            retry = json.loads(r2.request.content)
            for message in retry["messages"]:
                if message.get("role") == "tool":
                    message.pop("name", None)
            r2 = http_client.post("/v1/chat/completions", json=retry)
            assert r2.status_code == 200
            d2 = r2.json()
            assert d2["choices"][0]["finish_reason"] == "stop"
            assert d2["choices"][0]["message"]["role"] == "assistant"
            assert "128.50" in d2["choices"][0]["message"]["content"]
            assert d2["choices"][0]["message"].get("tool_calls") is None

            # Verify that upstream received properly transformed multi-turn history
            req2 = captured_requests[1]
            upstream_msgs = req2["messages"]

            # CRITICAL: Exactly 3 alternating turns (user -> assistant -> user)!
            assert len(upstream_msgs) == 3

            # Turn 0: initial user
            assert upstream_msgs[0]["role"] == "user"
            assert upstream_msgs[0]["content"] == "What is Tokyo weather and NVDA stock price?"

            # Turn 1: assistant with tool_use content blocks
            assert upstream_msgs[1]["role"] == "assistant"
            assistant_blocks = upstream_msgs[1]["content"]
            assert len(assistant_blocks) == 2
            assert assistant_blocks[0]["type"] == "tool_use"
            assert assistant_blocks[0]["id"] == "toolu_w1"
            assert assistant_blocks[0]["name"] == "get_weather"
            assert assistant_blocks[0]["input"] == {"city": "Tokyo"}
            assert assistant_blocks[1]["type"] == "tool_use"
            assert assistant_blocks[1]["id"] == "toolu_s1"
            assert assistant_blocks[1]["name"] == "get_stock_quote"
            assert assistant_blocks[1]["input"] == {"ticker": "NVDA"}

            # Turn 2: SINGLE merged user turn containing BOTH tool_result blocks!
            assert upstream_msgs[2]["role"] == "user"
            tool_result_blocks = upstream_msgs[2]["content"]
            assert len(tool_result_blocks) == 2
            assert tool_result_blocks[0]["type"] == "tool_result"
            assert tool_result_blocks[0]["tool_use_id"] == "toolu_w1"
            assert "Clear" in tool_result_blocks[0]["content"]
            assert tool_result_blocks[1]["type"] == "tool_result"
            assert tool_result_blocks[1]["tool_use_id"] == "toolu_s1"
            assert "128.50" in tool_result_blocks[1]["content"]
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_foundry_anthropic_tool_choice_variants_and_turn_alternation():
    """Verify tool_choice variants ('none', 'required', named) and strict turn alternation merging."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Understood."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    client = FoundryChatClient(
        base_url="https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions",
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    async def run_tests():
        # Variant 1: tool_choice="none" -> tools and tool_choice omitted
        await client.generate(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "Just talk"}],
            tools=[WEATHER_TOOL],
            tool_choice="none",
            resolved_config={"protocol": "anthropic_messages"},
        )
        req1 = captured_requests[0]
        assert "tools" not in req1
        assert "tool_choice" not in req1

        # Variant 2: tool_choice="required" -> maps to {"type": "any"}
        await client.generate(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "Must call a tool"}],
            tools=[WEATHER_TOOL],
            tool_choice="required",
            resolved_config={"protocol": "anthropic_messages"},
        )
        req2 = captured_requests[1]
        assert req2["tool_choice"] == {"type": "any"}

        # Variant 3: tool_choice={"name": "get_weather"} -> maps to {"type": "tool", "name": "get_weather"}
        await client.generate(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "Call weather"}],
            tools=[WEATHER_TOOL],
            tool_choice={"name": "get_weather"},
            resolved_config={"protocol": "anthropic_messages"},
        )
        req3 = captured_requests[2]
        assert req3["tool_choice"] == {"type": "tool", "name": "get_weather"}

        # Variant 4: Strict turn alternation: tool messages followed by user message
        await client.generate(
            model="claude-sonnet-5",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "First question"},
                {
                    "role": "assistant",
                    "content": "Checking...",
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
                {"role": "user", "content": "And tell me what jacket to wear."},
            ],
            tools=[WEATHER_TOOL],
            resolved_config={"protocol": "anthropic_messages"},
        )
        req4 = captured_requests[3]
        assert req4["system"] == "You are a helpful assistant."
        msgs4 = req4["messages"]
        # Exactly 3 messages: user -> assistant -> user (merged tool_result + user text)!
        assert len(msgs4) == 3
        assert msgs4[0]["role"] == "user"
        assert msgs4[0]["content"] == "First question"
        assert msgs4[1]["role"] == "assistant"
        # Turn 2 user content should have both tool_result and text block!
        assert msgs4[2]["role"] == "user"
        assert len(msgs4[2]["content"]) == 2
        assert msgs4[2]["content"][0]["type"] == "tool_result"
        assert msgs4[2]["content"][0]["content"] == "Sunny"
        assert msgs4[2]["content"][1]["type"] == "text"
        assert msgs4[2]["content"][1]["text"] == "And tell me what jacket to wear."

    asyncio.run(run_tests())
    asyncio.run(client.close())


def test_foundry_anthropic_stream_fallback_finish_reason():
    """Verify Anthropic stream finishes cleanly when upstream omits explicit message_delta finish_reason."""
    stream_payload = (
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":0}}}\n\n'
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call_ping","name":"ping","input":{}}}\n\n'
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{}"}}\n\n'
        b'data: {"type":"content_block_stop","index":0}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(
        base_url="https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions",
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    async def run():
        events = []
        async for event in client.stream_chat(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "ping"}],
            tools=[WEATHER_TOOL],
            resolved_config={"protocol": "anthropic_messages"},
        ):
            events.append(event)
        return events

    events = asyncio.run(run())
    asyncio.run(client.close())

    # Fallback chunk with finish_reason="tool_calls" should be generated because saw_tool_calls was True
    assert any(e.get("finish_reason") == "tool_calls" for e in events)


def test_foundry_anthropic_tool_calling_with_text_and_empty_args():
    """Verify Anthropic response with text block before tool_use and empty input."""
    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "msg_text_and_tool",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me look that up for you."},
                    {
                        "type": "tool_use",
                        "id": "call_time_01",
                        "name": "get_current_time",
                        "input": {},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 15},
            },
        )

    client = FoundryChatClient(
        base_url="https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions",
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    async def run():
        res = await client.generate(
            model="claude-sonnet-5",
            messages=[{"role": "user", "content": "What time is it?"}],
            resolved_config={"protocol": "anthropic_messages"},
        )
        assert res["text"] == "Let me look that up for you."
        assert res["finish_reason"] == "tool_calls"
        assert res["tool_calls"][0]["id"] == "call_time_01"
        assert res["tool_calls"][0]["function"]["name"] == "get_current_time"
        assert res["tool_calls"][0]["function"]["arguments"] == "{}"

    asyncio.run(run())
    asyncio.run(client.close())


# ==============================================================================
# Milestone 4 / Slice 4: Foundry xAI Responses Protocol (Grok Family)
# ==============================================================================


def _register_foundry_xai_alias(
    model_alias: str = "foundry:grok-4.6",
    provider_model: str = "grok-4.6",
) -> dict[str, dict]:
    old_registry = vertex.MODEL_REGISTRY.copy()
    vertex.MODEL_REGISTRY[model_alias] = {
        "provider": "foundry",
        "kind": "chat",
        "provider_model": provider_model,
        "protocol": "xai_responses",
    }
    return old_registry


def test_foundry_xai_tool_calling_non_stream():
    """Verify xAI protocol single and parallel function_call, tool_choice mapping, and finish_reason='tool_calls'."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/xai/v1/responses")
        assert request.headers.get("authorization") == "Bearer test-token"
        body = json.loads(request.content)
        captured_requests.append(body)

        # First request: single tool call
        if len(captured_requests) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "resp_01",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "call_w100",
                            "call_id": "call_w100",
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo", "unit": "celsius"}',
                        }
                    ],
                    "usage": {"input_tokens": 15, "output_tokens": 20, "total_tokens": 35},
                },
            )

        # Second request: parallel multiple tool calls
        return httpx.Response(
            200,
            json={
                "id": "resp_02",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "id": "call_s1",
                        "call_id": "call_s1",
                        "name": "get_stock_quote",
                        "arguments": '{"ticker": "NVDA"}',
                    },
                    {
                        "type": "function_call",
                        "id": "call_s2",
                        "call_id": "call_s2",
                        "name": "get_stock_quote",
                        "arguments": '{"ticker": "AAPL"}',
                    },
                ],
                "usage": {"input_tokens": 30, "output_tokens": 40, "total_tokens": 70},
            },
        )

    # 1. Unit test with direct FoundryChatClient
    client = FoundryChatClient(
        base_url="https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions",
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    async def run_unit_test():
        res1 = await client.generate(
            model="grok-4.6",
            messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            max_tokens=256,
            temperature=0.7,
            resolved_config={"protocol": "xai_responses"},
        )
        assert res1["text"] is None
        assert res1["finish_reason"] == "tool_calls"
        assert res1["tool_calls"] is not None
        assert len(res1["tool_calls"]) == 1
        assert res1["tool_calls"][0]["id"] == "call_w100"
        assert res1["tool_calls"][0]["type"] == "function"
        assert res1["tool_calls"][0]["function"]["name"] == "get_weather"
        assert json.loads(res1["tool_calls"][0]["function"]["arguments"]) == {"city": "Tokyo", "unit": "celsius"}
        assert res1["usage"]["total_tokens"] == 35

        # Check captured request 1
        req1 = captured_requests[0]
        assert req1["model"] == "grok-4.6"
        assert req1["max_output_tokens"] == 256
        assert req1["temperature"] == 0.7
        assert req1["tool_choice"] == "auto"
        assert req1["tools"] == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": WEATHER_TOOL["function"]["parameters"],
            }
        ]
        assert req1["input"] == [{"role": "user", "content": "What is the weather in Tokyo?"}]

        # Request 2: parallel tool calling with named dict tool_choice
        res2 = await client.generate(
            model="grok-4.6",
            messages=[{"role": "user", "content": "Fetch quotes for NVDA and AAPL"}],
            tools=[WEATHER_TOOL, STOCK_TOOL],
            tool_choice={"type": "function", "function": {"name": "get_stock_quote"}},
            resolved_config={"protocol": "xai_responses"},
        )
        assert res2["text"] is None
        assert res2["finish_reason"] == "tool_calls"
        assert len(res2["tool_calls"]) == 2
        assert res2["tool_calls"][0]["id"] == "call_s1"
        assert res2["tool_calls"][1]["id"] == "call_s2"
        assert res2["usage"]["total_tokens"] == 70

        # Check captured request 2
        req2 = captured_requests[1]
        assert req2["tool_choice"] == {"type": "function", "name": "get_stock_quote"}
        assert len(req2["tools"]) == 2

    asyncio.run(run_unit_test())
    asyncio.run(client.close())

    # 2. E2E test with TestClient + create_app
    captured_requests.clear()
    client2 = FoundryChatClient(
        base_url="https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions",
        token="test-token",
    )
    client2.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_xai_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client2,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            # Single tool call via public API
            resp1 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": [{"role": "user", "content": "What is the weather in Tokyo?"}],
                    "tools": [WEATHER_TOOL],
                    "tool_choice": "auto",
                },
            )
            assert resp1.status_code == 200
            data1 = resp1.json()
            assert data1["choices"][0]["finish_reason"] == "tool_calls"
            assert data1["choices"][0]["message"]["role"] == "assistant"
            assert data1["choices"][0]["message"]["content"] is None
            assert data1["choices"][0]["message"]["tool_calls"][0]["id"] == "call_w100"
            assert data1["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"

            # Parallel tool calls via public API
            resp2 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": [{"role": "user", "content": "Fetch quotes for NVDA and AAPL"}],
                    "tools": [WEATHER_TOOL, STOCK_TOOL],
                    "tool_choice": {"type": "function", "function": {"name": "get_stock_quote"}},
                },
            )
            assert resp2.status_code == 200
            data2 = resp2.json()
            assert data2["choices"][0]["finish_reason"] == "tool_calls"
            assert len(data2["choices"][0]["message"]["tool_calls"]) == 2
            assert data2["choices"][0]["message"]["tool_calls"][0]["id"] == "call_s1"
            assert data2["choices"][0]["message"]["tool_calls"][1]["id"] == "call_s2"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client2.close())


def test_foundry_xai_tool_calling_stream():
    """Verify xAI SSE response.output_item.* streaming, delta_tool_calls accumulation, finish_reason='tool_calls', and [DONE]."""
    stream_payload = (
        b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"call_tokyo_01","call_id":"call_tokyo_01","name":"get_weather","arguments":""}}\n\n'
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_tokyo_01","delta":"{\\"city\\": "}\n\n'
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_tokyo_01","delta":"\\"Tokyo\\"}"}\n\n'
        b'data: {"type":"response.output_item.done","output_index":0}\n\n'
        b'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":18,"output_tokens":22,"total_tokens":40}}}\n\n'
        b"data: [DONE]\n\n"
    )

    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/xai/v1/responses")
        body = json.loads(request.content)
        captured_requests.append(body)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    # 1. Unit test with direct FoundryChatClient.stream_chat
    client = FoundryChatClient(
        base_url="https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions",
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    async def run_unit_stream():
        events = []
        async for event in client.stream_chat(
            model="grok-4.6",
            messages=[{"role": "user", "content": "Weather in Tokyo?"}],
            tools=[WEATHER_TOOL],
            tool_choice="auto",
            resolved_config={"protocol": "xai_responses"},
        ):
            events.append(event)
        return events

    events = asyncio.run(run_unit_stream())
    asyncio.run(client.close())

    assert len(events) >= 3
    # First chunk contains tool call initialization
    first_tool_call = events[0]["delta_tool_calls"][0]
    assert first_tool_call["index"] == 0
    assert first_tool_call["id"] == "call_tokyo_01"
    assert first_tool_call["function"]["name"] == "get_weather"

    # Subsequent chunks contain partial JSON arguments
    arg_chunks = [
        e["delta_tool_calls"][0]["function"]["arguments"]
        for e in events
        if e.get("delta_tool_calls") and e["delta_tool_calls"][0]["function"].get("arguments")
    ]
    assert "".join(arg_chunks) == '{"city": "Tokyo"}'

    # Final event contains finish_reason and usage
    final_event = events[-1]
    assert final_event["finish_reason"] == "tool_calls"
    assert final_event["usage"]["prompt_tokens"] == 18
    assert final_event["usage"]["completion_tokens"] == 22
    assert final_event["usage"]["total_tokens"] == 40

    # 2. E2E test with TestClient + create_app
    client2 = FoundryChatClient(
        base_url="https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions",
        token="test-token",
    )
    client2.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_xai_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client2,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            resp = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": [{"role": "user", "content": "Weather in Tokyo?"}],
                    "tools": [WEATHER_TOOL],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]

            lines = resp.text.split("\n")
            json_chunks = []
            for line in lines:
                if line.startswith("data:") and not line.startswith("data: [DONE]"):
                    chunk_str = line[len("data:") :].strip()
                    if chunk_str:
                        json_chunks.append(json.loads(chunk_str))

            assert len(json_chunks) >= 3

            # Chunk 0 has tool call structure with name
            c0_delta = json_chunks[0]["choices"][0]["delta"]
            assert c0_delta["tool_calls"][0]["index"] == 0
            assert c0_delta["tool_calls"][0]["id"] == "call_tokyo_01"
            assert c0_delta["tool_calls"][0]["function"]["name"] == "get_weather"

            # Reconstruct arguments across chunks
            received_args = "".join(
                c["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
                for c in json_chunks
                if c["choices"][0]["delta"].get("tool_calls")
                and "arguments" in c["choices"][0]["delta"]["tool_calls"][0]["function"]
                and c["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
            )
            assert json.loads(received_args) == {"city": "Tokyo"}

            # Terminal finish_reason must be "tool_calls"
            final_choice = json_chunks[-1]["choices"][0]
            assert final_choice["finish_reason"] == "tool_calls"

            # Last non-empty SSE line must be data: [DONE]
            assert [l for l in lines if l.strip()][-1] == "data: [DONE]"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client2.close())


def test_foundry_xai_multi_turn_history():
    """Verify User -> Assistant(function_call) -> multiple Tool results(function_call_output) -> Assistant final synthesis."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)

        # Turn 1: user query -> assistant emits 2 parallel tool calls
        if len(captured_requests) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "resp_01",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "call_w1",
                            "call_id": "call_w1",
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo"}',
                        },
                        {
                            "type": "function_call",
                            "id": "call_s1",
                            "call_id": "call_s1",
                            "name": "get_stock_quote",
                            "arguments": '{"ticker": "NVDA"}',
                        },
                    ],
                    "usage": {"input_tokens": 15, "output_tokens": 25, "total_tokens": 40},
                },
            )

        # Turn 2: tool results provided -> assistant produces final answer
        return httpx.Response(
            200,
            json={
                "id": "resp_02",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "Tokyo is currently 22C and clear. NVIDIA (NVDA) is trading at $128.50 USD.",
                            }
                        ],
                    }
                ],
                "usage": {"input_tokens": 60, "output_tokens": 30, "total_tokens": 90},
            },
        )

    client = FoundryChatClient(
        base_url="https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions",
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_xai_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            # Turn 1: Initial query requesting both weather and stock
            r1 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": [{"role": "user", "content": "What is Tokyo weather and NVDA stock price?"}],
                    "tools": [WEATHER_TOOL, STOCK_TOOL],
                },
            )
            assert r1.status_code == 200
            d1 = r1.json()
            tool_calls = d1["choices"][0]["message"]["tool_calls"]
            assert len(tool_calls) == 2
            assert tool_calls[0]["id"] == "call_w1"
            assert tool_calls[1]["id"] == "call_s1"

            # Turn 2: Follow up with two role: "tool" messages
            messages_turn2 = [
                {"role": "user", "content": "What is Tokyo weather and NVDA stock price?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_calls,
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_w1",
                    "name": "get_weather",
                    "content": '{"city": "Tokyo", "temp": 22, "condition": "Clear"}',
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_s1",
                    "name": "get_stock_quote",
                    "content": '{"ticker": "NVDA", "price": 128.50, "currency": "USD"}',
                },
            ]

            r2 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": messages_turn2,
                    "tools": [WEATHER_TOOL, STOCK_TOOL],
                },
            )
            assert r2.status_code == 400
            error = r2.json()["error"]
            assert error["type"] == error["code"] == "invalid_request_error"
            assert error["param"] == "messages[2].name"
            assert "xai_responses" in error["message"]
            assert len(captured_requests) == 1
            retry = json.loads(r2.request.content)
            for message in retry["messages"]:
                if message.get("role") == "tool":
                    message.pop("name", None)
            r2 = http_client.post("/v1/chat/completions", json=retry)
            assert r2.status_code == 200
            d2 = r2.json()
            assert d2["choices"][0]["finish_reason"] == "stop"
            assert d2["choices"][0]["message"]["role"] == "assistant"
            assert "128.50" in d2["choices"][0]["message"]["content"]
            assert d2["choices"][0]["message"].get("tool_calls") is None

            # Verify that upstream received properly transformed xAI input items
            req2 = captured_requests[1]
            upstream_input = req2["input"]

            # Expected 5 items in sequence:
            # 1. user text
            # 2. function_call 1
            # 3. function_call 2
            # 4. function_call_output 1
            # 5. function_call_output 2
            assert len(upstream_input) == 5

            # Item 0: initial user
            assert upstream_input[0]["role"] == "user"
            assert upstream_input[0]["content"] == "What is Tokyo weather and NVDA stock price?"

            # Item 1: assistant function_call 1
            assert upstream_input[1]["type"] == "function_call"
            assert upstream_input[1]["call_id"] == "call_w1"
            assert upstream_input[1]["name"] == "get_weather"
            assert upstream_input[1]["arguments"] == '{"city": "Tokyo"}'

            # Item 2: assistant function_call 2
            assert upstream_input[2]["type"] == "function_call"
            assert upstream_input[2]["call_id"] == "call_s1"
            assert upstream_input[2]["name"] == "get_stock_quote"
            assert upstream_input[2]["arguments"] == '{"ticker": "NVDA"}'

            # Item 3: function_call_output 1
            assert upstream_input[3]["type"] == "function_call_output"
            assert upstream_input[3]["call_id"] == "call_w1"
            assert upstream_input[3]["output"] == '{"city": "Tokyo", "temp": 22, "condition": "Clear"}'

            # Item 4: function_call_output 2
            assert upstream_input[4]["type"] == "function_call_output"
            assert upstream_input[4]["call_id"] == "call_s1"
            assert upstream_input[4]["output"] == '{"ticker": "NVDA", "price": 128.50, "currency": "USD"}'
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_foundry_xai_stream_fallback_finish_reason():
    """Verify xAI stream finishes cleanly when upstream omits explicit response.completed finish_reason."""
    stream_payload = (
        b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"call_ping","call_id":"call_ping","name":"ping","arguments":""}}\n\n'
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_ping","delta":"{}"}\n\n'
        b'data: {"type":"response.output_item.done","output_index":0}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(
        base_url="https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions",
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    async def run():
        events = []
        async for event in client.stream_chat(
            model="grok-4.6",
            messages=[{"role": "user", "content": "ping"}],
            tools=[WEATHER_TOOL],
            resolved_config={"protocol": "xai_responses"},
        ):
            events.append(event)
        return events

    events = asyncio.run(run())
    asyncio.run(client.close())

    # Fallback chunk with finish_reason="tool_calls" should be generated because saw_tool_calls was True
    assert any(e.get("finish_reason") == "tool_calls" for e in events)


def test_foundry_xai_tool_calling_with_text_and_empty_args():
    """Verify xAI response with text before function_call and empty arguments."""
    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "resp_text_and_tool",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Let me look that up for you."}],
                    },
                    {
                        "type": "function_call",
                        "id": "call_time_01",
                        "call_id": "call_time_01",
                        "name": "get_current_time",
                        "arguments": "",
                    },
                ],
                "usage": {"input_tokens": 10, "output_tokens": 15, "total_tokens": 25},
            },
        )

    client = FoundryChatClient(
        base_url="https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions",
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    async def run():
        res = await client.generate(
            model="grok-4.6",
            messages=[{"role": "user", "content": "What time is it?"}],
            resolved_config={"protocol": "xai_responses"},
        )
        assert res["text"] == "Let me look that up for you."
        assert res["finish_reason"] == "tool_calls"
        assert res["tool_calls"][0]["id"] == "call_time_01"
        assert res["tool_calls"][0]["function"]["name"] == "get_current_time"
        assert res["tool_calls"][0]["function"]["arguments"] == "{}"

    asyncio.run(run())
    asyncio.run(client.close())


def test_foundry_xai_tool_choice_variants():
    """Verify xAI tool_choice variants ('none', 'required', named) and system message handling in input."""
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)
        return httpx.Response(
            200,
            json={
                "id": "resp_var",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "OK"}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            },
        )

    client = FoundryChatClient(
        base_url="https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions",
        token="test-token",
    )
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    async def run_tests():
        # Variant 1: tool_choice="none"
        await client.generate(
            model="grok-4.6",
            messages=[{"role": "user", "content": "No tools"}],
            tools=[WEATHER_TOOL],
            tool_choice="none",
            resolved_config={"protocol": "xai_responses"},
        )
        req1 = captured_requests[0]
        assert req1["tool_choice"] == "none"
        assert len(req1["tools"]) == 1

        # Variant 2: tool_choice="required"
        await client.generate(
            model="grok-4.6",
            messages=[{"role": "user", "content": "Must call tool"}],
            tools=[WEATHER_TOOL],
            tool_choice="required",
            resolved_config={"protocol": "xai_responses"},
        )
        req2 = captured_requests[1]
        assert req2["tool_choice"] == "required"

        # Variant 3: tool_choice={"name": "get_weather"}
        await client.generate(
            model="grok-4.6",
            messages=[{"role": "user", "content": "Call weather"}],
            tools=[WEATHER_TOOL],
            tool_choice={"name": "get_weather"},
            resolved_config={"protocol": "xai_responses"},
        )
        req3 = captured_requests[2]
        assert req3["tool_choice"] == {"type": "function", "name": "get_weather"}

        # Variant 4: system message in messages
        await client.generate(
            model="grok-4.6",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
            ],
            resolved_config={"protocol": "xai_responses"},
        )
        req4 = captured_requests[3]
        assert req4["input"][0] == {"role": "system", "content": "You are a helpful assistant."}
        assert req4["input"][1] == {"role": "user", "content": "Hello"}

    asyncio.run(run_tests())
    asyncio.run(client.close())


# ==============================================================================
# Milestone 5 / Slice 5: Error Unwrapping & OpenAI Standard Error Mapping Tests
# ==============================================================================


def test_foundry_e2e_error_unwrapping_llm_http_client_error_response_body():
    """Verify Palantir LanguageModelService:LlmHttpClientError with Optional[...] JSON in responseBody unwraps to 400 OpenAI error."""
    palantir_error_body = {
        "errorCode": "CUSTOM_CLIENT",
        "errorName": "LanguageModelService:LlmHttpClientError",
        "errorInstanceId": "e1f2a3b4-5678-90ab-cdef-1234567890ab",
        "parameters": {
            "responseBody": (
                'Optional[{"error": {"message": "Invalid model parameter", '
                '"type": "invalid_request_error", "code": "param_invalid"}}]'
            ),
            "errorCode": "Optional[param_invalid]",
        },
    }

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            headers={"content-type": "application/json"},
            json=palantir_error_body,
        )

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_openai_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            resp = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert resp.status_code == 400
            data = resp.json()
            assert "error" in data
            err = data["error"]
            assert err["message"] == "Invalid model parameter"
            assert err["type"] == "invalid_request_error"
            assert err["code"] == "param_invalid"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_foundry_e2e_error_fallback_error_message_and_code():
    """Verify parameters.errorMessage and parameters.errorCode fallback handling when responseBody is absent or unparseable."""
    calls = 0

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            # First call: Optional-wrapped errorMessage and errorCode with 429
            return httpx.Response(
                429,
                headers={"content-type": "application/json"},
                json={
                    "errorCode": "CUSTOM_CLIENT",
                    "errorName": "LanguageModelService:LlmHttpClientError",
                    "parameters": {
                        "errorMessage": "Optional[Upstream model rate limit exceeded]",
                        "errorCode": "Optional[rate_limit_exceeded]",
                    },
                },
            )
        # Second call: Plain string errorMessage and errorCode with 400
        return httpx.Response(
            400,
            headers={"content-type": "application/json"},
            json={
                "errorCode": "CUSTOM_CLIENT",
                "errorName": "LanguageModelService:LlmHttpClientError",
                "parameters": {
                    "errorMessage": "Context length exceeded maximum allowed tokens",
                    "errorCode": "context_length_exceeded",
                },
            },
        )

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_openai_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            # 1. 429 rate limit fallback
            resp1 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "Request 1"}],
                },
            )
            assert resp1.status_code == 429
            data1 = resp1.json()
            assert data1["error"]["message"] == "Upstream model rate limit exceeded"
            assert data1["error"]["code"] == "rate_limit_exceeded"
            assert data1["error"]["type"] == "rate_limit_error"

            # 2. 400 context length fallback
            resp2 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "Request 2"}],
                },
            )
            assert resp2.status_code == 400
            data2 = resp2.json()
            assert data2["error"]["message"] == "Context length exceeded maximum allowed tokens"
            assert data2["error"]["code"] == "context_length_exceeded"
            assert data2["error"]["type"] == "invalid_request_error"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_foundry_e2e_error_upstream_502_html_page():
    """Verify upstream 502 HTML error page is safely converted to 502 OpenAI standard error response."""
    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            502,
            headers={"content-type": "text/html; charset=utf-8"},
            content=b"<html><body>502 Bad Gateway</body></html>",
        )

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_openai_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            # Non-streaming 502 HTML
            resp = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )
            assert resp.status_code == 502
            data = resp.json()
            assert "error" in data
            assert data["error"]["type"] == "api_error"
            assert "502 Bad Gateway" in data["error"]["message"]

            # Streaming 502 HTML
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": [{"role": "user", "content": "ping"}],
                },
            ) as stream_resp:
                lines = list(stream_resp.iter_lines())

            assert stream_resp.status_code == 200
            error_events = [
                json.loads(line[len("data: ") :])
                for line in lines
                if line.startswith("data: ") and line.strip() != "data: [DONE]"
            ]
            assert len(error_events) >= 1
            assert "error" in error_events[0]
            assert error_events[0]["error"]["type"] == "api_error"
            assert "502 Bad Gateway" in error_events[0]["error"]["message"]
            assert [l for l in lines if l.strip()][-1] == "data: [DONE]"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_foundry_e2e_error_upstream_timeout_and_connection_error():
    """Verify upstream 504 Timeout and 502 Connection Error mapping in both non-streaming and streaming."""
    mode = "timeout"

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        if mode == "timeout":
            raise httpx.TimeoutException("Read timed out after 60 seconds", request=request)
        raise httpx.ConnectError("Failed to establish a new connection: Connection refused", request=request)

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_openai_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            # 1. Non-streaming Timeout -> 504
            mode = "timeout"
            resp_timeout = http_client.post(
                "/v1/chat/completions",
                json={"model": "foundry:gpt-6-astra", "messages": [{"role": "user", "content": "test"}]},
            )
            assert resp_timeout.status_code == 504
            data_t = resp_timeout.json()
            assert data_t["error"]["type"] == "api_error"
            assert data_t["error"]["code"] == "timeout"
            assert "timed out" in data_t["error"]["message"].lower()

            # 2. Non-streaming Connection Error -> 502
            mode = "connect_error"
            resp_conn = http_client.post(
                "/v1/chat/completions",
                json={"model": "foundry:gpt-6-astra", "messages": [{"role": "user", "content": "test"}]},
            )
            assert resp_conn.status_code == 502
            data_c = resp_conn.json()
            assert data_c["error"]["type"] == "api_error"
            assert data_c["error"]["code"] == "connection_error"
            assert "connection error" in data_c["error"]["message"].lower()

            # 3. Streaming Timeout -> SSE error event
            mode = "timeout"
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": "foundry:gpt-6-astra", "stream": True, "messages": [{"role": "user", "content": "test"}]},
            ) as s_resp_t:
                lines_t = list(s_resp_t.iter_lines())
            t_events = [json.loads(l[6:]) for l in lines_t if l.startswith("data: ") and l.strip() != "data: [DONE]"]
            assert len(t_events) >= 1
            assert t_events[0]["error"]["code"] == "timeout"
            assert [l for l in lines_t if l.strip()][-1] == "data: [DONE]"

            # 4. Streaming Connection Error -> SSE error event
            mode = "connect_error"
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": "foundry:gpt-6-astra", "stream": True, "messages": [{"role": "user", "content": "test"}]},
            ) as s_resp_c:
                lines_c = list(s_resp_c.iter_lines())
            c_events = [json.loads(l[6:]) for l in lines_c if l.startswith("data: ") and l.strip() != "data: [DONE]"]
            assert len(c_events) >= 1
            assert c_events[0]["error"]["code"] == "connection_error"
            assert [l for l in lines_c if l.strip()][-1] == "data: [DONE]"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


# ==============================================================================
# Milestone 5 / Slice 5: 4-Tier Multi-Turn Hermes Agent E2E Integration Tests
# ==============================================================================


def test_foundry_e2e_hermes_cycle_openai_non_stream():
    """Verify complete multi-turn Hermes cycle on OpenAI protocol (Non-streaming).
    Turn 1: User prompt -> Parallel tool calls (get_weather, get_stock_quote).
    Turn 2: Agent supplies 2 tool results -> Model final synthesis.
    """
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        captured_requests.append(body)

        if len(captured_requests) == 1:
            # Turn 1 response: parallel tool calls
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_tokyo_w",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city": "Tokyo", "unit": "celsius"}',
                                        },
                                    },
                                    {
                                        "id": "call_nvda_s",
                                        "type": "function",
                                        "function": {
                                            "name": "get_stock_quote",
                                            "arguments": '{"ticker": "NVDA"}',
                                        },
                                    },
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 25, "completion_tokens": 35, "total_tokens": 60},
                },
            )

        # Turn 2 response: final synthesis
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Tokyo weather is 22C and clear, while NVDA is trading at $128.50 USD.",
                            "tool_calls": None,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 70, "completion_tokens": 20, "total_tokens": 90},
            },
        )

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_openai_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            # Turn 1: Hermes initial request with 2 tools
            t1_resp = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "Check Tokyo weather and NVDA stock"}],
                    "tools": [WEATHER_TOOL, STOCK_TOOL],
                },
            )
            assert t1_resp.status_code == 200
            t1_data = t1_resp.json()
            assert t1_data["choices"][0]["finish_reason"] == "tool_calls"
            tool_calls = t1_data["choices"][0]["message"]["tool_calls"]
            assert len(tool_calls) == 2
            assert tool_calls[0]["id"] == "call_tokyo_w"
            assert tool_calls[1]["id"] == "call_nvda_s"

            # Turn 2: Hermes executes tools and sends results back
            t2_messages = [
                {"role": "user", "content": "Check Tokyo weather and NVDA stock"},
                {"role": "assistant", "content": None, "tool_calls": tool_calls},
                {
                    "role": "tool",
                    "tool_call_id": "call_tokyo_w",
                    "name": "get_weather",
                    "content": '{"city": "Tokyo", "temp": 22, "condition": "Clear"}',
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_nvda_s",
                    "name": "get_stock_quote",
                    "content": '{"ticker": "NVDA", "price": 128.50, "currency": "USD"}',
                },
            ]
            t2_resp = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": t2_messages,
                    "tools": [WEATHER_TOOL, STOCK_TOOL],
                },
            )
            assert t2_resp.status_code == 200
            t2_data = t2_resp.json()
            assert t2_data["choices"][0]["finish_reason"] == "stop"
            content = t2_data["choices"][0]["message"]["content"]
            assert "Tokyo weather is 22C" in content
            assert "128.50" in content

            # Verify upstream request preservation
            assert len(captured_requests) == 2
            assert len(captured_requests[1]["messages"]) == 4
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_foundry_e2e_hermes_cycle_openai_stream():
    """Verify complete multi-turn Hermes cycle on OpenAI protocol (Streaming)."""
    t1_stream_payload = (
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_w1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\": \\"Tokyo\\"}"}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_s1","type":"function","function":{"name":"get_stock_quote","arguments":""}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\\"ticker\\": \\"NVDA\\"}"}}]},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":25,"completion_tokens":35,"total_tokens":60}}\n\n'
        b"data: [DONE]\n\n"
    )
    t2_stream_payload = (
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Tokyo is 22C "},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"content":"and NVDA is $128.50."},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":70,"completion_tokens":15,"total_tokens":85}}\n\n'
        b"data: [DONE]\n\n"
    )

    call_count = 0

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=t1_stream_payload)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=t2_stream_payload)

    client = FoundryChatClient(base_url="https://foundry.example.com/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_openai_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            # Turn 1: Streaming parallel tool calls
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Check weather and stock"}],
                    "tools": [WEATHER_TOOL, STOCK_TOOL],
                },
            ) as r1:
                lines1 = list(r1.iter_lines())

            events1 = [json.loads(l[6:]) for l in lines1 if l.startswith("data: ") and l.strip() != "data: [DONE]"]
            tool_call_chunks = [ev["choices"][0]["delta"]["tool_calls"] for ev in events1 if "tool_calls" in ev["choices"][0]["delta"]]
            assert len(tool_call_chunks) >= 2
            assert events1[-1]["choices"][0]["finish_reason"] == "tool_calls"

            # Turn 2: Streaming synthesis
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": [
                        {"role": "user", "content": "Check weather and stock"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {"id": "call_w1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'}},
                                {"id": "call_s1", "type": "function", "function": {"name": "get_stock_quote", "arguments": '{"ticker": "NVDA"}'}},
                            ],
                        },
                        {"role": "tool", "tool_call_id": "call_w1", "content": "22C clear"},
                        {"role": "tool", "tool_call_id": "call_s1", "content": "$128.50"},
                    ],
                    "tools": [WEATHER_TOOL, STOCK_TOOL],
                },
            ) as r2:
                lines2 = list(r2.iter_lines())

            events2 = [json.loads(l[6:]) for l in lines2 if l.startswith("data: ") and l.strip() != "data: [DONE]"]
            text_chunks = [
                ev["choices"][0]["delta"]["content"]
                for ev in events2
                if "content" in ev["choices"][0]["delta"] and ev["choices"][0]["delta"]["content"]
            ]
            full_text = "".join(text_chunks)
            assert "Tokyo is 22C" in full_text
            assert "NVDA is $128.50" in full_text
            assert events2[-1]["choices"][0]["finish_reason"] == "stop"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_foundry_e2e_hermes_cycle_anthropic_non_stream():
    """Verify complete multi-turn Hermes cycle on Anthropic protocol (Non-streaming).
    Verifies automatic tool result merging into single user turn and strict turn alternation.
    """
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/anthropic/v1/messages")
        body = json.loads(request.content)
        captured_requests.append(body)

        if len(captured_requests) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "msg_anthropic_t1",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_weather_01", "name": "get_weather", "input": {"city": "Tokyo"}},
                        {"type": "tool_use", "id": "toolu_stock_01", "name": "get_stock_quote", "input": {"ticker": "NVDA"}},
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 20, "output_tokens": 30},
                },
            )

        return httpx.Response(
            200,
            json={
                "id": "msg_anthropic_t2",
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Tokyo is currently 22C and clear, while NVDA is at $128.50."}
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 60, "output_tokens": 25},
            },
        )

    client = FoundryChatClient(base_url="https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_anthropic_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            # Turn 1
            r1 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:claude-sonnet-5",
                    "messages": [{"role": "user", "content": "Weather in Tokyo and NVDA quote?"}],
                    "tools": [WEATHER_TOOL, STOCK_TOOL],
                },
            )
            assert r1.status_code == 200
            d1 = r1.json()
            assert d1["choices"][0]["finish_reason"] == "tool_calls"
            tcalls = d1["choices"][0]["message"]["tool_calls"]
            assert len(tcalls) == 2
            assert tcalls[0]["id"] == "toolu_weather_01"
            assert tcalls[1]["id"] == "toolu_stock_01"

            # Turn 2: Consecutive tool messages must be merged for Anthropic
            r2 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:claude-sonnet-5",
                    "messages": [
                        {"role": "user", "content": "Weather in Tokyo and NVDA quote?"},
                        {"role": "assistant", "content": None, "tool_calls": tcalls},
                        {"role": "tool", "tool_call_id": "toolu_weather_01", "name": "get_weather", "content": '{"temp": 22}'},
                        {"role": "tool", "tool_call_id": "toolu_stock_01", "name": "get_stock_quote", "content": '{"price": 128.50}'},
                    ],
                    "tools": [WEATHER_TOOL, STOCK_TOOL],
                },
            )
            assert r2.status_code == 400
            error = r2.json()["error"]
            assert error["type"] == error["code"] == "invalid_request_error"
            assert error["param"] == "messages[2].name"
            assert "anthropic_messages" in error["message"]
            assert len(captured_requests) == 1
            retry = json.loads(r2.request.content)
            for message in retry["messages"]:
                if message.get("role") == "tool":
                    message.pop("name", None)
            r2 = http_client.post("/v1/chat/completions", json=retry)
            assert r2.status_code == 200
            d2 = r2.json()
            assert d2["choices"][0]["finish_reason"] == "stop"
            assert "Tokyo is currently 22C" in d2["choices"][0]["message"]["content"]

            # Verify Anthropic upstream request structure
            upstream_msgs = captured_requests[1]["messages"]
            assert len(upstream_msgs) == 3  # Exactly user -> assistant -> user (merged tool results)
            assert upstream_msgs[1]["role"] == "assistant"
            assert len(upstream_msgs[1]["content"]) == 2
            assert upstream_msgs[2]["role"] == "user"
            assert len(upstream_msgs[2]["content"]) == 2
            assert upstream_msgs[2]["content"][0]["type"] == "tool_result"
            assert upstream_msgs[2]["content"][1]["type"] == "tool_result"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_foundry_e2e_hermes_cycle_anthropic_stream():
    """Verify complete multi-turn Hermes cycle on Anthropic protocol (Streaming)."""
    t1_payload = (
        b'data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"usage":{"input_tokens":15,"output_tokens":0}}}\n\n'
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_w1","name":"get_weather","input":{}}}\n\n'
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"city\\": \\"Tokyo\\"}"}}\n\n'
        b'data: {"type":"content_block_stop","index":0}\n\n'
        b'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":25}}\n\n'
        b'data: {"type":"message_stop"}\n\n'
        b"data: [DONE]\n\n"
    )
    t2_payload = (
        b'data: {"type":"message_start","message":{"id":"msg_2","type":"message","role":"assistant","content":[],"usage":{"input_tokens":40,"output_tokens":0}}}\n\n'
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Tokyo is clear and 22C."}}\n\n'
        b'data: {"type":"content_block_stop","index":0}\n\n'
        b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}\n\n'
        b'data: {"type":"message_stop"}\n\n'
        b"data: [DONE]\n\n"
    )

    call_count = 0

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        assert request.url.path.endswith("/anthropic/v1/messages")
        call_count += 1
        if call_count == 1:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=t1_payload)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=t2_payload)

    client = FoundryChatClient(base_url="https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_anthropic_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            # Turn 1 streaming
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:claude-sonnet-5",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Weather in Tokyo?"}],
                    "tools": [WEATHER_TOOL],
                },
            ) as r1:
                lines1 = list(r1.iter_lines())

            ev1 = [json.loads(l[6:]) for l in lines1 if l.startswith("data: ") and l.strip() != "data: [DONE]"]
            assert ev1[0]["choices"][0]["delta"]["tool_calls"][0]["id"] == "toolu_w1"
            assert ev1[-1]["choices"][0]["finish_reason"] == "tool_calls"

            # Turn 2 streaming
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:claude-sonnet-5",
                    "stream": True,
                    "messages": [
                        {"role": "user", "content": "Weather in Tokyo?"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{"id": "toolu_w1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'}}],
                        },
                        {"role": "tool", "tool_call_id": "toolu_w1", "name": "get_weather", "content": '{"temp": 22}'},
                    ],
                    "tools": [WEATHER_TOOL],
                },
            ) as r2:
                assert r2.status_code == 400
                r2.read()
                error = r2.json()["error"]
                assert error["type"] == error["code"] == "invalid_request_error"
                assert error["param"] == "messages[2].name"
                assert "anthropic_messages" in error["message"]
                assert call_count == 1
                retry = json.loads(r2.request.content)
            for message in retry["messages"]:
                if message.get("role") == "tool":
                    message.pop("name", None)
            with http_client.stream("POST", "/v1/chat/completions", json=retry) as r2:
                assert r2.status_code == 200
                lines2 = list(r2.iter_lines())

            ev2 = [json.loads(l[6:]) for l in lines2 if l.startswith("data: ") and l.strip() != "data: [DONE]"]
            text_chunks = [c["choices"][0]["delta"]["content"] for c in ev2 if "content" in c["choices"][0]["delta"] and c["choices"][0]["delta"]["content"]]
            assert "".join(text_chunks) == "Tokyo is clear and 22C."
            assert ev2[-1]["choices"][0]["finish_reason"] == "stop"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_foundry_e2e_hermes_cycle_xai_non_stream():
    """Verify complete multi-turn Hermes cycle on xAI Responses protocol (Non-streaming).
    Verifies transformation of assistant tool_calls and tool results to function_call/function_call_output items.
    """
    captured_requests: list[dict[str, Any]] = []

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/xai/v1/responses")
        body = json.loads(request.content)
        captured_requests.append(body)

        if len(captured_requests) == 1:
            return httpx.Response(
                200,
                json={
                    "id": "resp_xai_t1",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "id": "call_x_w1",
                            "call_id": "call_x_w1",
                            "name": "get_weather",
                            "arguments": '{"city": "Tokyo"}',
                        },
                        {
                            "type": "function_call",
                            "id": "call_x_s1",
                            "call_id": "call_x_s1",
                            "name": "get_stock_quote",
                            "arguments": '{"ticker": "NVDA"}',
                        },
                    ],
                    "usage": {"input_tokens": 18, "output_tokens": 28, "total_tokens": 46},
                },
            )

        return httpx.Response(
            200,
            json={
                "id": "resp_xai_t2",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Tokyo is 22C and NVDA is trading at $128.50."}
                        ],
                    }
                ],
                "usage": {"input_tokens": 55, "output_tokens": 20, "total_tokens": 75},
            },
        )

    client = FoundryChatClient(base_url="https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_xai_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            # Turn 1
            r1 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": [{"role": "user", "content": "Weather and stock"}],
                    "tools": [WEATHER_TOOL, STOCK_TOOL],
                },
            )
            assert r1.status_code == 200
            d1 = r1.json()
            assert d1["choices"][0]["finish_reason"] == "tool_calls"
            tcalls = d1["choices"][0]["message"]["tool_calls"]
            assert len(tcalls) == 2
            assert tcalls[0]["id"] == "call_x_w1"
            assert tcalls[1]["id"] == "call_x_s1"

            # Turn 2
            r2 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "messages": [
                        {"role": "user", "content": "Weather and stock"},
                        {"role": "assistant", "content": None, "tool_calls": tcalls},
                        {"role": "tool", "tool_call_id": "call_x_w1", "name": "get_weather", "content": '{"temp": 22}'},
                        {"role": "tool", "tool_call_id": "call_x_s1", "name": "get_stock_quote", "content": '{"price": 128.50}'},
                    ],
                    "tools": [WEATHER_TOOL, STOCK_TOOL],
                },
            )
            assert r2.status_code == 400
            error = r2.json()["error"]
            assert error["type"] == error["code"] == "invalid_request_error"
            assert error["param"] == "messages[2].name"
            assert "xai_responses" in error["message"]
            assert len(captured_requests) == 1
            retry = json.loads(r2.request.content)
            for message in retry["messages"]:
                if message.get("role") == "tool":
                    message.pop("name", None)
            r2 = http_client.post("/v1/chat/completions", json=retry)
            assert r2.status_code == 200
            d2 = r2.json()
            assert d2["choices"][0]["finish_reason"] == "stop"
            assert "Tokyo is 22C" in d2["choices"][0]["message"]["content"]

            # Verify xAI input items sequence
            input_items = captured_requests[1]["input"]
            assert len(input_items) == 5
            assert input_items[0]["role"] == "user"
            assert input_items[1]["type"] == "function_call"
            assert input_items[2]["type"] == "function_call"
            assert input_items[3]["type"] == "function_call_output"
            assert input_items[4]["type"] == "function_call_output"
            assert input_items[3]["call_id"] == "call_x_w1"
            assert input_items[4]["call_id"] == "call_x_s1"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_foundry_e2e_hermes_cycle_xai_stream():
    """Verify complete multi-turn Hermes cycle on xAI Responses protocol (Streaming)."""
    t1_payload = (
        b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"call_x_w1","call_id":"call_x_w1","name":"get_weather","arguments":""}}\n\n'
        b'data: {"type":"response.function_call_arguments.delta","call_id":"call_x_w1","delta":"{\\"city\\": \\"Tokyo\\"}"}\n\n'
        b'data: {"type":"response.output_item.done","output_index":0}\n\n'
        b'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":15,"output_tokens":20,"total_tokens":35}}}\n\n'
        b"data: [DONE]\n\n"
    )
    t2_payload = (
        b'data: {"type":"response.output_text.delta","delta":"Tokyo is 22C and sunny."}\n\n'
        b'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":40,"output_tokens":15,"total_tokens":55}}}\n\n'
        b"data: [DONE]\n\n"
    )

    call_count = 0

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        assert request.url.path.endswith("/xai/v1/responses")
        call_count += 1
        if call_count == 1:
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=t1_payload)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=t2_payload)

    client = FoundryChatClient(base_url="https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions", token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(mock_handler))

    old_registry = _register_foundry_xai_alias()
    try:
        app = create_app(
            embedding_client_factory=_DummyProvider,
            chat_client_factory=_DummyProvider,
            rerank_client_factory=_DummyProvider,
            ollama_chat_client_factory=_DummyProvider,
            foundry_chat_client_factory=lambda: client,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as http_client:
            # Turn 1
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Weather in Tokyo?"}],
                    "tools": [WEATHER_TOOL],
                },
            ) as r1:
                lines1 = list(r1.iter_lines())

            ev1 = [json.loads(l[6:]) for l in lines1 if l.startswith("data: ") and l.strip() != "data: [DONE]"]
            assert ev1[0]["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_x_w1"
            assert ev1[-1]["choices"][0]["finish_reason"] == "tool_calls"

            # Turn 2
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "stream": True,
                    "messages": [
                        {"role": "user", "content": "Weather in Tokyo?"},
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [{"id": "call_x_w1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'}}],
                        },
                        {"role": "tool", "tool_call_id": "call_x_w1", "name": "get_weather", "content": '{"temp": 22}'},
                    ],
                    "tools": [WEATHER_TOOL],
                },
            ) as r2:
                assert r2.status_code == 400
                r2.read()
                error = r2.json()["error"]
                assert error["type"] == error["code"] == "invalid_request_error"
                assert error["param"] == "messages[2].name"
                assert "xai_responses" in error["message"]
                assert call_count == 1
                retry = json.loads(r2.request.content)
            for message in retry["messages"]:
                if message.get("role") == "tool":
                    message.pop("name", None)
            with http_client.stream("POST", "/v1/chat/completions", json=retry) as r2:
                assert r2.status_code == 200
                lines2 = list(r2.iter_lines())

            ev2 = [json.loads(l[6:]) for l in lines2 if l.startswith("data: ") and l.strip() != "data: [DONE]"]
            text_chunks = [c["choices"][0]["delta"]["content"] for c in ev2 if "content" in c["choices"][0]["delta"] and c["choices"][0]["delta"]["content"]]
            assert "".join(text_chunks) == "Tokyo is 22C and sunny."
            assert ev2[-1]["choices"][0]["finish_reason"] == "stop"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())



