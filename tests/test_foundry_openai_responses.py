"""Tests for the Foundry OpenAI Responses protocol (GPT family) and the xAI effort clamp."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import openai_compatible_bridge.providers.vertex as vertex
from openai_compatible_bridge.main import create_app
from openai_compatible_bridge.core.transforms import _clamp_xai_reasoning_effort
from openai_compatible_bridge.providers.foundry import (
    FoundryChatClient,
)
from openai_compatible_bridge.providers.vertex import VertexAPIError


class _DummyProvider:
    async def close(self) -> None:
        pass


ECHO_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "contract_echo",
        "description": "Echo a value",
        "parameters": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    },
}

FOUNDRY_BASE = "https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions"


def _register_responses_alias(
    model_alias: str = "foundry:gpt-6-astra",
    provider_model: str = "gpt-6-astra",
) -> dict[str, dict]:
    old_registry = vertex.MODEL_REGISTRY.copy()
    vertex.MODEL_REGISTRY[model_alias] = {
        "provider": "foundry",
        "kind": "chat",
        "provider_model": provider_model,
        "protocol": "openai_responses",
    }
    return old_registry


def _restore_registry(old_registry: dict[str, dict]) -> None:
    vertex.MODEL_REGISTRY.clear()
    vertex.MODEL_REGISTRY.update(old_registry)


def _sse(event: dict[str, Any]) -> bytes:
    return ("data: " + json.dumps(event) + "\n\n").encode()


def _responses_tool_payload() -> dict[str, Any]:
    return {
        "id": "resp_1",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "contract_echo",
                "arguments": '{"value": "alpha"}',
            }
        ],
        "usage": {"input_tokens": 15, "output_tokens": 20, "total_tokens": 35},
    }


def test_registry_accepts_openai_responses_protocol(monkeypatch):
    custom = json.dumps(
        {
            "foundry:gpt-6-astra": {
                "provider": "foundry",
                "kind": "chat",
                "provider_model": "gpt-6-astra",
                "protocol": "openai_responses",
            }
        }
    )
    monkeypatch.setenv("MODEL_REGISTRY_JSON", custom)
    registry = vertex._build_registry()
    cfg = registry.get("foundry:gpt-6-astra")
    assert cfg is not None
    assert cfg["protocol"] == "openai_responses"


def test_build_openai_responses_request_body_maps_fields():
    body = FoundryChatClient._build_openai_responses_request_body(
        model="gpt-6-astra",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
        ],
        max_tokens=123,
        stream=False,
        tools=[ECHO_TOOL],
        tool_choice={"type": "function", "function": {"name": "contract_echo"}},
    )
    assert body["model"] == "gpt-6-astra"
    assert body["stream"] is False
    assert body["max_output_tokens"] == 123
    assert "messages" not in body
    assert body["input"][0] == {"role": "system", "content": "sys"}
    assert body["input"][1] == {"role": "user", "content": "hello"}
    assert body["tools"] == [
        {
            "type": "function",
            "name": "contract_echo",
            "description": "Echo a value",
            "parameters": ECHO_TOOL["function"]["parameters"],
        }
    ]
    assert body["tool_choice"] == {"type": "function", "name": "contract_echo"}
    # Sampling/reasoning fields are intentionally omitted on this route.
    assert "temperature" not in body
    assert "reasoning" not in body
    assert "reasoning_effort" not in body


def test_openai_responses_non_stream_tool_call():
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, json=_responses_tool_payload())

    client = FoundryChatClient(base_url=FOUNDRY_BASE, token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        result = await client.generate(
            model="gpt-6-astra",
            messages=[{"role": "user", "content": "Echo alpha"}],
            tools=[ECHO_TOOL],
            tool_choice="auto",
            max_tokens=256,
            resolved_config={"protocol": "openai_responses"},
        )
        await client.close()
        return result

    result = asyncio.run(run())
    assert captured[0].url.path.endswith("/api/v2/llm/proxy/openai/v1/responses")
    upstream_body = json.loads(captured[0].content)
    assert upstream_body["max_output_tokens"] == 256
    assert upstream_body["tools"][0]["name"] == "contract_echo"
    assert upstream_body["tool_choice"] == "auto"
    assert result["text"] is None
    assert result["finish_reason"] == "tool_calls"
    assert result["tool_calls"][0]["id"] == "call_1"
    assert result["tool_calls"][0]["type"] == "function"
    assert result["tool_calls"][0]["function"]["name"] == "contract_echo"
    assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"value": "alpha"}
    assert result["usage"]["total_tokens"] == 35


def test_openai_responses_stream_tool_call():
    stream_payload = b"".join(
        [
            _sse({"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}}),
            _sse(
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "contract_echo",
                        "arguments": "",
                        "status": "in_progress",
                    },
                }
            ),
            _sse(
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_1",
                    "output_index": 0,
                    "delta": '{"value": ',
                }
            ),
            _sse(
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_1",
                    "output_index": 0,
                    "delta": '"alpha"}',
                }
            ),
            _sse({"type": "response.function_call_arguments.done", "item_id": "fc_1", "output_index": 0}),
            _sse({"type": "response.output_item.done", "output_index": 0}),
            _sse(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "status": "completed",
                        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    },
                }
            ),
            b"data: [DONE]\n\n",
        ]
    )

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url=FOUNDRY_BASE, token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        events = []
        async for event in client.stream_chat(
            model="gpt-6-astra",
            messages=[{"role": "user", "content": "Echo alpha"}],
            tools=[ECHO_TOOL],
            tool_choice="auto",
            resolved_config={"protocol": "openai_responses"},
        ):
            events.append(event)
        await client.close()
        return events

    events = asyncio.run(run())
    first_tool_event = next(e for e in events if e.get("delta_tool_calls"))
    assert first_tool_event["delta_tool_calls"][0]["id"] == "call_1"
    assert first_tool_event["delta_tool_calls"][0]["function"]["name"] == "contract_echo"
    arg_chunks = "".join(
        e["delta_tool_calls"][0]["function"].get("arguments", "")
        for e in events
        if e.get("delta_tool_calls")
    )
    assert json.loads(arg_chunks) == {"value": "alpha"}
    last_event = events[-1]
    assert last_event["finish_reason"] == "tool_calls"
    assert last_event["usage"]["total_tokens"] == 15


def test_openai_responses_stream_fallback_finish_reason():
    stream_payload = b"".join(
        [
            _sse(
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "id": "call_ping",
                        "call_id": "call_ping",
                        "name": "ping",
                        "arguments": "",
                    },
                }
            ),
            _sse({"type": "response.function_call_arguments.delta", "item_id": "call_ping", "output_index": 0, "delta": "{}"}),
            _sse({"type": "response.output_item.done", "output_index": 0}),
            b"data: [DONE]\n\n",
        ]
    )

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url=FOUNDRY_BASE, token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        events = []
        async for event in client.stream_chat(
            model="gpt-6-astra",
            messages=[{"role": "user", "content": "ping"}],
            tools=[ECHO_TOOL],
            resolved_config={"protocol": "openai_responses"},
        ):
            events.append(event)
        await client.close()
        return events

    events = asyncio.run(run())
    assert any(e.get("finish_reason") == "tool_calls" for e in events)


def test_openai_responses_stream_text_delta():
    stream_payload = b"".join(
        [
            _sse({"type": "response.output_text.delta", "delta": "OK"}),
            _sse(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_2",
                        "status": "completed",
                        "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                    },
                }
            ),
            b"data: [DONE]\n\n",
        ]
    )

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url=FOUNDRY_BASE, token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        events = []
        async for event in client.stream_chat(
            model="gpt-6-astra",
            messages=[{"role": "user", "content": "hi"}],
            resolved_config={"protocol": "openai_responses"},
        ):
            events.append(event)
        await client.close()
        return events

    events = asyncio.run(run())
    assert "".join(e["delta_text"] for e in events if e.get("delta_text")) == "OK"
    assert events[-1]["finish_reason"] == "stop"


def test_private_responses_connect_timeout_has_safe_stage():
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("SYNTHETIC_PRIVATE_CONNECT_DETAIL")

    client = FoundryChatClient(base_url=FOUNDRY_BASE, token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        try:
            with pytest.raises(VertexAPIError) as caught:
                async for _event in client.stream_chat(
                    model="gpt-6-astra",
                    messages=[{"role": "user", "content": "synthetic"}],
                    resolved_config={"protocol": "openai_responses"},
                    _require_complete=True,
                ):
                    pass
            return caught.value
        finally:
            await client.close()

    error = asyncio.run(run())
    assert error.status_code == 504 and error.code == "timeout"
    assert error.stage == "provider_connect"


def test_private_responses_http_504_has_safe_stage():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(504, json={"error": {"message": "SYNTHETIC_PRIVATE_HTTP_BODY"}})

    client = FoundryChatClient(base_url=FOUNDRY_BASE, token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        try:
            with pytest.raises(VertexAPIError) as caught:
                async for _event in client.stream_chat(
                    model="gpt-6-astra",
                    messages=[{"role": "user", "content": "synthetic"}],
                    resolved_config={"protocol": "openai_responses"},
                    _require_complete=True,
                ):
                    pass
            return caught.value
        finally:
            await client.close()

    error = asyncio.run(run())
    assert error.status_code == 504
    assert error.stage == "provider_http_status"


def test_private_responses_read_timeout_has_safe_stage():
    class ReadTimeoutStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            raise httpx.ReadTimeout("SYNTHETIC_PRIVATE_READ_DETAIL")
            yield b""  # pragma: no cover - makes this an async generator

        async def aclose(self) -> None:
            pass

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=ReadTimeoutStream())

    client = FoundryChatClient(base_url=FOUNDRY_BASE, token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        try:
            with pytest.raises(VertexAPIError) as caught:
                async for _event in client.stream_chat(
                    model="gpt-6-astra",
                    messages=[{"role": "user", "content": "synthetic"}],
                    resolved_config={"protocol": "openai_responses"},
                    _require_complete=True,
                ):
                    pass
            return caught.value
        finally:
            await client.close()

    error = asyncio.run(run())
    assert error.status_code == 504 and error.code == "timeout"
    assert error.stage == "provider_read"


def test_openai_responses_multi_turn_history():
    captured: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "id": "resp_2",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "The tool returned alpha-result."}],
                    }
                ],
                "usage": {"input_tokens": 30, "output_tokens": 10, "total_tokens": 40},
            },
        )

    client = FoundryChatClient(base_url=FOUNDRY_BASE, token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        result = await client.generate(
            model="gpt-6-astra",
            messages=[
                {"role": "user", "content": "q1"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "contract_echo", "arguments": '{"value": "alpha"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_a", "content": "alpha-result"},
                {"role": "user", "content": "summarize"},
            ],
            tools=[ECHO_TOOL],
            resolved_config={"protocol": "openai_responses"},
        )
        await client.close()
        return result

    result = asyncio.run(run())
    items = json.loads(captured[0].content)["input"]
    assert items[0] == {"role": "user", "content": "q1"}
    assert items[1]["type"] == "function_call"
    assert items[1]["call_id"] == "call_a"
    assert items[1]["name"] == "contract_echo"
    assert items[1]["arguments"] == '{"value": "alpha"}'
    assert items[2]["type"] == "function_call_output"
    assert items[2]["call_id"] == "call_a"
    assert items[2]["output"] == "alpha-result"
    assert items[3] == {"role": "user", "content": "summarize"}
    assert result["text"] == "The tool returned alpha-result."
    assert result["finish_reason"] == "stop"
    assert result["tool_calls"] is None


def test_openai_responses_alias_e2e_tool_call():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_responses_tool_payload())

    client2 = FoundryChatClient(base_url=FOUNDRY_BASE, token="test-token")
    client2.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    old_registry = _register_responses_alias()
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
            response = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "Echo alpha"}],
                    "tools": [ECHO_TOOL],
                    "tool_choice": "auto",
                },
            )
        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["finish_reason"] == "tool_calls"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] is None
        assert data["choices"][0]["message"]["tool_calls"][0]["id"] == "call_1"
        assert data["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "contract_echo"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client2.close())


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("minimal", "minimal"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "xhigh"),
        ("max", "xhigh"),
        ("ultra", "xhigh"),
        ("none", None),
        ("bogus", None),
        (None, None),
    ],
)
def test_clamp_xai_reasoning_effort(requested, expected):
    assert _clamp_xai_reasoning_effort(requested) == expected


def _xai_body(reasoning_effort):
    return FoundryChatClient._build_xai_request_body(
        model="grok-4.6",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=64,
        temperature=None,
        top_p=None,
        stop=None,
        response_format=None,
        reasoning_effort=reasoning_effort,
        stream=False,
    )


def test_xai_request_maps_max_to_xhigh_and_omits_none():
    assert _xai_body("max")["reasoning"] == {"effort": "xhigh"}
    assert _xai_body("ultra")["reasoning"] == {"effort": "xhigh"}
    assert "reasoning" not in _xai_body("none")
    assert _xai_body("medium")["reasoning"] == {"effort": "medium"}


def _collect_xai_stream(stream_payload: bytes, *, tools: list[dict[str, Any]] | None = None):
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url=FOUNDRY_BASE, token="test-token")
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def run():
        events = []
        async for event in client.stream_chat(
            model="grok-4.6",
            messages=[{"role": "user", "content": "Echo alpha"}],
            tools=tools if tools is not None else [ECHO_TOOL],
            resolved_config={"protocol": "xai_responses"},
        ):
            events.append(event)
        await client.close()
        return events

    return asyncio.run(run())


def test_xai_stream_function_call_without_added_event_carries_metadata():
    """Foundry xAI streams function calls without response.output_item.added; the
    id/name arrive on response.output_item.done and must be emitted exactly once."""
    stream_payload = b"".join(
        [
            _sse({"type": "response.created", "response": {"id": "resp_1", "status": "in_progress"}}),
            _sse(
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_1",
                    "output_index": 1,
                    "delta": '{"value": "alpha"}',
                }
            ),
            _sse(
                {
                    "type": "response.output_item.done",
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "contract_echo",
                        "arguments": '{"value": "alpha"}',
                        "status": "completed",
                    },
                }
            ),
            _sse(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_1",
                        "status": "completed",
                        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    },
                }
            ),
            b"data: [DONE]\n\n",
        ]
    )

    events = _collect_xai_stream(stream_payload)

    name_chunks = [
        e
        for e in events
        if e.get("delta_tool_calls") and e["delta_tool_calls"][0]["function"].get("name")
    ]
    assert len(name_chunks) == 1
    metadata = name_chunks[0]["delta_tool_calls"][0]
    assert metadata["index"] == 0
    assert metadata["id"] == "call_1"
    assert metadata["function"]["name"] == "contract_echo"
    assert metadata["function"]["arguments"] == ""
    args = "".join(
        e["delta_tool_calls"][0]["function"].get("arguments", "")
        for e in events
        if e.get("delta_tool_calls")
    )
    assert json.loads(args) == {"value": "alpha"}
    assert events[-1]["finish_reason"] == "tool_calls"


def test_xai_stream_arguments_only_in_done_are_emitted():
    stream_payload = b"".join(
        [
            _sse({"type": "response.created", "response": {"id": "resp_2", "status": "in_progress"}}),
            _sse(
                {
                    "type": "response.output_item.done",
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_2",
                        "call_id": "call_2",
                        "name": "contract_echo",
                        "arguments": '{"value": "beta"}',
                        "status": "completed",
                    },
                }
            ),
            _sse(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_2",
                        "status": "completed",
                        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    },
                }
            ),
            b"data: [DONE]\n\n",
        ]
    )

    events = _collect_xai_stream(stream_payload)
    tool_chunks = [e["delta_tool_calls"][0] for e in events if e.get("delta_tool_calls")]
    assert tool_chunks[0]["id"] == "call_2"
    assert tool_chunks[0]["function"]["name"] == "contract_echo"
    assert json.loads(tool_chunks[0]["function"]["arguments"]) == {"value": "beta"}


def test_xai_stream_parallel_function_calls_without_added_events():
    stream_payload = b"".join(
        [
            _sse({"type": "response.created", "response": {"id": "resp_3", "status": "in_progress"}}),
            _sse(
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_1",
                    "output_index": 1,
                    "delta": '{"value": "one"}',
                }
            ),
            _sse(
                {
                    "type": "response.output_item.done",
                    "output_index": 1,
                    "item": {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_a",
                        "name": "contract_one",
                        "arguments": '{"value": "one"}',
                    },
                }
            ),
            _sse(
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": "fc_2",
                    "output_index": 2,
                    "delta": '{"value": "two"}',
                }
            ),
            _sse(
                {
                    "type": "response.output_item.done",
                    "output_index": 2,
                    "item": {
                        "type": "function_call",
                        "id": "fc_2",
                        "call_id": "call_b",
                        "name": "contract_two",
                        "arguments": '{"value": "two"}',
                    },
                }
            ),
            _sse(
                {
                    "type": "response.completed",
                    "response": {
                        "id": "resp_3",
                        "status": "completed",
                        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    },
                }
            ),
            b"data: [DONE]\n\n",
        ]
    )

    events = _collect_xai_stream(stream_payload)
    name_chunks = [
        e["delta_tool_calls"][0]
        for e in events
        if e.get("delta_tool_calls") and e["delta_tool_calls"][0]["function"].get("name")
    ]
    assert [(c["index"], c["id"], c["function"]["name"]) for c in name_chunks] == [
        (0, "call_a", "contract_one"),
        (1, "call_b", "contract_two"),
    ]
