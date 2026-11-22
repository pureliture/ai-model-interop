from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from openai_compatible_bridge.providers.jetbrains import JetBrainsChatClient
from openai_compatible_bridge.providers.vertex import VertexAPIError


TOOL = {"type": "function", "function": {"name": "echo", "parameters": {"type": "object", "properties": {}}}}


@pytest.mark.parametrize("protocol,requested,expected", [
    ("openai_responses", "none", "none"),
    ("openai_responses", "high", "high"),
    ("openai_responses", "xhigh", "xhigh"),
    ("xai_responses", "low", "low"),
    ("xai_responses", "xhigh", "xhigh"),
])
@pytest.mark.parametrize("stream", [False, True])
def test_reasoning_reaches_responses_with_tools(protocol, requested, expected, stream):
    requests = []
    completed = {"type": "response.completed", "response": {"status": "completed", "output": []}}

    def handler(request):
        requests.append(request)
        if stream:
            return httpx.Response(200, content=("data: " + json.dumps(completed) + "\n\n").encode(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json={"status": "completed", "output": []})

    async def run():
        client = JetBrainsChatClient(base_url="https://example.test", token="synthetic")
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            args = dict(model="native", messages=[{"role": "user", "content": "hi"}],
                        resolved_config={"protocol": protocol}, reasoning_effort=requested,
                        tools=[TOOL], tool_choice="auto")
            if stream:
                _ = [event async for event in client.stream_chat(**args)]
            else:
                await client.generate(**args)
        finally:
            await client.close()

    asyncio.run(run())
    assert len(requests) == 1
    request = requests[0]
    body = json.loads(request.content)
    assert request.url.path == "/v1/responses"
    assert request.headers["x-llm-model"] == ("grok" if protocol == "xai_responses" else "openai")
    assert body["reasoning"] == {"effort": expected}
    assert body["stream"] is stream
    assert body["tools"][0]["name"] == "echo"
    assert body["tool_choice"] == "auto"


@pytest.mark.parametrize("protocol,effort,code", [
    ("anthropic_messages", "minimal", "unsupported_parameter"),
    ("anthropic_messages", "none", "unsupported_parameter"),
    ("google_generate_content", "max", "unsupported_parameter"),
    ("google_generate_content", "none", "unsupported_parameter"),
    ("xai_responses", "max", "unsupported_parameter"),
    ("xai_responses", "none", "unsupported_parameter"),
    ("xai_responses", "minimal", "unsupported_parameter"),
    ("openai_responses", "invalid", "invalid_request_error"),
    ("xai_responses", "invalid", "invalid_request_error"),
])
@pytest.mark.parametrize("stream", [False, True])
def test_unverified_or_invalid_reasoning_is_rejected_before_http(protocol, effort, code, stream):
    def handler(request):
        pytest.fail("reasoning must be rejected before the upstream request")

    async def run():
        client = JetBrainsChatClient(base_url="https://example.test", token="synthetic")
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            args = dict(model="native", messages=[], resolved_config={"protocol": protocol}, reasoning_effort=effort)
            with pytest.raises(VertexAPIError) as exc:
                if stream:
                    _ = [event async for event in client.stream_chat(**args)]
                else:
                    await client.generate(**args)
            assert exc.value.code == code
        finally:
            await client.close()

    asyncio.run(run())


def test_anthropic_reasoning_body_includes_effort_and_thinking_blocks():
    messages = [
        {"role": "assistant", "content": None,
         "thinking_blocks": [{"type": "thinking", "thinking": "", "signature": "SIG"}],
         "tool_calls": [{"id": "call_1", "type": "function",
                          "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]
    body = JetBrainsChatClient._build_request_body(
        JetBrainsChatClient(base_url="https://example.test", token="synthetic"),
        protocol="anthropic_messages", model="claude-test", messages=messages,
        max_tokens=4096, temperature=None, top_p=None, stop=None,
        response_format=None, reasoning_effort="high", tools=[TOOL], stream=False,
    )
    assert body["output_config"] == {"effort": "high"}
    assert body["thinking"] == {"type": "adaptive"}
    assert body["messages"][0]["content"][0] == {
        "type": "thinking", "thinking": "", "signature": "SIG",
    }
    assert body["messages"][0]["content"][1]["type"] == "tool_use"


def test_google_reasoning_body_includes_level_and_signature():
    messages = [{"role": "assistant", "content": None, "tool_calls": [{
        "id": "call_1", "type": "function",
        "function": {"name": "lookup", "arguments": "{}"},
        "extra_content": {"google": {"thought_signature": "SIG"}},
    }]}]
    body = JetBrainsChatClient._build_request_body(
        JetBrainsChatClient(base_url="https://example.test", token="synthetic"),
        protocol="google_generate_content", model="gemini-test", messages=messages,
        max_tokens=64, temperature=None, top_p=None, stop=None,
        response_format=None, reasoning_effort="high", tools=[TOOL], stream=False,
    )
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "HIGH"}
    part = body["contents"][0]["parts"][0]
    assert part["thoughtSignature"] == "SIG"
    assert part["functionCall"]["name"] == "lookup"


def test_anthropic_generate_returns_thinking_blocks():
    async def run():
        client = JetBrainsChatClient(base_url="https://example.test", token="synthetic")
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "SIG"},
                    {"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}},
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            })
        ))
        try:
            return await client.generate(model="claude-test", messages=[],
                                         resolved_config={"protocol": "anthropic_messages"})
        finally:
            await client.close()
    result = asyncio.run(run())
    assert result["thinking_blocks"] == [{"type": "thinking", "thinking": "", "signature": "SIG"}]
    assert result["tool_calls"][0]["id"] == "call_1"


def test_google_generate_returns_thought_signature_extra_content():
    async def run():
        client = JetBrainsChatClient(base_url="https://example.test", token="synthetic")
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"candidates": [{
                "content": {"parts": [{
                    "functionCall": {"name": "lookup", "args": {}, "id": "call_1"},
                    "thoughtSignature": "SIG",
                }]},
                "finishReason": "STOP",
            }], "usageMetadata": {}})
        ))
        try:
            return await client.generate(model="gemini-test", messages=[],
                                         resolved_config={"protocol": "google_generate_content"})
        finally:
            await client.close()
    result = asyncio.run(run())
    assert result["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "SIG"}}


def test_google_tool_schema_normalizes_union_types():
    from openai_compatible_bridge.core.transforms import _google_tools

    declaration = _google_tools([{"type": "function", "function": {
        "name": "grep",
        "parameters": {"type": "object", "additionalProperties": False, "properties": {
            "skip": {"type": ["number", "null"]},
            "mode": {"type": ["string", "integer"]},
        }},
    }}])[0]["functionDeclarations"][0]
    schema = declaration["parameters"]
    assert schema["properties"]["skip"] == {"type": "number", "nullable": True}
    assert schema["properties"]["mode"]["anyOf"] == [{"type": "string"}, {"type": "integer"}]
    assert "additionalProperties" not in schema


def _run_jetbrains(handler, *, protocol, stream, messages, model="native") -> Any:
    async def run():
        client = JetBrainsChatClient(base_url="https://example.test", token="synthetic")
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            args = dict(
                model=model,
                messages=messages,
                resolved_config={"protocol": protocol},
            )
            if stream:
                return [event async for event in client.stream_chat(**args)]
            return await client.generate(**args)
        finally:
            await client.close()

    return asyncio.run(run())


def test_include_reasoning_encrypted_content_only_for_openai_responses() -> None:
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"status": "completed", "output": []})

    _run_jetbrains(
        handler,
        protocol="openai_responses",
        stream=False,
        messages=[{"role": "user", "content": "hi"}],
    )
    body = json.loads(requests[0].content)
    assert body["include"] == ["reasoning.encrypted_content"]


def test_reasoning_item_capture_stream_and_non_stream() -> None:
    reasoning = {
        "id": "rs_1",
        "type": "reasoning",
        "status": "completed",
        "summary": [],
        "encrypted_content": "enc_full",
    }
    payload = {
        "status": "completed",
        "output": [
            reasoning,
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}],
            },
        ],
    }

    def handler(request):
        return httpx.Response(200, json=payload)

    result = _run_jetbrains(
        handler,
        protocol="openai_responses",
        stream=False,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert result["reasoning_items"] == [
        {"type": "reasoning", "id": "rs_1", "summary": [], "status": "completed",
         "encrypted_content": "enc_full"}
    ]

    events = [
        {"type": "response.output_item.done", "item": reasoning},
        {"type": "response.output_text.delta", "delta": "hi"},
        {"type": "response.completed", "response": {"status": "completed", "output": []}},
    ]

    def stream_handler(request):
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})

    streamed = _run_jetbrains(
        stream_handler,
        protocol="openai_responses",
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
    )
    items = [item for event in streamed for item in (event.get("reasoning_items") or [])]
    assert items == [
        {"type": "reasoning", "id": "rs_1", "summary": [], "status": "completed",
         "encrypted_content": "enc_full"}
    ]


def test_reasoning_item_replayed_before_assistant_turn() -> None:
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"status": "completed", "output": []})

    _run_jetbrains(
        handler,
        protocol="openai_responses",
        stream=False,
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "hello",
                "thinking_blocks": [
                    {"type": "reasoning_item", "id": "rs_1", "encrypted_content": "enc_1", "summary": []}
                ],
            },
            {"role": "user", "content": "again"},
        ],
    )
    body = json.loads(requests[0].content)
    input_items = body["input"]
    reasoning_index = next(i for i, item in enumerate(input_items) if item.get("type") == "reasoning")
    assistant_index = next(i for i, item in enumerate(input_items) if item.get("role") == "assistant")
    assert reasoning_index < assistant_index
    assert input_items[reasoning_index] == {
        "type": "reasoning",
        "id": "rs_1",
        "encrypted_content": "enc_1",
        "summary": [],
    }


def test_reasoning_replay_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("BRIDGE_REASONING_REPLAY", "0")
    reasoning = {
        "id": "rs_off",
        "type": "reasoning",
        "status": "completed",
        "summary": [],
        "encrypted_content": "enc_off",
    }
    events = [
        {"type": "response.output_item.done", "item": reasoning},
        {"type": "response.completed", "response": {"status": "completed", "output": []}},
    ]

    def stream_handler(request):
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})

    streamed = _run_jetbrains(
        stream_handler,
        protocol="openai_responses",
        stream=True,
        messages=[{"role": "user", "content": "hi"}],
    )
    assert [item for event in streamed for item in (event.get("reasoning_items") or [])] == []

    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"status": "completed", "output": []})

    _run_jetbrains(
        handler,
        protocol="openai_responses",
        stream=False,
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "hello",
                "thinking_blocks": [
                    {"type": "reasoning_item", "id": "rs_1", "encrypted_content": "enc_1", "summary": []}
                ],
            },
        ],
    )
    body = json.loads(requests[0].content)
    assert all(item.get("type") != "reasoning" for item in body["input"])
