"""Adversarial stress-testing suite for Milestone 5 (Slice 5).

Focus Areas:
1. Palantir Foundry error unwrapping mechanism:
   - `LanguageModelService:LlmHttpClientError` unwrapping
   - `parameters.responseBody` with deeply nested JSON, Anthropic errors, primitive JSON types
   - `Optional[...]` corrupted / malformed / nested / empty / non-string formats
   - Missing fields fallback (`errorMessage`, `errorCode`, missing all parameters)
   - Non-dict / empty / raw HTML error responses
2. Mid-stream upstream error handling and SSE event emission:
   - OpenAI protocol mid-stream error event -> SSE error chunk + terminal `[DONE]`
   - Anthropic protocol mid-stream error event -> SSE error chunk + terminal `[DONE]`
   - xAI Responses protocol mid-stream error event -> SSE error chunk + terminal `[DONE]`
   - Mid-stream Palantir-wrapped error in SSE chunk -> unwrapped SSE error chunk + terminal `[DONE]`
   - Mid-stream corrupted SSE chunk tolerance (garbage JSON, numbers, null, raw strings)
3. Multi-turn Hermes cycle error resilience:
   - Turn 1 success (tool calls emitted) -> Turn 2 upstream failure (500 LlmHttpClientError)
   - Turn 2 Anthropic & xAI protocol failure handling and session isolation
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
from openai_compatible_bridge.core.transforms import _openai_error_from_payload, _unwrap_optional
from openai_compatible_bridge.providers.foundry import (
    FOUNDRY_ANTHROPIC_PROTOCOL,
    FOUNDRY_OPENAI_PROTOCOL,
    FOUNDRY_XAI_RESPONSES_PROTOCOL,
    FoundryChatClient,
    _parse_foundry_error,
)
from openai_compatible_bridge.providers.vertex import VertexAPIError

FOUNDRY_TEST_BASE_URL = "https://foundry.example.com/api/v2/llm/proxy/openai/v1/chat/completions"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

STOCK_TOOL = {
    "type": "function",
    "function": {
        "name": "get_stock_quote",
        "description": "Get latest stock price quote for ticker",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
            },
            "required": ["ticker"],
        },
    },
}


class _DummyProvider:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def close(self) -> None:
        pass


def _register_foundry_openai_alias():
    old = dict(vertex.MODEL_REGISTRY)
    vertex.MODEL_REGISTRY["foundry:gpt-6-astra"] = {
        "provider": "foundry",
        "kind": "chat",
        "provider_model": "gpt-6-astra",
        "protocol": FOUNDRY_OPENAI_PROTOCOL,
    }
    return old


def _register_foundry_anthropic_alias():
    old = dict(vertex.MODEL_REGISTRY)
    vertex.MODEL_REGISTRY["foundry:claude-sonnet-5"] = {
        "provider": "foundry",
        "kind": "chat",
        "provider_model": "claude-sonnet-5",
        "protocol": FOUNDRY_ANTHROPIC_PROTOCOL,
    }
    return old


def _register_foundry_xai_alias():
    old = dict(vertex.MODEL_REGISTRY)
    vertex.MODEL_REGISTRY["foundry:grok-4.6"] = {
        "provider": "foundry",
        "kind": "chat",
        "provider_model": "grok-4.6",
        "protocol": FOUNDRY_XAI_RESPONSES_PROTOCOL,
    }
    return old


def _restore_registry(old: dict[str, Any]) -> None:
    vertex.MODEL_REGISTRY.clear()
    vertex.MODEL_REGISTRY.update(old)


# ==============================================================================
# Dimension 1: Palantir Foundry Error Unwrapping Extreme Edge Cases
# ==============================================================================


def test_adv_error_unwrap_optional_variants():
    """Verify _unwrap_optional handles all edge cases of Optional[...] wrapping."""
    # Normal unwrapping
    assert _unwrap_optional("Optional[value]") == "value"
    assert _unwrap_optional("Optional[  spaces  ]") == "spaces"
    assert _unwrap_optional("  Optional[trimmed]  ") == "trimmed"

    # Empty optional variants
    assert _unwrap_optional("Optional[]") is None
    assert _unwrap_optional("Optional[   ]") is None
    assert _unwrap_optional("") is None
    assert _unwrap_optional("   ") is None
    assert _unwrap_optional(None) is None

    # Corrupted / unclosed / unopened brackets
    assert _unwrap_optional("Optional[unclosed") == "Optional[unclosed"
    assert _unwrap_optional("unopened]") == "unopened]"
    assert _unwrap_optional("[bracketed]") == "[bracketed]"
    assert _unwrap_optional("Optional") == "Optional"

    # Non-string types coerced safely
    assert _unwrap_optional(12345) == "12345"
    assert _unwrap_optional(True) == "True"
    assert _unwrap_optional(False) == "False"
    assert _unwrap_optional(["list"]) == "['list']"
    assert _unwrap_optional({"key": "val"}) == "{'key': 'val'}"

    # Nested Optional (single unwrap per contract)
    assert _unwrap_optional("Optional[Optional[inner]]") == "Optional[inner]"


def test_adv_error_unwrapping_deeply_nested_json():
    """Verify recursive unwrapping when parameters.responseBody contains multi-level nested errors."""
    # 3-level nesting: Palantir wrapper -> Palantir nested wrapper -> OpenAI standard error
    deep_openai_error = {
        "error": {
            "message": "Deep token budget exhausted",
            "type": "tokens",
            "code": "context_length_exceeded",
        }
    }
    level_2 = {
        "errorCode": "CUSTOM_CLIENT",
        "errorName": "LanguageModelService:LlmHttpClientError",
        "parameters": {
            "responseBody": f"Optional[{json.dumps(deep_openai_error)}]",
        },
    }
    level_1 = {
        "errorCode": "CUSTOM_CLIENT",
        "errorName": "LanguageModelService:LlmHttpClientError",
        "parameters": {
            "responseBody": f"Optional[{json.dumps(level_2)}]",
        },
    }

    result = _openai_error_from_payload(level_1)
    assert result is not None
    msg, code = result
    assert msg == "Deep token budget exhausted"
    assert code == "context_length_exceeded"


def test_adv_error_unwrapping_anthropic_error_format():
    """Verify parameters.responseBody containing Anthropic error schema is correctly extracted."""
    anthropic_error_body = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "prompt is too long: 250000 tokens > 200000 maximum",
        },
    }
    palantir_payload = {
        "errorCode": "CUSTOM_CLIENT",
        "errorName": "LanguageModelService:LlmHttpClientError",
        "parameters": {
            "responseBody": f"Optional[{json.dumps(anthropic_error_body)}]",
        },
    }

    result = _openai_error_from_payload(palantir_payload)
    assert result is not None
    msg, code = result
    assert msg == "prompt is too long: 250000 tokens > 200000 maximum"
    assert code == "invalid_request_error"


def test_adv_error_unwrapping_primitive_response_bodies():
    """Verify parameters.responseBody containing primitive JSON (int, bool, list, null, empty dict)
    falls back cleanly to parameters.errorMessage without crashing."""
    primitives = [
        ("12345", "Fallback msg 1"),
        ("true", "Fallback msg 2"),
        ('"raw_json_string"', "Fallback msg 3"),
        ("null", "Fallback msg 4"),
        ("[]", "Fallback msg 5"),
        ("{}", "Fallback msg 6"),
        ('{"foo": "bar", "non_error": 123}', "Fallback msg 7"),
    ]

    for raw_body, fallback_msg in primitives:
        payload = {
            "errorCode": "CUSTOM_CLIENT",
            "errorName": "LanguageModelService:LlmHttpClientError",
            "parameters": {
                "responseBody": f"Optional[{raw_body}]",
                "errorMessage": f"Optional[{fallback_msg}]",
                "errorCode": "Optional[fallback_code]",
            },
        }
        result = _openai_error_from_payload(payload)
        assert result is not None, f"Failed for raw_body: {raw_body}"
        msg, code = result
        assert msg == fallback_msg
        assert code == "fallback_code"


def test_adv_error_unwrapping_missing_all_parameters():
    """Verify payload with missing parameters dict or empty parameters falls back to status code."""
    # 1. No parameters at all
    p1 = {"errorCode": "CUSTOM_CLIENT", "errorName": "Default:Internal"}
    assert _openai_error_from_payload(p1) is None

    resp1 = httpx.Response(500, json=p1, request=httpx.Request("POST", "https://foundry.example.com"))
    err1 = _parse_foundry_error(resp1)
    assert err1.status_code == 500
    assert err1.message == "Foundry request failed."
    assert err1.code == "500"

    # 2. Empty parameters dict
    p2 = {"errorCode": "CUSTOM_CLIENT", "parameters": {}}
    assert _openai_error_from_payload(p2) is None

    # 3. Non-dict parameters
    p3 = {"errorCode": "CUSTOM_CLIENT", "parameters": "string_parameters"}
    assert _openai_error_from_payload(p3) is None


def test_adv_error_unwrapping_corrupted_response_body_and_broken_json():
    """Verify corrupted JSON syntax inside responseBody falls back gracefully."""
    payload = {
        "errorCode": "CUSTOM_CLIENT",
        "errorName": "LanguageModelService:LlmHttpClientError",
        "parameters": {
            "responseBody": 'Optional[{"error": {"message": "broken", "code": unquoted}}]',
            "errorMessage": "Safe fallback message",
            "errorCode": "fallback_err",
        },
    }
    result = _openai_error_from_payload(payload)
    assert result is not None
    msg, code = result
    assert msg == "Safe fallback message"
    assert code == "fallback_err"


def test_adv_error_unwrapping_non_dict_error_field():
    """Verify error field being a string, list, or number does not crash _openai_error_from_payload."""
    # error is string
    p1 = {"error": "Some simple error string"}
    assert _openai_error_from_payload(p1) is None

    # error is list
    p2 = {"error": ["err1", "err2"]}
    assert _openai_error_from_payload(p2) is None

    # error.message is non-string (e.g. dict or list or int)
    p3 = {"error": {"message": {"detail": "Structured failure", "subcode": 42}, "code": 400}}
    res3 = _openai_error_from_payload(p3)
    assert res3 is not None
    assert "Structured failure" in res3[0]
    assert res3[1] == "400"


def test_adv_error_unwrapping_empty_response_text_and_html():
    """Verify empty response body and HTML pages parsed via _parse_foundry_error."""
    # 1. Empty body
    r_empty = httpx.Response(502, text="", request=httpx.Request("POST", "https://foundry.example.com"))
    err_empty = _parse_foundry_error(r_empty)
    assert err_empty.status_code == 502
    assert err_empty.message == "Foundry request failed."
    assert err_empty.code == "502"

    # 2. HTML body
    r_html = httpx.Response(503, text="<html><body>503 Service Unavailable</body></html>", request=httpx.Request("POST", "https://foundry.example.com"))
    err_html = _parse_foundry_error(r_html)
    assert err_html.status_code == 503
    assert "503 Service Unavailable" in err_html.message
    assert err_html.code is None


# ==============================================================================
# Dimension 2: Mid-Stream Upstream Error Handling and SSE Emission
# ==============================================================================


def test_adv_stream_midstream_error_openai_protocol():
    """Verify that when an upstream OpenAI SSE stream yields normal chunks and then
    an error event mid-stream, the bridge emits an SSE error chunk followed immediately by terminal [DONE]."""
    # Stream sends 1 text delta, then an upstream error payload, then more chunks (which should not be processed)
    stream_payload = (
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Computing answer..."}}]}\n\n'
        b'data: {"error":{"message":"Engine quota exceeded mid-stream","type":"insufficient_quota","code":"quota_exceeded"}}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"content":"Unreached text"}}]}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Compute something"}],
                },
            ) as response:
                lines = list(response.iter_lines())

            assert response.status_code == 200

            # Filter non-empty data lines
            data_lines = [l for l in lines if l.startswith("data: ")]
            assert len(data_lines) >= 3

            # Event 1: Assistant text chunk
            e1 = json.loads(data_lines[0][len("data: ") :])
            assert e1["choices"][0]["delta"]["role"] == "assistant"
            assert e1["choices"][0]["delta"]["content"] == "Computing answer..."

            # Event 2: SSE error event
            e2 = json.loads(data_lines[1][len("data: ") :])
            assert "error" in e2
            assert e2["error"]["message"] == "Engine quota exceeded mid-stream"
            assert e2["error"]["code"] == "quota_exceeded"
            assert e2["error"]["type"] == "api_error"

            # Event 3: Terminal [DONE]
            assert data_lines[2].strip() == "data: [DONE]"
            # No trailing events after [DONE]
            assert len(data_lines) == 3
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_adv_stream_midstream_error_anthropic_protocol():
    """Verify that when Anthropic upstream emits an error event mid-stream
    (after initial tool_use block), the bridge emits an SSE error chunk and terminal [DONE]."""
    stream_payload = (
        b'data: {"type":"message_start","message":{"id":"msg_mid_err","type":"message","role":"assistant","content":[],"usage":{"input_tokens":10,"output_tokens":0}}}\n\n'
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_mid_01","name":"get_weather","input":{}}}\n\n'
        b'data: {"type":"error","error":{"type":"overloaded_error","message":"Claude cluster temporarily overloaded"}}\n\n'
        b'data: {"type":"content_block_stop","index":0}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:claude-sonnet-5",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Weather query"}],
                    "tools": [WEATHER_TOOL],
                },
            ) as response:
                lines = list(response.iter_lines())

            assert response.status_code == 200
            data_lines = [l for l in lines if l.startswith("data: ")]

            # First event: Assistant tool_calls start
            e1 = json.loads(data_lines[0][len("data: ") :])
            assert e1["choices"][0]["delta"]["role"] == "assistant"
            assert e1["choices"][0]["delta"]["tool_calls"][0]["id"] == "toolu_mid_01"

            # Second event: Error chunk
            e2 = json.loads(data_lines[1][len("data: ") :])
            assert "error" in e2
            assert e2["error"]["message"] == "Claude cluster temporarily overloaded"
            assert e2["error"]["code"] == "overloaded_error"

            # Third event: Terminal [DONE]
            assert data_lines[2].strip() == "data: [DONE]"
            assert len(data_lines) == 3
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_adv_stream_midstream_error_xai_protocol():
    """Verify that when xAI upstream emits response.failed mid-stream,
    the bridge catches the error, emits an SSE error chunk and terminates cleanly with [DONE]."""
    stream_payload = (
        b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"call_mid_x1","name":"get_stock_quote","arguments":""}}\n\n'
        b'data: {"type":"response.failed","error":{"code":"inference_failure","message":"Grok model worker terminated unexpectedly"}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Stock query"}],
                    "tools": [STOCK_TOOL],
                },
            ) as response:
                lines = list(response.iter_lines())

            assert response.status_code == 200
            data_lines = [l for l in lines if l.startswith("data: ")]

            # Event 1: tool call init
            e1 = json.loads(data_lines[0][len("data: ") :])
            assert e1["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_mid_x1"

            # Event 2: SSE error
            e2 = json.loads(data_lines[1][len("data: ") :])
            assert "error" in e2
            assert e2["error"]["message"] == "Grok model worker terminated unexpectedly"
            assert e2["error"]["code"] == "inference_failure"

            # Event 3: [DONE]
            assert data_lines[2].strip() == "data: [DONE]"
            assert len(data_lines) == 3
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_adv_stream_midstream_palantir_wrapped_error_in_sse():
    """Verify that when Palantir Foundry wraps an error payload inside an SSE event chunk,
    the bridge unwraps the nested responseBody and emits an OpenAI standard SSE error event."""
    nested_err = {
        "error": {
            "message": "Palantir upstream backend rate limited",
            "type": "tokens",
            "code": "rate_limit_exceeded",
        }
    }
    palantir_sse_chunk = {
        "errorCode": "CUSTOM_CLIENT",
        "errorName": "LanguageModelService:LlmHttpClientError",
        "parameters": {
            "responseBody": f"Optional[{json.dumps(nested_err)}]",
        },
    }
    stream_payload = (
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Starting..."}}]}\n\n'
        + f"data: {json.dumps(palantir_sse_chunk)}\n\n".encode("utf-8")
        + b"data: [DONE]\n\n"
    )

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            ) as response:
                lines = list(response.iter_lines())

            assert response.status_code == 200
            data_lines = [l for l in lines if l.startswith("data: ")]

            assert len(data_lines) == 3
            # First event: text delta
            assert json.loads(data_lines[0][len("data: ") :])["choices"][0]["delta"]["content"] == "Starting..."

            # Second event: unwrapped error
            err_ev = json.loads(data_lines[1][len("data: ") :])
            assert "error" in err_ev
            assert err_ev["error"]["message"] == "Palantir upstream backend rate limited"
            assert err_ev["error"]["code"] == "rate_limit_exceeded"

            # Third event: [DONE]
            assert data_lines[2].strip() == "data: [DONE]"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_adv_stream_garbage_and_malformed_chunk_tolerance():
    """Verify that corrupt, non-JSON, null, integer, and empty SSE chunks
    do not crash the stream generator and normal chunks proceed to finish_reason and [DONE]."""
    stream_payload = (
        b": keep-alive comment line\n\n"
        b"data: \n\n"
        b"data: null\n\n"
        b"data: 123456\n\n"
        b"data: not a valid json object\n\n"
        b'data: {"invalid_json_missing_brace": \n\n'
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Surviving "}}]}\n\n'
        b"data: [UNEXPECTED_STRING]\n\n"
        b'data: {"choices":[{"index":0,"delta":{"content":"garbage data."}}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Robustness check"}],
                },
            ) as response:
                lines = list(response.iter_lines())

            assert response.status_code == 200
            data_lines = [l for l in lines if l.startswith("data: ")]

            # Must have received surviving and garbage data text chunks
            events = [json.loads(l[6:]) for l in data_lines if l.strip() != "data: [DONE]"]
            text_chunks = [
                ev["choices"][0]["delta"].get("content", "")
                for ev in events
                if "content" in ev["choices"][0]["delta"]
            ]
            full_text = "".join(text_chunks)
            assert full_text == "Surviving garbage data."
            assert events[-1]["choices"][0]["finish_reason"] == "stop"
            assert data_lines[-1].strip() == "data: [DONE]"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


# ==============================================================================
# Dimension 3: Hermes Multi-Turn Cycle Error Injection & Isolation
# ==============================================================================


def test_adv_hermes_multiturn_turn2_upstream_500():
    """Verify multi-turn Hermes cycle where Turn 1 succeeds (tool calls generated),
    but Turn 2 (agent returns tool results) fails with upstream 500 LlmHttpClientError.
    Bridge must return 500 OpenAI standard error and keep server state clean for subsequent requests."""
    captured_calls = 0

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_calls
        captured_calls += 1
        if captured_calls == 1:
            # Turn 1: success with tool calls
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
                                        "id": "call_m5_fail",
                                        "type": "function",
                                        "function": {"name": "get_weather", "arguments": '{"city": "Seoul"}'},
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
                },
            )
        elif captured_calls == 2:
            # Turn 2: Upstream 500 error with Palantir LlmHttpClientError
            return httpx.Response(
                500,
                json={
                    "errorCode": "CUSTOM_CLIENT",
                    "errorName": "LanguageModelService:LlmHttpClientError",
                    "parameters": {
                        "responseBody": 'Optional[{"error": {"message": "Internal worker segfault", "type": "api_error", "code": "internal_worker_error"}}]',
                        "errorMessage": "Worker crash",
                        "errorCode": "worker_crash",
                    },
                },
            )
        else:
            # Turn 3 (recovery): normal success response
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Recovered successfully."},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            )

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
            # Turn 1: Success
            r1 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "Weather in Seoul"}],
                    "tools": [WEATHER_TOOL],
                },
            )
            assert r1.status_code == 200
            tc = r1.json()["choices"][0]["message"]["tool_calls"]
            assert tc[0]["id"] == "call_m5_fail"

            # Turn 2: Upstream failure during tool result resolution
            r2 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [
                        {"role": "user", "content": "Weather in Seoul"},
                        {"role": "assistant", "content": None, "tool_calls": tc},
                        {"role": "tool", "tool_call_id": "call_m5_fail", "content": '{"temp": 18}'},
                    ],
                    "tools": [WEATHER_TOOL],
                },
            )
            assert r2.status_code == 500
            err_data = r2.json()
            assert "error" in err_data
            assert err_data["error"]["message"] == "Internal worker segfault"
            assert err_data["error"]["code"] == "internal_worker_error"
            assert err_data["error"]["type"] == "api_error"

            # Recovery turn: Verify bridge state not corrupted
            r3 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:gpt-6-astra",
                    "messages": [{"role": "user", "content": "Hello again"}],
                },
            )
            assert r3.status_code == 200
            assert r3.json()["choices"][0]["message"]["content"] == "Recovered successfully."
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_adv_hermes_multiturn_turn2_anthropic_400_validation_error():
    """Verify multi-turn Hermes cycle on Anthropic protocol where Turn 2 fails with 400 invalid_request_error."""
    captured_calls = 0

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_calls
        captured_calls += 1
        if captured_calls == 1:
            return httpx.Response(
                200,
                json={
                    "id": "msg_ant_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_a1", "name": "get_weather", "input": {"city": "Busan"}},
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 15, "output_tokens": 20},
                },
            )
        # Turn 2: Anthropic returns 400 error
        return httpx.Response(
            400,
            json={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "messages.1: consecutive tool messages without user wrapping",
                },
            },
        )

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
                    "messages": [{"role": "user", "content": "Busan weather"}],
                    "tools": [WEATHER_TOOL],
                },
            )
            assert r1.status_code == 200
            tc = r1.json()["choices"][0]["message"]["tool_calls"]

            # Turn 2
            r2 = http_client.post(
                "/v1/chat/completions",
                json={
                    "model": "foundry:claude-sonnet-5",
                    "messages": [
                        {"role": "user", "content": "Busan weather"},
                        {"role": "assistant", "content": None, "tool_calls": tc},
                        {"role": "tool", "tool_call_id": "toolu_a1", "content": '{"temp": 20}'},
                    ],
                    "tools": [WEATHER_TOOL],
                },
            )
            assert r2.status_code == 400
            data2 = r2.json()
            assert "error" in data2
            assert "consecutive tool messages" in data2["error"]["message"]
            assert data2["error"]["type"] == "invalid_request_error"
            assert data2["error"]["code"] == "invalid_request_error"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_adv_hermes_multiturn_turn2_xai_rate_limit_stream():
    """Verify multi-turn Hermes cycle on xAI protocol streaming where Turn 2 hits 429 rate limit."""
    captured_calls = 0

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal captured_calls
        captured_calls += 1
        if captured_calls == 1:
            t1_payload = (
                b'data: {"type":"response.output_item.added","item":{"type":"function_call","id":"call_x_r1","name":"get_stock_quote","arguments":""}}\n\n'
                b'data: {"type":"response.completed","response":{"status":"completed"}}\n\n'
                b"data: [DONE]\n\n"
            )
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=t1_payload)
        # Turn 2: upstream 429 rate limit
        return httpx.Response(
            429,
            headers={"content-type": "application/json"},
            json={
                "error": {
                    "message": "xAI rate limit exceeded for model grok-4.6",
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                }
            },
        )

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
                    "messages": [{"role": "user", "content": "Quote"}],
                    "tools": [STOCK_TOOL],
                },
            ) as r1:
                lines1 = list(r1.iter_lines())
            assert r1.status_code == 200

            # Turn 2: stream requested but upstream returns HTTP 429 immediately
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "stream": True,
                    "messages": [
                        {"role": "user", "content": "Quote"},
                        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_x_r1", "type": "function", "function": {"name": "get_stock_quote", "arguments": '{"ticker": "NVDA"}'}}]},
                        {"role": "tool", "tool_call_id": "call_x_r1", "content": "128.5"},
                    ],
                    "tools": [STOCK_TOOL],
                },
            ) as r2:
                lines2 = list(r2.iter_lines())

            assert r2.status_code == 200
            data_lines2 = [l for l in lines2 if l.startswith("data: ")]
            err_event = json.loads(data_lines2[0][len("data: ") :])
            assert "error" in err_event
            assert "rate limit exceeded" in err_event["error"]["message"].lower()
            assert err_event["error"]["code"] == "rate_limit_exceeded"
            assert data_lines2[-1].strip() == "data: [DONE]"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


# ==============================================================================
# Dimension 4: HTTP Status Code Matrix & Protocol Edge Cases
# ==============================================================================


@pytest.mark.parametrize(
    "status_code, expected_type, expected_code",
    [
        (400, "invalid_request_error", "bad_param"),
        (401, "authentication_error", "invalid_token"),
        (403, "permission_error", "forbidden_access"),
        (404, "invalid_request_error", "resource_missing"),
        (429, "rate_limit_error", "rate_limit_hit"),
        (500, "api_error", "internal_foundry_error"),
        (503, "api_error", "cluster_overload"),
    ],
)
def test_adv_error_http_status_code_matrix(status_code: int, expected_type: str, expected_code: str):
    """Verify that every HTTP error status code from Foundry maps to the correct OpenAI error type."""
    palantir_payload = {
        "errorCode": "CUSTOM_CLIENT",
        "errorName": "LanguageModelService:LlmHttpClientError",
        "parameters": {
            "errorMessage": f"Foundry error for HTTP {status_code}",
            "errorCode": expected_code,
        },
    }

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=palantir_payload)

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
                json={"model": "foundry:gpt-6-astra", "messages": [{"role": "user", "content": "test"}]},
            )
            assert resp.status_code == status_code
            data = resp.json()
            assert "error" in data
            assert data["error"]["type"] == expected_type
            assert data["error"]["code"] == expected_code
            assert f"HTTP {status_code}" in data["error"]["message"]
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_adv_stream_anthropic_interleaved_text_and_tool_call_midstream_error():
    """Verify Anthropic stream where model outputs text explanation first,
    begins a tool call, and then encounters an upstream error event."""
    stream_payload = (
        b'data: {"type":"message_start","message":{"id":"msg_int","type":"message","role":"assistant","content":[],"usage":{"input_tokens":20,"output_tokens":0}}}\n\n'
        b'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n'
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"I will check the weather for you."}}\n\n'
        b'data: {"type":"content_block_stop","index":0}\n\n'
        b'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_int_01","name":"get_weather","input":{}}}\n\n'
        b'data: {"type":"error","error":{"type":"rate_limit_error","message":"Anthropic per-minute rate limit hit"}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:claude-sonnet-5",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Weather please"}],
                    "tools": [WEATHER_TOOL],
                },
            ) as response:
                lines = list(response.iter_lines())

            assert response.status_code == 200
            data_lines = [l for l in lines if l.startswith("data: ")]

            # Check sequence:
            # 1. First event: assistant role & text content
            e1 = json.loads(data_lines[0][len("data: ") :])
            assert e1["choices"][0]["delta"]["content"] == "I will check the weather for you."

            # 2. Second event: tool_calls block started
            e2 = json.loads(data_lines[1][len("data: ") :])
            assert e2["choices"][0]["delta"]["tool_calls"][0]["id"] == "toolu_int_01"

            # 3. Third event: upstream error caught and emitted as SSE error
            e3 = json.loads(data_lines[2][len("data: ") :])
            assert "error" in e3
            assert e3["error"]["message"] == "Anthropic per-minute rate limit hit"
            assert e3["error"]["code"] == "rate_limit_error"

            # 4. Final line is [DONE]
            assert data_lines[3].strip() == "data: [DONE]"
            assert len(data_lines) == 4
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_adv_stream_xai_parallel_tool_calls_midstream_failure():
    """Verify xAI stream where two parallel function calls are added, but during arguments
    for the second tool call, the upstream emits response.failed."""
    stream_payload = (
        b'data: {"type":"response.output_item.added","output_index":0,"item":{"type":"function_call","id":"call_p1","name":"get_weather","arguments":""}}\n\n'
        b'data: {"type":"response.function_call_arguments.delta","output_index":0,"delta":"{\\"city\\":\\"Tokyo\\"}"}\n\n'
        b'data: {"type":"response.output_item.added","output_index":1,"item":{"type":"function_call","id":"call_p2","name":"get_stock_quote","arguments":""}}\n\n'
        b'data: {"type":"response.failed","error":{"code":"stream_timeout","message":"xAI upstream inference worker timed out"}}\n\n'
        b"data: [DONE]\n\n"
    )

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_payload)

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
            with http_client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "foundry:grok-4.6",
                    "stream": True,
                    "messages": [{"role": "user", "content": "Weather & stock"}],
                    "tools": [WEATHER_TOOL, STOCK_TOOL],
                },
            ) as response:
                lines = list(response.iter_lines())

            assert response.status_code == 200
            data_lines = [l for l in lines if l.startswith("data: ")]

            # Events before failure
            e1 = json.loads(data_lines[0][len("data: ") :])
            assert e1["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_p1"
            assert e1["choices"][0]["delta"]["tool_calls"][0]["index"] == 0

            # Error event
            err_ev = [
                json.loads(l[6:])
                for l in data_lines
                if l.strip() != "data: [DONE]" and "error" in json.loads(l[6:])
            ][0]
            assert err_ev["error"]["message"] == "xAI upstream inference worker timed out"
            assert err_ev["error"]["code"] == "stream_timeout"

            # Final line is [DONE]
            assert data_lines[-1].strip() == "data: [DONE]"
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())


def test_adv_error_unwrapping_broken_json_arguments_pass_through():
    """Verify that when upstream returns non-JSON or broken JSON in tool arguments,
    the bridge preserves the raw string without crashing, complying with OpenAI spec."""
    broken_args = '{"city": "Tokyo", "invalid_tail": '

    async def mock_handler(_request: httpx.Request) -> httpx.Response:
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
                                    "id": "call_broken_args",
                                    "type": "function",
                                    "function": {"name": "get_weather", "arguments": broken_args},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
            },
        )

    client = FoundryChatClient(base_url=FOUNDRY_TEST_BASE_URL, token="test-token")
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
                    "messages": [{"role": "user", "content": "Weather"}],
                    "tools": [WEATHER_TOOL],
                },
            )
            assert resp.status_code == 200
            tc = resp.json()["choices"][0]["message"]["tool_calls"]
            assert tc[0]["function"]["arguments"] == broken_args
    finally:
        _restore_registry(old_registry)
        asyncio.run(client.close())

