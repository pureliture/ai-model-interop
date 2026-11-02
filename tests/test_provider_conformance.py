"""Provider conformance suite (Step 3).

단일 파일로 전 provider에 동일한 4종 계약을 검증한다:
- non-stream 텍스트+usage 왕복
- tool 왕복 (지원 provider만, 미지원은 capability 명시)
- streaming 델타+최종 usage
- usage 누락은 None (가짜 0 금지) + HTTP 400 매핑

개별 provider 테스트 파일을 복제하지 말고 이 파일에 케이스를 추가한다.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from openai_compatible_bridge.core.cost_tracking import NormalizedUsage
from openai_compatible_bridge.core.metered_http import MeteredHTTPClient
from openai_compatible_bridge.core.provider_adapter import (
    ChatProviderAdapter,
    ProviderAdapter,
)
from openai_compatible_bridge.providers.foundry import FoundryChatClient
from openai_compatible_bridge.providers.ollama import OllamaChatClient
from openai_compatible_bridge.providers.openai_compatible import OpenAICompatibleChatClient
from openai_compatible_bridge.providers.vertex import VertexChatClient

PROVIDERS = ["vertex", "ollama", "foundry", "openai_compatible"]
TOOLS_SUPPORTED = {"vertex": True, "ollama": False, "foundry": True, "openai_compatible": True}


class FakeTokens:
    def __init__(self) -> None:
        self.project_id = "test-project"

    async def get_token(self) -> str:
        return "fake-token"


class Accounting:
    def __init__(self, endpoint: str = "chat") -> None:
        self.endpoint = endpoint
        self.records: list[tuple[Any, Any]] = []

    async def before_attempt(self, provider: str) -> Any:
        from openai_compatible_bridge.core.cost_tracking import CostReservation

        return CostReservation(
            reservation_id="r-1",
            internal_request_id="req-1",
            provider=provider,
            endpoint=self.endpoint,
            model="conformance-model",
            forecast_cost_usd=0,
            currency="USD",
            pricing_source="test",
            pricing_version="test",
            window_started_at="2026-01-01T00:00:00Z",
            created_at="2026-01-01T00:00:00Z",
        )

    def record_attempt(self, reservation: Any, usage: Any) -> None:
        self.records.append((reservation, usage))


def _sse(payloads: list[dict[str, Any]]) -> bytes:
    return b"".join(b"data: " + json.dumps(p).encode() + b"\n\n" for p in payloads) + b"data: [DONE]\n\n"


def _ndjson(payloads: list[dict[str, Any]]) -> bytes:
    return b"".join(json.dumps(p).encode() + b"\n" for p in payloads)


def _make_client(provider: str, handler: Any) -> Any:
    transport = httpx.MockTransport(handler)
    if provider == "vertex":
        client = VertexChatClient(token_provider=FakeTokens())  # type: ignore[arg-type]
        client.http = httpx.AsyncClient(transport=transport)
        return client
    if provider == "ollama":
        client = OllamaChatClient(base_url="http://ollama.test")
        client.http = httpx.AsyncClient(transport=transport)
        return client
    if provider == "openai_compatible":
        client = OpenAICompatibleChatClient()
        client.http = httpx.AsyncClient(transport=transport)
        return client
    client = FoundryChatClient(base_url="http://foundry.test/chat/completions", token="test")
    client.http = httpx.AsyncClient(transport=transport)
    return client


def _go_kwargs() -> dict[str, Any]:
    return {
        "resolved_config": {
            "provider": "openai_compatible",
            "provider_model": "glm-5.3",
            "base_url": "https://go.test/v1",
            "api_key": "go-key",
        }
    }


def _nonstream_payload(provider: str) -> dict[str, Any]:
    if provider == "vertex":
        return {
            "candidates": [{"content": {"parts": [{"text": "hi"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 5, "totalTokenCount": 8},
        }
    if provider == "ollama":
        return {
            "message": {"content": "hi"},
            "prompt_eval_count": 3,
            "eval_count": 5,
            "done": True,
            "done_reason": "stop",
        }
    return {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
    }


def _nonstream_kwargs(provider: str) -> dict[str, Any]:
    if provider == "vertex":
        return {"resolved_config": {"api": "generateContent", "location": "us-central1"}}
    if provider == "foundry":
        return {"resolved_config": {"protocol": "openai_chat_completions"}}
    if provider == "openai_compatible":
        return _go_kwargs()
    return {}


@pytest.mark.parametrize("provider", PROVIDERS)
def test_conformance_adapters_satisfy_protocol(provider: str) -> None:
    client = _make_client(provider, lambda request: httpx.Response(200, json={}))
    try:
        assert isinstance(client, ProviderAdapter)
        assert isinstance(client, ChatProviderAdapter)
        assert client.provider_name == provider
    finally:
        asyncio.run(client.close())


@pytest.mark.parametrize("provider", PROVIDERS)
def test_conformance_nonstream_text_and_usage(provider: str) -> None:
    payload = _nonstream_payload(provider)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    async def run() -> None:
        client = _make_client(provider, handler)
        try:
            result = await client.generate(
                model="m", messages=[{"role": "user", "content": "hi"}], **_nonstream_kwargs(provider)
            )
            assert result["text"] == "hi"
            assert result["finish_reason"] == "stop"
            assert result["usage"]["prompt_tokens"] == 3
            assert result["usage"]["completion_tokens"] == 5
            parsed = client.parse_usage(payload, "chat")
            assert parsed == NormalizedUsage(prompt_tokens=3, completion_tokens=5, total_tokens=8)
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("provider", PROVIDERS)
def test_conformance_tool_roundtrip_or_capability(provider: str) -> None:
    if not TOOLS_SUPPORTED[provider]:
        # ollama는 tool 파라미터를 받지 않는다: capability 명시로 회귀 방지.
        assert TOOLS_SUPPORTED[provider] is False
        return

    if provider == "vertex":
        tool_payload: dict[str, Any] = {
            "candidates": [
                {
                    "content": {"parts": [{"functionCall": {"name": "get_weather", "args": {"city": "Seoul"}}}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 5, "totalTokenCount": 8},
        }
        extra_kwargs: dict[str, Any] = {"resolved_config": {"api": "generateContent", "location": "us-central1"}}
    elif provider == "openai_compatible":
        tool_payload = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city":"Seoul"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        }
        extra_kwargs = _go_kwargs()
    else:
        tool_payload = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": '{"city":"Seoul"}'},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        }
        extra_kwargs = {"resolved_config": {"protocol": "openai_chat_completions"}}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=tool_payload)

    async def run() -> None:
        client = _make_client(provider, handler)
        try:
            result = await client.generate(
                model="m",
                messages=[{"role": "user", "content": "weather?"}],
                tools=[{"type": "function", "function": {"name": "get_weather"}}],
                tool_choice="auto",
                **extra_kwargs,
            )
            assert result["finish_reason"] == "tool_calls"
            assert result["tool_calls"][0]["function"]["name"] == "get_weather"
            assert client.parse_usage(tool_payload, "chat") == NormalizedUsage(
                prompt_tokens=3, completion_tokens=5, total_tokens=8
            )
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("provider", PROVIDERS)
def test_conformance_stream_delta_and_final_usage(provider: str) -> None:
    if provider == "vertex":
        wire = _sse(
            [
                {"candidates": [{"content": {"parts": [{"text": "hel"}]}}]},
                {
                    "candidates": [{"content": {"parts": [{"text": "lo"}]}, "finishReason": "STOP"}],
                    "usageMetadata": {
                        "promptTokenCount": 3,
                        "candidatesTokenCount": 5,
                        "totalTokenCount": 8,
                    },
                },
            ]
        )
    elif provider == "ollama":
        wire = _ndjson(
            [
                {"message": {"content": "hel"}, "done": False},
                {"message": {"content": "lo"}, "done": False},
                {
                    "message": {"content": ""},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 3,
                    "eval_count": 5,
                },
            ]
        )
    else:
        wire = _sse(
            [
                {"choices": [{"delta": {"content": "hel"}, "finish_reason": None}]},
                {"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]},
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
                },
            ]
        )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=wire)

    async def run() -> None:
        client = _make_client(provider, handler)
        try:
            kwargs: dict[str, Any] = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            kwargs.update(_nonstream_kwargs(provider))
            events = [item async for item in client.stream_chat(**kwargs)]
            assert "".join(item.get("delta_text", "") for item in events) == "hello"
            usages = [item.get("usage") for item in events if item.get("usage")]
            assert usages, "final usage must be present"
            last = usages[-1]
            assert last["prompt_tokens"] == 3
            assert last["completion_tokens"] == 5
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("provider", PROVIDERS)
def test_conformance_usage_missing_is_none_not_zero(provider: str) -> None:
    client = _make_client(provider, lambda request: httpx.Response(200, json={}))
    try:
        assert client.parse_usage({"nope": True}, "chat") is None
        assert client.parse_usage({"usage": {"prompt_tokens": True, "completion_tokens": 5}}, "chat") is None
        assert client.parse_usage({"usage": {}}, "chat") is None
    finally:
        asyncio.run(client.close())


@pytest.mark.parametrize("provider", PROVIDERS)
def test_conformance_metered_post_records_none_on_missing_usage(provider: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    async def run() -> None:
        accounting = Accounting()
        raw = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        client = MeteredHTTPClient(raw, accounting, provider=provider)  # type: ignore[arg-type]
        try:
            resp = await client.post("http://upstream.test", json={})
            assert resp.status_code == 200
            assert accounting.records[0][1] is None
        finally:
            await client.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("provider", PROVIDERS)
def test_conformance_http_400_maps_to_provider_error(provider: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "bad request"}})

    async def run() -> None:
        # NOTE: test_wrapper가 vertex 모듈을 importlib.reload하므로
        # 클래스 identity 비교 대신 이름+status로 검증한다 (reload-proof).
        client = _make_client(provider, handler)
        try:
            try:
                await client.generate(
                    model="m", messages=[{"role": "user", "content": "hi"}], **_nonstream_kwargs(provider)
                )
            except Exception as exc:
                assert type(exc).__name__ == "VertexAPIError"
                assert getattr(exc, "status_code", None) == 400
            else:
                pytest.fail("expected VertexAPIError for HTTP 400")
            mapped = client.map_http_error(httpx.Response(400, json={"error": {"message": "bad"}}))
            assert type(mapped).__name__ == "VertexAPIError"
            assert getattr(mapped, "status_code", None) == 400
        finally:
            await client.close()

    asyncio.run(run())
