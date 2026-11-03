from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import openai_compatible_bridge.providers.vertex as vertex
from openai_compatible_bridge.main import create_app
from openai_compatible_bridge.providers.jetbrains import JetBrainsChatClient
from openai_compatible_bridge.providers.vertex import VertexAPIError


class _FakeProvider:
    async def close(self) -> None:
        pass


class _FakeJetBrains(_FakeProvider):
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "text": "hello from JetBrains",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }

    async def stream_chat(self, **kwargs):
        self.calls.append(kwargs)
        yield {"delta_text": "hello", "finish_reason": None, "usage": None}
        yield {
            "delta_text": " from JetBrains",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
        }


def _register_jetbrains_alias(
    *,
    protocol: str = "openai_responses",
    provider_model: str = "gpt-5.4",
) -> dict[str, dict]:
    old_registry = vertex.MODEL_REGISTRY.copy()
    vertex.MODEL_REGISTRY["jetbrains:openai-gpt-5-4"] = {
        "provider": "jetbrains",
        "kind": "chat",
        "provider_model": provider_model,
        "protocol": protocol,
        "llm_model_header": "openai",
    }
    return old_registry


def _restore_registry(old_registry: dict[str, dict]) -> None:
    vertex.MODEL_REGISTRY.clear()
    vertex.MODEL_REGISTRY.update(old_registry)


def test_jetbrains_alias_routes_non_stream() -> None:
    old_registry = _register_jetbrains_alias()
    try:
        fake = _FakeJetBrains()
        app = create_app(
            embedding_client_factory=_FakeProvider,
            chat_client_factory=_FakeProvider,
            rerank_client_factory=_FakeProvider,
            ollama_chat_client_factory=_FakeProvider,
            foundry_chat_client_factory=_FakeProvider,
            jetbrains_chat_client_factory=lambda: fake,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "jetbrains:openai-gpt-5-4",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert response.status_code == 200
            assert response.json()["choices"][0]["message"]["content"] == "hello from JetBrains"
            assert fake.calls
            assert fake.calls[0]["model"] == "gpt-5.4"
    finally:
        _restore_registry(old_registry)


def test_jetbrains_alias_streams() -> None:
    old_registry = _register_jetbrains_alias()
    try:
        fake = _FakeJetBrains()
        app = create_app(
            embedding_client_factory=_FakeProvider,
            chat_client_factory=_FakeProvider,
            rerank_client_factory=_FakeProvider,
            ollama_chat_client_factory=_FakeProvider,
            foundry_chat_client_factory=_FakeProvider,
            jetbrains_chat_client_factory=lambda: fake,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "jetbrains:openai-gpt-5-4",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
            assert response.status_code == 200
            assert "hello" in response.text
    finally:
        _restore_registry(old_registry)


def test_jetbrains_dynamic_prefix() -> None:
    old_registry = vertex.MODEL_REGISTRY.copy()
    try:
        fake = _FakeJetBrains()
        app = create_app(
            embedding_client_factory=_FakeProvider,
            chat_client_factory=_FakeProvider,
            rerank_client_factory=_FakeProvider,
            ollama_chat_client_factory=_FakeProvider,
            foundry_chat_client_factory=_FakeProvider,
            jetbrains_chat_client_factory=lambda: fake,
            cost_accounting_factory=lambda: None,
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "jetbrains:gpt-5.4",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert response.status_code == 200
            assert fake.calls
            assert fake.calls[0]["model"] == "gpt-5.4"
    finally:
        vertex.MODEL_REGISTRY.clear()
        vertex.MODEL_REGISTRY.update(old_registry)


def test_jetbrains_protocol_auto_inference() -> None:
    from openai_compatible_bridge.providers.jetbrains import infer_jetbrains_protocol

    assert infer_jetbrains_protocol("claude-sonnet-4-5") == "anthropic_messages"
    assert infer_jetbrains_protocol("jetbrains:claude-opus-4-6") == "anthropic_messages"
    assert infer_jetbrains_protocol("anthropic/claude-3-7-sonnet") == "anthropic_messages"
    assert infer_jetbrains_protocol("gemini-2.5-pro") == "google_generate_content"
    assert infer_jetbrains_protocol("jetbrains:gemini-3.1-pro") == "google_generate_content"
    assert infer_jetbrains_protocol("grok-3") == "xai_responses"
    assert infer_jetbrains_protocol("jetbrains:grok-4.6") == "xai_responses"
    assert infer_jetbrains_protocol("gpt-5.4") == "openai_responses"
    assert infer_jetbrains_protocol("o3-mini") == "openai_responses"

    client = JetBrainsChatClient(base_url="https://example.test", token="tok")
    try:
        assert client._protocol({"provider_model": "claude-sonnet-4-5"}) == "anthropic_messages"
        assert client._protocol({"provider_model": "gemini-2.5-pro"}) == "google_generate_content"
        assert client._protocol({"provider_model": "grok-3"}) == "xai_responses"
        assert client._protocol({"provider_model": "gpt-5.4"}) == "openai_responses"
        assert client._protocol({"provider_model": "claude-sonnet-4-5", "protocol": "openai_responses"}) == "openai_responses"
    finally:
        asyncio.run(client.close())


def test_jetbrains_grazie_headers_auth_modes_and_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    from openai_compatible_bridge.providers.jetbrains import _grazie_headers

    default_headers = _grazie_headers("anthropic", "tok-123")
    assert default_headers["Authorization"] == "Bearer tok-123"
    assert "Grazie-Authenticate-JWT" not in default_headers
    assert default_headers["anthropic-version"] == "2023-06-01"

    monkeypatch.setenv("JETBRAINS_AI_AUTH_MODE", "jwt")
    monkeypatch.setenv("JETBRAINS_GRAZIE_AGENT", '{"name":"codex:bridge","version":"1.0"}')
    jwt_headers = _grazie_headers("google", "jwt-xyz")
    assert jwt_headers["Grazie-Authenticate-JWT"] == "jwt-xyz"
    assert "Authorization" not in jwt_headers
    assert jwt_headers["Grazie-Agent"] == '{"name":"codex:bridge","version":"1.0"}'
    assert jwt_headers["X-Free-Google-Api"] == "true"


def test_jetbrains_anthropic_consecutive_turn_merging_and_is_error() -> None:
    body = JetBrainsChatClient._build_anthropic_request_body(
        model="claude-sonnet-4-5",
        messages=[
            {"role": "system", "content": "System 1"},
            {"role": "system", "content": "System 2"},
            {"role": "developer", "content": "Dev instruction"},
            {"role": "user", "content": "User prompt"},
            {
                "role": "assistant",
                "content": "Running command",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "shell", "arguments": '{"cmd":"ls"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "permission denied", "is_error": True},
            {"role": "user", "content": "Please retry"},
        ],
        max_tokens=1024,
        temperature=0.2,
        top_p=None,
        stop=None,
        stream=False,
        tools=[{"type": "function", "name": "shell", "description": "run", "parameters": {"type": "object"}}],
        parallel_tool_calls=False,
    )
    assert body["system"] == "System 1\n\nSystem 2"
    # Consecutive developer + user merge into a single user turn, and tool + user merge into a single user turn
    assert len(body["messages"]) == 3
    assert body["messages"][0]["role"] == "user"
    assert "Dev instruction\n\nUser prompt" == body["messages"][0]["content"]
    assert body["messages"][1]["role"] == "assistant"
    assert body["messages"][2]["role"] == "user"
    assert body["messages"][2]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "call_1",
        "content": "permission denied",
        "is_error": True,
    }
    assert body["messages"][2]["content"][1] == {"type": "text", "text": "Please retry"}
    # Flat tool preserved
    assert body["tools"][0]["name"] == "shell"
    assert body["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}


def test_jetbrains_reasoning_effort_mapping_and_rejection() -> None:
    # Anthropic thinking mapping + max_tokens adjustment + temperature stripped
    anth_body = JetBrainsChatClient._build_anthropic_request_body(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "think hard"}],
        max_tokens=1024,
        temperature=0.3,
        top_p=0.9,
        stop=None,
        stream=False,
        reasoning_effort="high",
    )
    assert anth_body["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert anth_body["max_tokens"] == 16384 + 4096
    assert "temperature" not in anth_body
    assert "top_p" not in anth_body

    # OpenAI responses mapping
    oai_body = JetBrainsChatClient._build_openai_request_body(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=512,
        temperature=None,
        top_p=None,
        stop=None,
        response_format=None,
        reasoning_effort="max",
        protocol="openai_responses",
        stream=False,
    )
    assert oai_body["reasoning"] == {"effort": "high"}

    # xAI responses clamping
    xai_body = JetBrainsChatClient._build_openai_request_body(
        model="grok-3",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=512,
        temperature=None,
        top_p=None,
        stop=None,
        response_format=None,
        reasoning_effort="max",
        protocol="xai_responses",
        stream=False,
    )
    assert xai_body["reasoning"] == {"effort": "xhigh"}

    # Google thinkingConfig mapping
    google_body = JetBrainsChatClient._build_google_request_body(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=512,
        temperature=None,
        top_p=None,
        stop=None,
        response_format=None,
        reasoning_effort="medium",
    )
    assert google_body["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 4096}

    # Unmappable effort rejected with 400
    with pytest.raises(VertexAPIError) as exc:
        JetBrainsChatClient._build_anthropic_request_body(
            model="claude-sonnet-4-5",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1024,
            temperature=None,
            top_p=None,
            stop=None,
            stream=False,
            reasoning_effort="ultracode",
        )
    assert exc.value.status_code == 400
    assert exc.value.code == "unsupported_parameter"


def test_jetbrains_pricing_fallback_and_error_mapping() -> None:
    from openai_compatible_bridge.core.provider_adapter import pricing_fallbacks

    assert pricing_fallbacks("jetbrains:claude-sonnet-4-5") == ("jetbrains:*",)
    assert pricing_fallbacks("jetbrains:*") == ()

    client = JetBrainsChatClient(base_url="https://example.test", token="tok")
    try:
        resp = httpx.Response(401, json={"error": {"message": "bad jwt", "code": "unauthorized"}})
        err = client.map_http_error(resp)
        assert err.status_code == 401
        assert err.message == "bad jwt"
        assert err.code == "unauthorized"
    finally:
        asyncio.run(client.close())


def test_conversation_state_cache_ttl_and_lru_eviction() -> None:
    from openai_compatible_bridge.core.conversation_state import ConversationStateCache

    now = [100.0]
    cache = ConversationStateCache(maxsize=2, ttl_seconds=10.0, timer=lambda: now[0])
    cache.save_response("resp_1", messages=[{"role": "user", "content": "one"}], custom_tool_names={"apply_patch"})
    cache.save_response("resp_2", messages=[{"role": "user", "content": "two"}], custom_tool_names=set())
    assert cache.get_response("resp_1") == {
        "messages": [{"role": "user", "content": "one"}],
        "custom_tool_names": {"apply_patch"},
    }
    # Adding 3rd entry evicts LRU ("resp_2" since "resp_1" was just accessed)
    cache.save_response("resp_3", messages=[{"role": "user", "content": "three"}], custom_tool_names=set())
    assert cache.get_response("resp_2") is None
    assert cache.get_response("resp_1") is not None
    assert cache.get_response("resp_3") is not None

    # Advance timer past TTL
    now[0] += 15.0
    assert cache.get_response("resp_1") is None
    assert cache.get_response("resp_3") is None


def test_gemini_thought_signature_cached_and_restored_across_stateless_turns() -> None:
    from openai_compatible_bridge.core.conversation_state import ConversationStateCache
    from openai_compatible_bridge.providers.vertex import VertexChatClient

    cache = ConversationStateCache(maxsize=16, ttl_seconds=60)
    captured_requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        captured_requests.append(body)
        if len(captured_requests) == 1:
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {"name": "read_file", "args": {"path": "README.md"}},
                                        "thoughtSignature": "sig-gemini-turn-1",
                                    }
                                ]
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 4, "totalTokenCount": 9},
                },
            )
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "File contents read."}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
            },
        )

    async def run() -> None:
        client = JetBrainsChatClient(base_url="https://example.test", token="tok", state_cache=cache)
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            turn1 = await client.generate(
                model="gemini-2.5-pro",
                messages=[{"role": "user", "content": "read README.md"}],
            )
            assert turn1["finish_reason"] == "tool_calls"
            call_id = turn1["tool_calls"][0]["id"]
            assert cache.get_google_call_part(call_id) is not None
            assert cache.get_google_call_part(call_id)["thoughtSignature"] == "sig-gemini-turn-1"

            # Stateless turn 2: client sends back standard OpenAI tool_calls without _google_part
            turn2 = await client.generate(
                model="gemini-2.5-pro",
                messages=[
                    {"role": "user", "content": "read README.md"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": call_id, "content": "# Title"},
                ],
            )
            assert turn2["text"] == "File contents read."
            model_turn = next(c for c in captured_requests[1]["contents"] if c["role"] == "model")
            assert model_turn["parts"][0]["thoughtSignature"] == "sig-gemini-turn-1"
            assert model_turn["parts"][0]["functionCall"] == {"name": "read_file", "args": {"path": "README.md"}}
        finally:
            await client.close()

    asyncio.run(run())

    # Also verify VertexChatClient._build_request_body restores from cache
    vertex_body = VertexChatClient._build_request_body(
        messages=[
            {"role": "user", "content": "read README.md"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_read_file_0",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_read_file_0", "content": "# Title"},
        ],
        max_tokens=128,
        temperature=None,
        top_p=None,
        stop=None,
        state_cache=cache,
    )
    vertex_model_turn = next(c for c in vertex_body["contents"] if c["role"] == "model")
    assert vertex_model_turn["parts"][0]["thoughtSignature"] == "sig-gemini-turn-1"
