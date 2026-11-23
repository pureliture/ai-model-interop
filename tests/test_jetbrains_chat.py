from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import openai_compatible_bridge.providers.vertex as vertex
from openai_compatible_bridge.main import create_app
from openai_compatible_bridge.providers.jetbrains import JetBrainsChatClient, validate_jetbrains_parameters
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


def test_jetbrains_drops_replayed_reasoning_content() -> None:
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
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {
                            "role": "assistant",
                            "content": "hello",
                            "reasoning_content": "replayed private reasoning",
                        },
                        {"role": "user", "content": "again"},
                    ],
                },
            )
            assert response.status_code == 200
            assert fake.calls
            assert all("reasoning_content" not in m for m in fake.calls[0]["messages"])
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
        tool_choice="auto",
    )
    assert body["system"] == "System 1\n\nSystem 2"
    # Tool and user messages merge into a single user turn.
    assert len(body["messages"]) == 3
    assert body["messages"][0]["role"] == "user"
    assert "User prompt" == body["messages"][0]["content"]
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
    assert body["tool_choice"] == {"type": "auto"}


def test_jetbrains_reasoning_effort_mapping_and_rejection() -> None:
    # Anthropic passthrough: output_config.effort + adaptive thinking, no budget/max_tokens rewrite.
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
    assert anth_body["output_config"] == {"effort": "high"}
    assert anth_body["thinking"] == {"type": "adaptive"}
    assert anth_body["max_tokens"] == 1024
    assert anth_body["temperature"] == 0.3
    assert anth_body["top_p"] == 0.9

    # OpenAI responses passthrough (no high/max rewrite).
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
    assert oai_body["reasoning"] == {"effort": "max"}

    # xAI responses passthrough (no clamp to xhigh here; validation handles support).
    xai_body = JetBrainsChatClient._build_openai_request_body(
        model="grok-3",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=512,
        temperature=None,
        top_p=None,
        stop=None,
        response_format=None,
        reasoning_effort="high",
        protocol="xai_responses",
        stream=False,
    )
    assert xai_body["reasoning"] == {"effort": "high"}

    # Google passthrough: thinkingLevel upper, no budget.
    google_body = JetBrainsChatClient._build_google_request_body(
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=512,
        temperature=None,
        top_p=None,
        stop=None,
        response_format=None,
        reasoning_effort="medium",
    )
    assert google_body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "MEDIUM"}

    # Unknown effort rejected with invalid_request_error via the validation contract.
    with pytest.raises(VertexAPIError) as exc:
        validate_jetbrains_parameters("anthropic_messages", reasoning_effort="ultracode")
    assert exc.value.status_code == 400
    assert exc.value.code == "invalid_request_error"


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


@pytest.mark.parametrize("protocol,message,field", [
    ("anthropic_messages", {"role": "developer", "content": "x"}, "role"),
    ("google_generate_content", {"role": "developer", "content": "x"}, "role"),
    ("xai_responses", {"role": "unknown", "content": "x"}, "role"),
    ("openai_responses", {"role": "user", "content": "x", "name": "alice"}, "name"),
    ("openai_responses", {"role": "tool", "content": "x"}, "tool_call_id"),
    ("openai_responses", {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]}, "image_url"),
    ("openai_responses", {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ==", "detail": "high"}}]}, "detail"),
])
def test_jetbrains_invalid_messages_fail_before_upstream(protocol, message, field):
    old_registry = _register_jetbrains_alias(protocol=protocol)
    fake = _FakeJetBrains()
    app = create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider,
                     rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider,
                     foundry_chat_client_factory=_FakeProvider, jetbrains_chat_client_factory=lambda: fake,
                     cost_accounting_factory=lambda: None)
    try:
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", json={"model": "jetbrains:openai-gpt-5-4", "messages": [message], "stream": True})
        assert response.status_code == 400
        error = response.json()["error"]
        assert field in error["param"]
        assert protocol in error["message"]
        assert error["code"] == "invalid_request_error"
        assert not fake.calls
    finally:
        _restore_registry(old_registry)


def test_responses_developer_rejected_before_stream_for_anthropic():
    old = _register_jetbrains_alias(protocol="anthropic_messages")
    fake = _FakeJetBrains()
    app = create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider,
                     rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider,
                     foundry_chat_client_factory=_FakeProvider, jetbrains_chat_client_factory=lambda: fake,
                     cost_accounting_factory=lambda: None)
    try:
        with TestClient(app) as client:
            response = client.post("/v1/responses", json={"model": "jetbrains:openai-gpt-5-4", "input": [{"role": "developer", "content": "rule"}], "stream": True})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request_error"
        assert response.json()["error"]["param"] == "messages[0].role"
        assert "anthropic_messages" in response.json()["error"]["message"]
        assert not fake.calls
    finally:
        _restore_registry(old)


def test_anthropic_unrepresentable_field_fails_before_upstream():
    fake = _FakeJetBrains()
    app = create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider,
                     rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider,
                     foundry_chat_client_factory=_FakeProvider, jetbrains_chat_client_factory=lambda: fake,
                     cost_accounting_factory=lambda: None)
    with TestClient(app) as client:
        response = client.post("/v1/messages", json={"model": "jetbrains:gpt-5.4", "max_tokens": 10, "messages": [{"role": "user", "content": "hi", "name": "alice"}], "stream": True})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "messages[0].name" in response.json()["error"]["message"]
    assert not fake.calls


def test_base64_image_transforms_preserve_data():
    from openai_compatible_bridge.core.transforms import _google_contents_and_system, _openai_input_messages
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}}]}]
    assert _openai_input_messages(messages)[0]["content"][0] == {"type": "input_image", "image_url": "data:image/png;base64,YQ=="}
    assert _google_contents_and_system(messages)[0][0]["parts"][0] == {"inlineData": {"mimeType": "image/png", "data": "YQ=="}}


@pytest.mark.parametrize("provider,protocol,message,field", [
    ("vertex", "google_generate_content", {"role": "developer", "content": "x"}, "role"),
    ("vertex", "google_generate_content", {"role": "user", "content": "x", "name": "n"}, "name"),
    ("foundry", "anthropic_messages", {"role": "developer", "content": "x"}, "role"),
    ("foundry", "xai_responses", {"role": "user", "content": "x", "name": "n"}, "name"),
    ("foundry", "openai_chat_completions", {"role": "unknown", "content": "x"}, "role"),
    ("openai_compatible", "openai_chat_completions", {"role": "unknown", "content": "x"}, "role"),
    ("ollama", "ollama_chat", {"role": "developer", "content": "x"}, "role"),
    ("ollama", "ollama_chat", {"role": "tool", "tool_call_id": "id", "content": "x"}, "tool_call_id"),
    ("foundry", "google_generate_content", {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.org/a"}}]}, "image_url"),
    ("openai_compatible", "openai_chat_completions", {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ==", "detail": "auto"}}]}, "detail"),
])
def test_other_providers_invalid_messages_fail_before_stream(provider, protocol, message, field):
    old = vertex.MODEL_REGISTRY.copy()
    alias = "guard-test"
    cfg = {"provider": provider, "kind": "chat", "provider_model": "model"}
    if provider == "foundry":
        cfg["protocol"] = protocol
    if provider == "vertex":
        cfg.update({"api": "generateContent", "location": "global", "vertex_model": "model"})
    vertex.MODEL_REGISTRY[alias] = cfg
    fake = _FakeJetBrains()
    app = create_app(embedding_client_factory=_FakeProvider, chat_client_factory=lambda: fake,
                     rerank_client_factory=_FakeProvider, ollama_chat_client_factory=lambda: fake,
                     foundry_chat_client_factory=lambda: fake, jetbrains_chat_client_factory=_FakeProvider,
                     openai_compatible_chat_client_factory=lambda: fake, cost_accounting_factory=lambda: None)
    try:
        with TestClient(app) as client:
            response = client.post("/v1/chat/completions", json={"model": alias, "messages": [message], "stream": True})
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert field in error["param"]
        assert protocol in error["message"]
        assert error["code"] == "invalid_request_error"
        assert not fake.calls
    finally:
        vertex.MODEL_REGISTRY.clear()
        vertex.MODEL_REGISTRY.update(old)


def test_non_jetbrains_image_builders_preserve_base64():
    from openai_compatible_bridge.providers.foundry import FoundryChatClient
    from openai_compatible_bridge.providers.vertex import VertexChatClient
    messages = [{"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}}]}]
    anthropic = FoundryChatClient._build_anthropic_request_body(model="m", messages=messages, max_tokens=8, stop=None, stream=False)
    assert anthropic["messages"][0]["content"][1]["source"]["data"] == "YQ=="
    google = VertexChatClient._build_request_body(messages=messages, max_tokens=8, temperature=None, top_p=None, stop=None)
    assert google["contents"][0]["parts"][1] == {"inlineData": {"mimeType": "image/png", "data": "YQ=="}}


@pytest.mark.parametrize("endpoint,payload,field", [
    ("/v1/responses", {"input": [{"role": "developer", "content": "rules"}]}, "role"),
    ("/v1/responses", {"input": [{"role": "user", "content": [{"type": "input_image", "file_id": "file-1"}]}]}, "file_id"),
    ("/v1/messages", {"max_tokens": 16, "messages": [{"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "https://example.org/a"}}]}]}, "source"),
])
def test_non_jetbrains_ingresses_reject_before_stream(endpoint, payload, field):
    old = vertex.MODEL_REGISTRY.copy()
    vertex.MODEL_REGISTRY["guard-test"] = {"provider": "foundry", "kind": "chat", "provider_model": "m", "protocol": "anthropic_messages"}
    fake = _FakeJetBrains()
    app = create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider,
                     rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider,
                     foundry_chat_client_factory=lambda: fake, jetbrains_chat_client_factory=_FakeProvider,
                     cost_accounting_factory=lambda: None)
    try:
        with TestClient(app) as client:
            response = client.post(endpoint, json={"model": "guard-test", "stream": True, **payload})
        assert response.status_code == 400
        error = response.json()["error"]
        assert field in error["message"]
        assert error.get("code", error.get("type")) == "invalid_request_error"
        if endpoint == "/v1/responses":
            assert field in error["param"]
        assert not fake.calls
    finally:
        vertex.MODEL_REGISTRY.clear()
        vertex.MODEL_REGISTRY.update(old)
