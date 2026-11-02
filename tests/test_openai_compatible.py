"""Generic OpenAI upstream provider (openai_compatible) tests."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import openai_compatible_bridge.providers.vertex as vertex_mod
from openai_compatible_bridge.main import create_app
from openai_compatible_bridge.providers.openai_compatible import OpenAICompatibleChatClient
from openai_compatible_bridge.providers.vertex import VertexAPIError


def _client(handler, *, env: dict[str, str] | None = None) -> OpenAICompatibleChatClient:
    if env:
        import os

        for key, value in env.items():
            os.environ[key] = value
    client = OpenAICompatibleChatClient()
    client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def test_generate_posts_to_alias_base_url_with_bearer_key():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "go ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10},
            },
        )

    async def run():
        client = _client(handler)
        try:
            return await client.generate(
                model="glm-5.3",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=16,
                resolved_config={
                    "provider": "openai_compatible",
                    "provider_model": "glm-5.3",
                    "base_url": "https://opencode.ai/inference/go/openai/v1/",
                    "api_key": "go-key",
                },
            )
        finally:
            await client.close()

    result = asyncio.run(run())
    assert seen["url"] == "https://opencode.ai/inference/go/openai/v1/chat/completions"
    assert seen["auth"] == "Bearer go-key"
    assert seen["body"]["model"] == "glm-5.3"
    assert seen["body"]["stream"] is False
    assert result["text"] == "go ok"
    assert result["usage"] == {"prompt_tokens": 4, "completion_tokens": 6, "total_tokens": 10}


def test_generate_reads_key_from_env_and_parses_tools(monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "env-key")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {"id": "call_1", "type": "function",
                                 "function": {"name": "bash", "arguments": '{"cmd":"ls"}'}}
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
            },
        )

    async def run():
        client = _client(handler)
        try:
            return await client.generate(
                model="glm-5.3",
                messages=[{"role": "user", "content": "ls"}],
                tools=[{"type": "function", "function": {"name": "bash"}}],
                tool_choice="auto",
                resolved_config={
                    "provider": "openai_compatible",
                    "provider_model": "glm-5.3",
                    "base_url": "https://go.test/v1",
                    "api_key_env": "OPENCODE_GO_API_KEY",
                },
            )
        finally:
            await client.close()

    result = asyncio.run(run())
    assert seen["auth"] == "Bearer env-key"
    assert result["finish_reason"] == "tool_calls"
    assert result["tool_calls"][0]["function"]["name"] == "bash"


def test_missing_base_url_and_bad_scheme_fail_closed():
    async def run():
        client = _client(lambda request: httpx.Response(200, json={}))
        try:
            with pytest.raises(VertexAPIError) as excinfo:
                await client.generate(model="m", messages=[], resolved_config={"provider": "openai_compatible"})
            assert excinfo.value.status_code == 503
            with pytest.raises(VertexAPIError):
                await client.generate(
                    model="m", messages=[],
                    resolved_config={"provider": "openai_compatible", "base_url": "ftp://x"},
                )
        finally:
            await client.close()

    asyncio.run(run())


def test_error_shape_unwrapped():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"error": {"message": "unknown model", "type": "invalid_request_error", "code": "model_not_found"}}
        )

    async def run():
        client = _client(handler)
        try:
            with pytest.raises(VertexAPIError) as excinfo:
                await client.generate(
                    model="nope", messages=[{"role": "user", "content": "x"}],
                    resolved_config={"provider": "openai_compatible", "base_url": "https://go.test/v1"},
                )
            assert excinfo.value.status_code == 400
            assert excinfo.value.code == "model_not_found"
        finally:
            await client.close()

    asyncio.run(run())


def test_stream_parses_deltas_and_usage():
    wire = (
        b'data: {"choices": [{"delta": {"content": "he"}}]}\n\n'
        b'data: {"choices": [{"delta": {"content": "llo"}, "finish_reason": "stop"}], '
        b'"usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}}\n\n'
        b'data: [DONE]\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=wire)

    async def run():
        client = _client(handler)
        try:
            return [
                event async for event in client.stream_chat(
                    model="glm-5.3",
                    messages=[{"role": "user", "content": "hi"}],
                    resolved_config={"provider": "openai_compatible", "base_url": "https://go.test/v1"},
                )
            ]
        finally:
            await client.close()

    events = asyncio.run(run())
    assert "".join(e["delta_text"] for e in events) == "hello"
    assert events[-1]["finish_reason"] == "stop"
    assert events[-1]["usage"]["total_tokens"] == 3


def _register_go_alias() -> dict:
    old = vertex_mod.MODEL_REGISTRY.copy()
    vertex_mod.MODEL_REGISTRY["go-test:glm"] = {
        "provider": "openai_compatible",
        "kind": "chat",
        "provider_model": "glm-5.3",
        "base_url": "https://go.test/v1",
        "api_key": "go-key",
    }
    return old


def _restore(old: dict) -> None:
    vertex_mod.MODEL_REGISTRY.clear()
    vertex_mod.MODEL_REGISTRY.update(old)


def _route_app(fake):
    class Fake:
        async def close(self): ...

    return create_app(
        embedding_client_factory=lambda: Fake(),
        chat_client_factory=lambda: Fake(),
        rerank_client_factory=lambda: Fake(),
        ollama_chat_client_factory=lambda: Fake(),
        foundry_chat_client_factory=lambda: Fake(),
        openai_compatible_chat_client_factory=lambda: fake,
        cost_accounting_factory=lambda: None,
    )


def test_route_chat_and_messages_through_go_alias():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://go.test/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "go hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            },
        )

    fake = _client(handler)
    old = _register_go_alias()
    try:
        app = _route_app(fake)
        with TestClient(app) as client:
            chat = client.post(
                "/v1/chat/completions",
                json={"model": "go-test:glm", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert chat.status_code == 200
            assert chat.json()["choices"][0]["message"]["content"] == "go hi"
            msg = client.post(
                "/v1/messages",
                json={"model": "go-test:glm", "max_tokens": 16,
                       "messages": [{"role": "user", "content": "hi"}]},
            )
            assert msg.status_code == 200
            assert msg.json()["content"] == [{"type": "text", "text": "go hi"}]
            models = client.get("/v1/models")
            entry = next(m for m in models.json()["data"] if m["id"] == "go-test:glm")
            assert entry["capabilities"]["tools"] is True
    finally:
        _restore(old)
        asyncio.run(fake.close())


def test_registry_rejects_openai_compatible_without_base_url(monkeypatch):
    import json as jsonlib

    monkeypatch.setenv(
        "MODEL_REGISTRY_JSON",
        jsonlib.dumps({"bad:alias": {"provider": "openai_compatible", "kind": "chat"}}),
    )
    with pytest.raises(ValueError, match="base_url"):
        vertex_mod._build_registry()


def test_dynamic_compat_namespace_needs_no_registry(monkeypatch):
    import openai_compatible_bridge.main as bridge_main

    monkeypatch.setattr(bridge_main, "OPENAI_COMPAT_DEFAULT_BASE_URL", "https://go.test/v1")
    monkeypatch.setattr(bridge_main, "OPENAI_COMPAT_DEFAULT_API_KEY", "go-key")

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://go.test/v1/chat/completions"
        body = json.loads(request.content.decode())
        assert body["model"] == "glm-5.3"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "dyn ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    fake = _client(handler)
    app = _route_app(fake)
    try:
        with TestClient(app) as client:
            chat = client.post(
                "/v1/chat/completions",
                json={"model": "compat:glm-5.3", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert chat.status_code == 200
            assert chat.json()["choices"][0]["message"]["content"] == "dyn ok"
            msg = client.post(
                "/v1/messages",
                json={"model": "compat:glm-5.3", "max_tokens": 8,
                       "messages": [{"role": "user", "content": "hi"}]},
            )
            assert msg.status_code == 200
            assert msg.json()["content"] == [{"type": "text", "text": "dyn ok"}]
            empty = client.post(
                "/v1/messages",
                json={"model": "compat:", "max_tokens": 8,
                       "messages": [{"role": "user", "content": "hi"}]},
            )
            assert empty.status_code == 400
    finally:
        asyncio.run(fake.close())


def test_dynamic_compat_without_base_url_rejected(monkeypatch):
    import openai_compatible_bridge.main as bridge_main

    monkeypatch.setattr(bridge_main, "OPENAI_COMPAT_DEFAULT_BASE_URL", "")
    monkeypatch.delenv("OPENAI_COMPAT_DEFAULT_BASE_URL", raising=False)
    app = _route_app(_client(lambda request: httpx.Response(200, json={})))
    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "compat:glm-5.3", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 400


def test_pricing_compat_wildcard_fallback():
    from openai_compatible_bridge.core.cost_tracking import CostEstimator, PricingCatalog

    catalog = PricingCatalog.from_json(
        json.dumps(
            {
                "source": "unit-test",
                "version": "2026-06-23",
                "currency": "USD",
                "models": {"compat:*": {"chat": {"input_per_million": "1.00", "output_per_million": "2.00"}}},
            }
        )
    )
    from openai_compatible_bridge.core.cost_tracking import NormalizedUsage

    cost = CostEstimator(catalog).estimate(
        model="compat:glm-5.3",
        endpoint="chat",
        usage=NormalizedUsage(prompt_tokens=1000, completion_tokens=1000),
    )
    assert str(cost) == "0.003"



def test_compat_base_config_merges_template_and_env(monkeypatch):
    import openai_compatible_bridge.main as bridge_main
    from openai_compatible_bridge.providers.openai_compatible import _resolve_extra_headers

    monkeypatch.setattr(bridge_main, "MODEL_REGISTRY", {
        "compat:*": {
            "base_url": "https://go.test/v1",
            "api_key_env": "GO_KEY",
            "extra_headers": {"x-opencode-session": "sid"},
            "context_length": 65536,
        }
    })
    monkeypatch.setenv("GO_KEY", "secret-key")
    cfg = bridge_main._openai_compatible_base_config()
    assert cfg["base_url"] == "https://go.test/v1"
    assert cfg["api_key"] == "secret-key"
    assert cfg["extra_headers"] == {"x-opencode-session": "sid"}
    assert cfg["context_length"] == 65536
    assert _resolve_extra_headers(cfg) == {"x-opencode-session": "sid"}


def test_compat_model_config_and_allowed_models_use_discovered_ids():
    import openai_compatible_bridge.main as bridge_main

    old_ids = set(bridge_main._COMPAT_MODEL_IDS)
    old_meta = dict(bridge_main._COMPAT_MODEL_META)
    bridge_main._COMPAT_MODEL_IDS.clear()
    bridge_main._COMPAT_MODEL_META.clear()
    bridge_main._COMPAT_MODEL_META["glm-5.3"] = {"id": "glm-5.3", "context_length": 131072}
    bridge_main._COMPAT_MODEL_IDS.add("compat:glm-5.3")
    try:
        cfg = bridge_main.model_config("compat:glm-5.3")
        assert cfg is not None
        assert cfg["provider"] == "openai_compatible"
        assert cfg["provider_model"] == "glm-5.3"
        assert cfg["context_length"] == 131072
        obj = bridge_main._model_object("compat:glm-5.3")
        assert obj["id"] == "compat:glm-5.3"
        assert obj["context_length"] == 131072
        assert obj["capabilities"]["tools"] is True
        assert "compat:glm-5.3" in bridge_main.allowed_models()
    finally:
        bridge_main._COMPAT_MODEL_IDS.clear()
        bridge_main._COMPAT_MODEL_IDS.update(old_ids)
        bridge_main._COMPAT_MODEL_META.clear()
        bridge_main._COMPAT_MODEL_META.update(old_meta)


def _tier_registry() -> dict:
    return {
        "compat:*": {
            "kind": "chat",
            "provider": "openai_compatible",
            "base_url": "https://go.test/v1",
        },
        "compat:opus-model": {
            "display_name": "Opus 5 (DeepSeek)",
            "anthropic_family_tier": "opus",
            "is_family_default": True,
            "max_tokens": 64000,
        },
        "gemini-2.5-flash": {"provider": "vertex"},
    }


def test_model_list_reports_anthropic_tier_metadata(monkeypatch):
    import openai_compatible_bridge.main as bridge_main

    monkeypatch.setattr(bridge_main, "MODEL_REGISTRY", _tier_registry())
    monkeypatch.setattr(vertex_mod, "MODEL_LIST_INCLUDE", "")
    monkeypatch.setattr(
        bridge_main, "_main_allowed_models", lambda: {"compat:opus-model", "gemini-2.5-flash"}
    )

    app = _route_app(_client(lambda request: httpx.Response(200, json={})))
    with TestClient(app) as client:
        data = client.get("/v1/models").json()["data"]

    by_id = {m["id"]: m for m in data}
    assert set(by_id) == {"compat:opus-model", "gemini-2.5-flash"}
    entry = by_id["compat:opus-model"]
    assert entry["display_name"] == "Opus 5 (DeepSeek)"
    assert entry["anthropic_family_tier"] == "opus"
    assert entry["is_family_default"] is True
    assert entry["max_tokens"] == 64000
    assert "anthropic_family_tier" not in by_id["gemini-2.5-flash"]


def test_model_list_include_filters_to_matching_ids(monkeypatch):
    import openai_compatible_bridge.main as bridge_main

    monkeypatch.setattr(bridge_main, "MODEL_REGISTRY", _tier_registry())
    monkeypatch.setattr(vertex_mod, "MODEL_LIST_INCLUDE", "compat:*")
    monkeypatch.setattr(
        bridge_main, "_main_allowed_models", lambda: {"compat:opus-model", "gemini-2.5-flash"}
    )

    app = _route_app(_client(lambda request: httpx.Response(200, json={})))
    with TestClient(app) as client:
        ids = [m["id"] for m in client.get("/v1/models").json()["data"]]

    assert ids == ["compat:opus-model"]


def test_model_list_reports_context_window_for_claude_code(monkeypatch):
    import openai_compatible_bridge.main as bridge_main

    registry = {
        "compat:*": {
            "kind": "chat",
            "provider": "openai_compatible",
            "base_url": "https://go.test/v1",
        },
        "compat:big": {"context_length": 1000000, "max_tokens": 64000},
    }
    monkeypatch.setattr(bridge_main, "MODEL_REGISTRY", registry)
    monkeypatch.setattr(vertex_mod, "MODEL_LIST_INCLUDE", "")
    monkeypatch.setattr(bridge_main, "_main_allowed_models", lambda: {"compat:big"})

    app = _route_app(_client(lambda request: httpx.Response(200, json={})))
    with TestClient(app) as client:
        entry = client.get("/v1/models").json()["data"][0]

    assert entry["context_length"] == 1000000
    assert entry["context_window"] == 1000000
    assert entry["runtime"]["max_input_tokens"] == 1000000
    assert entry["runtime"]["max_output_tokens"] == 64000


def test_registry_hot_reload_from_file(monkeypatch, tmp_path):
    import openai_compatible_bridge.main as bridge_main
    import openai_compatible_bridge.providers.vertex as vertex_mod
    import json, os, time

    hook_calls = {"n": 0}

    def hook():
        hook_calls["n"] += 1

    monkeypatch.setattr(vertex_mod, "MODEL_REGISTRY_RELOAD_HOOK", hook)
    # reset mtime cache
    vertex_mod._registry_mtime = 0.0

    f = tmp_path / "registry.json"
    f.write_text(
        json.dumps(
            {
                "compat:hot": {
                    "kind": "chat",
                    "provider": "openai_compatible",
                    "base_url": "https://go.test/v1",
                    "display_name": "Hot",
                }
            }
        )
    )
    monkeypatch.setenv("MODEL_REGISTRY_FILE", str(f))

    assert vertex_mod._reload_registry_if_changed() is True
    assert "compat:hot" in vertex_mod.MODEL_REGISTRY
    assert hook_calls["n"] == 1

    # change the file
    time.sleep(0.02)
    f.write_text(
        json.dumps(
            {
                "compat:hot2": {
                    "kind": "chat",
                    "provider": "openai_compatible",
                    "base_url": "https://go.test/v1",
                    "display_name": "Hot2",
                }
            }
        )
    )
    os.utime(str(f), (time.time() + 1, time.time() + 1))

    assert vertex_mod._reload_registry_if_changed() is True
    assert "compat:hot2" in vertex_mod.MODEL_REGISTRY
    assert "compat:hot" not in vertex_mod.MODEL_REGISTRY
    assert hook_calls["n"] == 2


def test_registry_hot_reload_noop_without_file(monkeypatch):
    import openai_compatible_bridge.providers.vertex as vertex_mod

    monkeypatch.delenv("MODEL_REGISTRY_FILE", raising=False)
    assert vertex_mod._reload_registry_if_changed() is False
