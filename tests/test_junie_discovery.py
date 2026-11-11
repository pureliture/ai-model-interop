from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

import openai_compatible_bridge.junie_discovery as discovery
import openai_compatible_bridge.main as main
import openai_compatible_bridge.providers.vertex as vertex
from test_jetbrains_chat import _FakeJetBrains, _FakeProvider


def _result(*values):
    return {"configOptions": [{"id": "model", "options": [
        {"group": "jetbrains-ai", "options": [{"value": value} for value in values]},
        {"group": "other", "options": [{"value": "v1:12:jetbrains-ai:gpt-hidden"}]},
    ]}]}


def test_parse_all_valid_native_ids_and_identify_known_families():
    parsed = discovery.parse_models(_result(
        "v1:12:jetbrains-ai:claude-sonnet-5", "v1:12:jetbrains-ai:gemini-3.1-pro-preview",
        "v1:12:jetbrains-ai:openai-gpt-5-4", "v1:12:jetbrains-ai:grok-4-5",
        "v1:12:jetbrains-ai:unknown-model", "v1:12:jetbrains-ai:gpt-../../secret",
        "v1:12:other:openai-gpt-5-4",
    ))
    assert {key: cfg["protocol"] for key, cfg in parsed.items()} == {
        "jetbrains:claude-sonnet-5": "anthropic_messages",
        "jetbrains:gemini-3.1-pro-preview": "google_generate_content",
        "jetbrains:openai-gpt-5-4": "openai_responses",
        "jetbrains:grok-4-5": "xai_responses",
        "jetbrains:unknown-model": None,
    }
    assert parsed["jetbrains:openai-gpt-5-4"]["provider_model"] == "openai-gpt-5-4"
    for family in ("claude-new-1", "gemini-new-1", "grok-new-1", "gpt-new-1"):
        assert discovery.parse_models(_result("v1:12:jetbrains-ai:" + family))["jetbrains:" + family]["protocol"]
    values = [f"v1:12:jetbrains-ai:future-{index}" for index in range(22)]
    assert len(discovery.parse_models(_result(*values))) == 22
    assert "jetbrains:vendor/model.v1" in discovery.parse_models(_result("v1:12:jetbrains-ai:vendor/model.v1"))
    with pytest.raises(ValueError):
        discovery.parse_models({"configOptions": []})


def test_cache_failure_keeps_snapshot_and_successful_empty_clears(monkeypatch):
    async def scenario():
        cache = discovery.JunieDiscovery(enabled=True)
        expected = discovery.parse_models(_result("v1:12:jetbrains-ai:openai-gpt-5-4"))
        calls = 0
        async def fetch(*args):
            nonlocal calls
            calls += 1
            if calls == 1:
                return expected
            if calls == 2:
                raise OSError("unavailable")
            return {}
        monkeypatch.setattr(discovery, "_fetch", fetch)
        assert await cache.refresh() == expected
        assert await cache.refresh() == expected
        assert calls == 1
        cache._next_refresh = 0
        assert await cache.refresh() == expected
        assert cache.stale
        cache._next_refresh = 0
        assert await cache.refresh() == {}
        assert not cache.stale
    asyncio.run(scenario())


def test_discovered_listing_chat_and_registry_precedence(monkeypatch):
    async def fetch(*args):
        return discovery.parse_models(_result("v1:12:jetbrains-ai:claude-sonnet-5", "v1:12:jetbrains-ai:unknown-model"))
    monkeypatch.setattr(discovery, "_fetch", fetch)
    monkeypatch.setattr(main, "discovery_from_env", lambda: discovery.JunieDiscovery(enabled=True))
    fake = _FakeJetBrains()
    app = main.create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider,
        rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider,
        foundry_chat_client_factory=_FakeProvider, jetbrains_chat_client_factory=lambda: fake,
        cost_accounting_factory=lambda: None)
    name = "jetbrains:claude-sonnet-5"
    with TestClient(app) as client:
        response = client.get("/v1/models")
        assert response.headers["x-junie-discovery-status"] == "fresh"
        row = next(item for item in response.json()["data"] if item["id"] == name)
        assert "context_length" not in row and "max_tokens" not in row
        assert client.get("/v1/models/" + name).status_code == 200
        assert any(item["id"] == "jetbrains:unknown-model" for item in response.json()["data"])
        response = client.post("/v1/chat/completions", json={"model": name, "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 200
        assert fake.calls[-1]["resolved_config"]["protocol"] == "anthropic_messages"
        assert fake.calls[-1]["resolved_config"]["provider_model"] == "claude-sonnet-5"
        calls = len(fake.calls)
        response = client.post("/v1/chat/completions", json={"model": "jetbrains:unknown-model", "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "unsupported_model"
        assert len(fake.calls) == calls
        response = client.post("/v1/messages", json={"model": "jetbrains:unknown-model", "max_tokens": 10, "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 400
        assert "unsupported protocol" in response.json()["error"]["message"]
        assert len(fake.calls) == calls
        response = client.post("/v1/chat/completions", json={"model": "jetbrains:unlisted-native", "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 200
        assert fake.calls[-1]["resolved_config"].get("protocol") is None
        assert fake.calls[-1]["model"] == "unlisted-native"
        monkeypatch.setitem(vertex.MODEL_REGISTRY, name, {"provider": "jetbrains", "kind": "chat", "provider_model": "explicit", "protocol": "openai_responses"})
        response = client.post("/v1/chat/completions", json={"model": name, "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 200
        assert fake.calls[-1]["resolved_config"]["provider_model"] == "explicit"
        monkeypatch.setitem(vertex.MODEL_REGISTRY, "jetbrains:unknown-model", {"provider": "jetbrains", "kind": "chat", "provider_model": "known-id", "protocol": "openai_responses"})
        response = client.post("/v1/chat/completions", json={"model": "jetbrains:unknown-model", "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 200
        assert fake.calls[-1]["model"] == "known-id"


def test_discovered_model_preserves_upstream_error(monkeypatch):
    async def fetch(*args):
        return discovery.parse_models(_result("v1:12:jetbrains-ai:gpt-unverified"))
    class FailingJetBrains(_FakeProvider):
        async def generate(self, **kwargs):
            raise vertex.VertexAPIError(404, "upstream model unavailable", code="upstream_missing")
    monkeypatch.setattr(discovery, "_fetch", fetch)
    monkeypatch.setattr(main, "discovery_from_env", lambda: discovery.JunieDiscovery(enabled=True))
    app = main.create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider,
        rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider,
        foundry_chat_client_factory=_FakeProvider, jetbrains_chat_client_factory=FailingJetBrains,
        cost_accounting_factory=lambda: None)
    with TestClient(app) as client:
        response = client.post("/v1/chat/completions", json={"model": "jetbrains:gpt-unverified", "messages": [{"role": "user", "content": "hello"}]})
        assert response.status_code == 404
        assert response.json()["error"]["message"] == "upstream model unavailable"
        assert response.json()["error"]["code"] == "upstream_missing"


def test_acp_handshake_uses_json_lines_and_terminates(monkeypatch):
    import json

    async def scenario():
        sent = []
        responses = [
            {"jsonrpc": "2.0", "method": "session/update", "params": {}},
            {"jsonrpc": "2.0", "id": 1, "result": {}},
            {"jsonrpc": "2.0", "id": 2, "result": _result("v1:12:jetbrains-ai:openai-gpt-5-4")},
        ]
        class Input:
            def write(self, line):
                sent.append(json.loads(line))
            async def drain(self):
                pass
        class Output:
            async def readline(self):
                return (json.dumps(responses.pop(0)) + "\n").encode()
        class Process:
            stdin, stdout, returncode = Input(), Output(), None
            killed = False
            def kill(self):
                self.killed = True
            async def wait(self):
                pass
        proc = Process()
        async def spawn(*args, **kwargs):
            assert args == ("junie", "--acp=true", "--skip-update-check")
            assert kwargs["stderr"] == asyncio.subprocess.DEVNULL
            return proc
        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        result = await discovery._fetch("junie", "/tmp")
        assert [item["method"] for item in sent] == ["initialize", "session/new"]
        assert sent[1]["params"] == {"cwd": "/tmp", "mcpServers": []}
        assert list(result) == ["jetbrains:openai-gpt-5-4"]
        assert proc.killed
    asyncio.run(scenario())


def test_timeout_preserves_cache_and_coalesces_requests(monkeypatch):
    async def scenario():
        cache = discovery.JunieDiscovery(enabled=True, timeout=0.001)
        cache.models = {"jetbrains:openai-gpt-5-4": {"provider": "jetbrains"}}
        calls = 0
        async def slow(*args):
            nonlocal calls
            calls += 1
            await asyncio.Event().wait()
        monkeypatch.setattr(discovery, "_fetch", slow)
        first, second = await asyncio.gather(cache.refresh(), cache.refresh())
        assert first == second == cache.models
        assert calls == 1 and cache.stale
    asyncio.run(scenario())
