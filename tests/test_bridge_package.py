from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient


def test_bridge_package_exposes_fastapi_app():
    from openai_compatible_bridge.main import app

    assert app.title == "AI Model Interop"


def test_create_app_accepts_provider_factories():
    from openai_compatible_bridge.main import create_app

    class FakeProvider:
        def __init__(self) -> None:
            self.closed = False

        async def close(self) -> None:
            self.closed = True

    fake_embedding = FakeProvider()
    fake_chat = FakeProvider()
    fake_rerank = FakeProvider()

    test_app = create_app(
        embedding_client_factory=lambda: fake_embedding,
        chat_client_factory=lambda: fake_chat,
        rerank_client_factory=lambda: fake_rerank,
        cost_accounting_factory=lambda: None,
    )

    with TestClient(test_app) as client:
        response = client.get("/healthz")

    assert response.json() == {"status": "ok"}
    assert response.headers["cache-control"] == "no-store"
    assert fake_embedding.closed is True
    assert fake_chat.closed is True
    assert fake_rerank.closed is True


def test_module_app_uses_bridge_factory_lifespan(monkeypatch):
    import openai_compatible_bridge.main as bridge_main

    events: list[str] = []

    class FakeProvider:
        def __init__(self, label: str) -> None:
            self.label = label
            events.append(f"created:{label}")

        async def close(self) -> None:
            events.append(f"closed:{self.label}")

    class FakeEmbeddingProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__("embedding")

    class FakeChatProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__("vertex-chat")

    class FakeRerankProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__("rerank")

    class FakeOllamaProvider(FakeProvider):
        def __init__(self) -> None:
            super().__init__("ollama-chat")

    monkeypatch.setattr(bridge_main, "VertexEmbeddingClient", FakeEmbeddingProvider)
    monkeypatch.setattr(bridge_main, "VertexChatClient", FakeChatProvider)
    monkeypatch.setattr(bridge_main, "VertexRerankClient", FakeRerankProvider)
    monkeypatch.setattr(bridge_main, "OllamaChatClient", FakeOllamaProvider)
    monkeypatch.setattr(bridge_main, "build_async_cost_accounting", lambda _env: None)

    with TestClient(bridge_main.app) as client:
        response = client.get("/healthz")
        assert response.json() == {"status": "ok"}
        assert client.app.state.ollama_chat_client.label == "ollama-chat"

    assert "created:ollama-chat" in events
    assert "closed:ollama-chat" in events


def test_readyz_is_never_served_from_cache_without_accounting():
    from openai_compatible_bridge.main import create_app

    class FakeProvider:
        async def close(self) -> None:
            pass

    test_app = create_app(
        embedding_client_factory=FakeProvider,
        chat_client_factory=FakeProvider,
        rerank_client_factory=FakeProvider,
        cost_accounting_factory=lambda: None,
    )

    with TestClient(test_app) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.headers["cache-control"] == "no-store"


def test_readyz_forbids_mime_sniffing_without_accounting():
    from openai_compatible_bridge.main import create_app

    class FakeProvider:
        async def close(self) -> None:
            pass

    test_app = create_app(
        embedding_client_factory=FakeProvider,
        chat_client_factory=FakeProvider,
        rerank_client_factory=FakeProvider,
        cost_accounting_factory=lambda: None,
    )

    with TestClient(test_app) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.headers["x-content-type-options"] == "nosniff"


def test_routes_resolve_models_from_current_registry():
    import openai_compatible_bridge.providers.vertex as vertex
    from openai_compatible_bridge.main import create_app

    class FakeProvider:
        async def close(self) -> None:
            pass

    old_registry = vertex.MODEL_REGISTRY.copy()
    try:
        vertex.MODEL_REGISTRY["llama-local"] = {
            "provider": "ollama",
            "kind": "chat",
            "provider_model": "llama3.1",
        }
        test_app = create_app(
            embedding_client_factory=FakeProvider,
            chat_client_factory=FakeProvider,
            rerank_client_factory=FakeProvider,
            cost_accounting_factory=lambda: None,
        )

        with TestClient(test_app) as client:
            response = client.get("/v1/models")

        ids = {item["id"] for item in response.json()["data"]}
        assert "llama-local" in ids
        assert response.headers["cache-control"] == "no-store"
    finally:
        vertex.MODEL_REGISTRY.clear()
        vertex.MODEL_REGISTRY.update(old_registry)


def test_bridge_api_key_replaces_wrapper_api_key(monkeypatch):
    import openai_compatible_bridge.main as bridge_main
    from openai_compatible_bridge.main import create_app

    class FakeProvider:
        async def close(self) -> None:
            pass

    assert not hasattr(bridge_main, "WRAPPER_API_KEY")
    monkeypatch.setattr(bridge_main, "BRIDGE_API_KEY", "secret-key")
    test_app = create_app(
        embedding_client_factory=FakeProvider,
        chat_client_factory=FakeProvider,
        rerank_client_factory=FakeProvider,
        cost_accounting_factory=lambda: None,
    )

    with TestClient(test_app) as client:
        bad = client.get("/v1/models")
        ok = client.get("/v1/models", headers={"Authorization": "Bearer secret-key"})

    assert bad.status_code == 401
    assert ok.status_code == 200
    assert ok.headers["cache-control"] == "no-store"


def test_vertex_auth_is_lazy_and_retries_after_credentials_become_available(monkeypatch):
    import openai_compatible_bridge.providers.vertex as vertex

    calls = []

    def default_without_credentials(**_kwargs):
        calls.append("missing")
        raise RuntimeError("credentials unavailable")

    monkeypatch.setattr(vertex.google.auth, "default", default_without_credentials)
    provider = vertex.GoogleAccessTokenProvider()
    assert calls == []

    with pytest.raises(vertex.VertexAPIError) as error:
        asyncio.run(provider.get_token())
    assert error.value.status_code == 503
    assert error.value.code == "vertex_auth_unavailable"

    class Credentials:
        token = "test-access-token"
        expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1)
        valid = True

    def default_with_credentials(**_kwargs):
        calls.append("available")
        return Credentials(), "test-project"

    monkeypatch.setattr(vertex.google.auth, "default", default_with_credentials)

    async def get_tokens():
        return await asyncio.gather(*(provider.get_token() for _ in range(8)))

    assert asyncio.run(get_tokens()) == ["test-access-token"] * 8
    assert calls == ["missing", "available"]
    assert provider.project_id == "test-project"


def test_vertex_project_is_checked_only_when_auth_is_used(monkeypatch):
    import openai_compatible_bridge.providers.vertex as vertex

    class Credentials:
        token = "test-access-token"
        expiry = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=1)
        valid = True

    calls = []

    def default_without_project(**_kwargs):
        calls.append(True)
        return Credentials(), None

    monkeypatch.setattr(vertex, "VERTEX_PROJECT", None)
    monkeypatch.setattr(vertex.google.auth, "default", default_without_project)
    provider = vertex.GoogleAccessTokenProvider()
    assert calls == []

    with pytest.raises(vertex.VertexAPIError) as error:
        asyncio.run(provider.get_token())
    assert error.value.status_code == 503
    assert error.value.code == "vertex_project_unavailable"

    monkeypatch.setattr(vertex, "VERTEX_PROJECT", "configured-project")
    assert asyncio.run(provider.get_token()) == "test-access-token"
    assert calls == [True, True]
    assert provider.project_id == "configured-project"


def test_app_starts_without_vertex_credentials_and_keeps_other_provider_available(monkeypatch):
    import openai_compatible_bridge.main as bridge_main
    import openai_compatible_bridge.providers.vertex as vertex
    from openai_compatible_bridge.providers.jetbrains import JetBrainsChatClient

    monkeypatch.setitem(
        vertex.MODEL_REGISTRY,
        "test-jetbrains",
        {
            "provider": "jetbrains",
            "kind": "chat",
            "provider_model": "gpt-test",
            "protocol": "openai_chat_completions",
        },
    )
    calls = []

    def no_credentials(**_kwargs):
        calls.append(True)
        raise RuntimeError("credentials unavailable")

    async def jetbrains_generate(_self, **_kwargs):
        return {
            "text": "JetBrains is available",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    monkeypatch.setattr(vertex.google.auth, "default", no_credentials)
    monkeypatch.setattr(JetBrainsChatClient, "generate", jetbrains_generate)
    monkeypatch.setattr(bridge_main, "BRIDGE_API_KEY", "")
    app = bridge_main.create_app(cost_accounting_factory=lambda: None)

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert calls == []

        jetbrains = client.post(
            "/v1/chat/completions",
            json={"model": "test-jetbrains", "messages": [{"role": "user", "content": "hello"}]},
        )
        assert jetbrains.status_code == 200
        assert jetbrains.json()["choices"][0]["message"]["content"] == "JetBrains is available"
        assert calls == []

        vertex_requests = [
            client.post("/v1/embeddings", json={"model": "gemini-embedding-001", "input": "hello"}),
            client.post(
                "/v1/chat/completions",
                json={"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "hello"}]},
            ),
            client.post(
                "/v1/chat/completions",
                json={"model": "gemini-2.5-flash", "messages": [{"role": "user", "content": "hello"}], "stream": True},
            ),
            client.post(
                "/v1/rerank",
                json={"model": "semantic-ranker-512@latest", "query": "hello", "documents": ["world"]},
            ),
        ]

    for index, response in enumerate(vertex_requests):
        if index == 2:
            assert response.status_code == 200
            assert "vertex_auth_unavailable" in response.text
            assert "data: [DONE]" in response.text
            continue
        assert response.status_code == 503, f"Vertex request {index} returned {response.status_code}: {response.text}"
        assert "vertex_auth_unavailable" in response.text
    assert calls
