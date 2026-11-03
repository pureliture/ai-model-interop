from __future__ import annotations

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
