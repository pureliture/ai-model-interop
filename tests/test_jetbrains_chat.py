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
