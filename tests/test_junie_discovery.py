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


def test_discovered_effort_metadata_only_advertises_routable_levels(monkeypatch):
    async def fetch(*args):
        return {
            "jetbrains:gpt-test": {"provider": "jetbrains", "kind": "chat",
                "provider_model": "gpt-test", "protocol": "openai_responses",
                "reasoning_efforts": ["none", "low", "high", "max"]},
            "jetbrains:claude-test": {"provider": "jetbrains", "kind": "chat",
                "provider_model": "claude-test", "protocol": "anthropic_messages",
                "reasoning_efforts": ["low", "high"]},
        }
    monkeypatch.setattr(discovery, "_fetch", fetch)
    monkeypatch.setattr(main, "discovery_from_env", lambda: discovery.JunieDiscovery(enabled=True))
    app = main.create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider,
        rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider,
        foundry_chat_client_factory=_FakeProvider, jetbrains_chat_client_factory=_FakeJetBrains,
        cost_accounting_factory=lambda: None)
    with TestClient(app) as client:
        rows = {row["id"]: row for row in client.get("/v1/models").json()["data"]}
        expected = [{"effort": level} for level in ("none", "low", "high", "max")]
        assert rows["jetbrains:gpt-test"]["supported_reasoning_levels"] == expected
        assert rows["jetbrains:claude-test"]["supported_reasoning_levels"] == [
            {"effort": "low"}, {"effort": "high"},
        ]
        assert client.get("/v1/models/jetbrains:gpt-test").json()["supported_reasoning_levels"] == expected


def test_jar_limits_are_joined_only_to_discovered_models(monkeypatch):
    async def fetch(*args):
        return discovery.parse_models(_result(
            "v1:12:jetbrains-ai:gemini-4-argon",
            "v1:12:jetbrains-ai:claude-sonnet-5",
            "v1:12:jetbrains-ai:gemini-unknown",
        ))
    async def metadata(pattern):
        assert pattern == "/installed/junie.jar"
        return {
            "gemini-4-argon": {"context_length": 1048576},
            "claude-sonnet-5": {"context_length": 1000000, "max_tokens": 128000},
            "unlisted": {"context_length": 42},
        }
    monkeypatch.setenv("JUNIE_DISCOVERY_JAR", "/installed/junie.jar")
    monkeypatch.setattr(discovery, "_fetch", fetch)
    monkeypatch.setattr(discovery, "_jar_metadata", metadata)
    monkeypatch.setattr(main, "discovery_from_env", lambda: discovery.JunieDiscovery(enabled=True))
    app = main.create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider,
        rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider,
        foundry_chat_client_factory=_FakeProvider, jetbrains_chat_client_factory=_FakeJetBrains,
        cost_accounting_factory=lambda: None)
    with TestClient(app) as client:
        rows = {row["id"]: row for row in client.get("/v1/models").json()["data"]}
        assert rows["jetbrains:gemini-4-argon"]["context_length"] == 1048576
        assert "max_tokens" not in rows["jetbrains:gemini-4-argon"]
        assert rows["jetbrains:claude-sonnet-5"]["max_tokens"] == 128000
        assert "context_length" not in rows["jetbrains:gemini-unknown"]
        assert "jetbrains:unlisted" not in rows
        assert client.get("/v1/models/jetbrains:claude-sonnet-5").json()["context_length"] == 1000000


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


def test_model_specific_efforts_are_discovered_without_guessing(monkeypatch):
    import json

    async def scenario():
        sent = []
        values = ["v1:12:jetbrains-ai:gemini-one", "v1:12:jetbrains-ai:gpt-two",
                  "v1:12:jetbrains-ai:unknown-three"]
        session = {**_result(*values), "sessionId": "session"}
        class Input:
            def write(self, line):
                sent.append(json.loads(line))
            async def drain(self):
                pass
        class Output:
            async def readline(self):
                req = sent[-1]
                if req["id"] == 1:
                    result = {}
                elif req["id"] == 2:
                    result = session
                else:
                    value = req["params"]["value"]
                    options = {values[0]: ["low", "medium", "high"],
                               values[1]: ["none", "low", "xhigh", "max"]}.get(value)
                    result = {"configOptions": [{"id": "model", "currentValue": value}]}
                    if options:
                        result["configOptions"].append({"id": "effort", "type": "select",
                            "category": "thought_level", "options": [{"value": v} for v in options]})
                return (json.dumps({"jsonrpc": "2.0", "id": req["id"], "result": result}) + "\n").encode()
        class Process:
            stdin, stdout, returncode = Input(), Output(), None
            def kill(self):
                pass
            async def wait(self):
                pass
        async def spawn(*args, **kwargs):
            return Process()
        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        result = await discovery._fetch("junie", "/tmp")
        assert result["jetbrains:gemini-one"]["reasoning_efforts"] == ["low", "medium", "high"]
        assert result["jetbrains:gpt-two"]["reasoning_efforts"] == ["none", "low", "xhigh", "max"]
        assert "reasoning_efforts" not in result["jetbrains:unknown-three"]
        assert [r["params"]["value"] for r in sent[2:]] == values
    asyncio.run(scenario())


def test_selected_efforts_require_matching_model_and_valid_options():
    response = {"configOptions": [
        {"id": "model", "currentValue": "model-a"},
        {"id": "effort", "type": "select", "category": "thought_level",
         "options": [{"value": "low"}, {"value": "high"}]},
    ]}
    assert discovery._selected_efforts(response, "model-b") is None
    assert discovery._selected_efforts(response, "model-a") == ["low", "high"]
    response["configOptions"][1]["options"].append({"value": None})
    assert discovery._selected_efforts(response, "model-a") is None


def test_effort_probe_timeout_returns_model_snapshot(monkeypatch):
    import json

    async def scenario():
        sent = []
        class Input:
            def write(self, line):
                sent.append(json.loads(line))
            async def drain(self):
                pass
        class Output:
            async def readline(self):
                req = sent[-1]
                if req["id"] == 3:
                    await asyncio.Event().wait()
                result = {} if req["id"] == 1 else {
                    **_result("v1:12:jetbrains-ai:gpt-one", "v1:12:jetbrains-ai:gpt-two"),
                    "sessionId": "session"}
                return (json.dumps({"id": req["id"], "result": result}) + "\n").encode()
        class Process:
            stdin, stdout, returncode = Input(), Output(), None
            def kill(self):
                pass
            async def wait(self):
                pass
        async def spawn(*args, **kwargs):
            return Process()
        monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
        monkeypatch.setattr(discovery, "_PROBE_TIMEOUT", 0.001)
        result = await discovery._fetch("junie", "/tmp")
        assert len(result) == 2
        assert all("reasoning_efforts" not in model for model in result.values())
        assert len(sent) == 3
    asyncio.run(scenario())


def test_parse_models_keeps_acp_display_names():
    result = {"configOptions": [{"id": "model", "options": [{"group": "jetbrains-ai", "options": [
        {"value": "v1:12:jetbrains-ai:claude-sonnet-5-5", "name": "  Claude   Sonnet 5.5 "},
        {"value": "v1:12:jetbrains-ai:gpt-6-luna", "name": ""},
        {"value": "v1:12:jetbrains-ai:grok-4.7"},
    ]}]}]}
    parsed = discovery.parse_models(result)
    assert parsed["jetbrains:claude-sonnet-5-5"]["display_name"] == "Claude Sonnet 5.5"
    assert "display_name" not in parsed["jetbrains:gpt-6-luna"]
    assert "display_name" not in parsed["jetbrains:grok-4.7"]


def test_jar_rows_distinguish_null_output_cap_from_missing_limits():
    stdout = (
        b"gemini-4-argon\t1048576\t\t1\n"
        b"claude-sonnet-5\t1000000\t128000\t1\n"
        b"grok-4.7\t\t\t0\n"
        b"gpt-6-luna\t1000000\t0\t0\n"
        b"legacy-three-columns\t1000000\t128000\n"
    )
    assert discovery._jar_limits(stdout) == {
        "gemini-4-argon": {"context_length": 1048576, "max_tokens_unspecified": True, "image": True},
        "claude-sonnet-5": {"context_length": 1000000, "max_tokens": 128000, "image": True},
        "grok-4.7": {"max_tokens_unspecified": True, "image": False},
        "gpt-6-luna": {"context_length": 1000000, "image": False},
        "legacy-three-columns": {"context_length": 1000000, "max_tokens": 128000},
    }


def test_jar_vision_advertises_images_without_manual_allowlist(monkeypatch):
    async def fetch(*args):
        return discovery.parse_models(_named_result(
            ("v1:12:jetbrains-ai:gemini-4-hydrogen", "Gemini 4 Hydrogen"),
            ("v1:12:jetbrains-ai:grok-code-fast", "Grok Code Fast"),
            ("v1:12:jetbrains-ai:gemini-unknown", "Gemini Unknown"),
        ))
    async def metadata(pattern):
        return {
            "gemini-4-hydrogen": {"context_length": 1048576, "max_tokens_unspecified": True, "image": True},
            "grok-code-fast": {"context_length": 256000, "max_tokens_unspecified": True, "image": False},
        }
    monkeypatch.setenv("JUNIE_DISCOVERY_JAR", "/installed/junie.jar")
    monkeypatch.delenv("JETBRAINS_AI_IMAGE_MODELS", raising=False)
    monkeypatch.setattr(discovery, "_fetch", fetch)
    monkeypatch.setattr(discovery, "_jar_metadata", metadata)
    monkeypatch.setattr(main, "discovery_from_env", lambda: discovery.JunieDiscovery(enabled=True))
    app = _app_with_discovery(_FakeJetBrains())
    with TestClient(app) as client:
        rows = {row["id"]: row for row in client.get("/v1/models").json()["data"]}
        assert rows["jetbrains:gemini-4-hydrogen"]["input"] == ["text", "image"]
        assert rows["jetbrains:gemini-4-hydrogen"]["capabilities"]["images"] is True
        assert "input" not in rows["jetbrains:grok-code-fast"]
        assert "input" not in rows["jetbrains:gemini-unknown"]
        monkeypatch.setenv("JETBRAINS_AI_IMAGE_MODELS", "grok-code-fast,gemini-unknown")
        rows = {row["id"]: row for row in client.get("/v1/models").json()["data"]}
        assert rows["jetbrains:grok-code-fast"]["input"] == ["text", "image"]
        assert rows["jetbrains:gemini-unknown"]["input"] == ["text", "image"]


def _app_with_discovery(fake):
    app = main.create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider,
        rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider,
        foundry_chat_client_factory=_FakeProvider, jetbrains_chat_client_factory=lambda: fake,
        cost_accounting_factory=lambda: None)
    return app


def _named_result(*entries):
    return {"configOptions": [{"id": "model", "options": [{"group": "jetbrains-ai", "options": [
        {"value": value, "name": name} for value, name in entries
    ]}]}]}


def test_listing_exposes_acp_display_name_metadata(monkeypatch):
    async def fetch(*args):
        return discovery.parse_models(_named_result(
            ("v1:12:jetbrains-ai:claude-sonnet-5-5", "Claude Sonnet 5.5"),
            ("v1:12:jetbrains-ai:gemini-4-argon", "Gemini 4 Argon"),
        ))
    monkeypatch.setattr(discovery, "_fetch", fetch)
    monkeypatch.setattr(main, "discovery_from_env", lambda: discovery.JunieDiscovery(enabled=True))
    app = _app_with_discovery(_FakeJetBrains())
    with TestClient(app) as client:
        rows = {row["id"]: row for row in client.get("/v1/models").json()["data"]}
        assert rows["jetbrains:claude-sonnet-5-5"]["metadata"] == {"display_name": "Claude Sonnet 5.5"}
        assert client.get("/v1/models/jetbrains:claude-sonnet-5-5").json()["metadata"] == {
            "display_name": "Claude Sonnet 5.5"
        }


def test_discovery_default_max_tokens_is_dropped_only_for_null_output_caps(monkeypatch):
    async def fetch(*args):
        return discovery.parse_models(_named_result(
            ("v1:12:jetbrains-ai:gemini-4-argon", "Gemini 4 Argon"),
            ("v1:12:jetbrains-ai:claude-sonnet-5-5", "Claude Sonnet 5.5"),
            ("v1:12:jetbrains-ai:gpt-6-astra", "GPT 6 Astra"),
        ))
    async def metadata(pattern):
        assert pattern == "/installed/junie.jar"
        return {
            "gemini-4-argon": {"context_length": 1048576, "max_tokens_unspecified": True},
            "claude-sonnet-5-5": {"context_length": 1000000, "max_tokens": 128000},
        }
    monkeypatch.setenv("JUNIE_DISCOVERY_JAR", "/installed/junie.jar")
    monkeypatch.setattr(discovery, "_fetch", fetch)
    monkeypatch.setattr(discovery, "_jar_metadata", metadata)
    monkeypatch.setattr(main, "discovery_from_env", lambda: discovery.JunieDiscovery(enabled=True))
    fake = _FakeJetBrains()
    app = _app_with_discovery(fake)
    with TestClient(app) as client:
        def ask(model, max_tokens, stream=False):
            body = {"model": model, "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": "hello"}]}
            if stream:
                body["stream"] = True
            assert client.post("/v1/chat/completions", json=body).status_code == 200
            return fake.calls[-1]

        assert ask("jetbrains:gemini-4-argon", 32768)["max_tokens"] is None
        assert ask("jetbrains:gemini-4-argon", 4096)["max_tokens"] == 4096
        assert ask("jetbrains:gemini-4-argon", 32768, stream=True)["max_tokens"] is None
        # A JAR-declared cap keeps the client value even at the default.
        assert ask("jetbrains:claude-sonnet-5-5", 32768)["max_tokens"] == 32768
        # A model the JAR did not report is not treated as null.
        assert ask("jetbrains:gpt-6-astra", 32768)["max_tokens"] == 32768
