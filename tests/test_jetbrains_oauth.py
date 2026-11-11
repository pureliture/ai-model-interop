import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from openai_compatible_bridge.providers import jetbrains_oauth
from openai_compatible_bridge.providers.foundry import FoundryChatClient
from openai_compatible_bridge.providers.jetbrains import JetBrainsChatClient
from openai_compatible_bridge.providers.jetbrains_oauth import JetBrainsOAuthError, JetBrainsOAuthStore


def make_store(tmp_path):
    os.chmod(tmp_path, 0o700)
    path = tmp_path / "oauth.json"
    path.write_text(json.dumps({"token_url": "https://auth.example.test/token", "client_id": "client",
                                "refresh_token": "old-refresh", "expires_at": 0}))
    os.chmod(path, 0o600)
    return path


def test_concurrent_refresh_serializes_and_persists_rotation(tmp_path, monkeypatch):
    path = make_store(tmp_path)
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, url, *, data, headers):
            calls.append(data.copy())
            time.sleep(0.02)
            return httpx.Response(200, json={"access_token": "new-access", "refresh_token": "new-refresh", "expires_in": 3600})

    monkeypatch.setattr(jetbrains_oauth.httpx, "Client", FakeClient)
    with ThreadPoolExecutor(max_workers=8) as executor:
        tokens = list(executor.map(lambda _: JetBrainsOAuthStore(str(path))._access_token(), range(8)))
    assert tokens == ["new-access"] * 8
    assert len(calls) == 1
    state = json.loads(path.read_text())
    assert state["refresh_token"] == "new-refresh"
    assert state["refresh_in_progress"] is False
    assert path.stat().st_mode & 0o777 == 0o600


def test_ambiguous_failure_blocks_reuse(tmp_path, monkeypatch):
    path = make_store(tmp_path)
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            calls.append(True)
            raise httpx.ReadTimeout("sensitive upstream details")

    monkeypatch.setattr(jetbrains_oauth.httpx, "Client", FakeClient)
    store = JetBrainsOAuthStore(str(path))
    for _ in range(2):
        with pytest.raises(JetBrainsOAuthError) as error:
            store._access_token()
        assert "sensitive" not in str(error.value)
    assert len(calls) == 1
    assert json.loads(path.read_text())["refresh_in_progress"] is True


def test_insecure_permissions_fail_before_network(tmp_path):
    path = make_store(tmp_path)
    os.chmod(path, 0o644)
    with pytest.raises(JetBrainsOAuthError):
        JetBrainsOAuthStore(str(path))._access_token()


def test_generate_and_stream_use_request_specific_tokens(monkeypatch):
    class Store:
        async def access_token(self):
            return "refreshed-token"

    async def fake_stream(self, **kwargs):
        yield self._headers("openai_responses")["Authorization"]

    async def run():
        client = JetBrainsChatClient(token="")
        client.oauth_store = Store()
        captured = []

        async def handler(request):
            captured.append(request.headers["authorization"])
            return httpx.Response(200, json={"output": [], "usage": {}})

        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        monkeypatch.setattr(FoundryChatClient, "stream_chat", fake_stream)
        await client.generate(model="model", messages=[])
        events = [event async for event in client.stream_chat(model="model", messages=[])]
        await client.close()
        assert captured == ["Bearer refreshed-token"]
        assert events == ["Bearer refreshed-token"]
        assert client.token == ""

    asyncio.run(run())


def test_concurrent_generate_and_stream_share_refreshed_token(tmp_path, monkeypatch):
    path = make_store(tmp_path)
    refresh_calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def post(self, *args, **kwargs):
            refresh_calls.append(True)
            time.sleep(0.02)
            return httpx.Response(200, json={"access_token": "fresh", "refresh_token": "rotated", "expires_in": 3600})

    async def fake_stream(self, **kwargs):
        yield self._headers("openai_responses")["Authorization"]

    async def run():
        client = JetBrainsChatClient(token="")
        client.oauth_store = JetBrainsOAuthStore(str(path))
        captured = []

        async def handler(request):
            captured.append(request.headers["authorization"])
            return httpx.Response(200, json={"output": [], "usage": {}})

        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))

        async def stream():
            return [event async for event in client.stream_chat(model="model", messages=[])]

        results = await asyncio.gather(*(client.generate(model="model", messages=[]) for _ in range(8)),
                                       *(stream() for _ in range(8)))
        await client.close()
        assert captured == ["Bearer fresh"] * 8
        assert results[8:] == [["Bearer fresh"]] * 8

    monkeypatch.setattr(jetbrains_oauth.httpx, "Client", FakeClient)
    monkeypatch.setattr(FoundryChatClient, "stream_chat", fake_stream)
    asyncio.run(run())
    assert len(refresh_calls) == 1


def test_explicit_static_token_takes_precedence(monkeypatch, tmp_path):
    monkeypatch.setenv("JETBRAINS_AI_OAUTH_FILE", str(tmp_path / "unused.json"))
    client = JetBrainsChatClient(token="static")
    assert client.oauth_store is None
    assert asyncio.run(client._request_token()) == "static"
