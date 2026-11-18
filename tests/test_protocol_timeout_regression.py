import asyncio
import os
import subprocess
import sys

import httpx
import pytest

from openai_compatible_bridge.providers.foundry import FoundryChatClient
from openai_compatible_bridge.providers.jetbrains import JetBrainsChatClient


@pytest.mark.parametrize('provider', ['foundry', 'jetbrains'])
@pytest.mark.parametrize('stream', [False, True])
def test_provider_http_error_mapping(provider, stream):
    async def run():
        cls = FoundryChatClient if provider == 'foundry' else JetBrainsChatClient
        client = cls(base_url='https://example.test', token='test')
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(429, json={'parameters': {'errorMessage': 'foundry nested', 'errorCode': 'nested'}})))
        try:
            with pytest.raises(Exception) as caught:
                if stream:
                    _ = [event async for event in client.stream_chat(model='m', messages=[])]
                else:
                    await client.generate(model='m', messages=[])
            error = caught.value
            assert error.status_code == 429
            if provider == 'foundry':
                assert error.message == 'foundry nested'
                assert error.code == 'nested'
            else:
                assert error.message != 'foundry nested'
                assert error.code == '429'
        finally:
            await client.close()
    asyncio.run(run())


def test_provider_clients_default_to_unlimited_timeout():
    env = os.environ.copy()
    for key in ('HTTP_TIMEOUT_SECONDS', 'JETBRAINS_AI_HTTP_TIMEOUT_SECONDS', 'FOUNDRY_HTTP_TIMEOUT_SECONDS', 'OLLAMA_HTTP_TIMEOUT_SECONDS', 'OPENAI_COMPAT_HTTP_TIMEOUT_SECONDS'):
        env[key] = ''
    script = '''
import asyncio
from openai_compatible_bridge.providers.foundry import FoundryChatClient
from openai_compatible_bridge.providers.jetbrains import JetBrainsChatClient
from openai_compatible_bridge.providers.ollama import OllamaChatClient
from openai_compatible_bridge.providers.openai_compatible import OpenAICompatibleChatClient
from openai_compatible_bridge.providers.vertex import VertexChatClient, VertexEmbeddingClient, VertexRerankClient
class Token: pass
clients = [FoundryChatClient(), JetBrainsChatClient(token=''), OllamaChatClient(), OpenAICompatibleChatClient(), VertexChatClient(Token()), VertexEmbeddingClient(Token()), VertexRerankClient(Token())]
assert all(c.http.timeout.read is None and c.http.timeout.connect is None for c in clients)
async def close():
    for c in clients: await c.close()
asyncio.run(close())
'''
    result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('key', ['JETBRAINS_AI_HTTP_TIMEOUT_SECONDS', 'FOUNDRY_HTTP_TIMEOUT_SECONDS',
                                 'OLLAMA_HTTP_TIMEOUT_SECONDS', 'OPENAI_COMPAT_HTTP_TIMEOUT_SECONDS'])
def test_timeout_override_and_global_fallback(monkeypatch, key):
    from openai_compatible_bridge.core.http_timeout import http_timeout_seconds
    monkeypatch.setenv('HTTP_TIMEOUT_SECONDS', '61')
    monkeypatch.setenv(key, '123')
    assert http_timeout_seconds(key) == 123
    monkeypatch.setenv(key, ' ')
    assert http_timeout_seconds(key) == 61
    monkeypatch.setenv('HTTP_TIMEOUT_SECONDS', '')
    assert http_timeout_seconds(key) is None
