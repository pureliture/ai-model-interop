import json

import openai_compatible_bridge.providers.vertex as vertex


def test_legacy_registry_provider_keeps_cost_contract(monkeypatch):
    from openai_compatible_bridge.core.provider_adapter import PROVIDER_ENDPOINTS, parse_usage_for_provider
    monkeypatch.setenv('MODEL_REGISTRY_JSON', json.dumps({'legacy': {'provider': 'openai_compat', 'kind': 'chat', 'base_url': 'https://example.test'}}))
    monkeypatch.delenv('MODEL_REGISTRY_FILE', raising=False)
    cfg = vertex._build_registry()['legacy']
    assert cfg['provider'] == 'openai_compat'
    assert PROVIDER_ENDPOINTS['openai_compat'] == ('chat',)
    assert parse_usage_for_provider('openai_compat', {'usage': {'prompt_tokens': 3, 'completion_tokens': 2}}, 'chat').total_tokens == 5


def test_jetbrains_ingresses_forward_config(monkeypatch):
    from fastapi.testclient import TestClient
    from openai_compatible_bridge.main import create_app
    from test_jetbrains_chat import _FakeProvider, _FakeJetBrains

    cfg = {'provider': 'jetbrains', 'kind': 'chat', 'provider_model': 'native', 'protocol': 'anthropic_messages'}
    monkeypatch.setitem(vertex.MODEL_REGISTRY, 'jetbrains:alias', cfg)
    fake = _FakeJetBrains()
    app = create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider,
                     rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider,
                     foundry_chat_client_factory=_FakeProvider, jetbrains_chat_client_factory=lambda: fake,
                     cost_accounting_factory=lambda: None)
    with TestClient(app) as client:
        for path in ('/v1/chat/completions', '/v1/messages'):
            for stream in (False, True):
                response = client.post(path, json={'model': 'jetbrains:alias', 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 16, 'stream': stream})
                assert response.status_code == 200, response.text
                assert fake.calls[-1]['resolved_config']['protocol'] == 'anthropic_messages', (path, stream, fake.calls[-1])


def test_jetbrains_client_uses_instance_token():
    from openai_compatible_bridge.providers.jetbrains import JetBrainsChatClient
    client = JetBrainsChatClient(token='instance-token')
    assert client._headers('anthropic_messages')['Authorization'] == 'Bearer instance-token'


def test_compose_empty_timeout_env_imports_in_fresh_process():
    import os
    import subprocess
    import sys
    env = dict(os.environ, OLLAMA_HTTP_TIMEOUT_SECONDS='', FOUNDRY_HTTP_TIMEOUT_SECONDS='', JETBRAINS_AI_HTTP_TIMEOUT_SECONDS='')
    result = subprocess.run([sys.executable, '-c', 'import openai_compatible_bridge.main'], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_legacy_billing_allows_compat_request(monkeypatch):
    import httpx
    from fastapi.testclient import TestClient
    from openai_compatible_bridge.core.async_cost import AsyncCostAccounting
    from openai_compatible_bridge.main import create_app
    from openai_compatible_bridge.providers.openai_compatible import OpenAICompatibleChatClient
    from test_jetbrains_chat import _FakeProvider

    monkeypatch.setitem(vertex.MODEL_REGISTRY, 'legacy-model', {'provider': 'openai_compat', 'kind': 'chat', 'base_url': 'https://example.test', 'provider_model': 'upstream'})
    upstream = OpenAICompatibleChatClient()
    upstream.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={'choices': [{'message': {'content': 'ok'}, 'finish_reason': 'stop'}], 'usage': {'prompt_tokens': 1, 'completion_tokens': 1}})))
    app = create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider, rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider, foundry_chat_client_factory=_FakeProvider, jetbrains_chat_client_factory=_FakeProvider, openai_compatible_chat_client_factory=lambda: upstream, cost_accounting_factory=lambda: AsyncCostAccounting(config=None, billing={'openai_compat': 'subscription'}))
    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json={'model': 'legacy-model', 'messages': [{'role': 'user', 'content': 'hi'}]})
        assert response.status_code == 200, response.text


def test_jetbrains_real_anthropic_protocol_through_both_ingresses(monkeypatch):
    import httpx
    from fastapi.testclient import TestClient
    from openai_compatible_bridge.main import create_app
    from openai_compatible_bridge.providers.jetbrains import JetBrainsChatClient
    from test_jetbrains_chat import _FakeProvider
    monkeypatch.setitem(vertex.MODEL_REGISTRY, 'jetbrains:real', {'provider': 'jetbrains', 'kind': 'chat', 'provider_model': 'claude-native', 'protocol': 'anthropic_messages'})
    requests = []
    def handler(request):
        requests.append(request)
        body = json.loads(request.content)
        assert request.url.path == '/v1/messages'
        assert request.headers['x-llm-model'] == 'anthropic'
        assert request.headers['authorization'] == 'Bearer test-token'
        assert body['model'] == 'claude-native'
        if body.get('stream'):
            return httpx.Response(200, text='data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hello"}}\n\ndata: {"type":"message_stop"}\n\n', headers={'content-type': 'text/event-stream'})
        return httpx.Response(200, json={'content': [{'type': 'text', 'text': 'hello'}], 'usage': {'input_tokens': 1, 'output_tokens': 1}})
    upstream = JetBrainsChatClient(base_url='https://example.test', token='test-token')
    upstream.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider, rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider, foundry_chat_client_factory=_FakeProvider, jetbrains_chat_client_factory=lambda: upstream, cost_accounting_factory=lambda: None)
    with TestClient(app) as client:
        for path in ('/v1/chat/completions', '/v1/messages'):
            for stream in (False, True):
                response = client.post(path, json={'model': 'jetbrains:real', 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 16, 'stream': stream})
                assert response.status_code == 200, response.text
                assert 'hello' in response.text, (path, stream, response.text)
    assert len(requests) == 4


def test_jetbrains_google_nonstream_returns_transformed_result():
    import asyncio
    import httpx
    from openai_compatible_bridge.providers.jetbrains import JetBrainsChatClient
    async def run():
        upstream = JetBrainsChatClient(base_url='https://example.test', token='test-token')
        upstream.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={'candidates': [{'content': {'parts': [{'text': 'hello google'}]}, 'finishReason': 'STOP'}], 'usageMetadata': {'promptTokenCount': 1, 'candidatesTokenCount': 2, 'totalTokenCount': 3}})))
        try:
            result = await upstream.generate(model='gemini', messages=[{'role': 'user', 'content': 'hi'}], resolved_config={'protocol': 'google_generate_content'})
            assert result['text'] == 'hello google'
        finally:
            await upstream.close()
    asyncio.run(run())


def test_billing_compatibility_aliases_and_conflicts():
    import pytest
    from openai_compatible_bridge.core.async_cost import AsyncCostAccounting
    from openai_compatible_bridge.core.cost_tracking import CostConfigError
    accounting = AsyncCostAccounting(config=None, billing={'openai_compatible': 'subscription'})
    try:
        assert accounting.billing['openai_compat'] == 'subscription'
    finally:
        import asyncio
        asyncio.run(accounting.aclose())
    with pytest.raises(CostConfigError, match='conflicting'):
        AsyncCostAccounting(config=None, billing={'openai_compat': 'metered', 'openai_compatible': 'subscription'})
