import asyncio
from decimal import Decimal

import httpx
import pytest
from fastapi.testclient import TestClient

from openai_compatible_bridge.core.async_cost import AsyncCostAccounting
from openai_compatible_bridge.core.cost_tracking import CostBudgetExceeded, CostTrackingConfig, NormalizedUsage, PricingCatalog, SQLiteCostRepository
from openai_compatible_bridge.main import create_app
from openai_compatible_bridge.providers.jetbrains import JetBrainsChatClient
from openai_compatible_bridge.providers.vertex import VertexAPIError
from test_cost_backend import cost_env
from test_jetbrains_chat import _FakeProvider, _FakeJetBrains


@pytest.mark.parametrize('configured', ['openai_compat', 'openai_compatible', 'both'])
def test_alias_ledger_both_names_count_once_for_admission(tmp_path, configured):
    config = CostTrackingConfig.from_env(cost_env(tmp_path) | {'COST_SHORT_WINDOW_LIMIT_USD': '0.00025'})
    ledger = SQLiteCostRepository(config.ledger_path)
    ledger.initialize()
    pricing = PricingCatalog.from_config(config)
    billing = {name: 'metered' for name in ('openai_compat', 'openai_compatible')}
    seed = AsyncCostAccounting(config=config, ledger=ledger, pricing=pricing, billing=billing)
    usage = NormalizedUsage(prompt_tokens=100)
    for provider in billing:
        seed.gate.preflight(endpoint='chat', model='gemini-2.5-flash', forecast_usage=usage, provider=provider)
    selected = billing if configured == 'both' else {configured: 'metered'}
    accounting = AsyncCostAccounting(config=config, ledger=ledger, pricing=pricing, billing=selected)
    # Both existing reservations total .0002, not .0001 or .0004.
    with pytest.raises(CostBudgetExceeded) as exc:
        accounting.gate.preflight(endpoint='chat', model='gemini-2.5-flash', forecast_usage=usage, provider=configured if configured != 'both' else 'openai_compat')
    assert exc.value.block.current_estimated_spend == Decimal('0.0002')
    assert len([event for event in ledger.fetch_events() if event['status'] == 'reserved']) == 2
    ledger.close()
    asyncio.run(seed.aclose())
    asyncio.run(accounting.aclose())


@pytest.mark.parametrize('stream', [False, True])
@pytest.mark.parametrize('anthropic', [False, True])
def test_jetbrains_ingress_rejects_unmappable_reasoning_before_stream_or_upstream(stream, anthropic):
    fake = _FakeJetBrains()
    app = create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider, rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider, foundry_chat_client_factory=_FakeProvider, jetbrains_chat_client_factory=lambda: fake, cost_accounting_factory=lambda: None)
    body = {'model': 'jetbrains:native', 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': stream}
    if anthropic:
        body.update(max_tokens=100, output_config={'effort': 'ultracode'})
    else:
        body['reasoning_effort'] = 'ultracode'
    with TestClient(app) as client:
        response = client.post('/v1/messages' if anthropic else '/v1/chat/completions', json=body)
    assert response.status_code == 400, response.text
    assert 'reasoning' in response.text.lower() or 'effort' in response.text.lower()
    assert not fake.calls


@pytest.mark.parametrize('stream', [False, True])
def test_jetbrains_ingress_rejects_unmappable_reasoning_object(stream):
    fake = _FakeJetBrains()
    app = create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider, rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider, foundry_chat_client_factory=_FakeProvider, jetbrains_chat_client_factory=lambda: fake, cost_accounting_factory=lambda: None)
    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json={'model': 'jetbrains:native', 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': stream, 'reasoning': {'effort': 'ultracode'}})
    assert response.status_code == 400
    assert not fake.calls


@pytest.mark.parametrize('protocol', ['openai_responses', 'xai_responses', 'anthropic_messages', 'google_generate_content'])
@pytest.mark.parametrize('stream', [False, True])
def test_jetbrains_direct_client_rejects_unmappable_reasoning(protocol, stream):
    async def run():
        client = JetBrainsChatClient(base_url='https://example.test', token='synthetic')
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={'status': 'completed', 'output': []})))
        try:
            args = dict(model='native', messages=[], resolved_config={'protocol': protocol}, reasoning_effort='ultracode')
            with pytest.raises(VertexAPIError) as exc:
                if stream:
                    _ = [event async for event in client.stream_chat(**args)]
                else:
                    await client.generate(**args)
            assert exc.value.code == 'unsupported_parameter'
        finally:
            await client.close()
    asyncio.run(run())


@pytest.mark.parametrize('provider', ['openai_compat', 'openai_compatible'])
def test_admin_status_alias_includes_both_ledger_names(tmp_path, provider):
    config = CostTrackingConfig.from_env(cost_env(tmp_path) | {'COST_ADMIN_ENABLED': 'true', 'COST_ADMIN_API_KEY': 'synthetic-admin'})
    ledger = SQLiteCostRepository(config.ledger_path)
    ledger.initialize()
    accounting = AsyncCostAccounting(config=config, ledger=ledger, pricing=PricingCatalog.from_config(config), billing={'openai_compat': 'metered'})
    for name in ('openai_compat', 'openai_compatible'):
        accounting.gate.preflight(endpoint='chat', model='gemini-2.5-flash', forecast_usage=NormalizedUsage(prompt_tokens=100), provider=name)
    app = create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider, rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider, foundry_chat_client_factory=_FakeProvider, jetbrains_chat_client_factory=_FakeProvider, cost_accounting_factory=lambda: accounting)
    with TestClient(app) as client:
        response = client.get('/admin/cost/status', params={'provider': provider}, headers={'Authorization': 'Bearer synthetic-admin'})
        assert response.status_code == 200
        data = response.json()
        for window in ('short_window', 'daily'):
            assert Decimal(data[window]['estimated_spend']) == Decimal('0.0002')
            assert data[window]['event_count'] == 2
        assert data['view_provider'] == provider
        unrelated = client.get('/admin/cost/status', params={'provider': 'vertex'}, headers={'Authorization': 'Bearer synthetic-admin'}).json()
        assert Decimal(unrelated['daily']['estimated_spend']) == 0
        assert unrelated['view_provider'] == 'vertex'
    ledger.close()


@pytest.mark.parametrize('protocol,extra', [
    ('anthropic_messages', {'response_format': {'type': 'json_object'}}),
    ('anthropic_messages', {'reasoning_effort': 'ultracode'}),
    ('google_generate_content', {'reasoning_effort': 'ultracode'}),
])
@pytest.mark.parametrize('stream', [False, True])
def test_jetbrains_unsupported_parameters_rejected_before_sse(monkeypatch, protocol, extra, stream):
    import openai_compatible_bridge.providers.vertex as vertex
    monkeypatch.setitem(vertex.MODEL_REGISTRY, 'jetbrains:invalid', {'provider': 'jetbrains', 'kind': 'chat', 'provider_model': 'native', 'protocol': protocol})
    sent = []
    upstream = JetBrainsChatClient(base_url='https://example.test', token='synthetic')
    upstream.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: sent.append(request) or httpx.Response(200, json={})))
    app = create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider, rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider, foundry_chat_client_factory=_FakeProvider, jetbrains_chat_client_factory=lambda: upstream, cost_accounting_factory=lambda: None)
    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json={'model': 'jetbrains:invalid', 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': stream, **extra})
        assert response.status_code == 400, response.text
        assert response.json()['error']['code'] == 'unsupported_parameter'
        assert not sent


@pytest.mark.parametrize('stream', [False, True])
def test_jetbrains_bad_json_schema_fails_before_sse(monkeypatch, stream):
    import openai_compatible_bridge.providers.vertex as vertex
    monkeypatch.setitem(vertex.MODEL_REGISTRY, 'jetbrains:invalid-schema', {'provider': 'jetbrains', 'kind': 'chat', 'provider_model': 'native', 'protocol': 'openai_responses'})
    sent = []
    upstream = JetBrainsChatClient(base_url='https://example.test', token='synthetic')
    upstream.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: sent.append(request) or httpx.Response(200, json={})))
    app = create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider, rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider, foundry_chat_client_factory=_FakeProvider, jetbrains_chat_client_factory=lambda: upstream, cost_accounting_factory=lambda: None)
    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json={'model': 'jetbrains:invalid-schema', 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': stream, 'response_format': {'type': 'json_schema'}})
        assert response.status_code == 400, response.text
        assert not sent


def test_incomplete_jetbrains_stream_settles_sqlite_actual_cost(tmp_path):
    from openai_compatible_bridge.core.metered_http import MeteredHTTPClient
    async def run():
        config = CostTrackingConfig.from_env(cost_env(tmp_path))
        ledger = SQLiteCostRepository(config.ledger_path)
        ledger.initialize()
        accounting = AsyncCostAccounting(config=config, ledger=ledger, pricing=PricingCatalog.from_config(config), billing={'jetbrains': 'metered'})
        wire = 'data: {"type":"response.incomplete","response":{"status":"incomplete","usage":{"input_tokens":100,"output_tokens":50}}}\n\ndata: [DONE]\n\n'
        client = JetBrainsChatClient(base_url='https://example.test', token='synthetic')
        client.http = MeteredHTTPClient(httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, text=wire))), accounting, provider='jetbrains')
        try:
            async with accounting.reservation(endpoint='chat', model='gemini-2.5-flash', forecast_usage=NormalizedUsage(prompt_tokens=1), provider='jetbrains'):
                _ = [x async for x in client.stream_chat(model='m', messages=[{'role':'user','content':'hi'}], resolved_config={'protocol':'openai_responses'})]
            await accounting.flush()
            rows = ledger.fetch_events()
            assert len(rows) == 1
            assert rows[0]['status'] == 'finalized'
            assert Decimal(rows[0]['estimated_cost_usd']) == Decimal('0.0002')
        finally:
            await client.close()
            await accounting.aclose()
            ledger.close()
    asyncio.run(run())
