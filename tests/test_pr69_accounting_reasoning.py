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
def test_jetbrains_ingress_rejects_reasoning_before_stream_or_upstream(stream, anthropic):
    fake = _FakeJetBrains()
    app = create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider, rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider, foundry_chat_client_factory=_FakeProvider, jetbrains_chat_client_factory=lambda: fake, cost_accounting_factory=lambda: None)
    body = {'model': 'jetbrains:native', 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': stream}
    if anthropic:
        body.update(max_tokens=100, output_config={'effort': 'high'})
    else:
        body['reasoning_effort'] = 'high'
    with TestClient(app) as client:
        response = client.post('/v1/messages' if anthropic else '/v1/chat/completions', json=body)
    assert response.status_code == 400, response.text
    assert 'reasoning' in response.text.lower() or 'effort' in response.text.lower()
    assert not fake.calls


@pytest.mark.parametrize('stream', [False, True])
def test_jetbrains_ingress_rejects_reasoning_object(stream):
    fake = _FakeJetBrains()
    app = create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider, rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider, foundry_chat_client_factory=_FakeProvider, jetbrains_chat_client_factory=lambda: fake, cost_accounting_factory=lambda: None)
    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json={'model': 'jetbrains:native', 'messages': [{'role': 'user', 'content': 'hi'}], 'stream': stream, 'reasoning': {'effort': 'high'}})
    assert response.status_code == 400
    assert not fake.calls


@pytest.mark.parametrize('protocol', ['openai_responses', 'xai_responses', 'anthropic_messages', 'google_generate_content'])
@pytest.mark.parametrize('stream', [False, True])
def test_jetbrains_direct_client_rejects_reasoning(protocol, stream):
    async def run():
        client = JetBrainsChatClient(base_url='https://example.test', token='synthetic')
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={'status': 'completed', 'output': []})))
        try:
            args = dict(model='native', messages=[], resolved_config={'protocol': protocol}, reasoning_effort='high')
            with pytest.raises(VertexAPIError) as exc:
                if stream:
                    _ = [event async for event in client.stream_chat(**args)]
                else:
                    await client.generate(**args)
            assert exc.value.code == 'unsupported_parameter'
        finally:
            await client.close()
    asyncio.run(run())
