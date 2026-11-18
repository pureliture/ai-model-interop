import asyncio
import json

import httpx
import pytest

from openai_compatible_bridge.providers.jetbrains import JetBrainsChatClient
from openai_compatible_bridge.providers.vertex import VertexAPIError


def run_client(protocol, payload, *, stream=False, **kwargs):
    async def run():
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            if stream:
                return httpx.Response(200, text=''.join('data: ' + json.dumps(e) + '\n\n' for e in payload))
            return httpx.Response(200, json=payload)
        client = JetBrainsChatClient(base_url='https://example.test', token='test')
        await client.http.aclose()
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            params = dict(model='native', messages=[{'role': 'user', 'content': 'hello'}], resolved_config={'protocol': protocol}, **kwargs)
            result = [event async for event in client.stream_chat(**params)] if stream else await client.generate(**params)
            return result, requests[0]
        finally:
            await client.close()
    return asyncio.run(run())


@pytest.mark.parametrize('protocol,events', [
    ('anthropic_messages', [
        {'type':'message_start','message':{'usage':{'input_tokens':4}}},
        {'type':'content_block_start','index':0,'content_block':{'type':'tool_use','id':'call_1','name':'lookup','input':{}}},
        {'type':'content_block_delta','index':0,'delta':{'type':'input_json_delta','partial_json':'{"x":1}'}},
        {'type':'message_delta','delta':{'stop_reason':'tool_use'},'usage':{'output_tokens':3}},
        {'type':'message_stop'}]),
    ('openai_responses', [
        {'type':'response.output_item.added','output_index':0,'item':{'type':'function_call','id':'item_1','call_id':'call_1','name':'lookup'}},
        {'type':'response.function_call_arguments.delta','item_id':'item_1','output_index':0,'delta':'{"x":1}'},
        {'type':'response.completed','response':{'status':'completed','usage':{'input_tokens':4,'output_tokens':3,'total_tokens':7}}}]),
    ('xai_responses', [
        {'type':'response.output_item.added','output_index':0,'item':{'type':'function_call','id':'item_1','call_id':'call_1','name':'lookup'}},
        {'type':'response.function_call_arguments.delta','item_id':'item_1','output_index':0,'delta':'{"x":1}'},
        {'type':'response.completed','response':{'status':'completed','usage':{'input_tokens':4,'output_tokens':3,'total_tokens':7}}}]),
    ('google_generate_content', [
        {'candidates':[{'content':{'parts':[{'thought':True,'text':'SECRET THOUGHT'},{'functionCall':{'name':'lookup','args':{'x':1}}}]},'finishReason':'STOP'}], 'usageMetadata':{'promptTokenCount':4,'candidatesTokenCount':3,'totalTokenCount':7}}]),
])
def test_native_stream_tool_calls_usage_and_thought(protocol, events):
    result, _ = run_client(protocol, events, stream=True)
    calls = [call for event in result for call in event.get('delta_tool_calls', []) or []]
    assert calls, result
    assert any(call.get('function', {}).get('name') == 'lookup' for call in calls)
    assert any(event.get('finish_reason') == 'tool_calls' for event in result)
    assert any((event.get('usage') or {}).get('total_tokens') == 7 for event in result)
    assert 'SECRET THOUGHT' not in json.dumps(result)


@pytest.mark.parametrize('protocol,event', [
    ('anthropic_messages', {'type':'error','error':{'type':'overloaded_error','message':'failed'}}),
    ('openai_responses', {'type':'response.failed','response':{'error':{'code':'server_error','message':'failed'}}}),
    ('xai_responses', {'type':'response.failed','response':{'error':{'code':'server_error','message':'failed'}}}),
    ('google_generate_content', {'error':{'message':'failed','code':500}}),
])
def test_native_stream_errors_not_silently_successful(protocol, event):
    with pytest.raises(VertexAPIError):
        run_client(protocol, [event], stream=True)


@pytest.mark.parametrize('protocol,payload,reason', [
    ('anthropic_messages', {'content':[{'type':'text','text':'partial'}], 'stop_reason':'max_tokens'}, 'length'),
    ('openai_responses', {'output':[], 'status':'completed'}, 'stop'),
    ('xai_responses', {'output':[], 'status':'incomplete','incomplete_details':{'reason':'max_output_tokens'}}, 'length'),
])
def test_native_nonstream_finish_reason(protocol, payload, reason):
    result, _ = run_client(protocol, payload)
    assert result['finish_reason'] == reason


@pytest.mark.parametrize('protocol,payload', [
    ('openai_responses', {'status':'failed','error':{'message':'failed'}}),
    ('xai_responses', {'status':'failed','error':{'message':'failed'}}),
    ('anthropic_messages', {'type':'error','error':{'message':'failed'}}),
    ('google_generate_content', {'error':{'message':'failed'}}),
])
def test_native_nonstream_error_never_success(protocol, payload):
    with pytest.raises(VertexAPIError):
        run_client(protocol, payload)


@pytest.mark.parametrize('stream', [False, True])
def test_responses_json_schema_shape(stream):
    fmt = {'type':'json_schema','json_schema':{'name':'result','strict':True,'schema':{'type':'object','properties':{},'additionalProperties':False}}}
    payload = [{'type':'response.completed','response':{'status':'completed'}}] if stream else {'output':[], 'status':'completed'}
    _, body = run_client('openai_responses', payload, stream=stream, response_format=fmt)
    assert body['text']['format'] == {'type':'json_schema', **fmt['json_schema']}


@pytest.mark.parametrize('protocol', ['anthropic_messages', 'google_generate_content'])
def test_unsupported_parallel_option_rejected(protocol):
    with pytest.raises(VertexAPIError) as exc:
        run_client(protocol, {}, parallel_tool_calls=False)
    assert exc.value.code == 'unsupported_parameter'


def test_old_module_import_compatibility():
    from openai_compatible_bridge.providers.openai_compat import OpenAICompatibleChatClient as old
    from openai_compatible_bridge.providers.openai_compatible import OpenAICompatibleChatClient as new
    assert old is new


@pytest.mark.parametrize('protocol,payload,events', [
    ('anthropic_messages', {'content':[{'type':'tool_use','id':'c1','name':'lookup','input':{'x':1}}], 'stop_reason':'tool_use','usage':{'input_tokens':4,'output_tokens':3}}, [
        {'type':'message_start','message':{'usage':{'input_tokens':4}}}, {'type':'content_block_start','index':0,'content_block':{'type':'tool_use','id':'c1','name':'lookup'}}, {'type':'content_block_delta','index':0,'delta':{'type':'input_json_delta','partial_json':'{"x":1}'}}, {'type':'message_delta','delta':{'stop_reason':'tool_use'},'usage':{'output_tokens':3}}, {'type':'message_stop'}]),
    *[(protocol, {'output':[{'type':'function_call','call_id':'c1','name':'lookup','arguments':'{"x":1}'}], 'status':'completed','usage':{'input_tokens':4,'output_tokens':3}}, [
        {'type':'response.output_item.added','output_index':0,'item':{'type':'function_call','id':'i1','call_id':'c1','name':'lookup'}}, {'type':'response.function_call_arguments.delta','item_id':'i1','output_index':0,'delta':'{"x":1}'}, {'type':'response.completed','response':{'status':'completed','usage':{'input_tokens':4,'output_tokens':3}}}]) for protocol in ('openai_responses','xai_responses')],
    ('google_generate_content', {'candidates':[{'content':{'parts':[{'thought':True,'text':'hidden'},{'functionCall':{'name':'lookup','args':{'x':1}}}]},'finishReason':'STOP'}], 'usageMetadata':{'promptTokenCount':4,'candidatesTokenCount':3,'totalTokenCount':7}}, [
        {'candidates':[{'content':{'parts':[{'thought':True,'text':'hidden'},{'functionCall':{'name':'lookup','args':{'x':1}}}]},'finishReason':'STOP'}], 'usageMetadata':{'promptTokenCount':4,'candidatesTokenCount':3,'totalTokenCount':7}}]),
])
def test_real_provider_through_openai_ingress(monkeypatch, protocol, payload, events):
    from fastapi.testclient import TestClient
    import openai_compatible_bridge.providers.vertex as vertex
    from openai_compatible_bridge.main import create_app
    from test_jetbrains_chat import _FakeProvider
    monkeypatch.setitem(vertex.MODEL_REGISTRY, 'jetbrains:integration', {'provider':'jetbrains','kind':'chat','provider_model':'native','protocol':protocol})
    def handler(request):
        body = json.loads(request.content)
        is_stream = body.get('stream') or 'streamGenerateContent' in str(request.url)
        if is_stream:
            return httpx.Response(200, text=''.join('data: '+json.dumps(event)+'\n\n' for event in events))
        return httpx.Response(200, json=payload)
    upstream = JetBrainsChatClient(base_url='https://example.test', token='test')
    asyncio.run(upstream.http.aclose())
    upstream.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider, rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider, foundry_chat_client_factory=_FakeProvider, jetbrains_chat_client_factory=lambda: upstream, cost_accounting_factory=lambda: None)
    with TestClient(app) as client:
        for stream in (False, True):
            body = {'model':'jetbrains:integration','messages':[{'role':'user','content':'hi'}], 'stream':stream}
            if stream:
                body['stream_options'] = {'include_usage':True}
            response = client.post('/v1/chat/completions', json=body)
            assert response.status_code == 200, response.text
            assert 'lookup' in response.text and 'tool_calls' in response.text
            assert 'hidden' not in response.text
            assert '"total_tokens": 7' in response.text or '"total_tokens":7' in response.text


def test_http_ingress_does_not_turn_stream_error_into_stop(monkeypatch):
    from fastapi.testclient import TestClient
    import openai_compatible_bridge.providers.vertex as vertex
    from openai_compatible_bridge.main import create_app
    from test_jetbrains_chat import _FakeProvider
    monkeypatch.setitem(vertex.MODEL_REGISTRY, 'jetbrains:error', {'provider':'jetbrains','kind':'chat','provider_model':'native','protocol':'openai_responses'})
    event = {'type':'response.failed','response':{'error':{'code':'upstream_failure','message':'failure'}}}
    upstream = JetBrainsChatClient(base_url='https://example.test', token='test')
    asyncio.run(upstream.http.aclose())
    upstream.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, text='data: '+json.dumps(event)+'\n\n')))
    app = create_app(embedding_client_factory=_FakeProvider, chat_client_factory=_FakeProvider, rerank_client_factory=_FakeProvider, ollama_chat_client_factory=_FakeProvider, foundry_chat_client_factory=_FakeProvider, jetbrains_chat_client_factory=lambda: upstream, cost_accounting_factory=lambda: None)
    with TestClient(app) as client:
        response = client.post('/v1/chat/completions', json={'model':'jetbrains:error','messages':[{'role':'user','content':'hi'}],'stream':True})
        assert 'upstream_failure' in response.text
        assert '"finish_reason": "stop"' not in response.text


def test_responses_parallel_option_forwarded():
    _, body = run_client('openai_responses', {'output':[], 'status':'completed'}, parallel_tool_calls=False)
    assert body['parallel_tool_calls'] is False
