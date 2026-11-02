from __future__ import annotations

import asyncio
import json
from decimal import Decimal

import httpx
import pytest

from openai_compatible_bridge.core.cost_tracking import CostReservation, NormalizedUsage
from openai_compatible_bridge.core.metered_http import (
    MeteredHTTPClient,
    _raw_usage,
    _StreamUsage,
)
from openai_compatible_bridge.providers.foundry import FoundryChatClient
from openai_compatible_bridge.providers.ollama import OllamaChatClient
from openai_compatible_bridge.providers.vertex import (
    VertexAPIError,
    VertexChatClient,
    VertexEmbeddingClient,
    VertexRerankClient,
)

MESSAGES = [{"role": "user", "content": "hello"}]
CHAT_USAGE = NormalizedUsage(prompt_tokens=3, completion_tokens=5, total_tokens=8)


@pytest.mark.parametrize("key", ["tokenCount", "token_count", "inputTokens", "input_tokens"])
def test_embedding_metadata_aliases_preserve_known_and_unknown_usage(key):
    for value in (0, 3):
        assert _raw_usage({"usageMetadata": {key: value}}, "embeddings") == NormalizedUsage(
            embedding_tokens=value, total_tokens=value,
        )
    for value in (-1, True, "3", None):
        assert _raw_usage({"usageMetadata": {key: value}}, "embeddings") is None


class Accounting:
    def __init__(self, endpoint="chat", *, exempt=False, reject_at=None):
        self.endpoint = endpoint
        self.exempt = exempt
        self.reject_at = reject_at
        self.before = []
        self.records = []
        self.events = []

    async def before_attempt(self, provider):
        self.before.append(provider)
        self.events.append("before")
        if len(self.before) == self.reject_at:
            raise PermissionError("admission rejected")
        if self.exempt:
            return None
        return CostReservation(
            reservation_id=str(len(self.before)),
            internal_request_id="logical-request",
            provider=provider,
            endpoint=self.endpoint,
            model="logical-model",
            forecast_cost_usd=Decimal(1),
            currency="USD",
            pricing_source="test",
            pricing_version="1",
            window_started_at="2026-09-26T00:00:00+00:00",
            created_at="2026-09-26T01:00:00+00:00",
        )

    def record_attempt(self, reservation, usage):
        self.events.append("record")
        self.records.append((reservation, usage))


class Tokens:
    project_id = "test-project"

    def __init__(self, accounting):
        self.accounting = accounting

    async def get_token(self):
        self.accounting.events.append("token")
        return "test-token"


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, *chunks, error=None):
        self.chunks = chunks
        self.error = error
        self.reads = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            self.reads += 1
            yield chunk
        if self.error is not None:
            raise self.error

    async def aclose(self):
        self.closed = True


async def wrap(client, accounting, handler, provider):
    await client.http.aclose()
    client.http = MeteredHTTPClient(
        httpx.AsyncClient(transport=httpx.MockTransport(handler)), accounting, provider=provider
    )
    return client


def sse(*events):
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()


@pytest.mark.parametrize(
    "payload, endpoint, expected",
    [
        ({"usage": {"prompt_tokens": 3, "completion_tokens": 5}}, "chat", CHAT_USAGE),
        ({"usage": {"input_tokens": 3, "output_tokens": 5}}, "chat", CHAT_USAGE),
        ({"prompt_eval_count": 3, "eval_count": 5}, "chat", CHAT_USAGE),
        ({"usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "thoughtsTokenCount": 3}}, "chat", CHAT_USAGE),
        ({"usage": {"prompt_tokens": 0, "completion_tokens": 0}}, "chat", NormalizedUsage()),
        ({"usage": {"prompt_tokens": 3, "total_tokens": 3}}, "embeddings", NormalizedUsage(embedding_tokens=3, total_tokens=3)),
        ({"embedding": {"statistics": {"token_count": 0}}}, "embeddings", NormalizedUsage()),
        ({"usageMetadata": {"promptTokenCount": 3}}, "embeddings", NormalizedUsage(embedding_tokens=3, total_tokens=3)),
        ({"usageMetadata": {"totalTokenCount": 3}}, "embeddings", NormalizedUsage(embedding_tokens=3, total_tokens=3)),
        ({"predictions": [{"embeddings": {"statistics": {"token_count": 1}}}, {"embeddings": {"statistics": {"token_count": 2}}}]}, "embeddings", NormalizedUsage(embedding_tokens=3, total_tokens=3)),
        ({"records": [{"id": "1", "score": 0.8}]}, "rerank", None),
        ({"usage": {"rerank_units": 0}}, "rerank", NormalizedUsage()),
        ({"usage": {"total_tokens": 8}}, "chat", None),
        ({"usage": {"prompt_tokens": 3}}, "chat", None),
        ({"usage": {"output_tokens": 5}}, "chat", None),
        ({"usageMetadata": {"promptTokenCount": 3, "thoughtsTokenCount": 5}}, "chat", None),
        ({"predictions": [{"embeddings": {"statistics": {"token_count": 3}}}, {"embeddings": {}}]}, "embeddings", None),
        ({"predictions": []}, "embeddings", None),
        ({"usage": {}}, "chat", None),
        ({}, "chat", None),
        ([], "chat", None),
        ({"error": {"message": "failed"}, "usage": {"prompt_tokens": 3, "completion_tokens": 5}}, "chat", None),
    ],
)
def test_raw_usage_requires_explicit_endpoint_dimensions(payload, endpoint, expected):
    assert _raw_usage(payload, endpoint) == expected


@pytest.mark.parametrize("invalid", [None, True, False, -1, 1.5, "3", [], {}])
@pytest.mark.parametrize("field", ["prompt_tokens", "completion_tokens", "total_tokens"])
def test_raw_usage_rejects_malformed_counts(invalid, field):
    usage = {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8}
    usage[field] = invalid
    assert _raw_usage({"usage": usage}, "chat") is None


@pytest.mark.parametrize("api, count", [("predict", 2), ("embedContent", 3)])
def test_vertex_parallel_embeddings_gate_each_actual_send(api, count):
    async def run():
        accounting = Accounting("embeddings")
        sent = []
        all_started = asyncio.Event()

        async def handler(request):
            assert accounting.events[-1] == "before"
            accounting.events.append("send")
            sent.append(request)
            if len(sent) == count:
                all_started.set()
            await asyncio.wait_for(all_started.wait(), 2)
            body = json.loads(request.content)
            embedding = {"values": [0.5], "statistics": {"token_count": 2}}
            if api == "predict":
                payload = {"predictions": [{"embeddings": embedding} for _ in body["instances"]]}
            else:
                payload = {"embedding": embedding}
            return httpx.Response(200, json=payload)

        client = await wrap(VertexEmbeddingClient(Tokens(accounting)), accounting, handler, "vertex")
        try:
            result = await client.embed(
                model="logical-model", texts=["a", "b", "c"], dimensions=None,
                task_type="RETRIEVAL_DOCUMENT", title=None,
                resolved_config={"api": api, "max_instances": 2, "location": "global"},
            )
            assert len(result) == 3
            assert len(sent) == len(accounting.before) == len(accounting.records) == count
            assert len({r.reservation_id for r, _ in accounting.records}) == count
            assert sum(u.embedding_tokens for _, u in accounting.records) == 6
            assert all(r.model == "logical-model" for r, _ in accounting.records)
            assert all(request.url.path.endswith(f":{api}") for request in sent)
            assert accounting.events.count("token") == count
        finally:
            await client.close()

    asyncio.run(run())


def test_vertex_rank_does_not_fabricate_usage():
    async def run():
        accounting = Accounting("rerank")

        def handler(request):
            assert request.url.host == "discoveryengine.googleapis.com"
            assert accounting.events == ["token", "before"]
            return httpx.Response(200, json={"records": [{"id": "a", "score": 0.9}]})

        client = await wrap(VertexRerankClient(Tokens(accounting)), accounting, handler, "vertex")
        try:
            result = await client.rank(model="ranker", query="q", records=[{"id": "a", "content": "a"}])
            assert result == [{"id": "a", "score": 0.9}]
            assert [u for _, u in accounting.records] == [None]
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("fallback", [False, True])
def test_vertex_stream_and_post_fallback_are_both_metered(fallback):
    async def run():
        accounting = Accounting()

        def handler(request):
            accounting.events.append("send")
            if fallback:
                assert request.url.path.endswith("/chat/completions")
                return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 5}})
            assert request.url.path.endswith(":streamGenerateContent")
            return httpx.Response(200, stream=ByteStream(sse({
                "candidates": [{"content": {"parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "thoughtsTokenCount": 3},
            })))

        client = await wrap(VertexChatClient(Tokens(accounting)), accounting, handler, "vertex")
        try:
            config = {"api": "openapiChatCompletions" if fallback else "generateContent"}
            result = [item async for item in client.stream_chat(model="m", messages=MESSAGES, resolved_config=config)]
            assert "".join(item["delta_text"] for item in result) == "ok"
            assert accounting.events == ["token", "before", "send", "record"]
            assert [u for _, u in accounting.records] == [CHAT_USAGE]
        finally:
            await client.close()

    asyncio.run(run())


PROTOCOL_EVENTS = [
    ("openai_chat_completions", [
        {"choices": [{"delta": {"content": "ok"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 1}},
        {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 5}},
    ]),
    ("anthropic_messages", [
        {"type": "message_start", "message": {"usage": {"input_tokens": 3, "output_tokens": 0}}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "ok"}},
        {"type": "message_delta", "delta": {}, "usage": {"output_tokens": 1}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}},
        {"type": "message_stop"},
    ]),
    ("xai_responses", [
        {"type": "response.output_text.delta", "delta": "ok"},
        {"type": "response.completed", "response": {"status": "completed", "usage": {"input_tokens": 3, "output_tokens": 5}}},
    ]),
    ("openai_responses", [
        {"type": "response.output_text.delta", "delta": "ok"},
        {"type": "response.completed", "response": {"status": "completed", "usage": {"input_tokens": 3, "output_tokens": 5}}},
    ]),
    ("google_generate_content", [
        {"candidates": [{"content": {"parts": [{"text": "ok"}]}}], "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1}},
        {"candidates": [{"finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "thoughtsTokenCount": 3}},
    ]),
]


@pytest.mark.parametrize("protocol, events", PROTOCOL_EVENTS)
def test_foundry_real_protocol_streams_keep_last_cumulative_usage(protocol, events):
    async def run():
        accounting = Accounting()
        wire = sse(*events) + b"data: [DONE]\n\n"
        stream = ByteStream(*(wire[i:i + 7] for i in range(0, len(wire), 7)))

        def handler(request):
            assert accounting.events == ["before"]
            accounting.events.append("send")
            return httpx.Response(200, stream=stream)

        client = await wrap(
            FoundryChatClient(base_url="https://foundry.test/api/v2/llm/proxy/openai/v1/chat/completions", token="test"),
            accounting, handler, "foundry",
        )
        try:
            result = [item async for item in client.stream_chat(model="m", messages=MESSAGES, resolved_config={"protocol": protocol})]
            assert "".join(item["delta_text"] for item in result) == "ok"
            assert accounting.events == ["before", "send", "record"]
            assert [u for _, u in accounting.records] == [CHAT_USAGE]
            assert stream.closed
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("protocol", [protocol for protocol, _ in PROTOCOL_EVENTS])
def test_foundry_real_protocol_posts_observe_original_usage(protocol):
    async def run():
        accounting = Accounting()
        payloads = {
            "openai_chat_completions": {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 5},
            },
            "anthropic_messages": {
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 3, "output_tokens": 5},
            },
            "google_generate_content": {
                "candidates": [{"content": {"parts": [{"text": "ok"}]}}],
                "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2, "thoughtsTokenCount": 3},
            },
        }
        responses_payload = {
            "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
            "usage": {"input_tokens": 3, "output_tokens": 5},
        }

        def handler(request):
            accounting.events.append("send")
            return httpx.Response(200, json=payloads.get(protocol, responses_payload))

        client = await wrap(
            FoundryChatClient(base_url="https://foundry.test/api/v2/llm/proxy/openai/v1/chat/completions", token="test"),
            accounting, handler, "foundry",
        )
        try:
            result = await client.generate(model="m", messages=MESSAGES, resolved_config={"protocol": protocol})
            assert result["text"] == "ok"
            assert accounting.events == ["before", "send", "record"]
            assert [u for _, u in accounting.records] == [CHAT_USAGE]
        finally:
            await client.close()

    asyncio.run(run())


def test_ollama_schema_repair_attempts_each_gate_and_record():
    async def run():
        accounting = Accounting()
        sent = []

        def handler(request):
            assert len(accounting.before) == len(sent) + 1
            sent.append(json.loads(request.content))
            return httpx.Response(200, json={
                "message": {"content": '"bad"' if len(sent) == 1 else '{"ok": true}'},
                "prompt_eval_count": 3, "eval_count": 5,
            })

        client = await wrap(OllamaChatClient(base_url="https://ollama.test"), accounting, handler, "ollama")
        response_format = {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}}
        try:
            with pytest.raises(VertexAPIError, match="did not satisfy"):
                await client.generate(model="m", messages=MESSAGES, response_format=response_format)
            result = await client.generate(model="m", messages=MESSAGES + [{"role": "user", "content": "repair"}], response_format=response_format)
            assert result["text"] == '{"ok": true}'
            assert len(sent) == 2
            assert len(sent[1]["messages"]) == 2
            assert [u for _, u in accounting.records] == [CHAT_USAGE, CHAT_USAGE]
            assert [r.reservation_id for r, _ in accounting.records] == ["1", "2"]
        finally:
            await client.close()

    asyncio.run(run())


def test_rejected_ollama_repair_does_not_repeat_upstream():
    async def run():
        accounting = Accounting(reject_at=2)
        sent = []

        def handler(request):
            sent.append(request)
            return httpx.Response(200, json={
                "message": {"content": '"bad"'}, "prompt_eval_count": 3, "eval_count": 5,
            })

        client = await wrap(OllamaChatClient(base_url="https://ollama.test"), accounting, handler, "ollama")
        response_format = {"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}}
        try:
            with pytest.raises(VertexAPIError, match="did not satisfy"):
                await client.generate(model="m", messages=MESSAGES, response_format=response_format)
            with pytest.raises(PermissionError):
                await client.generate(model="m", messages=MESSAGES, response_format=response_format)
            assert len(sent) == len(accounting.records) == 1
            assert len(accounting.before) == 2
        finally:
            await client.close()

    asyncio.run(run())


def test_ollama_ndjson_stream():
    async def run():
        accounting = Accounting()
        wire = b'{"message":{"content":"ok"},"done":false}\n' + b'{"done":true,"prompt_eval_count":3,"eval_count":5,"done_reason":"stop"}\n'
        client = await wrap(OllamaChatClient(base_url="https://ollama.test"), accounting, lambda _: httpx.Response(200, stream=ByteStream(wire[:13], wire[13:])), "ollama")
        try:
            result = [item async for item in client.stream_chat(model="m", messages=MESSAGES)]
            assert "".join(item["delta_text"] for item in result) == "ok"
            assert [u for _, u in accounting.records] == [CHAT_USAGE]
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("streaming", [False, True])
def test_rejection_never_sends_or_records(streaming):
    async def run():
        accounting = Accounting(reject_at=1)
        sent = []
        client = await wrap(OllamaChatClient(base_url="https://ollama.test"), accounting, lambda request: sent.append(request), "ollama")
        try:
            with pytest.raises(PermissionError, match="admission rejected"):
                if streaming:
                    _ = [item async for item in client.stream_chat(model="m", messages=MESSAGES)]
                else:
                    await client.generate(model="m", messages=MESSAGES)
            assert accounting.before == ["ollama"]
            assert sent == accounting.records == []
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("failure", ["status", "timeout", "cancel"])
def test_attempt_failure_preserves_exception_and_records_unknown_once(streaming, failure):
    async def run():
        accounting = Accounting()
        error = httpx.ReadTimeout("timeout") if failure == "timeout" else asyncio.CancelledError()

        def handler(request):
            accounting.events.append("send")
            if failure != "status":
                raise error
            return httpx.Response(503, json={"usage": {"prompt_tokens": 3, "completion_tokens": 5}})

        client = MeteredHTTPClient(httpx.AsyncClient(transport=httpx.MockTransport(handler)), accounting, provider="p")
        try:
            async def call():
                if streaming:
                    async with client.stream("POST", "https://upstream.test") as response:
                        assert response.status_code == 503
                        await response.aread()
                        assert response.json()["usage"]["prompt_tokens"] == 3
                else:
                    response = await client.post("https://upstream.test")
                    assert response.status_code == 503

            if failure == "status":
                await call()
            else:
                with pytest.raises(type(error)) as caught:
                    await call()
                assert caught.value is error
            assert accounting.events == ["before", "send", "record"]
            assert [u for _, u in accounting.records] == [None]
        finally:
            await client.aclose()
        assert client.is_closed

    asyncio.run(run())


@pytest.mark.parametrize("exempt", [False, True])
def test_missing_raw_usage_is_not_provider_fake_zero(exempt):
    async def run():
        accounting = Accounting(exempt=exempt)
        client = await wrap(OllamaChatClient(base_url="https://ollama.test"), accounting, lambda _: httpx.Response(200, json={"message": {"content": "ok"}}), "ollama")
        try:
            result = await client.generate(model="m", messages=MESSAGES)
            assert result["usage"]["total_tokens"] == 0
            assert [u for _, u in accounting.records] == ([] if exempt else [None])
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("ending", ["exhausted", "early", "error", "cancel"])
def test_stream_observation_never_reads_ahead_and_only_finalizes_completion(ending):
    async def run():
        accounting = Accounting()
        error = httpx.ReadError("broken") if ending == "error" else asyncio.CancelledError() if ending == "cancel" else None
        stream = ByteStream(sse({"usage": {"prompt_tokens": 3, "completion_tokens": 5}}), b": keepalive\n\n", error=error)
        client = MeteredHTTPClient(httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))), accounting, provider="p")
        context = client.stream("POST", "https://upstream.test")
        assert accounting.before == []
        try:
            response = await context.__aenter__()
            assert stream.reads == 0
            lines = response.aiter_lines()
            first = await anext(lines)
            assert first.startswith("data:")
            assert stream.reads == 1
            assert accounting.records == []
            if ending != "early":
                if error is None:
                    _ = [line async for line in lines]
                else:
                    with pytest.raises(type(error)):
                        _ = [line async for line in lines]
            await context.__aexit__(None, None, None)
            await lines.aclose()
            assert stream.closed
            assert response._observer.buffered_bytes == 0
            assert [u for _, u in accounting.records] == [CHAT_USAGE if ending == "exhausted" else None]
        finally:
            await client.aclose()

    asyncio.run(run())


def test_bounded_multiline_sse_and_malformed_final_snapshot():
    observer = _StreamUsage("chat")
    for line in ['event: message', 'data: {"usage":', 'data: {"prompt_tokens": 3, "completion_tokens": 5}}', '']:
        observer.feed_line(line)
    assert observer.usage == CHAT_USAGE
    for line in ['data: {"usage":{"prompt_tokens":3,"completion_tokens":true}}', '']:
        observer.feed_line(line)
    assert observer.usage is None

    observer = _StreamUsage("chat")
    for _ in range(100):
        observer.feed_line('data: ' + 'x' * 1024)
        assert observer.buffered_bytes <= 65536
    observer.feed_line('')
    for line in sse({"usage": {"prompt_tokens": 3, "completion_tokens": 5}}).decode().splitlines():
        observer.feed_line(line)
    assert observer.usage is None


@pytest.mark.parametrize("payload", [
    {"usageMetadata": {"promptTokenCount": True, "candidatesTokenCount": 5}},
    {"usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 5, "thoughtsTokenCount": "2"}},
    {"prompt_eval_count": 3, "eval_count": -1},
    {"usage": {"input_tokens": 3.0, "output_tokens": 5}},
])
def test_other_protocols_reject_invalid_dimensions(payload):
    assert _raw_usage(payload, "chat") is None


@pytest.mark.parametrize("wire", [
    b"data: not-json\n\n",
    b'data: {"usage":{"prompt_tokens":3}}\n\n',
    b'data: {"type":"message_start","message":{"usage":{"input_tokens":3,"output_tokens":0}}}\n\n',
    b'data: {"type":"response.completed","response":{"usage":null}}\n\n',
    sse({"usage": {"prompt_tokens": 3, "completion_tokens": 5}}, {"error": {"message": "failed"}}),
    sse({"usage": {"prompt_tokens": 3, "completion_tokens": 5}}, {"type": "response.incomplete"}),
    b'{"done":false,"prompt_eval_count":3,"eval_count":5}\n',
    b'{"done":true,"prompt_eval_count":3,"eval_count":5}\nnot-json\n',
    b'data: {"content":"' + b"x" * 65536 + b'"}\n\n',
], ids=["invalid-json", "missing-output", "anthropic-start-only", "responses-null", "error", "incomplete", "ollama-not-done", "invalid-ndjson-tail", "oversized"])
def test_unknown_stream_usage_keeps_original_lines(wire):
    async def run():
        accounting = Accounting()
        stream = ByteStream(wire)
        client = MeteredHTTPClient(httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))), accounting, provider="p")
        try:
            async with client.stream("POST", "https://upstream.test") as response:
                lines = [line async for line in response.aiter_lines()]
                assert lines == wire.decode().splitlines()
            assert [u for _, u in accounting.records] == [None]
        finally:
            await client.aclose()

    asyncio.run(run())


def test_cancelling_real_provider_consumer_closes_and_preserves_reservation():
    async def run():
        accounting = Accounting()
        yielded = asyncio.Event()
        blocked = asyncio.Event()
        stream = ByteStream(sse({
            "choices": [{"delta": {"content": "ok"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 5},
        }))
        client = await wrap(
            FoundryChatClient(base_url="https://foundry.test/chat", token="test"),
            accounting, lambda _: httpx.Response(200, stream=stream), "foundry",
        )

        async def consume():
            events = client.stream_chat(model="m", messages=MESSAGES)
            try:
                async for _ in events:
                    yielded.set()
                    await blocked.wait()
            finally:
                await events.aclose()

        try:
            task = asyncio.create_task(consume())
            await asyncio.wait_for(yielded.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert stream.closed
            assert [u for _, u in accounting.records] == [None]
            assert accounting.before == ["foundry"]
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("streaming", [False, True])
def test_exempt_response_bypasses_usage_observation(streaming):
    async def run():
        accounting = Accounting(exempt=True)
        original = httpx.Response(200, json={"usage": {"prompt_tokens": 3, "completion_tokens": 5}})
        client = MeteredHTTPClient(httpx.AsyncClient(transport=httpx.MockTransport(lambda _: original)), accounting, provider="subscription")
        try:
            if streaming:
                async with client.stream("POST", "https://upstream.test") as response:
                    assert response is original
                    await response.aread()
            else:
                assert await client.post("https://upstream.test") is original
            assert accounting.records == []
            assert accounting.before == ["subscription"]
        finally:
            await client.aclose()

    asyncio.run(run())


def test_post_parse_and_record_failures_do_not_change_upstream_response():
    async def run():
        accounting = Accounting()

        def record(reservation, usage):
            accounting.records.append((reservation, usage))
            raise RuntimeError("record failed")

        accounting.record_attempt = record
        original = httpx.Response(200, content=b"invalid json")
        sent = []

        def handler(request):
            sent.append(request)
            return original

        client = MeteredHTTPClient(httpx.AsyncClient(transport=httpx.MockTransport(handler)), accounting, provider="p")
        try:
            response = await client.post("https://upstream.test")
            assert response is original
            assert response.content == b"invalid json"
            assert len(sent) == 1
            assert [u for _, u in accounting.records] == [None]
        finally:
            await client.aclose()

    asyncio.run(run())

def test_jetbrains_incomplete_stream_records_actual_usage():
    from openai_compatible_bridge.providers.jetbrains import JetBrainsChatClient
    async def run():
        accounting = Accounting()
        wire = sse({'type': 'response.incomplete', 'response': {'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'}, 'usage': {'input_tokens': 100, 'output_tokens': 50, 'total_tokens': 150}}}) + b'data: [DONE]\n\n'
        client = await wrap(JetBrainsChatClient(base_url='https://example.test', token='synthetic'), accounting, lambda _: httpx.Response(200, stream=ByteStream(wire)), 'jetbrains')
        try:
            _ = [event async for event in client.stream_chat(model='m', messages=MESSAGES, resolved_config={'protocol': 'openai_responses'})]
            assert accounting.records[0][1] == NormalizedUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        finally:
            await client.close()
    asyncio.run(run())


@pytest.mark.parametrize('response', [
    {'status': 'incomplete', 'usage': {'input_tokens': 100}},
    {'status': 'incomplete', 'usage': {'input_tokens': True, 'output_tokens': 50}},
    {'status': 'incomplete', 'error': {'message': 'failed'}, 'usage': {'input_tokens': 100, 'output_tokens': 50}},
    {'status': 'failed', 'usage': {'input_tokens': 100, 'output_tokens': 50}},
])
def test_incomplete_invalid_usage_cannot_be_replaced_by_later_data(response):
    observer = _StreamUsage('chat', 'jetbrains')
    wire = sse({'type': 'response.incomplete', 'response': response}, {'usage': {'prompt_tokens': 1, 'completion_tokens': 1}})
    for line in wire.decode().splitlines():
        observer.feed_line(line)
    assert observer.usage is None


def test_jetbrains_incomplete_nonstream_records_actual_usage():
    from openai_compatible_bridge.providers.jetbrains import JetBrainsChatClient
    async def run():
        accounting = Accounting()
        payload = {'status': 'incomplete', 'output': [], 'incomplete_details': {'reason': 'max_output_tokens'}, 'usage': {'input_tokens': 100, 'output_tokens': 50, 'total_tokens': 150}}
        client = await wrap(JetBrainsChatClient(base_url='https://example.test', token='synthetic'), accounting, lambda _: httpx.Response(200, json=payload), 'jetbrains')
        try:
            await client.generate(model='m', messages=MESSAGES, resolved_config={'protocol': 'openai_responses'})
            assert accounting.records[0][1] == NormalizedUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        finally:
            await client.close()
    asyncio.run(run())
