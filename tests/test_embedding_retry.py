from __future__ import annotations

import asyncio
import json
import socket
from dataclasses import replace
from datetime import UTC, datetime
from email.utils import format_datetime

import httpx
import pytest
from test_cost_backend import NoCallProvider
from test_metered_http import Accounting, Tokens, wrap

from openai_compatible_bridge import main
from openai_compatible_bridge.core import embedding_resilience as resilience
from openai_compatible_bridge.core.cost_tracking import CostConfigError, CostSubsystemUnhealthy, DisabledCostAccounting
from openai_compatible_bridge.embedding_limits import EmbeddingRequestLimits
from openai_compatible_bridge.providers import vertex


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for name in (
        "EMBEDDINGS_MAX_INFLIGHT", "EMBEDDINGS_RETRY_MAX_ATTEMPTS",
        "EMBEDDINGS_RETRY_INITIAL_DELAY_SECONDS", "EMBEDDINGS_RETRY_MAX_DELAY_SECONDS",
        "EMBEDDINGS_RETRY_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    def blocked(*args, **kwargs):
        pytest.fail("임베딩 재시도 테스트는 외부 연결을 열 수 없습니다")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(main, "BRIDGE_API_KEY", "")


@pytest.fixture(params=["predict", "embedContent"])
def api(request):
    return request.param


@pytest.fixture
def clock(monkeypatch):
    elapsed = [100.0]
    delays = []

    async def sleep(delay):
        delays.append(delay)
        elapsed[0] += delay
        await asyncio.sleep(0)

    monkeypatch.setattr(resilience, "sleep", sleep)
    monkeypatch.setattr(resilience, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(resilience, "uniform", lambda low, high: 1.0)
    return delays


def success(request):
    embedding = {"values": [0.5], "statistics": {"token_count": 2}}
    if request.url.path.endswith(":predict"):
        body = json.loads(request.content)
        return httpx.Response(200, json={"predictions": [
            {"embeddings": embedding} for _ in body["instances"]
        ]})
    return httpx.Response(200, json={"embedding": embedding, "usageMetadata": {"tokenCount": 2}})


async def embed(client, api):
    return await client.embed(
        model="logical-model", texts=["x"], dimensions=None,
        task_type="RETRIEVAL_DOCUMENT", title=None,
        resolved_config={"api": api, "max_instances": 1, "location": "global"},
    )


def test_exponential_retry_gates_each_transmission(api, clock):
    async def run():
        accounting = Accounting("embeddings")
        sent = []
        statuses = iter([429, 503, 429, 200])

        def handler(request):
            sent.append(request.content)
            status = next(statuses)
            return success(request) if status == 200 else httpx.Response(status, json={"error": {"message": "temporary"}})

        client = await wrap(vertex.VertexEmbeddingClient(Tokens(accounting)), accounting, handler, "vertex")
        try:
            assert await embed(client, api) == [{"values": [0.5], "token_count": 2}]
            assert clock == [1, 2, 4]
            assert len(sent) == len(accounting.before) == len(accounting.records) == 4
            assert len(set(sent)) == 1
            assert len({r.reservation_id for r, _ in accounting.records}) == 4
            assert [u for _, u in accounting.records[:-1]] == [None] * 3
            assert accounting.records[-1][1].embedding_tokens == 2
            print(f"retry_success: api={api}, delays={clock}, sends=4, admissions=4, reservations=4")
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("attempts", [1, 4])
def test_retry_attempt_limit_preserves_last_error(api, clock, monkeypatch, attempts):
    monkeypatch.setenv("EMBEDDINGS_RETRY_MAX_ATTEMPTS", str(attempts))

    async def run():
        accounting = Accounting("embeddings")
        client = await wrap(vertex.VertexEmbeddingClient(Tokens(accounting)), accounting,
                            lambda _: httpx.Response(503, json={"error": {"message": "temporary"}}), "vertex")
        try:
            with pytest.raises(vertex.VertexAPIError) as error:
                await embed(client, api)
            assert error.value.status_code == 503
            assert len(accounting.before) == attempts
            assert clock == ([1, 2, 4] if attempts == 4 else [])
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("factor,expected", [
    (0.8, [0.8, 1.6, 3.2, 6.4, 6.4]),
    (1.2, [1.2, 2.4, 4.8, 8, 8]),
])
def test_jitter_and_delay_cap(api, clock, monkeypatch, factor, expected):
    monkeypatch.setenv("EMBEDDINGS_RETRY_MAX_ATTEMPTS", "6")
    sampled = []

    def jitter(low, high):
        sampled.append((low, high))
        return factor

    monkeypatch.setattr(resilience, "uniform", jitter)

    async def run():
        accounting = Accounting("embeddings")
        client = await wrap(vertex.VertexEmbeddingClient(Tokens(accounting)), accounting,
                            lambda _: httpx.Response(429), "vertex")
        try:
            with pytest.raises(vertex.VertexAPIError):
                await embed(client, api)
            assert clock == pytest.approx(expected)
            assert sampled == [(0.8, 1.2)] * 5
            assert len(accounting.before) == 6
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("header,expected", [
    ("3", 3), ("invalid", 1), ("nan", 1), ("inf", 1), ("-1", 1),
    (format_datetime(datetime.fromtimestamp(1700000005, UTC), usegmt=True), 5),
])
def test_retry_after_header(api, clock, monkeypatch, header, expected):
    monkeypatch.setattr(resilience, "time", lambda: 1700000000)

    async def run():
        accounting = Accounting("embeddings")

        def handler(request):
            if len(accounting.before) == 1:
                return httpx.Response(429, headers={"Retry-After": header})
            return success(request)

        client = await wrap(vertex.VertexEmbeddingClient(Tokens(accounting)), accounting, handler, "vertex")
        try:
            await embed(client, api)
            assert clock == [expected]
            assert len(accounting.before) == 2
        finally:
            await client.close()

    asyncio.run(run())


def test_retry_after_beyond_budget_does_not_retry_early(api, clock):
    async def run():
        accounting = Accounting("embeddings")
        client = await wrap(vertex.VertexEmbeddingClient(Tokens(accounting)), accounting,
                            lambda _: httpx.Response(503, headers={"Retry-After": "31"}), "vertex")
        try:
            with pytest.raises(vertex.VertexAPIError) as error:
                await embed(client, api)
            assert error.value.status_code == 503
            assert len(accounting.before) == 1
            assert clock == []
        finally:
            await client.close()

    asyncio.run(run())


def test_retry_budget_is_shared_across_all_attempts(api, clock):
    async def run():
        accounting = Accounting("embeddings")
        client = await wrap(vertex.VertexEmbeddingClient(Tokens(accounting)), accounting,
                            lambda _: httpx.Response(503), "vertex")
        client.retry_policy = replace(client.retry_policy, timeout=2.5)
        try:
            with pytest.raises(vertex.VertexAPIError):
                await embed(client, api)
            assert len(accounting.before) == 2 and clock == [1]
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("status,code", [
    (400, "invalid"), (401, "unauthorized"), (403, "forbidden"), (500, "internal"),
    (429, "budget_exceeded"), (503, "cost_config_error"), (503, "cost_tracking_unavailable"),
])
def test_permanent_and_cost_errors_never_retry(api, clock, status, code):
    async def run():
        accounting = Accounting("embeddings")
        client = await wrap(vertex.VertexEmbeddingClient(Tokens(accounting)), accounting,
                            lambda _: httpx.Response(status, json={"error": {"code": code}}), "vertex")
        try:
            with pytest.raises(vertex.VertexAPIError) as error:
                await embed(client, api)
            assert error.value.code == code and error.value.status_code == status
            assert len(accounting.before) == 1 and clock == []
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("error_type,status", [(httpx.ReadTimeout, 504), (httpx.ConnectError, 502)])
def test_uncertain_transport_failures_never_retry(api, clock, error_type, status):
    async def run():
        accounting = Accounting("embeddings")

        def handler(_request):
            raise error_type("synthetic failure")

        client = await wrap(vertex.VertexEmbeddingClient(Tokens(accounting)), accounting, handler, "vertex")
        try:
            with pytest.raises(vertex.VertexAPIError) as error:
                await embed(client, api)
            assert error.value.status_code == status
            assert len(accounting.before) == len(accounting.records) == 1 and clock == []
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("error_type", [CostConfigError, CostSubsystemUnhealthy])
def test_retry_admission_failure_blocks_transmission(api, clock, error_type):
    class FailingAccounting(Accounting):
        async def before_attempt(self, provider):
            reservation = await super().before_attempt(provider)
            if len(self.before) == 2:
                raise error_type("synthetic admission failure")
            return reservation

    async def run():
        accounting = FailingAccounting("embeddings")
        sent = []

        def handler(request):
            sent.append(request)
            return httpx.Response(503)

        client = await wrap(vertex.VertexEmbeddingClient(Tokens(accounting)), accounting, handler, "vertex")
        try:
            with pytest.raises(error_type):
                await embed(client, api)
            assert len(accounting.before) == 2
            assert len(sent) == len(accounting.records) == 1
            assert clock == [1]
        finally:
            await client.close()

    asyncio.run(run())


def test_cancellation_during_backoff_returns_http_slot(api, monkeypatch):
    async def run():
        entered = asyncio.Event()
        accounting = Accounting("embeddings")

        async def wait(_delay):
            entered.set()
            await asyncio.Future()

        monkeypatch.setattr(resilience, "sleep", wait)
        client = await wrap(vertex.VertexEmbeddingClient(Tokens(accounting)), accounting,
                            lambda request: httpx.Response(503) if len(accounting.before) == 1 else success(request), "vertex")
        task = asyncio.create_task(embed(client, api))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            assert client.semaphore._value == client.max_inflight
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert len(accounting.before) == 1
            assert await embed(client, api) == [{"values": [0.5], "token_count": 2}]
            assert client.semaphore._value == client.max_inflight
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await client.close()

    asyncio.run(run())


def test_retry_time_budget_interrupts_inflight_retry(api):
    async def run():
        accounting = Accounting("embeddings")

        async def handler(_request):
            if len(accounting.before) == 1:
                return httpx.Response(503)
            await asyncio.Future()

        client = await wrap(vertex.VertexEmbeddingClient(Tokens(accounting)), accounting, handler, "vertex")
        client.retry_policy = replace(client.retry_policy, initial_delay=0.01, timeout=0.1)
        try:
            with pytest.raises(vertex.VertexAPIError) as error:
                await asyncio.wait_for(embed(client, api), 1)
            assert error.value.status_code == 503
            assert len(accounting.before) == len(accounting.records) == 2
            assert all(usage is None for _, usage in accounting.records)
            assert client.semaphore._value == client.max_inflight
        finally:
            await client.close()

    asyncio.run(run())


def test_retry_time_budget_includes_http_slot_wait(api, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_MAX_INFLIGHT", "1")

    async def run():
        accounting = Accounting("embeddings")
        client = await wrap(vertex.VertexEmbeddingClient(Tokens(accounting)), accounting,
                            lambda _: httpx.Response(503), "vertex")
        client.retry_policy = replace(client.retry_policy, initial_delay=0.01, timeout=0.1)

        async def occupy_slot(_delay):
            await client.semaphore.acquire()

        monkeypatch.setattr(resilience, "sleep", occupy_slot)
        try:
            with pytest.raises(vertex.VertexAPIError) as error:
                await asyncio.wait_for(embed(client, api), 1)
            assert error.value.status_code == 503
            assert len(accounting.before) == len(accounting.records) == 1
            assert client.semaphore._value == 0
            client.semaphore.release()
            assert client.semaphore._value == 1
        finally:
            await client.close()

    asyncio.run(run())


def test_failed_batch_cancels_sibling_backoff_before_returning(api, monkeypatch):
    async def run():
        accounting = Accounting("embeddings")
        waiting, cancelled = asyncio.Event(), asyncio.Event()

        async def sleep(_delay):
            waiting.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        async def handler(request):
            body = json.loads(request.content)
            text = body["instances"][0]["content"] if api == "predict" else body["content"]["parts"][0]["text"]
            if text == "retry":
                return httpx.Response(429)
            if text == "bad":
                await waiting.wait()
                return httpx.Response(400)
            return success(request)

        monkeypatch.setattr(resilience, "sleep", sleep)
        client = await wrap(vertex.VertexEmbeddingClient(Tokens(accounting)), accounting, handler, "vertex")
        try:
            with pytest.raises(vertex.VertexAPIError) as error:
                await asyncio.wait_for(client.embed(
                    model="logical-model", texts=["retry", "bad"], dimensions=None,
                    task_type="RETRIEVAL_DOCUMENT", title=None,
                    resolved_config={"api": api, "max_instances": 1, "location": "global"},
                ), 1)
            assert error.value.status_code == 400
            assert cancelled.is_set()
            assert len(accounting.before) == len(accounting.records) == 2
            assert client.semaphore._value == client.max_inflight
            assert await embed(client, api) == [{"values": [0.5], "token_count": 2}]
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("capacity", [32, 3])
def test_real_asgi_provider_path_uses_shared_capacity(api, monkeypatch, capacity):
    if capacity != 32:
        monkeypatch.setenv("EMBEDDINGS_MAX_INFLIGHT", str(capacity))
    monkeypatch.setattr(vertex, "MAX_CONCURRENCY", 1)

    async def run():
        accounting = Accounting("embeddings")
        active = peak = 0

        async def handler(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.02)
                return success(request)
            finally:
                active -= 1

        provider = await wrap(vertex.VertexEmbeddingClient(Tokens(accounting)), accounting, handler, "vertex")
        assert EmbeddingRequestLimits(None).max_inflight == provider.max_inflight == capacity
        app = main.create_app(
            embedding_client_factory=lambda: provider, chat_client_factory=NoCallProvider,
            rerank_client_factory=NoCallProvider, ollama_chat_client_factory=NoCallProvider,
            foundry_chat_client_factory=NoCallProvider, openai_compatible_chat_client_factory=NoCallProvider,
            cost_accounting_factory=DisabledCostAccounting,
        )
        model = "text-embedding-005" if api == "predict" else "gemini-embedding-2"
        async with app.router.lifespan_context(app), httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            responses = await asyncio.gather(*(client.post("/v1/embeddings", json={"model": model, "input": "x"}) for _ in range(50)))
            assert [response.status_code for response in responses] == [200] * 50
            assert peak == capacity
            assert len(accounting.before) == len(accounting.records) == 50
            assert all(response.json()["data"][0]["embedding"] == [0.5] for response in responses)
            assert (await client.get("/readyz")).json()["embeddings"] == {
                "inflight": 0, "queue_depth": 0, "capacity": capacity,
            }
        print(f"provider_burst: api={api}, requests=50, statuses=50x200, ingress={capacity}, upstream_peak={peak}")

    asyncio.run(run())


def test_token_preparation_shares_embedding_capacity(api, monkeypatch):
    monkeypatch.setenv("EMBEDDINGS_MAX_INFLIGHT", "3")

    async def run():
        accounting = Accounting("embeddings")
        active = peak = 0

        class BoundedTokens(Tokens):
            async def get_token(self):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                try:
                    await asyncio.sleep(0.01)
                    return await super().get_token()
                finally:
                    active -= 1

        async def handler(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.01)
                return success(request)
            finally:
                active -= 1

        client = await wrap(vertex.VertexEmbeddingClient(BoundedTokens(accounting)), accounting, handler, "vertex")
        try:
            results = await client.embed(
                model="logical-model", texts=["x"] * 20, dimensions=None,
                task_type="RETRIEVAL_DOCUMENT", title=None,
                resolved_config={"api": api, "max_instances": 1, "location": "global"},
            )
            assert len(results) == len(accounting.before) == 20
            assert peak == 3
            assert client.semaphore._value == 3
        finally:
            await client.close()

    asyncio.run(run())


@pytest.mark.parametrize("name,value", [
    ("EMBEDDINGS_RETRY_MAX_ATTEMPTS", "0"), ("EMBEDDINGS_RETRY_MAX_ATTEMPTS", "9"),
    ("EMBEDDINGS_RETRY_MAX_ATTEMPTS", "1.5"), ("EMBEDDINGS_RETRY_MAX_ATTEMPTS", "invalid"),
    ("EMBEDDINGS_RETRY_INITIAL_DELAY_SECONDS", "0"), ("EMBEDDINGS_RETRY_INITIAL_DELAY_SECONDS", "nan"),
    ("EMBEDDINGS_RETRY_MAX_DELAY_SECONDS", "0.5"), ("EMBEDDINGS_RETRY_MAX_DELAY_SECONDS", "inf"),
    ("EMBEDDINGS_RETRY_TIMEOUT_SECONDS", "0"), ("EMBEDDINGS_RETRY_TIMEOUT_SECONDS", "301"),
])
def test_invalid_retry_configuration_fails_before_client_creation(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        vertex.VertexEmbeddingClient(Tokens(Accounting()))
