"""실제 HTTP 시도 경계에서 예약하고 원본 사용량만 비차단 기록에 전달한다."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from typing import Any, Protocol

import httpx

from openai_compatible_bridge.core.cost_tracking import CostReservation, NormalizedUsage
from openai_compatible_bridge.core.provider_adapter import (
    _chat_usage,
    _count,
    _embedding_count,
    _is_error,
    _object,
    parse_usage_for_provider,
)


class _AttemptAccounting(Protocol):
    async def before_attempt(self, provider: str) -> CostReservation | None: ...

    def record_attempt(self, reservation: CostReservation, usage: NormalizedUsage | None) -> None: ...


# NOTE: _count/_object/_chat_usage/_embedding_count/_is_error 구현은
# provider_adapter가 단일 소유한다. 아래 _raw_usage는 legacy shim으로
# 동일한 본문을 유지하되 헬퍼는 adapter에서 import해 쓴다.


def _raw_usage(payload: Any, endpoint: str) -> NormalizedUsage | None:
    """누락과 명시적 0을 구별하며 해당 endpoint의 모든 과금 차원을 요구한다."""
    payload = _object(payload)
    if _is_error(payload):
        return None
    if endpoint == "embeddings":
        tokens = _embedding_count(payload)
        return None if tokens is None else NormalizedUsage(embedding_tokens=tokens, total_tokens=tokens)
    if endpoint == "rerank":
        units = _count(_object(payload.get("usage")).get("rerank_units"))
        return None if units is None else NormalizedUsage(rerank_units=units)
    if endpoint != "chat":
        return None
    if payload.get("type") == "response.completed":
        payload = _object(payload.get("response"))
        if _is_error(payload):
            return None
    if "usageMetadata" in payload:
        return _chat_usage(
            _object(payload["usageMetadata"]), "promptTokenCount", "candidatesTokenCount",
            "totalTokenCount", thoughts_key="thoughtsTokenCount",
        )
    if "prompt_eval_count" in payload or "eval_count" in payload:
        return _chat_usage(payload, "prompt_eval_count", "eval_count")
    usage = _object(payload.get("usage"))
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        return _chat_usage(usage, "prompt_tokens", "completion_tokens")
    return _chat_usage(usage, "input_tokens", "output_tokens")


def _observe_usage(provider: str | None, payload: Any, endpoint: str) -> NormalizedUsage | None:
    """Step 1 dispatcher: provider 소유 파서 우선, unknown이면 legacy 폴백.

    known provider가 None을 반환해도 legacy로 재시도해 기존 동작을 보존한다.
    완전 제거는 legacy _raw_usage/_StreamUsage 호출자가 없을 때 별도 작업으로.
    """
    if (provider == "jetbrains" and endpoint == "chat" and isinstance(payload, dict)
            and payload.get("status") == "incomplete" and payload.get("error") is None):
        usage = _object(payload.get("usage"))
        return _chat_usage(usage, "input_tokens", "output_tokens")
    if provider is not None:
        parsed = parse_usage_for_provider(provider, payload, endpoint)
        if parsed is not None:
            return parsed
        if provider in ("vertex", "ollama", "foundry", "openai_compat", "openai_compatible", "jetbrains"):
            return _raw_usage(payload, endpoint)
    return _raw_usage(payload, endpoint)


class _StreamUsage:
    """최대 64KiB의 SSE 이벤트만 임시 보관하고 누적 사용량은 마지막 값으로 교체한다."""

    _MAX_BYTES = 64 * 1024

    def __init__(self, endpoint: str, provider: str | None = None) -> None:
        self.endpoint = endpoint
        self.provider = provider
        self._usage: NormalizedUsage | None = None
        self._input_tokens: int | None = None
        self._data: list[str] = []
        self.buffered_bytes = 0
        self.invalid = False
        self.done = False

    @property
    def usage(self) -> NormalizedUsage | None:
        return None if self.invalid else self._usage

    def _discard(self) -> None:
        self.invalid = True
        self._data.clear()
        self.buffered_bytes = 0

    def feed_line(self, line: str) -> None:
        if self.invalid:
            return
        try:
            self._feed_line(line)
        except (ValueError, TypeError, RecursionError):
            # 관찰 실패는 원 응답/예외 대신 보수적 예약을 남긴다.
            self._discard()

    def _feed_line(self, line: str) -> None:
        if len(line) > self._MAX_BYTES or len(line.encode("utf-8")) > self._MAX_BYTES:
            self._discard()
            return
        if not line:
            self.finish()
        elif line.startswith("data:"):
            data = line[5:]
            data = data.removeprefix(" ")
            if data.strip() == "[DONE]":
                self.finish()
                self.done = True
                return
            size = len(data.encode("utf-8")) + 1
            if self.buffered_bytes + size > self._MAX_BYTES:
                self._discard()
                return
            self._data.append(data)
            self.buffered_bytes += size
        elif line.startswith((":", "event:", "id:", "retry:")):
            return
        elif line.lstrip().startswith(("{", "[")):
            self.finish()
            self._observe(json.loads(line))
        else:
            self._discard()

    def finish(self) -> None:
        if not self._data:
            return
        data = "\n".join(self._data)
        self._data.clear()
        self.buffered_bytes = 0
        try:
            self._observe(json.loads(data))
        except (ValueError, TypeError, RecursionError):
            self._discard()

    def _observe(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            self._discard()
            return
        event_type = payload.get("type")
        if event_type == "response.incomplete":
            response = _object(payload.get("response"))
            if payload.get("error") is not None or response.get("error") is not None or response.get("status") != "incomplete":
                self._discard()
                return
            self._usage = _observe_usage(self.provider, {"usage": response.get("usage")}, self.endpoint)
            if self._usage is None:
                self._discard()
            return
        if _is_error(payload):
            self._discard()
            return
        if event_type == "message_start":
            usage = _object(_object(payload.get("message")).get("usage"))
            self._input_tokens = _count(usage.get("input_tokens"))
            self._usage = None
        elif event_type == "message_delta":
            usage = _object(payload.get("usage"))
            merged = {"input_tokens": self._input_tokens, **usage}
            self._usage = _observe_usage(self.provider, {"usage": merged}, self.endpoint)
        elif event_type == "response.completed":
            self._usage = _observe_usage(self.provider, payload, self.endpoint)
        elif isinstance(event_type, str) and event_type.startswith("response."):
            return
        elif "done" in payload:
            if payload["done"] is True:
                self._usage = _observe_usage(self.provider, payload, self.endpoint)
        elif any(key in payload for key in ("usage", "usageMetadata", "prompt_eval_count", "eval_count")):
            self._usage = _observe_usage(self.provider, payload, self.endpoint)


class _MeteredResponse:
    def __init__(self, response: httpx.Response, endpoint: str, provider: str | None = None) -> None:
        self._response = response
        self._observer = _StreamUsage(endpoint, provider)
        self._exhausted = False
        self._failed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._response, name)

    async def aiter_lines(self) -> AsyncIterator[str]:
        try:
            async for line in self._response.aiter_lines():
                self._observer.feed_line(line)
                yield line
            self._observer.finish()
            self._exhausted = True
        except GeneratorExit:
            # Foundry는 [DONE]을 받으면 EOF를 읽지 않고 반복을 마친다.
            if not self._observer.done:
                self._failed = True
            raise
        except BaseException:
            self._failed = True
            raise

    def collected_usage(self) -> NormalizedUsage | None:
        if self._failed or self._response.status_code >= 400:
            return None
        if not (self._exhausted or self._observer.done):
            return None
        return self._observer.usage

    def discard_observation(self) -> None:
        self._observer._discard()


class MeteredHTTPClient:
    """기존 client를 한 번 감싼다. transport 생성이나 회계 목적 재시도는 하지 않는다.

    accounting은 요청 ContextVar로 forecast를 결정한다. record_attempt는 동기식,
    무 I/O 큐 전달이어야 하며 실제 DB 기록/실패 처리는 accounting이 소유한다.
    """

    def __init__(self, http: httpx.AsyncClient, accounting: _AttemptAccounting, *, provider: str) -> None:
        self._http = http
        self._accounting = accounting
        self._provider = provider

    def __getattr__(self, name: str) -> Any:
        return getattr(self._http, name)

    def _record(self, reservation: CostReservation | None, usage: NormalizedUsage | None) -> None:
        if reservation is not None:
            # 최외곽 nonraising 계약: 기록 서비스 오류나 동기 로그로 응답을 중단하지 않는다.
            with suppress(Exception):
                self._accounting.record_attempt(reservation, usage)

    async def post(self, *args: Any, **kwargs: Any) -> httpx.Response:
        reservation = await self._accounting.before_attempt(self._provider)
        usage = None
        try:
            response = await self._http.post(*args, **kwargs)
            if reservation is not None and response.status_code < 400:
                with suppress(ValueError, TypeError, RecursionError):
                    usage = _observe_usage(self._provider, response.json(), reservation.endpoint)
            return response
        finally:
            self._record(reservation, usage)

    @asynccontextmanager
    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        task = asyncio.current_task()
        cancellations = task.cancelling() if task is not None else 0
        proxy = None
        reservation = await self._accounting.before_attempt(self._provider)
        usage = None
        try:
            async with self._http.stream(*args, **kwargs) as response:
                if reservation is None:
                    yield response
                else:
                    proxy = _MeteredResponse(response, reservation.endpoint, self._provider)
                    yield proxy
            # Provider의 수동 __aexit__(None, None, None)도 취소를 성공으로 바꾸지 않는다.
            if proxy is not None and (task is None or task.cancelling() == cancellations):
                usage = proxy.collected_usage()
        finally:
            if proxy is not None:
                proxy.discard_observation()
            self._record(reservation, usage)

    async def aclose(self) -> None:
        await self._http.aclose()