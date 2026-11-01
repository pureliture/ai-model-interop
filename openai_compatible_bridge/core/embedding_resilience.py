"""임베딩 동시성 설정과 응답이 확인된 일시 오류의 제한된 재시도."""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from random import uniform
from time import monotonic, time
from typing import Any

import httpx

from .env_limits import LIMITS

sleep = asyncio.sleep
_NON_RETRYABLE_CODES = {"budget_exceeded", "cost_config_error", "cost_tracking_unavailable"}


def embedding_max_inflight(source: Mapping[str, str] | None = None) -> int:
    env = os.environ if source is None else source
    try:
        value = int(env.get("EMBEDDINGS_MAX_INFLIGHT", "32"))
    except ValueError:
        raise ValueError("EMBEDDINGS_MAX_INFLIGHT must be an integer") from None
    if value < 1:
        raise ValueError("EMBEDDINGS_MAX_INFLIGHT must be positive")
    return value


def _seconds(env: Mapping[str, str], name: str, default: str, minimum: float, maximum: float) -> float:
    try:
        value = float(env.get(name, default))
    except ValueError:
        raise ValueError(f"{name} must be a number") from None
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be within {minimum}..{maximum}")
    return value


def _retryable(response: httpx.Response) -> bool:
    if response.status_code not in {429, 503}:
        return False
    try:
        payload = response.json()
    except ValueError:
        return True
    error = payload.get("error") if isinstance(payload, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    return not isinstance(code, str) or code not in _NON_RETRYABLE_CODES


def _retry_after(response: httpx.Response) -> float:
    raw = response.headers.get("Retry-After")
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        try:
            date = parsedate_to_datetime(raw)
            if date.tzinfo is None:
                return 0.0
            value = date.timestamp() - time()
        except (ValueError, TypeError, OverflowError):
            return 0.0
    return max(0.0, value) if math.isfinite(value) else 0.0


@dataclass(frozen=True)
class EmbeddingRetryPolicy:
    max_attempts: int = 4
    initial_delay: float = 1.0
    max_delay: float = 8.0
    timeout: float = 30.0

    @classmethod
    def from_env(cls, source: Mapping[str, str] | None = None) -> EmbeddingRetryPolicy:
        env = os.environ if source is None else source
        try:
            attempts = int(env.get("EMBEDDINGS_RETRY_MAX_ATTEMPTS", "4"))
        except ValueError:
            raise ValueError("EMBEDDINGS_RETRY_MAX_ATTEMPTS must be an integer") from None
        attempt_limits = LIMITS["EMBEDDINGS_RETRY_MAX_ATTEMPTS"]
        if not attempt_limits["min"] <= attempts <= attempt_limits["max"]:
            raise ValueError("EMBEDDINGS_RETRY_MAX_ATTEMPTS must be within 1..8")
        initial_spec = LIMITS["EMBEDDINGS_RETRY_INITIAL_DELAY_SECONDS"]
        max_spec = LIMITS["EMBEDDINGS_RETRY_MAX_DELAY_SECONDS"]
        timeout_spec = LIMITS["EMBEDDINGS_RETRY_TIMEOUT_SECONDS"]
        initial = _seconds(env, "EMBEDDINGS_RETRY_INITIAL_DELAY_SECONDS", "1", initial_spec["min"], initial_spec["max"])
        maximum = _seconds(env, "EMBEDDINGS_RETRY_MAX_DELAY_SECONDS", "8", max_spec["min"], max_spec["max"])
        if maximum < initial:
            raise ValueError("EMBEDDINGS_RETRY_MAX_DELAY_SECONDS must cover the initial delay")
        return cls(
            max_attempts=attempts, initial_delay=initial, max_delay=maximum,
            timeout=_seconds(env, "EMBEDDINGS_RETRY_TIMEOUT_SECONDS", "30", timeout_spec["min"], timeout_spec["max"]),
        )

    async def post(
        self, http: Any, semaphore: asyncio.Semaphore, url: str, **kwargs: Any,
    ) -> httpx.Response:
        # http는 MeteredHTTPClient를 포함한다. 매 전송마다 새 admission을 거친다.
        async with semaphore:
            response = await http.post(url, **kwargs)
        deadline = monotonic() + self.timeout
        for retry in range(self.max_attempts - 1):
            if not _retryable(response):
                break
            base = min(self.max_delay, self.initial_delay * 2**retry)
            delay = max(min(self.max_delay, base * uniform(0.8, 1.2)), _retry_after(response))
            remaining = deadline - monotonic()
            if delay >= remaining:
                break
            await response.aclose()
            # backoff 중에는 HTTP 슬롯을 반환한다. 최초 전송의 기존 HTTP timeout은 유지한다.
            budget = asyncio.timeout(remaining)
            try:
                async with budget:
                    await sleep(delay)
                    async with semaphore:
                        if monotonic() >= deadline:
                            break
                        response = await http.post(url, **kwargs)
            except TimeoutError:
                if not budget.expired():
                    raise
                # 후속 전송의 결과가 불확실해도 추가 전송하지 않고 마지막 오류를 유지한다.
                break
        return response
