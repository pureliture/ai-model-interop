from __future__ import annotations

import asyncio
import json
import os
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import replace
from functools import partial
from typing import Any, Self

from .env_limits import LIMITS
from .cost_tracking import (
    BudgetGate,
    BudgetReservationContext,
    CostBudgetExceeded,
    CostConfigError,
    CostReservation,
    CostSubsystemUnhealthy,
    CostTrackingConfig,
    DisabledCostAccounting,
    NormalizedUsage,
    PricingCatalog,
    SQLiteCostRepository,
    _iso,
    _parse_bool,
    _utcnow,
)

_ACTIVE: ContextVar[AttemptContext | None] = ContextVar("cost_attempt_context", default=None)


class AttemptContext(BudgetReservationContext):
    """요청 메타데이터만 전달한다. 실제 예약은 HTTP 호출 직전에 수행한다."""

    def preflight_now(self) -> AttemptContext:
        return self

    async def __aenter__(self) -> Self:
        self._token = _ACTIVE.set(self)
        return self

    async def __aexit__(self, *args: object) -> bool:
        _ACTIVE.reset(self._token)
        return False

    def complete_attempt(self, usage: NormalizedUsage, estimated_reason: str) -> None:
        self.actual_usage = usage

    def release_attempt(self, reason: str) -> None:
        pass

    def finalize_interrupted_stream(self, reason: str) -> None:
        pass

    def renew(self, *, model: str, forecast_usage: NormalizedUsage, provider: str | None = None) -> None:
        self.model = model
        self.forecast_usage = forecast_usage
        if provider is not None:
            self.provider = provider


class AttemptBudgetGate(BudgetGate):
    def _mark_ledger_failure(self, operation: str, reservation_id: str | None = None) -> None:
        # 실패는 호출자에게 전달하되, 기록 실패로 후속 판정을 영구 차단하지 않는다.
        raise CostSubsystemUnhealthy("cost ledger unavailable") from None


class AsyncCostAccounting:
    enabled = True

    def __init__(
        self, *, config: CostTrackingConfig | None, ledger: Any = None,
        pricing: PricingCatalog | None = None, billing: Mapping[str, str],
        queue_size: int = 256, admission_workers: int = 4,
        admission_timeout: float = 10.0, shutdown_timeout: float = 1.0,
    ) -> None:
        if min(queue_size, admission_workers, admission_timeout, shutdown_timeout) <= 0:
            raise ValueError("cost execution bounds must be positive")
        self.config = config
        self.billing = dict(billing)
        legacy = self.billing.get("openai_compat")
        modern = self.billing.get("openai_compatible")
        if legacy is not None and modern is not None and legacy != modern:
            raise CostConfigError("conflicting openai compatibility billing classifications")
        if modern is not None and legacy is None:
            self.billing["openai_compat"] = modern
        if legacy is not None and modern is None:
            self.billing["openai_compatible"] = legacy
        self.ledger = ledger
        self.gate = None
        if config is not None and ledger is not None and pricing is not None:
            metered = tuple(sorted(key for key, value in self.billing.items() if value == "metered"))
            self.gate = AttemptBudgetGate(
                config=replace(config, tracked_providers=metered), ledger=ledger, pricing=pricing,
            )
        self._admission_executor = ThreadPoolExecutor(max_workers=admission_workers, thread_name_prefix="cost-admission")
        self._record_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cost-record")
        self._admission_workers = admission_workers
        self._admission_slots = asyncio.BoundedSemaphore(admission_workers)
        self._admission_waiters = 0
        self._admission_timeout = admission_timeout
        self._shutdown_timeout = shutdown_timeout
        self._admissions: set[asyncio.Future] = set()
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self._writer: asyncio.Task | None = None
        self._closed = False
        self._sqlite_lock = threading.Lock()
        self._counts = dict.fromkeys((
            "admission_timeouts", "admission_rejected", "admission_failures",
            "records_written", "records_dropped", "record_failures", "usage_missing",
            "record_in_flight",
        ), 0)
        self._last_record_success_at: str | None = None

    def reservation(self, **kwargs: Any) -> AttemptContext:
        return AttemptContext(self, **kwargs)

    def _run(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(self.ledger, SQLiteCostRepository):
            # SQLite connection은 생성된 worker에서만 사용하고 닫는다.
            with self._sqlite_lock:
                try:
                    self.ledger.initialize()
                    return operation(*args, **kwargs)
                finally:
                    self.ledger.close()
        return operation(*args, **kwargs)

    async def _admit_io(self, operation: Any, *args: Any, **kwargs: Any) -> Any:
        if self._closed:
            self._counts["admission_rejected"] += 1
            raise CostSubsystemUnhealthy("cost admission capacity unavailable")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._admission_timeout
        if self._admission_slots.locked():
            self._admission_waiters += 1
            try:
                await asyncio.wait_for(
                    self._admission_slots.acquire(), timeout=max(0.0, deadline - loop.time()),
                )
            except TimeoutError:
                self._counts["admission_timeouts"] += 1
                self._counts["admission_rejected"] += 1
                raise CostSubsystemUnhealthy("cost admission timed out") from None
            finally:
                self._admission_waiters -= 1
        else:
            await self._admission_slots.acquire()

        if self._closed:
            self._admission_slots.release()
            self._counts["admission_rejected"] += 1
            raise CostSubsystemUnhealthy("cost admission capacity unavailable")
        remaining = deadline - loop.time()
        if remaining <= 0:
            self._admission_slots.release()
            self._counts["admission_timeouts"] += 1
            self._counts["admission_rejected"] += 1
            raise CostSubsystemUnhealthy("cost admission timed out") from None

        try:
            future = loop.run_in_executor(
                self._admission_executor, partial(self._run, operation, *args, **kwargs),
            )
        except Exception:  # noqa: BLE001 - 제출 실패도 슬롯을 반환하고 유료 전송을 차단한다.
            self._admission_slots.release()
            self._counts["admission_rejected"] += 1
            raise CostSubsystemUnhealthy("cost admission capacity unavailable") from None
        self._admissions.add(future)

        def finished(done: asyncio.Future) -> None:
            self._admissions.discard(done)
            self._admission_slots.release()
            if not done.cancelled():
                done.exception()

        future.add_done_callback(finished)
        try:
            return await asyncio.wait_for(asyncio.shield(future), remaining)
        except TimeoutError:
            self._counts["admission_timeouts"] += 1
            raise CostSubsystemUnhealthy("cost admission timed out") from None

    def _preflight(self, **kwargs: Any) -> CostReservation:
        self.ledger.check_health()
        return self.gate.preflight(**kwargs)

    async def before_attempt(self, provider: str) -> CostReservation | None:
        mode = self.billing.get(provider)
        if mode in {"subscription", "nonbillable"}:
            return None
        if mode != "metered" or self.gate is None:
            missing = []
            if mode != "metered":
                missing.append("COST_PROVIDER_BILLING_JSON")
            if self.gate is None:
                missing.append("COST_PRICING_JSON")
            raise CostConfigError(
                "cost provider billing or pricing configuration unavailable: "
                f"provider={provider!r} has classification={mode!r}; "
                f"missing={'/'.join(missing)}"
            )
        ctx = _ACTIVE.get()
        if ctx is None or ctx.accounting is not self or ctx.provider != provider:
            raise CostConfigError("cost request context unavailable")
        try:
            return await self._admit_io(
                self._preflight, endpoint=ctx.endpoint, model=ctx.model,
                forecast_usage=ctx.forecast_usage, provider=provider,
            )
        except (CostConfigError, CostSubsystemUnhealthy):
            self._counts["admission_failures"] += 1
            raise
        except CostBudgetExceeded:
            raise
        except Exception:  # noqa: BLE001 - 예상 못한 저장소 오류도 유료 전송 없이 안전하게 차단한다.
            self._counts["admission_failures"] += 1
            raise CostSubsystemUnhealthy("cost admission unavailable") from None

    def record_attempt(self, reservation: CostReservation | None, usage: NormalizedUsage | None) -> None:
        if reservation is None:
            return
        if usage is None:
            self._counts["usage_missing"] += 1
            return
        if self._closed:
            self._counts["records_dropped"] += 1
            return
        try:
            self._queue.put_nowait((reservation, usage))
        except asyncio.QueueFull:
            self._counts["records_dropped"] += 1
            return
        if self._writer is None:
            self._writer = asyncio.create_task(self._write_records())

    async def _write_records(self) -> None:
        while True:
            reservation, usage = await self._queue.get()
            self._counts["record_in_flight"] = 1
            try:
                await asyncio.get_running_loop().run_in_executor(
                    self._record_executor,
                    partial(self._run, self.gate.finalize_success, reservation, usage),
                )
                self._counts["records_written"] += 1
                self._last_record_success_at = _iso(_utcnow())
            except asyncio.CancelledError:
                self._counts["records_dropped"] += 1
                raise
            except Exception:  # noqa: BLE001 - 최외곽 기록 경계는 실패를 계수하고 다음 job을 계속 처리한다.
                self._counts["record_failures"] += 1
                self._counts["records_dropped"] += 1
            finally:
                self._counts["record_in_flight"] = 0
                self._queue.task_done()

    def metrics(self) -> dict[str, Any]:
        return {
            **self._counts,
            "queue_depth": self._queue.qsize(), "queue_capacity": self._queue.maxsize,
            "admission_in_flight": len(self._admissions),
            "admission_capacity": self._admission_workers,
            "admission_queue_depth": self._admission_waiters,
            "last_record_success_at": self._last_record_success_at,
        }

    async def readiness(self) -> dict[str, Any]:
        available = False
        if self.gate is not None:
            try:
                await self._admit_io(self.ledger.check_health)
                available = True
            except (CostConfigError, CostSubsystemUnhealthy, OSError):
                available = False
        return {
            "enabled": True, "backend": self.config.backend if self.config else "unconfigured",
            "database_available": available, "healthy": available and bool(self.billing),
            "reason": None if available else "cost admission unavailable",
            "billing": self.billing, "recording": self.metrics(),
        }

    async def admin_status(self, *, provider: str | None = None) -> dict[str, Any]:
        if self.gate is None:
            return await self.readiness()
        result = await self._admit_io(self.gate.admin_status, provider=provider)
        result.update(billing=self.billing, recording=self.metrics(), guarantee="forecast_admission_only")
        return result

    async def admin_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        if self.gate is None:
            return []
        return await self._admit_io(self.gate.admin_events, limit=limit)

    async def admin_reconciliation(self) -> dict[str, Any]:
        if self.gate is None:
            return {"status": "unavailable"}
        return await self._admit_io(self.gate.admin_reconciliation)

    async def flush(self) -> None:
        await self._queue.join()

    async def aclose(self) -> None:
        self._closed = True
        try:
            await asyncio.wait_for(self.flush(), self._shutdown_timeout)
        except TimeoutError:
            pass
        if self._writer is not None:
            self._writer.cancel()
            try:
                await self._writer
            except asyncio.CancelledError:
                pass
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()
            self._counts["records_dropped"] += 1
        self._admission_executor.shutdown(wait=False, cancel_futures=True)
        self._record_executor.shutdown(wait=False, cancel_futures=True)
        # pool 기반 ledger가 물리 연결을 유지하므로 shutdown에서 닫는다.
        if self.ledger is not None:
            try:
                self.ledger.close()
            except Exception:  # noqa: BLE001 - 종료 경로는 남은 정리를 막지 않는다
                pass


def _billing_config(source: Mapping[str, str]) -> dict[str, str]:
    try:
        values = json.loads(source.get("COST_PROVIDER_BILLING_JSON", "{}"))
        if not isinstance(values, dict) or any(
            not isinstance(key, str) or not key or value not in {"metered", "subscription", "nonbillable"}
            for key, value in values.items()
        ):
            return {}
        return values
    except (ValueError, TypeError):
        return {}


def _strict_pricing(config: CostTrackingConfig) -> PricingCatalog:
    """Legacy shim: 검증은 PricingCatalog.from_config -> adapter 스펙으로 위임.

    async_cost는 예약/원장 배선만 담당하고 가격 차원 지식은 소유하지 않는다.
    """
    return PricingCatalog.from_config(config)


def _admission_timeout_seconds(source: Mapping[str, str]) -> float:
    """Operator-tunable cost-admission budget; keep the fail-closed default."""
    raw = str(source.get("COST_ADMISSION_TIMEOUT_SECONDS") or "").strip()
    if not raw:
        return 10.0
    try:
        value = float(raw)
    except ValueError:
        raise CostConfigError("COST_ADMISSION_TIMEOUT_SECONDS must be a number") from None
    spec = LIMITS["COST_ADMISSION_TIMEOUT_SECONDS"]
    if not spec["min"] <= value <= spec["max"]:
        raise CostConfigError("COST_ADMISSION_TIMEOUT_SECONDS must be within 0.5..30")
    return value


def _admission_worker_count(source: Mapping[str, str]) -> int:
    """Bound the number of concurrent PostgreSQL cost-admission operations."""
    raw = str(source.get("COST_ADMISSION_WORKERS") or "").strip()
    if not raw:
        return 4
    try:
        value = int(raw)
    except ValueError:
        raise CostConfigError("COST_ADMISSION_WORKERS must be an integer") from None
    spec = LIMITS["COST_ADMISSION_WORKERS"]
    if not spec["min"] <= value <= spec["max"]:
        raise CostConfigError("COST_ADMISSION_WORKERS must be within 1..64")
    return value


def build_async_cost_accounting(env: Mapping[str, str] | None = None) -> AsyncCostAccounting | DisabledCostAccounting:
    source = os.environ if env is None else env
    if not _parse_bool(source.get("COST_TRACKING_ENABLED")):
        return DisabledCostAccounting()
    billing = _billing_config(source)
    config = None
    try:
        config = CostTrackingConfig.from_env(source)
        pricing = _strict_pricing(config)
        admission_workers = _admission_worker_count(source)
        admission_timeout = _admission_timeout_seconds(source)
        if config.backend == "postgres":
            from .postgres_cost_repository import PostgresCostRepository

            ledger = PostgresCostRepository(
                config.postgres_dsn, request_retention_days=config.request_retention_days,
                aggregate_retention_months=config.aggregate_retention_months,
            )
        else:
            ledger = SQLiteCostRepository(
                config.ledger_path, request_retention_days=config.request_retention_days,
                aggregate_retention_months=config.aggregate_retention_months,
            )
        return AsyncCostAccounting(
            config=config, ledger=ledger, pricing=pricing, billing=billing,
            admission_workers=admission_workers, admission_timeout=admission_timeout,
        )
    except (CostConfigError, OSError, ValueError, TypeError, AttributeError):
        return AsyncCostAccounting(config=config, billing=billing)
