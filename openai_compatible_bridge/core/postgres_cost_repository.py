from __future__ import annotations

import contextlib
import hashlib
import os
import threading
import time
import uuid
import weakref
from collections.abc import Iterator, Mapping
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Any

import psycopg
from psycopg.pq import TransactionStatus
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool, PoolTimeout

from openai_compatible_bridge.core.cost_tracking import (
    LEDGER_ALLOWED_FIELDS,
    CostLedgerValidationError,
    CostSubsystemUnhealthy,
    CostUsageSummary,
    ICostRepository,
    NormalizedUsage,
    ReconciliationResult,
    _iso,
    _utcnow,
)

SCHEMA = "bridge_cost"
CURRENT_SCHEMA_VERSION = 1
ADVISORY_LOCK_KEY = 0x4252494447454354

_MIGRATIONS = ((1, "0001_initial.sql"),)
_MONEY_FIELDS = {"forecast_cost_usd", "estimated_cost_usd"}
_TIME_FIELDS = {"created_at", "window_started_at", "finalized_at"}
_USAGE_FIELDS = {"prompt_tokens", "completion_tokens", "total_tokens", "embedding_tokens", "rerank_units"}
_TABLE_COLUMNS = {
    "cost_events": LEDGER_ALLOWED_FIELDS,
    "cost_daily_aggregates": ("day", "estimated_cost_usd", "currency", "updated_at"),
    "cost_reconciliation_results": (
        "day", "wrapper_estimated_cost_usd", "billing_export_cost", "delta_usd",
        "status", "checked_at", "error_message",
    ),
}


@contextlib.contextmanager
def _database_errors() -> Iterator[None]:
    try:
        yield
    except psycopg.Error:
        raise CostSubsystemUnhealthy("PostgreSQL 비용 저장소 작업에 실패했습니다.") from None


def _connect(dsn: str) -> psycopg.Connection:
    return psycopg.connect(
        dsn,
        autocommit=True,
        row_factory=dict_row,
        connect_timeout=3,
        options=(
            "-c statement_timeout=5000 -c lock_timeout=5000 "
            "-c idle_in_transaction_session_timeout=10000 -c timezone=UTC -c search_path=pg_catalog"
        ),
    )


def _pool_max_size() -> int:
    """admission 4 + record 1 + admin/readiness 여유분. env로 조정한다."""
    raw = os.getenv("COST_LEDGER_POOL_MAX_SIZE", "8").strip()
    try:
        value = int(raw)
    except ValueError:
        return 8
    return value if 1 <= value <= 64 else 8


def _pool_timeout_seconds() -> float:
    """pool 대기는 admission/caller timeout보다 짧게 유지한다."""
    raw = os.getenv("COST_LEDGER_POOL_TIMEOUT_SECONDS", "3").strip()
    try:
        value = float(raw)
    except ValueError:
        return 3.0
    return value if 0.1 <= value <= 30 else 3.0


def _schema_verify_ttl_seconds() -> float:
    """schema 검증 캐시 TTL. 0이면 매 transaction마다 검증한다(fail-closed 최대)."""
    raw = os.getenv("COST_SCHEMA_VERIFY_TTL_SECONDS", "30").strip()
    try:
        value = float(raw)
    except ValueError:
        return 30.0
    return value if 0.0 <= value <= 3600 else 30.0


@lru_cache(maxsize=1)
def _migrations() -> tuple[tuple[int, str, str], ...]:
    try:
        result = []
        for version, filename in _MIGRATIONS:
            content = (Path(__file__).with_name("migrations") / filename).read_bytes()
            result.append((version, content.decode("utf-8"), hashlib.sha256(content).hexdigest()))
        return tuple(result)
    except (OSError, UnicodeError):
        raise CostSubsystemUnhealthy("PostgreSQL migration 파일을 읽을 수 없습니다.") from None


def _migration_history(conn: psycopg.Connection) -> list[tuple[int, str]]:
    rows = conn.execute(f"SELECT version, checksum FROM {SCHEMA}.schema_migrations ORDER BY version").fetchall()
    return [(row["version"], row["checksum"]) for row in rows]


def _verify_schema(conn: psycopg.Connection) -> None:
    expected = [(version, checksum) for version, _, checksum in _migrations()]
    if _migration_history(conn) != expected:
        raise CostSubsystemUnhealthy("PostgreSQL 비용 schema version/checksum이 일치하지 않습니다.")
    tables = {**_TABLE_COLUMNS, "schema_migrations": ("version", "checksum", "applied_at")}
    rows = conn.execute(
        """
        SELECT c.relname, c.relkind, c.relrowsecurity, a.attname,
               format_type(a.atttypid, a.atttypmod) AS column_type
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_attribute a ON a.attrelid = c.oid
        WHERE n.nspname = %s AND c.relname = ANY(%s)
          AND a.attnum > 0 AND NOT a.attisdropped
        """,
        (SCHEMA, list(tables)),
    ).fetchall()
    expected_types = {}
    for table, columns in tables.items():
        for name in columns:
            column_type = "text"
            if name in _MONEY_FIELDS | {"wrapper_estimated_cost_usd", "billing_export_cost", "delta_usd"}:
                column_type = "numeric"
            elif name in _TIME_FIELDS | {"applied_at", "checked_at", "updated_at"}:
                column_type = "timestamp with time zone"
            elif name in _USAGE_FIELDS:
                column_type = "bigint"
            elif name in {"version", "billing_eligible"}:
                column_type = "integer"
            expected_types[(table, name)] = column_type
    actual_types = {(row["relname"], row["attname"]): row["column_type"] for row in rows}
    if actual_types != expected_types or any(row["relkind"] != "r" or row["relrowsecurity"] for row in rows):
        raise CostSubsystemUnhealthy("PostgreSQL 비용 schema 구조가 일치하지 않습니다.")
    for table, columns in _TABLE_COLUMNS.items():
        conn.execute(f"SELECT {', '.join(columns)} FROM {SCHEMA}.{table} LIMIT 0")


def _ensure_transaction(conn: psycopg.Connection) -> None:
    if conn.info.transaction_status != TransactionStatus.INTRANS:
        raise CostSubsystemUnhealthy("PostgreSQL 비용 transaction이 중단되었습니다.")


def apply_migrations(dsn: str) -> int:
    """운영자가 명시적으로 실행한다. DB/role 생성과 runtime 시작에서는 호출하지 않는다."""
    with _database_errors(), _connect(dsn) as conn, conn.transaction():
        conn.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (ADVISORY_LOCK_KEY,))
        rights = conn.execute(
            "SELECT has_database_privilege(current_database(), 'CREATE') AS can_create, "
            "current_setting('transaction_read_only') AS read_only, to_regnamespace(%s) AS schema_oid",
            (SCHEMA,),
        ).fetchone()
        if not rights["can_create"] or rights["read_only"] == "on":
            raise CostSubsystemUnhealthy("PostgreSQL migration 실행 권한이 없습니다.")
        applied = []
        if rights["schema_oid"] is not None:
            if not conn.execute("SELECT has_schema_privilege(%s, 'CREATE') AS allowed", (SCHEMA,)).fetchone()["allowed"]:
                raise CostSubsystemUnhealthy("PostgreSQL migration 실행 권한이 없습니다.")
            applied = _migration_history(conn)
            expected = [(version, checksum) for version, _, checksum in _migrations()]
            if not applied or applied != expected[:len(applied)]:
                raise CostSubsystemUnhealthy("PostgreSQL migration 이력이 일치하지 않습니다.")
        for version, content, checksum in _migrations()[len(applied):]:
            conn.execute(content)
            conn.execute(
                f"INSERT INTO {SCHEMA}.schema_migrations (version, checksum) VALUES (%s, %s)",
                (version, checksum),
            )
        _verify_schema(conn)
    return CURRENT_SCHEMA_VERSION


def _money(value: Any, *, signed: bool = False) -> Decimal:
    if isinstance(value, (float, bool)) or not isinstance(value, (str, int, Decimal)):
        raise CostLedgerValidationError("비용은 float이 아닌 유한 Decimal 값이어야 합니다.")
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError):
        raise CostLedgerValidationError("비용 Decimal 값이 유효하지 않습니다.") from None
    if not result.is_finite() or (not signed and result < 0):
        raise CostLedgerValidationError("비용은 유한한 음이 아닌 값이어야 합니다.")
    return result


def _timestamp(value: Any) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except (ValueError, TypeError, OverflowError):
        raise CostLedgerValidationError("비용 timestamp가 유효하지 않습니다.") from None


def _day(value: str | date) -> date:
    try:
        if isinstance(value, datetime):
            return _timestamp(value).date()
        return value if isinstance(value, date) else date.fromisoformat(value)
    except (ValueError, TypeError):
        raise CostLedgerValidationError("비용 날짜가 유효하지 않습니다.") from None


def _normalize(name: str, value: Any) -> Any:
    if value is None:
        return None
    if name in _MONEY_FIELDS:
        return str(_money(value))
    if name in _TIME_FIELDS:
        return _iso(_timestamp(value))
    if name == "billing_eligible":
        if not isinstance(value, int) or value not in (0, 1):
            raise CostLedgerValidationError("billing_eligible은 0 또는 1이어야 합니다.")
        return int(value)
    if name in _USAGE_FIELDS:
        if not isinstance(value, int) or value < 0 or value > 2**63 - 1:
            raise CostLedgerValidationError("사용량은 음이 아닌 BIGINT 값이어야 합니다.")
        return int(value)
    return str(value)


def _api_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: _iso(value) if isinstance(value, datetime) else str(value) if isinstance(value, Decimal) else value
        for name, value in row.items()
    }


def _db_value(name: str, value: Any) -> Any:
    if value is None:
        return None
    if name in _MONEY_FIELDS:
        return _money(value)
    if name in _TIME_FIELDS:
        return _timestamp(value)
    return value


def _same_fields(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(_db_value(name, value) == _db_value(name, right[name]) for name, value in left.items())


def _filters(statuses: tuple[str, ...] | None, providers: tuple[str, ...] | None) -> tuple[str, list[Any]]:
    clause = ""
    params: list[Any] = []
    if statuses is not None:
        clause += " AND status = ANY(%s)"
        params.append(list(statuses))
    if providers:
        clause += " AND provider = ANY(%s)"
        params.append(list(providers))
    return clause, params


_TRACKED_REPOSITORIES: weakref.WeakSet[PostgresCostRepository] = weakref.WeakSet()


def close_tracked_repositories() -> int:
    """테스트/운영 정리용: 남아 있는 repository의 pool을 닫는다.

    pool은 물리 연결을 유지하므로, 소유자가 close()를 잊으면 DROP DATABASE 같은
    정리 작업이 ObjectInUse로 실패한다. fixture가 teardown에서 호출한다.
    """
    closed = 0
    for repo in list(_TRACKED_REPOSITORIES):
        with contextlib.suppress(Exception):
            repo.close()
            closed += 1
    return closed


class PostgresCostRepository(ICostRepository):
    def __init__(
        self,
        dsn: str,
        *,
        request_retention_days: int = 90,
        aggregate_retention_months: int = 13,
        now_fn: Any = _utcnow,
    ) -> None:
        if request_retention_days <= 0 or aggregate_retention_months <= 0:
            raise CostLedgerValidationError("비용 보존 기간은 양수여야 합니다.")
        self._dsn = dsn
        self.request_retention_days = request_retention_days
        self.aggregate_retention_months = aggregate_retention_months
        self._now_fn = now_fn
        self._local = threading.local()
        self._pool: ConnectionPool | None = None
        self._pool_dsn: str | None = None
        self._retired_pools: list[ConnectionPool] = []
        self._pool_lock = threading.Lock()
        self._verify_at: float | None = None
        self._verify_dsn: str | None = None
        _TRACKED_REPOSITORIES.add(self)

    def _get_pool(self) -> ConnectionPool:
        """물리 연결은 pool이 소유한다. admission마다 TCP 연결을 새로 열지 않는다."""
        pool = self._pool
        if pool is not None and self._pool_dsn == self._dsn:
            return pool
        with self._pool_lock:
            if self._pool is not None and self._pool_dsn == self._dsn:
                return self._pool
            stale = self._pool
            self._pool = None
            self._pool_dsn = None
            if stale is not None:
                self._retired_pools.append(stale)
            pool = ConnectionPool(
                conninfo=self._dsn,
                min_size=0,
                max_size=_pool_max_size(),
                timeout=_pool_timeout_seconds(),
                open=False,
                kwargs={
                    "autocommit": True,
                    "row_factory": dict_row,
                    "connect_timeout": 3,
                    "options": (
                        "-c statement_timeout=5000 -c lock_timeout=5000 "
                        "-c idle_in_transaction_session_timeout=10000 "
                        "-c timezone=UTC -c search_path=pg_catalog"
                    ),
                },
                name="bridge-cost-ledger",
            )
            pool.open(wait=False)
            self._pool = pool
            self._pool_dsn = self._dsn
            return pool

    @property
    def connection(self) -> psycopg.Connection:
        conn = getattr(self._local, "connection", None)
        if conn is None:
            raise CostSubsystemUnhealthy("명시적 비용 transaction 안에서만 connection을 사용할 수 있습니다.")
        return conn

    @contextlib.contextmanager
    def _connection(self) -> Iterator[psycopg.Connection]:
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            with _database_errors():
                yield conn
            return
        pool = self._get_pool()
        try:
            with pool.connection(timeout=_pool_timeout_seconds()) as conn:
                with _database_errors(), conn.transaction():
                    conn.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                    self._local.connection = conn
                    try:
                        yield conn
                        _ensure_transaction(conn)
                    except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn):
                        # schema 변경/복구 흔적이면 캐시를 버리고 다음 호출에서 재검증한다.
                        self._verify_at = None
                        raise
                    finally:
                        del self._local.connection
        except PoolTimeout:
            raise CostSubsystemUnhealthy("PostgreSQL 비용 저장소 connection pool이 포화되었습니다.") from None
        except psycopg.Error:
            raise CostSubsystemUnhealthy("PostgreSQL 비용 저장소 작업에 실패했습니다.") from None

    def _verify_schema_cached(self, conn: psycopg.Connection) -> None:
        """성공한 schema 검증만 TTL 동안 캐시한다. 실패는 캐시하지 않는다."""
        ttl = _schema_verify_ttl_seconds()
        verified_at = self._verify_at
        now = time.monotonic()
        if ttl > 0 and verified_at is not None and self._verify_dsn == self._dsn:
            if now - verified_at < ttl:
                return
        _verify_schema(conn)
        self._verify_at = now
        self._verify_dsn = self._dsn

    @contextlib.contextmanager
    def transaction(self) -> Iterator[None]:
        outer = getattr(self._local, "connection", None) is None
        with self._connection() as conn:
            if outer:
                self._verify_schema_cached(conn)
                conn.execute("SELECT pg_advisory_xact_lock(%s)", (ADVISORY_LOCK_KEY,))
                yield
            else:
                with conn.transaction():
                    yield
                    _ensure_transaction(conn)

    @contextlib.contextmanager
    def _read(self) -> Iterator[psycopg.Connection]:
        outer = getattr(self._local, "connection", None) is None
        with self._connection() as conn:
            if outer:
                self._verify_schema_cached(conn)
            yield conn

    def initialize(self) -> None:
        with self._read():
            pass

    def check_health(self) -> None:
        with self._read() as conn:
            if conn.execute("SHOW transaction_read_only").fetchone()["transaction_read_only"] == "on":
                raise CostSubsystemUnhealthy("PostgreSQL 비용 저장소가 read-only 상태입니다.")
            rights = conn.execute(
                """
                WITH schema_rights AS (
                    SELECT has_schema_privilege(%s, 'USAGE') AS schema_allowed
                ), function_rights AS (
                    SELECT schema_allowed,
                           CASE WHEN schema_allowed THEN
                               has_function_privilege('pg_catalog.pg_advisory_xact_lock(bigint)', 'EXECUTE')
                           ELSE false END AS function_allowed
                    FROM schema_rights
                )
                SELECT schema_allowed, function_allowed,
                       CASE WHEN schema_allowed AND function_allowed THEN (
                           SELECT bool_and(has_table_privilege(table_name, privilege))
                           FROM unnest(%s::text[]) AS tables(table_name)
                           CROSS JOIN unnest(%s::text[]) AS privileges(privilege)
                       ) ELSE false END AS tables_allowed
                FROM function_rights
                """,
                (SCHEMA, [f"{SCHEMA}.{table}" for table in _TABLE_COLUMNS], ["SELECT", "INSERT", "UPDATE", "DELETE"]),
            ).fetchone()
            if not rights["schema_allowed"]:
                raise CostSubsystemUnhealthy("PostgreSQL 비용 schema 접근 권한이 없습니다.")
            if not rights["function_allowed"]:
                raise CostSubsystemUnhealthy("PostgreSQL 비용 lock 실행 권한이 없습니다.")
            if not rights["tables_allowed"]:
                raise CostSubsystemUnhealthy("PostgreSQL 비용 저장소 runtime 권한이 부족합니다.")

    def close(self) -> None:
        if getattr(self._local, "connection", None) is not None:
            raise CostSubsystemUnhealthy("진행 중인 비용 transaction은 먼저 종료해야 합니다.")
        pool = self._pool
        retired: list[ConnectionPool] = []
        if pool is not None:
            with self._pool_lock:
                if self._pool is pool:
                    self._pool = None
                    self._pool_dsn = None
                    retired, self._retired_pools = self._retired_pools, []
        for item in ([pool] if pool is not None else []) + retired:
            item.close(timeout=5.0)

    def prepare_event(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        if set(fields) - set(LEDGER_ALLOWED_FIELDS):
            raise CostLedgerValidationError("비용 event에 허용되지 않은 필드가 있습니다.")
        row = {name: fields.get(name) for name in LEDGER_ALLOWED_FIELDS}
        row["event_id"] = row["event_id"] or f"costevt-{uuid.uuid4().hex}"
        row["created_at"] = row["created_at"] or self._now_fn()
        row = {name: _normalize(name, value) for name, value in row.items()}
        if row["status"] == "reserved" and (
            not row["reservation_id"] or row["billing_eligible"] != 1
            or row["forecast_cost_usd"] is None or row["estimated_cost_usd"] is None
            or _money(row["forecast_cost_usd"]) != _money(row["estimated_cost_usd"])
        ):
            raise CostLedgerValidationError("예약에는 고유 ID, 청구 가능 표시 및 동일한 forecast/estimated 비용이 필요합니다.")
        return row

    def record_event(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        return self.insert_event(fields)

    def insert_event(self, fields: Mapping[str, Any]) -> dict[str, Any]:
        normalized = self.prepare_event(fields)
        with self.transaction():
            conn = self.connection
            existing = conn.execute(
                f"SELECT * FROM {SCHEMA}.cost_events WHERE event_id = %s", (normalized["event_id"],),
            ).fetchone()
            if existing is not None:
                stored = _api_row(existing)
                if not fields.get("created_at"):
                    normalized["created_at"] = stored["created_at"]
                if not _same_fields(normalized, stored):
                    raise CostLedgerValidationError("동일 event ID의 내용이 충돌합니다.")
                return stored
            if normalized["reservation_id"] is not None and conn.execute(
                f"SELECT 1 FROM {SCHEMA}.cost_events WHERE reservation_id = %s", (normalized["reservation_id"],),
            ).fetchone():
                raise CostLedgerValidationError("이미 사용 중인 reservation ID입니다.")
            columns = ", ".join(LEDGER_ALLOWED_FIELDS)
            placeholders = ", ".join("%s" for _ in LEDGER_ALLOWED_FIELDS)
            values = [_db_value(name, normalized[name]) for name in LEDGER_ALLOWED_FIELDS]
            row = conn.execute(
                f"INSERT INTO {SCHEMA}.cost_events ({columns}) VALUES ({placeholders}) RETURNING *", values,
            ).fetchone()
            return _api_row(row)

    def fetch_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._read() as conn:
            rows = conn.execute(
                f"SELECT * FROM {SCHEMA}.cost_events ORDER BY created_at ASC, event_id ASC LIMIT %s", (limit,),
            ).fetchall()
            return [_api_row(row) for row in rows]

    def sum_estimated_since(
        self,
        cutoff: datetime,
        statuses: tuple[str, ...] | None = None,
        providers: tuple[str, ...] | None = None,
    ) -> Decimal:
        clause, params = _filters(statuses, providers)
        window = "created_at >= %s"
        cost = "estimated_cost_usd"
        if statuses is not None and "reserved" in statuses:
            window = "(created_at >= %s OR status = 'reserved')"
            cost = "CASE WHEN status = 'reserved' THEN forecast_cost_usd ELSE estimated_cost_usd END"
        with self._read() as conn:
            return conn.execute(
                f"SELECT COALESCE(SUM({cost}), 0) AS total FROM {SCHEMA}.cost_events "
                f"WHERE {window} AND billing_eligible = 1 {clause}",
                (_timestamp(cutoff), *params),
            ).fetchone()["total"]

    def daily_estimated_spend(self, day: str | date, providers: tuple[str, ...] | None = None) -> Decimal:
        day_value = _day(day)
        start = datetime(day_value.year, day_value.month, day_value.day, tzinfo=UTC)
        clause, params = _filters(("reserved", "finalized", "estimated_only"), providers)
        with self._read() as conn:
            return conn.execute(
                f"SELECT COALESCE(SUM(estimated_cost_usd), 0) AS total FROM {SCHEMA}.cost_events "
                f"WHERE created_at >= %s AND created_at < %s AND billing_eligible = 1 {clause}",
                (start, start + timedelta(days=1), *params),
            ).fetchone()["total"]

    def usage_summary_since(
        self,
        cutoff: datetime,
        statuses: tuple[str, ...] | None = None,
        providers: tuple[str, ...] | None = None,
    ) -> CostUsageSummary:
        clause, params = _filters(statuses, providers)
        with self._read() as conn:
            row = conn.execute(
                f"""
                SELECT COALESCE(SUM(estimated_cost_usd), 0) AS estimated_cost_usd,
                       COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                       COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                       COALESCE(SUM(CASE WHEN total_tokens > 0 THEN total_tokens
                           ELSE prompt_tokens::NUMERIC + completion_tokens
                                + embedding_tokens END), 0) AS total_tokens,
                       COUNT(*) AS event_count
                FROM {SCHEMA}.cost_events
                WHERE created_at >= %s AND billing_eligible = 1 {clause}
                """,
                (_timestamp(cutoff), *params),
            ).fetchone()
            return CostUsageSummary(
                estimated_cost_usd=row["estimated_cost_usd"], prompt_tokens=int(row["prompt_tokens"]),
                completion_tokens=int(row["completion_tokens"]), total_tokens=int(row["total_tokens"]),
                event_count=int(row["event_count"]),
            )

    def update_reservation(
        self,
        reservation_id: str,
        *,
        status: str,
        billing_eligible: bool,
        usage: NormalizedUsage,
        estimated_cost_usd: Decimal,
        finalized_at: str,
    ) -> None:
        if not isinstance(reservation_id, str) or not reservation_id:
            raise CostLedgerValidationError("정산할 reservation ID가 없습니다.")
        if not isinstance(status, str) or (
            status not in {"finalized", "estimated_only"} and not status.startswith("released_")
        ):
            raise CostLedgerValidationError("지원하지 않는 예약 정산 상태입니다.")
        if billing_eligible is None:
            raise CostLedgerValidationError("정산에는 billing_eligible 값이 필요합니다.")
        changes = {
            "status": status, "billing_eligible": billing_eligible,
            "prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens or usage.prompt_tokens + usage.completion_tokens + usage.embedding_tokens,
            "embedding_tokens": usage.embedding_tokens, "rerank_units": usage.rerank_units,
            "estimated_cost_usd": _money(estimated_cost_usd),
        }
        changes = {name: _normalize(name, value) for name, value in changes.items()}
        finalized = _timestamp(finalized_at)
        with self.transaction():
            conn = self.connection
            existing = conn.execute(
                f"SELECT * FROM {SCHEMA}.cost_events WHERE reservation_id = %s", (reservation_id,),
            ).fetchone()
            if existing is None:
                raise CostLedgerValidationError("정산할 reservation ID가 없습니다.")
            if existing["status"] != "reserved":
                if _same_fields(changes, _api_row(existing)):
                    return
                raise CostLedgerValidationError("종료된 예약의 정산 내용이 충돌합니다.")
            assignments = ", ".join(f"{name} = %s" for name in changes)
            conn.execute(
                f"UPDATE {SCHEMA}.cost_events SET {assignments}, finalized_at = %s WHERE reservation_id = %s",
                (*(_db_value(name, value) for name, value in changes.items()), finalized, reservation_id),
            )

    def record_reconciliation_result(self, result: ReconciliationResult) -> None:
        values = (
            _day(result.day).isoformat(),
            None if result.wrapper_estimated_cost_usd is None else _money(result.wrapper_estimated_cost_usd),
            None if result.billing_export_cost is None else _money(result.billing_export_cost),
            None if result.delta_usd is None else _money(result.delta_usd, signed=True),
            result.status, _timestamp(result.checked_at), result.error_message,
        )
        columns = _TABLE_COLUMNS["cost_reconciliation_results"]
        assignments = ", ".join(f"{name} = EXCLUDED.{name}" for name in columns if name != "day")
        with self.transaction():
            self.connection.execute(
                f"INSERT INTO {SCHEMA}.cost_reconciliation_results ({', '.join(columns)}) "
                f"VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (day) DO UPDATE SET {assignments}", values,
            )

    def latest_reconciliation_result(self) -> dict[str, Any] | None:
        with self._read() as conn:
            row = conn.execute(
                f"SELECT * FROM {SCHEMA}.cost_reconciliation_results ORDER BY checked_at DESC, day DESC LIMIT 1",
            ).fetchone()
            return _api_row(row) if row is not None else None

    def prune(self, *, now: datetime | None = None) -> dict[str, int]:
        current = _timestamp(now or self._now_fn())
        request_cutoff = current - timedelta(days=self.request_retention_days)
        aggregate_cutoff = (current - timedelta(days=self.aggregate_retention_months * 31)).date().isoformat()
        with self.transaction():
            conn = self.connection
            events = conn.execute(
                f"DELETE FROM {SCHEMA}.cost_events WHERE created_at < %s AND status IS DISTINCT FROM 'reserved'",
                (request_cutoff,),
            ).rowcount
            counts = {"cost_events": events}
            for table in ("cost_daily_aggregates", "cost_reconciliation_results"):
                counts[table] = conn.execute(f"DELETE FROM {SCHEMA}.{table} WHERE day < %s", (aggregate_cutoff,)).rowcount
            return counts