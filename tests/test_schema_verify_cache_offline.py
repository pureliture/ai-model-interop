from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import psycopg
import pytest

from openai_compatible_bridge.core import postgres_cost_repository as repository


@pytest.fixture
def cache(monkeypatch):
    clock = SimpleNamespace(monotonic=0.0, wall=1000.0)
    monkeypatch.setattr("time.monotonic", lambda: clock.monotonic)
    monkeypatch.setattr(repository, "_utcnow", lambda: datetime.fromtimestamp(clock.wall, UTC))
    monkeypatch.setenv("COST_SCHEMA_VERIFY_TTL_SECONDS", "30")
    verify = Mock()
    monkeypatch.setattr(repository, "_verify_schema", verify)
    return repository.PostgresCostRepository("offline-dsn"), clock, verify


def test_backward_wall_clock_does_not_extend_ttl(cache):
    repo, clock, verify = cache
    conn = object()
    repo._verify_schema_cached(conn)
    assert verify.call_count == 1
    clock.monotonic = 29.0
    clock.wall = 500.0
    repo._verify_schema_cached(conn)
    assert verify.call_count == 1
    clock.monotonic = 30.0
    repo._verify_schema_cached(conn)
    assert verify.call_count == 2


@pytest.mark.parametrize("error", [psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn])
def test_schema_error_invalidates_cache_even_near_monotonic_origin(cache, monkeypatch, error):
    repo, clock, verify = cache
    conn = Mock()
    conn.transaction.return_value = nullcontext()
    pool = Mock()
    pool.connection.return_value = nullcontext(conn)
    monkeypatch.setattr(repo, "_get_pool", lambda: pool)
    repo._verify_schema_cached(conn)
    assert verify.call_count == 1
    clock.monotonic = 1.0
    with pytest.raises(repository.CostSubsystemUnhealthy):
        with repo._connection():
            raise error("offline test")
    repo._verify_schema_cached(conn)
    assert verify.call_count == 2


def test_cache_disabled_dsn_change_and_failed_verification(cache, monkeypatch):
    repo, clock, verify = cache
    conn = object()
    repo._verify_schema_cached(conn)
    repo._dsn = "other-offline-dsn"
    repo._verify_schema_cached(conn)
    assert verify.call_count == 2
    monkeypatch.setenv("COST_SCHEMA_VERIFY_TTL_SECONDS", "0")
    repo._verify_schema_cached(conn)
    repo._verify_schema_cached(conn)
    assert verify.call_count == 4
    monkeypatch.setenv("COST_SCHEMA_VERIFY_TTL_SECONDS", "30")
    clock.monotonic = 30.0
    verify.side_effect = [repository.CostSubsystemUnhealthy("failed"), None]
    with pytest.raises(repository.CostSubsystemUnhealthy):
        repo._verify_schema_cached(conn)
    repo._verify_schema_cached(conn)
    assert verify.call_count == 6
