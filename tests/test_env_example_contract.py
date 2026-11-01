"""Keep the committed env example and its source-owned admission contract aligned."""
from __future__ import annotations

import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def _env_example() -> dict[str, str]:
    values = {}
    for line in (ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Z][A-Z0-9_]*)=(.*)", line)
        assert match is not None, "invalid .env.example assignment"
        key, value = match.groups()
        assert key not in values, f"duplicate .env.example setting: {key}"
        values[key] = value.strip().strip("\"'")
    return values


def _contract() -> dict:
    from openai_compatible_bridge.core.env_contract import validate_contract

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate contract key: {key}")
            result[key] = value
        return result

    result = json.loads((ROOT / "contracts" / "bridge-env-contract.json").read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    validate_contract(result)
    return result


def test_env_example_and_contract_have_the_same_settings():
    contract = _contract()
    assert contract["schema"] == "bridge-env-example-contract/v1"
    assert set(contract["settings"]) == set(_env_example())


def test_contract_limits_match_non_embedding_runtime_settings():
    from decimal import Decimal

    from openai_compatible_bridge.core.cost_tracking import (
        CostConfigError,
        _parse_budget_limit,
        _parse_positive_int,
    )
    from openai_compatible_bridge.main import SUPPORTED_TASK_TYPES
    from openai_compatible_bridge.providers.ollama import _ollama_think_from_env

    settings = _contract()["settings"]
    assert set(settings["VERTEX_TASK_TYPE_DEFAULT"]["values"]) == SUPPORTED_TASK_TYPES
    assert settings["OLLAMA_THINK"]["case_insensitive"] is True
    for value in settings["OLLAMA_THINK"]["values"]:
        _ollama_think_from_env(value)
        _ollama_think_from_env(value.upper())

    assert settings["COST_SHORT_WINDOW_SECONDS"] == {"type": "integer", "min": 1}
    assert settings["COST_SHORT_WINDOW_LIMIT_USD"]["type"] == "budget"
    assert settings["COST_DAILY_LIMIT_USD"]["type"] == "budget"
    assert _parse_budget_limit("unlimited", "test") is None
    assert _parse_budget_limit("1.25", "test") == Decimal("1.25")
    try:
        _parse_budget_limit("0", "test")
    except CostConfigError:
        pass
    else:
        raise AssertionError("runtime budget parser accepted zero")
    assert _parse_positive_int("1", "test") == 1
    assert settings["COST_LEDGER_BACKEND"]["case_insensitive"] is True
    assert settings["STRUCTURED_OUTPUT_REPAIR_MODELS"]["max_items"] == 3


def test_embedding_runtime_limits_are_declared_in_the_source_contract():
    settings = _contract()["settings"]
    assert settings["EMBEDDINGS_QUEUE_TIMEOUT_SECONDS"] == {
        "type": "number", "min": 0,
    }


def test_example_embedding_defaults_are_accepted_by_runtime_validators(monkeypatch):
    from openai_compatible_bridge.core.embedding_resilience import EmbeddingRetryPolicy
    from openai_compatible_bridge.embedding_limits import EmbeddingRequestLimits

    values = _env_example()
    policy = EmbeddingRetryPolicy.from_env(values)
    assert policy.max_attempts == int(values["EMBEDDINGS_RETRY_MAX_ATTEMPTS"])

    for key in (
        "EMBEDDINGS_MAX_REQUEST_BYTES", "EMBEDDINGS_MAX_INFLIGHT",
        "EMBEDDINGS_QUEUE_TIMEOUT_SECONDS",
    ):
        monkeypatch.setenv(key, values[key])
    limits = EmbeddingRequestLimits(lambda _scope: None)
    assert limits.max_bytes == int(values["EMBEDDINGS_MAX_REQUEST_BYTES"])
    assert limits.max_inflight == int(values["EMBEDDINGS_MAX_INFLIGHT"])
    assert limits.queue_timeout == float(values["EMBEDDINGS_QUEUE_TIMEOUT_SECONDS"])


def test_runtime_uses_shared_retry_limit_contract(monkeypatch):
    import pytest
    from openai_compatible_bridge.core import env_limits
    from openai_compatible_bridge.core.embedding_resilience import EmbeddingRetryPolicy

    monkeypatch.setitem(env_limits.LIMITS["EMBEDDINGS_RETRY_MAX_ATTEMPTS"], "max", 2)
    with pytest.raises(ValueError):
        EmbeddingRetryPolicy.from_env({"EMBEDDINGS_RETRY_MAX_ATTEMPTS": "3"})


def test_shared_runtime_limits_match_published_contract():
    from openai_compatible_bridge.core.env_limits import LIMITS

    settings = _contract()["settings"]
    for name, spec in LIMITS.items():
        assert settings[name] == spec


def test_contract_schema_rejects_malformed_specs():
    import copy
    import pytest
    from openai_compatible_bridge.core.env_contract import validate_contract

    good = _contract()
    validate_contract(good)
    cases = []
    bad = copy.deepcopy(good); bad["settings"]["HTTP_TIMEOUT_SECONDS"]["typo"] = 1; cases.append(bad)
    bad = copy.deepcopy(good); bad["settings"]["HTTP_TIMEOUT_SECONDS"]["min"] = True; cases.append(bad)
    bad = copy.deepcopy(good); bad["settings"]["HTTP_TIMEOUT_SECONDS"]["max"] = 0; cases.append(bad)
    bad = copy.deepcopy(good); bad["settings"]["EMBEDDINGS_RETRY_MAX_DELAY_SECONDS"]["minimum_key"] = "MISSING"; cases.append(bad)
    bad = copy.deepcopy(good); bad["settings"]["OLLAMA_THINK"]["values"] = []; cases.append(bad)
    for bad in cases:
        with pytest.raises(ValueError):
            validate_contract(bad)


def test_all_shared_bounds_drive_runtime(monkeypatch):
    import pytest
    from openai_compatible_bridge.core import env_limits
    from openai_compatible_bridge.core.async_cost import _admission_timeout_seconds, _admission_worker_count
    from openai_compatible_bridge.core.cost_tracking import CostConfigError
    from openai_compatible_bridge.core.embedding_resilience import EmbeddingRetryPolicy

    for name in (
        "EMBEDDINGS_RETRY_MAX_ATTEMPTS", "EMBEDDINGS_RETRY_INITIAL_DELAY_SECONDS",
        "EMBEDDINGS_RETRY_MAX_DELAY_SECONDS", "EMBEDDINGS_RETRY_TIMEOUT_SECONDS",
    ):
        spec = env_limits.LIMITS[name]
        with monkeypatch.context() as patch:
            patch.setitem(spec, "max", spec["min"])
            with pytest.raises(ValueError):
                EmbeddingRetryPolicy.from_env({name: str(spec["min"] + 1)})
    for name, parser in (("COST_ADMISSION_WORKERS", _admission_worker_count),
                         ("COST_ADMISSION_TIMEOUT_SECONDS", _admission_timeout_seconds)):
        spec = env_limits.LIMITS[name]
        with monkeypatch.context() as patch:
            patch.setitem(spec, "max", spec["min"])
            with pytest.raises(CostConfigError):
                parser({name: str(spec["min"] + 1)})


def test_runtime_limits_load_from_contract_file(tmp_path):
    from openai_compatible_bridge.core.env_limits import load_limits

    contract = _contract()
    contract["settings"]["EMBEDDINGS_RETRY_MAX_ATTEMPTS"]["max"] = 3
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    assert load_limits(path)["EMBEDDINGS_RETRY_MAX_ATTEMPTS"]["max"] == 3


def test_runtime_limits_reject_invalid_contract(tmp_path):
    import pytest
    from openai_compatible_bridge.core.env_limits import load_limits

    contract = _contract()
    del contract["settings"]["COST_ADMISSION_WORKERS"]["max"]
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError):
        load_limits(path)
    path.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_limits(path)
