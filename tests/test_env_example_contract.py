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
    return json.loads((ROOT / "contracts" / "bridge-env-contract.json").read_text(encoding="utf-8"))


def test_env_example_and_contract_have_the_same_settings():
    contract = _contract()
    assert contract["schema"] == "bridge-env-example-contract/v1"
    assert set(contract["settings"]) == set(_env_example())


def test_embedding_runtime_limits_are_declared_in_the_source_contract():
    settings = _contract()["settings"]
    assert settings["EMBEDDINGS_QUEUE_TIMEOUT_SECONDS"] == {
        "type": "number", "min": 0,
    }
    assert settings["EMBEDDINGS_RETRY_MAX_ATTEMPTS"] == {
        "type": "integer", "min": 1, "max": 8,
    }
    assert settings["EMBEDDINGS_RETRY_INITIAL_DELAY_SECONDS"] == {
        "type": "number", "min": 0.01, "max": 30,
    }
    assert settings["EMBEDDINGS_RETRY_MAX_DELAY_SECONDS"] == {
        "type": "number", "min": 0.01, "max": 60,
        "minimum_key": "EMBEDDINGS_RETRY_INITIAL_DELAY_SECONDS",
    }
    assert settings["EMBEDDINGS_RETRY_TIMEOUT_SECONDS"] == {
        "type": "number", "min": 0.1, "max": 300,
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
