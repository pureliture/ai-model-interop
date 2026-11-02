"""Load the six runtime-enforced bounds from the published contract."""

import json
from pathlib import Path

from .env_contract import validate_contract

_CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "bridge-env-contract.json"
_RUNTIME_TYPES = {
    "EMBEDDINGS_RETRY_MAX_ATTEMPTS": "integer",
    "EMBEDDINGS_RETRY_INITIAL_DELAY_SECONDS": "number",
    "EMBEDDINGS_RETRY_MAX_DELAY_SECONDS": "number",
    "EMBEDDINGS_RETRY_TIMEOUT_SECONDS": "number",
    "COST_ADMISSION_WORKERS": "integer",
    "COST_ADMISSION_TIMEOUT_SECONDS": "number",
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate contract key: {key}")
        result[key] = value
    return result


def load_limits(path):
    with Path(path).open(encoding="utf-8") as source:
        contract = json.load(source, object_pairs_hook=_unique_object)
    validate_contract(contract)
    settings = contract["settings"]
    result = {}
    for name, kind in _RUNTIME_TYPES.items():
        spec = settings.get(name)
        if not isinstance(spec, dict) or spec.get("type") != kind or not {"min", "max"} <= spec.keys():
            raise ValueError(f"invalid runtime limit: {name}")
        result[name] = spec
    maximum_delay = result["EMBEDDINGS_RETRY_MAX_DELAY_SECONDS"]
    if maximum_delay.get("minimum_key") != "EMBEDDINGS_RETRY_INITIAL_DELAY_SECONDS":
        raise ValueError("invalid runtime limit: EMBEDDINGS_RETRY_MAX_DELAY_SECONDS")
    return result


LIMITS = load_limits(_CONTRACT)
