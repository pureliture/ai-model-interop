"""ProviderAdapter 프로토콜 + provider별 usage/error 파서.

Step 1: metered_http의 body 스니핑을 provider 소유로 옮긴다.
- 기존 `_raw_usage`의 분기 로직을 provider별로 분리하되 동작은 동일 유지.
- `MeteredHTTPClient`는 `provider`로 dispatcher를 호출한다.
- 미지정/unknown provider는 legacy 폴백으로 기존 동작 유지 (테스트 호환).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from openai_compatible_bridge.core.cost_tracking import NormalizedUsage

SUPPORTED_PROVIDERS: tuple[str, ...] = ("vertex", "ollama", "foundry", "openai_compat", "openai_compatible", "jetbrains")


def _count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _chat_usage(
    usage: dict[str, Any], input_key: str, output_key: str, total_key: str = "total_tokens",
    *, thoughts_key: str | None = None,
) -> NormalizedUsage | None:
    prompt = _count(usage.get(input_key))
    completion = _count(usage.get(output_key))
    if prompt is None or completion is None:
        return None
    if thoughts_key is not None and thoughts_key in usage:
        thoughts = _count(usage[thoughts_key])
        if thoughts is None:
            return None
        completion += thoughts
    total = _count(usage[total_key]) if total_key in usage else prompt + completion
    if total is None:
        return None
    return NormalizedUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=total)


def _is_error(payload: Any) -> bool:
    return (
        payload.get("error") is not None
        if isinstance(payload, dict)
        else False
    ) or (
        payload.get("type") in ("error", "response.failed", "response.incomplete")
        if isinstance(payload, dict)
        else False
    ) or (
        payload.get("status") in ("failed", "cancelled", "incomplete")
        if isinstance(payload, dict)
        else False
    )


def _embedding_count(payload: dict[str, Any]) -> int | None:
    if "usageMetadata" in payload:
        usage = _object(payload["usageMetadata"])
        if "totalTokenCount" in usage and _count(usage["totalTokenCount"]) is None:
            return None
        for key in ("promptTokenCount", "totalTokenCount", "tokenCount", "token_count", "inputTokens", "input_tokens"):
            if key in usage:
                return _count(usage[key])
        return None
    if "usage" in payload:
        usage = _object(payload["usage"])
        key = "prompt_tokens" if "prompt_tokens" in usage else "input_tokens"
        if "total_tokens" in usage and _count(usage["total_tokens"]) is None:
            return None
        return _count(usage.get(key))
    if "embedding" in payload:
        stats = _object(_object(payload["embedding"]).get("statistics"))
        return _count(stats.get("token_count"))
    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        return None
    total = 0
    for prediction in predictions:
        embedding = _object(_object(prediction).get("embeddings"))
        tokens = _count(_object(embedding.get("statistics")).get("token_count"))
        if tokens is None:
            return None
        total += tokens
    return total


def parse_vertex_usage(payload: Any, endpoint: str) -> NormalizedUsage | None:
    """Vertex provider가 emit할 수 있는 shape만 처리한다."""
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
    usage = _object(payload.get("usage"))
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        return _chat_usage(usage, "prompt_tokens", "completion_tokens")
    # openapiChatCompletions 외에 Vertex가 직접 input_tokens를 내진 않지만
    # 하위호환용으로 유지한다.
    if "input_tokens" in usage or "output_tokens" in usage:
        return _chat_usage(usage, "input_tokens", "output_tokens")
    return None


def parse_ollama_usage(payload: Any, endpoint: str) -> NormalizedUsage | None:
    """Ollama provider가 emit할 수 있는 shape만 처리한다."""
    payload = _object(payload)
    if _is_error(payload):
        return None
    if endpoint != "chat":
        return None
    if payload.get("type") == "response.completed":
        return None
    if "prompt_eval_count" in payload or "eval_count" in payload:
        return _chat_usage(payload, "prompt_eval_count", "eval_count")
    usage = _object(payload.get("usage"))
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        return _chat_usage(usage, "prompt_tokens", "completion_tokens")
    return None


def parse_protocol_usage(payload: Any, endpoint: str) -> NormalizedUsage | None:
    """Parse usage emitted by the supported native chat protocols.

    openai_chat_completions / anthropic_messages / xai_responses /
    openai_responses / google_generate_content의 raw payload를 커버한다.
    """
    payload = _object(payload)
    if _is_error(payload):
        return None
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
        return None
    usage = _object(payload.get("usage"))
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        return _chat_usage(usage, "prompt_tokens", "completion_tokens")
    if "input_tokens" in usage or "output_tokens" in usage:
        return _chat_usage(usage, "input_tokens", "output_tokens")
    return None


def parse_openai_compat_usage(payload: Any, endpoint: str) -> NormalizedUsage | None:
    """Generic OpenAI upstream이 emit하는 OpenAI-shape usage만 처리한다."""
    payload = _object(payload)
    if _is_error(payload):
        return None
    if endpoint != "chat":
        return None
    if payload.get("type") == "response.completed":
        payload = _object(payload.get("response"))
        if _is_error(payload):
            return None
    usage = _object(payload.get("usage"))
    if "prompt_tokens" in usage or "completion_tokens" in usage:
        return _chat_usage(usage, "prompt_tokens", "completion_tokens")
    if "input_tokens" in usage or "output_tokens" in usage:
        return _chat_usage(usage, "input_tokens", "output_tokens")
    return None


def parse_usage_for_provider(provider: str | None, payload: Any, endpoint: str) -> NormalizedUsage | None:
    """provider별 파서 dispatcher. unknown/None이면 None을 반환해 caller가 legacy로 폴백."""
    if provider == "vertex":
        return parse_vertex_usage(payload, endpoint)
    if provider == "ollama":
        return parse_ollama_usage(payload, endpoint)
    if provider == "foundry":
        return parse_protocol_usage(payload, endpoint)
    if provider in {"openai_compat", "openai_compatible"}:
        return parse_openai_compat_usage(payload, endpoint)
    if provider == "jetbrains":
        return parse_protocol_usage(payload, endpoint)
    return None


# ---------------------------------------------------------------------------
# Step 2: 가격/파서 선언형 스펙 (adapter 소유, async_cost는 예약만)
# ---------------------------------------------------------------------------

ENDPOINT_PRICE_DIMENSIONS: dict[str, frozenset[str]] = {
    "chat": frozenset({"input_per_million", "output_per_million"}),
    "embeddings": frozenset({"embedding_per_million"}),
    "rerank": frozenset({"rerank_per_unit"}),
}

PROVIDER_ENDPOINTS: dict[str, tuple[str, ...]] = {
    "vertex": ("chat", "embeddings", "rerank"),
    "ollama": ("chat",),
    "foundry": ("chat",),
    "openai_compat": ("chat",),
    "openai_compatible": ("chat",),
    "jetbrains": ("chat",),
}

# endpoint별 usage 스키마 (adapter가 소유하는 과금 차원 선언).
ENDPOINT_USAGE_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "chat": ("prompt_tokens", "completion_tokens"),
    "embeddings": ("embedding_tokens",),
    "rerank": ("rerank_units",),
}


def required_price_dimensions(endpoint: str) -> frozenset[str]:
    return ENDPOINT_PRICE_DIMENSIONS.get(endpoint, frozenset())


def validate_pricing_payload(data: Any) -> None:
    """가격 JSON의 endpoint/차원을 선언형 스펙으로 검증한다 (fail-closed 유지)."""
    from openai_compatible_bridge.core.cost_tracking import CostConfigError

    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, dict):
        return
    for endpoint_map in models.values():
        if not isinstance(endpoint_map, dict):
            continue
        for endpoint, price in endpoint_map.items():
            required = ENDPOINT_PRICE_DIMENSIONS.get(endpoint)
            if required is None or not isinstance(price, dict):
                raise CostConfigError("explicit pricing rates are required")
            if not required <= price.keys():
                raise CostConfigError("explicit pricing rates are required")


def pricing_fallbacks(model: str) -> tuple[str, ...]:
    """모델별 가격 fallback 체인 (adapter 소유).

    - `ollama:<native>` dynamic 모델은 `ollama:*`로 폴백 (legacy 동작 유지).
    - `compat:<upstream>` dynamic 모델은 `compat:*`로 폴백.
    - 그 외는 exact 매칭만 (fail-closed).
    """
    if model.startswith("ollama:") and model != "ollama:*":
        return ("ollama:*",)
    if model.startswith("compat:") and model != "compat:*":
        return ("compat:*",)
    return ()


def map_http_error_for_provider(provider: str | None, response: Any) -> Any:
    """provider별 HTTP 에러 매핑. lazy import로 순환 참조를 피한다."""
    if provider == "foundry":
        from openai_compatible_bridge.providers.foundry import _parse_foundry_error

        return _parse_foundry_error(response)
    if provider == "ollama":
        from openai_compatible_bridge.providers.vertex import VertexAPIError

        # legacy inline과 byte-identical: raw 미첨부, json 파싱 없음.
        return VertexAPIError(
            response.status_code, "Ollama request failed",
            code=str(response.status_code),
        )
    if provider in {"openai_compat", "openai_compatible"}:
        from openai_compatible_bridge.providers.vertex import VertexAPIError

        try:
            error_payload = response.json()
        except Exception:
            error_payload = {"error": {"message": response.text}}
        err = error_payload.get("error", {}) if isinstance(error_payload, dict) else {}
        if isinstance(err, str):
            err = {"message": err}
        message = err.get("message") or response.text or "OpenAI-compat request failed"
        code = err.get("code") or err.get("type") or str(response.status_code)
        return VertexAPIError(
            response.status_code, message=str(message), code=str(code), raw=error_payload
        )
    if provider == "jetbrains":
        from openai_compatible_bridge.providers.vertex import VertexAPIError

        try:
            error_payload = response.json()
        except Exception:
            error_payload = {"error": {"message": response.text}}
        err = error_payload.get("error", {}) if isinstance(error_payload, dict) else {}
        if isinstance(err, str):
            err = {"message": err}
        message = err.get("message") or response.text or "JetBrains request failed"
        code = err.get("code") or err.get("type") or str(response.status_code)
        return VertexAPIError(
            response.status_code, message=str(message), code=str(code), raw=error_payload
        )
    # vertex + unknown은 기존 vertex 파서를 기본값으로 사용한다.
    from openai_compatible_bridge.providers.vertex import _parse_vertex_error

    return _parse_vertex_error(response)


@runtime_checkable
class ProviderAdapter(Protocol):
    """embed/rerank/chat provider 공통 최소 계약 (Step 1)."""

    provider_name: str

    async def close(self) -> None: ...

    def parse_usage(self, payload: Any, endpoint: str) -> NormalizedUsage | None: ...

    def map_http_error(self, response: Any) -> Any: ...


@runtime_checkable
class ChatProviderAdapter(ProviderAdapter, Protocol):
    """chat provider 추가 계약."""

    async def generate(self, **kwargs: Any) -> dict[str, Any]: ...

    def stream_chat(self, **kwargs: Any) -> Any: ...
