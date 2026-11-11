"""OpenAI-compatible embedding proxy for Google Cloud Vertex AI (Gemini Enterprise Agent Platform).

OpenAI-API 호환 임베딩 client가 호출하는 POST /v1/embeddings 를 받아서
Vertex AI :predict 엔드포인트로 통역한다.

- 인증: 서비스 계정(ADC)에서 short-lived OAuth2 access token을 자동 발급/캐시/갱신.
- 배치 분할: client는 요청당 여러 텍스트를 보내지만, Vertex 모델별 요청당 instance
  한도에 맞춰 쪼개 병렬 호출.
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import re

import httpx
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict

from openai_compatible_bridge.anthropic_ingress import (
    AnthropicBadRequest,
    AnthropicStreamState,
    DEFAULT_CONTEXT_LENGTH,
    PROVIDER_TOOL_SUPPORT,
    anthropic_error_response,
    map_status_to_anthropic_error_type,
    new_anthropic_message_id,
    openai_result_to_anthropic,
    parse_anthropic_chat_params,
    sse_event,
)
from openai_compatible_bridge.core.async_cost import AsyncCostAccounting, build_async_cost_accounting
from openai_compatible_bridge.core.cost_tracking import (
    BudgetBlock,
    BudgetReservationContext,
    CostBudgetExceeded,
    CostConfigError,
    CostSubsystemUnhealthy,
    DisabledCostAccounting,
    NormalizedUsage,
)
from openai_compatible_bridge.providers.foundry import FoundryChatClient
from openai_compatible_bridge.providers.jetbrains import JetBrainsChatClient, validate_jetbrains_parameters
from openai_compatible_bridge.junie_discovery import discovery_from_env
from openai_compatible_bridge.providers.ollama import OllamaChatClient
from openai_compatible_bridge.providers.openai_compatible import (
    OpenAICompatibleChatClient,
    _resolve_extra_headers,
)
from openai_compatible_bridge.providers.vertex import (
    SUPPORTED_RESPONSE_FORMAT_TYPES,
    VertexAPIError,
    VertexChatClient,
    VertexEmbeddingClient,
    VertexRerankClient,
    allowed_models as _vertex_allowed_models,
    model_config as _vertex_model_config,
    MODEL_REGISTRY,
)
from openai_compatible_bridge.providers import vertex as _vertex_mod

SUPPORTED_TASK_TYPES = {
    "UNSPECIFIED",
    "RETRIEVAL_QUERY",
    "RETRIEVAL_DOCUMENT",
    "SEMANTIC_SIMILARITY",
    "CLASSIFICATION",
    "CLUSTERING",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
    "CODE_RETRIEVAL_QUERY",
}

VERTEX_TASK_TYPE_DEFAULT = os.getenv("VERTEX_TASK_TYPE_DEFAULT", "RETRIEVAL_DOCUMENT")
BRIDGE_API_KEY = os.getenv("BRIDGE_API_KEY")
_COMPAT_MODEL_IDS: set[str] = set()
_JETBRAINS_MODEL_IDS: set[str] = set()
_COMPAT_MODEL_META: dict[str, dict[str, Any]] = {}


def _openai_compatible_base_config() -> dict[str, Any]:
    cfg: dict[str, Any] = {}
    template = MODEL_REGISTRY.get("compat:*")
    if isinstance(template, dict):
        cfg.update(template)
    if not cfg.get("base_url"):
        url = (OPENAI_COMPAT_DEFAULT_BASE_URL or os.getenv("OPENAI_COMPAT_DEFAULT_BASE_URL", "")).strip().rstrip("/")
        if url:
            cfg["base_url"] = url
    if not cfg.get("api_key") and not cfg.get("api_key_env"):
        key = (OPENAI_COMPAT_DEFAULT_API_KEY or os.getenv("OPENAI_COMPAT_DEFAULT_API_KEY", "")).strip()
        if key:
            cfg["api_key"] = key
    if not cfg.get("api_key"):
        env_name = cfg.get("api_key_env")
        if isinstance(env_name, str) and env_name:
            val = (os.getenv(env_name) or "").strip()
            if val:
                cfg["api_key"] = val
    cfg.setdefault("provider", "openai_compatible")
    cfg.setdefault("kind", "chat")
    cfg.setdefault("context_length", DEFAULT_CONTEXT_LENGTH)
    return cfg


def _compat_model_config(model_id: str) -> dict[str, Any] | None:
    upstream = model_id[len(OPENAI_COMPAT_DYNAMIC_MODEL_PREFIX):].strip()
    if not upstream:
        return None
    cfg = dict(_openai_compatible_base_config())
    cfg["provider"] = "openai_compatible"
    cfg["kind"] = "chat"
    cfg["provider_model"] = upstream
    cfg.setdefault("context_length", DEFAULT_CONTEXT_LENGTH)
    meta = _COMPAT_MODEL_META.get(upstream)
    if isinstance(meta, dict) and meta.get("context_length") is not None:
        try:
            cfg["context_length"] = int(meta["context_length"])
        except (TypeError, ValueError):
            pass
    exact = MODEL_REGISTRY.get(model_id)
    if isinstance(exact, dict):
        for key in (
            "display_name",
            "anthropic_family_tier",
            "is_family_default",
            "max_tokens",
            "context_length",
        ):
            if key in exact and exact[key] is not None:
                cfg[key] = exact[key]
    return cfg


def _main_model_config(model: str) -> dict[str, Any] | None:
    if model.startswith(OPENAI_COMPAT_DYNAMIC_MODEL_PREFIX):
        return _compat_model_config(model)
    return _vertex_model_config(model)


model_config = _main_model_config


def _main_allowed_models() -> set[str]:
    return _vertex_allowed_models() | _COMPAT_MODEL_IDS | _JETBRAINS_MODEL_IDS


allowed_models = _main_allowed_models
def _discover_compat_models() -> None:
    base = _openai_compatible_base_config()
    url = (base.get("base_url") or "").rstrip("/")
    if not url:
        return
    headers = {"Accept": "application/json"}
    api_key = base.get("api_key")
    if not api_key:
        env_name = base.get("api_key_env")
        if env_name:
            api_key = (os.getenv(env_name) or "").strip() or None
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        headers.update(_resolve_extra_headers(base))
    except Exception:
        pass
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{url}/models", headers=headers)
            resp.raise_for_status()
            data = resp.json()
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            items = data.get("models") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return
        ids: list[str] = []
        meta: dict[str, dict[str, Any]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            mid = item.get("id")
            if not isinstance(mid, str) or not mid:
                continue
            ids.append(mid)
            meta[mid] = item
        _COMPAT_MODEL_IDS.clear()
        _COMPAT_MODEL_IDS.update(f"{OPENAI_COMPAT_DYNAMIC_MODEL_PREFIX}{mid}" for mid in ids)
        _COMPAT_MODEL_META.clear()
        _COMPAT_MODEL_META.update(meta)
    except Exception as exc:  # pragma: no cover - best effort, network dependent
        logging.getLogger("openai_compatible_bridge").warning("compat model discovery failed: %s", exc)


def _reload_compat_discovery() -> None:
    _COMPAT_MODEL_IDS.clear()
    _COMPAT_MODEL_META.clear()
    try:
        _discover_compat_models()
    except Exception:
        pass


async def _registry_hot_reload_dispatch(request: Request, call_next):
    _vertex_mod._reload_registry_if_changed()
    return await call_next(request)


_vertex_mod.MODEL_REGISTRY_RELOAD_HOOK = _reload_compat_discovery


ALLOWED_MODELS = _main_allowed_models()
OLLAMA_DYNAMIC_MODEL_PREFIX = "ollama:"
OPENAI_COMPAT_DYNAMIC_MODEL_PREFIX = "compat:"
JETBRAINS_DYNAMIC_MODEL_PREFIX = "jetbrains:"
OPENAI_COMPAT_DEFAULT_BASE_URL = os.getenv("OPENAI_COMPAT_BASE_URL", "").strip().rstrip("/")
OPENAI_COMPAT_DEFAULT_API_KEY = os.getenv("OPENAI_COMPAT_API_KEY", "").strip()
MODEL_LIST_INCLUDE = os.getenv("MODEL_LIST_INCLUDE", "").strip()
STREAM_INCLUDE_USAGE_DEFAULT = os.getenv("STREAM_INCLUDE_USAGE_DEFAULT", "false").lower() in ("true", "1")
STRUCTURED_OUTPUT_REPAIR_LOGGER = logging.getLogger("structured_output_repair")
STRUCTURED_OUTPUT_REPAIR_LOGGER.setLevel(logging.INFO)
STRUCTURED_OUTPUT_REPAIR_RUNTIME_LOGGER = logging.getLogger("uvicorn.error")
STRUCTURED_OUTPUT_REPAIR_DEFAULT_MODELS = (
    "ollama:qwen3.5:cloud",
    "ollama:gemma4:31b-cloud",
    "ollama:glm-5.2:cloud",
)


def current_allowed_models() -> set[str]:
    return _main_allowed_models()


def _resolve_chat_model(model: str, discovered: dict[str, dict[str, Any]] | None = None) -> tuple[dict[str, Any] | None, JSONResponse | None]:
    if model not in _vertex_allowed_models() and model in (discovered or {}):
        cfg = discovered[model]
        if cfg.get("protocol") is None:
            return None, openai_error_response(
                message="The discovered JetBrains model has an unsupported protocol; configure an explicit registry entry.",
                status_code=400, error_type="invalid_request_error", code="unsupported_model", param="model",
            )
        return cfg, None
    if model.startswith(OLLAMA_DYNAMIC_MODEL_PREFIX):
        provider_model = model[len(OLLAMA_DYNAMIC_MODEL_PREFIX):].strip()
        if not provider_model:
            return None, openai_error_response(
                message="Dynamic Ollama model must use the form 'ollama:<native-model>'.",
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_model",
                param="model",
            )
        return {
            "provider": "ollama",
            "kind": "chat",
            "provider_model": provider_model,
        }, None

    if model.startswith(OPENAI_COMPAT_DYNAMIC_MODEL_PREFIX):
        provider_model = model[len(OPENAI_COMPAT_DYNAMIC_MODEL_PREFIX):].strip()
        if not provider_model:
            return None, openai_error_response(
                message="Dynamic compat model must use the form 'compat:<upstream-model>'.",
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_model",
                param="model",
            )
        base_cfg = _openai_compatible_base_config()
        if not base_cfg.get("base_url"):
            return None, openai_error_response(
                message="OPENAI_COMPAT_BASE_URL (or compat:* base_url) is not configured for dynamic compat models.",
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_model",
                param="model",
            )
        resolved = dict(base_cfg)
        resolved["provider"] = "openai_compatible"
        resolved["kind"] = "chat"
        resolved["provider_model"] = provider_model
        return resolved, None

    if model not in current_allowed_models() and model.startswith(JETBRAINS_DYNAMIC_MODEL_PREFIX):
        provider_model = model[len(JETBRAINS_DYNAMIC_MODEL_PREFIX):].strip()
        if not provider_model:
            return None, openai_error_response(
                message="Dynamic JetBrains model must use the form 'jetbrains:<native-model>'.",
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_model",
                param="model",
            )
        base_cfg = {
            "provider": "jetbrains",
            "kind": "chat",
            "provider_model": provider_model,
        }
        token = (os.getenv("JETBRAINS_AI_TOKEN") or "").strip()
        if token:
            base_cfg["api_key"] = token
        return base_cfg, None

    if model not in current_allowed_models():
        return None, openai_error_response(
            message=f"The model '{model}' does not exist.",
            status_code=404,
            error_type="invalid_request_error",
            code="model_not_found",
            param="model",
        )

    cfg = _main_model_config(model)
    if cfg and cfg.get("kind") != "chat":
        return None, openai_error_response(
            message=f"The model '{model}' is not a chat model.",
            status_code=400,
            error_type="invalid_request_error",
            code="invalid_model",
            param="model",
        )
    return cfg, None


class OpenAIEmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    input: str | list[str]
    model: str
    encoding_format: Literal["float", "base64"] = "float"
    dimensions: int | None = None
    user: str | None = None


class OpenAIChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[Any] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


def _normalize_chat_message(m: OpenAIChatMessage) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": m.role}
    if m.content is not None:
        msg["content"] = m.content
    else:
        msg["content"] = None
    if m.tool_calls is not None:
        msg["tool_calls"] = m.tool_calls
    if m.tool_call_id is not None:
        msg["tool_call_id"] = m.tool_call_id
    if m.name is not None:
        msg["name"] = m.name
    extra = getattr(m, "__pydantic_extra__", None) or {}
    for k, v in extra.items():
        if k not in msg and v is not None:
            msg[k] = v
    return msg


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")

    include_usage: bool | None = None


class OpenAIChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[OpenAIChatMessage]
    temperature: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    stream: bool | None = None
    stream_options: StreamOptions | dict[str, Any] | None = None
    user: str | None = None
    response_format: dict[str, Any] | None = None
    reasoning_effort: str | None = None
    reasoning: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None


def _should_include_stream_usage(payload: OpenAIChatRequest) -> bool:
    if payload.stream_options is not None:
        if isinstance(payload.stream_options, dict):
            return bool(payload.stream_options.get("include_usage"))
        return bool(getattr(payload.stream_options, "include_usage", False))
    return STREAM_INCLUDE_USAGE_DEFAULT


class CohereRerankRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    query: str
    documents: list[str | dict[str, Any]]
    top_n: int | None = None


def openai_error_response(
    *,
    message: str,
    status_code: int,
    error_type: str,
    code: str | None = None,
    param: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"message": message, "type": error_type, "param": param, "code": code}},
    )


def budget_exceeded_response(block: BudgetBlock) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "message": "Cost budget exceeded.",
                "type": "rate_limit_error",
                "param": None,
                "code": "budget_exceeded",
                "limit_type": block.limit_type,
                "reset_at": block.reset_at,
                "current_estimated_spend": str(block.current_estimated_spend),
                "configured_limit": str(block.configured_limit),
                "currency": block.currency,
            }
        },
    )


def map_vertex_status_to_openai_type(status_code: int) -> str:
    return {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "invalid_request_error",
        429: "rate_limit_error",
    }.get(status_code, "api_error" if status_code >= 500 else "invalid_request_error")


def coerce_string_inputs(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    raise ValueError("This wrapper supports only string or array[string] inputs.")


def encode_embedding(values: list[float], fmt: str) -> list[float] | str:
    if fmt == "base64":
        import base64
        import struct
        return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode("ascii")
    return values


def _estimate_text_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return 0 if value == "" else max(1, (len(value) + 3) // 4)
    if isinstance(value, list):
        return sum(_estimate_text_tokens(item) for item in value)
    if isinstance(value, dict):
        total = 0
        for key in ("text", "content"):
            item = value.get(key)
            if isinstance(item, (str, list, dict)):
                total += _estimate_text_tokens(item)
        return total
    return 0


def _default_chat_completion_tokens() -> int:
    try:
        return max(1, int(os.getenv("COST_CHAT_DEFAULT_MAX_OUTPUT_TOKENS", "4096")))
    except ValueError:
        return 4096


def _chat_max_tokens(payload: OpenAIChatRequest) -> int | None:
    """Return the OpenAI max-output-token field with modern-field precedence."""
    if payload.max_completion_tokens is not None:
        return payload.max_completion_tokens
    return payload.max_tokens


def _chat_forecast_usage(payload: OpenAIChatRequest) -> NormalizedUsage:
    prompt_tokens = sum(_estimate_text_tokens(message.content) for message in payload.messages)
    requested_max_tokens = _chat_max_tokens(payload)
    completion_tokens = (
        requested_max_tokens if requested_max_tokens is not None and requested_max_tokens > 0
        else _default_chat_completion_tokens()
    )
    return NormalizedUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _embedding_forecast_usage(texts: list[str]) -> NormalizedUsage:
    embedding_tokens = sum(_estimate_text_tokens(text) for text in texts)
    return NormalizedUsage(embedding_tokens=embedding_tokens, total_tokens=embedding_tokens)


def _chat_usage_from_mapping(usage: MappingLike) -> NormalizedUsage:
    prompt_tokens = _safe_int(usage.get("prompt_tokens"))
    completion_tokens = _safe_int(usage.get("completion_tokens"))
    total_tokens = _safe_int(usage.get("total_tokens")) or prompt_tokens + completion_tokens
    return NormalizedUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _safe_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


MappingLike = dict[str, Any]


def _cost_accounting(request: Request) -> Any:
    return getattr(request.app.state, "cost_accounting", None)


async def _cost_query(accounting: Any, method: str, **kwargs: Any) -> Any:
    try:
        if isinstance(accounting, AsyncCostAccounting):
            return await getattr(accounting, method)(**kwargs)
        return getattr(accounting, method)(**kwargs)
    except (CostBudgetExceeded, CostConfigError, CostSubsystemUnhealthy) as exc:
        return _cost_tracking_error_response(exc)


def _install_cost_clients(app: FastAPI) -> None:
    accounting = app.state.cost_accounting
    if not isinstance(accounting, AsyncCostAccounting):
        return
    from openai_compatible_bridge.core.metered_http import MeteredHTTPClient

    for name, provider in (
        ("vertex_client", "vertex"), ("vertex_chat_client", "vertex"),
        ("vertex_rerank_client", "vertex"), ("ollama_chat_client", "ollama"),
        ("foundry_chat_client", "foundry"),
        ("openai_compatible_chat_client", "openai_compat"),
        ("jetbrains_chat_client", "jetbrains"),
    ):
        client = getattr(app.state, name, None)
        if client is not None and hasattr(client, "http"):
            client.http = MeteredHTTPClient(client.http, accounting, provider=provider)


async def _close_cost_accounting(accounting: Any) -> None:
    if isinstance(accounting, AsyncCostAccounting):
        await accounting.aclose()
    elif accounting is not None:
        accounting.close()


def _cost_tracking_error_response(
    exc: CostBudgetExceeded | CostConfigError | CostSubsystemUnhealthy,
) -> JSONResponse:
    if isinstance(exc, CostBudgetExceeded):
        return budget_exceeded_response(exc.block)
    if isinstance(exc, CostConfigError):
        return openai_error_response(
            message=str(exc),
            status_code=503,
            error_type="api_error",
            code="cost_config_error",
        )
    return openai_error_response(
        message=str(exc),
        status_code=503,
        error_type="api_error",
        code="cost_tracking_unavailable",
    )


def _preflight_reservation_now(ctx: BudgetReservationContext) -> JSONResponse | None:
    try:
        ctx.preflight_now()
    except (CostBudgetExceeded, CostConfigError, CostSubsystemUnhealthy) as exc:
        return _cost_tracking_error_response(exc)
    return None


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _is_json_schema_response_format(response_format: dict[str, Any] | None) -> bool:
    return isinstance(response_format, dict) and response_format.get("type") == "json_schema"


def _structured_output_repair_enabled() -> bool:
    return _env_flag("STRUCTURED_OUTPUT_REPAIR_ENABLED")


def _structured_output_repair_model_ids() -> tuple[str, ...]:
    raw = os.getenv("STRUCTURED_OUTPUT_REPAIR_MODELS")
    if raw:
        values = tuple(item.strip() for item in raw.split(",") if item.strip())
    else:
        values = STRUCTURED_OUTPUT_REPAIR_DEFAULT_MODELS
    return values[:3]


def _dynamic_ollama_native_model(model_id: str) -> str | None:
    if not model_id.startswith(OLLAMA_DYNAMIC_MODEL_PREFIX):
        return None
    native = model_id[len(OLLAMA_DYNAMIC_MODEL_PREFIX):].strip()
    return native or None


def _is_ollama_cloud_native_model(native_model: str) -> bool:
    return native_model.endswith(":cloud") or native_model.endswith("-cloud")


def _structured_output_repair_applies(
    *,
    payload: "OpenAIChatRequest",
    provider: str,
    provider_model: str,
) -> bool:
    return (
        _structured_output_repair_enabled()
        and provider == "ollama"
        and payload.model.startswith(OLLAMA_DYNAMIC_MODEL_PREFIX)
        and _is_ollama_cloud_native_model(provider_model)
        and _is_json_schema_response_format(payload.response_format)
    )


def _structured_output_repair_native_models() -> tuple[str, ...]:
    models: list[str] = []
    invalid: list[str] = []
    for model_id in _structured_output_repair_model_ids():
        native = _dynamic_ollama_native_model(model_id)
        if native is not None and _is_ollama_cloud_native_model(native):
            models.append(native)
        else:
            invalid.append(model_id)
    if invalid or not models:
        raise VertexAPIError(
            503,
            "STRUCTURED_OUTPUT_REPAIR_MODELS must contain dynamic Ollama Cloud model ids.",
            code="cost_config_error",
            raw={"cost_managed": True},
        )
    return tuple(models)


def _build_structured_output_repair_messages(
    *,
    messages: list[dict[str, Any]],
    response_format: dict[str, Any] | None,
    failure_code: str | None,
) -> list[dict[str, Any]]:
    json_schema_obj = response_format.get("json_schema") if isinstance(response_format, dict) else None
    schema = json_schema_obj.get("schema") if isinstance(json_schema_obj, dict) else {}
    schema_json = json.dumps(schema, ensure_ascii=False, sort_keys=True)
    instruction = (
        "Return only JSON that validates against the provided JSON Schema. "
        "Do not include markdown, code fences, commentary, or reasoning text. "
        f"Previous failure category: {failure_code or 'invalid_schema_output'}. "
        f"JSON Schema: {schema_json}"
    )
    return [{"role": "system", "content": instruction}, *messages]


async def _generate_ollama_chat_once(
    chat_client: VertexChatClient,
    *,
    model: str,
    messages: list[dict[str, Any]],
    payload: "OpenAIChatRequest",
    repair_attempt: bool = False,
) -> dict[str, Any]:
    temperature = payload.temperature
    if repair_attempt and temperature is None:
        temperature = 0
    return await chat_client.generate(
        model=model,
        messages=messages,
        max_tokens=_chat_max_tokens(payload),
        temperature=temperature,
        top_p=payload.top_p,
        stop=payload.stop,
        response_format=payload.response_format,
        reasoning_effort=payload.reasoning_effort,
        reasoning=payload.reasoning,
    )


def _structured_output_latency_bucket(elapsed_seconds: float) -> str:
    if elapsed_seconds < 1:
        return "lt_1s"
    if elapsed_seconds < 5:
        return "lt_5s"
    if elapsed_seconds < 15:
        return "lt_15s"
    if elapsed_seconds < 60:
        return "lt_60s"
    return "gte_60s"


def _log_structured_output_repair_event(
    *,
    attempted_models: list[str],
    final_status: str,
    failure_category: str | None,
    content_chars: int,
    started_at: float,
) -> None:
    event = {
        "event": "structured_output_repair",
        "enabled": True,
        "attempt_count": max(0, len(attempted_models) - 1),
        "attempted_models": attempted_models,
        "final_status": final_status,
        "failure_category": failure_category or "invalid_schema_output",
        "latency_bucket": _structured_output_latency_bucket(time.monotonic() - started_at),
        "content_chars": max(0, content_chars),
    }
    STRUCTURED_OUTPUT_REPAIR_LOGGER.info(
        "structured_output_repair %s",
        json.dumps(event, ensure_ascii=False, sort_keys=True),
    )
    STRUCTURED_OUTPUT_REPAIR_RUNTIME_LOGGER.info(
        "structured_output_repair %s",
        json.dumps(event, ensure_ascii=False, sort_keys=True),
    )


def _usage_from_error(exc: VertexAPIError) -> NormalizedUsage:
    raw = exc.raw if isinstance(exc.raw, dict) else {}
    usage = raw.get("usage") if isinstance(raw, dict) else None
    return _chat_usage_from_mapping(usage if isinstance(usage, dict) else {})


def _add_usage(left: NormalizedUsage, right: NormalizedUsage) -> NormalizedUsage:
    prompt_tokens = left.prompt_tokens + right.prompt_tokens
    completion_tokens = left.completion_tokens + right.completion_tokens
    return NormalizedUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def _usage_to_mapping(usage: NormalizedUsage) -> dict[str, int]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


async def _generate_ollama_with_structured_output_repair(
    chat_client: VertexChatClient,
    *,
    payload: "OpenAIChatRequest",
    messages: list[dict[str, Any]],
    provider_model: str,
    ctx: BudgetReservationContext,
) -> dict[str, Any]:
    started_at = time.monotonic()
    attempted_models = [payload.model]
    aggregate_usage = NormalizedUsage()
    try:
        result = await _generate_ollama_chat_once(
            chat_client,
            model=provider_model,
            messages=messages,
            payload=payload,
        )
        result_usage = _chat_usage_from_mapping(result.get("usage", {}))
        ctx.complete(result_usage)
        return result
    except VertexAPIError as exc:
        if not (
            exc.code == "invalid_schema_output"
            and _structured_output_repair_applies(
                payload=payload,
                provider="ollama",
                provider_model=provider_model,
            )
        ):
            raise
        last_failure_code = exc.code
        initial_usage = _usage_from_error(exc)
        aggregate_usage = _add_usage(aggregate_usage, initial_usage)
        ctx.complete_attempt(
            initial_usage,
            estimated_reason="structured_output_repair_initial_schema_failure",
        )

    repair_messages = _build_structured_output_repair_messages(
        messages=messages,
        response_format=payload.response_format,
        failure_code=last_failure_code,
    )
    for repair_model in _structured_output_repair_native_models():
        repair_model_id = f"{OLLAMA_DYNAMIC_MODEL_PREFIX}{repair_model}"
        attempted_models.append(repair_model_id)
        ctx.renew(
            model=repair_model_id,
            forecast_usage=_chat_forecast_usage(payload),
            provider="ollama",
        )
        try:
            result = await _generate_ollama_chat_once(
                chat_client,
                model=repair_model,
                messages=repair_messages,
                payload=payload,
                repair_attempt=True,
            )
            result_usage = _chat_usage_from_mapping(result.get("usage", {}))
            aggregate_usage = _add_usage(aggregate_usage, result_usage)
            ctx.complete_attempt(
                result_usage,
                estimated_reason="structured_output_repair_success_missing_usage",
            )
            result["usage"] = _usage_to_mapping(aggregate_usage)
            _log_structured_output_repair_event(
                attempted_models=attempted_models,
                final_status="success",
                failure_category=last_failure_code,
                content_chars=len(str(result.get("text") or "")),
                started_at=started_at,
            )
            return result
        except VertexAPIError as exc:
            if exc.code != "invalid_schema_output":
                ctx.release_attempt("upstream_error")
                raise VertexAPIError(
                    exc.status_code,
                    "Ollama structured output repair attempt failed.",
                    code="structured_output_repair_error",
                ) from exc
            last_failure_code = exc.code
            attempt_usage = _usage_from_error(exc)
            aggregate_usage = _add_usage(aggregate_usage, attempt_usage)
            ctx.complete_attempt(
                attempt_usage,
                estimated_reason="structured_output_repair_schema_failure",
            )

    _log_structured_output_repair_event(
        attempted_models=attempted_models,
        final_status="failure",
        failure_category=last_failure_code,
        content_chars=0,
        started_at=started_at,
    )
    raise VertexAPIError(
        502,
        "Ollama structured output repair failed after configured attempts.",
        code="invalid_schema_output",
    )


def _authorize_cost_admin(request: Request, authorization: str | None) -> JSONResponse | None:
    accounting = _cost_accounting(request)
    config = getattr(accounting, "config", None)
    if accounting is None or not getattr(accounting, "enabled", False) or config is None or not config.admin_enabled:
        return openai_error_response(
            message="Cost admin API is disabled.",
            status_code=404,
            error_type="invalid_request_error",
            code="cost_admin_disabled",
        )
    if not config.admin_api_key:
        return openai_error_response(
            message="Cost admin API is disabled.",
            status_code=404,
            error_type="invalid_request_error",
            code="cost_admin_disabled",
        )
    if authorization is None:
        return openai_error_response(
            message="Missing cost admin API key.",
            status_code=401,
            error_type="authentication_error",
            code="missing_admin_api_key",
        )
    if authorization != f"Bearer {config.admin_api_key}":
        return openai_error_response(
            message="Invalid cost admin API key.",
            status_code=403,
            error_type="permission_error",
            code="invalid_admin_api_key",
        )
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.junie_discovery = discovery_from_env()
    app.state.vertex_client = VertexEmbeddingClient()
    app.state.vertex_chat_client = VertexChatClient()
    app.state.vertex_rerank_client = VertexRerankClient()
    app.state.jetbrains_chat_client = JetBrainsChatClient()
    app.state.cost_accounting = build_async_cost_accounting(os.environ)
    _install_cost_clients(app)
    try:
        await asyncio.to_thread(_discover_compat_models)
    except Exception as exc:  # pragma: no cover - best effort
        logging.getLogger("openai_compatible_bridge").warning("compat model discovery failed: %s", exc)
    try:
        yield
    finally:
        cost_accounting = getattr(app.state, "cost_accounting", None)
        await _close_cost_accounting(cost_accounting)
        await app.state.vertex_client.close()
        await app.state.vertex_chat_client.close()
        await app.state.vertex_rerank_client.close()


app = FastAPI(title="AI Model Interop", version="0.1.0", lifespan=lifespan)
app.router.redirect_slashes = False
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    detail = exc.errors()
    msg = detail[0].get("msg", "Invalid request") if detail else "Invalid request"
    loc = detail[0].get("loc") if detail else None
    param = ".".join(str(p) for p in loc[1:]) if loc and len(loc) > 1 else None
    return openai_error_response(
        message=str(msg),
        status_code=400,
        error_type="invalid_request_error",
        code="invalid_request",
        param=param,
    )


@app.get("/healthz")
async def healthz() -> JSONResponse:
    return JSONResponse(content={"status": "ok"}, headers={"Cache-Control": "no-store"})


@app.get("/readyz")
async def readyz(request: Request) -> JSONResponse:
    accounting = _cost_accounting(request)
    if accounting is None:
        status = {"enabled": None, "database_available": False, "healthy": False}
    else:
        status = await _cost_query(accounting, "readiness")
    # 리미터 미들웨어가 없거나 metrics()가 실패해도 readyz는 cost_tracking만으로 판정한다.
    embedding_metrics = None
    limits = getattr(request.app.state, "embedding_limits", None)
    if limits is not None:
        try:
            embedding_metrics = limits.metrics()
        except Exception:
            embedding_metrics = None
    return JSONResponse(
        status_code=200 if status["healthy"] else 503,
        content={"status": "ok" if status["healthy"] else "unavailable", "cost_tracking": status,
                 "embeddings": embedding_metrics},
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


@app.get("/admin/cost/status", response_model=None)
async def admin_cost_status(
    request: Request,
    authorization: str | None = Header(default=None),
    provider: str | None = None,
) -> JSONResponse | dict[str, Any]:
    auth_error = _authorize_cost_admin(request, authorization)
    if auth_error is not None:
        return auth_error
    return await _cost_query(_cost_accounting(request), "admin_status", provider=provider)


@app.get("/admin/cost/events", response_model=None)
async def admin_cost_events(
    request: Request,
    authorization: str | None = Header(default=None),
    limit: int = 100,
) -> JSONResponse | dict[str, Any]:
    auth_error = _authorize_cost_admin(request, authorization)
    if auth_error is not None:
        return auth_error
    result = await _cost_query(_cost_accounting(request), "admin_events", limit=limit)
    return result if isinstance(result, JSONResponse) else {"data": result}


@app.get("/admin/cost/reconciliation", response_model=None)
async def admin_cost_reconciliation(
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse | dict[str, Any]:
    auth_error = _authorize_cost_admin(request, authorization)
    if auth_error is not None:
        return auth_error
    return await _cost_query(_cost_accounting(request), "admin_reconciliation")


def _model_object(model_id: str) -> dict[str, Any]:
    cfg = _main_model_config(model_id) or {}
    provider = cfg.get("provider", "vertex")
    context_length = cfg.get("context_length", DEFAULT_CONTEXT_LENGTH)
    try:
        context_length = int(context_length)
    except (TypeError, ValueError):
        context_length = DEFAULT_CONTEXT_LENGTH
    obj: dict[str, Any] = {
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "google",
        "context_length": max(1, context_length),
        "context_window": max(1, context_length),
        "capabilities": {
            "tools": bool(PROVIDER_TOOL_SUPPORT.get(provider, False)),
            "streaming": True,
            "images": False,
        },
    }
    display_name = cfg.get("display_name")
    if isinstance(display_name, str) and display_name.strip():
        obj["display_name"] = display_name.strip()
    tier = cfg.get("anthropic_family_tier")
    if isinstance(tier, str) and tier.strip():
        obj["anthropic_family_tier"] = tier.strip()
    family_default = cfg.get("is_family_default")
    if isinstance(family_default, bool):
        obj["is_family_default"] = family_default
    max_tokens = cfg.get("max_tokens")
    if max_tokens is not None:
        try:
            obj["max_tokens"] = max(1, int(max_tokens))
        except (TypeError, ValueError):
            pass
    runtime: dict[str, Any] = {"max_input_tokens": max(1, context_length)}
    if isinstance(obj.get("max_tokens"), int):
        runtime["max_output_tokens"] = obj["max_tokens"]
    obj["runtime"] = runtime
    return obj


def _model_in_list(model_id: str) -> bool:
    patterns = [p.strip() for p in _vertex_mod.MODEL_LIST_INCLUDE.split(",") if p.strip()]
    if not patterns:
        return True
    return any(fnmatch.fnmatchcase(model_id, pattern) for pattern in patterns)


def _select_chat_client(request: Request, provider: str) -> Any:
    if provider == "ollama":
        return request.app.state.ollama_chat_client
    if provider == "foundry":
        return request.app.state.foundry_chat_client
    if provider == "jetbrains":
        return request.app.state.jetbrains_chat_client
    if provider in {"openai_compat", "openai_compatible"}:
        return request.app.state.openai_compatible_chat_client
    return request.app.state.vertex_chat_client


@app.get("/v1/models", response_model=None)
async def list_models(request: Request, authorization: str | None = Header(default=None)) -> JSONResponse | dict[str, Any]:
    if BRIDGE_API_KEY and authorization != f"Bearer {BRIDGE_API_KEY}":
        return openai_error_response(
            message="Invalid wrapper API key.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )
    discovery = request.app.state.junie_discovery
    discovered = await discovery.refresh()
    registered = current_allowed_models()
    return JSONResponse(
        content={
            "object": "list",
            "data": [(_model_object(m) if m in registered else {"id": m, "object": "model", "created": 0, "owned_by": "jetbrains"})
                     for m in sorted(registered | discovered.keys()) if _model_in_list(m)],
        },
        headers={"Cache-Control": "no-store", "X-Junie-Discovery-Status":
                 "disabled" if not discovery.enabled else "stale" if discovery.stale else "fresh"},
    )


@app.get("/v1/models/{model_id:path}", response_model=None)
async def retrieve_model(
    model_id: str, request: Request, authorization: str | None = Header(default=None)
) -> JSONResponse | dict[str, Any]:
    if BRIDGE_API_KEY and authorization != f"Bearer {BRIDGE_API_KEY}":
        return openai_error_response(
            message="Invalid wrapper API key.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )
    discovered = await request.app.state.junie_discovery.refresh() if model_id.startswith(JETBRAINS_DYNAMIC_MODEL_PREFIX) else {}
    if model_id not in current_allowed_models() and model_id in discovered:
        return {"id": model_id, "object": "model", "created": 0, "owned_by": "jetbrains"}
    if model_id not in current_allowed_models():
        return openai_error_response(
            message=f"The model '{model_id}' does not exist.",
            status_code=404,
            error_type="invalid_request_error",
            code="model_not_found",
            param="model",
        )
    return _model_object(model_id)


@app.post("/v1/embeddings", response_model=None)
async def create_embeddings(
    payload: OpenAIEmbeddingsRequest,
    request: Request,
    authorization: str | None = Header(default=None),
    x_vertex_task_type: str | None = Header(default=None, alias="X-Vertex-Task-Type"),
    x_vertex_title: str | None = Header(default=None, alias="X-Vertex-Title"),
) -> JSONResponse | dict[str, Any]:
    if BRIDGE_API_KEY and authorization != f"Bearer {BRIDGE_API_KEY}":
        return openai_error_response(
            message="Invalid wrapper API key.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )

    allowed = current_allowed_models()
    if payload.model not in allowed:
        return openai_error_response(
            message=f"The model '{payload.model}' does not exist. Allowed: {sorted(allowed)}",
            status_code=404,
            error_type="invalid_request_error",
            code="model_not_found",
            param="model",
        )

    _embed_cfg = _main_model_config(payload.model)
    if _embed_cfg and _embed_cfg.get("kind") != "embedding":
        return openai_error_response(
            message=f"The model '{payload.model}' is not an embedding model.",
            status_code=400,
            error_type="invalid_request_error",
            code="invalid_model",
            param="model",
        )

    if payload.dimensions is not None and payload.dimensions < 1:
        return openai_error_response(
            message="dimensions must be >= 1.",
            status_code=400,
            error_type="invalid_request_error",
            code="invalid_dimensions",
            param="dimensions",
        )

    try:
        texts = coerce_string_inputs(payload.input)
    except ValueError as exc:
        return openai_error_response(
            message=str(exc),
            status_code=400,
            error_type="invalid_request_error",
            code="unsupported_input_shape",
            param="input",
        )
    if not texts:
        return openai_error_response(
            message="input must not be empty.",
            status_code=400,
            error_type="invalid_request_error",
            code="empty_input",
            param="input",
        )

    task_type = x_vertex_task_type or VERTEX_TASK_TYPE_DEFAULT
    if task_type not in SUPPORTED_TASK_TYPES:
        return openai_error_response(
            message=f"Unsupported task type: {task_type}",
            status_code=400,
            error_type="invalid_request_error",
            code="unsupported_task_type",
            param="X-Vertex-Task-Type",
        )

    accounting = _cost_accounting(request) or DisabledCostAccounting()
    try:
        async with accounting.reservation(
            endpoint="embeddings",
            model=payload.model,
            forecast_usage=_embedding_forecast_usage(texts),
            provider=(_embed_cfg or {}).get("provider", "vertex"),
        ) as ctx:
            vertex_client: VertexEmbeddingClient = request.app.state.vertex_client
            provider_model = (_embed_cfg or {}).get("provider_model", payload.model)

            chunk_results = await vertex_client.embed(
                model=provider_model,
                texts=texts,
                dimensions=payload.dimensions,
                task_type=task_type,
                title=x_vertex_title,
                resolved_config=_embed_cfg,
            )

            data: list[dict[str, Any]] = []
            total_tokens = 0
            for index, item in enumerate(chunk_results):
                values = item.get("values")
                if not isinstance(values, list):
                    ctx.complete_attempt(NormalizedUsage(), "malformed_upstream_response")
                    return openai_error_response(
                        message="Malformed Vertex AI response: embeddings.values missing.",
                        status_code=502,
                        error_type="api_error",
                        code="bad_gateway",
                    )
                try:
                    total_tokens += int(item.get("token_count", 0))
                except (TypeError, ValueError):
                    pass
                data.append(
                    {
                        "object": "embedding",
                        "index": index,
                        "embedding": encode_embedding(values, payload.encoding_format),
                    }
                )

            ctx.complete(NormalizedUsage(embedding_tokens=total_tokens, total_tokens=total_tokens))
            return {
                "object": "list",
                "data": data,
                "model": payload.model,
                "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
            }
    except (CostBudgetExceeded, CostConfigError, CostSubsystemUnhealthy) as exc:
        return _cost_tracking_error_response(exc)
    except VertexAPIError as exc:
        return openai_error_response(
            message=exc.message,
            status_code=exc.status_code,
            error_type=map_vertex_status_to_openai_type(exc.status_code),
            code=exc.code,
        )


def _new_chat_completion_id() -> str:
    """매 요청 고유한 OpenAI 호환 chat completion id를 생성한다."""
    return f"chatcmpl-{uuid.uuid4().hex[:16]}"


class CostManagedStreamingResponse(StreamingResponse):
    def __init__(self, content: Any, *, cost_context: BudgetReservationContext, **kwargs: Any) -> None:
        super().__init__(content, **kwargs)
        self._cost_context = cost_context

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._cost_context.finalize_interrupted_stream("client_disconnect")


def _chat_completions_stream(
    chat_client: VertexChatClient,
    payload: "OpenAIChatRequest",
    messages: list[dict[str, Any]],
    ctx: BudgetReservationContext,
    provider_model: str | None = None,
    resolved_config: dict[str, Any] | None = None,
    provider: str = "vertex",
) -> StreamingResponse:
    """stream=true 요청을 OpenAI 호환 SSE로 변환하는 StreamingResponse를 만든다."""
    completion_id = _new_chat_completion_id()
    ctx.is_stream = True
    include_usage = _should_include_stream_usage(payload)

    def _chunk(delta: dict[str, Any], finish_reason: str | None) -> str:
        obj = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": payload.model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    async def event_generator():
        first = True
        final_finish_reason: str | None = None
        accumulated_text_parts: list[str] = []
        try:
            async with ctx:
                stream_kwargs = {
                    "model": provider_model or payload.model,
                    "messages": messages,
                    "max_tokens": _chat_max_tokens(payload),
                    "temperature": payload.temperature,
                    "top_p": payload.top_p,
                    "stop": payload.stop,
                    "response_format": payload.response_format,
                }
                if payload.tools is not None:
                    stream_kwargs["tools"] = payload.tools
                if payload.tool_choice is not None:
                    stream_kwargs["tool_choice"] = payload.tool_choice
                if payload.parallel_tool_calls is not None:
                    stream_kwargs["parallel_tool_calls"] = payload.parallel_tool_calls
                if provider == "ollama":
                    stream_kwargs["reasoning_effort"] = payload.reasoning_effort
                    stream_kwargs["reasoning"] = payload.reasoning
                elif resolved_config is not None:
                    if provider == "foundry":
                        stream_kwargs["reasoning_effort"] = payload.reasoning_effort
                    elif provider in {"openai_compat", "openai_compatible"} and payload.reasoning_effort is not None:
                        stream_kwargs["reasoning_effort"] = payload.reasoning_effort
                    stream_kwargs["resolved_config"] = resolved_config
                async def events():
                    async for event in chat_client.stream_chat(**stream_kwargs):
                        yield event

                async for event in events():
                    ctx.stream_saw_event = True
                    delta_text = event.get("delta_text", "") or ""
                    if delta_text:
                        accumulated_text_parts.append(delta_text)
                    delta_tool_calls = event.get("delta_tool_calls")
                    fr = event.get("finish_reason")
                    if fr is not None:
                        final_finish_reason = fr
                    usage = event.get("usage")
                    if isinstance(usage, dict):
                        ctx.actual_usage = _chat_usage_from_mapping(usage)

                    delta: dict[str, Any] = {}
                    if first:
                        delta["role"] = "assistant"
                        first = False
                    if delta_text:
                        delta["content"] = delta_text
                    if delta_tool_calls:
                        delta["tool_calls"] = delta_tool_calls

                    if delta:
                        yield _chunk(delta, None)
        except (CostBudgetExceeded, CostConfigError, CostSubsystemUnhealthy) as exc:
            error = _cost_tracking_error_response(exc)
            yield f"data: {error.body.decode()}\n\n"
            yield "data: [DONE]\n\n"
            return
        except VertexAPIError as exc:
            err_obj = {
                "error": {
                    "message": exc.message,
                    "type": map_vertex_status_to_openai_type(exc.status_code),
                    "param": None,
                    "code": exc.code,
                }
            }
            yield f"data: {json.dumps(err_obj, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        if first:
            yield _chunk({"role": "assistant"}, None)

        # 종료 청크: finish_reason 담기 (없으면 stop으로 폴백).
        yield _chunk({}, final_finish_reason or "stop")

        if include_usage:
            usage_dict: dict[str, int]
            if ctx.actual_usage is not None:
                usage_dict = {
                    "prompt_tokens": ctx.actual_usage.prompt_tokens,
                    "completion_tokens": ctx.actual_usage.completion_tokens,
                    "total_tokens": ctx.actual_usage.total_tokens,
                }
            else:
                prompt_tokens = sum(_estimate_text_tokens(m.get("content")) for m in messages)
                completion_tokens = _estimate_text_tokens("".join(accumulated_text_parts))
                usage_dict = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
            usage_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": 0,
                "model": payload.model,
                "choices": [],
                "usage": usage_dict,
            }
            yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"

    return CostManagedStreamingResponse(
        event_generator(),
        cost_context=ctx,
        media_type="text/event-stream",
    )


def _chat_completions_stream_buffered_structured_repair(
    chat_client: VertexChatClient,
    payload: "OpenAIChatRequest",
    messages: list[dict[str, Any]],
    ctx: BudgetReservationContext,
    *,
    provider_model: str,
) -> StreamingResponse:
    completion_id = _new_chat_completion_id()
    include_usage = _should_include_stream_usage(payload)

    def _chunk(delta: dict[str, Any], finish_reason: str | None) -> str:
        obj = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": 0,
            "model": payload.model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    async def event_generator():
        try:
            async with ctx:
                result = await _generate_ollama_with_structured_output_repair(
                    chat_client,
                    payload=payload,
                    messages=messages,
                    provider_model=provider_model,
                    ctx=ctx,
                )
                yield _chunk({"role": "assistant"}, None)
                yield _chunk({"content": result["text"]}, None)
                yield _chunk({}, result.get("finish_reason") or "stop")
                if include_usage:
                    usage_data = result.get("usage")
                    if isinstance(usage_data, dict):
                        usage_dict = {
                            "prompt_tokens": int(usage_data.get("prompt_tokens", 0) or 0),
                            "completion_tokens": int(usage_data.get("completion_tokens", 0) or 0),
                            "total_tokens": int(usage_data.get("total_tokens", 0) or 0),
                        }
                    else:
                        usage_dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                    usage_chunk = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": payload.model,
                        "choices": [],
                        "usage": usage_dict,
                    }
                    yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
        except (CostBudgetExceeded, CostConfigError, CostSubsystemUnhealthy) as exc:
            error = _cost_tracking_error_response(exc)
            yield f"data: {error.body.decode()}\n\n"
            yield "data: [DONE]\n\n"
            return
        except VertexAPIError as exc:
            err_obj = {
                "error": {
                    "message": exc.message,
                    "type": map_vertex_status_to_openai_type(exc.status_code),
                    "param": None,
                    "code": exc.code,
                }
            }
            yield f"data: {json.dumps(err_obj, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

    return CostManagedStreamingResponse(
        event_generator(),
        cost_context=ctx,
        media_type="text/event-stream",
    )


@app.post("/v1/chat/completions", response_model=None)
async def create_chat_completions(
    payload: OpenAIChatRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse | dict[str, Any]:
    if BRIDGE_API_KEY and authorization != f"Bearer {BRIDGE_API_KEY}":
        return openai_error_response(
            message="Invalid wrapper API key.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )

    discovered = await request.app.state.junie_discovery.refresh() if payload.model.startswith(JETBRAINS_DYNAMIC_MODEL_PREFIX) else {}
    _chat_cfg, model_error = _resolve_chat_model(payload.model, discovered)
    if model_error is not None:
        return model_error

    # response_format.type 유효성 검사: 지원하지 않는 type은 400으로 거부한다.
    # 허용 타입은 vertex.SUPPORTED_RESPONSE_FORMAT_TYPES 단일 출처를 공유한다(드리프트 방지).
    if payload.response_format is not None:
        rf_type = payload.response_format.get("type")
        if rf_type not in SUPPORTED_RESPONSE_FORMAT_TYPES:
            return openai_error_response(
                message=f"Unsupported response_format.type: {rf_type!r}. "
                        f"Allowed: {sorted(SUPPORTED_RESPONSE_FORMAT_TYPES)}",
                status_code=400,
                error_type="invalid_request_error",
                code="invalid_request",
                param="response_format",
            )

    provider = (_chat_cfg or {}).get("provider", "vertex")
    if provider == "jetbrains" and (payload.reasoning_effort is not None or payload.reasoning is not None):
        param = "reasoning_effort" if payload.reasoning_effort is not None else "reasoning"
        return openai_error_response(
            message=f"{param} is not supported for JetBrains.", status_code=400,
            error_type="invalid_request_error", code="unsupported_parameter", param=param,
        )
    if provider == "jetbrains":
        try:
            validate_jetbrains_parameters(
                (_chat_cfg or {}).get("protocol") or "openai_responses",
                parallel_tool_calls=payload.parallel_tool_calls,
                response_format=payload.response_format,
            )
        except VertexAPIError as exc:
            return openai_error_response(message=exc.message, status_code=exc.status_code,
                                         error_type=map_vertex_status_to_openai_type(exc.status_code), code=exc.code)
    if provider == "openai_compatible":
        provider = "openai_compat"
    provider_model = (_chat_cfg or {}).get("provider_model", payload.model)
    chat_client = _select_chat_client(request, provider)

    messages = [_normalize_chat_message(m) for m in payload.messages]

    accounting = _cost_accounting(request) or DisabledCostAccounting()

    # 스트리밍 경로: StreamingResponse가 반환된 뒤 event_generator가 소비되므로
    # outer async with 가 generator 실행 전에 닫혀버린다.
    # 따라서 ctx를 직접 생성해 event_generator 내부 `async with ctx:` 가 라이프사이클을 단독 관리한다.
    if payload.stream:
        ctx = accounting.reservation(
            endpoint="chat",
            model=payload.model,
            forecast_usage=_chat_forecast_usage(payload),
            provider=provider,
        )
        preflight_error = _preflight_reservation_now(ctx)
        if preflight_error is not None:
            return preflight_error
        if _structured_output_repair_applies(
            payload=payload,
            provider=provider,
            provider_model=provider_model,
        ):
            return _chat_completions_stream_buffered_structured_repair(
                chat_client,
                payload,
                messages,
                ctx,
                provider_model=provider_model,
            )
        return _chat_completions_stream(
            chat_client,
            payload,
            messages,
            ctx,
            provider_model=provider_model,
            resolved_config=_chat_cfg if provider in {"vertex", "foundry", "openai_compat", "openai_compatible", "jetbrains"} else None,
            provider=provider,
        )

    # 비스트리밍 경로: async with 가 요청 완료 전에 닫히므로 정상 동작.
    try:
        async with accounting.reservation(
            endpoint="chat",
            model=payload.model,
            forecast_usage=_chat_forecast_usage(payload),
            provider=provider,
        ) as ctx:
            generate_kwargs = {
                "model": provider_model,
                "messages": messages,
                "max_tokens": _chat_max_tokens(payload),
                "temperature": payload.temperature,
                "top_p": payload.top_p,
                "stop": payload.stop,
                "response_format": payload.response_format,
            }
            if payload.tools is not None:
                generate_kwargs["tools"] = payload.tools
            if payload.tool_choice is not None:
                generate_kwargs["tool_choice"] = payload.tool_choice
            if payload.parallel_tool_calls is not None:
                generate_kwargs["parallel_tool_calls"] = payload.parallel_tool_calls
            if provider == "foundry":
                generate_kwargs["reasoning_effort"] = payload.reasoning_effort
            elif provider in {"openai_compat", "openai_compatible"} and payload.reasoning_effort is not None:
                generate_kwargs["reasoning_effort"] = payload.reasoning_effort
            if provider in {"vertex", "foundry", "openai_compat", "openai_compatible", "jetbrains"}:
                generate_kwargs["resolved_config"] = _chat_cfg
                result = await chat_client.generate(**generate_kwargs)
                result_usage = _chat_usage_from_mapping(result.get("usage", {}))
                ctx.complete(result_usage)
            elif provider == "ollama":
                result = await _generate_ollama_with_structured_output_repair(
                    chat_client,
                    payload=payload,
                    messages=messages,
                    provider_model=provider_model,
                    ctx=ctx,
                )
            else:
                result = await chat_client.generate(**generate_kwargs)
                result_usage = _chat_usage_from_mapping(result.get("usage", {}))
                ctx.complete(result_usage)

            choice_message: dict[str, Any] = {"role": "assistant"}
            if result.get("text") is not None:
                choice_message["content"] = result["text"]
            elif "tool_calls" in result and result["tool_calls"]:
                choice_message["content"] = None
            else:
                choice_message["content"] = ""
            if "tool_calls" in result and result["tool_calls"]:
                choice_message["tool_calls"] = result["tool_calls"]

            return {
                "id": _new_chat_completion_id(),
                "object": "chat.completion",
                "created": 0,
                "model": payload.model,
                "choices": [
                    {
                        "index": 0,
                        "message": choice_message,
                        "finish_reason": result["finish_reason"],
                    }
                ],
                "usage": result.get("usage", {}),
            }
    except (CostBudgetExceeded, CostConfigError, CostSubsystemUnhealthy) as exc:
        return _cost_tracking_error_response(exc)
    except VertexAPIError as exc:
        return openai_error_response(
            message=exc.message,
            status_code=exc.status_code,
            error_type=map_vertex_status_to_openai_type(exc.status_code),
            code=exc.code,
        )


def _anthropic_forecast_usage(messages: list[dict[str, Any]], max_tokens: int) -> NormalizedUsage:
    prompt_tokens = sum(_estimate_text_tokens(m.get("content")) for m in messages)
    return NormalizedUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=max_tokens,
        total_tokens=prompt_tokens + max_tokens,
    )


def _anthropic_generate_kwargs(
    *,
    params: dict[str, Any],
    provider_model: str,
    resolved_config: dict[str, Any] | None,
    provider: str,
) -> dict[str, Any]:
    generate_kwargs: dict[str, Any] = {
        "model": provider_model,
        "messages": params["messages"],
        "max_tokens": params["max_tokens"],
        "temperature": params["temperature"],
        "top_p": params["top_p"],
        "stop": params["stop"],
    }
    if params["tools"] is not None:
        generate_kwargs["tools"] = params["tools"]
    if params["tool_choice"] is not None:
        generate_kwargs["tool_choice"] = params["tool_choice"]
    if provider in {"openai_compat", "openai_compatible"} and params["reasoning_effort"] is not None:
        generate_kwargs["reasoning_effort"] = params["reasoning_effort"]
    if provider == "foundry":
        generate_kwargs["reasoning_effort"] = None
    if provider in {"vertex", "foundry", "openai_compat", "openai_compatible", "jetbrains"}:
        generate_kwargs["resolved_config"] = resolved_config
    return generate_kwargs


def _anthropic_stream(
    chat_client: Any,
    params: dict[str, Any],
    ctx: BudgetReservationContext,
    provider_model: str,
    resolved_config: dict[str, Any] | None,
    provider: str,
) -> StreamingResponse:
    """Anthropic SSE로 변환하는 StreamingResponse를 만든다."""
    message_id = new_anthropic_message_id()
    ctx.is_stream = True

    async def event_generator():
        state = AnthropicStreamState(message_id=message_id, model=params["model"])
        accumulated_text_parts: list[str] = []
        try:
            async with ctx:
                stream_kwargs = _anthropic_generate_kwargs(
                    params=params,
                    provider_model=provider_model,
                    resolved_config=resolved_config,
                    provider=provider,
                )
                if provider == "ollama":
                    stream_kwargs.pop("tools", None)
                    stream_kwargs.pop("tool_choice", None)
                    stream_kwargs["reasoning_effort"] = None
                    stream_kwargs["reasoning"] = None
                elif provider == "foundry":
                    stream_kwargs["reasoning_effort"] = None
                async for event in chat_client.stream_chat(**stream_kwargs):
                    ctx.stream_saw_event = True
                    delta_text = event.get("delta_text", "") or ""
                    if delta_text:
                        accumulated_text_parts.append(delta_text)
                    usage = event.get("usage")
                    if isinstance(usage, dict):
                        ctx.actual_usage = _chat_usage_from_mapping(usage)
                    for chunk in state.feed(event):
                        yield chunk
        except (CostBudgetExceeded, CostConfigError, CostSubsystemUnhealthy) as exc:
            error = _cost_tracking_error_response(exc)
            try:
                payload_err = json.loads(error.body.decode())
                err_obj = payload_err.get("error", {})
            except (ValueError, TypeError):
                err_obj = {"message": str(exc), "type": "api_error"}
            yield sse_event(
                {
                    "type": "error",
                    "error": {
                        "type": "rate_limit_error" if isinstance(exc, CostBudgetExceeded) else "api_error",
                        "message": err_obj.get("message", str(exc)),
                    },
                }
            )
            return
        except VertexAPIError as exc:
            yield sse_event(
                {
                    "type": "error",
                    "error": {
                        "type": map_status_to_anthropic_error_type(exc.status_code),
                        "message": exc.message,
                    },
                }
            )
            return

        prompt_tokens = sum(_estimate_text_tokens(m.get("content")) for m in params["messages"])
        for chunk in state.finish(
            prompt_tokens=prompt_tokens, completion_text="".join(accumulated_text_parts)
        ):
            yield chunk

    return CostManagedStreamingResponse(
        event_generator(),
        cost_context=ctx,
        media_type="text/event-stream",
    )


@app.post("/v1/messages", response_model=None)
async def create_anthropic_messages(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> JSONResponse | dict[str, Any]:
    """Anthropic Messages ingress. Claude Code / Claude Desktop용 ingress이며,
    내부는 OpenAI-shape 파이프라인(registry 라우팅 + cost gate)을 그대로 탄다."""
    credential = None
    if authorization and authorization.startswith("Bearer "):
        credential = authorization[len("Bearer "):]
    elif x_api_key:
        credential = x_api_key
    if BRIDGE_API_KEY and credential != BRIDGE_API_KEY:
        return anthropic_error_response(
            message="Invalid wrapper API key.",
            status_code=401,
            error_type="authentication_error",
        )

    try:
        raw_payload = await request.json()
    except Exception:
        return anthropic_error_response(
            message="Request body must be valid JSON.",
            status_code=400,
            error_type="invalid_request_error",
        )
    try:
        params = parse_anthropic_chat_params(raw_payload)
    except AnthropicBadRequest as exc:
        return anthropic_error_response(
            message=exc.message, status_code=400, error_type="invalid_request_error"
        )

    cfg: dict[str, Any] | None
    provider_model: str
    if params["model"].startswith(OLLAMA_DYNAMIC_MODEL_PREFIX):
        native = params["model"][len(OLLAMA_DYNAMIC_MODEL_PREFIX):].strip()
        if not native:
            return anthropic_error_response(
                message="Dynamic Ollama model must use the form 'ollama:<native-model>'.",
                status_code=400,
                error_type="invalid_request_error",
            )
        cfg = {"provider": "ollama", "kind": "chat", "provider_model": native}
        provider_model = native
    elif params["model"].startswith(OPENAI_COMPAT_DYNAMIC_MODEL_PREFIX):
        native = params["model"][len(OPENAI_COMPAT_DYNAMIC_MODEL_PREFIX):].strip()
        if not native:
            return anthropic_error_response(
                message="Dynamic compat model must use the form 'compat:<upstream-model>'.",
                status_code=400,
                error_type="invalid_request_error",
            )
        base_cfg = _openai_compatible_base_config()
        if not base_cfg.get("base_url"):
            return anthropic_error_response(
                message="OPENAI_COMPAT_BASE_URL (or compat:* base_url) is not configured for dynamic compat models.",
                status_code=400,
                error_type="invalid_request_error",
            )
        cfg = dict(base_cfg)
        cfg["provider"] = "openai_compatible"
        cfg["kind"] = "chat"
        cfg["provider_model"] = native
        provider_model = native
    elif params["model"] not in _vertex_allowed_models() and params["model"] in (discovered := await request.app.state.junie_discovery.refresh() if params["model"].startswith(JETBRAINS_DYNAMIC_MODEL_PREFIX) else {}):
        cfg = discovered[params["model"]]
        if cfg.get("protocol") is None:
            return anthropic_error_response(
                message="The discovered JetBrains model has an unsupported protocol; configure an explicit registry entry.",
                status_code=400, error_type="invalid_request_error",
            )
        provider_model = cfg["provider_model"]
    elif params["model"] not in current_allowed_models() and params["model"].startswith(JETBRAINS_DYNAMIC_MODEL_PREFIX):
        native = params["model"][len(JETBRAINS_DYNAMIC_MODEL_PREFIX):].strip()
        if not native:
            return anthropic_error_response(
                message="Dynamic JetBrains model must use the form 'jetbrains:<native-model>'.",
                status_code=400,
                error_type="invalid_request_error",
            )
        cfg = {
            "provider": "jetbrains",
            "kind": "chat",
            "provider_model": native,
        }
        provider_model = native
    elif params["model"] not in current_allowed_models():
        return anthropic_error_response(
            message=f"The model '{params['model']}' does not exist.",
            status_code=404,
            error_type="not_found_error",
        )
    else:
        cfg = _main_model_config(params["model"])
        if cfg and cfg.get("kind") != "chat":
            return anthropic_error_response(
                message=f"The model '{params['model']}' is not a chat model.",
                status_code=400,
                error_type="invalid_request_error",
            )
        provider_model = (cfg or {}).get("provider_model", params["model"])

    provider = (cfg or {}).get("provider", "vertex")
    if provider == "jetbrains" and params["reasoning_effort"] is not None:
        return anthropic_error_response(
            message="output_config.effort is not supported for JetBrains.",
            status_code=400, error_type="invalid_request_error",
        )
    if provider == "openai_compatible":
        provider = "openai_compat"
    chat_client = _select_chat_client(request, provider)

    if params["tools"] is not None and not PROVIDER_TOOL_SUPPORT.get(provider, False):
        return anthropic_error_response(
            message=f"Tools are not supported for provider '{provider}'.",
            status_code=400,
            error_type="invalid_request_error",
        )

    accounting = _cost_accounting(request) or DisabledCostAccounting()

    if params["stream"]:
        ctx = accounting.reservation(
            endpoint="chat",
            model=params["model"],
            forecast_usage=_anthropic_forecast_usage(params["messages"], params["max_tokens"]),
            provider=provider,
        )
        preflight_error = _preflight_reservation_now(ctx)
        if preflight_error is not None:
            try:
                payload_err = json.loads(preflight_error.body.decode())
                err_obj = payload_err.get("error", {})
            except (ValueError, TypeError):
                err_obj = {}
            if preflight_error.status_code == 429:
                return anthropic_error_response(
                    message=err_obj.get("message", "Cost budget exceeded."),
                    status_code=429,
                    error_type="rate_limit_error",
                )
            return anthropic_error_response(
                message=err_obj.get("message", "Cost tracking unavailable."),
                status_code=503,
                error_type="api_error",
            )
        return _anthropic_stream(
            chat_client, params, ctx, provider_model, cfg if provider in {"vertex", "foundry", "openai_compat", "openai_compatible", "jetbrains"} else None, provider
        )

    try:
        async with accounting.reservation(
            endpoint="chat",
            model=params["model"],
            forecast_usage=_anthropic_forecast_usage(params["messages"], params["max_tokens"]),
            provider=provider,
        ) as ctx:
            if provider == "ollama":
                result = await chat_client.generate(
                    model=provider_model,
                    messages=params["messages"],
                    max_tokens=params["max_tokens"],
                    temperature=params["temperature"],
                    top_p=params["top_p"],
                    stop=params["stop"],
                )
                ctx.complete(_chat_usage_from_mapping(result.get("usage", {})))
            else:
                generate_kwargs = _anthropic_generate_kwargs(
                    params=params,
                    provider_model=provider_model,
                    resolved_config=cfg,
                    provider=provider,
                )
                result = await chat_client.generate(**generate_kwargs)
                ctx.complete(_chat_usage_from_mapping(result.get("usage", {})))

            return openai_result_to_anthropic(
                result, model=params["model"], message_id=new_anthropic_message_id()
            )
    except (CostBudgetExceeded, CostConfigError, CostSubsystemUnhealthy) as exc:
        if isinstance(exc, CostBudgetExceeded):
            return anthropic_error_response(
                message="Cost budget exceeded.", status_code=429, error_type="rate_limit_error"
            )
        return anthropic_error_response(
            message=str(exc), status_code=503, error_type="api_error"
        )
    except VertexAPIError as exc:
        return anthropic_error_response(
            message=exc.message,
            status_code=exc.status_code,
            error_type=map_status_to_anthropic_error_type(exc.status_code),
        )


@app.post("/v1/rerank", response_model=None)
async def rerank(
    payload: CohereRerankRequest,
    request: Request,
    authorization: str | None = Header(default=None),
) -> JSONResponse | dict[str, Any]:
    if BRIDGE_API_KEY and authorization != f"Bearer {BRIDGE_API_KEY}":
        return openai_error_response(
            message="Invalid wrapper API key.",
            status_code=401,
            error_type="authentication_error",
            code="invalid_api_key",
        )

    if payload.model not in current_allowed_models():
        return openai_error_response(
            message=f"The model '{payload.model}' does not exist.",
            status_code=404,
            error_type="invalid_request_error",
            code="model_not_found",
            param="model",
        )

    _rank_cfg = _main_model_config(payload.model)
    if _rank_cfg and _rank_cfg.get("kind") != "rerank":
        return openai_error_response(
            message=f"The model '{payload.model}' is not a rerank model.",
            status_code=400,
            error_type="invalid_request_error",
            code="invalid_model",
            param="model",
        )

    if not payload.documents:
        return openai_error_response(
            message="documents must not be empty.",
            status_code=400,
            error_type="invalid_request_error",
            code="empty_documents",
            param="documents",
        )

    rerank_client: VertexRerankClient = request.app.state.vertex_rerank_client

    records: list[dict[str, Any]] = []
    for i, doc in enumerate(payload.documents):
        content = doc.get("text", "") if isinstance(doc, dict) else str(doc)
        records.append({"id": str(i), "content": content})

    accounting = _cost_accounting(request) or DisabledCostAccounting()
    try:
        async with accounting.reservation(
            endpoint="rerank",
            model=payload.model,
            forecast_usage=NormalizedUsage(rerank_units=1),
            provider=(_rank_cfg or {}).get("provider", "vertex"),
        ) as ctx:
            try:
                records_out = await rerank_client.rank(
                    model=(_rank_cfg or {}).get("provider_model", payload.model),
                    query=payload.query,
                    records=records,
                    top_n=payload.top_n,
                    ignore_record_details_in_response=True,
                    location=_rank_cfg.get("location", "global") if _rank_cfg else "global",
                )
            except VertexAPIError as exc:
                raise exc

            results = []
            for r in records_out:
                try:
                    idx = int(r.get("id", "0"))
                except ValueError:
                    idx = 0
                results.append({
                    "index": idx,
                    "relevance_score": float(r.get("score", 0.0))
                })

            ctx.complete(NormalizedUsage(rerank_units=1))
            return {"results": results}
    except (CostBudgetExceeded, CostConfigError, CostSubsystemUnhealthy) as exc:
        return _cost_tracking_error_response(exc)
    except VertexAPIError as exc:
        return openai_error_response(
            message=exc.message,
            status_code=exc.status_code,
            error_type=map_vertex_status_to_openai_type(exc.status_code),
            code=exc.code,
        )


def _lifespan_with_factories(
    *,
    embedding_client_factory: Any,
    chat_client_factory: Any,
    rerank_client_factory: Any,
    ollama_chat_client_factory: Any,
    foundry_chat_client_factory: Any,
    openai_compatible_chat_client_factory: Any | None = None,
    jetbrains_chat_client_factory: Any | None = None,
    cost_accounting_factory: Any,
) -> Any:
    @asynccontextmanager
    async def managed_lifespan(app: FastAPI):
        app.state.junie_discovery = discovery_from_env()
        app.state.vertex_client = embedding_client_factory()
        app.state.vertex_chat_client = chat_client_factory()
        app.state.vertex_rerank_client = rerank_client_factory()
        app.state.ollama_chat_client = ollama_chat_client_factory()
        app.state.foundry_chat_client = foundry_chat_client_factory()
        app.state.openai_compatible_chat_client = (
            openai_compatible_chat_client_factory()
            if openai_compatible_chat_client_factory is not None
            else OpenAICompatibleChatClient()
        )
        app.state.jetbrains_chat_client = (
            jetbrains_chat_client_factory()
            if jetbrains_chat_client_factory is not None
            else JetBrainsChatClient()
        )
        app.state.cost_accounting = cost_accounting_factory()
        _install_cost_clients(app)
        try:
            await asyncio.to_thread(_discover_compat_models)
        except Exception as exc:  # pragma: no cover - best effort
            logging.getLogger("openai_compatible_bridge").warning("compat model discovery failed: %s", exc)
        try:
            yield
        finally:
            cost_accounting = getattr(app.state, "cost_accounting", None)
            await _close_cost_accounting(cost_accounting)
            await app.state.vertex_client.close()
            await app.state.vertex_chat_client.close()
            await app.state.vertex_rerank_client.close()
            await app.state.ollama_chat_client.close()
            await app.state.foundry_chat_client.close()
            await app.state.openai_compatible_chat_client.close()
            await app.state.jetbrains_chat_client.close()

    return managed_lifespan


_ROUTE_SOURCE_APP = app


def create_app(
    *,
    embedding_client_factory: Any | None = None,
    chat_client_factory: Any | None = None,
    rerank_client_factory: Any | None = None,
    ollama_chat_client_factory: Any | None = None,
    foundry_chat_client_factory: Any | None = None,
    openai_compatible_chat_client_factory: Any | None = None,
    jetbrains_chat_client_factory: Any | None = None,
    cost_accounting_factory: Any | None = None,
) -> FastAPI:
    embedding_factory = embedding_client_factory or (lambda: VertexEmbeddingClient())
    chat_factory = chat_client_factory or (lambda: VertexChatClient())
    rerank_factory = rerank_client_factory or (lambda: VertexRerankClient())
    ollama_factory = ollama_chat_client_factory or (lambda: OllamaChatClient())
    foundry_factory = foundry_chat_client_factory or (lambda: FoundryChatClient())
    openai_compatible_factory = openai_compatible_chat_client_factory
    jetbrains_factory = jetbrains_chat_client_factory
    cost_factory = cost_accounting_factory or (lambda: build_async_cost_accounting(os.environ))
    bridge_app = FastAPI(
        title="AI Model Interop",
        version="0.1.0",
        lifespan=_lifespan_with_factories(
            embedding_client_factory=embedding_factory,
            chat_client_factory=chat_factory,
            rerank_client_factory=rerank_factory,
            ollama_chat_client_factory=ollama_factory,
            foundry_chat_client_factory=foundry_factory,
            openai_compatible_chat_client_factory=openai_compatible_factory,
            jetbrains_chat_client_factory=jetbrains_factory,
            cost_accounting_factory=cost_factory,
        ),
    )
    bridge_app.router.redirect_slashes = False
    from .embedding_limits import EmbeddingRequestLimits
    bridge_app.add_middleware(EmbeddingRequestLimits, state=bridge_app.state)
    bridge_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    bridge_app.add_middleware(BaseHTTPMiddleware, dispatch=_registry_hot_reload_dispatch)
    bridge_app.add_exception_handler(RequestValidationError, validation_exception_handler)
    for route in _ROUTE_SOURCE_APP.router.routes:
        if isinstance(route, APIRoute):
            bridge_app.router.routes.append(route)
    return bridge_app


app = create_app()
