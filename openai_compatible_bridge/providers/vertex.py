from __future__ import annotations

import asyncio
import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, Iterable

import google.auth
import httpx
from google.auth.transport.requests import Request as GoogleAuthRequest

from openai_compatible_bridge.core.cost_tracking import NormalizedUsage
from openai_compatible_bridge.core.embedding_resilience import EmbeddingRetryPolicy, embedding_max_inflight
from openai_compatible_bridge.core.provider_adapter import (
    map_http_error_for_provider,
    parse_vertex_usage,
)

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
VERTEX_PROJECT = os.getenv("VERTEX_PROJECT")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
TOKEN_REFRESH_SKEW_SECONDS = int(os.getenv("TOKEN_REFRESH_SKEW_SECONDS", "300"))
HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "60"))
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "8"))
VERTEX_AUTO_TRUNCATE = os.getenv("VERTEX_AUTO_TRUNCATE", "true").lower() in {"1", "true", "yes", "on"}
DEFAULT_MAX_INSTANCES = int(os.getenv("DEFAULT_MAX_INSTANCES", "1"))

# ---------------------------------------------------------------------------
# Model registry (config-driven)
# ---------------------------------------------------------------------------
_REGISTRY_RELOAD_LOCK = threading.RLock()
MODEL_REGISTRY_RELOAD_HOOK = None
_registry_mtime = 0.0

_SUPPORTED_APIS = {"predict", "embedContent", "generateContent", "openapiChatCompletions", "rank"}
_SUPPORTED_PROVIDERS = {"vertex", "ollama", "foundry", "openai_compat", "openai_compatible", "jetbrains"}
_SUPPORTED_FOUNDRY_PROTOCOLS = {
    "openai_chat_completions",
    "anthropic_messages",
    "xai_responses",
    "openai_responses",
    "google_generate_content",
}

# kind 기본값 결정: api별 default kind
_API_DEFAULT_KIND: dict[str, str] = {
    "predict": "embedding",
    "embedContent": "embedding",
    "generateContent": "chat",
    "openapiChatCompletions": "chat",
    "rank": "rerank",
}

_BUILTIN_REGISTRY: dict[str, dict[str, Any]] = {
    "text-embedding-005": {"api": "predict", "max_instances": 5},
    "text-multilingual-embedding-002": {"api": "predict", "max_instances": 5},
    "gemini-embedding-001": {"api": "predict", "max_instances": 1},
    "gemini-embedding-2": {"api": "embedContent", "location": "global", "max_instances": 1},
    "semantic-ranker-512@latest": {"api": "rank", "location": "global"},
    "semantic-ranker-default-004": {"api": "rank", "location": "global"},
    "semantic-ranker-fast-004": {"api": "rank", "location": "global"},
    # Chat models
    # thinking 모델은 thinking_budget=0으로 사고 토큰을 끈다 -> 작은 max_tokens에도 본문이 비지 않음
    # (client verify가 작은 max_tokens로 호출해 빈 응답->실패하던 문제 방지).
    "gemini-2.5-flash": {"api": "generateContent", "kind": "chat", "location": "us-central1", "thinking_budget": 0},
    "gemini-2.5-flash-thinking": {"api": "generateContent", "kind": "chat", "location": "us-central1", "vertex_model": "gemini-2.5-flash"},
    "gemini-2.5-pro": {"api": "generateContent", "kind": "chat", "location": "us-central1"},
    "gemini-3.5-flash": {"api": "generateContent", "kind": "chat", "location": "global", "thinking_budget": 0},
    "gemini-3.5-flash-thinking": {"api": "generateContent", "kind": "chat", "location": "global", "vertex_model": "gemini-3.5-flash"},
    "gemma-4-26b-a4b-it-maas": {
        "api": "openapiChatCompletions",
        "kind": "chat",
        "location": "global",
        "openapi_model": "google/gemma-4-26b-a4b-it-maas",
    },
}

def _build_registry() -> dict[str, dict[str, Any]]:
    """환경변수를 반영한 최종 모델 레지스트리를 빌드한다."""
    registry: dict[str, dict[str, Any]] = {k: dict(v) for k, v in _BUILTIN_REGISTRY.items()}

    # MODEL_REGISTRY_JSON: 기본값 위에 merge
    # 파일 기반(MODEL_REGISTRY_FILE)이 설정되면 그 내용이 우선한다.
    registry_json_str = ""
    model_registry_file = os.getenv("MODEL_REGISTRY_FILE", "").strip()
    if model_registry_file:
        try:
            with open(model_registry_file, "r", encoding="utf-8") as fh:
                registry_json_str = fh.read().strip()
        except OSError:
            registry_json_str = os.getenv("MODEL_REGISTRY_JSON", "").strip()
    else:
        registry_json_str = os.getenv("MODEL_REGISTRY_JSON", "").strip()
    if registry_json_str:
        # invalid JSON이면 loudly fail
        override = json.loads(registry_json_str)
        if not isinstance(override, dict):
            raise ValueError("MODEL_REGISTRY_JSON must be a JSON object")
        for model_id, cfg in override.items():
            if model_id == "_include":
                global MODEL_LIST_INCLUDE
                MODEL_LIST_INCLUDE = str(cfg).strip()
                continue
            if model_id in registry:
                registry[model_id] = {**registry[model_id], **cfg}
            else:
                registry[model_id] = dict(cfg)

    # EXTRA_MODELS: comma-separated backward compat
    extra_models_str = os.getenv("EXTRA_MODELS", "").strip()
    if extra_models_str:
        for name in (m.strip() for m in extra_models_str.split(",") if m.strip()):
            if name not in registry:
                registry[name] = {"api": "predict", "max_instances": DEFAULT_MAX_INSTANCES}

    # provider/api 검증. 기존 api 기반 Vertex entries는 계속 허용하고,
    # 새 provider 기반 entries는 provider별 계약으로 검증한다.
    for model_id, cfg in registry.items():
        provider = cfg.get("provider")
        api = cfg.get("api")
        if provider is None:
            provider = "vertex"
            cfg["provider"] = provider
        if provider not in _SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Model '{model_id}' has invalid provider={provider!r}; "
                f"must be one of {sorted(_SUPPORTED_PROVIDERS)}"
            )
        if provider == "vertex" and api not in _SUPPORTED_APIS:
            raise ValueError(
                f"Model '{model_id}' has invalid api={api!r}; "
                f"must be one of {sorted(_SUPPORTED_APIS)}"
            )
        if provider == "ollama" and cfg.get("kind", "chat") != "chat":
            raise ValueError("Ollama provider currently supports only kind='chat'")
        if provider in {"openai_compat", "openai_compatible"}:
            if cfg.get("kind", "chat") != "chat":
                raise ValueError("openai_compat provider currently supports only kind='chat'")
            if not isinstance(cfg.get("base_url"), str) or not cfg["base_url"].strip():
                raise ValueError(f"Model '{model_id}' with provider='openai_compatible' requires base_url.")
            api_key = cfg.get("api_key")
            if api_key is not None and not isinstance(api_key, str):
                raise ValueError(f"Model '{model_id}' api_key must be a string.")
            api_key_env = cfg.get("api_key_env")
            if api_key_env is not None and (not isinstance(api_key_env, str) or not api_key_env.strip()):
                raise ValueError(f"Model '{model_id}' api_key_env must be a non-empty string.")
        if provider == "foundry" and cfg.get("kind", "chat") != "chat":
            raise ValueError("Foundry provider currently supports only kind='chat'")
        if provider == "foundry":
            protocol = cfg.setdefault("protocol", "openai_chat_completions")
            if protocol not in _SUPPORTED_FOUNDRY_PROTOCOLS:
                raise ValueError(
                    f"Model '{model_id}' has invalid Foundry protocol={protocol!r}; "
                    f"must be one of {sorted(_SUPPORTED_FOUNDRY_PROTOCOLS)}"
                )

    return registry


MODEL_REGISTRY: dict[str, dict[str, Any]] = _build_registry()

MODEL_LIST_INCLUDE: str = os.getenv("MODEL_LIST_INCLUDE", "").strip()


def _reload_registry_if_changed() -> bool:
    """파일 기반 레지스트리가 바뀌었으면 in-place 갱신한다 (재시작 불필요)."""
    path = os.getenv("MODEL_REGISTRY_FILE", "").strip()
    if not path:
        return False
    try:
        st = os.stat(path)
    except OSError:
        return False
    with _REGISTRY_RELOAD_LOCK:
        global _registry_mtime
        if st.st_mtime == _registry_mtime and st.st_mtime != 0:
            return False
        try:
            new_registry = _build_registry()
        except (ValueError, json.JSONDecodeError, OSError):
            return False
        MODEL_REGISTRY.clear()
        MODEL_REGISTRY.update(new_registry)
        _registry_mtime = st.st_mtime
        # _include 키가 있으면 필터도 갱신
        try:
            raw = open(path, "r", encoding="utf-8").read().strip()
            parsed = json.loads(raw) if raw else {}
            if isinstance(parsed, dict) and "_include" in parsed:
                global MODEL_LIST_INCLUDE
                MODEL_LIST_INCLUDE = str(parsed["_include"]).strip()
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    hook = MODEL_REGISTRY_RELOAD_HOOK
    if callable(hook):
        try:
            hook()
        except Exception:
            pass
    return True

# Backward-compat: 기존 코드가 KNOWN_MAX_INSTANCES를 직접 참조하는 경우를 위해 유지
KNOWN_MAX_INSTANCES: dict[str, int] = {
    k: v["max_instances"] for k, v in MODEL_REGISTRY.items() if v.get("api") == "predict"
}


def model_config(model: str) -> dict[str, Any] | None:
    """모델의 resolved config dict를 반환한다.

    반환 dict는 최소 api, kind, location, max_instances 키를 포함한다.
    - location: 엔트리에 명시된 경우 그 값, embedContent면 "global", predict면 VERTEX_LOCATION.
    - kind: 엔트리에 명시된 경우 그 값, 없으면 api에 따라 결정 (_API_DEFAULT_KIND).
    """
    entry = MODEL_REGISTRY.get(model)
    if entry is None:
        return None
    cfg = dict(entry)
    provider = cfg.get("provider", "vertex")
    cfg["provider"] = provider
    if provider in {"ollama", "foundry", "openai_compat", "openai_compatible", "jetbrains"}:
        if "kind" not in cfg:
            cfg["kind"] = "chat"
        if "provider_model" not in cfg:
            cfg["provider_model"] = model
        return cfg
    if "location" not in cfg:
        if cfg.get("api") in ("embedContent", "rank"):
            cfg["location"] = "global"
        else:
            cfg["location"] = VERTEX_LOCATION
    if "max_instances" not in cfg:
        cfg["max_instances"] = DEFAULT_MAX_INSTANCES
    if "kind" not in cfg:
        cfg["kind"] = _API_DEFAULT_KIND.get(cfg.get("api", ""), "embedding")
    # provider_model/vertex_model: provider-native model id. Vertex legacy aliases may
    # still use vertex_model, while provider-generic aliases can use provider_model.
    if "vertex_model" not in cfg:
        cfg["vertex_model"] = cfg.get("provider_model", model)
    if "provider_model" not in cfg:
        cfg["provider_model"] = cfg["vertex_model"]
    if cfg.get("api") == "openapiChatCompletions" and "openapi_model" not in cfg:
        cfg["openapi_model"] = cfg["vertex_model"]
    return cfg


def allowed_models() -> set[str]:
    """레지스트리에 등록된 모든 모델 id를 반환한다."""
    return set(MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class VertexAPIError(Exception):
    def __init__(self, status_code: int, message: str, code: str | None = None, raw: Any = None,
                 stage: str | None = None, upstream_status: int | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.code = code
        self.raw = raw
        self.stage = stage
        self.upstream_status = upstream_status


def _parse_vertex_error(resp: httpx.Response) -> VertexAPIError:
    """Vertex AI HTTP 에러 응답(4xx/5xx)을 VertexAPIError로 변환한다.

    predict / embedContent 등 어댑터가 공유하는 에러 파싱 로직.
    """
    try:
        payload = resp.json()
    except Exception:
        payload = {"error": {"message": resp.text}}
    err = payload.get("error", {}) if isinstance(payload, dict) else {}
    message = err.get("message") or resp.text or "Vertex AI request failed"
    code = err.get("status") or err.get("code") or str(resp.status_code)
    return VertexAPIError(resp.status_code, message=message, code=str(code), raw=payload)


def _vertex_host(location: str) -> str:
    """Vertex AI 호스트. location이 'global'이면 region prefix가 없다."""
    if location == "global":
        return "aiplatform.googleapis.com"
    return f"{location}-aiplatform.googleapis.com"


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    size = max(1, size)
    for i in range(0, len(items), size):
        yield items[i : i + size]

class GoogleAccessTokenProvider:
    """서비스 계정(ADC) 기반 access token을 만료 전 선갱신하며 캐시한다."""
    def __init__(self) -> None:
        creds, detected_project = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
        self._creds = creds
        self.project_id = VERTEX_PROJECT or detected_project
        if not self.project_id:
            raise RuntimeError(
                "Google Cloud project를 결정할 수 없습니다. VERTEX_PROJECT 환경변수를 설정하거나 ADC project를 구성하세요."
            )
        self._lock = threading.Lock()
        self._request = GoogleAuthRequest()

    def _valid_with_skew(self) -> bool:
        token = getattr(self._creds, "token", None)
        expiry = getattr(self._creds, "expiry", None)
        if not token:
            return False
        if expiry is None:
            return bool(self._creds.valid)
        remaining = (expiry - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds()
        return bool(self._creds.valid) and remaining > TOKEN_REFRESH_SKEW_SECONDS

    def _get_token_sync(self) -> str:
        with self._lock:
            if not self._valid_with_skew():
                self._creds.refresh(self._request)
            token = getattr(self._creds, "token", None)
            if not token:
                raise RuntimeError("Google access token 발급에 실패했습니다.")
            return token

    async def get_token(self) -> str:
        return await asyncio.to_thread(self._get_token_sync)


# ---------------------------------------------------------------------------
# Vertex AI HTTP client
# ---------------------------------------------------------------------------

class VertexEmbeddingClient:
    provider_name = "vertex"

    def parse_usage(self, payload: Any, endpoint: str) -> NormalizedUsage | None:
        return parse_vertex_usage(payload, endpoint)

    def map_http_error(self, response: httpx.Response) -> VertexAPIError:
        return map_http_error_for_provider(self.provider_name, response)

    def __init__(self, token_provider: GoogleAccessTokenProvider | None = None) -> None:
        self.max_inflight = embedding_max_inflight()
        self.retry_policy = EmbeddingRetryPolicy.from_env()
        self.token_provider = token_provider or GoogleAccessTokenProvider()
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS))
        self.semaphore = asyncio.Semaphore(self.max_inflight)

    async def close(self) -> None:
        await self.http.aclose()

    async def _gather_embeddings(self, operations: Iterable[Any]) -> list[Any]:
        tasks = [asyncio.create_task(operation) for operation in operations]
        try:
            return await asyncio.gather(*tasks)
        except (Exception, asyncio.CancelledError):
            # 일부 배치가 실패하면 다른 배치의 backoff·후속 전송도 종료한다.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    def _predict_url(self, model: str, location: str) -> str:
        project = self.token_provider.project_id
        return (
            f"https://{_vertex_host(location)}/"
            f"v1/projects/{project}/locations/{location}/"
            f"publishers/google/models/{model}:predict"
        )

    def _embed_content_url(self, model: str, location: str) -> str:
        project = self.token_provider.project_id
        return (
            f"https://{_vertex_host(location)}/"
            f"v1/projects/{project}/locations/{location}/"
            f"publishers/google/models/{model}:embedContent"
        )

    async def _predict(
        self,
        *,
        model: str,
        texts: list[str],
        dimensions: int | None,
        task_type: str,
        title: str | None,
        auto_truncate: bool,
        location: str,
    ) -> list[dict[str, Any]]:
        """predict API를 호출하고 예측 목록을 반환한다."""
        async with self.semaphore:
            token = await self.token_provider.get_token()
        url = self._predict_url(model, location)

        instances: list[dict[str, Any]] = []
        for text in texts:
            item: dict[str, Any] = {"content": text}
            if task_type and task_type != "UNSPECIFIED":
                item["task_type"] = task_type
            if title:
                item["title"] = title
            instances.append(item)

        parameters: dict[str, Any] = {"autoTruncate": auto_truncate}
        if dimensions is not None:
            parameters["outputDimensionality"] = dimensions

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        try:
            resp = await self.retry_policy.post(
                self.http, self.semaphore, url,
                headers=headers, json={"instances": instances, "parameters": parameters},
            )
        except httpx.TimeoutException as exc:
            raise VertexAPIError(504, f"Vertex AI request timed out: {exc}", code="timeout") from exc
        except httpx.RequestError as exc:
            raise VertexAPIError(502, f"Vertex AI connection error: {exc}", code="connection_error") from exc

        if resp.status_code >= 400:
            raise _parse_vertex_error(resp)

        try:
            data = resp.json()
        except Exception as exc:
            raise VertexAPIError(502, f"Invalid JSON from Vertex AI: {exc}", code="bad_gateway") from exc

        predictions = data.get("predictions")
        if not isinstance(predictions, list):
            raise VertexAPIError(502, "Malformed Vertex AI response: missing predictions[]", code="bad_gateway")
        if len(predictions) != len(texts):
            raise VertexAPIError(
                502,
                f"Malformed predict response: prediction count mismatch "
                f"(expected {len(texts)}, got {len(predictions)})",
                code="bad_gateway",
            )
        return predictions

    async def _embed_content_single(
        self,
        *,
        model: str,
        text: str,
        dimensions: int | None,
        task_type: str,
        location: str,
    ) -> dict[str, Any]:
        """embedContent API를 1개 텍스트에 대해 호출하고 raw 응답을 반환한다."""
        async with self.semaphore:
            token = await self.token_provider.get_token()
        url = self._embed_content_url(model, location)

        body: dict[str, Any] = {
            "content": {"parts": [{"text": text}]},
        }
        if dimensions is not None:
            body["outputDimensionality"] = dimensions
        if task_type and task_type != "UNSPECIFIED":
            body["taskType"] = task_type

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        try:
            resp = await self.retry_policy.post(self.http, self.semaphore, url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise VertexAPIError(504, f"Vertex AI request timed out: {exc}", code="timeout") from exc
        except httpx.RequestError as exc:
            raise VertexAPIError(502, f"Vertex AI connection error: {exc}", code="connection_error") from exc

        if resp.status_code >= 400:
            raise _parse_vertex_error(resp)

        try:
            data = resp.json()
        except Exception as exc:
            raise VertexAPIError(502, f"Invalid JSON from Vertex AI: {exc}", code="bad_gateway") from exc

        return data

    async def embed(
        self,
        *,
        model: str,
        texts: list[str],
        dimensions: int | None,
        task_type: str,
        title: str | None,
        resolved_config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """모든 텍스트에 대한 embedding을 반환한다.

        Returns:
            list[dict] in input order, each dict = {"values": list[float], "token_count": int}
        """
        cfg = resolved_config or model_config(model) or {
            "api": "predict",
            "location": VERTEX_LOCATION,
            "max_instances": DEFAULT_MAX_INSTANCES,
            "vertex_model": model,
        }
        api = cfg.get("api", "predict")
        location = cfg.get("location", VERTEX_LOCATION)
        batch_size = cfg.get("max_instances", DEFAULT_MAX_INSTANCES)
        vertex_model = cfg.get("vertex_model") or cfg.get("provider_model") or model

        if api == "embedContent":
            return await self._embed_all_embed_content(
                model=vertex_model,
                texts=texts,
                dimensions=dimensions,
                task_type=task_type,
                location=location,
            )
        else:
            return await self._embed_all_predict(
                model=vertex_model,
                texts=texts,
                dimensions=dimensions,
                task_type=task_type,
                title=title,
                batch_size=batch_size,
                location=location,
            )

    async def _embed_all_predict(
        self,
        *,
        model: str,
        texts: list[str],
        dimensions: int | None,
        task_type: str,
        title: str | None,
        batch_size: int,
        location: str,
    ) -> list[dict[str, Any]]:
        """predict API를 사용하여 배치 처리 후 flat list로 반환."""
        async def one_chunk(chunk: list[str]) -> list[dict[str, Any]]:
            return await self._predict(
                model=model,
                texts=chunk,
                dimensions=dimensions,
                task_type=task_type,
                title=title,
                auto_truncate=VERTEX_AUTO_TRUNCATE,
                location=location,
            )

        chunk_results = await self._gather_embeddings(one_chunk(chunk) for chunk in chunked(texts, batch_size))

        # flat list로 변환
        results: list[dict[str, Any]] = []
        for predictions in chunk_results:
            for pred in predictions:
                emb = pred.get("embeddings", {}) if isinstance(pred, dict) else {}
                values = emb.get("values", []) if isinstance(emb, dict) else []
                stats = emb.get("statistics", {}) if isinstance(emb, dict) else {}
                try:
                    token_count = int(stats.get("token_count", 0))
                except (TypeError, ValueError):
                    token_count = 0
                results.append({"values": values, "token_count": token_count})
        return results

    # ------------------------------------------------------------------
    # Private helpers for embed_content path
    # ------------------------------------------------------------------

    async def _embed_all_embed_content(
        self,
        *,
        model: str,
        texts: list[str],
        dimensions: int | None,
        task_type: str,
        location: str,
    ) -> list[dict[str, Any]]:
        """embedContent API를 사용하여 1개씩 호출 후 flat list로 반환. 입력 순서 보장."""
        async def one_text(text: str) -> dict[str, Any]:
            return await self._embed_content_single(
                model=model,
                text=text,
                dimensions=dimensions,
                task_type=task_type,
                location=location,
            )

        raw_results = await self._gather_embeddings(one_text(text) for text in texts)

        results: list[dict[str, Any]] = []
        for data in raw_results:
            embedding = data.get("embedding", {})
            values = embedding.get("values", []) if isinstance(embedding, dict) else []
            if not values:
                raise VertexAPIError(
                    502,
                    "Malformed embedContent response: embedding.values missing",
                    code="bad_gateway",
                )
            usage = data.get("usageMetadata", {}) or {}
            # usageMetadata 키는 다양할 수 있어 방어적으로 파싱
            token_count = 0
            if isinstance(usage, dict):
                for key in ("tokenCount", "token_count", "inputTokens", "input_tokens"):
                    val = usage.get(key)
                    if val is not None:
                        try:
                            token_count = int(val)
                        except (TypeError, ValueError):
                            token_count = 0
                        break
            results.append({"values": values, "token_count": token_count})
        return results


# ---------------------------------------------------------------------------
# Vertex AI Chat client (generateContent)
# ---------------------------------------------------------------------------

_RESPONSE_FORMAT_SANITIZE_KEYS = frozenset({"$schema", "$id", "$ref", "definitions", "$defs"})

# response_format.type 허용값 단일 출처. app 계층 검증과 _apply_response_format 분기가 이를 공유한다.
SUPPORTED_RESPONSE_FORMAT_TYPES: frozenset[str] = frozenset({"text", "json_object", "json_schema"})


def _sanitize_schema(schema: Any) -> Any:
    """JSON Schema에서 Vertex 미지원 메타 키를 제거하고 $ref/$defs를 inline 해소한 새 구조를 반환한다.

    원본을 변경하지 않고 새 dict/list를 반환한다.
    $ref는 루트 $defs/definitions에서 참조 대상을 찾아 inline한다.
    제거 대상: $schema, $id, definitions, $defs
    보존: type, properties, required, items, enum, description, propertyOrdering, additionalProperties 등
    """
    defs: dict[str, Any] = {}
    if isinstance(schema, dict):
        for key in ("$defs", "definitions"):
            section = schema.get(key)
            if isinstance(section, dict):
                defs.update(section)
    return _sanitize_schema_node(schema, defs, frozenset())


def _sanitize_schema_node(node: Any, defs: dict[str, Any], seen: frozenset[str]) -> Any:
    if isinstance(node, dict):
        ref = node.get("$ref")
        if isinstance(ref, str):
            target_name = ref.rsplit("/", 1)[-1]
            target = defs.get(target_name)
            if target is not None and target_name not in seen:
                siblings = {k: v for k, v in node.items() if k != "$ref"}
                return _sanitize_schema_node({**target, **siblings}, defs, seen | {target_name})
            return {
                k: _sanitize_schema_node(v, defs, seen)
                for k, v in node.items()
                if k not in _RESPONSE_FORMAT_SANITIZE_KEYS
            }
        return {
            k: _sanitize_schema_node(v, defs, seen)
            for k, v in node.items()
            if k not in _RESPONSE_FORMAT_SANITIZE_KEYS
        }
    if isinstance(node, list):
        return [_sanitize_schema_node(item, defs, seen) for item in node]
    return node


def _apply_response_format(gen_cfg: dict[str, Any], response_format: dict[str, Any] | None) -> None:
    """response_format 매핑을 gen_cfg dict에 적용한다 (in-place).

    - type=="json_object": responseMimeType = "application/json"
    - type=="json_schema": responseMimeType = "application/json" + responseJsonSchema = sanitized schema
    - type=="text" 또는 None: 변경 없음
    """
    if response_format is None:
        return
    fmt_type = response_format.get("type")
    if fmt_type == "text" or fmt_type is None:
        return
    if fmt_type == "json_object":
        gen_cfg["responseMimeType"] = "application/json"
    elif fmt_type == "json_schema":
        gen_cfg["responseMimeType"] = "application/json"
        json_schema_obj = response_format.get("json_schema") or {}
        raw_schema = json_schema_obj.get("schema") if isinstance(json_schema_obj, dict) else None
        # 빈/누락 스키마({} 또는 None)면 responseJsonSchema 없이 mimeType만 설정 -> Vertex가 자유 JSON 출력.
        if raw_schema is not None and raw_schema != {}:
            gen_cfg["responseJsonSchema"] = _sanitize_schema(raw_schema)
    # 그 외 type은 app 계층에서 이미 400으로 검증/차단된다. 여기서는 방어적으로 no-op.


def _map_finish_reason(vertex_reason: str) -> str:
    """Vertex finishReason -> OpenAI finish_reason 매핑."""
    return {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
    }.get(vertex_reason, "stop")


def _extract_message_text(content: Any) -> str:
    """OpenAI 메시지 content(str 또는 parts list)에서 텍스트를 추출한다."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content) if content is not None else ""


def _coerce_openai_usage(usage: Any) -> dict[str, Any]:
    """OpenAI Chat Completions usage 객체를 wrapper usage dict로 정규화한다.

    캐시 읽기/쓰기 토큰은 제공업체가 실제 필드를 반환한 경우에만 보존한다.
    없는 필드를 0으로 채우지 않으며, 이 함수는 캐시 적중을 추정하지 않는다.
    """
    usage = usage if isinstance(usage, dict) else {}

    def parse_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        if parsed < 0:
            return None
        return parsed

    def int_value(key: str) -> int:
        parsed = parse_int(usage.get(key, 0))
        return 0 if parsed is None else parsed

    prompt = int_value("prompt_tokens")
    completion = int_value("completion_tokens")
    total = int_value("total_tokens") or prompt + completion
    normalized: dict[str, Any] = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }
    for detail_key in ("prompt_tokens_details", "completion_tokens_details"):
        raw_detail = usage.get(detail_key)
        if not isinstance(raw_detail, dict):
            continue
        detail = {
            key: parsed
            for key, value in raw_detail.items()
            if isinstance(key, str) and (parsed := parse_int(value)) is not None
        }
        if detail:
            normalized[detail_key] = detail
    for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
        if key not in usage:
            continue
        parsed = parse_int(usage.get(key))
        if parsed is not None:
            normalized[key] = parsed
    return normalized


def _parse_stream_chunk(chunk: dict[str, Any]) -> tuple[str, str | None, dict[str, Any] | None]:
    """Vertex streamGenerateContent SSE chunk를 (delta_text, finish_reason, usage)로 파싱한다.

    - delta_text: candidates[0].content.parts[*].text 연결 (없으면 "").
    - finish_reason: candidates[0].finishReason가 있으면 OpenAI 매핑값, 없으면 None.
    - usage: usageMetadata가 있으면 usage dict, 없으면 None.
    """
    delta_text = ""
    finish_reason: str | None = None

    candidates = chunk.get("candidates", []) if isinstance(chunk, dict) else []
    if candidates:
        candidate = candidates[0] or {}
        content_obj = candidate.get("content", {}) or {}
        parts_list = content_obj.get("parts", []) or []
        delta_text = "".join(
            p.get("text", "") for p in parts_list if isinstance(p, dict)
        )
        raw_finish = candidate.get("finishReason")
        if raw_finish:
            finish_reason = _map_finish_reason(raw_finish)

    usage: dict[str, Any] | None = None
    usage_meta = chunk.get("usageMetadata") if isinstance(chunk, dict) else None
    if isinstance(usage_meta, dict) and usage_meta:
        usage = {
            "prompt_tokens": int(usage_meta.get("promptTokenCount", 0) or 0),
            "completion_tokens": int(usage_meta.get("candidatesTokenCount", 0) or 0),
            "total_tokens": int(usage_meta.get("totalTokenCount", 0) or 0),
        }

    return delta_text, finish_reason, usage


class VertexChatClient:
    """Vertex AI generateContent API를 사용하는 채팅 클라이언트."""

    provider_name = "vertex"

    def parse_usage(self, payload: Any, endpoint: str) -> NormalizedUsage | None:
        return parse_vertex_usage(payload, endpoint)

    def map_http_error(self, response: httpx.Response) -> VertexAPIError:
        return map_http_error_for_provider(self.provider_name, response)

    def __init__(self, token_provider: GoogleAccessTokenProvider | None = None) -> None:
        self.token_provider = token_provider or GoogleAccessTokenProvider()
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS))
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def close(self) -> None:
        await self.http.aclose()

    def _generate_content_url(self, model: str, location: str) -> str:
        project = self.token_provider.project_id
        return (
            f"https://{_vertex_host(location)}/"
            f"v1/projects/{project}/locations/{location}/"
            f"publishers/google/models/{model}:generateContent"
        )

    def _stream_generate_content_url(self, model: str, location: str) -> str:
        project = self.token_provider.project_id
        return (
            f"https://{_vertex_host(location)}/"
            f"v1/projects/{project}/locations/{location}/"
            f"publishers/google/models/{model}:streamGenerateContent?alt=sse"
        )

    def _openapi_chat_completions_url(self, location: str) -> str:
        project = self.token_provider.project_id
        return (
            f"https://{_vertex_host(location)}/"
            f"v1/projects/{project}/locations/{location}/"
            f"endpoints/openapi/chat/completions"
        )

    async def _generate_openapi_chat_completions(
        self,
        *,
        cfg: dict[str, Any],
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Agent Platform OpenAI Chat Completions endpoint를 호출한다."""
        location = cfg.get("location", VERTEX_LOCATION)
        openapi_model = cfg.get("openapi_model") or cfg.get("vertex_model") or model

        token = await self.token_provider.get_token()
        url = self._openapi_chat_completions_url(location)

        body: dict[str, Any] = {
            "model": openapi_model,
            "messages": messages,
            "stream": False,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if stop is not None:
            body["stop"] = stop
        if response_format is not None:
            body["response_format"] = response_format
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        try:
            async with self.semaphore:
                resp = await self.http.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise VertexAPIError(504, f"Vertex AI request timed out: {exc}", code="timeout") from exc
        except httpx.RequestError as exc:
            raise VertexAPIError(502, f"Vertex AI connection error: {exc}", code="connection_error") from exc

        if resp.status_code >= 400:
            raise _parse_vertex_error(resp)

        try:
            data = resp.json()
        except Exception as exc:
            raise VertexAPIError(502, f"Invalid JSON from Vertex AI: {exc}", code="bad_gateway") from exc

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise VertexAPIError(502, "Malformed Vertex AI response: missing choices[]", code="bad_gateway")

        choice = choices[0] or {}
        message = choice.get("message", {}) or {}
        content = message.get("content", "")
        text_out = _extract_message_text(content)

        choice = choices[0] or {}
        message = choice.get("message", {}) or {}
        content = message.get("content", "")
        text_out = _extract_message_text(content)
        tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else None
        finish_reason = choice.get("finish_reason") or ("tool_calls" if tool_calls else "stop")

        result: dict[str, Any] = {
            "text": text_out if text_out or not tool_calls else None,
            "finish_reason": finish_reason,
            "usage": _coerce_openai_usage(data.get("usage") if isinstance(data, dict) else None),
        }
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    @staticmethod
    def _openai_tools_to_function_declarations(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        """OpenAI tools -> Vertex functionDeclarations. v1: pass-through + 최소 검증."""
        if tools is None:
            return None
        declarations: list[dict[str, Any]] = []
        for tool in tools:
            fn = tool.get("function", {}) if isinstance(tool, dict) else {}
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                raise VertexAPIError(400, "Tool function requires a name.", code="invalid_request")
            parameters = fn.get("parameters", {})
            if not isinstance(parameters, dict):
                raise VertexAPIError(400, "Tool function parameters must be an object.", code="invalid_request")
            declarations.append(
                {
                    "name": name,
                    "description": str(fn.get("description", "")),
                    "parameters": parameters,
                }
            )
        return declarations or None

    @staticmethod
    def _openai_tool_choice_to_config(
        tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if tool_choice is None or tool_choice == "auto":
            return None
        if tool_choice == "required" or (isinstance(tool_choice, dict) and not tool_choice.get("type")):
            return {"functionCallingConfig": {"mode": "ANY"}}
        if isinstance(tool_choice, str):
            if tool_choice == "none":
                return {"functionCallingConfig": {"mode": "NONE"}}
            raise VertexAPIError(400, f"Unsupported tool_choice: {tool_choice!r}.", code="invalid_request")
        choice_type = tool_choice.get("type")
        if choice_type == "function":
            name = (tool_choice.get("function") or {}).get("name")
            if not name:
                raise VertexAPIError(400, "tool_choice function requires a name.", code="invalid_request")
            return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [str(name)]}}
        if choice_type == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
        raise VertexAPIError(400, f"Unsupported tool_choice: {tool_choice!r}.", code="invalid_request")

    @staticmethod
    def _build_request_body(
        *,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        stop: str | list[str] | None,
        thinking_budget: int | None = None,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """OpenAI messages/params -> Vertex generateContent request body.

        비스트림(generate)과 스트림(stream_chat)이 공유하는 매핑 로직.
        assistant tool_calls -> functionCall parts, role=tool 메시지 ->
        functionResponse parts로 매핑한다 (functionResponse name은 같은 턴의
        tool_calls id에서 역조회, 없으면 tool_call_id를 name으로 사용).
        """
        # --- 메시지 매핑 ---
        system_parts: list[dict[str, Any]] = []
        contents: list[dict[str, Any]] = []

        call_id_to_name: dict[str, str] = {}
        for msg in messages:
            for call in msg.get("tool_calls") or []:
                if isinstance(call, dict) and call.get("id"):
                    fn = call.get("function", {}) if isinstance(call.get("function"), dict) else {}
                    if fn.get("name"):
                        call_id_to_name[str(call["id"])] = str(fn["name"])

        for msg in messages:
            role = msg.get("role", "user")
            raw_content = msg.get("content", "")
            text = _extract_message_text(raw_content)

            if role == "system":
                system_parts.append({"text": text})
            elif role == "assistant":
                parts: list[dict[str, Any]] = []
                if text:
                    parts.append({"text": text})
                for call in msg.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    fn = call.get("function", {}) if isinstance(call.get("function"), dict) else {}
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except (TypeError, ValueError):
                        args = {}
                    if not isinstance(args, dict):
                        args = {}
                    parts.append({"functionCall": {"name": str(fn.get("name", "")), "args": args}})
                contents.append({"role": "model", "parts": parts or [{"text": ""}]})
            elif role == "tool":
                tool_call_id = str(msg.get("tool_call_id") or msg.get("id") or "")
                name = call_id_to_name.get(tool_call_id, tool_call_id)
                try:
                    response_obj = json.loads(raw_content) if isinstance(raw_content, str) and raw_content.strip().startswith(("{", "[")) else raw_content
                except (TypeError, ValueError):
                    response_obj = raw_content
                if not isinstance(response_obj, dict):
                    response_obj = {"content": response_obj}
                contents.append(
                    {"role": "user", "parts": [{"functionResponse": {"name": name, "response": response_obj}}]}
                )
            else:
                contents.append({"role": "user", "parts": [{"text": text}]})

        body: dict[str, Any] = {"contents": contents}

        if system_parts:
            body["systemInstruction"] = {"parts": system_parts}

        declarations = VertexChatClient._openai_tools_to_function_declarations(tools)
        if declarations is not None:
            body["tools"] = [{"functionDeclarations": declarations}]
        tool_config = VertexChatClient._openai_tool_choice_to_config(tool_choice)
        if tool_config is not None:
            body["toolConfig"] = tool_config

        # --- generationConfig 매핑 ---
        gen_cfg: dict[str, Any] = {}
        if max_tokens is not None:
            gen_cfg["maxOutputTokens"] = max_tokens
        if temperature is not None:
            gen_cfg["temperature"] = temperature
        if top_p is not None:
            gen_cfg["topP"] = top_p
        if stop is not None:
            gen_cfg["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)
        if thinking_budget is not None:
            gen_cfg["thinkingConfig"] = {"thinkingBudget": thinking_budget}

        # --- response_format 매핑 (responseMimeType / responseJsonSchema) ---
        _apply_response_format(gen_cfg, response_format)

        if gen_cfg:
            body["generationConfig"] = gen_cfg

        return body

    @staticmethod
    def _parse_candidate_parts(
        parts_list: list[Any], *, used_call_ids: set[str] | None = None,
    ) -> tuple[str | None, list[dict[str, Any]] | None, str | None]:
        """generateContent parts -> (text, tool_calls, raw_finish) with synthesized ids."""
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        used = set(used_call_ids or ())
        for part_index, part in enumerate(parts_list):
            if not isinstance(part, dict) or part.get("thought") is True:
                continue
            text = part.get("text")
            if text is not None:
                text_parts.append(str(text))
            function_call = part.get("functionCall")
            if isinstance(function_call, dict):
                name = str(function_call.get("name", ""))
                args = function_call.get("args", {})
                if not isinstance(args, dict):
                    args = {}
                call_id = f"call_{name}_{part_index}"
                suffix = 1
                base_id = call_id
                while call_id in used:
                    call_id = f"{base_id}_{suffix}"
                    suffix += 1
                used.add(call_id)
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                    }
                )
        if tool_calls:
            return "".join(text_parts) or None, tool_calls, "tool_calls"
        return "".join(text_parts), None, None

    async def generate(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        resolved_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """generateContent를 호출하고 정규화된 결과를 반환한다.

        Returns:
            {"text": str | None, "tool_calls": [...] | None, "finish_reason": str, "usage": {...}}
        """
        cfg = resolved_config or model_config(model) or {"location": VERTEX_LOCATION, "vertex_model": model}
        if cfg.get("api") == "openapiChatCompletions":
            return await self._generate_openapi_chat_completions(
                cfg=cfg,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
            )

        location = cfg.get("location", VERTEX_LOCATION)
        vertex_model = cfg.get("vertex_model", model)

        token = await self.token_provider.get_token()
        url = self._generate_content_url(vertex_model, location)

        body = self._build_request_body(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            thinking_budget=cfg.get("thinking_budget"),
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
        )

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        try:
            async with self.semaphore:
                resp = await self.http.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise VertexAPIError(504, f"Vertex AI request timed out: {exc}", code="timeout") from exc
        except httpx.RequestError as exc:
            raise VertexAPIError(502, f"Vertex AI connection error: {exc}", code="connection_error") from exc

        if resp.status_code >= 400:
            raise _parse_vertex_error(resp)

        try:
            data = resp.json()
        except Exception as exc:
            raise VertexAPIError(502, f"Invalid JSON from Vertex AI: {exc}", code="bad_gateway") from exc

        # --- 응답 파싱 ---
        candidates = data.get("candidates", [])
        if not candidates:
            raise VertexAPIError(502, "Malformed Vertex AI response: no candidates", code="bad_gateway")

        candidate = candidates[0]
        content_obj = candidate.get("content", {}) or {}
        parts_list = content_obj.get("parts", []) or []
        used_call_ids = {
            str(call["id"]) for message in messages
            for call in message.get("tool_calls") or []
            if isinstance(call, dict) and call.get("id")
        }
        text_out, tool_calls, tool_finish = self._parse_candidate_parts(
            parts_list, used_call_ids=used_call_ids,
        )

        finish_reason = tool_finish or _map_finish_reason(candidate.get("finishReason", "STOP"))

        usage_meta = data.get("usageMetadata", {}) or {}
        prompt_tokens = int(usage_meta.get("promptTokenCount", 0) or 0)
        completion_tokens = int(usage_meta.get("candidatesTokenCount", 0) or 0)
        total_tokens = int(usage_meta.get("totalTokenCount", 0) or 0)

        result: dict[str, Any] = {
            "text": text_out,
            "finish_reason": finish_reason,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    async def stream_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: str | list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        resolved_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        **_: Any,
    ):
        """streamGenerateContent(SSE)를 호출하고 델타를 순차 yield하는 async generator.

        각 yield는 dict:
            {"delta_text": str, "delta_tool_calls": [...] | None, "finish_reason": str | None, "usage": dict | None}
        - delta_text: 이 chunk의 델타 텍스트 (없으면 "").
        - finish_reason: 마지막(혹은 finishReason이 있는) chunk에서 OpenAI 매핑값, 아니면 None.
        - usage: usageMetadata가 있는 chunk에서 usage dict, 아니면 None.

        tool 호출이 포함된 요청은 buffered 단일 이벤트로 반환한다
        (Ollama repair / openapiChatCompletions 폴백과 같은 방식).

        스트림 시작 전 Vertex 4xx/5xx면 VertexAPIError를 raise한다.
        스트림 도중 끊김/파싱 실패는 방어적으로 해당 줄을 건너뛴다.
        """
        cfg = resolved_config or model_config(model) or {"location": VERTEX_LOCATION, "vertex_model": model}
        if cfg.get("api") == "openapiChatCompletions" or tools is not None:
            result = await self.generate(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                response_format=response_format,
                resolved_config=cfg,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
            )
            buffered: dict[str, Any] = {
                "delta_text": result.get("text") or "",
                "finish_reason": result.get("finish_reason", "stop"),
                "usage": result.get("usage"),
            }
            if result.get("tool_calls"):
                buffered["delta_tool_calls"] = result["tool_calls"]
            yield buffered
            return

        location = cfg.get("location", VERTEX_LOCATION)
        vertex_model = cfg.get("vertex_model", model)

        token = await self.token_provider.get_token()
        url = self._stream_generate_content_url(vertex_model, location)

        body = self._build_request_body(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            thinking_budget=cfg.get("thinking_budget"),
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
        )

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        # 세마포어는 연결+헤더 수신(및 4xx 에러 처리)까지만 점유하고,
        # aiter_lines 스트림 루프에 진입하기 전에 해제한다. 그렇지 않으면 긴
        # 스트림들이 MAX_CONCURRENCY 슬롯을 점유한 채 비스트림/후속 요청을 굶긴다.
        stream_ctx = self.http.stream("POST", url, headers=headers, json=body)
        await self.semaphore.acquire()
        entered = False
        try:
            try:
                resp = await stream_ctx.__aenter__()
                entered = True
            except httpx.TimeoutException as exc:
                raise VertexAPIError(504, f"Vertex AI request timed out: {exc}", code="timeout") from exc
            except httpx.RequestError as exc:
                raise VertexAPIError(502, f"Vertex AI connection error: {exc}", code="connection_error") from exc

            if resp.status_code >= 400:
                # 에러 본문을 읽어 VertexAPIError로 변환 (스트림 시작 전 에러).
                try:
                    await resp.aread()
                except Exception:
                    pass
                # raise 전에 열린 stream_ctx를 정리한다.
                await stream_ctx.__aexit__(None, None, None)
                entered = False
                raise _parse_vertex_error(resp)
        finally:
            self.semaphore.release()
            # __aenter__ 자체가 실패해 진입하지 못한 경우엔 닫을 컨텍스트가 없다.
            # 4xx로 위에서 이미 닫은 경우엔 entered=False라 중복 호출하지 않는다.

        # 여기부터 세마포어는 해제된 상태. resp는 열린 채로 스트리밍한다.
        try:
            try:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    stripped = line.strip()
                    if not stripped.startswith("data:"):
                        continue
                    payload = stripped[len("data:"):].strip()
                    if not payload or payload == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(payload)
                    except (ValueError, TypeError):
                        # 부분 줄/파싱 실패는 방어적으로 건너뜀
                        continue

                    delta_text, finish_reason, usage = _parse_stream_chunk(chunk)
                    yield {
                        "delta_text": delta_text,
                        "finish_reason": finish_reason,
                        "usage": usage,
                    }
            except httpx.TimeoutException as exc:
                raise VertexAPIError(504, f"Vertex AI request timed out: {exc}", code="timeout") from exc
            except httpx.RequestError as exc:
                raise VertexAPIError(502, f"Vertex AI connection error: {exc}", code="connection_error") from exc
        finally:
            # 정상 종료/에러/클라이언트 끊김(GeneratorExit) 모든 경로에서 닫는다.
            if entered:
                await stream_ctx.__aexit__(None, None, None)


# ---------------------------------------------------------------------------
# Vertex AI Rerank client
# ---------------------------------------------------------------------------

class VertexRerankClient:
    """Vertex AI Search Ranking API (Rerank) 클라이언트."""

    provider_name = "vertex"

    def parse_usage(self, payload: Any, endpoint: str) -> NormalizedUsage | None:
        return parse_vertex_usage(payload, endpoint)

    def map_http_error(self, response: httpx.Response) -> VertexAPIError:
        return map_http_error_for_provider(self.provider_name, response)

    def __init__(self, token_provider: GoogleAccessTokenProvider | None = None) -> None:
        self.token_provider = token_provider or GoogleAccessTokenProvider()
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS))
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    async def close(self) -> None:
        await self.http.aclose()

    def _rank_url(self, location: str) -> str:
        project = self.token_provider.project_id
        return (
            f"https://discoveryengine.googleapis.com/"
            f"v1/projects/{project}/locations/{location}/"
            f"rankingConfigs/default_ranking_config:rank"
        )

    async def rank(
        self,
        *,
        model: str,
        query: str,
        records: list[dict[str, Any]],
        top_n: int | None = None,
        ignore_record_details_in_response: bool = True,
        location: str = "global",
    ) -> list[dict[str, Any]]:
        token = await self.token_provider.get_token()
        url = self._rank_url(location)

        body: dict[str, Any] = {
            "query": query,
            "records": records,
            "ignoreRecordDetailsInResponse": ignore_record_details_in_response,
        }
        if top_n is not None:
            body["topN"] = top_n
        if model:
            body["model"] = model

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        try:
            async with self.semaphore:
                resp = await self.http.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise VertexAPIError(504, f"Vertex AI request timed out: {exc}", code="timeout") from exc
        except httpx.RequestError as exc:
            raise VertexAPIError(502, f"Vertex AI connection error: {exc}", code="connection_error") from exc

        if resp.status_code >= 400:
            raise _parse_vertex_error(resp)

        try:
            data = resp.json()
        except Exception as exc:
            raise VertexAPIError(502, f"Invalid JSON from Vertex AI: {exc}", code="bad_gateway") from exc

        records_out = data.get("records")
        if not isinstance(records_out, list):
            if "records" not in data:
                return []
            raise VertexAPIError(502, "Malformed Vertex AI response: missing records[]", code="bad_gateway")

        return records_out
