"""Generic OpenAI-compatible upstream provider.

Operator가 registry alias마다 `base_url`(+키)을 지정하면 해당 upstream의
`/chat/completions`를 호출한다. 용도: OpenCode Go 같은 외부 OpenAI gateway를
bridge의 registry 라우팅 + cost gate 뒤에 두기.

- client가 URL/key를 지정할 수 없다. alias 해석은 registry(env)만 본다.
- 키는 alias의 `api_key_env`가 가리키는 환경변수에서 호출 시점에 읽는다
  (재시작 없이 로테이션 가능). `api_key` 직접 기재도 허용하지만 권장하지 않는다.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import urlsplit

import httpx

from openai_compatible_bridge.core.provider_adapter import (
    map_http_error_for_provider,
    parse_openai_compat_usage,
)
from openai_compatible_bridge.providers.vertex import VertexAPIError

HTTP_TIMEOUT_SECONDS = float(os.getenv("OPENAI_COMPAT_HTTP_TIMEOUT_SECONDS", os.getenv("HTTP_TIMEOUT_SECONDS", "60")))

_HEADER_NAME = re.compile(r"^[A-Za-z0-9!#$%&'*+.^_`|~-]{1,64}$")
_RESERVED_HEADERS = frozenset({
    "authorization", "content-type", "content-length", "host", "transfer-encoding",
})
MAX_EXTRA_HEADERS = 16


def _resolve_extra_headers(resolved_config: dict[str, Any] | None) -> dict[str, str]:
    """Alias별 고정 헤더.

    일부 upstream gateway는 라우팅용 헤더를 요구한다(예: OpenCode Go의
    `x-opencode-session`). registry alias의 `extra_headers`로만 지정할 수 있고,
    인증·전송 헤더는 덮어쓸 수 없다.
    """
    cfg = resolved_config or {}
    extra = cfg.get("extra_headers")
    if extra is None:
        return {}
    if not isinstance(extra, dict) or len(extra) > MAX_EXTRA_HEADERS:
        raise VertexAPIError(
            503,
            "openai_compat extra_headers must be an object with at most "
            f"{MAX_EXTRA_HEADERS} entries.",
            code="provider_config_error",
        )
    headers: dict[str, str] = {}
    for name, value in extra.items():
        if not isinstance(name, str) or not _HEADER_NAME.fullmatch(name) or not isinstance(value, str):
            raise VertexAPIError(
                503,
                "openai_compat extra_headers entries must be a header name and a string value.",
                code="provider_config_error",
            )
        if name.lower() in _RESERVED_HEADERS or "\n" in value or "\r" in value:
            raise VertexAPIError(
                503,
                f"openai_compat extra_headers may not set {name!r}.",
                code="provider_config_error",
            )
        headers[name] = value
    return headers


def _resolve_base_url(resolved_config: dict[str, Any] | None) -> str:
    base_url = ((resolved_config or {}).get("base_url") or "").strip().rstrip("/")
    if not base_url:
        raise VertexAPIError(
            503,
            "openai_compat alias requires a base_url.",
            code="provider_config_error",
        )
    scheme = urlsplit(base_url).scheme.lower()
    if scheme not in ("http", "https"):
        raise VertexAPIError(
            503,
            "openai_compat base_url must be http(s).",
            code="provider_config_error",
        )
    return base_url


def _resolve_api_key(resolved_config: dict[str, Any] | None) -> str | None:
    cfg = resolved_config or {}
    direct = (cfg.get("api_key") or "").strip() if isinstance(cfg.get("api_key"), str) else ""
    if direct:
        return direct
    env_name = (cfg.get("api_key_env") or "").strip() if isinstance(cfg.get("api_key_env"), str) else ""
    if env_name:
        return (os.getenv(env_name) or "").strip() or None
    return None


class OpenAICompatibleChatClient:
    """Alias별 base_url로 고정된 OpenAI /chat/completions 클라이언트."""

    provider_name = "openai_compatible"

    def __init__(self, *, timeout_seconds: float | None = None) -> None:
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds or HTTP_TIMEOUT_SECONDS))

    async def close(self) -> None:
        await self.http.aclose()

    def parse_usage(self, payload: Any, endpoint: str) -> Any:
        return parse_openai_compat_usage(payload, endpoint)

    def map_http_error(self, response: httpx.Response) -> VertexAPIError:
        return map_http_error_for_provider(self.provider_name, response)

    @staticmethod
    def _build_request_body(
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        stop: str | list[str] | None,
        response_format: dict[str, Any] | None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        reasoning_effort: str | None = None,
        stream: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": stream,
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
        if parallel_tool_calls is not None:
            body["parallel_tool_calls"] = parallel_tool_calls
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        return body

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
        reasoning_effort: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        base_url = _resolve_base_url(resolved_config)
        api_key = _resolve_api_key(resolved_config)
        headers = {"Content-Type": "application/json", **_resolve_extra_headers(resolved_config)}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        body = self._build_request_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            reasoning_effort=reasoning_effort,
            stream=False,
        )
        try:
            response = await self.http.post(f"{base_url}/chat/completions", headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise VertexAPIError(504, f"OpenAI-compat request timed out: {exc}", code="timeout") from exc
        except httpx.RequestError as exc:
            raise VertexAPIError(502, f"OpenAI-compat connection error: {exc}", code="connection_error") from exc

        if response.status_code >= 400:
            raise self.map_http_error(response)

        try:
            payload = response.json()
        except Exception as exc:
            raise VertexAPIError(502, f"Invalid JSON from OpenAI-compat upstream: {exc}", code="bad_gateway") from exc

        choices = payload.get("choices") if isinstance(payload, dict) else None
        if not isinstance(choices, list) or not choices:
            raise VertexAPIError(502, "Malformed OpenAI-compat response: missing choices[]", code="bad_gateway")
        choice = choices[0] or {}
        message = choice.get("message", {}) if isinstance(choice.get("message"), dict) else {}
        tool_calls = message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else None
        from openai_compatible_bridge.providers.vertex import _extract_message_text

        content = message.get("content")
        if tool_calls:
            text = _extract_message_text(content) if content is not None else ""
            finish_reason = choice.get("finish_reason") or "tool_calls"
            text = text if text else None
        else:
            text = _extract_message_text(content or "")
            finish_reason = choice.get("finish_reason") or "stop"
        usage = payload.get("usage") if isinstance(payload, dict) else None
        return {
            "text": text,
            "tool_calls": tool_calls if tool_calls else None,
            "finish_reason": finish_reason,
            "usage": {
                "prompt_tokens": int((usage or {}).get("prompt_tokens", 0) or 0),
                "completion_tokens": int((usage or {}).get("completion_tokens", 0) or 0),
                "total_tokens": int((usage or {}).get("total_tokens", 0) or 0),
            },
        }

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
        reasoning_effort: str | None = None,
        **_: Any,
    ):
        from openai_compatible_bridge.providers.vertex import _coerce_openai_usage, _extract_message_text

        base_url = _resolve_base_url(resolved_config)
        api_key = _resolve_api_key(resolved_config)
        headers = {"Content-Type": "application/json", **_resolve_extra_headers(resolved_config)}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        body = self._build_request_body(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            response_format=response_format,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            reasoning_effort=reasoning_effort,
            stream=True,
        )
        stream_context = self.http.stream("POST", f"{base_url}/chat/completions", headers=headers, json=body)
        try:
            try:
                response = await stream_context.__aenter__()
            except httpx.TimeoutException as exc:
                raise VertexAPIError(504, f"OpenAI-compat request timed out: {exc}", code="timeout") from exc
            except httpx.RequestError as exc:
                raise VertexAPIError(502, f"OpenAI-compat connection error: {exc}", code="connection_error") from exc

            if response.status_code >= 400:
                try:
                    await response.aread()
                except Exception:
                    pass
                raise self.map_http_error(response)

            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                raw = line[len("data:"):].strip()
                if raw == "[DONE]":
                    break
                try:
                    event = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if not isinstance(event, dict):
                    continue
                choices = event.get("choices")
                choice = choices[0] if isinstance(choices, list) and choices else {}
                delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                delta = delta if isinstance(delta, dict) else {}
                delta_text = _extract_message_text(delta.get("content", ""))
                raw_tool_calls = delta.get("tool_calls")
                delta_tool_calls = raw_tool_calls if isinstance(raw_tool_calls, list) and raw_tool_calls else None
                finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
                usage = event.get("usage")
                normalized_usage = _coerce_openai_usage(usage) if isinstance(usage, dict) else None
                yield {
                    "delta_text": delta_text,
                    "delta_tool_calls": delta_tool_calls,
                    "finish_reason": finish_reason,
                    "usage": normalized_usage,
                }
        finally:
            await stream_context.__aexit__(None, None, None)
