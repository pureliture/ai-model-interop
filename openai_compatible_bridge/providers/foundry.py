"""Palantir Foundry OpenAI-compatible chat provider."""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from openai_compatible_bridge.core.cost_tracking import NormalizedUsage
from openai_compatible_bridge.core.http_timeout import http_timeout_seconds
from openai_compatible_bridge.core.protocol_stream import ProtocolStream
from openai_compatible_bridge.core.provider_adapter import (
    map_http_error_for_provider,
    parse_protocol_usage,
)
from openai_compatible_bridge.providers.vertex import VertexAPIError

FOUNDRY_BASE_URL = os.getenv("FOUNDRY_BASE_URL", "").strip().rstrip("/")
FOUNDRY_TOKEN = os.getenv("FOUNDRY_TOKEN", "").strip()
HTTP_TIMEOUT_SECONDS = http_timeout_seconds("FOUNDRY_HTTP_TIMEOUT_SECONDS")

FOUNDRY_OPENAI_PROTOCOL = "openai_chat_completions"
FOUNDRY_ANTHROPIC_PROTOCOL = "anthropic_messages"
FOUNDRY_XAI_RESPONSES_PROTOCOL = "xai_responses"
FOUNDRY_OPENAI_RESPONSES_PROTOCOL = "openai_responses"
FOUNDRY_GOOGLE_GENERATE_CONTENT_PROTOCOL = "google_generate_content"
FOUNDRY_OPENAI_PATH = "/api/v2/llm/proxy/openai/v1/chat/completions"
FOUNDRY_ANTHROPIC_PATH = "/api/v2/llm/proxy/anthropic/v1/messages"
FOUNDRY_XAI_RESPONSES_PATH = "/api/v2/llm/proxy/xai/v1/responses"
FOUNDRY_OPENAI_RESPONSES_PATH = "/api/v2/llm/proxy/openai/v1/responses"
FOUNDRY_GOOGLE_PATH_PREFIX = "/api/v2/llm/proxy/google/v1/models/"


from openai_compatible_bridge.core.transforms import _openai_error_from_payload

def _parse_foundry_error(response: httpx.Response) -> VertexAPIError:
    try:
        payload = response.json()
    except Exception:
        payload = {"error": {"message": response.text}}

    parsed = _openai_error_from_payload(payload)
    if parsed is None:
        message = "Foundry request failed."
        code = str(response.status_code)
    else:
        message, code = parsed

    return VertexAPIError(
        response.status_code,
        message=message,
        code=code,
        raw=payload,
        stage="provider_http_status",
        upstream_status=response.status_code,
    )


class FoundryChatClient(ProtocolStream):
    """OpenAI Chat Completions client for a fixed Foundry endpoint.

    The endpoint is deployment configuration, not request input. This prevents a
    caller from redirecting the bridge's bearer token to an arbitrary URL.
    """

    provider_name = "foundry"
    provider_label = "Foundry"

    def parse_usage(self, payload: Any, endpoint: str) -> NormalizedUsage | None:
        return parse_protocol_usage(payload, endpoint)

    def map_http_error(self, response: httpx.Response) -> VertexAPIError:
        return map_http_error_for_provider(self.provider_name, response)

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token: str | None = None,
    ) -> None:
        self.base_url = (base_url if base_url is not None else FOUNDRY_BASE_URL).strip().rstrip("/")
        self.token = token if token is not None else FOUNDRY_TOKEN
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS))

    async def close(self) -> None:
        await self.http.aclose()

    def _protocol(self, resolved_config: dict[str, Any] | None) -> str:
        protocol = (resolved_config or {}).get("protocol", FOUNDRY_OPENAI_PROTOCOL)
        if protocol is None:
            protocol = FOUNDRY_OPENAI_PROTOCOL
        if protocol not in {
            FOUNDRY_OPENAI_PROTOCOL,
            FOUNDRY_ANTHROPIC_PROTOCOL,
            FOUNDRY_XAI_RESPONSES_PROTOCOL,
            FOUNDRY_OPENAI_RESPONSES_PROTOCOL,
            FOUNDRY_GOOGLE_GENERATE_CONTENT_PROTOCOL,
        }:
            raise VertexAPIError(
                503,
                f"Unsupported Foundry protocol: {protocol}.",
                code="provider_config_error",
            )
        return protocol

    def _url_for_protocol(
        self,
        protocol: str,
        *,
        model: str | None = None,
        stream: bool = False,
    ) -> str:
        if not self.base_url:
            raise VertexAPIError(
                503,
                "FOUNDRY_BASE_URL is not configured.",
                code="provider_config_error",
            )
        if protocol == FOUNDRY_OPENAI_PROTOCOL:
            return self.base_url

        parsed = urlsplit(self.base_url)
        if parsed.query or parsed.fragment or not parsed.path.endswith(FOUNDRY_OPENAI_PATH):
            raise VertexAPIError(
                503,
                "FOUNDRY_BASE_URL must be the documented OpenAI chat proxy URL "
                "when using a non-OpenAI Foundry protocol.",
                code="provider_config_error",
            )
        root = parsed.path[: -len(FOUNDRY_OPENAI_PATH)]
        if protocol == FOUNDRY_GOOGLE_GENERATE_CONTENT_PROTOCOL:
            if not model:
                raise VertexAPIError(
                    503,
                    "Foundry Google protocol requires a provider model.",
                    code="provider_config_error",
                )
            encoded_model = quote(str(model), safe="")
            action = "streamGenerateContent?alt=sse" if stream else "generateContent"
            path = f"{root}{FOUNDRY_GOOGLE_PATH_PREFIX}{encoded_model}:{action}"
            return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
        path = {
            FOUNDRY_ANTHROPIC_PROTOCOL: FOUNDRY_ANTHROPIC_PATH,
            FOUNDRY_XAI_RESPONSES_PROTOCOL: FOUNDRY_XAI_RESPONSES_PATH,
            FOUNDRY_OPENAI_RESPONSES_PROTOCOL: FOUNDRY_OPENAI_RESPONSES_PATH,
        }[protocol]
        return urlunsplit((parsed.scheme, parsed.netloc, root + path, "", ""))

    def _headers(self, protocol: str) -> dict[str, str]:
        if not self.token:
            raise VertexAPIError(
                503,
                "FOUNDRY_TOKEN is not configured.",
                code="provider_config_error",
            )
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        if protocol == FOUNDRY_ANTHROPIC_PROTOCOL:
            headers["anthropic-version"] = "2023-06-01"
        return headers


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
        reasoning_effort: str | None = None,
        resolved_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        protocol = self._protocol(resolved_config)
        body = self._build_request_body(
            protocol=protocol,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
            stream=False,
        )

        try:
            response = await self.http.post(
                self._url_for_protocol(protocol, model=model),
                headers=self._headers(protocol),
                json=body,
            )
        except httpx.TimeoutException as exc:
            stage = "provider_connect" if isinstance(exc, httpx.ConnectTimeout) else (
                "provider_read" if isinstance(exc, httpx.ReadTimeout) else "provider_transport"
            )
            raise VertexAPIError(504, f"Foundry request timed out: {exc}", code="timeout", stage=stage) from exc
        except httpx.RequestError as exc:
            stage = "provider_connect" if isinstance(exc, httpx.ConnectError) else "provider_transport"
            raise VertexAPIError(502, f"Foundry connection error: {exc}", code="connection_error", stage=stage) from exc

        if response.status_code >= 400:
            raise self.map_http_error(response)

        try:
            payload = response.json()
        except Exception as exc:
            raise VertexAPIError(502, f"Invalid JSON from Foundry: {exc}", code="bad_gateway") from exc

        return self._parse_protocol_response(payload, protocol, messages, max_tokens)
