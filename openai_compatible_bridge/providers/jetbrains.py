"""JetBrains AI (Grazie Gateway) OpenAI-compatible chat provider.

JetBrains AI API는 Grazie Gateway를 통해 접근한다.
각 프로바이더(OpenAI, Anthropic, Grok, Google)별로 다른 endpoint를 사용하며,
X-LLM-Model 헤더로 라우팅된다.

transform 로직은 core/transforms.py에서 import하여 재사용한다.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from openai_compatible_bridge.core.conversation_state import (
    ConversationStateCache,
    get_default_conversation_state_cache,
)
from openai_compatible_bridge.core.cost_tracking import NormalizedUsage
from openai_compatible_bridge.core.provider_adapter import (
    map_http_error_for_provider,
    parse_foundry_usage,
)
from openai_compatible_bridge.core.transforms import (
    ANTHROPIC_THINKING_BUDGETS,
    GOOGLE_THINKING_BUDGETS,
    STANDARD_REASONING_EFFORTS,
    _anthropic_usage,
    _append_anthropic_turn,
    _clamp_xai_reasoning_effort,
    _google_contents_and_system,
    _google_json_object,
    _google_response_result,
    _google_tool_config,
    _google_tools,
    _google_usage,
    _map_finish_reason,
    _normalize_standard_reasoning_effort,
    _openai_input_messages,
    _unwrap_optional,
    _usage,
    _xai_input_messages,
    _xai_text,
    _xai_usage,
)
from openai_compatible_bridge.providers.vertex import (
    VertexAPIError,
    _coerce_openai_usage,
    _extract_message_text,
    _apply_response_format,
)

JETBRAINS_AI_BASE_URL = os.getenv("JETBRAINS_AI_BASE_URL", "https://ingrazzio-cloud-prod.labs.jb.gg").strip().rstrip("/")
JETBRAINS_AI_TOKEN = os.getenv("JETBRAINS_AI_TOKEN", "").strip()
HTTP_TIMEOUT_SECONDS = float(
    (os.getenv("JETBRAINS_AI_HTTP_TIMEOUT_SECONDS") or os.getenv("HTTP_TIMEOUT_SECONDS") or "60")
)

# 프로토콜 상수
JETBRAINS_OPENAI_PROTOCOL = "openai_responses"
JETBRAINS_ANTHROPIC_PROTOCOL = "anthropic_messages"
JETBRAINS_XAI_PROTOCOL = "xai_responses"
JETBRAINS_GOOGLE_PROTOCOL = "google_generate_content"

# endpoint 경로
JETBRAINS_OPENAI_PATH = "/v1/responses"
JETBRAINS_ANTHROPIC_PATH = "/v1/messages"
JETBRAINS_GOOGLE_PATH_PREFIX = "/v1beta1/projects/jetbrains-grazie/locations/global/publishers/google/models/"

GRAZIE_AGENT_HEADER = '{"name":"junie:cli","version":" "}'


def infer_jetbrains_protocol(model_name: str | None) -> str:
    """Infer JetBrains Grazie protocol from a model identifier when protocol is omitted."""
    raw = str(model_name or "").strip().lower()
    if raw.startswith("jetbrains:"):
        raw = raw[len("jetbrains:") :].strip()
    base = raw.split("/")[-1].split(":")[-1].strip()
    if base.startswith("claude"):
        return JETBRAINS_ANTHROPIC_PROTOCOL
    if base.startswith("gemini"):
        return JETBRAINS_GOOGLE_PROTOCOL
    if base.startswith("grok"):
        return JETBRAINS_XAI_PROTOCOL
    return JETBRAINS_OPENAI_PROTOCOL


def _grazie_headers(
    llm_model: str,
    token: str,
    *,
    auth_mode: str | None = None,
    grazie_agent: str | None = None,
) -> dict[str, str]:
    """Grazie Gateway 라우팅 헤더를 생성한다."""
    effective_auth_mode = (auth_mode or os.getenv("JETBRAINS_AI_AUTH_MODE") or "bearer").strip().lower()
    effective_agent = (
        (grazie_agent if grazie_agent is not None else os.getenv("JETBRAINS_GRAZIE_AGENT")) or GRAZIE_AGENT_HEADER
    ).strip() or GRAZIE_AGENT_HEADER
    headers: dict[str, str] = {
        "Content-Type": "application/json",
        "Grazie-Agent": effective_agent,
        "X-LLM-Model": llm_model,
        "X-Keep-Path": "true",
        "X-Accept-EAP-License": "false",
    }
    if effective_auth_mode in {"jwt", "grazie-jwt", "grazie_authenticate_jwt", "personal"}:
        headers["Grazie-Authenticate-JWT"] = token
    else:
        headers["Authorization"] = f"Bearer {token}"
    if llm_model == "google":
        headers["X-Free-Google-Api"] = "true"
    if llm_model == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    return headers


def validate_jetbrains_parameters(
    protocol: str,
    *,
    reasoning_effort: str | None = None,
    parallel_tool_calls: bool | None = None,
    response_format: dict[str, Any] | None = None,
) -> None:
    if reasoning_effort is not None:
        _normalize_standard_reasoning_effort(reasoning_effort)
    if parallel_tool_calls is not None and not isinstance(parallel_tool_calls, bool):
        raise VertexAPIError(400, "parallel_tool_calls must be a boolean.", code="invalid_request_error")
    if protocol == JETBRAINS_ANTHROPIC_PROTOCOL and response_format is not None:
        raise VertexAPIError(400, "response_format is not supported for this JetBrains protocol.", code="unsupported_parameter")
    if (
        protocol in {JETBRAINS_OPENAI_PROTOCOL, JETBRAINS_XAI_PROTOCOL}
        and response_format is not None
        and response_format.get("type") == "json_schema"
        and not isinstance(response_format.get("json_schema"), dict)
    ):
        raise VertexAPIError(400, "Invalid json_schema response format.", code="invalid_request_error")


class JetBrainsChatClient:
    """OpenAI Chat Completions client for JetBrains Grazie Gateway.

    Each provider (OpenAI, Anthropic, Grok, Google) uses a different endpoint.
    Routing is done via X-LLM-Model header.
    """

    provider_name = "jetbrains"

    def parse_usage(self, payload: Any, endpoint: str) -> NormalizedUsage | None:
        return parse_foundry_usage(payload, endpoint)

    def map_http_error(self, response: httpx.Response) -> VertexAPIError:
        return map_http_error_for_provider("jetbrains", response)

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token: str | None = None,
        auth_mode: str | None = None,
        grazie_agent: str | None = None,
        state_cache: ConversationStateCache | None = None,
    ) -> None:
        self.base_url = (base_url if base_url is not None else JETBRAINS_AI_BASE_URL).strip().rstrip("/")
        self.token = token if token is not None else JETBRAINS_AI_TOKEN
        self.auth_mode = auth_mode
        self.grazie_agent = grazie_agent
        self.state_cache = state_cache if state_cache is not None else get_default_conversation_state_cache()
        self._active_resolved_config: dict[str, Any] | None = None
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS))

    async def close(self) -> None:
        await self.http.aclose()

    def _protocol(self, resolved_config: dict[str, Any] | None, *, model: str | None = None) -> str:
        cfg = resolved_config or {}
        protocol = cfg.get("protocol")
        if not protocol:
            protocol = infer_jetbrains_protocol(cfg.get("provider_model") or model)
        if protocol not in {
            JETBRAINS_OPENAI_PROTOCOL,
            JETBRAINS_ANTHROPIC_PROTOCOL,
            JETBRAINS_XAI_PROTOCOL,
            JETBRAINS_GOOGLE_PROTOCOL,
        }:
            raise VertexAPIError(
                503,
                f"Unsupported JetBrains protocol: {protocol}.",
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
                "JETBRAINS_AI_BASE_URL is not configured.",
                code="provider_config_error",
            )
        if protocol == JETBRAINS_OPENAI_PROTOCOL:
            return f"{self.base_url}{JETBRAINS_OPENAI_PATH}"
        if protocol == JETBRAINS_ANTHROPIC_PROTOCOL:
            return f"{self.base_url}{JETBRAINS_ANTHROPIC_PATH}"
        if protocol == JETBRAINS_XAI_PROTOCOL:
            return f"{self.base_url}{JETBRAINS_OPENAI_PATH}"
        if protocol == JETBRAINS_GOOGLE_PROTOCOL:
            if not model:
                raise VertexAPIError(
                    503,
                    "JetBrains Google protocol requires a provider model.",
                    code="provider_config_error",
                )
            encoded_model = quote(str(model), safe="")
            action = "streamGenerateContent?alt=sse" if stream else "generateContent"
            return f"{self.base_url}{JETBRAINS_GOOGLE_PATH_PREFIX}{encoded_model}:{action}"
        raise VertexAPIError(
            503,
            f"Unsupported JetBrains protocol: {protocol}.",
            code="provider_config_error",
        )

    def _llm_model_header(self, protocol: str) -> str:
        return {
            JETBRAINS_OPENAI_PROTOCOL: "openai",
            JETBRAINS_ANTHROPIC_PROTOCOL: "anthropic",
            JETBRAINS_XAI_PROTOCOL: "grok",
            JETBRAINS_GOOGLE_PROTOCOL: "google",
        }.get(protocol, "openai")

    def _headers(self, protocol: str, *, resolved_config: dict[str, Any] | None = None) -> dict[str, str]:
        cfg = resolved_config if resolved_config is not None else (self._active_resolved_config or {})
        effective_token = (cfg.get("api_key") or self.token or "").strip()
        if not effective_token:
            raise VertexAPIError(
                503,
                "JETBRAINS_AI_TOKEN is not configured.",
                code="provider_config_error",
            )
        return _grazie_headers(
            self._llm_model_header(protocol),
            effective_token,
            auth_mode=cfg.get("auth_mode") or self.auth_mode,
            grazie_agent=cfg.get("grazie_agent") or self.grazie_agent,
        )

    @staticmethod
    def _build_openai_request_body(
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        stop: str | list[str] | None,
        response_format: dict[str, Any] | None,
        reasoning_effort: str | None = None,
        protocol: str = JETBRAINS_OPENAI_PROTOCOL,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        stream: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "input": _xai_input_messages(messages),
            "stream": stream,
        }
        if max_tokens is not None:
            body["max_output_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if stop is not None:
            body["stop"] = stop
        if reasoning_effort is not None:
            effort = _normalize_standard_reasoning_effort(reasoning_effort)
            if effort is not None:
                if protocol == JETBRAINS_XAI_PROTOCOL:
                    clamped = _clamp_xai_reasoning_effort(effort)
                    if clamped is not None:
                        body["reasoning"] = {"effort": clamped}
                else:
                    body["reasoning"] = {"effort": "high" if effort == "max" else effort}
        if response_format is not None:
            format_type = response_format.get("type")
            if format_type == "json_object":
                body["text"] = {"format": {"type": "json_object"}}
            elif format_type == "json_schema":
                schema = response_format.get("json_schema")
                if not isinstance(schema, dict):
                    raise VertexAPIError(400, "Invalid json_schema response format.", code="invalid_request_error")
                body["text"] = {"format": {**schema, "type": "json_schema"}}
        if tools is not None:
            grazie_tools: list[dict[str, Any]] = []
            for t in tools:
                fn = t.get("function") if t.get("type") == "function" and isinstance(t.get("function"), dict) else t
                grazie_tools.append(
                    {
                        "type": "function",
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                    }
                )
            body["tools"] = grazie_tools
        if parallel_tool_calls is not None:
            body["parallel_tool_calls"] = parallel_tool_calls
        if tool_choice is not None:
            if isinstance(tool_choice, dict):
                fn_dict = tool_choice.get("function")
                name = fn_dict.get("name") if isinstance(fn_dict, dict) else tool_choice.get("name")
                if name:
                    body["tool_choice"] = {"type": "function", "name": name}
                else:
                    body["tool_choice"] = tool_choice
            elif isinstance(tool_choice, str):
                body["tool_choice"] = tool_choice
        return body

    @staticmethod
    def _build_anthropic_request_body(
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        stop: str | list[str] | None,
        stream: bool,
        reasoning_effort: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
    ) -> dict[str, Any]:
        system_parts: list[str] = []
        chat_messages: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content")
            if role == "system":
                if content:
                    text = _extract_message_text(content)
                    if text:
                        system_parts.append(text)
            elif role == "assistant":
                tool_calls = message.get("tool_calls")
                if tool_calls:
                    blocks: list[dict[str, Any]] = []
                    if content:
                        text = _extract_message_text(content)
                        if text:
                            blocks.append({"type": "text", "text": text})
                    for tc in tool_calls:
                        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                        raw_args = fn.get("arguments", "{}")
                        if isinstance(raw_args, str):
                            try:
                                input_dict = json.loads(raw_args) if raw_args.strip() else {}
                            except Exception:
                                input_dict = {}
                        elif isinstance(raw_args, dict):
                            input_dict = raw_args
                        else:
                            input_dict = {}
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.get("id", ""),
                                "name": fn.get("name", ""),
                                "input": input_dict,
                            }
                        )
                    _append_anthropic_turn(chat_messages, "assistant", blocks)
                else:
                    _append_anthropic_turn(chat_messages, "assistant", content if content is not None else "")
            elif role == "tool":
                call_id = message.get("tool_call_id") or message.get("id") or ""
                if isinstance(content, (dict, list)):
                    content_text = json.dumps(content, ensure_ascii=False)
                else:
                    content_text = _extract_message_text(content) if content is not None else ""
                result_block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": content_text,
                }
                if message.get("is_error"):
                    result_block["is_error"] = True
                _append_anthropic_turn(chat_messages, "user", [result_block])
            else:
                _append_anthropic_turn(chat_messages, "user", content if content is not None else "")

        if not chat_messages:
            chat_messages.append({"role": "user", "content": ""})

        effective_effort = _normalize_standard_reasoning_effort(reasoning_effort)
        effective_max_tokens = max_tokens or 4096
        if effective_effort is not None:
            budget = ANTHROPIC_THINKING_BUDGETS[effective_effort]
            effective_max_tokens = max(max_tokens or 0, budget + 4096)

        body: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": effective_max_tokens,
            "stream": stream,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if effective_effort is not None:
            body["thinking"] = {"type": "enabled", "budget_tokens": ANTHROPIC_THINKING_BUDGETS[effective_effort]}
        else:
            if temperature is not None:
                body["temperature"] = temperature
            if top_p is not None:
                body["top_p"] = top_p
        if stop is not None:
            body["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
        if tools is not None:
            grazie_tools: list[dict[str, Any]] = []
            for t in tools:
                fn = t.get("function") if t.get("type") == "function" and isinstance(t.get("function"), dict) else t
                grazie_tools.append(
                    {
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
                    }
                )
            body["tools"] = grazie_tools

        anthropic_tool_choice: dict[str, Any] | None = None
        if tool_choice is not None:
            if tool_choice == "auto":
                anthropic_tool_choice = {"type": "auto"}
            elif tool_choice == "none":
                anthropic_tool_choice = {"type": "none"}
            elif tool_choice == "required":
                anthropic_tool_choice = {"type": "any"}
            elif isinstance(tool_choice, dict):
                fn_dict = tool_choice.get("function")
                name = fn_dict.get("name") if isinstance(fn_dict, dict) else tool_choice.get("name")
                if name and tool_choice.get("type") in (None, "function", "tool"):
                    anthropic_tool_choice = {"type": "tool", "name": name}
                else:
                    anthropic_tool_choice = dict(tool_choice)
        if parallel_tool_calls is False and tools:
            if anthropic_tool_choice is None:
                anthropic_tool_choice = {"type": "auto", "disable_parallel_tool_use": True}
            elif anthropic_tool_choice.get("type") != "none":
                anthropic_tool_choice = {**anthropic_tool_choice, "disable_parallel_tool_use": True}
        if anthropic_tool_choice is not None:
            body["tool_choice"] = anthropic_tool_choice
        return body

    @staticmethod
    def _build_google_request_body(
        *,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        stop: str | list[str] | None,
        response_format: dict[str, Any] | None,
        reasoning_effort: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        state_cache: ConversationStateCache | None = None,
    ) -> dict[str, Any]:
        contents, system_parts = _google_contents_and_system(messages, state_cache=state_cache)
        body: dict[str, Any] = {"contents": contents}
        if system_parts:
            body["systemInstruction"] = {"parts": system_parts}
        generation_config: dict[str, Any] = {}
        if max_tokens is not None:
            generation_config["maxOutputTokens"] = max_tokens
        if temperature is not None:
            generation_config["temperature"] = temperature
        if top_p is not None:
            generation_config["topP"] = top_p
        if stop is not None:
            generation_config["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)
        if reasoning_effort is not None:
            effort = _normalize_standard_reasoning_effort(reasoning_effort)
            if effort is not None:
                generation_config["thinkingConfig"] = {"thinkingBudget": GOOGLE_THINKING_BUDGETS[effort]}
        _apply_response_format(generation_config, response_format)
        if generation_config:
            body["generationConfig"] = generation_config
        google_tools = _google_tools(tools)
        if google_tools is not None:
            body["tools"] = google_tools
        tool_config = _google_tool_config(tool_choice)
        if tool_config is not None:
            body["toolConfig"] = tool_config
        return body

    def _build_request_body(
        self,
        *,
        protocol: str,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        stop: str | list[str] | None,
        response_format: dict[str, Any] | None,
        reasoning_effort: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        validate_jetbrains_parameters(
            protocol,
            reasoning_effort=reasoning_effort,
            parallel_tool_calls=parallel_tool_calls,
            response_format=response_format,
        )
        if protocol == JETBRAINS_OPENAI_PROTOCOL or protocol == JETBRAINS_XAI_PROTOCOL:
            return self._build_openai_request_body(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                response_format=response_format,
                reasoning_effort=reasoning_effort,
                protocol=protocol,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
                stream=stream,
            )
        if protocol == JETBRAINS_ANTHROPIC_PROTOCOL:
            return self._build_anthropic_request_body(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                stream=stream,
                reasoning_effort=reasoning_effort,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=parallel_tool_calls,
            )
        if protocol == JETBRAINS_GOOGLE_PROTOCOL:
            return self._build_google_request_body(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                response_format=response_format,
                reasoning_effort=reasoning_effort,
                tools=tools,
                tool_choice=tool_choice,
                state_cache=self.state_cache,
            )
        raise VertexAPIError(503, f"Unsupported protocol: {protocol}", code="provider_config_error")

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
        native_model = (resolved_config or {}).get("provider_model", model)
        protocol = self._protocol(resolved_config, model=native_model)
        url = self._url_for_protocol(protocol, model=native_model, stream=False)
        headers = self._headers(protocol, resolved_config=resolved_config)
        body = self._build_request_body(
            protocol=protocol,
            model=native_model,
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
            response = await self.http.post(url, headers=headers, json=body)
        except httpx.TimeoutException as exc:
            raise VertexAPIError(504, f"JetBrains request timed out: {exc}", code="timeout") from exc
        except httpx.RequestError as exc:
            raise VertexAPIError(502, f"JetBrains connection error: {exc}", code="connection_error") from exc

        if response.status_code >= 400:
            raise self.map_http_error(response)

        try:
            payload = response.json()
        except Exception as exc:
            raise VertexAPIError(502, f"Invalid JSON from JetBrains: {exc}", code="bad_gateway") from exc

        if not isinstance(payload, dict) or payload.get("error") or payload.get("type") == "error" or payload.get("status") in {"failed", "cancelled"}:
            raise VertexAPIError(502, "JetBrains returned an upstream error.", code="bad_gateway")
        if protocol == JETBRAINS_ANTHROPIC_PROTOCOL and not isinstance(payload.get("content"), list):
            raise VertexAPIError(502, "Malformed JetBrains Anthropic response.", code="bad_gateway")
        if protocol in {JETBRAINS_OPENAI_PROTOCOL, JETBRAINS_XAI_PROTOCOL} and not isinstance(payload.get("output"), list):
            raise VertexAPIError(502, "Malformed JetBrains Responses response.", code="bad_gateway")

        if protocol == JETBRAINS_GOOGLE_PROTOCOL:
            return _google_response_result(payload, state_cache=self.state_cache)
        elif protocol == JETBRAINS_ANTHROPIC_PROTOCOL:
            # Anthropic response → OpenAI format
            content = payload.get("content", [])
            text_parts = []
            tool_calls = None
            for block in content if isinstance(content, list) else []:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif block.get("type") == "tool_use":
                        if tool_calls is None:
                            tool_calls = []
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input", {})),
                            },
                        })
            usage = payload.get("usage", {})
            return {
                "text": "".join(text_parts) or None,
                "tool_calls": tool_calls,
                "finish_reason": "tool_calls" if tool_calls else (_map_finish_reason(payload.get("stop_reason")) or "stop"),
                "usage": {
                    "prompt_tokens": int(usage.get("input_tokens", 0) or 0),
                    "completion_tokens": int(usage.get("output_tokens", 0) or 0),
                    "total_tokens": int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0),
                },
            }
        else:
            # OpenAI Responses API → OpenAI Chat Completions format
            text = _xai_text(payload)
            tool_calls = None
            for item in payload.get("output", []) or []:
                if isinstance(item, dict) and item.get("type") == "function_call":
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append({
                        "id": item.get("call_id", ""),
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}"),
                        },
                    })
            usage = payload.get("usage", {})
            return {
                "text": text or None,
                "tool_calls": tool_calls,
                "finish_reason": "tool_calls" if tool_calls else (_map_finish_reason((payload.get("incomplete_details") or {}).get("reason")) or _map_finish_reason(payload.get("status")) or "stop"),
                "usage": {
                    "prompt_tokens": int(usage.get("input_tokens", 0) or 0),
                    "completion_tokens": int(usage.get("output_tokens", 0) or 0),
                    "total_tokens": int(usage.get("total_tokens", 0) or 0) or (int(usage.get("input_tokens", 0) or 0) + int(usage.get("output_tokens", 0) or 0)),
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
        # Both gateways use the same native protocol event formats. Reuse the
        # established parser while keeping JetBrains URLs, headers and bodies.
        from openai_compatible_bridge.providers.foundry import FoundryChatClient

        native_model = (resolved_config or {}).get("provider_model", model)
        protocol = self._protocol(resolved_config, model=native_model)
        effective_config = {**(resolved_config or {}), "provider_model": native_model, "protocol": protocol}
        prev_cfg = self._active_resolved_config
        self._active_resolved_config = effective_config
        try:
            async for event in FoundryChatClient.stream_chat(
                self, model=native_model, messages=messages, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p, stop=stop,
                response_format=response_format, resolved_config=effective_config,
                tools=tools, tool_choice=tool_choice, parallel_tool_calls=parallel_tool_calls,
                reasoning_effort=reasoning_effort, _require_complete=True,
            ):
                yield event
        finally:
            self._active_resolved_config = prev_cfg
