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

from openai_compatible_bridge.core.cost_tracking import NormalizedUsage
from openai_compatible_bridge.core.provider_adapter import (
    map_http_error_for_provider,
    parse_foundry_usage,
)
from openai_compatible_bridge.core.transforms import (
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


def _grazie_headers(llm_model: str, token: str) -> dict[str, str]:
    """Grazie Gateway 라우팅 헤더를 생성한다."""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Grazie-Agent": GRAZIE_AGENT_HEADER,
        "X-LLM-Model": llm_model,
        "X-Keep-Path": "true",
        "X-Accept-EAP-License": "false",
    }
    if llm_model == "google":
        headers["X-Free-Google-Api"] = "true"
    if llm_model == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
    return headers


class JetBrainsChatClient:
    """OpenAI Chat Completions client for JetBrains Grazie Gateway.

    Each provider (OpenAI, Anthropic, Grok, Google) uses a different endpoint.
    Routing is done via X-LLM-Model header.
    """

    provider_name = "jetbrains"

    def parse_usage(self, payload: Any, endpoint: str) -> NormalizedUsage | None:
        return parse_foundry_usage(payload, endpoint)

    def map_http_error(self, response: httpx.Response) -> VertexAPIError:
        return map_http_error_for_provider("openai_compatible", response)

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token: str | None = None,
    ) -> None:
        self.base_url = (base_url if base_url is not None else JETBRAINS_AI_BASE_URL).strip().rstrip("/")
        self.token = token if token is not None else JETBRAINS_AI_TOKEN
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(HTTP_TIMEOUT_SECONDS))

    async def close(self) -> None:
        await self.http.aclose()

    def _protocol(self, resolved_config: dict[str, Any] | None) -> str:
        protocol = (resolved_config or {}).get("protocol", JETBRAINS_OPENAI_PROTOCOL)
        if protocol is None:
            protocol = JETBRAINS_OPENAI_PROTOCOL
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

    def _headers(self, protocol: str) -> dict[str, str]:
        if not self.token:
            raise VertexAPIError(
                503,
                "JETBRAINS_AI_TOKEN is not configured.",
                code="provider_config_error",
            )
        return _grazie_headers(self._llm_model_header(protocol), self.token)

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
        if response_format is not None:
            format_type = response_format.get("type")
            if format_type == "json_object":
                body["text"] = {"format": {"type": "json_object"}}
            elif format_type == "json_schema":
                body["text"] = {"format": response_format}
        if tools is not None:
            grazie_tools: list[dict[str, Any]] = []
            for t in tools:
                fn = t.get("function", {}) if t.get("type") == "function" else t
                grazie_tools.append(
                    {
                        "type": "function",
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                    }
                )
            body["tools"] = grazie_tools
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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        system_parts: list[str] = []
        chat_messages: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content")
            if role == "system":
                if content:
                    system_parts.append(_extract_message_text(content))
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
                    chat_messages.append({"role": "assistant", "content": blocks})
                else:
                    chat_messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                call_id = message.get("tool_call_id") or message.get("id") or ""
                content_text = _extract_message_text(content) if content is not None else ""
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": content_text,
                }
                # Anthropic requires tool_result in user turn
                if chat_messages and chat_messages[-1]["role"] == "user" and isinstance(chat_messages[-1].get("content"), list):
                    chat_messages[-1]["content"].append(result_block)
                else:
                    chat_messages.append({"role": "user", "content": [result_block]})
            else:
                chat_messages.append({"role": role, "content": content})

        body: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens or 4096,
            "stream": stream,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if stop is not None:
            body["stop_sequences"] = [stop] if isinstance(stop, str) else list(stop)
        if tools is not None:
            grazie_tools: list[dict[str, Any]] = []
            for t in tools:
                fn = t.get("function", {}) if t.get("type") == "function" else t
                grazie_tools.append(
                    {
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
                    }
                )
            body["tools"] = grazie_tools
        if tool_choice is not None:
            if tool_choice == "auto":
                body["tool_choice"] = {"type": "auto"}
            elif tool_choice == "none":
                body["tool_choice"] = {"type": "none"}
            elif tool_choice == "required":
                body["tool_choice"] = {"type": "any"}
            elif isinstance(tool_choice, dict):
                fn_dict = tool_choice.get("function")
                name = fn_dict.get("name") if isinstance(fn_dict, dict) else tool_choice.get("name")
                if name:
                    body["tool_choice"] = {"type": "tool", "name": name}
                else:
                    body["tool_choice"] = tool_choice
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
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        contents, system_parts = _google_contents_and_system(messages)
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
        if protocol == JETBRAINS_OPENAI_PROTOCOL or protocol == JETBRAINS_XAI_PROTOCOL:
            return self._build_openai_request_body(
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
                tools=tools,
                tool_choice=tool_choice,
            )
        if protocol == JETBRAINS_GOOGLE_PROTOCOL:
            return self._build_google_request_body(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
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
        protocol = self._protocol(resolved_config)
        llm_model = self._llm_model_header(protocol)
        url = self._url_for_protocol(protocol, model=(resolved_config or {}).get("provider_model", model), stream=False)
        headers = self._headers(protocol)
        body = self._build_request_body(
            protocol=protocol,
            model=(resolved_config or {}).get("provider_model", model),
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

        if protocol == JETBRAINS_GOOGLE_PROTOCOL:
            return _google_response_result(payload)
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
                "finish_reason": "tool_calls" if tool_calls else "stop",
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
                "finish_reason": "tool_calls" if tool_calls else (payload.get("status") or "stop"),
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
        protocol = self._protocol(resolved_config)
        llm_model = self._llm_model_header(protocol)
        url = self._url_for_protocol(protocol, model=(resolved_config or {}).get("provider_model", model), stream=True)
        headers = self._headers(protocol)
        body = self._build_request_body(
            protocol=protocol,
            model=(resolved_config or {}).get("provider_model", model),
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
            stream=True,
        )
        stream_context = self.http.stream("POST", url, headers=headers, json=body)
        try:
            try:
                response = await stream_context.__aenter__()
            except httpx.TimeoutException as exc:
                raise VertexAPIError(504, f"JetBrains request timed out: {exc}", code="timeout") from exc
            except httpx.RequestError as exc:
                raise VertexAPIError(502, f"JetBrains connection error: {exc}", code="connection_error") from exc

            if response.status_code >= 400:
                try:
                    await response.aread()
                except Exception:
                    pass
                raise self.map_http_error(response)

            if protocol == JETBRAINS_GOOGLE_PROTOCOL:
                # Google streaming: parse SSE
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
                    # Yield delta from Google response
                    candidates = event.get("candidates", [])
                    if candidates and isinstance(candidates[0], dict):
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", []) if isinstance(content, dict) else []
                        for part in parts:
                            if isinstance(part, dict) and "text" in part:
                                yield {
                                    "delta_text": part["text"],
                                    "delta_tool_calls": None,
                                    "finish_reason": None,
                                    "usage": None,
                                }
            elif protocol == JETBRAINS_ANTHROPIC_PROTOCOL:
                # Anthropic streaming: parse SSE
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
                    event_type = event.get("type", "")
                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if isinstance(delta, dict) and delta.get("type") == "text_delta":
                            yield {
                                "delta_text": delta.get("text", ""),
                                "delta_tool_calls": None,
                                "finish_reason": None,
                                "usage": None,
                            }
                    elif event_type == "message_stop":
                        yield {
                            "delta_text": "",
                            "delta_tool_calls": None,
                            "finish_reason": "stop",
                            "usage": None,
                        }
            else:
                # OpenAI Responses streaming
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
                    # Extract delta text from Responses API events
                    event_type = event.get("type", "")
                    if event_type == "response.output_text.delta":
                        yield {
                            "delta_text": event.get("delta", ""),
                            "delta_tool_calls": None,
                            "finish_reason": None,
                            "usage": None,
                        }
                    elif event_type == "response.completed":
                        usage = event.get("response", {}).get("usage", {})
                        yield {
                            "delta_text": "",
                            "delta_tool_calls": None,
                            "finish_reason": "stop",
                            "usage": {
                                "prompt_tokens": int(usage.get("input_tokens", 0) or 0),
                                "completion_tokens": int(usage.get("output_tokens", 0) or 0),
                                "total_tokens": int(usage.get("total_tokens", 0) or 0),
                            },
                        }
        finally:
            await stream_context.__aexit__(None, None, None)
