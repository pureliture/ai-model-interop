"""Provider-neutral native protocol request builders and SSE streaming."""
from __future__ import annotations

import json
import sys
from typing import Any
import httpx

from openai_compatible_bridge.core.transforms import (
    _anthropic_usage, _append_anthropic_turn, _clamp_xai_reasoning_effort, _google_contents_and_system,
    _google_json_object, _google_response_result, _google_tool_config, _google_tools,
    _google_usage, _map_finish_reason, _openai_error_from_payload, _xai_input_messages, _xai_usage,
)
from openai_compatible_bridge.providers.vertex import (
    VertexAPIError, _apply_response_format, _coerce_openai_usage, _extract_message_text,
)

OPENAI_CHAT = "openai_chat_completions"
ANTHROPIC = "anthropic_messages"
XAI_RESPONSES = "xai_responses"
OPENAI_RESPONSES = "openai_responses"
GOOGLE = "google_generate_content"

class ProtocolStream:
    """Protocol conversion over provider-supplied URL, headers and error mapping."""
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
            "messages": messages,
            "stream": stream,
        }
        # Foundry's current OpenAI backend rejects max_tokens and requires this
        # newer field name. The bridge's public contract remains max_tokens.
        if max_tokens is not None:
            body["max_completion_tokens"] = max_tokens
        # gpt-6-astra currently accepts only the default temperature. Omitting a
        # caller-supplied non-default is the compatible behavior for this backend;
        # forwarding it produces a deterministic upstream 400.
        if temperature is not None and temperature == 1.0:
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
        return body

    @staticmethod
    def _build_anthropic_request_body(
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
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
                    _append_anthropic_turn(chat_messages, "assistant", blocks)
                else:
                    _append_anthropic_turn(chat_messages, "assistant", content if content is not None else "")
            elif role == "tool":
                content_val = message.get("content", "")
                if isinstance(content_val, (dict, list)):
                    result_content = json.dumps(content_val, ensure_ascii=False)
                else:
                    result_content = str(content_val if content_val is not None else "")
                tool_block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id") or message.get("id") or "",
                    "content": result_content,
                }
                if message.get("is_error"):
                    tool_block["is_error"] = True
                _append_anthropic_turn(chat_messages, "user", [tool_block])
            else:
                _append_anthropic_turn(chat_messages, "user", content if content is not None else "")

        if not chat_messages:
            chat_messages.append({"role": "user", "content": ""})

        body: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": max_tokens or 4096,
            "stream": stream,
        }
        if system_parts:
            body["system"] = "\n\n".join(part for part in system_parts if part)
        if stop is not None:
            body["stop_sequences"] = [stop] if isinstance(stop, str) else stop

        if tools and tool_choice != "none":
            anthropic_tools: list[dict[str, Any]] = []
            for t in tools:
                fn = t.get("function", {}) if t.get("type") == "function" else t
                tool_def: dict[str, Any] = {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
                }
                anthropic_tools.append(tool_def)
            body["tools"] = anthropic_tools

            if tool_choice == "auto":
                body["tool_choice"] = {"type": "auto"}
            elif tool_choice == "required":
                body["tool_choice"] = {"type": "any"}
            elif isinstance(tool_choice, dict):
                name = (
                    tool_choice.get("function", {}).get("name")
                    if isinstance(tool_choice.get("function"), dict)
                    else tool_choice.get("name")
                )
                if name:
                    body["tool_choice"] = {"type": "tool", "name": name}

        # Sonnet 5 and Opus 5 reject non-default sampling fields; intentionally
        # do not forward temperature/top_p from the public OpenAI contract.
        return body

    @staticmethod
    def _build_xai_request_body(
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        stop: str | list[str] | None,
        response_format: dict[str, Any] | None,
        reasoning_effort: str | None,
        stream: bool,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
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
        clamped_effort = _clamp_xai_reasoning_effort(reasoning_effort)
        if clamped_effort is not None:
            body["reasoning"] = {"effort": clamped_effort}
        if response_format is not None:
            format_type = response_format.get("type")
            if format_type == "json_object":
                body["text"] = {"format": {"type": "json_object"}}
            elif format_type == "json_schema":
                body["text"] = {"format": response_format}

        if tools is not None:
            xai_tools: list[dict[str, Any]] = []
            for t in tools:
                fn = t.get("function", {}) if t.get("type") == "function" else t
                xai_tools.append(
                    {
                        "type": "function",
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                    }
                )
            body["tools"] = xai_tools

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
    def _build_openai_responses_request_body(
        *,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        stream: bool,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """OpenAI Responses proxy body for the gpt-6-astra family.

        These models reject function tools on /v1/chat/completions (Foundry
        returns HTTP 400 for tools with reasoning), while /v1/responses accepts
        tools with default reasoning. The verified working shape is
        model/input/tools/tool_choice/max_output_tokens; sampling and reasoning
        fields are intentionally omitted on this route.
        """
        body: dict[str, Any] = {
            "model": model,
            "input": _xai_input_messages(messages),
            "stream": stream,
        }
        if max_tokens is not None:
            body["max_output_tokens"] = max_tokens

        if tools is not None:
            responses_tools: list[dict[str, Any]] = []
            for t in tools:
                fn = t.get("function", {}) if t.get("type") == "function" else t
                responses_tools.append(
                    {
                        "type": "function",
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters") or {"type": "object", "properties": {}},
                    }
                )
            body["tools"] = responses_tools

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
    def _build_google_request_body(
        *,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        stop: str | list[str] | None,
        response_format: dict[str, Any] | None,
        reasoning_effort: str | None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if reasoning_effort is not None:
            raise VertexAPIError(
                400,
                "Google protocol does not expose reasoning controls.",
                code="unsupported_parameter",
            )

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
        reasoning_effort: str | None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        stream: bool,
    ) -> dict[str, Any]:
        if protocol == GOOGLE:
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
            )
        if protocol == OPENAI_CHAT:
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
        if protocol == ANTHROPIC:
            return self._build_anthropic_request_body(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stop=stop,
                stream=stream,
                tools=tools,
                tool_choice=tool_choice,
            )
        if protocol == OPENAI_RESPONSES:
            return self._build_openai_responses_request_body(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                stream=stream,
                tools=tools,
                tool_choice=tool_choice,
            )
        return self._build_xai_request_body(
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
            stream=stream,
        )

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
        reasoning_effort: str | None = None,
        resolved_config: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        parallel_tool_calls: bool | None = None,
        **_: Any,
    ):
        provider_label = self.provider_label
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
            stream=True,
        )
        if protocol == OPENAI_CHAT and _.get("_require_complete"):
            # Native Chat usage is opt-in even when public SSE usage is disabled.
            body["stream_options"] = {"include_usage": True}
        stream_context = self.http.stream(
            "POST",
            self._url_for_protocol(protocol, model=model, stream=True),
            headers=self._headers(protocol),
            json=body,
        )
        entered = False
        try:
            try:
                response = await stream_context.__aenter__()
                entered = True
            except httpx.TimeoutException as exc:
                stage = None
                if _.get("_require_complete"):
                    if isinstance(exc, httpx.ConnectTimeout):
                        stage = "provider_connect"
                    elif isinstance(exc, httpx.ReadTimeout):
                        stage = "provider_read"
                    else:
                        stage = "provider_transport"
                raise VertexAPIError(504, f"{provider_label} request timed out: {exc}", code="timeout", stage=stage) from exc
            except httpx.RequestError as exc:
                stage = None
                if _.get("_require_complete"):
                    stage = "provider_connect" if isinstance(exc, httpx.ConnectError) else "provider_transport"
                raise VertexAPIError(502, f"{provider_label} connection error: {exc}", code="connection_error", stage=stage) from exc

            if response.status_code >= 400:
                try:
                    await response.aread()
                except Exception:
                    pass
                error = self.map_http_error(response)
                if _.get("_require_complete"):
                    error.stage = "provider_http_status"
                raise error

            anthropic_tool_call_map: dict[int, int] = {}
            anthropic_input_usage: dict[str, Any] = {}
            xai_tool_call_map: dict[str, int] = {}
            google_parts: dict[str, dict[str, Any]] = {}
            google_used_ids = {str(call.get("id")) for message in messages
                               for call in message.get("tool_calls") or [] if call.get("id")}
            tool_calls_initialized: set[int] = set()
            xai_args_seen: set[int] = set()
            stream_usage: dict[str, int] | None = None
            stream_finish_reason: str | None = None
            saw_tool_calls: bool = False
            terminal_response = False

            private_stream = bool(_.get("_require_complete"))
            private_activity = _.get("_private_stream_activity") if private_stream else None

            async def response_lines():
                try:
                    async for raw_line in response.aiter_lines():
                        if callable(private_activity) and (raw_line.startswith("data:") or raw_line.startswith(":")):
                            private_activity()
                        yield raw_line
                except httpx.TimeoutException as exc:
                    if private_stream:
                        raise VertexAPIError(
                            504, f"{provider_label} response stream read timed out.", code="timeout", stage="provider_read",
                        ) from exc
                    raise
                except httpx.RequestError as exc:
                    if private_stream:
                        raise VertexAPIError(
                            502, f"{provider_label} response stream read failed.", code="connection_error", stage="provider_read",
                        ) from exc
                    raise

            async for line in response_lines():
                if not line or not line.startswith("data:"):
                    continue
                raw = line[len("data:") :].strip()
                if raw == "[DONE]":
                    if protocol == OPENAI_CHAT and stream_finish_reason in {
                        "stop", "length", "tool_calls", "content_filter", "function_call",
                    }:
                        terminal_response = True
                    break
                try:
                    event = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if not isinstance(event, dict):
                    continue

                delta_text = ""
                delta_tool_calls: list[dict[str, Any]] | None = None
                delta_thinking_blocks: list[dict[str, Any]] | None = None
                finish_reason: str | None = None
                normalized_usage: dict[str, int] | None = None
                google_native_parts = None

                if protocol == OPENAI_CHAT:
                    stream_error = _openai_error_from_payload(event)
                    if stream_error is not None and "choices" not in event:
                        message, code = stream_error
                        raise VertexAPIError(
                            502, message, code=code, raw=event,
                            stage="provider_sse_error", upstream_status=response.status_code,
                        )
                    choices = event.get("choices")
                    choice = choices[0] if isinstance(choices, list) and choices else {}
                    delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                    delta = delta if isinstance(delta, dict) else {}
                    delta_text = _extract_message_text(delta.get("content", ""))
                    raw_tool_calls = delta.get("tool_calls")
                    if isinstance(raw_tool_calls, list) and raw_tool_calls:
                        delta_tool_calls = raw_tool_calls
                        saw_tool_calls = True
                    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
                    if finish_reason is not None:
                        stream_finish_reason = finish_reason
                    usage = event.get("usage")
                    normalized_usage = _coerce_openai_usage(usage) if isinstance(usage, dict) else None
                    if _.get("_require_complete"):
                        # Private buffered rounds must not turn missing/invalid native
                        # counts into measured zero. Leave the excluded default path alone.
                        if not isinstance(usage, dict) or any(
                            not isinstance(usage.get(key), int) or isinstance(usage.get(key), bool) or usage[key] < 0
                            for key in ("prompt_tokens", "completion_tokens")
                        ) or ("total_tokens" in usage and (
                            not isinstance(usage["total_tokens"], int) or isinstance(usage["total_tokens"], bool)
                            or usage["total_tokens"] < 0
                        )):
                            normalized_usage = None
                        elif normalized_usage is not None:
                            for key in ("prompt_tokens_details", "completion_tokens_details"):
                                details = usage.get(key)
                                valid = {k: v for k, v in details.items()
                                         if isinstance(v, int) and not isinstance(v, bool) and v >= 0} if isinstance(details, dict) else {}
                                normalized_usage.pop(key, None)
                                if valid:
                                    normalized_usage[key] = valid
                    if normalized_usage is not None and finish_reason is None and stream_finish_reason is not None:
                        finish_reason = stream_finish_reason

                elif protocol == GOOGLE:
                    stream_error = _openai_error_from_payload(event)
                    if stream_error is not None and "candidates" not in event:
                        message, code = stream_error
                        raise VertexAPIError(
                            502, message, code=code, raw=event,
                            stage="provider_sse_error", upstream_status=response.status_code,
                        )
                    candidates = event.get("candidates") or []
                    candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
                    blocked = (event.get("promptFeedback") or {}).get("blockReason")
                    native_finish = candidate.get("finishReason")
                    if blocked or native_finish in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "MALFORMED_FUNCTION_CALL", "OTHER"}:
                        raise VertexAPIError(
                            502, f"{provider_label} Google stream was blocked.", code="content_filter",
                            stage="provider_sse_error", upstream_status=response.status_code,
                        )
                    texts = []
                    for position, part in enumerate((candidate.get("content") or {}).get("parts") or []):
                        if not isinstance(part, dict) or part.get("thought") is True:
                            continue
                        if part.get("text") is not None:
                            texts.append(str(part["text"]))
                        call = part.get("functionCall")
                        if not isinstance(call, dict):
                            continue
                        key = "id:" + str(call["id"]) if call.get("id") else f"slot:{position}:{call.get('name', '')}"
                        previous = google_parts.get(key, {})
                        previous_call = previous.get("functionCall", {})
                        # Google args are object snapshots, not JSON-string deltas.
                        args = {**previous_call.get("args", {}), **_google_json_object(call.get("args", {}), string_key="value")}
                        google_parts[key] = {**previous, **part, "functionCall": {**previous_call, **call, "args": args}}
                    delta_text = "".join(texts)
                    google_size = len(json.dumps(list(google_parts.values()), ensure_ascii=False).encode("utf-8"))
                    if google_size > max(65536, (max_tokens or 4096) * 64):
                        raise VertexAPIError(502, "Context compaction stream exceeded its buffer limit.", code="context_compaction_stream_limit")
                    native_usage = event.get("usageMetadata")
                    if isinstance(native_usage, dict):
                        normalized_usage = _google_usage(native_usage) if all(
                            type(native_usage.get(key)) is int and native_usage[key] >= 0
                            for key in ("promptTokenCount", "candidatesTokenCount", "totalTokenCount")
                        ) else None
                        stream_usage = normalized_usage
                        if not native_finish and not terminal_response:
                            normalized_usage = None
                    if native_finish and not terminal_response:
                        if not isinstance(native_usage, dict):
                            stream_usage = None
                        terminal_response = True
                        result = _google_response_result({"candidates": [{"content": {"parts": list(google_parts.values())},
                                                                         "finishReason": native_finish}]},
                                                         used_call_ids=google_used_ids, max_tokens=max_tokens)
                        delta_tool_calls = [{"index": index, **call} for index, call in enumerate(result["tool_calls"] or [])]
                        saw_tool_calls = bool(delta_tool_calls)
                        google_native_parts = result.get("_google_call_parts")
                        finish_reason = result["finish_reason"]
                        stream_finish_reason = finish_reason
                        normalized_usage = stream_usage

                elif protocol == ANTHROPIC:
                    event_type = event.get("type")
                    if event_type == "message_start":
                        message = event.get("message", {}) or {}
                        usage = message.get("usage", {}) or {}
                        # Input/cache counters arrive at start; its output count is
                        # provisional, not a substitute for missing terminal usage.
                        anthropic_input_usage = {key: usage[key] for key in (
                            "input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens",
                        ) if key in usage}
                    elif event_type == "content_block_start":
                        cb = event.get("content_block", {}) or {}
                        if cb.get("type") == "redacted_thinking" and isinstance(cb.get("data"), str):
                            delta_thinking_blocks = [{"type": "redacted_thinking", "data": cb["data"]}]
                        if cb.get("type") == "tool_use":
                            block_idx = event.get("index", 0)
                            tool_idx = len(anthropic_tool_call_map)
                            anthropic_tool_call_map[block_idx] = tool_idx
                            delta_tool_calls = [
                                {
                                    "index": tool_idx,
                                    "id": cb.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": cb.get("name", ""),
                                        "arguments": "",
                                    },
                                }
                            ]
                            saw_tool_calls = True
                    elif event_type == "content_block_delta":
                        delta = event.get("delta", {}) or {}
                        delta_type = delta.get("type")
                        if delta_type == "text_delta":
                            delta_text = str(delta.get("text", ""))
                        elif delta_type == "thinking_delta":
                            thinking = delta.get("thinking", "")
                            if isinstance(thinking, str) and thinking:
                                delta_thinking_blocks = [{"type": "thinking", "thinking": thinking,
                                                          "signature": ""}]
                        elif delta_type == "signature_delta":
                            signature = delta.get("signature", "")
                            if isinstance(signature, str) and signature:
                                delta_thinking_blocks = [{"type": "thinking", "thinking": "",
                                                          "signature": signature}]
                        elif delta_type == "input_json_delta":
                            block_idx = event.get("index", 0)
                            tool_idx = anthropic_tool_call_map.get(block_idx, 0)
                            partial_json = delta.get("partial_json", "")
                            delta_tool_calls = [
                                {
                                    "index": tool_idx,
                                    "function": {
                                        "arguments": partial_json,
                                    },
                                }
                            ]
                            saw_tool_calls = True
                    elif event_type == "message_delta":
                        delta = event.get("delta", {}) or {}
                        usage = event.get("usage", {}) or {}
                        native_usage = {**anthropic_input_usage, "output_tokens": usage.get("output_tokens")}
                        if all(type(native_usage.get(key)) is int and native_usage[key] >= 0
                               for key in ("input_tokens", "output_tokens")):
                            stream_usage = _anthropic_usage({"usage": native_usage})
                        else:
                            stream_usage = None
                        stop_reason = delta.get("stop_reason")
                        if stop_reason == "tool_use":
                            finish_reason = "tool_calls"
                        else:
                            finish_reason = _map_finish_reason(stop_reason)
                        if finish_reason is not None:
                            stream_finish_reason = finish_reason
                        normalized_usage = stream_usage
                    elif event_type == "message_stop":
                        terminal_response = True
                    elif event_type in {"error"}:
                        error = event.get("error", {}) or {}
                        message = error.get("message") or "Anthropic stream failed"
                        raise VertexAPIError(
                            502, str(message), code=error.get("type"), raw=event,
                            stage="provider_sse_error", upstream_status=response.status_code,
                        )

                else:
                    event_type = event.get("type")
                    if event_type == "response.output_item.added":
                        item = event.get("item", {}) or {}
                        if item.get("type") == "function_call":
                            call_id = item.get("call_id") or item.get("id") or ""
                            tool_idx = len(set(xai_tool_call_map.values()))
                            if call_id:
                                xai_tool_call_map[call_id] = tool_idx
                            if item.get("id"):
                                xai_tool_call_map[item["id"]] = tool_idx
                            if "output_index" in event:
                                xai_tool_call_map[f"idx_{event['output_index']}"] = tool_idx
                            delta_tool_calls = [
                                {
                                    "index": tool_idx,
                                    "id": call_id,
                                    "type": "function",
                                    "function": {
                                        "name": item.get("name", ""),
                                        "arguments": "",
                                    },
                                }
                            ]
                            saw_tool_calls = True
                            tool_calls_initialized.add(tool_idx)
                    elif event_type == "response.function_call_arguments.delta":
                        call_id = event.get("call_id") or event.get("id") or event.get("item_id")
                        output_index = event.get("output_index")
                        tool_idx = None
                        if call_id and call_id in xai_tool_call_map:
                            tool_idx = xai_tool_call_map[call_id]
                        elif output_index is not None and f"idx_{output_index}" in xai_tool_call_map:
                            tool_idx = xai_tool_call_map[f"idx_{output_index}"]
                        elif output_index is not None and xai_tool_call_map:
                            # Some xAI streams emit function calls without a preceding
                            # response.output_item.added; a fresh output_index marks a
                            # new tool call, so allocate the next slot for it.
                            tool_idx = len(set(xai_tool_call_map.values()))
                        elif xai_tool_call_map:
                            tool_idx = list(xai_tool_call_map.values())[-1]
                        else:
                            tool_idx = 0
                        if call_id:
                            xai_tool_call_map.setdefault(call_id, tool_idx)
                        if output_index is not None:
                            xai_tool_call_map.setdefault(f"idx_{output_index}", tool_idx)
                        delta_arg = str(event.get("delta", ""))
                        if delta_arg:
                            xai_args_seen.add(tool_idx)
                        delta_tool_calls = [
                            {
                                "index": tool_idx,
                                "function": {
                                    "arguments": delta_arg,
                                },
                            }
                        ]
                        saw_tool_calls = True
                    elif event_type == "response.output_item.done":
                        item = event.get("item", {}) or {}
                        if item.get("type") == "function_call":
                            call_id = item.get("call_id") or item.get("id") or ""
                            item_id = item.get("id") or ""
                            output_index = event.get("output_index")
                            if call_id and call_id in xai_tool_call_map:
                                tool_idx = xai_tool_call_map[call_id]
                            elif item_id and item_id in xai_tool_call_map:
                                tool_idx = xai_tool_call_map[item_id]
                            elif output_index is not None and f"idx_{output_index}" in xai_tool_call_map:
                                tool_idx = xai_tool_call_map[f"idx_{output_index}"]
                            else:
                                tool_idx = len(set(xai_tool_call_map.values()))
                            if call_id:
                                xai_tool_call_map.setdefault(call_id, tool_idx)
                            if item_id:
                                xai_tool_call_map.setdefault(item_id, tool_idx)
                            if tool_idx in tool_calls_initialized and tool_idx not in xai_args_seen and item.get("arguments"):
                                delta_tool_calls = [{"index": tool_idx, "function": {"arguments": str(item["arguments"])}}]
                                xai_args_seen.add(tool_idx)
                            if tool_idx not in tool_calls_initialized:
                                # The xAI route reveals id/name only on the completed
                                # item; emit them once so streaming consumers can
                                # assemble a valid tool call. Include the arguments
                                # only when none were streamed as deltas.
                                arguments = "" if tool_idx in xai_args_seen else str(item.get("arguments") or "")
                                delta_tool_calls = [
                                    {
                                        "index": tool_idx,
                                        "id": call_id,
                                        "type": "function",
                                        "function": {
                                            "name": item.get("name", ""),
                                            "arguments": arguments,
                                        },
                                    }
                                ]
                                tool_calls_initialized.add(tool_idx)
                                saw_tool_calls = True
                    elif event_type == "response.output_text.delta":
                        delta_text = str(event.get("delta", ""))
                    elif event_type in {"response.completed", "response.incomplete"}:
                        terminal_response = True
                        completed = event.get("response", {}) or {}
                        normalized_usage = _xai_usage(completed) if isinstance(completed.get("usage"), dict) else None
                        if normalized_usage is not None and protocol in {OPENAI_RESPONSES, XAI_RESPONSES}:
                            native_usage = completed["usage"]
                            if not all(type(native_usage.get(key)) is int and native_usage[key] >= 0
                                       for key in ("input_tokens", "output_tokens")):
                                normalized_usage = None
                            else:
                                details = native_usage.get("input_tokens_details")
                                if not (isinstance(details, dict) and type(details.get("cached_tokens")) is int
                                        and details["cached_tokens"] >= 0):
                                    details = None
                                normalized_usage = _coerce_openai_usage({
                                    **normalized_usage, "prompt_tokens_details": details,
                                })
                        stream_usage = normalized_usage
                        if saw_tool_calls:
                            finish_reason = "tool_calls"
                        else:
                            status = completed.get("status")
                            finish_reason = "stop" if status == "completed" else None
                            incomplete = completed.get("incomplete_details")
                            if isinstance(incomplete, dict):
                                finish_reason = _map_finish_reason(incomplete.get("reason")) or finish_reason
                            finish_reason = finish_reason or "stop"
                        stream_finish_reason = finish_reason
                    elif event_type in {"response.failed", "error"}:
                        error = event.get("error") or (event.get("response") or {}).get("error") or {}
                        message = error.get("message") or "xAI Responses stream failed"
                        raise VertexAPIError(
                            502, str(message), code=error.get("code"), raw=event,
                            stage="provider_sse_error", upstream_status=response.status_code,
                        )

                if (delta_text or delta_tool_calls or delta_thinking_blocks
                        or finish_reason is not None or normalized_usage is not None):
                    event_dict: dict[str, Any] = {
                        "delta_text": delta_text,
                        "finish_reason": finish_reason,
                        "usage": normalized_usage,
                    }
                    if delta_tool_calls:
                        event_dict["delta_tool_calls"] = delta_tool_calls
                    if delta_thinking_blocks:
                        event_dict["thinking_blocks"] = delta_thinking_blocks
                    if google_native_parts:
                        event_dict["_google_call_parts"] = google_native_parts
                    yield event_dict

            if _.get("_require_complete") and not terminal_response:
                raise VertexAPIError(502, f"{provider_label} stream ended before completion.", code="incomplete_stream")

            if (
                protocol
                in {
                    OPENAI_CHAT,
                    ANTHROPIC,
                    XAI_RESPONSES,
                    OPENAI_RESPONSES,
                    GOOGLE,
                }
                and saw_tool_calls
                and stream_finish_reason is None
            ):
                yield {
                    "delta_text": "",
                    "delta_tool_calls": None,
                    "finish_reason": "tool_calls",
                    "usage": stream_usage,
                }
        finally:
            if entered:
                # HTTP EOF alone does not prove protocol completion: propagate
                # parser/terminal failures so metering retains unknown usage.
                await stream_context.__aexit__(*sys.exc_info())

    @staticmethod
    def _parse_protocol_response(payload, protocol, messages, max_tokens):
        if protocol == GOOGLE:
            used_call_ids = {
                str(call["id"]) for message in messages
                for call in message.get("tool_calls") or []
                if isinstance(call, dict) and call.get("id")
            }
            return _google_response_result(payload, used_call_ids=used_call_ids, max_tokens=max_tokens)

        if protocol == OPENAI_CHAT:
            choices = payload.get("choices") if isinstance(payload, dict) else None
            if not isinstance(choices, list) or not choices:
                raise VertexAPIError(502, "Malformed chat response: missing choices[]", code="bad_gateway")
            choice = choices[0] or {}
            message = choice.get("message", {}) or {}
            tool_calls = message.get("tool_calls")
            content = message.get("content")
            if tool_calls:
                extracted = _extract_message_text(content) if content is not None else ""
                text = extracted if extracted else None
                finish_reason = choice.get("finish_reason") or "tool_calls"
            else:
                text = _extract_message_text(content or "")
                finish_reason = choice.get("finish_reason") or "stop"
            return {
                "text": text,
                "tool_calls": tool_calls if tool_calls else None,
                "finish_reason": finish_reason,
                "usage": _coerce_openai_usage(payload.get("usage") if isinstance(payload, dict) else None),
            }

        if protocol == ANTHROPIC:
            content = payload.get("content") if isinstance(payload, dict) else None
            if not isinstance(content, list):
                raise VertexAPIError(502, "Malformed Anthropic response: missing content[]", code="bad_gateway")
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    text_parts.append(str(block.get("text", "")))
                elif block_type == "tool_use":
                    raw_input = block.get("input")
                    if isinstance(raw_input, (dict, list)):
                        args_str = json.dumps(raw_input, ensure_ascii=False)
                    elif isinstance(raw_input, str):
                        args_str = raw_input
                    else:
                        args_str = "{}"
                    tool_calls.append(
                        {
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": args_str,
                            },
                        }
                    )
            stop_reason = payload.get("stop_reason") if isinstance(payload, dict) else None
            if stop_reason == "tool_use" or (tool_calls and stop_reason is None):
                finish_reason = "tool_calls"
            else:
                finish_reason = _map_finish_reason(stop_reason) or "stop"

            extracted_text = "".join(text_parts)
            if tool_calls:
                text = extracted_text if extracted_text else None
            else:
                text = extracted_text

            return {
                "text": text,
                "tool_calls": tool_calls if tool_calls else None,
                "finish_reason": finish_reason,
                "usage": _anthropic_usage(payload),
            }

        output_list = payload.get("output") if isinstance(payload, dict) else None
        if not isinstance(output_list, list):
            raise VertexAPIError(502, "Malformed xAI response: missing output[]", code="bad_gateway")

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for item in output_list:
            if not isinstance(item, dict):
                continue
            itype = item.get("type")
            if itype == "message":
                for c in item.get("content", []) or []:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        text_parts.append(str(c.get("text", "")))
            elif itype == "function_call":
                raw_args = item.get("arguments")
                if not raw_args:
                    args_str = "{}"
                elif isinstance(raw_args, str):
                    args_str = raw_args
                elif isinstance(raw_args, (dict, list)):
                    args_str = json.dumps(raw_args, ensure_ascii=False)
                else:
                    args_str = "{}"
                call_id = item.get("call_id") or item.get("id") or ""
                tool_calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": args_str,
                        },
                    }
                )

        extracted_text = "".join(text_parts)
        if tool_calls:
            text = extracted_text if extracted_text else None
            finish_reason = "tool_calls"
        else:
            text = extracted_text
            status = payload.get("status") if isinstance(payload, dict) else None
            incomplete = payload.get("incomplete_details") if isinstance(payload, dict) else None
            finish_reason = "stop" if status == "completed" else None
            if isinstance(incomplete, dict):
                finish_reason = _map_finish_reason(incomplete.get("reason")) or finish_reason
            finish_reason = finish_reason or "stop"

        usage = _xai_usage(payload)
        if protocol in {OPENAI_RESPONSES, XAI_RESPONSES}:
            # Both Responses protocols use input_tokens_details, not the Chat Completions key.
            # Preserve returned cache evidence for aggregation across private rounds.
            native_usage = payload.get("usage") or {}
            details = native_usage.get("input_tokens_details")
            if protocol == XAI_RESPONSES:
                cached = details.get("cached_tokens") if isinstance(details, dict) else None
                details = {"cached_tokens": cached} if (
                    isinstance(cached, int) and not isinstance(cached, bool) and cached >= 0
                ) else None
            usage = _coerce_openai_usage({
                **usage,
                "prompt_tokens_details": details,
            })
        return {
            "text": text,
            "tool_calls": tool_calls if tool_calls else None,
            "finish_reason": finish_reason,
            "usage": usage,
        }
