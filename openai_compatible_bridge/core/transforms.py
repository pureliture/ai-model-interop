"""Shared transform functions for OpenAI/Anthropic/Google API conversions.

Extracted from providers/foundry.py so that multiple providers (Foundry, JetBrains, etc.)
can reuse the same transform logic without code duplication.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from openai_compatible_bridge.core.cost_tracking import NormalizedUsage
from openai_compatible_bridge.core.images import data_url, normalize_content
from openai_compatible_bridge.providers.vertex import (
    VertexAPIError,
    _extract_message_text,
    _apply_response_format,
)


def _unwrap_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.startswith("Optional[") and text.endswith("]"):
        text = text[len("Optional[") : -1].strip()
    return text or None


def _openai_error_from_payload(payload: Any) -> tuple[str, str | None] | None:
    if not isinstance(payload, dict):
        return None

    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        code = error.get("code") or error.get("type")
        if message:
            return str(message), str(code) if code is not None else None

    parameters = payload.get("parameters")
    if isinstance(parameters, dict):
        response_body = _unwrap_optional(parameters.get("responseBody"))
        if response_body:
            try:
                nested = json.loads(response_body)
            except (TypeError, ValueError):
                nested = None
            parsed = _openai_error_from_payload(nested)
            if parsed is not None:
                return parsed

        message = _unwrap_optional(parameters.get("errorMessage"))
        code = _unwrap_optional(parameters.get("errorCode"))
        if message:
            return message, code

    return None




def _usage(prompt_tokens: Any, completion_tokens: Any, total_tokens: Any = None) -> dict[str, int]:
    def as_int(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    prompt = as_int(prompt_tokens)
    completion = as_int(completion_tokens)
    total = as_int(total_tokens) or prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _anthropic_usage(payload: Any) -> dict[str, int]:
    usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
    normalized = _usage(usage.get("input_tokens"), usage.get("output_tokens"), usage.get("total_tokens"))
    # Preserve native cache counters for per-call accounting and private-loop sums.
    # Missing counters must stay missing rather than becoming measured zeroes.
    for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
        value = usage.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            normalized[key] = value
    return normalized


def _xai_usage(payload: Any) -> dict[str, int]:
    usage = payload.get("usage", {}) if isinstance(payload, dict) else {}
    return _usage(usage.get("input_tokens"), usage.get("output_tokens"), usage.get("total_tokens"))


def _map_finish_reason(reason: Any) -> str | None:
    if reason is None:
        return None
    value = str(reason)
    return {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "completed": "stop",
        "STOP": "stop",
        "max_tokens": "length",
        "max_output_tokens": "length",
        "MAX_TOKENS": "length",
        "length": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "BLOCKLIST": "content_filter",
        "PROHIBITED_CONTENT": "content_filter",
        "SPII": "content_filter",
        "MALFORMED_FUNCTION_CALL": "tool_calls",
    }.get(value, value)


_XAI_SUPPORTED_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")


def _clamp_xai_reasoning_effort(effort: Any) -> str | None:
    """Map Hermes-style effort values onto Foundry xAI's accepted set.

    Live probing on foundry:grok-4.6 shows minimal|low|medium|high|xhigh are
    accepted while 'none' and 'max'/'ultra' are rejected with HTTP 400. This
    clamps unsupported values instead of forwarding a deterministic 400.
    """
    if effort is None:
        return None
    value = str(effort).strip().lower()
    if not value:
        return None
    if value in _XAI_SUPPORTED_EFFORTS:
        return value
    if value in {"max", "ultra"}:
        return "xhigh"
    # 'none' (and unknown values) cannot be represented on this route: omitting
    # the field lets the server default apply instead of a rejected value.
    return None


def _xai_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    parts: list[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") == "output_text":
                parts.append(str(content.get("text", "")))
    return "".join(parts)


def _xai_input_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transform message history into xAI Responses input items sequence."""
    input_items: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")
        tool_calls = message.get("tool_calls")

        if role == "tool":
            content_val = content
            if isinstance(content_val, (dict, list)):
                out_str = json.dumps(content_val, ensure_ascii=False)
            else:
                out_str = str(content_val if content_val is not None else "")
            call_id = message.get("tool_call_id") or message.get("id") or ""
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": out_str,
                }
            )
        elif role == "assistant":
            if content:
                text = _extract_message_text(content)
                if text:
                    input_items.append({"role": "assistant", "content": text})
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    raw_args = fn.get("arguments")
                    if not raw_args:
                        args_str = "{}"
                    elif isinstance(raw_args, str):
                        args_str = raw_args
                    else:
                        args_str = json.dumps(raw_args, ensure_ascii=False)
                    call_id = tc.get("id") or tc.get("call_id") or ""
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": call_id,
                            "name": fn.get("name", ""),
                            "arguments": args_str,
                        }
                    )
            elif not content:
                input_items.append({"role": "assistant", "content": ""})
        elif role in {"system", "developer", "user"}:
            blocks = normalize_content(content)
            if any(block["type"] == "image" for block in blocks):
                if role != "user":
                    raise VertexAPIError(
                        400, "Images are supported only in user messages.",
                        code="invalid_request_error", stage="messages",
                    )
                parts: list[dict[str, Any]] = []
                for block in blocks:
                    if block["type"] == "text":
                        parts.append({"type": "input_text", "text": block["text"]})
                    else:
                        parts.append({"type": "input_image", "detail": "auto",
                                      "image_url": data_url(block)})
                input_items.append({"role": role, "content": parts})
            else:
                text = "".join(block["text"] for block in blocks if block["type"] == "text")
                input_items.append({"role": role, "content": text})
        else:
            text = _extract_message_text(content) if content is not None else ""
            input_items.append({"role": role, "content": text})
    return input_items


_openai_input_messages = _xai_input_messages



def _google_json_object(value: Any, *, string_key: str = "content") -> dict[str, Any]:
    """Convert an OpenAI tool payload into a Google object value."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return {string_key: value}
        if isinstance(parsed, dict):
            return parsed
        return {string_key: parsed}
    if value is None:
        return {}
    return {string_key: value}


def _google_function_call_part(tool_call: dict[str, Any]) -> tuple[dict[str, Any], str]:
    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    function = function if isinstance(function, dict) else {}
    name = str(function.get("name", ""))
    args = _google_json_object(function.get("arguments", "{}"), string_key="value")
    native = tool_call.get("_google_part")
    if isinstance(native, dict) and isinstance(native.get("functionCall"), dict):
        return native, name
    part: dict[str, Any] = {"functionCall": {"name": name, "args": args}}
    extra = tool_call.get("extra_content") if isinstance(tool_call, dict) else None
    google = extra.get("google") if isinstance(extra, dict) else None
    signature = google.get("thought_signature") if isinstance(google, dict) else None
    if isinstance(signature, str) and signature:
        part["thoughtSignature"] = signature
    return part, name


def _google_contents_and_system(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Map OpenAI messages to Google GenerateContent contents/systemInstruction."""
    contents: list[dict[str, Any]] = []
    system_parts: list[dict[str, str]] = []
    tool_names: dict[str, str] = {}
    native_ids: dict[str, str] = {}

    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content")
        if role == "system":
            blocks = normalize_content(content)
            if any(block["type"] == "image" for block in blocks):
                raise VertexAPIError(
                    400, "Images are supported only in user messages.",
                    code="invalid_request_error", stage="messages",
                )
            text = "".join(block["text"] for block in blocks if block["type"] == "text")
            if text:
                system_parts.append({"text": text})
            continue

        if role == "assistant":
            blocks = normalize_content(content)
            if any(block["type"] == "image" for block in blocks):
                raise VertexAPIError(
                    400, "Images are supported only in user messages.",
                    code="invalid_request_error", stage="messages",
                )
            parts: list[dict[str, Any]] = []
            text = "".join(block["text"] for block in blocks if block["type"] == "text")
            if text:
                parts.append({"text": text})
            for tool_call in message.get("tool_calls") or []:
                part, name = _google_function_call_part(tool_call)
                parts.append(part)
                call_id = tool_call.get("id") if isinstance(tool_call, dict) else None
                if call_id:
                    tool_names[str(call_id)] = name
                    native_id = part["functionCall"].get("id")
                    if native_id:
                        native_ids[str(call_id)] = native_id
            if not parts:
                parts.append({"text": ""})
            contents.append({"role": "model", "parts": parts})
            continue

        if role == "tool":
            call_id = str(message.get("tool_call_id") or message.get("id") or "")
            name = str(message.get("name") or tool_names.get(call_id) or call_id)
            response = _google_json_object(content, string_key="content")
            function_response = {"name": name, "response": response}
            if call_id in native_ids:
                function_response["id"] = native_ids[call_id]
            part = {"functionResponse": function_response}
            if contents and contents[-1]["role"] == "user" and all(
                "functionResponse" in previous for previous in contents[-1]["parts"]
            ):
                contents[-1]["parts"].append(part)
            else:
                contents.append({"role": "user", "parts": [part]})
            continue

        blocks = normalize_content(content)
        parts: list[dict[str, Any]] = []
        for block in blocks:
            if block["type"] == "text":
                parts.append({"text": block["text"]})
            else:
                parts.append({"inlineData": {"mimeType": block["mime_type"], "data": block["data"]}})
        contents.append({"role": "user", "parts": parts or [{"text": ""}]})

    if not contents:
        contents.append({"role": "user", "parts": [{"text": ""}]})
    return contents, system_parts


_GOOGLE_SCHEMA_FIELDS = (
    "type", "format", "title", "description", "nullable", "enum", "minimum", "maximum",
    "minItems", "maxItems", "minLength", "maxLength", "minProperties", "maxProperties",
    "pattern", "propertyOrdering", "required",
)


def _google_schema(value: Any) -> dict[str, Any]:
    """Project JSON Schema onto Google's Schema proto without unsupported fields."""
    if not isinstance(value, dict):
        return {"type": "string"}
    out = {key: value[key] for key in _GOOGLE_SCHEMA_FIELDS if key in value}
    schema_type = out.get("type")
    if isinstance(schema_type, list):
        non_null = [item for item in schema_type if isinstance(item, str) and item != "null"]
        if len(non_null) == 1:
            out["type"] = non_null[0]
            if "null" in schema_type:
                out["nullable"] = True
        else:
            out.pop("type", None)
            out["anyOf"] = [_google_schema({"type": item}) for item in non_null] or [{"type": "string"}]
    if "required" in out:
        properties = value.get("properties") if isinstance(value.get("properties"), dict) else {}
        out["required"] = [name for name in out["required"]
                           if isinstance(name, str) and name in properties]
    if isinstance(value.get("properties"), dict):
        out["properties"] = {name: _google_schema(spec) for name, spec in value["properties"].items()}
    if isinstance(value.get("items"), dict):
        out["items"] = _google_schema(value["items"])
    if isinstance(value.get("anyOf"), list):
        out["anyOf"] = [_google_schema(item) for item in value["anyOf"]]
    return out


def _google_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    declarations: list[dict[str, Any]] = []
    for tool in tools:
        function = tool.get("function", {}) if isinstance(tool, dict) else {}
        function = function if isinstance(function, dict) else {}
        name = str(function.get("name", ""))
        if not name:
            continue
        declaration: dict[str, Any] = {"name": name}
        description = function.get("description")
        if description:
            declaration["description"] = str(description)
        parameters = function.get("parameters")
        if parameters is not None:
            declaration["parameters"] = _google_schema(parameters)
        declarations.append(declaration)
    return [{"functionDeclarations": declarations}] if declarations else None


def _google_tool_config(tool_choice: str | dict[str, Any] | None) -> dict[str, Any] | None:
    if tool_choice is None:
        return None
    mode = "AUTO"
    allowed: list[str] | None = None
    if tool_choice == "none":
        mode = "NONE"
    elif tool_choice == "required":
        mode = "ANY"
    elif isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        name = function.get("name") if isinstance(function, dict) else tool_choice.get("name")
        mode = "ANY"
        if name:
            allowed = [str(name)]
    config: dict[str, Any] = {"mode": mode}
    if allowed:
        config["allowedFunctionNames"] = allowed
    return {"functionCallingConfig": config}


def _google_usage(usage: Any) -> dict[str, Any]:
    usage = usage if isinstance(usage, dict) else {}

    def as_int(value: Any) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    prompt = as_int(usage.get("promptTokenCount"))
    completion = as_int(usage.get("candidatesTokenCount"))
    total = as_int(usage.get("totalTokenCount")) or prompt + completion
    normalized: dict[str, Any] = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}
    cached = usage.get("cachedContentTokenCount")
    if isinstance(cached, int) and not isinstance(cached, bool) and cached >= 0:
        normalized["prompt_tokens_details"] = {"cached_tokens": cached}
    return normalized


def _google_response_result(
    payload: Any, *, used_call_ids: set[str] | None = None, max_tokens: int | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise VertexAPIError(502, "Malformed Google response: expected object", code="bad_gateway")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise VertexAPIError(502, "Malformed Google response: missing candidates[]", code="bad_gateway")

    candidate = candidates[0] if isinstance(candidates[0], dict) else {}
    content = candidate.get("content", {}) if isinstance(candidate.get("content", {}), dict) else {}
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    native_parts: dict[str, dict[str, Any]] = {}
    native_bytes = 2  # Serialized mapping braces.
    # Use the same token-derived byte budget as the native stream collector.
    native_limit = max(65536, (max_tokens or 4096) * 64)
    used = set(used_call_ids or ())
    for part_index, part in enumerate(content.get("parts", []) or []):
        if not isinstance(part, dict) or part.get("thought") is True:
            continue
        text = part.get("text")
        if text is not None:
            text_parts.append(str(text))
        function_call = part.get("functionCall")
        if isinstance(function_call, dict):
            name = str(function_call.get("name", ""))
            args = _google_json_object(function_call.get("args", {}), string_key="value")
            call_id = str(function_call.get("id") or f"call_{name}_{part_index}")
            if not function_call.get("id"):
                base_id = call_id
                suffix = 1
                while call_id in used:
                    call_id = f"{base_id}_{suffix}"
                    suffix += 1
            used.add(call_id)
            if "id" in function_call or "thoughtSignature" in part:
                native_bytes += len(json.dumps({call_id: part}, ensure_ascii=False).encode("utf-8")) - 2
                if native_parts:
                    native_bytes += 2  # Mapping entry separator.
                if native_bytes > native_limit:
                    raise VertexAPIError(
                        502, "Context compaction native parts exceeded their buffer limit.",
                        code="context_compaction_native_limit", stage="collector_buffer",
                    )
                native_parts[call_id] = part
            tool_call = {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
            }
            signature = part.get("thoughtSignature")
            if isinstance(signature, str) and signature:
                tool_call["extra_content"] = {"google": {"thought_signature": signature}}
            tool_calls.append(tool_call)

    if tool_calls:
        finish_reason = "tool_calls"
        text: str | None = "".join(text_parts) or None
    else:
        finish_reason = _map_finish_reason(candidate.get("finishReason") or "STOP")
        text = "".join(text_parts)

    return {
        "text": text,
        "tool_calls": tool_calls if tool_calls else None,
        "finish_reason": finish_reason,
        "usage": _google_usage(payload.get("usageMetadata")),
        **({"_google_call_parts": native_parts} if native_parts else {}),
    }


def _append_anthropic_turn(chat_messages: list[dict[str, Any]], role: str, content: Any) -> None:
    if not chat_messages:
        chat_messages.append({"role": role, "content": content})
        return

    if chat_messages[-1]["role"] != role:
        chat_messages.append({"role": role, "content": content})
        return

    prev_content = chat_messages[-1]["content"]
    if isinstance(prev_content, list) and isinstance(content, list):
        prev_content.extend(content)
    elif isinstance(prev_content, list) and not isinstance(content, list):
        text = _extract_message_text(content)
        if text:
            prev_content.append({"type": "text", "text": text})
    elif not isinstance(prev_content, list) and isinstance(content, list):
        prev_text = _extract_message_text(prev_content)
        new_blocks: list[dict[str, Any]] = []
        if prev_text:
            new_blocks.append({"type": "text", "text": prev_text})
        new_blocks.extend(content)
        chat_messages[-1]["content"] = new_blocks
    else:
        prev_text = _extract_message_text(prev_content)
        curr_text = _extract_message_text(content)
        if prev_text and curr_text:
            chat_messages[-1]["content"] = f"{prev_text}\n\n{curr_text}"
        elif curr_text:
            chat_messages[-1]["content"] = curr_text
        else:
            chat_messages[-1]["content"] = prev_text


