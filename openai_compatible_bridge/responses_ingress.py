"""OpenAI Responses API ingress/egress (POST /v1/responses) translation layer.

Translates Codex CLI and OpenAI Responses API requests, freeform custom tools
(such as `apply_patch`), non-streaming responses, and SSE streams to and from
the bridge's internal OpenAI-shape chat pipeline.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from openai_compatible_bridge.core.conversation_state import ConversationStateCache
from openai_compatible_bridge.core.transforms import STANDARD_REASONING_EFFORTS


class ResponsesBadRequest(Exception):
    def __init__(
        self,
        message: str,
        *,
        param: str | None = None,
        code: str = "invalid_request_error",
    ) -> None:
        super().__init__(message)
        self.message = message
        self.param = param
        self.code = code


def new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex[:24]}"


def responses_sse_event(event_type: str, payload: dict[str, Any]) -> str:
    body = {"type": event_type, **payload}
    return f"event: {event_type}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"


def _text_value(text: Any, *, index: int) -> str:
    if isinstance(text, str):
        return text
    if text is None:
        return ""
    try:
        return json.dumps(text, ensure_ascii=False)
    except (TypeError, ValueError):
        raise ResponsesBadRequest(
            f"Message content block[{index}] has a text value that cannot be represented.",
            param="input",
        )


def _extract_responses_content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for index, block in enumerate(content):
            if isinstance(block, str):
                parts.append(block)
                continue
            if not isinstance(block, dict):
                raise ResponsesBadRequest(
                    f"Message content block[{index}] must be an object or string.",
                    param="input",
                )
            block_type = block.get("type")
            if block_type in ("input_text", "output_text", "text", None):
                parts.append(_text_value(block.get("text", ""), index=index))
            elif block_type in ("reasoning", "thinking", "redacted_thinking", "summary_text"):
                continue
            else:
                raise ResponsesBadRequest(
                    f"Unsupported content block type in Responses input: {block_type!r}.",
                    param="input",
                )
        return "".join(parts)
    if isinstance(content, dict):
        if "text" in content:
            return _text_value(content["text"], index=0)
        return json.dumps(content, ensure_ascii=False)
    return _text_value(content, index=0)


def _extract_tool_output_text(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item and isinstance(item["text"], str):
                    parts.append(item["text"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(output, dict):
        return json.dumps(output, ensure_ascii=False)
    return str(output)


def _stringify_input_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return None


def extract_custom_tool_input(raw_arguments: Any) -> str | None:
    """Extract raw freeform string input from a wrapped custom tool JSON arguments payload."""
    if raw_arguments is None:
        return None
    if isinstance(raw_arguments, dict):
        if "input" in raw_arguments:
            return _stringify_input_value(raw_arguments.get("input"))
        return json.dumps(raw_arguments, ensure_ascii=False)
    if isinstance(raw_arguments, str):
        stripped = raw_arguments.strip()
        if not stripped:
            return ""
        try:
            parsed = json.loads(stripped)
        except (TypeError, ValueError):
            return raw_arguments
        if isinstance(parsed, dict) and "input" in parsed:
            return _stringify_input_value(parsed.get("input"))
        if isinstance(parsed, str):
            return parsed
        return json.dumps(parsed, ensure_ascii=False)
    stringified = _stringify_input_value(raw_arguments)
    return raw_arguments if stringified is None else stringified


def responses_input_to_messages(
    input_value: Any,
    *,
    instructions: str | None = None,
    state_cache: ConversationStateCache | None = None,
) -> tuple[list[dict[str, Any]], set[str]]:
    """Normalize Responses `instructions` and `input` into internal OpenAI-shape messages."""
    messages: list[dict[str, Any]] = []
    inferred_custom_tools: set[str] = set()
    system_parts: list[str] = []
    saw_conversation_turn = False

    if instructions is not None:
        if not isinstance(instructions, str):
            raise ResponsesBadRequest("instructions must be a string.", param="instructions")
        if instructions.strip():
            system_parts.append(instructions)

    if isinstance(input_value, str):
        items: list[Any] = [{"type": "message", "role": "user", "content": input_value}]
    elif isinstance(input_value, list):
        items = input_value
    else:
        raise ResponsesBadRequest(
            "input must be a string or an array of input items.",
            param="input",
        )

    for index, item in enumerate(items):
        if isinstance(item, str):
            saw_conversation_turn = True
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            raise ResponsesBadRequest(f"input[{index}] must be an object or string.", param="input")

        item_type = item.get("type")
        role = item.get("role")

        if item_type == "item_reference":
            ref_id = item.get("id")
            if not isinstance(ref_id, str) or not ref_id.strip():
                raise ResponsesBadRequest(
                    f"input[{index}] (item_reference) requires a non-empty id.",
                    param="input",
                )
            if state_cache is None:
                raise ResponsesBadRequest(
                    f"input[{index}] (item_reference) cannot be resolved without a conversation state cache.",
                    param="input",
                )
            resolved = state_cache.get_response(ref_id.strip())
            if not isinstance(resolved, dict) or not isinstance(resolved.get("messages"), list):
                raise ResponsesBadRequest(
                    f"input[{index}] references unknown response '{ref_id.strip()}'.",
                    param="input",
                )
            saw_conversation_turn = True
            messages.extend(resolved["messages"])
            continue

        if item_type == "reasoning":
            continue

        if item_type in ("function_call", "custom_tool_call"):
            saw_conversation_turn = True
            call_id = str(item.get("call_id") or item.get("id") or "")
            name = str(item.get("name") or "")
            if not name:
                raise ResponsesBadRequest(
                    f"input[{index}] ({item_type}) requires a non-empty name.",
                    param="input",
                )
            if not call_id:
                call_id = f"call_{name}_{index}"
            if item_type == "custom_tool_call":
                inferred_custom_tools.add(name)
                raw_input = item.get("input", "")
                args_str = json.dumps(
                    {"input": "" if raw_input is None else str(raw_input)},
                    ensure_ascii=False,
                )
            else:
                raw_args = item.get("arguments", "{}")
                if isinstance(raw_args, str):
                    args_str = raw_args if raw_args.strip() else "{}"
                else:
                    try:
                        args_str = json.dumps(raw_args, ensure_ascii=False)
                    except (TypeError, ValueError):
                        raise ResponsesBadRequest(
                            f"input[{index}] (function_call) has arguments that cannot be represented as JSON.",
                            param="input",
                        )
            tc_entry = {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": args_str},
            }
            if messages and messages[-1].get("role") == "assistant":
                messages[-1].setdefault("tool_calls", []).append(tc_entry)
            else:
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [tc_entry],
                    }
                )
            continue

        if item_type in ("function_call_output", "custom_tool_call_output"):
            saw_conversation_turn = True
            call_id = str(item.get("call_id") or item.get("id") or "")
            if not call_id:
                raise ResponsesBadRequest(
                    f"input[{index}] ({item_type}) requires call_id.",
                    param="input",
                )
            tool_msg: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": call_id,
                "content": _extract_tool_output_text(item.get("output", "")),
            }
            if item.get("is_error") is True:
                tool_msg["is_error"] = True
            messages.append(tool_msg)
            continue

        if item_type in (None, "message"):
            effective_role = str(role or "user")
            if effective_role not in ("user", "assistant", "system", "developer"):
                raise ResponsesBadRequest(
                    f"Unsupported message role in input[{index}]: {effective_role!r}.",
                    param=f"input[{index}].role",
                )
            extra = set(item) - {"type", "role", "content"}
            if extra:
                field = sorted(extra)[0]
                raise ResponsesBadRequest(
                    f"input[{index}].{field} cannot be represented in the chat protocol.",
                    param=f"input[{index}].{field}",
                )
            content = item.get("content")
            if isinstance(content, list):
                mapped = []
                has_image = False
                for block_index, block in enumerate(content):
                    if isinstance(block, dict) and block.get("type") == "input_image":
                        path = f"input[{index}].content[{block_index}]"
                        extra = block.keys() - {"type", "image_url", "detail", "file_id"}
                        if extra:
                            field = sorted(extra)[0]
                            raise ResponsesBadRequest(f"{path}.{field} cannot be represented in chat protocol.", param=f"{path}.{field}")
                        image = {"type": "image_url", "image_url": {"url": block.get("image_url")}}
                        if "detail" in block:
                            image["image_url"]["detail"] = block["detail"]
                        if "file_id" in block:
                            image["file_id"] = block["file_id"]
                        mapped.append(image)
                        has_image = True
                    elif isinstance(block, dict) and block.get("type") in {"input_text", "output_text", "text"}:
                        extra = block.keys() - {"type", "text"}
                        if extra:
                            field = sorted(extra)[0]
                            path = f"input[{index}].content[{block_index}].{field}"
                            raise ResponsesBadRequest(f"{path} cannot be represented in chat protocol.", param=path)
                        mapped.append({"type": "text", "text": block.get("text", "")})
                    else:
                        mapped.append(block)
                content = mapped if has_image else _extract_responses_content_text(content)
            else:
                content = _extract_responses_content_text(content)
            saw_conversation_turn = True
            messages.append({"role": effective_role, "content": content})
            continue

        raise ResponsesBadRequest(
            f"Unsupported input item type at input[{index}]: {item_type!r}.",
            param="input",
        )

    if system_parts:
        messages = [{"role": "system", "content": "\n\n".join(system_parts)}, *messages]

    return messages, inferred_custom_tools


def responses_tools_to_openai(tools: Any) -> tuple[list[dict[str, Any]] | None, set[str]]:
    """Normalize flat/nested function tools and wrap freeform custom tools (`apply_patch`)."""
    if tools is None:
        return None, set()
    if not isinstance(tools, list):
        raise ResponsesBadRequest("tools must be an array.", param="tools")
    if not tools:
        return None, set()

    openai_tools: list[dict[str, Any]] = []
    custom_tool_names: set[str] = set()
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise ResponsesBadRequest(f"tools[{index}] must be an object.", param="tools")
        tool_type = tool.get("type", "function")
        if tool_type == "function":
            fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            name = fn.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ResponsesBadRequest(
                    f"tools[{index}].name must be a non-empty string.",
                    param="tools",
                )
            parameters = fn.get("parameters")
            if parameters is not None and not isinstance(parameters, dict):
                raise ResponsesBadRequest(
                    f"tools[{index}].parameters must be a JSON Schema object.",
                    param="tools",
                )
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name.strip(),
                        "description": str(fn.get("description") or ""),
                        "parameters": parameters or {"type": "object", "properties": {}},
                    },
                }
            )
        elif tool_type == "custom":
            name = tool.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ResponsesBadRequest(
                    f"tools[{index}].name must be a non-empty string.",
                    param="tools",
                )
            clean_name = name.strip()
            custom_tool_names.add(clean_name)
            description = str(
                tool.get("description")
                or f"Freeform custom tool '{clean_name}'. Provide the raw text payload in the 'input' string argument."
            )
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": clean_name,
                        "description": description,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {
                                    "type": "string",
                                    "description": f"Raw freeform text input for {clean_name}.",
                                }
                            },
                            "required": ["input"],
                            "additionalProperties": False,
                        },
                    },
                }
            )
        else:
            raise ResponsesBadRequest(
                f"Unsupported tool type in tools[{index}]: {tool_type!r}.",
                param="tools",
            )

    return openai_tools, custom_tool_names


def responses_tool_choice_to_openai(tool_choice: Any) -> str | dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        if tool_choice in ("auto", "none", "required"):
            return tool_choice
        raise ResponsesBadRequest(
            f"Unsupported tool_choice value: {tool_choice!r}.",
            param="tool_choice",
        )
    if isinstance(tool_choice, dict):
        choice_type = tool_choice.get("type")
        if choice_type in ("auto", "none", "required"):
            return str(choice_type)
        if choice_type in ("function", "custom", "tool"):
            fn_obj = tool_choice.get("function")
            name = fn_obj.get("name") if isinstance(fn_obj, dict) else tool_choice.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ResponsesBadRequest(
                    f"tool_choice of type {choice_type!r} requires a non-empty name.",
                    param="tool_choice",
                )
            return {"type": "function", "function": {"name": name.strip()}}
        raise ResponsesBadRequest(
            f"Unsupported tool_choice type: {choice_type!r}.",
            param="tool_choice",
        )
    raise ResponsesBadRequest("tool_choice must be a string or object.", param="tool_choice")


def _parse_responses_format(payload: dict[str, Any]) -> dict[str, Any] | None:
    response_format = payload.get("response_format")
    if isinstance(response_format, dict):
        return response_format
    text_cfg = payload.get("text")
    if isinstance(text_cfg, dict):
        fmt = text_cfg.get("format")
        if isinstance(fmt, dict):
            fmt_type = fmt.get("type")
            if fmt_type == "text" or fmt_type is None:
                return None
            if fmt_type == "json_object":
                return {"type": "json_object"}
            if fmt_type == "json_schema":
                if isinstance(fmt.get("json_schema"), dict):
                    return {"type": "json_schema", "json_schema": fmt["json_schema"]}
                schema_fields = {k: v for k, v in fmt.items() if k != "type"}
                return {"type": "json_schema", "json_schema": schema_fields}
            return fmt
    return None


def parse_responses_chat_params(
    payload: Any,
    *,
    state_cache: ConversationStateCache | None = None,
) -> dict[str, Any]:
    """Validate POST /v1/responses payload and normalize into internal chat params."""
    if not isinstance(payload, dict):
        raise ResponsesBadRequest("Request body must be a JSON object.")

    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ResponsesBadRequest("model must be a non-empty string.", param="model")
    model = model.strip()

    previous_response_id = payload.get("previous_response_id")
    prior_messages: list[dict[str, Any]] = []
    prior_custom_tools: set[str] = set()
    if previous_response_id is not None:
        if not isinstance(previous_response_id, str) or not previous_response_id.strip():
            raise ResponsesBadRequest(
                "previous_response_id must be a non-empty string when provided.",
                param="previous_response_id",
            )
        previous_response_id = previous_response_id.strip()
        if state_cache is None:
            raise ResponsesBadRequest(
                f"Previous response '{previous_response_id}' was not found or has expired.",
                param="previous_response_id",
            )
        cached = state_cache.get_response(previous_response_id)
        if cached is None:
            raise ResponsesBadRequest(
                f"Previous response '{previous_response_id}' was not found or has expired.",
                param="previous_response_id",
            )
        prior_messages = list(cached.get("messages") or [])
        prior_custom_tools = set(cached.get("custom_tool_names") or ())

    raw_input = payload.get("input")
    if raw_input is None:
        raise ResponsesBadRequest("input is required.", param="input")

    messages, input_custom_tools = responses_input_to_messages(
        raw_input,
        instructions=payload.get("instructions"),
        state_cache=state_cache,
    )

    if prior_messages:
        # Merge without duplicating an identical leading system prompt
        if (
            messages
            and messages[0].get("role") == "system"
            and prior_messages
            and prior_messages[0].get("role") == "system"
        ):
            if messages[0].get("content") == prior_messages[0].get("content"):
                messages = [*prior_messages, *messages[1:]]
            else:
                merged_sys = f"{prior_messages[0].get('content', '')}\n\n{messages[0].get('content', '')}".strip()
                messages = [{"role": "system", "content": merged_sys}, *prior_messages[1:], *messages[1:]]
        else:
            messages = [*prior_messages, *messages]

    if not messages:
        raise ResponsesBadRequest("input produced no messages.", param="input")

    max_tokens = payload.get("max_output_tokens")
    if max_tokens is None:
        max_tokens = payload.get("max_tokens")
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ResponsesBadRequest(
                "max_output_tokens must be a positive integer.",
                param="max_output_tokens",
            )

    temperature = payload.get("temperature")
    if temperature is not None and (isinstance(temperature, bool) or not isinstance(temperature, (int, float))):
        raise ResponsesBadRequest("temperature must be a number.", param="temperature")

    top_p = payload.get("top_p")
    if top_p is not None and (isinstance(top_p, bool) or not isinstance(top_p, (int, float))):
        raise ResponsesBadRequest("top_p must be a number.", param="top_p")

    stop = payload.get("stop")
    if stop is not None and not (
        isinstance(stop, str) or (isinstance(stop, list) and all(isinstance(s, str) for s in stop))
    ):
        raise ResponsesBadRequest("stop must be a string or an array of strings.", param="stop")

    reasoning_effort: str | None = None
    raw_reasoning = payload.get("reasoning")
    if raw_reasoning is not None:
        if not isinstance(raw_reasoning, dict):
            raise ResponsesBadRequest("reasoning must be an object.", param="reasoning")
        if "effort" in raw_reasoning and raw_reasoning.get("effort") is not None:
            eff = raw_reasoning.get("effort")
            if not isinstance(eff, str) or eff.strip().lower() not in STANDARD_REASONING_EFFORTS:
                raise ResponsesBadRequest(
                    f"Unsupported reasoning.effort: {eff!r}. Allowed: {sorted(STANDARD_REASONING_EFFORTS)}",
                    param="reasoning.effort",
                    code="unsupported_parameter",
                )
            reasoning_effort = eff.strip().lower()
    elif payload.get("reasoning_effort") is not None:
        eff = payload.get("reasoning_effort")
        if not isinstance(eff, str) or eff.strip().lower() not in STANDARD_REASONING_EFFORTS:
            raise ResponsesBadRequest(
                f"Unsupported reasoning_effort: {eff!r}. Allowed: {sorted(STANDARD_REASONING_EFFORTS)}",
                param="reasoning_effort",
                code="unsupported_parameter",
            )
        reasoning_effort = eff.strip().lower()

    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise ResponsesBadRequest("stream must be a boolean.", param="stream")

    parallel_tool_calls = payload.get("parallel_tool_calls")
    if parallel_tool_calls is not None and not isinstance(parallel_tool_calls, bool):
        raise ResponsesBadRequest(
            "parallel_tool_calls must be a boolean.",
            param="parallel_tool_calls",
        )

    openai_tools, tool_custom_names = responses_tools_to_openai(payload.get("tools"))
    tool_choice = responses_tool_choice_to_openai(payload.get("tool_choice"))
    response_format = _parse_responses_format(payload)

    custom_tool_names = prior_custom_tools | input_custom_tools | tool_custom_names

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "reasoning_effort": reasoning_effort,
        "stop": stop,
        "stream": stream,
        "tools": openai_tools,
        "tool_choice": tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
        "response_format": response_format,
        "custom_tool_names": custom_tool_names,
        "previous_response_id": previous_response_id,
    }


def build_assistant_history_message(result: dict[str, Any]) -> dict[str, Any]:
    """Build internal OpenAI-shape assistant message from a generate/stream result for state caching."""
    text = result.get("text")
    tool_calls = result.get("tool_calls")
    msg: dict[str, Any] = {
        "role": "assistant",
        "content": str(text) if text else (None if tool_calls else ""),
    }
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _coerce_usage_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _clean_detail_map(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    detail: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        parsed = _coerce_usage_int(value)
        if parsed is not None:
            detail[key] = parsed
    return detail


def _responses_usage_object(usage_raw: Any) -> dict[str, Any]:
    usage = usage_raw if isinstance(usage_raw, dict) else {}
    input_tokens = _coerce_usage_int(usage.get("input_tokens"))
    if input_tokens is None:
        input_tokens = _coerce_usage_int(usage.get("prompt_tokens")) or 0
    output_tokens = _coerce_usage_int(usage.get("output_tokens"))
    if output_tokens is None:
        output_tokens = _coerce_usage_int(usage.get("completion_tokens")) or 0
    total_tokens = _coerce_usage_int(usage.get("total_tokens"))
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens
    usage_obj: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }
    input_details = _clean_detail_map(usage.get("input_tokens_details"))
    if not input_details:
        input_details = _clean_detail_map(usage.get("prompt_tokens_details"))
    if input_details:
        usage_obj["input_tokens_details"] = input_details
    output_details = _clean_detail_map(usage.get("output_tokens_details"))
    if not output_details:
        output_details = _clean_detail_map(usage.get("completion_tokens_details"))
    if output_details:
        usage_obj["output_tokens_details"] = output_details
    return usage_obj


def _responses_status(finish_reason: str | None) -> tuple[str, dict[str, str] | None]:
    if finish_reason == "length":
        return "incomplete", {"reason": "max_output_tokens"}
    if finish_reason == "content_filter":
        return "incomplete", {"reason": "content_filter"}
    return "completed", None


def openai_result_to_responses(
    result: dict[str, Any],
    *,
    model: str,
    response_id: str | None = None,
    custom_tool_names: set[str] | None = None,
) -> dict[str, Any]:
    """Convert internal {text, tool_calls, finish_reason, usage} into an OpenAI Responses object."""
    rid = response_id or new_response_id()
    custom_names = set(custom_tool_names or ())
    text = result.get("text")
    tool_calls = result.get("tool_calls") or []
    finish_reason = result.get("finish_reason") or "stop"

    output_items: list[dict[str, Any]] = []
    if text or not tool_calls:
        output_items.append(
            {
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": str(text or ""),
                        "annotations": [],
                    }
                ],
            }
        )

    for index, call in enumerate(tool_calls):
        if not isinstance(call, dict):
            call = {"id": None, "function": {"name": "", "arguments": call}}
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(fn.get("name") or "")
        call_id = str(call.get("id") or f"call_{name or 'tool'}_{index}")
        raw_args = fn.get("arguments", "{}")
        if name in custom_names:
            output_items.append(
                {
                    "id": f"ctc_{uuid.uuid4().hex[:24]}",
                    "type": "custom_tool_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": name,
                    "input": extract_custom_tool_input(raw_args),
                }
            )
        else:
            if isinstance(raw_args, str):
                args_str = raw_args
            else:
                try:
                    args_str = json.dumps(raw_args, ensure_ascii=False)
                except (TypeError, ValueError):
                    raise ResponsesBadRequest(
                        "Upstream tool call arguments cannot be represented as JSON.",
                        param="output",
                    )
            output_items.append(
                {
                    "id": f"fc_{uuid.uuid4().hex[:24]}",
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": name,
                    "arguments": args_str,
                }
            )

    usage_obj = _responses_usage_object(result.get("usage"))
    status, incomplete_details = _responses_status(finish_reason if isinstance(finish_reason, str) else "stop")
    response_obj: dict[str, Any] = {
        "id": rid,
        "object": "response",
        "created_at": int(time.time()),
        "status": status,
        "model": model,
        "output": output_items,
        "output_text": str(text or ""),
        "usage": usage_obj,
    }
    if incomplete_details is not None:
        response_obj["incomplete_details"] = incomplete_details
    return response_obj


class ResponsesStreamState:
    """State machine converting internal stream events into OpenAI Responses SSE events."""

    def __init__(
        self,
        *,
        response_id: str,
        model: str,
        custom_tool_names: set[str] | None = None,
    ) -> None:
        self.response_id = response_id
        self.model = model
        self.custom_tool_names = set(custom_tool_names or ())
        self.created_at = int(time.time())
        self.started = False
        self.msg_item_id = f"msg_{uuid.uuid4().hex[:24]}"
        self.msg_open = False
        self.msg_output_index = 0
        self.text_parts: list[str] = []
        self.tool_buffers: dict[int, dict[str, str]] = {}
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None
        self.completed_items: list[dict[str, Any]] = []

    def _base_response(
        self,
        *,
        status: str,
        output: list[dict[str, Any]],
        usage: dict[str, Any] | None = None,
        incomplete_details: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "output": output,
        }
        if usage is not None:
            obj["usage"] = usage
        if status == "incomplete":
            obj["incomplete_details"] = dict(incomplete_details) if incomplete_details else {"reason": "max_output_tokens"}
        return obj

    def _ensure_start(self) -> list[str]:
        if self.started:
            return []
        self.started = True
        return [
            responses_sse_event(
                "response.created",
                {"response": self._base_response(status="in_progress", output=[])},
            ),
            responses_sse_event(
                "response.in_progress",
                {"response": self._base_response(status="in_progress", output=[])},
            ),
        ]

    def _ensure_message_open(self) -> list[str]:
        if self.msg_open:
            return []
        self.msg_open = True
        self.msg_output_index = len(self.completed_items)
        msg_item = {
            "id": self.msg_item_id,
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        part = {"type": "output_text", "text": "", "annotations": []}
        return [
            responses_sse_event(
                "response.output_item.added",
                {"output_index": self.msg_output_index, "item": msg_item},
            ),
            responses_sse_event(
                "response.content_part.added",
                {
                    "item_id": self.msg_item_id,
                    "output_index": self.msg_output_index,
                    "content_index": 0,
                    "part": part,
                },
            ),
        ]

    def _close_message(self) -> list[str]:
        if not self.msg_open:
            return []
        self.msg_open = False
        full_text = "".join(self.text_parts)
        done_part = {"type": "output_text", "text": full_text, "annotations": []}
        done_item = {
            "id": self.msg_item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [done_part],
        }
        self.completed_items.append(done_item)
        return [
            responses_sse_event(
                "response.output_text.done",
                {
                    "item_id": self.msg_item_id,
                    "output_index": self.msg_output_index,
                    "content_index": 0,
                    "text": full_text,
                },
            ),
            responses_sse_event(
                "response.content_part.done",
                {
                    "item_id": self.msg_item_id,
                    "output_index": self.msg_output_index,
                    "content_index": 0,
                    "part": done_part,
                },
            ),
            responses_sse_event(
                "response.output_item.done",
                {"output_index": self.msg_output_index, "item": done_item},
            ),
        ]

    def on_event(self, event: dict[str, Any]) -> list[str]:
        out = self._ensure_start()
        delta_text = event.get("delta_text", "")
        if delta_text is None:
            delta_text = ""
        if not isinstance(delta_text, str):
            try:
                delta_text = json.dumps(delta_text, ensure_ascii=False)
            except (TypeError, ValueError):
                raise ResponsesBadRequest(
                    "Upstream stream text delta cannot be represented as JSON.",
                    param="output",
                )
        if delta_text:
            out.extend(self._ensure_message_open())
            self.text_parts.append(delta_text)
            out.append(
                responses_sse_event(
                    "response.output_text.delta",
                    {
                        "item_id": self.msg_item_id,
                        "output_index": self.msg_output_index,
                        "content_index": 0,
                        "delta": delta_text,
                    },
                )
            )

        for position, call in enumerate(event.get("delta_tool_calls") or []):
            if not isinstance(call, dict):
                call = {"index": position, "function": {"arguments": call}}
            key = call.get("index", position)
            try:
                key = int(key)
            except (TypeError, ValueError):
                key = position
            buf = self.tool_buffers.setdefault(key, {"id": "", "name": "", "arguments": ""})
            if call.get("id"):
                buf["id"] = str(call["id"])
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            if fn.get("name"):
                buf["name"] = str(fn["name"])
            args = fn.get("arguments", "")
            if isinstance(args, str):
                buf["arguments"] += args
            elif args is not None:
                try:
                    buf["arguments"] += json.dumps(args, ensure_ascii=False)
                except (TypeError, ValueError):
                    raise ResponsesBadRequest(
                        "Upstream stream tool call arguments cannot be represented as JSON.",
                        param="output",
                    )

        if event.get("finish_reason") is not None:
            self.finish_reason = event["finish_reason"]
        if isinstance(event.get("usage"), dict):
            self.usage = event["usage"]
        return out

    feed = on_event

    def finalize(self, *, prompt_tokens: int = 0, completion_text: str = "") -> list[str]:
        out = self._ensure_start()

        if not self.msg_open and not self.text_parts and not self.tool_buffers:
            out.extend(self._ensure_message_open())

        if self.msg_open:
            out.extend(self._close_message())

        for key in sorted(self.tool_buffers):
            buf = self.tool_buffers[key]
            name = buf["name"]
            call_id = buf["id"] or f"call_{name or 'tool'}_{key}"
            raw_args = buf["arguments"]
            output_index = len(self.completed_items)

            if name in self.custom_tool_names:
                item_id = f"ctc_{uuid.uuid4().hex[:24]}"
                custom_input = extract_custom_tool_input(raw_args)
                added_item = {
                    "id": item_id,
                    "type": "custom_tool_call",
                    "status": "in_progress",
                    "call_id": call_id,
                    "name": name,
                    "input": "",
                }
                done_item = {
                    "id": item_id,
                    "type": "custom_tool_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": name,
                    "input": custom_input,
                }
                self.completed_items.append(done_item)
                out.append(
                    responses_sse_event(
                        "response.output_item.added",
                        {"output_index": output_index, "item": added_item},
                    )
                )
                if custom_input:
                    out.append(
                        responses_sse_event(
                            "response.custom_tool_call_input.delta",
                            {
                                "item_id": item_id,
                                "output_index": output_index,
                                "delta": custom_input,
                            },
                        )
                    )
                out.append(
                    responses_sse_event(
                        "response.custom_tool_call_input.done",
                        {
                            "item_id": item_id,
                            "output_index": output_index,
                            "input": custom_input,
                        },
                    )
                )
                out.append(
                    responses_sse_event(
                        "response.output_item.done",
                        {"output_index": output_index, "item": done_item},
                    )
                )
            else:
                item_id = f"fc_{uuid.uuid4().hex[:24]}"
                args_str = raw_args if raw_args.strip() else "{}"
                added_item = {
                    "id": item_id,
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": call_id,
                    "name": name,
                    "arguments": "",
                }
                done_item = {
                    "id": item_id,
                    "type": "function_call",
                    "status": "completed",
                    "call_id": call_id,
                    "name": name,
                    "arguments": args_str,
                }
                self.completed_items.append(done_item)
                out.append(
                    responses_sse_event(
                        "response.output_item.added",
                        {"output_index": output_index, "item": added_item},
                    )
                )
                if args_str:
                    out.append(
                        responses_sse_event(
                            "response.function_call_arguments.delta",
                            {
                                "item_id": item_id,
                                "output_index": output_index,
                                "delta": args_str,
                            },
                        )
                    )
                out.append(
                    responses_sse_event(
                        "response.function_call_arguments.done",
                        {
                            "item_id": item_id,
                            "output_index": output_index,
                            "arguments": args_str,
                        },
                    )
                )
                out.append(
                    responses_sse_event(
                        "response.output_item.done",
                        {"output_index": output_index, "item": done_item},
                    )
                )

        if isinstance(self.usage, dict):
            usage_obj = _responses_usage_object(self.usage)
        else:
            input_tokens = prompt_tokens
            text_for_est = completion_text or "".join(self.text_parts)
            output_tokens = max(0, (len(text_for_est) + 3) // 4) if text_for_est else 0
            usage_obj = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
        status, incomplete_details = _responses_status(self.finish_reason if isinstance(self.finish_reason, str) else "stop")
        out.append(
            responses_sse_event(
                "response.completed",
                {
                    "response": self._base_response(
                        status=status,
                        output=self.completed_items,
                        usage=usage_obj,
                        incomplete_details=incomplete_details,
                    )
                },
            )
        )
        return out

    finish = finalize

    def assistant_turn_message(self) -> dict[str, Any]:
        """Reconstruct internal OpenAI-shape assistant message for ConversationStateCache."""
        text = "".join(self.text_parts)
        tool_calls: list[dict[str, Any]] = []
        for key in sorted(self.tool_buffers):
            buf = self.tool_buffers[key]
            name = buf["name"]
            call_id = buf["id"] or f"call_{name or 'tool'}_{key}"
            raw_args = buf["arguments"] if buf["arguments"].strip() else "{}"
            tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": raw_args},
                }
            )
        msg: dict[str, Any] = {
            "role": "assistant",
            "content": text if text else (None if tool_calls else ""),
        }
        if tool_calls:
            msg["tool_calls"] = tool_calls
        return msg
