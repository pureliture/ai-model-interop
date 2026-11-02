"""Anthropic Messages ingress (POST /v1/messages) translation layer.

Claude Code / Claude Desktop이 말하는 Anthropic 규격을 bridge 내부의
OpenAI-shape 파이프라인으로 변환한다. 순수 함수만 두어 단위 테스트 가능하게 한다.
전송·과금·SSE 프레이밍은 main.py 라우트가 담당한다.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from fastapi.responses import JSONResponse

ANTHROPIC_VERSION = "2023-06-01"

# tool 호출을 지원하는 provider (live 기준).
PROVIDER_TOOL_SUPPORT = {"vertex": True, "foundry": True, "openai_compatible": True, "jetbrains": True, "ollama": False}

# Desktop validator용 기본 context window. 레지스트리 alias별
# `context_length` 키로 override한다 (정확한 값은 운영자가 기입).
DEFAULT_CONTEXT_LENGTH = 128000


class AnthropicBadRequest(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def new_anthropic_message_id() -> str:
    return f"msg_{uuid.uuid4().hex[:24]}"


def anthropic_error_response(
    *,
    message: str,
    status_code: int,
    error_type: str,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"type": "error", "error": {"type": error_type, "message": message}},
    )


def map_status_to_anthropic_error_type(status_code: int) -> str:
    return {
        400: "invalid_request_error",
        401: "authentication_error",
        403: "permission_error",
        404: "not_found_error",
        429: "rate_limit_error",
    }.get(status_code, "api_error" if status_code >= 500 else "invalid_request_error")


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return str(content)


def _system_to_text(system: Any) -> str:
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in system
            if not isinstance(block, dict) or block.get("type") in ("text", None) or "text" in block
        )
    raise AnthropicBadRequest("system must be a string or a list of text blocks.")


def _tool_result_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif block.get("type") in ("thinking", "redacted_thinking"):
                    continue
                elif "text" in block:
                    parts.append(str(block["text"]))
                else:
                    raise AnthropicBadRequest(
                        f"Unsupported tool_result block type: {block.get('type')!r}."
                    )
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def anthropic_messages_to_openai(messages: Any) -> list[dict[str, Any]]:
    """Anthropic messages 배열을 OpenAI-shape 배열로 변환한다.

    - assistant tool_use -> tool_calls
    - user tool_result -> role=tool 메시지 (is_error는 extra로 보존)
    - thinking/redacted_thinking -> 스킵, image 등 미디어 -> 400
    """
    if not isinstance(messages, list) or not messages:
        raise AnthropicBadRequest("messages must be a non-empty array.")

    openai_messages: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise AnthropicBadRequest(f"messages[{index}] must be an object.")
        role = message.get("role")
        if role not in ("user", "assistant", "system"):
            raise AnthropicBadRequest(f"messages[{index}].role must be user, assistant, or system.")
        content = message.get("content")

        if role == "system":
            text = _extract_text(content)
            openai_messages.append({"role": "system", "content": text})
            continue

        if isinstance(content, str):
            openai_messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            raise AnthropicBadRequest(f"messages[{index}].content must be a string or an array.")

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        tool_messages: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                raise AnthropicBadRequest(f"messages[{index}] content blocks must be objects.")
            block_type = block.get("type")
            if block_type == "text":
                text_parts.append(str(block.get("text", "")))
            elif block_type == "thinking" or block_type == "redacted_thinking":
                continue
            elif block_type == "tool_use":
                if role != "assistant":
                    raise AnthropicBadRequest("tool_use blocks must be in assistant messages.")
                name = block.get("name", "")
                raw_input = block.get("input", {})
                if not isinstance(name, str) or not name:
                    raise AnthropicBadRequest("tool_use blocks require a name.")
                if isinstance(raw_input, dict):
                    args_str = json.dumps(raw_input, ensure_ascii=False)
                elif isinstance(raw_input, str):
                    args_str = raw_input
                else:
                    raise AnthropicBadRequest("tool_use input must be an object.")
                tool_calls.append(
                    {
                        "id": str(block.get("id", "")),
                        "type": "function",
                        "function": {"name": name, "arguments": args_str},
                    }
                )
            elif block_type == "tool_result":
                if role != "user":
                    raise AnthropicBadRequest("tool_result blocks must be in user messages.")
                tool_use_id = block.get("tool_use_id", "")
                if not tool_use_id:
                    raise AnthropicBadRequest("tool_result blocks require tool_use_id.")
                tool_message: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": str(tool_use_id),
                    "content": _tool_result_to_text(block.get("content", "")),
                }
                if block.get("is_error") is True:
                    tool_message["is_error"] = True
                tool_messages.append(tool_message)
            else:
                raise AnthropicBadRequest(f"Unsupported content block type: {block_type!r}.")

        if role == "assistant":
            text = "".join(text_parts)
            entry: dict[str, Any] = {"role": "assistant", "content": text if text or not tool_calls else None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            openai_messages.append(entry)
        else:
            text = "".join(text_parts)
            if text or not tool_messages:
                openai_messages.append({"role": "user", "content": text})
            openai_messages.extend(tool_messages)

    return openai_messages


def anthropic_tools_to_openai(tools: Any) -> list[dict[str, Any]] | None:
    if tools is None:
        return None
    if not isinstance(tools, list) or not tools:
        raise AnthropicBadRequest("tools must be a non-empty array when present.")
    openai_tools: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        if not isinstance(tool, dict):
            raise AnthropicBadRequest(f"tools[{index}] must be an object.")
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise AnthropicBadRequest(f"tools[{index}].name must be a non-empty string.")
        input_schema = tool.get("input_schema", {})
        if not isinstance(input_schema, dict):
            raise AnthropicBadRequest(f"tools[{index}].input_schema must be an object.")
        entry: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": name,
                "description": str(tool.get("description", "")),
                "parameters": input_schema,
            },
        }
        openai_tools.append(entry)
    return openai_tools


def anthropic_tool_choice_to_openai(tool_choice: Any) -> str | dict[str, Any] | None:
    if tool_choice is None:
        return None
    if not isinstance(tool_choice, dict):
        raise AnthropicBadRequest("tool_choice must be an object.")
    choice_type = tool_choice.get("type")
    if choice_type == "auto":
        return "auto"
    if choice_type == "any":
        return "required"
    if choice_type == "none":
        return "none"
    if choice_type == "tool":
        name = (tool_choice.get("name") or tool_choice.get("tool", {}).get("name")
                if isinstance(tool_choice.get("tool"), dict) else tool_choice.get("name"))
        if not isinstance(name, str) or not name:
            raise AnthropicBadRequest("tool_choice of type tool requires a name.")
        return {"type": "function", "function": {"name": name}}
    raise AnthropicBadRequest(f"Unsupported tool_choice type: {choice_type!r}.")


def parse_anthropic_chat_params(payload: Any) -> dict[str, Any]:
    """POST /v1/messages body를 검증하고 내부 파라미터 dict로 변환한다."""
    if not isinstance(payload, dict):
        raise AnthropicBadRequest("Request body must be a JSON object.")
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise AnthropicBadRequest("model must be a non-empty string.")
    messages = anthropic_messages_to_openai(payload.get("messages"))
    max_tokens = payload.get("max_tokens")
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
        raise AnthropicBadRequest("max_tokens must be a positive integer.")

    system_text = _system_to_text(payload.get("system"))
    if system_text:
        messages = [{"role": "system", "content": system_text}, *messages]

    temperature = payload.get("temperature")
    if temperature is not None and not isinstance(temperature, (int, float)):
        raise AnthropicBadRequest("temperature must be a number.")
    top_p = payload.get("top_p")
    if top_p is not None and not isinstance(top_p, (int, float)):
        raise AnthropicBadRequest("top_p must be a number.")
    stop = payload.get("stop_sequences")
    if stop is not None and not (
        isinstance(stop, str) or (isinstance(stop, list) and all(isinstance(s, str) for s in stop))
    ):
        raise AnthropicBadRequest("stop_sequences must be a string or an array of strings.")

    output_config = payload.get("output_config")
    if output_config is not None and not isinstance(output_config, dict):
        raise AnthropicBadRequest("output_config must be an object.")
    reasoning_effort = (output_config or {}).get("effort")
    if reasoning_effort is not None and (
        not isinstance(reasoning_effort, str)
        or reasoning_effort not in {"low", "medium", "high", "xhigh", "max"}
    ):
        raise AnthropicBadRequest(
            "output_config.effort must be one of: low, medium, high, xhigh, max."
        )

    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise AnthropicBadRequest("stream must be a boolean.")

    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "reasoning_effort": reasoning_effort,
        "stop": stop,
        "stream": stream,
        "tools": anthropic_tools_to_openai(payload.get("tools")),
        "tool_choice": anthropic_tool_choice_to_openai(payload.get("tool_choice")),
    }


def openai_result_to_anthropic(
    result: dict[str, Any], *, model: str, message_id: str,
) -> dict[str, Any]:
    """내부 chat 결과 {text, tool_calls, finish_reason, usage}를 Anthropic 응답으로 변환."""
    content: list[dict[str, Any]] = []
    text = result.get("text")
    if text:
        content.append({"type": "text", "text": str(text)})
    for call in result.get("tool_calls") or []:
        fn = call.get("function", {}) if isinstance(call, dict) else {}
        raw_args = fn.get("arguments", "{}")
        try:
            parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (TypeError, ValueError):
            parsed_args = {}
        if not isinstance(parsed_args, dict):
            parsed_args = {}
        content.append(
            {
                "type": "tool_use",
                "id": str(call.get("id", "")),
                "name": str(fn.get("name", "")),
                "input": parsed_args,
            }
        )
    finish = result.get("finish_reason") or "stop"
    if result.get("tool_calls"):
        stop_reason = "tool_use"
    elif finish in ("stop", "tool_calls"):
        stop_reason = "end_turn" if finish == "stop" else "tool_use"
    elif finish == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"
    usage = result.get("usage", {}) if isinstance(result.get("usage"), dict) else {}
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "output_tokens": int(usage.get("completion_tokens", 0) or 0),
        },
    }


def sse_event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


class AnthropicStreamState:
    """provider 스트림 이벤트를 Anthropic SSE 이벤트 문자열로 변환하는 상태머신.

    provider 이벤트 shape: {"delta_text", "delta_tool_calls", "finish_reason", "usage"}.
    tool 델타는 버퍼링 후 종료 시점에 블록으로 방출한다 (live provider는
    tool 스트리밍을 하지 않으므로 정확성 우선).
    """

    def __init__(self, *, message_id: str, model: str) -> None:
        self.message_id = message_id
        self.model = model
        self.started = False
        self.text_open = False
        self.text_chars = 0
        self.tool_buffers: dict[int, dict[str, str]] = {}
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None
        self.saw_event = False

    def _ensure_start(self) -> list[str]:
        if self.started:
            return []
        self.started = True
        return [
            sse_event(
                {
                    "type": "message_start",
                    "message": {
                        "id": self.message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                    },
                }
            ),
            sse_event(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
            ),
        ]

    def feed(self, event: dict[str, Any]) -> list[str]:
        out = self._ensure_start()
        self.saw_event = True
        delta_text = event.get("delta_text", "") or ""
        if delta_text:
            self.text_open = True
            self.text_chars += len(delta_text)
            out.append(
                sse_event(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": delta_text},
                    }
                )
            )
        for position, call in enumerate(event.get("delta_tool_calls") or []):
            if not isinstance(call, dict):
                continue
            key = call.get("index", position)
            try:
                key = int(key)
            except (TypeError, ValueError):
                key = position
            buf = self.tool_buffers.setdefault(key, {"id": "", "name": "", "arguments": ""})
            if call.get("id"):
                buf["id"] = str(call["id"])
            fn = call.get("function", {}) if isinstance(call.get("function"), dict) else {}
            if fn.get("name"):
                buf["name"] = str(fn["name"])
            args = fn.get("arguments", "")
            if isinstance(args, str):
                buf["arguments"] += args
        if event.get("finish_reason") is not None:
            self.finish_reason = event["finish_reason"]
        if isinstance(event.get("usage"), dict):
            self.usage = event["usage"]
        return out

    def finish(self, *, prompt_tokens: int, completion_text: str) -> list[str]:
        out: list[str] = []
        if not self.started:
            out.extend(self._ensure_start())
        if self.text_open:
            out.append(sse_event({"type": "content_block_stop", "index": 0}))
            self.text_open = False
        for offset, key in enumerate(sorted(self.tool_buffers)):
            buf = self.tool_buffers[key]
            block_index = offset + 1
            try:
                parsed = json.loads(buf["arguments"]) if buf["arguments"].strip() else {}
            except (TypeError, ValueError):
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
            out.append(
                sse_event(
                    {
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": buf["id"] or f"toolu_stream_{key}",
                            "name": buf["name"],
                            "input": {},
                        },
                    }
                )
            )
            out.append(
                sse_event(
                    {
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": buf["arguments"],
                        },
                    }
                )
            )
            out.append(sse_event({"type": "content_block_stop", "index": block_index}))
        if self.finish_reason == "tool_calls":
            stop_reason = "tool_use"
        elif self.finish_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = "end_turn"
        if isinstance(self.usage, dict):
            input_tokens = int(self.usage.get("prompt_tokens", 0) or 0)
            output_tokens = int(self.usage.get("completion_tokens", 0) or 0)
        else:
            input_tokens = prompt_tokens
            output_tokens = max(0, (len(completion_text) + 3) // 4) if completion_text else 0
        out.append(
            sse_event(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                }
            )
        )
        out.append(sse_event({"type": "message_stop"}))
        return out
