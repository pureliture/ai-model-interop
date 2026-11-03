"""Unit tests for OpenAI Responses API ingress/egress (`responses_ingress.py`)."""

from __future__ import annotations

import json

import pytest

from openai_compatible_bridge.core.conversation_state import ConversationStateCache
from openai_compatible_bridge.responses_ingress import (
    ResponsesBadRequest,
    ResponsesStreamState,
    extract_custom_tool_input,
    new_response_id,
    openai_result_to_responses,
    parse_responses_chat_params,
)


def _parse_sse_frames(chunks: list[str]) -> list[tuple[str, dict]]:
    frames: list[tuple[str, dict]] = []
    raw = "".join(chunks)
    for block in raw.strip().split("\n\n"):
        if not block.strip():
            continue
        event_name = ""
        data_obj = {}
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_obj = json.loads(line[len("data:") :].strip())
        frames.append((event_name, data_obj))
    return frames


def test_parse_responses_instructions_developer_and_history_items() -> None:
    params = parse_responses_chat_params(
        {
            "model": "jetbrains:claude-sonnet-4-5",
            "instructions": "Base system prompt",
            "input": [
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "AGENTS.md rules"}],
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Refactor UserService"}],
                },
                {
                    "type": "reasoning",
                    "id": "rs_123",
                    "summary": [{"type": "summary_text", "text": "thinking..."}],
                    "encrypted_content": "enc_xyz",
                },
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_shell_1",
                    "name": "shell",
                    "arguments": '{"command":["ls"]}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_shell_1",
                    "output": "UserService.py\n",
                },
                {
                    "type": "message",
                    "role": "developer",
                    "content": "Mid-turn environment update",
                },
            ],
            "store": True,
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": "cache-1",
            "truncation": "auto",
        }
    )

    assert params["model"] == "jetbrains:claude-sonnet-4-5"
    messages = params["messages"]
    assert messages[0] == {
        "role": "system",
        "content": "Base system prompt\n\nAGENTS.md rules",
    }
    assert messages[1] == {"role": "user", "content": "Refactor UserService"}
    assert messages[2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_shell_1",
                "type": "function",
                "function": {"name": "shell", "arguments": '{"command":["ls"]}'},
            }
        ],
    }
    assert messages[3] == {
        "role": "tool",
        "tool_call_id": "call_shell_1",
        "content": "UserService.py\n",
    }
    # Mid-conversation developer message preserved in turn order as user message
    assert messages[4] == {"role": "user", "content": "Mid-turn environment update"}


def test_parse_responses_flat_and_custom_tools_and_reverse_mapping() -> None:
    params = parse_responses_chat_params(
        {
            "model": "jetbrains:claude-sonnet-4-5",
            "input": "Fix the bug in app.py",
            "tools": [
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "Execute shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                    },
                },
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "description": "Apply a patch to files",
                },
            ],
            "parallel_tool_calls": False,
            "reasoning": {"effort": "high"},
        }
    )

    assert params["reasoning_effort"] == "high"
    assert params["parallel_tool_calls"] is False
    assert params["custom_tool_names"] == {"apply_patch"}
    assert len(params["tools"]) == 2
    assert params["tools"][0] == {
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": "Execute shell command",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    }
    assert params["tools"][1]["type"] == "function"
    assert params["tools"][1]["function"]["name"] == "apply_patch"
    assert params["tools"][1]["function"]["parameters"]["properties"]["input"]["type"] == "string"

    # Non-streaming response converts apply_patch back to custom_tool_call
    patch_text = "*** Begin Patch\n*** Update File: app.py\n@@\n-old\n+new\n*** End Patch"
    resp = openai_result_to_responses(
        {
            "text": "Applying patch now.",
            "tool_calls": [
                {
                    "id": "call_patch_1",
                    "type": "function",
                    "function": {
                        "name": "apply_patch",
                        "arguments": json.dumps({"input": patch_text}),
                    },
                },
                {
                    "id": "call_exec_1",
                    "type": "function",
                    "function": {
                        "name": "exec_command",
                        "arguments": '{"cmd":"pytest"}',
                    },
                },
            ],
            "finish_reason": "tool_calls",
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        },
        model="jetbrains:claude-sonnet-4-5",
        response_id=new_response_id(),
        custom_tool_names=params["custom_tool_names"],
    )

    assert resp["object"] == "response"
    assert resp["status"] == "completed"
    assert len(resp["output"]) == 3
    assert resp["output"][0]["type"] == "message"
    assert resp["output"][0]["content"][0]["text"] == "Applying patch now."
    assert resp["output"][1]["type"] == "custom_tool_call"
    assert resp["output"][1]["call_id"] == "call_patch_1"
    assert resp["output"][1]["name"] == "apply_patch"
    assert resp["output"][1]["input"] == patch_text
    assert resp["output"][2]["type"] == "function_call"
    assert resp["output"][2]["call_id"] == "call_exec_1"
    assert resp["output"][2]["name"] == "exec_command"
    assert resp["output"][2]["arguments"] == '{"cmd":"pytest"}'


def test_parse_responses_previous_response_id_and_custom_tool_output() -> None:
    cache = ConversationStateCache(maxsize=16, ttl_seconds=60)
    cache.save_response(
        "resp_prev_1",
        messages=[
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "Patch app.py"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_patch_1",
                        "type": "function",
                        "function": {"name": "apply_patch", "arguments": '{"input":"patch"}'},
                    }
                ],
            },
        ],
        custom_tool_names={"apply_patch"},
    )

    params = parse_responses_chat_params(
        {
            "model": "jetbrains:claude-sonnet-4-5",
            "previous_response_id": "resp_prev_1",
            "instructions": "Sys",
            "input": [
                {
                    "type": "custom_tool_call_output",
                    "call_id": "call_patch_1",
                    "output": "Success. Updated 1 file.",
                }
            ],
        },
        state_cache=cache,
    )

    assert params["custom_tool_names"] == {"apply_patch"}
    assert len(params["messages"]) == 4
    assert params["messages"][0] == {"role": "system", "content": "Sys"}
    assert params["messages"][3] == {
        "role": "tool",
        "tool_call_id": "call_patch_1",
        "content": "Success. Updated 1 file.",
    }

    with pytest.raises(ResponsesBadRequest) as exc:
        parse_responses_chat_params(
            {
                "model": "jetbrains:claude-sonnet-4-5",
                "previous_response_id": "resp_missing",
                "input": "hello",
            },
            state_cache=cache,
        )
    assert exc.value.param == "previous_response_id"


def test_parse_responses_rejects_unmappable_reasoning_effort() -> None:
    with pytest.raises(ResponsesBadRequest) as exc:
        parse_responses_chat_params(
            {
                "model": "jetbrains:claude-sonnet-4-5",
                "input": "hello",
                "reasoning": {"effort": "ultracode"},
            }
        )
    assert exc.value.code == "unsupported_parameter"


def test_responses_stream_state_text_function_and_custom_tool_events() -> None:
    state = ResponsesStreamState(
        response_id="resp_stream_1",
        model="jetbrains:claude-sonnet-4-5",
        custom_tool_names={"apply_patch"},
    )
    chunks: list[str] = []
    chunks.extend(state.on_event({"delta_text": "Hello ", "finish_reason": None, "usage": None}))
    chunks.extend(state.on_event({"delta_text": "world!", "finish_reason": None, "usage": None}))
    chunks.extend(
        state.on_event(
            {
                "delta_text": "",
                "delta_tool_calls": [
                    {
                        "index": 0,
                        "id": "call_fn_1",
                        "function": {"name": "shell", "arguments": '{"cmd":'},
                    },
                    {
                        "index": 1,
                        "id": "call_patch_1",
                        "function": {"name": "apply_patch", "arguments": '{"input":"*** Begin Patch"}'},
                    },
                ],
                "finish_reason": None,
                "usage": None,
            }
        )
    )
    chunks.extend(
        state.on_event(
            {
                "delta_text": "",
                "delta_tool_calls": [
                    {
                        "index": 0,
                        "function": {"arguments": '"ls"}'},
                    }
                ],
                "finish_reason": "tool_calls",
                "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
            }
        )
    )
    chunks.extend(state.finalize())

    frames = _parse_sse_frames(chunks)
    event_names = [name for name, _ in frames]
    assert event_names[0] == "response.created"
    assert event_names[1] == "response.in_progress"
    assert "response.output_item.added" in event_names
    assert "response.content_part.added" in event_names
    assert "response.output_text.delta" in event_names
    assert "response.output_text.done" in event_names
    assert "response.function_call_arguments.delta" in event_names
    assert "response.function_call_arguments.done" in event_names
    assert "response.custom_tool_call_input.delta" in event_names
    assert "response.custom_tool_call_input.done" in event_names
    assert event_names[-1] == "response.completed"

    completed_payload = frames[-1][1]["response"]
    assert completed_payload["status"] == "completed"
    assert completed_payload["usage"] == {"input_tokens": 10, "output_tokens": 15, "total_tokens": 25}
    assert len(completed_payload["output"]) == 3
    assert completed_payload["output"][0]["type"] == "message"
    assert completed_payload["output"][0]["content"][0]["text"] == "Hello world!"
    assert completed_payload["output"][1]["type"] == "function_call"
    assert completed_payload["output"][1]["arguments"] == '{"cmd":"ls"}'
    assert completed_payload["output"][2]["type"] == "custom_tool_call"
    assert completed_payload["output"][2]["input"] == "*** Begin Patch"

    history_msg = state.assistant_turn_message()
    assert history_msg["role"] == "assistant"
    assert history_msg["content"] == "Hello world!"
    assert len(history_msg["tool_calls"]) == 2
