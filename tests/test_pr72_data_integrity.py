"""PR #72 data-integrity blockers: every test fails at PR head, passes after the fix."""
from __future__ import annotations

import asyncio
import json

import pytest

from openai_compatible_bridge.core.cost_tracking import BudgetReservationContext, NormalizedUsage

from openai_compatible_bridge.providers.jetbrains import _grazie_headers

from openai_compatible_bridge.providers.vertex import VertexAPIError
from openai_compatible_bridge.responses_ingress import (
    ResponsesBadRequest,
    ResponsesStreamState,
    openai_result_to_responses,
    parse_responses_chat_params,
)


def _dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _is_explicit_rejection(exc: BaseException, statuses: tuple[int, ...] = (400, 502)) -> bool:
    if isinstance(exc, ResponsesBadRequest):
        return True
    return isinstance(exc, VertexAPIError) and exc.status_code in statuses

def test_unsupported_responses_item_type_is_not_silently_empty_text():
    # 경로: responses_ingress.py :: responses_input_to_messages() message-type branch with a role
    payload = {
        "model": "gpt-5",
        "input": [{"type": "image_generation_call", "role": "user"}],
    }
    try:
        params = parse_responses_chat_params(payload)
    except (ResponsesBadRequest, VertexAPIError) as exc:
        assert _is_explicit_rejection(exc)
        return
    assert False, f"unsupported item type became empty text: {_dump(params['messages'])}"


def test_item_reference_is_not_silently_dropped():
    # 경로: responses_ingress.py :: _responses_input_to_items() item_reference branch
    payload = {
        "model": "gpt-5",
        "input": [
            {"type": "message", "role": "user", "content": "keep"},
            {
                "type": "item_reference",
                "id": "item_missing",
                "content": [{"type": "input_text", "text": "ITEM_REFERENCE_MARKER"}],
            },
        ],
    }
    try:
        params = parse_responses_chat_params(payload)
    except (ResponsesBadRequest, VertexAPIError) as exc:
        assert _is_explicit_rejection(exc)
        return
    assert "ITEM_REFERENCE_MARKER" in _dump(params["messages"]), (
        "item_reference was ignored; expected 400 or resolution"
    )


def test_function_call_non_string_arguments_are_serialized():
    # 경로: responses_ingress.py :: _responses_input_to_items() function_call branch
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": [{"q": "ARGUMENTS_MARKER"}],
            }
        ],
    }
    try:
        params = parse_responses_chat_params(payload)
    except (ResponsesBadRequest, VertexAPIError) as exc:
        assert _is_explicit_rejection(exc)
        return
    assert "ARGUMENTS_MARKER" in _dump(params["messages"]), (
        "non-string arguments collapsed to {} instead of being serialized"
    )


def test_non_string_content_text_is_not_discarded():
    # 경로: responses_ingress.py :: _extract_responses_content_text()
    payload = {
        "model": "gpt-5",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": {"value": "CONTENT_TEXT_MARKER"}}],
            }
        ],
    }
    try:
        params = parse_responses_chat_params(payload)
    except (ResponsesBadRequest, VertexAPIError) as exc:
        assert _is_explicit_rejection(exc)
        return
    assert "CONTENT_TEXT_MARKER" in _dump(params["messages"]), (
        "non-string text block was dropped; expected preservation or 400"
    )


def test_non_dict_tool_call_is_not_removed():
    # 경로: responses_ingress.py :: openai_result_to_responses() tool_calls loop
    result_payload = {
        "text": "ok",
        "finish_reason": "tool_calls",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}},
            "MALFORMED_TOOL_CALL_MARKER",
        ],
    }
    try:
        response = openai_result_to_responses(result_payload, model="gpt-5")
    except VertexAPIError as exc:
        assert exc.status_code == 502
        return
    assert "MALFORMED_TOOL_CALL_MARKER" in _dump(response), (
        "non-dict tool call was removed; expected preservation or 502"
    )


def test_non_string_stream_delta_is_not_dropped():
    # 경로: responses_ingress.py :: ResponsesStreamState.on_event() tool argument buffering
    state = ResponsesStreamState(response_id="resp_delta", model="gpt-5")
    try:
        state.on_event({
            "delta_tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "function": {"name": "lookup", "arguments": {"marker": "DELTA_MARKER"}},
                }
            ]
        })
        state.finalize()
    except VertexAPIError as exc:
        assert exc.status_code in (400, 502)
        return
    assert "DELTA_MARKER" in _dump(state.completed_items), (
        "non-string tool argument delta was dropped; expected accumulation or 400/502"
    )


def test_null_custom_tool_input_is_preserved():
    # 경로: responses_ingress.py :: _responses_input_to_items() custom_tool_call branch
    payload = {
        "model": "gpt-5",
        "input": [
            {"type": "custom_tool_call", "call_id": "call_1", "name": "apply_patch", "input": None}
        ],
    }
    try:
        params = parse_responses_chat_params(payload)
    except (ResponsesBadRequest, VertexAPIError) as exc:
        assert _is_explicit_rejection(exc)
        return
    arguments = params["messages"][0]["tool_calls"][0]["function"]["arguments"]
    assert "null" in arguments and '"input": ""' not in arguments, (
        "input:null was rewritten to an empty string"
    )


def test_usage_details_and_explicit_total_are_preserved():
    # 경로: responses_ingress.py :: openai_result_to_responses() usage mapping
    result = openai_result_to_responses(
        {
            "text": "ok",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 0,
                "prompt_tokens_details": {"cached_tokens": 7},
                "completion_tokens_details": {"reasoning_tokens": 5},
            },
        },
        model="gpt-5",
    )
    usage = result["usage"]
    assert usage.get("prompt_tokens_details", {}).get("cached_tokens") == 7
    assert usage.get("completion_tokens_details", {}).get("reasoning_tokens") == 5
    assert usage["total_tokens"] == 0, "explicit upstream total was recomputed"


def test_content_filter_finish_reason_keeps_incomplete_status():
    # 경로: responses_ingress.py :: openai_result_to_responses() status mapping
    result = openai_result_to_responses(
        {"text": "blocked", "finish_reason": "content_filter"}, model="gpt-5"
    )
    assert result["status"] == "incomplete", "content_filter was flattened to completed"
    details = result.get("incomplete_details") or result.get("details") or {}
    assert "content_filter" in _dump(details), "content_filter details were discarded"


def test_stream_error_does_not_replace_actual_usage_with_estimate():
    # 경로: core/cost_tracking.py :: BudgetReservationContext.__aexit__() / _finalize_failed_exit()
    class _Accounting:
        enabled = False

        def __init__(self) -> None:
            self.calls: list[tuple[str, object]] = []

        def finalize_success(self, reservation: object, usage: object) -> None:
            self.calls.append(("success", usage))

        def finalize_estimated_only(self, reservation: object, reason: object) -> None:
            self.calls.append(("estimated", reason))

        def release_nonbillable(self, reservation: object, reason: object) -> None:
            self.calls.append(("released", reason))

    accounting = _Accounting()
    ctx = BudgetReservationContext(
        accounting, "chat", "gpt-5", NormalizedUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    )
    ctx.is_stream = True
    ctx.stream_saw_event = True
    actual = NormalizedUsage(prompt_tokens=11, completion_tokens=3, total_tokens=14)
    ctx.complete(actual)

    async def fail() -> None:
        async with ctx:
            raise RuntimeError("stream broke")

    with pytest.raises(RuntimeError):
        asyncio.run(fail())

    assert accounting.calls, "no cost finalization recorded"
    kind, payload = accounting.calls[0]
    assert kind == "success" and payload == actual, (
        f"actual usage replaced by {payload!r} on stream error"
    )


def test_unknown_jetbrains_auth_mode_is_rejected(monkeypatch: pytest.MonkeyPatch):
    # 경로: providers/jetbrains.py :: _grazie_headers() auth_mode validation
    monkeypatch.setenv("JETBRAINS_AI_AUTH_MODE", "quantum")
    try:
        headers = _grazie_headers("gpt-5.4", "jb-token")
    except VertexAPIError as exc:
        assert exc.status_code in (400, 503)
        return
    assert "Authorization" not in headers, (
        f"unknown auth_mode silently became bearer: {_dump(headers)}"
    )
