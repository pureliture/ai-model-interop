"""End-to-end FastAPI route tests for POST /v1/responses (Codex -> Claude / Gemini / GPT)."""

from __future__ import annotations

import asyncio
import json

import httpx
from fastapi.testclient import TestClient

from openai_compatible_bridge.core.conversation_state import ConversationStateCache
from openai_compatible_bridge.main import create_app
from openai_compatible_bridge.providers.jetbrains import JetBrainsChatClient


class _FakeProvider:
    async def close(self) -> None:
        pass


def _make_app(jetbrains_client: JetBrainsChatClient):
    return create_app(
        embedding_client_factory=_FakeProvider,
        chat_client_factory=_FakeProvider,
        rerank_client_factory=_FakeProvider,
        ollama_chat_client_factory=_FakeProvider,
        foundry_chat_client_factory=_FakeProvider,
        jetbrains_chat_client_factory=lambda: jetbrains_client,
        cost_accounting_factory=lambda: None,
    )


def test_codex_to_claude_non_stream_and_sse_stream() -> None:
    captured_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/messages"
        assert request.headers["x-llm-model"] == "anthropic"
        body = json.loads(request.content.decode())
        captured_bodies.append(body)
        if body.get("stream"):
            sse = (
                'data: {"type":"message_start","message":{"usage":{"input_tokens":11}}}\n\n'
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Refactored "}}\n\n'
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"UserService."}}\n\n'
                'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}\n\n'
                'data: {"type":"message_stop"}\n\n'
            )
            return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "Refactored UserService."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 11, "output_tokens": 7},
            },
        )

    jb = JetBrainsChatClient(base_url="https://grazie.test", token="jb-token")
    asyncio.run(jb.http.aclose())
    jb.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = _make_app(jb)

    try:
        with TestClient(app) as client:
            # Non-streaming
            res = client.post(
                "/v1/responses",
                json={
                    "model": "jetbrains:claude-sonnet-4-5",
                    "instructions": "You are Codex.",
                    "input": "Analyze UserService.",
                    "reasoning": {"effort": "high"},
                    "max_output_tokens": 2048,
                },
            )
            assert res.status_code == 200, res.text
            data = res.json()
            assert data["object"] == "response"
            assert data["status"] == "completed"
            assert data["output_text"] == "Refactored UserService."
            assert data["output"][0]["type"] == "message"
            assert data["output"][0]["content"][0]["text"] == "Refactored UserService."
            assert data["usage"] == {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}

            # Verify upstream Anthropic request had thinking budget and adjusted max_tokens
            assert captured_bodies[0]["system"] == "You are Codex."
            assert captured_bodies[0]["thinking"] == {"type": "enabled", "budget_tokens": 16384}
            assert captured_bodies[0]["max_tokens"] == 16384 + 4096

            # SSE streaming
            with client.stream(
                "POST",
                "/v1/responses",
                json={
                    "model": "jetbrains:claude-sonnet-4-5",
                    "instructions": "You are Codex.",
                    "input": "Analyze UserService.",
                    "stream": True,
                },
            ) as stream_res:
                assert stream_res.status_code == 200
                raw_sse = stream_res.read().decode()

            events = [
                line[len("event:") :].strip()
                for line in raw_sse.splitlines()
                if line.startswith("event:")
            ]
            assert events[0] == "response.created"
            assert events[1] == "response.in_progress"
            assert "response.output_item.added" in events
            assert "response.content_part.added" in events
            assert "response.output_text.delta" in events
            assert "response.output_text.done" in events
            assert "response.output_item.done" in events
            assert events[-1] == "response.completed"
    finally:
        asyncio.run(jb.close())


def test_codex_to_claude_apply_patch_custom_tool_and_previous_response_id() -> None:
    captured_bodies: list[dict] = []
    patch_payload = "*** Begin Patch\n*** Update File: src/user.py\n@@\n-pass\n+return True\n*** End Patch"

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        captured_bodies.append(body)
        if len(captured_bodies) == 1:
            return httpx.Response(
                200,
                json={
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_patch_01",
                            "name": "apply_patch",
                            "input": {"input": patch_payload},
                        }
                    ],
                    "stop_reason": "tool_use",
                    "usage": {"input_tokens": 20, "output_tokens": 15},
                },
            )
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "Patch applied and verified."}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 35, "output_tokens": 10},
            },
        )

    state_cache = ConversationStateCache(maxsize=32, ttl_seconds=300)
    jb = JetBrainsChatClient(base_url="https://grazie.test", token="jb-token", state_cache=state_cache)
    asyncio.run(jb.http.aclose())
    jb.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = _make_app(jb)

    try:
        with TestClient(app) as client:
            app.state.conversation_state = state_cache
            # Turn 1: Codex sends freeform custom tool `apply_patch` + consecutive developer/user items
            turn1 = client.post(
                "/v1/responses",
                json={
                    "model": "jetbrains:claude-sonnet-4-5",
                    "instructions": "System instructions",
                    "input": [
                        {"type": "message", "role": "developer", "content": "AGENTS.md context"},
                        {"type": "message", "role": "user", "content": "Fix src/user.py"},
                        {"type": "message", "role": "developer", "content": "Permissions: workspace-write"},
                    ],
                    "tools": [
                        {"type": "custom", "name": "apply_patch", "description": "Patch files"},
                    ],
                    "parallel_tool_calls": False,
                },
            )
            assert turn1.status_code == 200, turn1.text
            t1_data = turn1.json()
            resp_id_1 = t1_data["id"]
            assert t1_data["output"][0] == {
                "id": t1_data["output"][0]["id"],
                "type": "custom_tool_call",
                "status": "completed",
                "call_id": "call_patch_01",
                "name": "apply_patch",
                "input": patch_payload,
            }

            # Verify upstream Anthropic request merged consecutive user/developer turns and mapped parallel_tool_calls=False
            assert len(captured_bodies[0]["messages"]) == 1
            assert captured_bodies[0]["messages"][0]["role"] == "user"
            assert "Fix src/user.py" in captured_bodies[0]["messages"][0]["content"]
            assert "Permissions: workspace-write" in captured_bodies[0]["messages"][0]["content"]
            assert captured_bodies[0]["tools"][0]["name"] == "apply_patch"
            assert captured_bodies[0]["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}

            # Turn 2: Codex continues using previous_response_id and custom_tool_call_output
            turn2 = client.post(
                "/v1/responses",
                json={
                    "model": "jetbrains:claude-sonnet-4-5",
                    "previous_response_id": resp_id_1,
                    "input": [
                        {
                            "type": "custom_tool_call_output",
                            "call_id": "call_patch_01",
                            "output": "Success. Updated the following files:\nM src/user.py",
                        }
                    ],
                },
            )
            assert turn2.status_code == 200, turn2.text
            t2_data = turn2.json()
            assert t2_data["output_text"] == "Patch applied and verified."
            # Upstream Turn 2 should have 3 messages: user -> assistant (tool_use) -> user (tool_result)
            assert len(captured_bodies[1]["messages"]) == 3
            assert captured_bodies[1]["messages"][1]["role"] == "assistant"
            assert captured_bodies[1]["messages"][1]["content"][0]["type"] == "tool_use"
            assert captured_bodies[1]["messages"][2]["role"] == "user"
            assert captured_bodies[1]["messages"][2]["content"][0]["type"] == "tool_result"
            assert captured_bodies[1]["messages"][2]["content"][0]["tool_use_id"] == "call_patch_01"

            # Expired/unknown previous_response_id returns HTTP 400
            bad_prev = client.post(
                "/v1/responses",
                json={
                    "model": "jetbrains:claude-sonnet-4-5",
                    "previous_response_id": "resp_does_not_exist",
                    "input": "continue",
                },
            )
            assert bad_prev.status_code == 400
            assert bad_prev.json()["error"]["param"] == "previous_response_id"

            # Unmappable reasoning effort returns HTTP 400
            bad_effort = client.post(
                "/v1/responses",
                json={
                    "model": "jetbrains:claude-sonnet-4-5",
                    "input": "hi",
                    "reasoning": {"effort": "ultracode"},
                },
            )
            assert bad_effort.status_code == 400
            assert bad_effort.json()["error"]["code"] == "unsupported_parameter"
    finally:
        asyncio.run(jb.close())
