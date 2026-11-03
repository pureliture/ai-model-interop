"""End-to-end tests for Claude Code (POST /v1/messages) -> JetBrains GPT, Gemini, Grok, and Claude."""

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


def test_claude_code_dynamic_routing_and_effort_mapping_across_gpt_grok_claude() -> None:
    captured: list[tuple[str, str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        llm_header = request.headers.get("x-llm-model", "")
        body = json.loads(request.content.decode())
        captured.append((request.url.path, llm_header, body))
        if llm_header == "anthropic":
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "text", "text": "hello from claude"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 4, "output_tokens": 3},
                },
            )
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": f"hello from {llm_header}"}],
                    }
                ],
                "usage": {"input_tokens": 4, "output_tokens": 3, "total_tokens": 7},
            },
        )

    jb = JetBrainsChatClient(base_url="https://grazie.test", token="jb-tok")
    asyncio.run(jb.http.aclose())
    jb.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = _make_app(jb)

    try:
        with TestClient(app) as client:
            # 1. Dynamic GPT route with output_config.effort = "high"
            r_gpt = client.post(
                "/v1/messages",
                json={
                    "model": "jetbrains:gpt-4.1",
                    "max_tokens": 512,
                    "output_config": {"effort": "high"},
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert r_gpt.status_code == 200, r_gpt.text
            assert r_gpt.json()["content"][0]["text"] == "hello from openai"
            assert captured[0][0] == "/v1/responses"
            assert captured[0][1] == "openai"
            assert captured[0][2]["reasoning"] == {"effort": "high"}

            # 2. Dynamic Grok route with output_config.effort = "max" -> clamped to "xhigh"
            r_grok = client.post(
                "/v1/messages",
                json={
                    "model": "jetbrains:grok-3",
                    "max_tokens": 512,
                    "output_config": {"effort": "max"},
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert r_grok.status_code == 200, r_grok.text
            assert r_grok.json()["content"][0]["text"] == "hello from grok"
            assert captured[1][0] == "/v1/responses"
            assert captured[1][1] == "grok"
            assert captured[1][2]["reasoning"] == {"effort": "xhigh"}

            # 3. Dynamic Claude route with output_config.effort = "medium" and disable_parallel_tool_use
            r_claude = client.post(
                "/v1/messages",
                json={
                    "model": "jetbrains:claude-sonnet-4-5",
                    "max_tokens": 1024,
                    "output_config": {"effort": "medium"},
                    "tools": [{"name": "bash", "input_schema": {"type": "object"}}],
                    "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert r_claude.status_code == 200, r_claude.text
            assert r_claude.json()["content"][0]["text"] == "hello from claude"
            assert captured[2][0] == "/v1/messages"
            assert captured[2][1] == "anthropic"
            assert captured[2][2]["thinking"] == {"type": "enabled", "budget_tokens": 4096}
            assert captured[2][2]["max_tokens"] == 4096 + 4096
            assert captured[2][2]["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    finally:
        asyncio.run(jb.close())


def test_claude_code_to_gemini_multi_turn_tool_call_restores_thought_signature() -> None:
    captured_bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-llm-model"] == "google"
        body = json.loads(request.content.decode())
        captured_bodies.append(body)
        if len(captured_bodies) == 1:
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "functionCall": {"name": "read_file", "args": {"path": "main.py"}},
                                        "thoughtSignature": "gemini-sig-claude-code",
                                    }
                                ]
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {"promptTokenCount": 6, "candidatesTokenCount": 4, "totalTokenCount": 10},
                },
            )
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Done reading main.py."}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 5, "totalTokenCount": 17},
            },
        )

    state_cache = ConversationStateCache(maxsize=32, ttl_seconds=300)
    jb = JetBrainsChatClient(base_url="https://grazie.test", token="jb-tok", state_cache=state_cache)
    asyncio.run(jb.http.aclose())
    jb.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = _make_app(jb)

    try:
        with TestClient(app) as client:
            app.state.conversation_state = state_cache
            # Turn 1: Claude Code requests tool call on Gemini with effort="low"
            t1 = client.post(
                "/v1/messages",
                json={
                    "model": "jetbrains:gemini-2.5-pro",
                    "max_tokens": 1024,
                    "output_config": {"effort": "low"},
                    "tools": [{"name": "read_file", "input_schema": {"type": "object"}}],
                    "messages": [{"role": "user", "content": "Read main.py"}],
                },
            )
            assert t1.status_code == 200, t1.text
            t1_body = t1.json()
            assert t1_body["stop_reason"] == "tool_use"
            tool_use_block = t1_body["content"][0]
            tool_use_id = tool_use_block["id"]
            assert captured_bodies[0]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 1024}

            # Turn 2: Claude Code sends back tool_use and tool_result (without Gemini thoughtSignature)
            t2 = client.post(
                "/v1/messages",
                json={
                    "model": "jetbrains:gemini-2.5-pro",
                    "max_tokens": 1024,
                    "tools": [{"name": "read_file", "input_schema": {"type": "object"}}],
                    "messages": [
                        {"role": "user", "content": "Read main.py"},
                        {"role": "assistant", "content": [tool_use_block]},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": "print('hello')",
                                }
                            ],
                        },
                    ],
                },
            )
            assert t2.status_code == 200, t2.text
            assert t2.json()["content"][0]["text"] == "Done reading main.py."
            model_turn = next(c for c in captured_bodies[1]["contents"] if c["role"] == "model")
            assert model_turn["parts"][0]["thoughtSignature"] == "gemini-sig-claude-code"
            assert model_turn["parts"][0]["functionCall"] == {"name": "read_file", "args": {"path": "main.py"}}
    finally:
        asyncio.run(jb.close())
