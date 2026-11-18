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

            # 2. Dynamic Grok route with output_config.effort = "max" is not supported -> 400
            r_grok = client.post(
                "/v1/messages",
                json={
                    "model": "jetbrains:grok-3",
                    "max_tokens": 512,
                    "output_config": {"effort": "max"},
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert r_grok.status_code == 400, r_grok.text
            assert len(captured) == 1, "unsupported effort must be rejected before upstream"

            # 2b. Supported Grok effort passes through unchanged.
            r_grok_ok = client.post(
                "/v1/messages",
                json={
                    "model": "jetbrains:grok-3",
                    "max_tokens": 512,
                    "output_config": {"effort": "high"},
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert r_grok_ok.status_code == 200, r_grok_ok.text
            assert r_grok_ok.json()["content"][0]["text"] == "hello from grok"
            assert captured[1][0] == "/v1/responses"
            assert captured[1][1] == "grok"
            assert captured[1][2]["reasoning"] == {"effort": "high"}

            # 3. Dynamic Claude route with output_config.effort = "medium" passthrough.
            r_claude = client.post(
                "/v1/messages",
                json={
                    "model": "jetbrains:claude-sonnet-4-5",
                    "max_tokens": 8192,
                    "output_config": {"effort": "medium"},
                    "tools": [{"name": "bash", "input_schema": {"type": "object"}}],
                    "tool_choice": {"type": "auto"},
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert r_claude.status_code == 200, r_claude.text
            assert r_claude.json()["content"][0]["text"] == "hello from claude"
            assert captured[2][0] == "/v1/messages"
            assert captured[2][1] == "anthropic"
            assert captured[2][2]["output_config"] == {"effort": "medium"}
            assert captured[2][2]["thinking"] == {"type": "adaptive"}
            assert captured[2][2]["max_tokens"] == 8192
            assert captured[2][2]["tool_choice"] == {"type": "auto"}

            # 3b. disable_parallel_tool_use is unsupported on this route -> 400, no silent mapping.
            r_parallel = client.post(
                "/v1/messages",
                json={
                    "model": "jetbrains:claude-sonnet-4-5",
                    "max_tokens": 1024,
                    "tools": [{"name": "bash", "input_schema": {"type": "object"}}],
                    "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert r_parallel.status_code == 400, r_parallel.text
            assert len(captured) == 3
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
            assert captured_bodies[0]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "LOW"}

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


def test_base64_images_reach_all_jetbrains_protocols_from_all_ingresses() -> None:
    from openai_compatible_bridge.providers.vertex import MODEL_REGISTRY

    data_url = "data:image/png;base64,YQ=="
    for protocol in ("openai_responses", "xai_responses", "anthropic_messages", "google_generate_content"):
        for ingress in ("chat", "responses", "anthropic"):
            captured = []
            def handler(request: httpx.Request) -> httpx.Response:
                captured.append(json.loads(request.content))
                if protocol == "anthropic_messages":
                    payload = {"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1}}
                elif protocol == "google_generate_content":
                    payload = {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}]}
                else:
                    payload = {"status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]}
                return httpx.Response(200, json=payload)
            jb = JetBrainsChatClient(base_url="https://grazie.test", token="test")
            asyncio.run(jb.http.aclose())
            jb.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            alias = "jetbrains:image-test"
            old = MODEL_REGISTRY.copy()
            MODEL_REGISTRY[alias] = {"provider": "jetbrains", "kind": "chat", "provider_model": "model", "protocol": protocol}
            try:
                if ingress == "chat":
                    path = "/v1/chat/completions"
                    payload = {"model": alias, "messages": [{"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": data_url}}]}]}
                elif ingress == "responses":
                    path = "/v1/responses"
                    payload = {"model": alias, "input": [{"role": "user", "content": [{"type": "input_text", "text": "look"}, {"type": "input_image", "image_url": data_url}]}]}
                else:
                    path = "/v1/messages"
                    payload = {"model": alias, "max_tokens": 32, "messages": [{"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "YQ=="}}]}]}
                with TestClient(_make_app(jb)) as client:
                    response = client.post(path, json=payload)
                assert response.status_code == 200, (protocol, ingress, response.text)
                assert len(captured) == 1
                body = captured[0]
                if protocol in {"openai_responses", "xai_responses"}:
                    assert body["input"][0]["content"][1] == {"type": "input_image", "image_url": data_url}
                elif protocol == "anthropic_messages":
                    assert body["messages"][0]["content"][1] == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "YQ=="}}
                else:
                    assert body["contents"][0]["parts"][1] == {"inlineData": {"mimeType": "image/png", "data": "YQ=="}}
            finally:
                MODEL_REGISTRY.clear()
                MODEL_REGISTRY.update(old)
