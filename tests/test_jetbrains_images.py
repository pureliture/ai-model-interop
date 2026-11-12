from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.testclient import TestClient

import openai_compatible_bridge.main as main
from openai_compatible_bridge.anthropic_ingress import anthropic_messages_to_openai
from openai_compatible_bridge.core.images import normalize_content
from openai_compatible_bridge.core.transforms import _google_contents_and_system, _xai_input_messages
from openai_compatible_bridge.providers.jetbrains import JetBrainsChatClient
from openai_compatible_bridge.providers.vertex import VertexAPIError

PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAusB9Wl2nWQAAAAASUVORK5CYII="
)
DATA_URL = f"data:image/png;base64,{PNG}"
IMAGE_BLOCK = {"type": "image_url", "image_url": {"url": DATA_URL}}
USER = [{"role": "user", "content": [
    {"type": "text", "text": "describe"},
    IMAGE_BLOCK,
]}]


def test_normalize_accepts_ordered_text_and_data_image():
    assert normalize_content(USER[0]["content"]) == [
        {"type": "text", "text": "describe"},
        {"type": "image", "mime_type": "image/png", "data": PNG},
    ]


@pytest.mark.parametrize("url", [
    "https://example.test/image.png",
    "data:image/svg+xml;base64,PHN2Zy8+",
    "data:image/png;base64,not-base64!",
])
def test_normalize_rejects_remote_and_invalid_images(url):
    with pytest.raises(VertexAPIError) as exc:
        normalize_content([{"type": "image_url", "image_url": {"url": url}}])
    assert exc.value.status_code == 400


def test_responses_input_preserves_images_with_tools():
    items = _xai_input_messages(USER)
    assert items == [{"role": "user", "content": [
        {"type": "input_text", "text": "describe"},
        {"type": "input_image", "detail": "auto", "image_url": DATA_URL},
    ]}]


def test_google_contents_preserve_images():
    contents, _ = _google_contents_and_system(USER)
    assert contents == [{"role": "user", "parts": [
        {"text": "describe"},
        {"inlineData": {"mimeType": "image/png", "data": PNG}},
    ]}]


def test_anthropic_body_preserves_images():
    async def run():
        client = JetBrainsChatClient(base_url="https://example.test", token="synthetic")
        await client.http.aclose()
        requests = []
        client.http = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: requests.append(request) or httpx.Response(200, json={
                "content": [{"type": "text", "text": "ok"}], "usage": {"input_tokens": 1, "output_tokens": 1},
            })
        ))
        try:
            body = client._build_request_body(
                protocol="anthropic_messages", model="claude-test", messages=USER,
                max_tokens=64, temperature=None, top_p=None, stop=None,
                response_format=None, stream=False,
            )
        finally:
            await client.close()
        return body
    body = asyncio.run(run())
    assert body["messages"][0]["content"][1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": PNG},
    }


def test_image_tokens_are_forecast_conservatively():
    assert main._estimate_text_tokens(USER[0]["content"]) > main._estimate_text_tokens("describe")


def test_anthropic_ingress_maps_base64_image_to_data_url():
    messages = anthropic_messages_to_openai([{
        "role": "user",
        "content": [
            {"type": "text", "text": "describe"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
        ],
    }])
    assert messages == [{"role": "user", "content": [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": DATA_URL}},
    ]}]


def test_images_outside_user_messages_are_rejected():
    with pytest.raises(VertexAPIError):
        _xai_input_messages([{"role": "system", "content": [IMAGE_BLOCK]}])
    with pytest.raises(VertexAPIError):
        _google_contents_and_system([{"role": "system", "content": [IMAGE_BLOCK]}])


def test_image_capability_advertised_only_when_configured(monkeypatch):
    cfg = {"provider": "jetbrains", "kind": "chat", "provider_model": "gpt-test",
           "protocol": "openai_responses"}
    monkeypatch.delenv("JETBRAINS_AI_IMAGE_MODELS", raising=False)
    assert "input" not in main._discovered_jetbrains_model_object("jetbrains:gpt-test", cfg)
    monkeypatch.setenv("JETBRAINS_AI_IMAGE_MODELS", "gpt-test")
    row = main._discovered_jetbrains_model_object("jetbrains:gpt-test", cfg)
    assert row["input"] == ["text", "image"]
    assert row["capabilities"]["images"] is True
