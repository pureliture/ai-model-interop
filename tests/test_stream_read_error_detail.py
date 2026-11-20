"""Upstream stream read failures keep a short, redacted cause for operators."""
from __future__ import annotations

import asyncio

import httpx
import pytest

from openai_compatible_bridge.providers.jetbrains import JetBrainsChatClient


class _FailingStream(httpx.AsyncByteStream):
    """One SSE line, then a dropped upstream connection."""

    def __init__(self, error: BaseException):
        self.error = error

    async def __aiter__(self):
        yield b'data: {"choices": [{"delta": {"content": "partial"}}]}\n\n'
        raise self.error


def test_jetbrains_stream_read_error_reports_redacted_cause():
    async def run():
        error = httpx.RemoteProtocolError(
            "peer closed connection Bearer super-secret-token access_token=leaky-value"
        )
        client = JetBrainsChatClient(base_url="https://grazie.test", token="test")
        await client.http.aclose()
        client.http = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, stream=_FailingStream(error))
            )
        )
        try:
            with pytest.raises(Exception) as caught:
                _ = [
                    event
                    async for event in client.stream_chat(
                        model="jetbrains:gpt-6.1-terra",
                        messages=[{"role": "user", "content": "ping"}],
                    )
                ]
            err = caught.value
            assert err.status_code == 502
            assert err.code == "connection_error"
            assert err.stage == "provider_read"
            message = err.message
            assert "RemoteProtocolError" in message
            assert "peer closed connection" in message
            assert "super-secret-token" not in message
            assert "leaky-value" not in message
        finally:
            await client.close()

    asyncio.run(run())
