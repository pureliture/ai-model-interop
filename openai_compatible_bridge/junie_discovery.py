"""Bounded, read-only Junie ACP model discovery."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any

_PREFIX = "v1:12:jetbrains-ai:"
_NATIVE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")


def _protocol(native: str) -> str | None:
    name = native.lower()
    for prefixes, protocol in (
        (("claude-",), "anthropic_messages"),
        (("gemini-",), "google_generate_content"),
        (("grok-",), "xai_responses"),
        (("gpt-", "openai-gpt-"), "openai_responses"),
    ):
        if name.startswith(prefixes):
            return protocol
    return None


def parse_models(result: Any) -> dict[str, dict[str, Any]]:
    """Accept only model options explicitly belonging to the JetBrains AI group."""
    if not isinstance(result, dict) or not isinstance(result.get("configOptions"), list):
        raise ValueError("invalid ACP session response")
    models: dict[str, dict[str, Any]] = {}
    found_group = False
    for option in result["configOptions"]:
        if not isinstance(option, dict) or option.get("id") != "model":
            continue
        groups = option.get("options", [])
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict) or group.get("group") != "jetbrains-ai":
                continue
            found_group = True
            entries = group.get("options", [])
            if not isinstance(entries, list):
                raise ValueError("invalid JetBrains AI model group")
            for entry in entries:
                value = entry.get("value") if isinstance(entry, dict) else None
                if not isinstance(value, str) or not value.startswith(_PREFIX):
                    continue
                native = value[len(_PREFIX):]
                if not _NATIVE.fullmatch(native) or any(part in {".", ".."} for part in native.split("/")):
                    continue
                models["jetbrains:" + native] = {
                    "provider": "jetbrains", "kind": "chat",
                    "provider_model": native, "protocol": _protocol(native),
                }
    if not found_group:
        raise ValueError("missing JetBrains AI model group")
    return models


async def _fetch(command: str, cwd: str) -> dict[str, dict[str, Any]]:
    proc = await asyncio.create_subprocess_exec(
        command, "--acp=true", "--skip-update-check", cwd=cwd,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL, limit=1024 * 1024,
    )
    assert proc.stdin is not None and proc.stdout is not None
    async def request(identifier: int, method: str, params: dict[str, Any]) -> Any:
        proc.stdin.write((json.dumps({"jsonrpc": "2.0", "id": identifier, "method": method, "params": params}) + "\n").encode())
        await proc.stdin.drain()
        for _ in range(100):
            line = await proc.stdout.readline()
            if not line:
                raise ValueError("ACP closed")
            message = json.loads(line)
            if not isinstance(message, dict):
                continue
            if message.get("id") == identifier and "method" not in message:
                if "error" in message or "result" not in message:
                    raise ValueError("ACP request failed")
                return message["result"]
            if "method" in message and "id" in message:
                proc.stdin.write((json.dumps({"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": "Method not supported"}}) + "\n").encode())
                await proc.stdin.drain()
        raise ValueError("too many ACP messages")
    try:
        await request(1, "initialize", {"protocolVersion": 1, "clientCapabilities": {}})
        result = await request(2, "session/new", {"cwd": cwd, "mcpServers": []})
        return parse_models(result)
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        await proc.wait()


class JunieDiscovery:
    def __init__(self, *, enabled: bool, command: str = "junie", cwd: str | None = None,
                 ttl: float = 300, timeout: float = 10) -> None:
        self.enabled, self.command = enabled, command
        self.cwd = cwd or os.getcwd()
        self.ttl, self.timeout = ttl, timeout
        self.models: dict[str, dict[str, Any]] = {}
        self.stale = False
        self.has_snapshot = False
        self._next_refresh = 0.0
        self._lock = asyncio.Lock()

    async def refresh(self) -> dict[str, dict[str, Any]]:
        if not self.enabled or time.monotonic() < self._next_refresh:
            return self.models
        async with self._lock:
            if time.monotonic() < self._next_refresh:
                return self.models
            try:
                models = await asyncio.wait_for(_fetch(self.command, self.cwd), timeout=self.timeout)
            except (OSError, ValueError, asyncio.TimeoutError, json.JSONDecodeError):
                # Keep the last successful snapshot and retry after a bounded delay.
                self.stale = True
                self._next_refresh = time.monotonic() + min(self.ttl, 30)
            else:
                self.models = models
                self.stale = False
                self.has_snapshot = True
                self._next_refresh = time.monotonic() + self.ttl
            return self.models


def discovery_from_env() -> JunieDiscovery:
    return JunieDiscovery(enabled=os.getenv("JUNIE_DISCOVERY_ENABLED", "false").lower() in {"1", "true", "yes"},
                          command=os.getenv("JUNIE_DISCOVERY_COMMAND", "junie"),
                          cwd=os.getenv("JUNIE_DISCOVERY_CWD") or None)
