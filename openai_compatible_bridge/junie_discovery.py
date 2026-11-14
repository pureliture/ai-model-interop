"""Bounded, read-only Junie ACP model discovery."""
from __future__ import annotations

import asyncio
import json
import glob
import os
import re
import time
from pathlib import Path
from typing import Any

_PREFIX = "v1:12:jetbrains-ai:"
_NATIVE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
_EFFORT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_PROBE_BUDGET = 4.0
_PROBE_TIMEOUT = 0.5
_MAX_MODEL_PROBES = 64


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
                model: dict[str, Any] = {
                    "provider": "jetbrains", "kind": "chat",
                    "provider_model": native, "protocol": _protocol(native),
                }
                label = entry.get("name")
                if isinstance(label, str):
                    display = " ".join(label.split())
                    if display and len(display) <= 120:
                        model["display_name"] = display
                models["jetbrains:" + native] = model
    if not found_group:
        raise ValueError("missing JetBrains AI model group")
    return models


def _selected_efforts(result: Any, expected: str) -> list[str] | None:
    """Return only effort values explicitly reported for the selected model."""
    if not isinstance(result, dict) or not isinstance(result.get("configOptions"), list):
        return None
    options = result["configOptions"]
    if not any(isinstance(o, dict) and o.get("id") == "model" and
               o.get("currentValue") == expected for o in options):
        return None
    for option in options:
        if not isinstance(option, dict) or option.get("id") != "effort" or option.get("category") != "thought_level" or option.get("type") != "select":
            continue
        entries = option.get("options")
        if not isinstance(entries, list) or not entries:
            return None
        values = [entry.get("value") if isinstance(entry, dict) else None for entry in entries]
        if any(not isinstance(value, str) or not _EFFORT.fullmatch(value) for value in values):
            return None
        return list(dict.fromkeys(values))
    return None


async def _fetch(command: str, cwd: str) -> dict[str, dict[str, Any]]:
    deadline = time.monotonic() + _PROBE_BUDGET
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
        models = parse_models(result)
        session_id = result.get("sessionId")
        if not isinstance(session_id, str) or not session_id:
            return models
        for identifier, (name, model) in enumerate(list(models.items())[:_MAX_MODEL_PROBES], start=3):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            value = _PREFIX + model["provider_model"]
            try:
                selected = await asyncio.wait_for(request(identifier, "session/set_config_option", {
                    "sessionId": session_id, "configId": "model", "value": value,
                }), timeout=min(_PROBE_TIMEOUT, remaining))
            except asyncio.TimeoutError:
                # A late reply would leave the stream out of sync; keep the snapshot.
                break
            except OSError:
                break
            except (ValueError, json.JSONDecodeError):
                continue
            efforts = _selected_efforts(selected, value)
            if efforts is not None:
                models[name]["reasoning_efforts"] = efforts
        return models
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        await proc.wait()


def _jar_limits(stdout: bytes) -> dict[str, dict[str, Any]]:
    """Parse the helper's `native<TAB>context<TAB>output` rows.

    The helper leaves the output column empty only for a null maxOutputTokens;
    a model missing from the JAR produces no row at all, and a JAR the probe
    could not read produces no output. Keeping those cases apart is what lets
    request handling drop a client default without dropping an explicit cap.
    """
    metadata: dict[str, dict[str, Any]] = {}
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        fields = line.split("\t")
        if len(fields) != 3 or not _NATIVE.fullmatch(fields[0]):
            continue
        limits: dict[str, Any] = {}
        context = fields[1]
        if context.isascii() and context.isdigit() and 0 < int(context) <= 100_000_000:
            limits["context_length"] = int(context)
        output = fields[2]
        if output.isascii() and output.isdigit() and 0 < int(output) <= 100_000_000:
            limits["max_tokens"] = int(output)
        elif output == "":
            limits["max_tokens_unspecified"] = True
        if limits:
            metadata[fields[0]] = limits
    return metadata


async def _jar_metadata(pattern: str) -> dict[str, dict[str, Any]]:
    """Read token limits from an explicitly configured, unambiguous Junie JAR.

    A model the JAR lists with an empty output column declares *no* output cap,
    which is distinct from a model (or a JAR) the probe could not read.
    """
    jars = glob.glob(os.path.expanduser(pattern))
    if len(jars) != 1:
        return {}
    source = Path(__file__).with_name("JunieModelMetadata.java")
    proc = await asyncio.create_subprocess_exec(
        os.getenv("JUNIE_DISCOVERY_JAVA", "java"), "--class-path", jars[0], str(source),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
    except asyncio.TimeoutError:
        return {}
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
    if proc.returncode != 0 or len(stdout) > 1024 * 1024:
        return {}
    return _jar_limits(stdout)


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
                jar_pattern = os.getenv("JUNIE_DISCOVERY_JAR", "")
                if jar_pattern:
                    try:
                        metadata = await _jar_metadata(jar_pattern)
                    except OSError:
                        metadata = {}
                    for model in models.values():
                        model.update(metadata.get(model["provider_model"], {}))
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
