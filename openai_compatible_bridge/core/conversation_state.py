"""Bounded in-memory conversation state cache (RAM only, TTL + LRU eviction).

Stores:
1. Prior conversation messages and custom tool names keyed by `response_id`
   for stateful OpenAI Responses API (`previous_response_id`) continuation.
2. Gemini `functionCall` parts (including `thoughtSignature` and native `id`)
   keyed by `tool_call_id` so stateless clients (Claude Code, Codex, Chat Completions)
   can complete multi-turn Gemini tool calls without losing signatures.
"""

from __future__ import annotations

import copy
import os
import threading
from typing import Any, Callable

from cachetools import TTLCache


def _env_positive_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        val = int(raw)
        return val if val > 0 else default
    except ValueError:
        return default


DEFAULT_STATE_CACHE_MAXSIZE = _env_positive_int("BRIDGE_STATE_CACHE_MAXSIZE", 512)
DEFAULT_STATE_CACHE_TTL_SECONDS = _env_positive_int("BRIDGE_STATE_CACHE_TTL_SECONDS", 3600)


class ConversationStateCache:
    """Thread-safe, bounded in-memory TTL/LRU cache for multi-turn bridge state."""

    def __init__(
        self,
        *,
        maxsize: int | None = None,
        ttl_seconds: int | float | None = None,
        timer: Callable[[], float] | None = None,
    ) -> None:
        resolved_maxsize = maxsize if maxsize is not None and maxsize > 0 else _env_positive_int("BRIDGE_STATE_CACHE_MAXSIZE", 512)
        resolved_ttl = ttl_seconds if ttl_seconds is not None and ttl_seconds > 0 else _env_positive_int("BRIDGE_STATE_CACHE_TTL_SECONDS", 3600)
        self.maxsize = int(resolved_maxsize)
        self.ttl_seconds = float(resolved_ttl)
        cache_kwargs: dict[str, Any] = {"maxsize": self.maxsize, "ttl": self.ttl_seconds}
        if timer is not None:
            cache_kwargs["timer"] = timer
        self._responses: TTLCache[str, dict[str, Any]] = TTLCache(**cache_kwargs)
        self._google_call_parts: TTLCache[str, dict[str, Any]] = TTLCache(**cache_kwargs)
        self._lock = threading.RLock()

    def save_response(
        self,
        response_id: str,
        *,
        messages: list[dict[str, Any]],
        custom_tool_names: set[str] | list[str] | None = None,
    ) -> None:
        if not response_id or not isinstance(response_id, str):
            return
        entry = {
            "messages": copy.deepcopy(messages),
            "custom_tool_names": set(custom_tool_names or ()),
        }
        with self._lock:
            self._responses[response_id] = entry

    save_response_state = save_response

    def get_response(self, response_id: str) -> dict[str, Any] | None:
        if not response_id or not isinstance(response_id, str):
            return None
        with self._lock:
            entry = self._responses.get(response_id)
            if entry is None:
                return None
            return {
                "messages": copy.deepcopy(entry.get("messages", [])),
                "custom_tool_names": set(entry.get("custom_tool_names") or ()),
            }

    get_response_state = get_response

    def save_google_call_part(self, tool_call_id: str, part: dict[str, Any]) -> None:
        if not tool_call_id or not isinstance(tool_call_id, str) or not isinstance(part, dict):
            return
        with self._lock:
            self._google_call_parts[tool_call_id] = copy.deepcopy(part)

    save_tool_call_meta = save_google_call_part

    def get_google_call_part(self, tool_call_id: str) -> dict[str, Any] | None:
        if not tool_call_id or not isinstance(tool_call_id, str):
            return None
        with self._lock:
            part = self._google_call_parts.get(tool_call_id)
            return copy.deepcopy(part) if isinstance(part, dict) else None

    get_tool_call_meta = get_google_call_part

    def clear(self) -> None:
        with self._lock:
            self._responses.clear()
            self._google_call_parts.clear()


DEFAULT_CONVERSATION_STATE_CACHE = ConversationStateCache()


def get_default_conversation_state_cache() -> ConversationStateCache:
    return DEFAULT_CONVERSATION_STATE_CACHE
