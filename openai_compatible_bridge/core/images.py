"""Strict OpenAI image-block normalization for native provider transforms.

Only client-supplied base64 ``data:`` images are accepted. Remote URLs are
rejected instead of fetched so the bridge cannot be used as an SSRF relay.
"""
from __future__ import annotations

import base64
import binascii
import re
from typing import Any

from openai_compatible_bridge.providers.vertex import VertexAPIError

MAX_IMAGES = 8
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_BYTES = 20 * 1024 * 1024
ALLOWED_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
_DATA_URL = re.compile(r"^data:(image/[A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/=\s]+)$")


def _reject(message: str, param: str = "messages") -> VertexAPIError:
    return VertexAPIError(400, message, code="invalid_request_error", stage=param)


def normalize_content(content: Any) -> list[dict[str, Any]]:
    """Return ordered text/image blocks, rejecting anything that cannot be forwarded safely.

    Text blocks use ``{"type": "text", "text": str}`` and images use
    ``{"type": "image", "mime_type": str, "data": base64}``. Other OpenAI block
    types (audio, files) are rejected rather than silently dropped.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        raise _reject("Message content must be a string or an array of content blocks.")

    blocks: list[dict[str, Any]] = []
    images = 0
    total_bytes = 0
    for raw in content:
        if isinstance(raw, str):
            blocks.append({"type": "text", "text": raw})
            continue
        if not isinstance(raw, dict):
            raise _reject("Invalid content block.")
        kind = raw.get("type")
        if kind in {"text", "input_text"}:
            text = raw.get("text")
            if not isinstance(text, str):
                raise _reject("Text content blocks require a string text field.")
            blocks.append({"type": "text", "text": text})
            continue
        if kind not in {"image_url", "input_image"}:
            raise _reject(f"Unsupported content block type: {kind!r}.")
        if "file_id" in raw:
            raise _reject("Image file_id cannot be represented.", "file_id")
        image = raw.get("image_url")
        if "detail" in raw or (isinstance(image, dict) and "detail" in image):
            raise _reject("Image detail cannot be represented.", "detail")
        if set(raw) - {"type", "image_url"} or (isinstance(image, dict) and set(image) - {"url"}):
            raise _reject("Image fields cannot be represented.", "image_url")
        url = image.get("url") if isinstance(image, dict) else image
        if not isinstance(url, str):
            raise _reject("Image content blocks require a data URL.", "image_url")
        match = _DATA_URL.fullmatch(url.strip())
        if match is None:
            raise _reject(
                "Only base64 data: image URLs are supported; remote URLs are not fetched.",
                "image_url",
            )
        mime_type = match.group(1).lower()
        if mime_type not in ALLOWED_MIME_TYPES:
            raise _reject(
                f"Unsupported image MIME type: {mime_type}. Allowed: {sorted(ALLOWED_MIME_TYPES)}.",
                "image_url",
            )
        data = re.sub(r"\s+", "", match.group(2))
        try:
            decoded = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError):
            raise _reject("Image data is not valid base64.", "image_url") from None
        if not decoded:
            raise _reject("Image data is empty.", "image_url")
        if len(decoded) > MAX_IMAGE_BYTES:
            raise _reject(f"Image exceeds {MAX_IMAGE_BYTES} decoded bytes.", "image_url")
        images += 1
        if images > MAX_IMAGES:
            raise _reject(f"At most {MAX_IMAGES} images are accepted per request.", "image_url")
        total_bytes += len(decoded)
        if total_bytes > MAX_TOTAL_BYTES:
            raise _reject(f"Total image bytes exceed {MAX_TOTAL_BYTES}.", "image_url")
        blocks.append({"type": "image", "mime_type": mime_type, "data": data})
    return blocks


def data_url(block: dict[str, Any]) -> str:
    return f"data:{block['mime_type']};base64,{block['data']}"


def validate_messages(messages: list[dict[str, Any]], provider: str, protocol: str) -> None:
    """Reject message data that the selected outgoing protocol cannot carry."""
    target = f"{provider}/{protocol}"
    chat = protocol == "openai_chat_completions"
    ollama = protocol == "ollama_chat"
    known = {"openai_chat_completions", "openai_responses", "xai_responses", "anthropic_messages", "google_generate_content", "ollama_chat"}
    if protocol not in known:
        raise _reject(f"messages: unsupported target protocol {target}.", "messages")
    roles = {"system", "user", "assistant", "tool"}
    if chat or protocol == "openai_responses":
        roles.add("developer")
    seen_turn = False
    for index, message in enumerate(messages):
        path = f"messages[{index}]"
        def reject(field: str, reason: str) -> None:
            raise _reject(f"{field}: {reason} for {target}.", field)
        role = message.get("role")
        if role not in roles:
            reject(f"{path}.role", f"Unsupported role {role!r}")
        if role == "system" and seen_turn and protocol in {"anthropic_messages", "google_generate_content"}:
            reject(f"{path}.role", "A system turn cannot be preserved at this position")
        if role != "system":
            seen_turn = True
        fields = {"role", "content"}
        if chat:
            fields.add("name")
        if role == "assistant":
            fields.add("tool_calls")
        if role == "tool":
            fields.add("tool_call_id")
            if protocol == "anthropic_messages":
                fields.add("is_error")
            if not message.get("tool_call_id"):
                reject(f"{path}.tool_call_id", "A tool result requires a connection ID")
            if ollama:
                reject(f"{path}.tool_call_id", "Tool result connection ID cannot be represented")
        extra = message.keys() - fields
        if extra:
            reject(f"{path}.{sorted(extra)[0]}", "Field cannot be represented")
        content = message.get("content")
        if isinstance(content, list):
            if ollama:
                reject(f"{path}.content", "Content blocks cannot be represented")
            for block_index, block in enumerate(content):
                block_path = f"{path}.content[{block_index}]"
                if not isinstance(block, dict):
                    reject(block_path, "Content block must be an object")
                kind = block.get("type")
                if kind in {"image_url", "input_image"}:
                    if role != "user":
                        reject(block_path, "Image requires a user role")
                    if ollama:
                        reject(block_path, "Image cannot be represented")
                    try:
                        normalize_content([block])
                    except VertexAPIError as exc:
                        suffix = exc.stage if exc.stage != "messages" else "image_url"
                        reject(f"{block_path}.{suffix}", exc.message)
                elif kind in {"text", "input_text"}:
                    if kind != "text":
                        reject(f"{block_path}.type", "Content block cannot be represented")
                    if not isinstance(block.get("text"), str):
                        reject(f"{block_path}.text", "Text must be a string")
                    extra_block = block.keys() - {"type", "text"}
                    if extra_block:
                        reject(f"{block_path}.{sorted(extra_block)[0]}", "Field cannot be represented")
                else:
                    reject(f"{block_path}.type", "Content block cannot be represented")
            try:
                normalize_content(content)
            except VertexAPIError as exc:
                reject(f"{path}.content", exc.message)
            if role == "tool" and not chat:
                reject(f"{path}.content", "Tool result content blocks cannot be represented")
        elif content is not None and not isinstance(content, str):
            reject(f"{path}.content", "Content cannot be represented")


def anthropic_image_content(content: Any) -> Any:
    """Map OpenAI content blocks to Anthropic text and base64 image blocks."""
    if not isinstance(content, list):
        return content if content is not None else ""
    return [
        {"type": "text", "text": block["text"]} if block["type"] == "text" else
        {"type": "image", "source": {"type": "base64", "media_type": block["mime_type"], "data": block["data"]}}
        for block in normalize_content(content)
    ]
