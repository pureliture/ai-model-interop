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
        image = raw.get("image_url")
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
