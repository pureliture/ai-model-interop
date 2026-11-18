"""Optional provider HTTP timeout configuration."""
import math
import os


def http_timeout_seconds(provider_key: str | None = None) -> float | None:
    for key in (provider_key, "HTTP_TIMEOUT_SECONDS"):
        if not key:
            continue
        raw = os.getenv(key, "").strip()
        if raw:
            value = float(raw)
            if not math.isfinite(value) or not 0.01 <= value <= 3600:
                raise ValueError(f"{key} must be between 0.01 and 3600 seconds")
            return value
    return None
