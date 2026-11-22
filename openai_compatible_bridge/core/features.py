"""Runtime feature switches for protocol behavior that can be disabled without a rebuild."""

from __future__ import annotations

import os


def reasoning_replay_enabled() -> bool:
    """Capture/emit/replay opaque reasoning items unless explicitly disabled."""
    raw = os.getenv("BRIDGE_REASONING_REPLAY", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}
