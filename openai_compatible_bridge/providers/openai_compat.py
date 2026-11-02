"""Backward-compatible alias for the renamed provider module."""
import sys

from openai_compatible_bridge.providers import openai_compatible as _module

# Preserve module identity so existing monkeypatches and configuration continue
# to affect the implementation, including its private helper names.
sys.modules[__name__] = _module
