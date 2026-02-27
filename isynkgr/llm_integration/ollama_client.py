"""Backward-compatible import path for the shared Ollama client.

This module exists to keep older imports working while avoiding duplicated
implementations.
"""

from isynkgr.llm.ollama import OllamaClient

__all__ = ["OllamaClient"]
