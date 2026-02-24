from __future__ import annotations

import json
import pytest

from isynkgr.llm.ollama import OllamaClient
from isynkgr.utils.caching import JsonCache


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_complete_json_ignores_cached_llm_errors_and_retries(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cache = JsonCache(root=str(tmp_path))
    client = OllamaClient(model="demo", base_url="http://example", cache=cache)

    key = "test-key"
    monkeypatch.setattr("isynkgr.llm.ollama.stable_hash", lambda _: key)
    cache.set(key, {"mappings": [], "_llm_error": {"type": "llm_request_failed"}})

    calls = {"n": 0}

    def _fake_urlopen(req, timeout=0):  # noqa: ANN001
        calls["n"] += 1
        return _FakeResponse({"response": json.dumps({"mappings": [{"source_path": "opcua://x", "target_path": "aas://y", "mapping_type": "equivalent"}]})})

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    result = client.complete_json("prompt", "MappingList", 42)

    assert calls["n"] == 1
    assert len(result["mappings"]) == 1
    assert cache.get(key) == result


def test_complete_json_does_not_cache_failed_calls(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cache = JsonCache(root=str(tmp_path))
    client = OllamaClient(model="demo", base_url="http://example", cache=cache)

    key = "test-key-2"
    monkeypatch.setattr("isynkgr.llm.ollama.stable_hash", lambda _: key)

    def _raise(req, timeout=0):  # noqa: ANN001
        raise OSError("network unavailable")

    monkeypatch.setattr("urllib.request.urlopen", _raise)

    result = client.complete_json("prompt", "MappingList", 42)

    assert result.get("_llm_error", {}).get("type") == "llm_request_failed"
    assert cache.get(key) is None
