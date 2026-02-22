from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from isynkgr.utils.caching import JsonCache
from isynkgr.utils.hashing import stable_hash


class OllamaClient:
    def __init__(self, model: str = "qwen3:0.6b", base_url: str | None = None, cache: JsonCache | None = None) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")).rstrip("/")
        self.cache = cache or JsonCache()
        self.last_error: dict | None = None

    def _call_generate(self, endpoint: str, prompt: str, seed: int) -> dict:
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"seed": seed, "temperature": 0},
        }
        req = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode())
            return json.loads(payload.get("response", "{}"))

    def _call_chat_compat(self, endpoint: str, prompt: str, seed: int) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": seed,
        }
        req = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode())
            content = (((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "{}")
            return json.loads(content)

    def complete_json(self, prompt: str, schema_name: str, seed: int) -> dict:
        key = stable_hash({"m": self.model, "p": prompt, "s": schema_name, "seed": seed})
        cached = self.cache.get(key)
        if cached:
            self.last_error = cached.get("_llm_error")
            return cached

        endpoints = [
            (f"{self.base_url}/api/generate", "generate"),
            (f"{self.base_url}/generate", "generate"),
            (f"{self.base_url}/v1/chat/completions", "chat"),
        ]
        parsed: dict = {"mappings": []}
        errors: list[dict] = []

        for endpoint, kind in endpoints:
            try:
                parsed = self._call_generate(endpoint, prompt, seed) if kind == "generate" else self._call_chat_compat(endpoint, prompt, seed)
                self.last_error = None
                self.cache.set(key, parsed)
                return parsed
            except urllib.error.HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode()
                except Exception:
                    body = ""
                errors.append({"endpoint": endpoint, "status": exc.code, "reason": str(exc.reason), "body": body[:500]})
            except Exception as exc:
                errors.append({"endpoint": endpoint, "error": str(exc)})

        llm_error = {
            "type": "llm_request_failed",
            "message": "All LLM endpoints failed",
            "attempts": errors,
            "hint": "Check OLLAMA_BASE_URL and provider API compatibility.",
        }
        parsed = {"mappings": [], "_llm_error": llm_error}
        self.last_error = llm_error
        self.cache.set(key, parsed)
        return parsed
