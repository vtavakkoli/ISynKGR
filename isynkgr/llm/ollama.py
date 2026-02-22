from __future__ import annotations

import json
import os
import urllib.request

from isynkgr.utils.caching import JsonCache
from isynkgr.utils.hashing import stable_hash


class OllamaClient:
    def __init__(self, model: str = "Qwen/Qwen3-0.6B", base_url: str | None = None, cache: JsonCache | None = None) -> None:
        self.model = model
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        self.cache = cache or JsonCache()

    def complete_json(self, prompt: str, schema_name: str, seed: int) -> dict:
        key = stable_hash({"m": self.model, "p": prompt, "s": schema_name, "seed": seed})
        cached = self.cache.get(key)
        if cached:
            return cached
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"seed": seed, "temperature": 0},
        }
        parsed = {"mappings": []}
        try:
            req = urllib.request.Request(f"{self.base_url}/api/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                payload = json.loads(resp.read().decode())
                parsed = json.loads(payload.get("response", "{}"))
        except Exception:
            pass
        self.cache.set(key, parsed)
        return parsed
