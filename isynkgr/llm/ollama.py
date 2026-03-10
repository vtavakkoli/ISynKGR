from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from isynkgr.utils.caching import JsonCache
from isynkgr.utils.hashing import stable_hash


class OllamaClient:
    def __init__(self, model: str = "qwen3.5:0.8b", base_url: str | None = None, cache: JsonCache | None = None) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")).rstrip("/")
        self.cache = cache or JsonCache()
        self.last_error: dict | None = None
        self.io_log_path = Path(os.getenv("OLLAMA_IO_LOG", "output/ollama_io.jsonl"))
        self.io_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log_io(self, event: str, payload: dict) -> None:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "model": self.model,
            **payload,
        }
        with self.io_log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _parse_model_json(response_text: str) -> dict:
        text = (response_text or "").strip()
        if not text:
            raise ValueError("Ollama returned empty model response")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError(f"Model response is not JSON: {text[:200]}") from None
            parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Model response JSON must be an object")
        return parsed

    def _call_generate(self, endpoint: str, prompt: str, seed: int) -> dict:
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"seed": seed, "temperature": 0},
        }
        self._log_io("request", {"endpoint": endpoint, "request_body": body})
        req = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw_text = resp.read().decode(errors="replace").strip()
        if not raw_text:
            raise ValueError("Empty response body from Ollama")
        raw = json.loads(raw_text)
        parsed = self._parse_model_json(str(raw.get("response", "")))
        self._log_io("response", {"endpoint": endpoint, "response_body": raw, "parsed": parsed})
        return parsed

    def complete_json(self, prompt: str, schema_name: str, seed: int) -> dict:
        key = stable_hash({"m": self.model, "p": prompt, "s": schema_name, "seed": seed})
        cached = self.cache.get(key)
        if cached and not cached.get("_llm_error"):
            self.last_error = cached.get("_llm_error")
            self._log_io("cache_hit", {"cache_key": key, "schema_name": schema_name, "cached": cached})
            return cached

        endpoint = f"{self.base_url}/api/generate"
        try:
            parsed = self._call_generate(endpoint, prompt, seed)
            self.last_error = None
            self.cache.set(key, parsed)
            return parsed
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode()
            except Exception:
                body = ""
            errors = [{"endpoint": endpoint, "status": exc.code, "reason": str(exc.reason), "body": body[:2000]}]
        except Exception as exc:
            errors = [{"endpoint": endpoint, "error": str(exc)}]

        llm_error = {
            "type": "llm_request_failed",
            "message": "Ollama request failed",
            "attempts": errors,
            "hint": "Check OLLAMA_BASE_URL and /api/generate availability.",
        }
        parsed = {"mappings": [], "_llm_error": llm_error}
        self.last_error = llm_error
        self._log_io("error", {"endpoint": endpoint, "error": llm_error, "prompt": prompt})
        return parsed
