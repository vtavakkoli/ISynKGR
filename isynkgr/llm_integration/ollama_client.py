from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import request

from isynkgr.common import stable_hash

logger = logging.getLogger(__name__)


@dataclass
class OllamaClient:
    base_url: str = "http://ollama:11434"
    model: str = "qwen3:0.6b"
    cache_dir: Path = Path("cache/llm")
    timeout_s: int = 120

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, prompt: str, context: dict[str, Any] | None = None) -> Path:
        ctx = json.dumps(context or {}, sort_keys=True)
        return self.cache_dir / f"{stable_hash(self.model + prompt + ctx)}.json"

    def generate(self, prompt: str, context: dict[str, Any] | None = None, json_mode: bool = True) -> dict[str, Any]:
        key = self._cache_key(prompt, context)
        if key.exists():
            return json.loads(key.read_text())
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "seed": 145162578},
            "format": "json" if json_mode else "",
        }
        url = f"{self.base_url}/api/generate"
        req = request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        with request.urlopen(req, timeout=self.timeout_s) as resp:  # nosec B310
            body = json.loads(resp.read().decode("utf-8"))
        text = body.get("response", "").strip()
        parsed = self._parse_response(text)
        result = {
            "model": self.model,
            "prompt": prompt,
            "context": context or {},
            "raw": text,
            "parsed": parsed,
            "eval_count": body.get("eval_count", 0),
            "prompt_eval_count": body.get("prompt_eval_count", 0),
            "total_duration": body.get("total_duration", 0),
        }
        key.write_text(json.dumps(result, indent=2, sort_keys=True))
        return result

    @staticmethod
    def _parse_response(text: str) -> dict[str, Any]:
        if not text:
            return {"status": "empty", "explanation": ""}
        try:
            loaded = json.loads(text)
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            pass
        return {"status": "unstructured", "explanation": text}
