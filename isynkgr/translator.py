from __future__ import annotations

from pathlib import Path
from typing import Literal

from isynkgr.canonical.schemas import TranslationResult
from isynkgr.llm.ollama import OllamaClient
from isynkgr.pipeline.hybrid import HybridPipeline, TranslatorConfig
from isynkgr.retrieval.graphrag import GraphRAGRetriever
from isynkgr.rules.engine import RuleEngine


class Translator:
    def __init__(self, config: TranslatorConfig | None = None) -> None:
        self.config = config or TranslatorConfig()
        self.pipeline = HybridPipeline(llm=OllamaClient(model=self.config.model_name), retriever=GraphRAGRetriever(), rules=RuleEngine())

    def translate(
        self,
        source_standard: str,
        target_standard: str,
        source_artifact_path: str | bytes | dict,
        mode: Literal["hybrid", "llm_only", "rag_only", "rule_only", "graph_only"] = "hybrid",
        config: TranslatorConfig | None = None,
    ) -> TranslationResult:
        cfg = config or self.config
        raw = source_artifact_path
        if isinstance(source_artifact_path, str) and Path(source_artifact_path).exists():
            raw = Path(source_artifact_path).read_bytes()
        return self.pipeline.run(source_standard, target_standard, raw, mode=mode, config=cfg)
