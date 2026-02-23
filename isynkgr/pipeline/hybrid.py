from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


from isynkgr.adapters.aas import AASAdapter
from isynkgr.adapters.opcua import OPCUAAdapter
from isynkgr.canonical.model import CanonicalModel
from isynkgr.canonical.schemas import Mapping, Provenance, TranslationResult
from isynkgr.icr.mapping_schema import ingest_mapping_payload
from isynkgr.llm.ollama import OllamaClient
from isynkgr.retrieval.graphrag import GraphRAGRetriever
from isynkgr.rules.engine import RuleEngine
from isynkgr.utils.hashing import stable_hash

Mode = Literal["hybrid", "llm_only", "rag_only", "rule_only", "graph_only"]


class TranslatorConfig:
    def __init__(self, model_name: str = "qwen3:0.6b", seed: int = 42, max_repair_iterations: int = 2, enable_vector_retrieval: bool = False) -> None:
        self.model_name = model_name
        self.seed = seed
        self.max_repair_iterations = max_repair_iterations
        self.enable_vector_retrieval = enable_vector_retrieval


ADAPTERS = {"opcua": OPCUAAdapter(), "aas": AASAdapter()}


def _mapping_key(mapping: Mapping) -> tuple[str, str, str]:
    return (mapping.source_path, mapping.target_path, str(mapping.mapping_type))


def _git_commit() -> str:
    head = Path(".git/HEAD")
    if not head.exists():
        return "unknown"
    ref = head.read_text().strip()
    if ref.startswith("ref:"):
        p = Path(".git") / ref.split(" ", 1)[1]
        if p.exists():
            return p.read_text().strip()[:12]
    return ref[:12]


@dataclass
class HybridPipeline:
    llm: OllamaClient
    retriever: GraphRAGRetriever
    rules: RuleEngine

    def run(self, source_standard: str, target_standard: str, source_raw: str | bytes | dict, mode: Mode, config: TranslatorConfig) -> TranslationResult:
        src = ADAPTERS[source_standard]
        tgt = ADAPTERS[target_standard]
        source_model = src.parse(source_raw)
        evidence = self.retriever.retrieve(source_model, target_standard) if mode in {"hybrid", "rag_only", "graph_only"} else []

        mappings: list[Mapping] = []
        if mode in {"hybrid", "rule_only"}:
            mappings.extend(self.rules.apply_rules(source_model, None))

        llm_error = None
        if mode in {"hybrid", "llm_only", "rag_only"}:
            prompt = f"Map nodes from {source_standard} to {target_standard}. Output JSON with mappings list, each with source_path,target_path,mapping_type,transform,confidence,rationale,evidence. Source nodes:{[n.model_dump() for n in source_model.nodes[:50]]}. Evidence:{[e.model_dump() for e in evidence[:10]]}"
            raw = self.llm.complete_json(prompt, "MappingList", config.seed)
            llm_error = raw.get("_llm_error")
            for m in raw.get("mappings", []):
                try:
                    mappings.append(ingest_mapping_payload(m, migrate_legacy=True))
                except Exception:
                    continue

        best_by_key: dict[tuple[str, str, str], Mapping] = {}
        for mapping in mappings:
            key = _mapping_key(mapping)
            current = best_by_key.get(key)
            if current is None or mapping.confidence > current.confidence:
                best_by_key[key] = mapping

        mappings = sorted(best_by_key.values(), key=_mapping_key)

        target_model = CanonicalModel(standard=target_standard, nodes=source_model.nodes, edges=source_model.edges)
        target_artifact = tgt.serialize(target_model, [m.model_dump() for m in mappings])
        validation = tgt.validate(target_artifact)
        metadata = {"mode": mode}
        if llm_error is not None:
            metadata["llm_error"] = llm_error
        prov = Provenance(model_name=config.model_name, prompt_hash=stable_hash({"mode": mode, "source": source_standard, "target": target_standard}), seed=config.seed, git_commit=_git_commit(), adapter_versions={"source": "1.0", "target": "1.0"}, metadata=metadata)
        return TranslationResult(target_artifact=target_artifact, mappings=mappings, evidence=evidence, provenance=prov, validation_report=validation)
