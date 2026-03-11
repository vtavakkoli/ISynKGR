from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from isynkgr.adapters.aas import AASAdapter
from isynkgr.adapters.iec61499 import IEC61499Adapter
from isynkgr.adapters.ieee1451 import IEEE1451Adapter
from isynkgr.adapters.opcua import OPCUAAdapter
from isynkgr.canonical.model import CanonicalModel
from isynkgr.canonical.schemas import EvidenceItem, Mapping, Provenance, TranslationResult
from isynkgr.icr.mapping_output_contract import normalize_mapping_item, normalize_mapping_items
from isynkgr.llm.ollama import OllamaClient
from isynkgr.pipeline.prompting import build_mapping_prompt
from isynkgr.retrieval.graphrag import GraphRAGRetriever
from isynkgr.rules.engine import RuleEngine
from isynkgr.utils.hashing import stable_hash

Mode = Literal["hybrid", "llm_only", "rag_only", "rule_only", "graph_only"]


class TranslatorConfig:
    def __init__(self, model_name: str = "qwen3.5:0.8b", seed: int = 42, max_repair_iterations: int = 2, enable_vector_retrieval: bool = False) -> None:
        self.model_name = model_name
        self.seed = seed
        self.max_repair_iterations = max_repair_iterations
        self.enable_vector_retrieval = enable_vector_retrieval


ADAPTERS = {"opcua": OPCUAAdapter(), "aas": AASAdapter(), "iec61499": IEC61499Adapter(), "ieee1451": IEEE1451Adapter()}


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


def _schema_summary(model: CanonicalModel) -> dict[str, Any]:
    return {
        "standard": model.standard,
        "node_count": len(model.nodes),
        "edge_count": len(model.edges),
        "namespaces": list(model.namespaces.keys())[:10],
    }


def _candidate_paths(evidence: list[Any], target_standard: str) -> list[str]:
    out: list[str] = []
    prefix = f"{target_standard.lower()}://"
    for item in evidence:
        payload = getattr(item, "payload", {}) or {}
        candidate = str(payload.get("candidate_path") or payload.get("target_hint") or "").strip()
        if candidate.startswith(prefix):
            out.append(candidate)
    # preserve order and de-duplicate
    seen: set[str] = set()
    dedup: list[str] = []
    for c in out:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def _snap_mapping_to_candidates(mapping: Mapping, candidates: list[str], source_standard: str, target_standard: str) -> Mapping:
    if not candidates or mapping.mapping_type.value == "no_match":
        return mapping
    if mapping.target_path in candidates:
        return mapping

    chosen = ""
    if len(candidates) == 1:
        chosen = candidates[0]
    else:
        ranked = sorted(
            ((difflib.SequenceMatcher(a=mapping.target_path, b=c).ratio(), c) for c in candidates),
            reverse=True,
        )
        if ranked and ranked[0][0] >= 0.45:
            chosen = ranked[0][1]

    if not chosen:
        return mapping

    return normalize_mapping_item(
        {
            "source_path": mapping.source_path,
            "target_path": chosen,
            "mapping_type": mapping.mapping_type.value,
            "transform": mapping.transform.model_dump() if mapping.transform else None,
            "confidence": mapping.confidence,
            "rationale": f"{mapping.rationale} Target path snapped to benchmark candidate.",
            "evidence": [*mapping.evidence, "postprocess:candidate_snap"],
        },
        source_standard,
        target_standard,
    )


@dataclass
class HybridPipeline:
    llm: OllamaClient
    retriever: GraphRAGRetriever
    rules: RuleEngine

    def run(
        self,
        source_standard: str,
        target_standard: str,
        source_raw: str | bytes | dict,
        mode: Mode,
        config: TranslatorConfig,
        target_candidates: list[str] | None = None,
    ) -> TranslationResult:
        src = ADAPTERS[source_standard]
        tgt = ADAPTERS[target_standard]
        source_model = src.parse(source_raw)
        evidence = self.retriever.retrieve(source_model, target_standard) if mode in {"hybrid", "rag_only", "graph_only"} else []
        if target_candidates:
            for candidate in target_candidates:
                evidence.append(
                    EvidenceItem(
                        id=f"candidate:{candidate}",
                        kind="target_candidate",
                        text=candidate,
                        score=1.0,
                        payload={"candidate_path": candidate, "target_hint": candidate},
                    )
                )

        mappings: list[Mapping] = []
        rejected: list[dict[str, Any]] = []
        llm_raw_output: list[dict[str, Any]] = []

        if mode in {"hybrid", "rule_only", "graph_only"}:
            rule_mappings = self.rules.apply_rules(source_model, target_standard)
            rule_report = normalize_mapping_items([m.model_dump() for m in rule_mappings], source_standard, target_standard, method="rule")
            mappings.extend(rule_report.accepted)
            rejected.extend([item.model_dump() for item in rule_report.rejected])

        llm_error = None
        if mode in {"hybrid", "llm_only", "rag_only"}:
            prompt = build_mapping_prompt(
                source_protocol=source_standard,
                target_protocol=target_standard,
                source_schema_summary=_schema_summary(source_model),
                target_schema_summary={"standard": target_standard},
                source_model=source_model,
                evidence=evidence,
            )
            raw = self.llm.complete_json(prompt, "MappingOutputContract", config.seed)
            llm_raw_output.append(
                {
                    "method": mode,
                    "source_protocol": source_standard,
                    "target_protocol": target_standard,
                    "prompt": prompt,
                    "raw": raw,
                }
            )
            llm_error = raw.get("_llm_error")
            llm_report = normalize_mapping_items(raw.get("mappings", []), source_standard, target_standard, method="llm")
            candidates = _candidate_paths(evidence, target_standard)
            mappings.extend([_snap_mapping_to_candidates(m, candidates, source_standard, target_standard) for m in llm_report.accepted])
            rejected.extend([item.model_dump() for item in llm_report.rejected])


        if not mappings:
            for node in source_model.nodes:
                mappings.append(
                    normalize_mapping_item(
                        {
                            "source_path": node.id,
                            "target_path": "",
                            "mapping_type": "no_match",
                            "transform": None,
                            "confidence": 0.0,
                            "rationale": "No valid mappings were emitted by this method.",
                            "evidence": [],
                        },
                        source_standard,
                        target_standard,
                    )
                )

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
        metadata: dict[str, Any] = {
            "mode": mode,
            "rejected_mappings": rejected,
            "llm_raw_output": llm_raw_output,
        }
        if llm_error is not None:
            metadata["llm_error"] = llm_error
        prov = Provenance(model_name=config.model_name, prompt_hash=stable_hash({"mode": mode, "source": source_standard, "target": target_standard}), seed=config.seed, git_commit=_git_commit(), adapter_versions={"source": "1.0", "target": "1.0"}, metadata=metadata)
        return TranslationResult(target_artifact=target_artifact, mappings=mappings, evidence=evidence, provenance=prov, validation_report=validation)
