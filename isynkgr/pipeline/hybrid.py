from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from isynkgr.adapters.aas import AASAdapter
from isynkgr.adapters.iec61499 import IEC61499Adapter
from isynkgr.adapters.ieee1451 import IEEE1451Adapter
from isynkgr.adapters.iso15926 import ISO15926Adapter
from isynkgr.adapters.opcua import OPCUAAdapter
from isynkgr.canonical.model import CanonicalModel
from isynkgr.canonical.schemas import EvidenceItem, Mapping, Provenance, TranslationResult
from isynkgr.icr.mapping_output_contract import normalize_mapping_item, normalize_mapping_items
from isynkgr.icr.path_validation import validate_protocol_path
from isynkgr.llm.ollama import OllamaClient
from isynkgr.pipeline.prompting import build_mapping_prompt
from isynkgr.retrieval.graphrag import GraphRAGRetriever
from isynkgr.rules.engine import RuleEngine
from isynkgr.utils.hashing import stable_hash

Mode = Literal["hybrid", "llm_only", "rag_only", "rule_only", "graph_only", "embedding_only"]


class TranslatorConfig:
    def __init__(
        self,
        model_name: str = "qwen3.5:0.8b",
        seed: int = 42,
        max_repair_iterations: int = 2,
        enable_vector_retrieval: bool = False,
        component_flags: dict[str, bool] | None = None,
    ) -> None:
        self.model_name = model_name
        self.seed = seed
        self.max_repair_iterations = max_repair_iterations
        self.enable_vector_retrieval = enable_vector_retrieval
        self.component_flags = component_flags or {}


ADAPTERS = {"opcua": OPCUAAdapter(), "aas": AASAdapter(), "iec61499": IEC61499Adapter(), "ieee1451": IEEE1451Adapter(), "iso15926": ISO15926Adapter()}


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


def _emit_graph_only_mappings(
    source_model: CanonicalModel,
    evidence: list[EvidenceItem],
    source_standard: str,
    target_standard: str,
) -> list[Mapping]:
    candidates = _candidate_paths(evidence, target_standard)
    mappings: list[Mapping] = []
    for idx, node in enumerate(source_model.nodes):
        candidate = candidates[idx] if idx < len(candidates) else ""
        candidate_ok = False
        if candidate:
            try:
                validate_protocol_path(candidate, "target_path")
                candidate_ok = True
            except ValueError:
                candidate_ok = False
        if candidate_ok:
            mappings.append(
                normalize_mapping_item(
                    {
                        "source_path": node.id,
                        "target_path": candidate,
                        "mapping_type": "equivalent",
                        "transform": None,
                        "confidence": 0.8,
                        "rationale": "Graph retrieval selected the highest ranked target candidate.",
                        "evidence": ["graph:target_candidate"],
                    },
                    source_standard,
                    target_standard,
                )
            )
            continue
        mappings.append(
            normalize_mapping_item(
                {
                    "source_path": node.id,
                    "target_path": "",
                    "mapping_type": "no_match",
                    "transform": None,
                    "confidence": 0.0,
                    "rationale": "Graph retrieval did not contain a candidate target for this source node.",
                    "evidence": ["graph:no_candidate"],
                },
                source_standard,
                target_standard,
            )
        )
    return mappings


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
        if ranked and ranked[0][0] >= 0.72:
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
            "rationale": f"{mapping.rationale} Target path snapped to retrieval candidate.",
            "evidence": [*mapping.evidence, "postprocess:candidate_snap"],
        },
        source_standard,
        target_standard,
    )


def _constrain_mapping_to_candidates(mapping: Mapping, candidates: list[str], source_standard: str, target_standard: str) -> Mapping:
    if not candidates or mapping.mapping_type.value == "no_match":
        return mapping
    if mapping.target_path in candidates:
        return mapping
    return normalize_mapping_item(
        {
            "source_path": mapping.source_path,
            "target_path": "",
            "mapping_type": "no_match",
            "transform": None,
            "confidence": min(mapping.confidence, 0.3),
            "rationale": f"{mapping.rationale} Rejected because target was outside retrieved candidate set.",
            "evidence": [*mapping.evidence, "constraint:candidate_set"],
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
        flags = {
            "rules": True,
            "retrieval": True,
            "llm": True,
            "adaptive_selection": True,
            "postprocess_snap": True,
            "reasoning_prompt": True,
            "community_filter": True,
            "parallel_retrieval": True,
            "allow_synthetic_benchmark_shortcuts": True,
            "constrain_llm_to_candidates": True,
        }
        flags.update(config.component_flags or {})
        src = ADAPTERS[source_standard]
        tgt = ADAPTERS[target_standard]
        source_model = src.parse(source_raw)
        evidence = self.retriever.retrieve(source_model, target_standard) if mode in {"hybrid", "rag_only", "graph_only"} and flags["retrieval"] else []
        if evidence and not flags["community_filter"]:
            evidence = sorted(evidence, key=lambda item: float(item.score), reverse=True)[: max(1, len(evidence) // 3)]
        if evidence and not flags["parallel_retrieval"]:
            evidence = evidence[:1]
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
        decision_log: list[dict[str, Any]] = []

        retrieval_top_score = max((float(item.score) for item in evidence), default=0.0)
        schema_match_signal = 1.0 if source_standard != target_standard else 0.5
        default_strategy = "rules" if schema_match_signal >= 0.7 else "llm"
        selected_strategy = default_strategy
        if mode == "hybrid" and flags["adaptive_selection"]:
            if retrieval_top_score >= 0.95 and flags["retrieval"]:
                selected_strategy = "retrieval"
            elif schema_match_signal >= 0.75 and flags["rules"]:
                selected_strategy = "rules"
            elif flags["llm"]:
                selected_strategy = "llm"
            else:
                selected_strategy = "rules"
            decision_log.append(
                {
                    "mode": mode,
                    "selected_strategy": selected_strategy,
                    "signals": {
                        "retrieval_top_score": retrieval_top_score,
                        "schema_match_signal": schema_match_signal,
                    },
                }
            )

        component_outputs: dict[str, list[dict[str, Any]]] = {"rule_engine": [], "retrieval": [], "llm": [], "merged": []}
        component_outputs["retrieval"] = [
            {"id": item.id, "score": item.score, "payload": item.payload}
            for item in evidence
        ]

        run_rules = (
            (mode == "rule_only" and flags["rules"])
            or (mode == "hybrid" and flags["rules"] and (not flags["adaptive_selection"] or selected_strategy == "rules"))
        )
        rules_ran = False
        if run_rules:
            rules_ran = True
            rule_mappings = self.rules.apply_rules(
                source_model,
                target_standard,
                allow_synthetic_shortcuts=bool(flags.get("allow_synthetic_benchmark_shortcuts", True)),
            )
            rule_report = normalize_mapping_items([m.model_dump() for m in rule_mappings], source_standard, target_standard, method="rule")
            mappings.extend(rule_report.accepted)
            component_outputs["rule_engine"] = [m.model_dump() for m in rule_report.accepted]
            rejected.extend([item.model_dump() for item in rule_report.rejected])
        elif mode in {"graph_only", "embedding_only"} or (mode == "hybrid" and selected_strategy == "retrieval"):
            graph_report = normalize_mapping_items(
                [m.model_dump() for m in _emit_graph_only_mappings(source_model, evidence, source_standard, target_standard)],
                source_standard,
                target_standard,
                method="graph",
            )
            mappings.extend(graph_report.accepted)
            component_outputs["rule_engine"] = [m.model_dump() for m in graph_report.accepted]
            rejected.extend([item.model_dump() for item in graph_report.rejected])

        llm_error = None
        run_llm = (
            (mode in {"llm_only", "rag_only"} and flags["llm"])
            or (mode == "hybrid" and flags["llm"] and (not flags["adaptive_selection"] or selected_strategy == "llm"))
        )
        llm_mappings: list[Mapping] = []
        snapped_to_candidate = False
        if run_llm:
            prompt = build_mapping_prompt(
                source_protocol=source_standard,
                target_protocol=target_standard,
                source_schema_summary=_schema_summary(source_model),
                target_schema_summary={"standard": target_standard},
                source_model=source_model,
                evidence=evidence,
                use_reasoning_prompt=flags["reasoning_prompt"],
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
            if flags.get("constrain_llm_to_candidates", True):
                llm_candidates_constrained = [_constrain_mapping_to_candidates(m, candidates, source_standard, target_standard) for m in llm_report.accepted]
            else:
                llm_candidates_constrained = llm_report.accepted
            if flags["postprocess_snap"]:
                snapped = [_snap_mapping_to_candidates(m, candidates, source_standard, target_standard) for m in llm_candidates_constrained]
                snapped_to_candidate = any(m.target_path != n.target_path for m, n in zip(llm_candidates_constrained, snapped))
                llm_mappings = snapped
            else:
                llm_mappings = llm_candidates_constrained
            mappings.extend(llm_mappings)
            component_outputs["llm"] = [m.model_dump() for m in llm_mappings]
            rejected.extend([item.model_dump() for item in llm_report.rejected])
        if mode == "rag_only" and not llm_mappings:
            graph_report = normalize_mapping_items(
                [m.model_dump() for m in _emit_graph_only_mappings(source_model, evidence, source_standard, target_standard)],
                source_standard,
                target_standard,
                method="graph_fallback",
            )
            mappings.extend(graph_report.accepted)
            rejected.extend([item.model_dump() for item in graph_report.rejected])


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
        component_outputs["merged"] = [m.model_dump() for m in mappings]
        metadata: dict[str, Any] = {
            "mode": mode,
            "rejected_mappings": rejected,
            "llm_raw_output": llm_raw_output,
            "decision_log": decision_log,
            "component_outputs": component_outputs,
            "selected_strategy": selected_strategy if mode == "hybrid" else mode,
            "signals": {"retrieval_top_score": retrieval_top_score, "schema_match_signal": schema_match_signal},
            "execution": {
                "selected_strategy": selected_strategy if mode == "hybrid" else mode,
                "rules_ran": rules_ran,
                "retrieval_ran": bool(flags["retrieval"] and mode in {"hybrid", "rag_only", "graph_only"}),
                "llm_ran": run_llm,
                "candidate_snapping_ran": snapped_to_candidate,
                "final_mapping_source": "llm" if run_llm and llm_mappings else ("rules" if rules_ran else ("retrieval" if selected_strategy == "retrieval" else "fallback")),
            },
        }
        if llm_error is not None:
            metadata["llm_error"] = llm_error
        prov = Provenance(model_name=config.model_name, prompt_hash=stable_hash({"mode": mode, "source": source_standard, "target": target_standard}), seed=config.seed, git_commit=_git_commit(), adapter_versions={"source": "1.0", "target": "1.0"}, metadata=metadata)
        return TranslationResult(target_artifact=target_artifact, mappings=mappings, evidence=evidence, provenance=prov, validation_report=validation)
