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
from isynkgr.canonical.model import CanonicalModel, CanonicalNode
from isynkgr.canonical.schemas import EvidenceItem, Mapping, Provenance, TranslationResult
from isynkgr.icr.mapping_output_contract import normalize_mapping_item, normalize_mapping_items
from isynkgr.icr.entities import build_endpoint_path, normalize_path
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


# Diagnosis note:
# - previous hybrid behavior collapsed to a single branch via selected_strategy.
# - retrieval candidates were flattened globally and mapped by list index.
# - synthetic shortcuts dominated rules unless explicitly disabled.
# - target model for rules/retrieval reused source nodes, which disabled real target reasoning.
# - merge step deduplicated tuples but did not pick a single best mapping per source node.


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


def _build_default_target_model(target_standard: str) -> CanonicalModel:
    std = target_standard.lower()
    if std == "aas":
        labels = ["temperature", "pressure", "flow", "state", "speed", "vibration"]
        nodes = [
            CanonicalNode(id=f"aas://asset/submodel/default/element/{label}/value", type="Property", label=label, attributes={"datatype": "FLOAT" if label != "state" else "STRING"})
            for label in labels
        ]
        return CanonicalModel(standard=std, nodes=nodes, edges=[])
    if std == "iec61499":
        nodes = [
            CanonicalNode(id="iec61499://Device0/Res1/FB1/temp", type="Signal", label="temperature", attributes={"dtype": "FLOAT", "unit": "C"}),
            CanonicalNode(id="iec61499://Device0/Res1/FB1/pressure", type="Signal", label="pressure", attributes={"dtype": "FLOAT", "unit": "bar"}),
            CanonicalNode(id="iec61499://Device0/Res1/FB1/state", type="Signal", label="state", attributes={"dtype": "STRING"}),
        ]
        return CanonicalModel(standard=std, nodes=nodes, edges=[])
    if std == "ieee1451":
        nodes = [
            CanonicalNode(id="ieee1451://teds0/ch0/value", type="Channel", label="temperature", attributes={"dtype": "FLOAT", "unit": "C"}),
            CanonicalNode(id="ieee1451://teds0/ch1/value", type="Channel", label="pressure", attributes={"dtype": "FLOAT", "unit": "bar"}),
        ]
        return CanonicalModel(standard=std, nodes=nodes, edges=[])
    if std == "opcua":
        nodes = [
            CanonicalNode(id="opcua://ns=2;s=Temperature", type="UAVariable", label="temperature", attributes={"datatype": "FLOAT", "unit": "C"}),
            CanonicalNode(id="opcua://ns=2;s=Pressure", type="UAVariable", label="pressure", attributes={"datatype": "FLOAT", "unit": "bar"}),
        ]
        return CanonicalModel(standard=std, nodes=nodes, edges=[])
    return CanonicalModel(standard=std, nodes=[CanonicalNode(id=f"{std}://candidate/default/value", type="Candidate", label="value", attributes={})], edges=[])


def _canonical_source_path(source_standard: str, raw_path: str) -> str:
    return normalize_path(raw_path if "://" in str(raw_path or "") else build_endpoint_path(source_standard, str(raw_path or "")))


def _group_evidence_by_source(evidence: list[EvidenceItem], source_paths: list[str]) -> dict[str, list[EvidenceItem]]:
    grouped: dict[str, list[EvidenceItem]] = {}
    for item in evidence:
        source_node = str((item.payload or {}).get("source_node", "")).strip()
        if not source_node and len(source_paths) == 1:
            source_node = source_paths[0]
        if not source_node:
            continue
        grouped.setdefault(source_node, []).append(item)
    for source_node in grouped:
        grouped[source_node] = sorted(grouped[source_node], key=lambda x: float(x.score), reverse=True)
    return grouped


def _path_list_for_source(by_source: dict[str, list[EvidenceItem]], source_path: str, target_standard: str) -> list[str]:
    prefix = f"{target_standard.lower()}://"
    out: list[str] = []
    seen: set[str] = set()
    for item in by_source.get(source_path, []):
        candidate = str((item.payload or {}).get("candidate_path") or (item.payload or {}).get("target_hint") or "").strip()
        if candidate.startswith(prefix) and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
    return out


def _target_compatibility(mapping: Mapping, target_model: CanonicalModel) -> float:
    if mapping.mapping_type.value == "no_match":
        return 0.25
    target_index = {n.id: n for n in target_model.nodes}
    node = target_index.get(mapping.target_path)
    if node is None:
        return 0.0
    return 1.0


def _merge_per_source(
    source_model: CanonicalModel,
    source_paths: list[str],
    source_standard: str,
    target_standard: str,
    target_model: CanonicalModel,
    retrieval_by_source: dict[str, list[EvidenceItem]],
    rules_by_source: dict[str, list[Mapping]],
    llm_by_source: dict[str, list[Mapping]],
    preference_by_source: dict[str, dict[str, float]],
) -> tuple[list[Mapping], dict[str, Any]]:
    final: list[Mapping] = []
    trace: dict[str, Any] = {"alternatives": {}, "winning_reason": {}}

    for _node, source_path in zip(source_model.nodes, source_paths):
        candidates: list[tuple[float, str, Mapping, str]] = []

        for m in rules_by_source.get(source_path, []):
            score = float(m.confidence) * preference_by_source[source_path]["rules"] * _target_compatibility(m, target_model)
            candidates.append((score, "rules", m, "rule confidence + compatibility"))

        for ev in retrieval_by_source.get(source_path, []):
            retrieval_mapping = normalize_mapping_item(
                {
                    "source_path": source_path,
                    "target_path": str(ev.payload.get("candidate_path") or ""),
                    "mapping_type": "equivalent",
                    "transform": None,
                    "confidence": float(ev.score),
                    "rationale": "Retrieval candidate for source node.",
                    "evidence": ["retrieval:ranked_candidate"],
                },
                source_standard,
                target_standard,
            )
            score = float(retrieval_mapping.confidence) * preference_by_source[source_path]["retrieval"] * _target_compatibility(retrieval_mapping, target_model)
            candidates.append((score, "retrieval", retrieval_mapping, "retrieval score + compatibility"))

        for m in llm_by_source.get(source_path, []):
            score = float(m.confidence) * preference_by_source[source_path]["llm"] * _target_compatibility(m, target_model)
            candidates.append((score, "llm", m, "llm confidence + compatibility"))

        if not candidates:
            winner = normalize_mapping_item(
                {
                    "source_path": source_path,
                    "target_path": "",
                    "mapping_type": "no_match",
                    "transform": None,
                    "confidence": 0.0,
                    "rationale": "No component produced candidate for this source node.",
                    "evidence": ["merge:no_candidates"],
                },
                source_standard,
                target_standard,
            )
            final.append(winner)
            trace["alternatives"][source_path] = []
            trace["winning_reason"][source_path] = "No candidates available."
            continue

        agreement_bonus: dict[str, float] = {}
        for _score, _component, m, _reason in candidates:
            agreement = sum(1 for _, _, other, _ in candidates if other.target_path and other.target_path == m.target_path)
            agreement_bonus[m.target_path] = 0.08 * max(0, agreement - 1)

        rescored = [
            (score + agreement_bonus.get(m.target_path, 0.0), component, m, reason)
            for score, component, m, reason in candidates
        ]
        rescored.sort(key=lambda row: row[0], reverse=True)
        best_score, best_component, best_mapping, best_reason = rescored[0]
        final.append(best_mapping)
        trace["alternatives"][source_path] = [
            {"component": component, "target_path": m.target_path, "score": s, "confidence": m.confidence, "mapping_type": m.mapping_type.value}
            for s, component, m, _ in rescored[:6]
        ]
        trace["winning_reason"][source_path] = (
            f"Selected {best_component} target {best_mapping.target_path or '<no_match>'} with merged score {best_score:.3f}; policy={best_reason}."
        )
    return final, trace


def _lexical_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=str(a or "").lower(), b=str(b or "").lower()).ratio()


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
            "constrain_llm_to_candidates": False,
            "strict_llm_on_high_confidence_retrieval": True,
        }
        flags.update(config.component_flags or {})

        if mode == "rule_only":
            flags.update({"retrieval": False, "llm": False})
        elif mode in {"rag_only", "graph_only", "embedding_only"}:
            flags.update({"rules": False, "llm": False, "retrieval": True})
        elif mode == "llm_only":
            flags.update({"rules": False, "retrieval": False, "llm": True})

        src = ADAPTERS[source_standard]
        tgt = ADAPTERS[target_standard]
        source_model = src.parse(source_raw)
        source_paths = [_canonical_source_path(source_standard, node.id) for node in source_model.nodes]
        def _resolve_source_key(candidate_source: str) -> str:
            if candidate_source in source_paths:
                return candidate_source
            return source_paths[0] if len(source_paths) == 1 else candidate_source
        target_model = _build_default_target_model(target_standard)

        evidence: list[EvidenceItem] = []
        if flags["retrieval"]:
            evidence = self.retriever.retrieve(
                source_model,
                target_standard,
                target_model=target_model,
                top_k=5,
                enable_vector=config.enable_vector_retrieval,
            )
        if target_candidates:
            for candidate in target_candidates:
                for source_path in source_paths:
                    evidence.append(
                        EvidenceItem(
                            id=f"candidate:{source_path}:{candidate}",
                            kind="target_candidate",
                            text=candidate,
                            score=0.99,
                            payload={"source_node": source_path, "candidate_path": candidate, "target_hint": candidate, "label": candidate.rsplit("/", 2)[-2]},
                        )
                    )

        retrieval_by_source = _group_evidence_by_source(evidence, source_paths=source_paths)
        rules_by_source: dict[str, list[Mapping]] = {}
        llm_by_source: dict[str, list[Mapping]] = {}
        llm_raw_output: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        component_outputs: dict[str, Any] = {"rule_engine": {}, "retrieval": {}, "llm": {}, "merged": []}

        if flags["rules"]:
            rule_mappings = self.rules.apply_rules(
                source_model,
                target_standard,
                target=target_model,
                allow_synthetic_shortcuts=bool(flags.get("allow_synthetic_benchmark_shortcuts", True)),
            )
            rule_report = normalize_mapping_items([m.model_dump() for m in rule_mappings], source_standard, target_standard, method="rule")
            for m in rule_report.accepted:
                normalized_source = _canonical_source_path(source_standard, m.source_path)
                normalized_source = _resolve_source_key(normalized_source)
                normalized_mapping = normalize_mapping_item(
                    {**m.model_dump(), "source_path": normalized_source},
                    source_standard,
                    target_standard,
                )
                rules_by_source.setdefault(normalized_source, []).append(normalized_mapping)
            component_outputs["rule_engine"] = {k: [m.model_dump() for m in v] for k, v in rules_by_source.items()}
            rejected.extend([item.model_dump() for item in rule_report.rejected])

        llm_error = None
        if flags["llm"]:
            prompt = build_mapping_prompt(
                source_protocol=source_standard,
                target_protocol=target_standard,
                source_schema_summary=_schema_summary(source_model),
                target_schema_summary=_schema_summary(target_model),
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
            for m in llm_report.accepted:
                normalized_source = _canonical_source_path(source_standard, m.source_path)
                normalized_source = _resolve_source_key(normalized_source)
                source_candidates = _path_list_for_source(retrieval_by_source, normalized_source, target_standard)
                top_score = retrieval_by_source.get(normalized_source, [EvidenceItem(id="", kind="", text="", score=0.0)])[0].score if retrieval_by_source.get(normalized_source) else 0.0
                constrained = bool(flags.get("strict_llm_on_high_confidence_retrieval", True) and top_score >= 0.85 and mode != "llm_only")
                llm_candidate = normalize_mapping_item(
                    {**m.model_dump(), "source_path": normalized_source},
                    source_standard,
                    target_standard,
                )
                if source_candidates and llm_candidate.target_path not in source_candidates:
                    snapped_target = ""
                    if flags["postprocess_snap"] and not constrained:
                        if len(source_candidates) == 1:
                            snapped_target = source_candidates[0]
                        else:
                            scored = sorted(
                                ((difflib.SequenceMatcher(a=llm_candidate.target_path, b=c).ratio(), c) for c in source_candidates),
                                reverse=True,
                            )
                            if scored and scored[0][0] >= 0.72:
                                snapped_target = scored[0][1]
                    if snapped_target:
                        llm_candidate = normalize_mapping_item(
                            {
                                **llm_candidate.model_dump(),
                                "source_path": normalized_source,
                                "target_path": snapped_target,
                                "rationale": f"{llm_candidate.rationale} (snapped to retrieved candidate)",
                                "evidence": [*llm_candidate.evidence, "llm:candidate_snap"],
                            },
                            source_standard,
                            target_standard,
                        )
                if source_candidates and llm_candidate.target_path not in source_candidates:
                    if constrained:
                        replacement = normalize_mapping_item(
                            {
                                "source_path": normalized_source,
                                "target_path": "",
                                "mapping_type": "no_match",
                                "transform": None,
                                "confidence": min(float(llm_candidate.confidence), 0.35),
                                "rationale": "LLM target rejected because high-confidence retrieval disagreed.",
                                "evidence": [*llm_candidate.evidence, "llm:outside_high_confidence_candidates"],
                            },
                            source_standard,
                            target_standard,
                        )
                        llm_by_source.setdefault(normalized_source, []).append(replacement)
                    else:
                        soft_confidence = max(0.35, float(llm_candidate.confidence) * 0.75)
                        softened = normalize_mapping_item(
                            {
                                "source_path": normalized_source,
                                "target_path": llm_candidate.target_path,
                                "mapping_type": llm_candidate.mapping_type.value,
                                "transform": llm_candidate.transform.model_dump() if llm_candidate.transform else None,
                                "confidence": soft_confidence,
                                "rationale": f"{llm_candidate.rationale} (kept as soft-guided LLM candidate outside retrieval set)",
                                "evidence": [*llm_candidate.evidence, "llm:outside_soft_guidance"],
                            },
                            source_standard,
                            target_standard,
                        )
                        llm_by_source.setdefault(normalized_source, []).append(softened)
                else:
                    llm_by_source.setdefault(normalized_source, []).append(llm_candidate)
            component_outputs["llm"] = {k: [m.model_dump() for m in v] for k, v in llm_by_source.items()}
            rejected.extend([item.model_dump() for item in llm_report.rejected])

        component_outputs["retrieval"] = {
            source_node: [
                {
                    "id": item.id,
                    "score": item.score,
                    "candidate_path": item.payload.get("candidate_path"),
                    "datatype": item.payload.get("datatype", ""),
                    "unit": item.payload.get("unit", ""),
                    "breakdown": item.payload.get("score_breakdown", {}),
                }
                for item in items
            ]
            for source_node, items in retrieval_by_source.items()
        }

        preference_by_source: dict[str, dict[str, float]] = {}
        decision_log: list[dict[str, Any]] = []
        for node, source_path in zip(source_model.nodes, source_paths):
            retrieval_items = retrieval_by_source.get(source_path, [])
            top1 = float(retrieval_items[0].score) if retrieval_items else 0.0
            top2 = float(retrieval_items[1].score) if len(retrieval_items) > 1 else 0.0
            margin = max(0.0, top1 - top2)
            source_label = str(node.label or node.id)
            top_label = str(retrieval_items[0].payload.get("label") if retrieval_items else "")
            lexical = _lexical_similarity(source_label, top_label)
            deterministic_rule = any(m.mapping_type.value in {"label_match", "equivalent"} for m in rules_by_source.get(source_path, []))
            rules_target = rules_by_source.get(source_path, [None])[0].target_path if rules_by_source.get(source_path) else ""
            retrieval_target = str(retrieval_items[0].payload.get("candidate_path")) if retrieval_items else ""
            rules_retrieval_agree = bool(rules_target and retrieval_target and rules_target == retrieval_target)

            retrieval_weight = 1.0 + (top1 * 0.8) + (margin * 0.3)
            rules_weight = 1.0 + (0.7 if deterministic_rule else 0.0) + (0.4 if rules_retrieval_agree else 0.0)
            llm_weight = 1.0 + (0.4 if not deterministic_rule else -0.1) + (0.2 if lexical < 0.55 else 0.0)
            if mode in {"rag_only", "graph_only", "embedding_only"}:
                rules_weight = 0.0
                llm_weight = 0.0
            if mode == "llm_only":
                rules_weight = 0.0
                retrieval_weight = 0.0
            if mode == "rule_only":
                retrieval_weight = 0.0
                llm_weight = 0.0
            preference_by_source[source_path] = {
                "retrieval": max(0.0, retrieval_weight),
                "rules": max(0.0, rules_weight),
                "llm": max(0.0, llm_weight),
            }
            selected = max(preference_by_source[source_path], key=preference_by_source[source_path].get)
            decision_log.append(
                {
                    "mode": mode,
                    "source_path": source_path,
                    "selected_strategy": selected,
                    "signals": {
                        "retrieval_top_score": top1,
                        "retrieval_margin_top1_top2": margin,
                        "lexical_similarity_top_candidate": lexical,
                        "deterministic_rules_fired": deterministic_rule,
                        "rules_retrieval_agree": rules_retrieval_agree,
                    },
                    "weights": preference_by_source[source_path],
                }
            )

        final_mappings, merge_trace = _merge_per_source(
            source_model=source_model,
            source_paths=source_paths,
            source_standard=source_standard,
            target_standard=target_standard,
            target_model=target_model,
            retrieval_by_source=retrieval_by_source,
            rules_by_source=rules_by_source,
            llm_by_source=llm_by_source,
            preference_by_source=preference_by_source,
        )

        # one-to-one contract in benchmark expects exactly one mapping per source node.
        dedup: dict[str, Mapping] = {}
        for m in final_mappings:
            dedup[m.source_path] = m
        mappings = [dedup[source_path] for source_path in source_paths if source_path in dedup]

        target_artifact = tgt.serialize(target_model, [m.model_dump() for m in mappings])
        validation = tgt.validate(target_artifact)
        component_outputs["merged"] = [m.model_dump() for m in mappings]

        metadata: dict[str, Any] = {
            "mode": mode,
            "rejected_mappings": rejected,
            "llm_raw_output": llm_raw_output,
            "decision_log": decision_log,
            "component_outputs": component_outputs,
            "selected_strategy": "hybrid_weighted_merge" if mode == "hybrid" else mode,
            "signals": {"vector_retrieval_enabled": bool(config.enable_vector_retrieval)},
            "merge_trace": merge_trace,
            "execution": {
                "selected_strategy": "hybrid_weighted_merge" if mode == "hybrid" else mode,
                "rules_ran": flags["rules"],
                "retrieval_ran": flags["retrieval"],
                "llm_ran": flags["llm"],
                "candidate_snapping_ran": False,
                "final_mapping_source": "merged",
            },
        }
        if llm_error is not None:
            metadata["llm_error"] = llm_error

        prov = Provenance(
            model_name=config.model_name,
            prompt_hash=stable_hash({"mode": mode, "source": source_standard, "target": target_standard}),
            seed=config.seed,
            git_commit=_git_commit(),
            adapter_versions={"source": "1.0", "target": "1.0"},
            metadata=metadata,
        )
        return TranslationResult(target_artifact=target_artifact, mappings=mappings, evidence=evidence, provenance=prov, validation_report=validation)
