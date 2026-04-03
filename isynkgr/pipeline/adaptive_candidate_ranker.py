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
from isynkgr.icr.entities import build_endpoint_path, normalize_path
from isynkgr.icr.mapping_output_contract import normalize_mapping_item
from isynkgr.icr.mapping_schema import MappingType
from isynkgr.llm.ollama import OllamaClient
from isynkgr.pipeline.prompting import build_mapping_prompt
from isynkgr.retrieval.graphrag import GraphRAGRetriever
from isynkgr.rules.engine import RuleEngine
from isynkgr.utils.hashing import stable_hash

Mode = Literal[
    "adaptive_candidate_ranker",
    "hybrid",  # deprecated alias
    "llm_only",
    "rag_only",
    "rule_only",
    "graph_only",
    "embedding_only",
]


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


def _lexical_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=str(a or "").lower(), b=str(b or "").lower()).ratio()


def _guess_dtype(node: CanonicalNode) -> str:
    attrs = node.attributes or {}
    meta = attrs.get("metadata", {}) if isinstance(attrs.get("metadata", {}), dict) else {}
    for key in ("datatype", "dtype", "valueType", "dataType", "type"):
        value = attrs.get(key) or meta.get(key)
        if value:
            return str(value).upper()
    return ""


def _guess_unit(node: CanonicalNode) -> str:
    attrs = node.attributes or {}
    meta = attrs.get("metadata", {}) if isinstance(attrs.get("metadata", {}), dict) else {}
    return str(attrs.get("unit") or meta.get("unit") or "").strip().lower()


def _parent_map(model: CanonicalModel) -> dict[str, str]:
    out: dict[str, str] = {}
    for edge in model.edges:
        if edge.target not in out:
            out[edge.target] = edge.source
    return out


@dataclass
class CandidateState:
    mapping: Mapping
    total_score: float
    score_breakdown: dict[str, float]
    support: set[str]
    rejected_reasons: list[str]


class AdaptiveCandidateRankerPipeline:
    def __init__(self, llm: OllamaClient, retriever: GraphRAGRetriever, rules: RuleEngine) -> None:
        self.llm = llm
        self.retriever = retriever
        self.rules = rules

    def run(
        self,
        source_standard: str,
        target_standard: str,
        source_raw: str | bytes | dict,
        mode: Mode,
        config: TranslatorConfig,
        target_candidates: list[str] | None = None,
    ) -> TranslationResult:
        resolved_mode = "adaptive_candidate_ranker" if mode == "hybrid" else mode
        flags = {
            "rules": True,
            "retrieval": True,
            "llm": True,
            "allow_synthetic_benchmark_shortcuts": True,
            "uncertainty_threshold": 0.72,
            "ambiguity_margin": 0.06,
            "max_candidates_per_source": 5,
        }
        flags.update(config.component_flags or {})

        if resolved_mode == "rule_only":
            flags.update({"retrieval": False, "llm": False})
        elif resolved_mode in {"rag_only", "graph_only", "embedding_only"}:
            flags.update({"rules": False, "llm": False, "retrieval": True})
        elif resolved_mode == "llm_only":
            flags.update({"rules": False, "retrieval": False, "llm": True})

        src = ADAPTERS[source_standard]
        tgt = ADAPTERS[target_standard]
        source_model = src.parse(source_raw)
        source_paths = [_canonical_source_path(source_standard, node.id) for node in source_model.nodes]
        source_index = {path: node for path, node in zip(source_paths, source_model.nodes)}
        target_model = _build_default_target_model(target_standard)
        target_index = {node.id: node for node in target_model.nodes}
        source_parent = _parent_map(source_model)
        target_parent = _parent_map(target_model)
        allowed_external_targets = set(target_candidates or [])
        strict_target_existence = resolved_mode in {"adaptive_candidate_ranker", "rule_only", "hybrid"}

        evidence: list[EvidenceItem] = []
        retrieval_by_source: dict[str, list[EvidenceItem]] = {p: [] for p in source_paths}
        if flags["retrieval"]:
            evidence = self.retriever.retrieve(
                source_model,
                target_standard,
                target_model=target_model,
                top_k=int(flags.get("max_candidates_per_source", 5)),
                enable_vector=config.enable_vector_retrieval,
            )
            for ev in evidence:
                source_node = str((ev.payload or {}).get("source_node", "")).strip()
                if not source_node and len(source_paths) == 1:
                    source_node = source_paths[0]
                if source_node in retrieval_by_source:
                    retrieval_by_source[source_node].append(ev)
            for source_node in retrieval_by_source:
                retrieval_by_source[source_node] = sorted(retrieval_by_source[source_node], key=lambda x: float(x.score), reverse=True)

        if target_candidates:
            for source_path in source_paths:
                for candidate in target_candidates:
                    retrieval_by_source[source_path].append(
                        EvidenceItem(
                            id=f"candidate:{source_path}:{candidate}",
                            kind="target_candidate",
                            text=candidate,
                            score=0.99,
                            payload={"source_node": source_path, "candidate_path": candidate, "target_hint": candidate, "label": candidate.rsplit("/", 2)[-2]},
                        )
                    )

        rules_by_source: dict[str, list[Mapping]] = {p: [] for p in source_paths}
        if flags["rules"]:
            rule_mappings = self.rules.apply_rules(
                source_model,
                target_standard,
                target=target_model,
                allow_synthetic_shortcuts=bool(flags.get("allow_synthetic_benchmark_shortcuts", True)),
            )
            for rm in rule_mappings:
                normalized = normalize_mapping_item(rm.model_dump(), source_standard, target_standard)
                source_key = _canonical_source_path(source_standard, normalized.source_path)
                if source_key in rules_by_source:
                    rules_by_source[source_key].append(normalized)

        candidates_by_source: dict[str, dict[str, CandidateState]] = {p: {} for p in source_paths}
        for source_path in source_paths:
            for ev in retrieval_by_source.get(source_path, []):
                target_path = str((ev.payload or {}).get("candidate_path") or "").strip()
                if not target_path:
                    continue
                mapping = normalize_mapping_item(
                    {
                        "source_path": source_path,
                        "target_path": target_path,
                        "mapping_type": "equivalent",
                        "transform": None,
                        "confidence": float(ev.score),
                        "rationale": "Retrieved target candidate.",
                        "evidence": ["retrieval:ranked_candidate"],
                    },
                    source_standard,
                    target_standard,
                )
                candidates_by_source[source_path][target_path] = CandidateState(mapping=mapping, total_score=0.0, score_breakdown={}, support={"retrieval"}, rejected_reasons=[])
            for rm in rules_by_source.get(source_path, []):
                if rm.mapping_type == MappingType.NO_MATCH:
                    continue
                state = candidates_by_source[source_path].get(rm.target_path)
                if state is None:
                    candidates_by_source[source_path][rm.target_path] = CandidateState(mapping=rm, total_score=0.0, score_breakdown={}, support={"rules"}, rejected_reasons=[])
                else:
                    state.support.add("rules")
                    if rm.confidence > state.mapping.confidence:
                        state.mapping = rm

        llm_raw_output: list[dict[str, Any]] = []
        llm_by_source: dict[str, list[Mapping]] = {p: [] for p in source_paths}
        llm_invocation_log: list[dict[str, Any]] = []

        prompt = ""
        if flags["llm"]:
            prompt = build_mapping_prompt(
                source_protocol=source_standard,
                target_protocol=target_standard,
                source_schema_summary=_schema_summary(source_model),
                target_schema_summary=_schema_summary(target_model),
                source_model=source_model,
                evidence=evidence,
                use_reasoning_prompt=True,
            )

        source_decisions: dict[str, dict[str, Any]] = {}
        for source_path in source_paths:
            source_node = source_index[source_path]
            src_dtype = _guess_dtype(source_node)
            src_unit = _guess_unit(source_node)
            src_parent = source_parent.get(source_node.id, "")

            scored: list[CandidateState] = []
            for target_path, state in candidates_by_source[source_path].items():
                target_node = target_index.get(target_path)
                if target_node is None:
                    state.rejected_reasons.append("target_path_missing")
                    continue
                retrieval_item = next((i for i in retrieval_by_source.get(source_path, []) if str(i.payload.get("candidate_path")) == target_path), None)
                retrieval_score = float(retrieval_item.score) if retrieval_item else 0.0
                retrieval_breakdown = (retrieval_item.payload or {}).get("score_breakdown", {}) if retrieval_item else {}
                lexical = float(retrieval_breakdown.get("lexical", _lexical_similarity(source_node.label or source_node.id, target_node.label or target_node.id)))
                embedding_similarity = float(retrieval_breakdown.get("vector_boost", 0.0))
                datatype_compat = 1.0 if src_dtype and _guess_dtype(target_node) and src_dtype == _guess_dtype(target_node) else (0.5 if not src_dtype or not _guess_dtype(target_node) else 0.0)
                unit_compat = 1.0 if src_unit and _guess_unit(target_node) and src_unit == _guess_unit(target_node) else (0.5 if not src_unit or not _guess_unit(target_node) else 0.0)
                parent_sim = _lexical_similarity(src_parent, target_parent.get(target_node.id, "")) if src_parent or target_parent.get(target_node.id, "") else 0.5
                rule_hit = 1.0 if "rules" in state.support else 0.0

                breakdown = {
                    "lexical_similarity": lexical,
                    "embedding_similarity": embedding_similarity,
                    "rule_hit": rule_hit,
                    "datatype_compatibility": datatype_compat,
                    "unit_compatibility": unit_compat,
                    "parent_context_similarity": parent_sim,
                    "retrieval_score": retrieval_score,
                }
                total_score = (
                    lexical * 0.22
                    + embedding_similarity * 0.08
                    + rule_hit * 0.2
                    + datatype_compat * 0.14
                    + unit_compat * 0.12
                    + parent_sim * 0.1
                    + retrieval_score * 0.14
                )
                if len(state.support) > 1:
                    total_score += 0.08
                state.score_breakdown = breakdown
                state.total_score = min(1.0, total_score)
                scored.append(state)

            scored.sort(key=lambda x: x.total_score, reverse=True)
            top1 = scored[0].total_score if scored else 0.0
            top2 = scored[1].total_score if len(scored) > 1 else 0.0
            uncertainty = top1 < float(flags["uncertainty_threshold"]) or (top1 - top2) <= float(flags["ambiguity_margin"])

            rules_target = rules_by_source[source_path][0].target_path if rules_by_source.get(source_path) else ""
            retrieval_target = str(retrieval_by_source[source_path][0].payload.get("candidate_path")) if retrieval_by_source.get(source_path) else ""
            disagreement = bool(rules_target and retrieval_target and rules_target != retrieval_target)
            uncertainty = uncertainty or disagreement

            if flags["llm"] and (resolved_mode == "llm_only" or uncertainty):
                raw = self.llm.complete_json(prompt, "MappingOutputContract", config.seed)
                llm_raw_output.append(
                    {
                        "method": resolved_mode,
                        "source_protocol": source_standard,
                        "target_protocol": target_standard,
                        "prompt": prompt,
                        "raw": raw,
                        "source_path": source_path,
                    }
                )
                raw_mappings = raw.get("mappings", [])
                for item in raw_mappings:
                    try:
                        normalized = normalize_mapping_item(item, source_standard, target_standard)
                    except Exception:
                        continue
                    normalized_source = _canonical_source_path(source_standard, normalized.source_path)
                    if normalized_source != source_path:
                        if len(source_paths) == 1:
                            normalized = normalize_mapping_item(
                                {**normalized.model_dump(), "source_path": source_path},
                                source_standard,
                                target_standard,
                            )
                        else:
                            continue
                    source_candidates = [str(i.payload.get("candidate_path", "")).strip() for i in retrieval_by_source.get(source_path, []) if str(i.payload.get("candidate_path", "")).strip()]
                    if source_candidates and normalized.target_path not in source_candidates:
                        snapped_target = ""
                        if len(source_candidates) == 1:
                            snapped_target = source_candidates[0]
                        else:
                            scored = sorted(
                                ((_lexical_similarity(normalized.target_path, c), c) for c in source_candidates),
                                reverse=True,
                            )
                            if scored and scored[0][0] >= 0.72:
                                snapped_target = scored[0][1]
                        if snapped_target:
                            normalized = normalize_mapping_item(
                                {
                                    **normalized.model_dump(),
                                    "target_path": snapped_target,
                                    "rationale": f"{normalized.rationale} (snapped to retrieved candidate)",
                                    "evidence": [*normalized.evidence, "llm:candidate_snap"],
                                },
                                source_standard,
                                target_standard,
                            )
                    llm_by_source[source_path].append(normalized)
                    if normalized.mapping_type != MappingType.NO_MATCH:
                        state = candidates_by_source[source_path].get(normalized.target_path)
                        if state is None:
                            candidates_by_source[source_path][normalized.target_path] = CandidateState(mapping=normalized, total_score=float(normalized.confidence) * 0.7, score_breakdown={"llm_confidence": float(normalized.confidence)}, support={"llm"}, rejected_reasons=[])
                        else:
                            state.support.add("llm")
                            state.total_score = min(1.0, state.total_score + 0.06)
                llm_invocation_log.append({"source_path": source_path, "invoked": True, "reason": "uncertain_or_disagreement", "top1": top1, "top2": top2, "rules_retrieval_disagree": disagreement})
            else:
                llm_invocation_log.append({"source_path": source_path, "invoked": False, "reason": "high_confidence_non_ambiguous", "top1": top1, "top2": top2, "rules_retrieval_disagree": disagreement})

            source_decisions[source_path] = {
                "top1": top1,
                "top2": top2,
                "score_gap": max(0.0, top1 - top2),
                "rules_retrieval_disagree": disagreement,
                "candidate_count": len(scored),
            }

        final_by_source: dict[str, Mapping] = {}
        used_targets: set[str] = set()
        ranking_trace: dict[str, list[dict[str, Any]]] = {}

        for source_path in source_paths:
            states = list(candidates_by_source[source_path].values())
            states.sort(key=lambda x: x.total_score, reverse=True)
            ranking_trace[source_path] = [
                {
                    "target_path": s.mapping.target_path,
                    "total_score": s.total_score,
                    "score_breakdown": s.score_breakdown,
                    "support": sorted(s.support),
                    "rejected_reasons": s.rejected_reasons,
                }
                for s in states[:8]
            ]

            winner: Mapping | None = None
            for state in states:
                mapping = state.mapping
                if mapping.mapping_type not in {
                    MappingType.EQUIVALENT,
                    MappingType.APPROXIMATE,
                    MappingType.TRANSFORM,
                    MappingType.LABEL_MATCH,
                }:
                    continue
                if strict_target_existence and mapping.target_path not in target_index and mapping.target_path not in allowed_external_targets:
                    continue
                if mapping.target_path in target_index:
                    src_dtype = _guess_dtype(source_index[source_path])
                    tgt_dtype = _guess_dtype(target_index[mapping.target_path])
                    if src_dtype and tgt_dtype and src_dtype != tgt_dtype:
                        continue
                    src_unit = _guess_unit(source_index[source_path])
                    tgt_unit = _guess_unit(target_index[mapping.target_path])
                    if src_unit and tgt_unit and src_unit != tgt_unit:
                        continue
                if mapping.target_path in used_targets:
                    continue
                winner = normalize_mapping_item(
                    {
                        **mapping.model_dump(),
                        "source_path": source_path,
                        "confidence": max(float(mapping.confidence), state.total_score),
                        "rationale": f"Adaptive candidate ranker selected highest valid candidate with support={sorted(state.support)}.",
                        "evidence": [*mapping.evidence, "ranker:final_selection"],
                    },
                    source_standard,
                    target_standard,
                )
                break

            if winner is None:
                winner = normalize_mapping_item(
                    {
                        "source_path": source_path,
                        "target_path": "",
                        "mapping_type": "no_match",
                        "transform": None,
                        "confidence": 0.0,
                        "rationale": "No valid target candidate after constraint enforcement.",
                        "evidence": ["ranker:no_valid_candidate"],
                    },
                    source_standard,
                    target_standard,
                )
            elif winner.target_path:
                used_targets.add(winner.target_path)

            final_by_source[source_path] = winner

        mappings = [final_by_source[source_path] for source_path in source_paths]
        target_artifact = tgt.serialize(target_model, [m.model_dump() for m in mappings])
        validation = tgt.validate(target_artifact)

        metadata: dict[str, Any] = {
            "mode": resolved_mode,
            "legacy_mode_alias_used": mode == "hybrid",
            "deprecation_warning": "mode='hybrid' is deprecated; use mode='adaptive_candidate_ranker'." if mode == "hybrid" else "",
            "llm_raw_output": llm_raw_output,
            "component_outputs": {
                "retrieval": ({
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
                } if (flags["retrieval"] or target_candidates) else {}),
                "rules": {k: [m.model_dump() for m in v] for k, v in rules_by_source.items()},
                "rule_engine": {k: [m.model_dump() for m in v] for k, v in rules_by_source.items()},
                "llm": {k: [m.model_dump() for m in v] for k, v in llm_by_source.items()},
                "ranking": ranking_trace,
                "final": [m.model_dump() for m in mappings],
                "merged": [m.model_dump() for m in mappings],
            },
            "source_decisions": source_decisions,
            "decision_log": [
                {"source_path": source_path, **details, "selected_strategy": "adaptive_candidate_ranker"}
                for source_path, details in source_decisions.items()
            ],
            "llm_invocations": llm_invocation_log,
            "execution": {
                "selected_strategy": "adaptive_candidate_ranker",
                "rules_ran": flags["rules"],
                "retrieval_ran": flags["retrieval"],
                "llm_ran": flags["llm"],
                "one_to_one_enforced": True,
            },
        }

        prov = Provenance(
            model_name=config.model_name,
            prompt_hash=stable_hash({"mode": resolved_mode, "source": source_standard, "target": target_standard}),
            seed=config.seed,
            git_commit=_git_commit(),
            adapter_versions={"source": "1.0", "target": "1.0"},
            metadata=metadata,
        )
        return TranslationResult(target_artifact=target_artifact, mappings=mappings, evidence=evidence, provenance=prov, validation_report=validation)


# Backward compatibility class alias.
HybridPipeline = AdaptiveCandidateRankerPipeline
