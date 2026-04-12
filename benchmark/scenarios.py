from __future__ import annotations

from dataclasses import dataclass

CANONICAL_SCENARIOS: tuple[str, ...] = (
    "full_framework",
    "rule_based_only",
    "llm_only",
    "rag_only",
    "embedding_similarity",
    "ablation_no_rules",
    "ablation_no_retrieval",
    "ablation_no_llm",
)

DEPRECATED_SCENARIO_ALIASES: dict[str, str] = {
    "baseline": "rule_based_only",
    "ablation_no_graphrag": "ablation_no_retrieval",
    "ablation_no_parallel": "rag_only",
    "ablation_no_community": "embedding_similarity",
    "ablation_no_reasoning": "ablation_no_llm",
    "ablation_no_graph_expansion": "full_framework",
    "ablation_no_reasoning_prompt": "ablation_no_llm",
    "ablation_no_community_filter": "full_framework",
    "ablation_no_parallel_retrieval": "full_framework",
}

COMPONENT_FLAGS: dict[str, dict[str, bool]] = {
    "full_framework": {"postprocess_snap": False},
    "rule_based_only": {"retrieval": False, "llm": False, "adaptive_selection": False},
    "llm_only": {"rules": False, "retrieval": False},
    "rag_only": {"rules": False, "llm": False},
    "embedding_similarity": {"rules": False, "llm": False, "adaptive_selection": False},
    "ablation_no_rules": {"rules": False},
    "ablation_no_retrieval": {"retrieval": False},
    "ablation_no_llm": {"llm": False},
}


@dataclass(frozen=True)
class ScenarioRuntime:
    name: str
    mode: str
    component_flags: dict[str, bool]


SCENARIO_RUNTIME: dict[str, ScenarioRuntime] = {
    "full_framework": ScenarioRuntime("full_framework", "adaptive_candidate_ranker", COMPONENT_FLAGS["full_framework"]),
    "rule_based_only": ScenarioRuntime("rule_based_only", "rule_only", COMPONENT_FLAGS["rule_based_only"]),
    "llm_only": ScenarioRuntime("llm_only", "llm_only", COMPONENT_FLAGS["llm_only"]),
    "rag_only": ScenarioRuntime("rag_only", "rag_only", COMPONENT_FLAGS["rag_only"]),
    "embedding_similarity": ScenarioRuntime("embedding_similarity", "embedding_only", COMPONENT_FLAGS["embedding_similarity"]),
    "ablation_no_rules": ScenarioRuntime("ablation_no_rules", "adaptive_candidate_ranker", COMPONENT_FLAGS["ablation_no_rules"]),
    "ablation_no_retrieval": ScenarioRuntime("ablation_no_retrieval", "adaptive_candidate_ranker", COMPONENT_FLAGS["ablation_no_retrieval"]),
    "ablation_no_llm": ScenarioRuntime("ablation_no_llm", "adaptive_candidate_ranker", COMPONENT_FLAGS["ablation_no_llm"]),
}


def resolve_scenario_name(name: str) -> tuple[str, bool]:
    lowered = str(name).strip()
    if lowered in SCENARIO_RUNTIME:
        return lowered, False
    mapped = DEPRECATED_SCENARIO_ALIASES.get(lowered)
    if mapped:
        return mapped, True
    valid = ", ".join(CANONICAL_SCENARIOS)
    raise ValueError(f"Unknown scenario '{name}'. Valid canonical scenarios: {valid}")
