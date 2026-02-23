from __future__ import annotations

import re
from pathlib import Path

CANONICAL_BENCHMARK_ENTRYPOINTS: tuple[str, ...] = (
    "benchmark.sample_validate",
    "benchmark.orchestrate",
    "benchmark.run",
)


def build_benchmark_orchestrate_steps(
    python_executable: str,
    scenarios: list[str],
    *,
    config_path: str = "benchmark/config.json",
    max_items: int = 100,
) -> list[tuple[str, list[str]]]:
    steps: list[tuple[str, list[str]]] = [
        (
            "STEP 1/4: sample validation",
            [python_executable, "-u", "-m", "benchmark.sample_validate"],
        )
    ]
    for scenario in scenarios:
        steps.append(
            (
                f" - scenario={scenario}",
                [
                    python_executable,
                    "-u",
                    "-m",
                    "benchmark.run",
                    "--scenario",
                    scenario,
                    "--config",
                    config_path,
                    "--out",
                    f"results/{scenario}",
                    "--max-items",
                    str(max_items),
                ],
            )
        )
    return steps


def extract_python_module_entrypoints(compose_path: Path) -> set[str]:
    content = compose_path.read_text()
    return set(re.findall(r"-m\s+([\w\.]+)", content))
