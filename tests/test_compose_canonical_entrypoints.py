from pathlib import Path

from isynkgr.validation.orchestration import CANONICAL_BENCHMARK_ENTRYPOINTS, extract_python_module_entrypoints


def test_docker_compose_uses_only_canonical_benchmark_entrypoints() -> None:
    compose_modules = extract_python_module_entrypoints(Path("docker-compose.yml"))
    benchmark_modules = {module for module in compose_modules if module.startswith("benchmark.")}
    assert benchmark_modules <= set(CANONICAL_BENCHMARK_ENTRYPOINTS)
