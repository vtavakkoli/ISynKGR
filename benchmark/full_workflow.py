from __future__ import annotations

from benchmark.orchestrate import main as orchestrate_main


def run_full_workflow() -> int:
    return orchestrate_main()


if __name__ == "__main__":
    raise SystemExit(run_full_workflow())
