from __future__ import annotations

import warnings

from benchmark.orchestrate import main as benchmark_main


def main() -> int:
    warnings.warn(
        "isynkgr.evaluation.benchmark_runner is deprecated; use benchmark.orchestrate instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return benchmark_main()


if __name__ == "__main__":
    raise SystemExit(main())
