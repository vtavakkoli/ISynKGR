#!/usr/bin/env bash
set -euo pipefail
python -m src.evaluation.benchmark_runner "$@"
