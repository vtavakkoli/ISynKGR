#!/usr/bin/env bash
set -euo pipefail
python -m src.evaluation.data_gen.generate_samples
python -m src.evaluation.data_gen.generate_ground_truth
python -m src.evaluation.data_gen.validate_data
