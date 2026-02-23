# Datasets
`datasets/v1` contains OPC UA and AAS synthetic (100 each), semi-real (10 each), crosswalk GT and manifest hashes.


## benchmark/data_gen pipeline
Generate deterministic multi-standard benchmark artifacts/GT with:
`python -m benchmark.data_gen.pipeline --out-dir benchmark/data_gen/out --seed 42 --sample-size 100`

Env overrides:
- `BENCHMARK_DATA_GEN_SEED`
- `BENCHMARK_DATA_GEN_SAMPLE_SIZE`
- `BENCHMARK_DATA_GEN_OUT`
