# ISynKGR

ISynKGR is an industrial translation library for standards-aware artifact translation (hybrid LLM + KG/GraphRAG + rules), plus a container-isolated benchmark framework.

## Quick start

```bash
make install
make example-opcua-aas
```

## Benchmark quick start

```bash
make benchmark-small
make benchmark-full
```

## Design choices
- Container-isolated SUTs for fair baseline comparison.
- Standard adapters with explicit parse/serialize/validate.
- Mapping-level scoring + structural validity + robustness metrics.
- Reproducibility via seeds, pinned deps, manifest hashes, and provenance.

## Extend adapters
Add a new adapter in `isynkgr/adapters/`, implement `parse/serialize/validate`, and register it in `isynkgr/pipeline/hybrid.py`.

## Add a new baseline SUT
Create Dockerfile under `docker/sut/<name>/`, add service in `docker/compose/docker-compose.bench.yml`, then include baseline in `benchmark/harness.py`.
