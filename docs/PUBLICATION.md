# Publication Companion

ISynKGR is the research-software companion for work on adaptive cross-standard semantic interoperability.

## Associated publication

Mohsenzadegan, K., Tavakkoli, V., & Kyamakya, K. (2026). **ISynKGR: An Adaptive Benchmark Framework for Cross-Standard Semantic Interoperability.** Accepted for presentation at the International Workshop on AI and Mathematical Methods for Real-world Impact (AI2M4RI), in conjunction with the 23rd International Conference on Mobile Systems and Pervasive Computing (MobiSPC), Athens, Greece, August 18–20, 2026.

### BibTeX

```bibtex
@inproceedings{mohsenzadegan2026isynkgr,
  author    = {Kabeh Mohsenzadegan and Vahid Tavakkoli and Kyandoghere Kyamakya},
  title     = {ISynKGR: An Adaptive Benchmark Framework for Cross-Standard Semantic Interoperability},
  booktitle = {International Workshop on AI and Mathematical Methods for Real-world Impact (AI2M4RI), in conjunction with the 23rd International Conference on Mobile Systems and Pervasive Computing (MobiSPC)},
  address   = {Athens, Greece},
  year      = {2026},
  note      = {Accepted for presentation, August 18--20, 2026}
}
```

## What this repository contributes

The repository operationalizes the benchmark methodology as executable research software. It provides:

- deterministic generation and validation of benchmark data;
- adapters for multiple industrial semantic representations;
- canonical full, baseline, and ablation scenarios;
- multi-seed experiment orchestration;
- mapping and retrieval metrics;
- robustness and error-analysis artifacts;
- Markdown and HTML reporting;
- local and Docker-based execution paths.

## Relationship between paper and code

The paper is the scientific reference for the framework and its motivation. The repository is the executable artifact used to inspect, reproduce, extend, and validate the benchmark methodology.

When reporting repository-derived results, cite the publication and identify the exact repository commit or release used. Results obtained from newer commits may differ from the accepted-paper experiments if datasets, adapters, prompts, models, scenario definitions, or evaluation logic have changed.

## Recommended artifact statement

A publication or report using this code can include a statement such as:

> Experiments were executed with the ISynKGR research-software artifact at a recorded Git revision. Benchmark configuration, seed policy, model/runtime settings, and generated run artifacts were retained to support reproducibility.

## Publication-quality evidence

For every experiment that may support a scientific claim, retain:

- the Git revision;
- `artifacts/<RUN_ID>/metrics.json`;
- pair-level datasets and ground truth;
- scenario/seed output directories;
- logs and error-analysis files;
- generated reports and plots;
- the model/runtime configuration for LLM-backed scenarios;
- hardware and software environment information for performance claims.

See `REPRODUCIBILITY.md` for the complete protocol.

## Scope statement

ISynKGR is a benchmark framework and reference implementation. It does not claim complete conformance with OPC UA, AAS, IEEE 1451, IEC 61499, or ISO 15926, and it should not be treated as a certified replacement for standard-specific industrial tooling.
