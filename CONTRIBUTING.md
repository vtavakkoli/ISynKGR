# Contributing to ISynKGR

Thank you for helping improve ISynKGR. Contributions are welcome when they preserve the repository's scientific reproducibility and make changes easy to evaluate.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
make check
```

## Contribution workflow

1. Create a focused branch from `main`.
2. Keep changes scoped to one logical improvement.
3. Add or update tests for behavior changes.
4. Run `make check` before opening a pull request.
5. Update documentation when commands, benchmark semantics, datasets, scenarios, or reported outputs change.
6. Explain any effect on reproducibility, comparability, runtime requirements, or benchmark interpretation in the pull request.

## Scientific changes

Changes to datasets, ground truth, adapters, candidate generation, ranking, prompts, evaluation, metrics, scenario definitions, or default seeds can change scientific conclusions. Such pull requests should therefore include:

- a precise description of the methodological change;
- the motivation and expected effect;
- tests that cover the changed behavior;
- the experiment configuration used for validation;
- before/after metrics when the change affects reported results;
- a note about whether existing results remain comparable;
- updated limitations or threats-to-validity documentation when appropriate.

Do not silently replace ground truth, alter canonical path conventions, or change headline metric semantics.

## Code quality

The project targets Python 3.10+ and uses `ruff` and `pytest`.

```bash
make lint
make test
```

Prefer deterministic behavior for benchmark infrastructure. Randomized procedures must expose and record seeds.

## Documentation

Public APIs, environment variables, benchmark scenarios, and experiment commands should be documented in the same pull request that changes them. Keep the root README concise enough to orient new users; place detailed research protocol material in `docs/`.

## Reporting results

When contributing benchmark results, record at minimum:

- repository commit or release;
- source/target standard pairs;
- dataset tier and size;
- scenario name;
- seed policy;
- LLM model and runtime endpoint when applicable;
- relevant environment-variable overrides;
- generated artifact/run identifier.

## Issues and security

Use GitHub issues for reproducible bugs and feature proposals. Do not disclose security-sensitive findings publicly; follow `SECURITY.md` instead.
