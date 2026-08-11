# Repository agent instructions

## Scope and precedence

These instructions apply to the entire repository. If a future directory contains a more specific `AGENTS.md`, follow that file for work inside its directory. Direct maintainer instructions take precedence.

## Project context

- **Project:** EstateScope AI
- **Repository shape:** Python, Docker, Jupyter notebooks
- Read `README.md` before changing behaviour, architecture, data handling, or public claims.
- Treat `CONTRIBUTING.md` as the authoritative development workflow and command reference.
- Follow `SECURITY.md` for vulnerability reporting and `CODE_OF_CONDUCT.md` for collaboration.
- Use the workflows under `.github/workflows/` as the final cross-platform CI contract.

## Working agreement

1. Keep each change focused on a clear problem; avoid unrelated cleanup.
2. Inspect nearby tests, configuration, documentation, and generated artifacts before editing.
3. Add or update tests for behavioural changes, including failure and boundary cases.
4. Preserve public APIs and file formats unless the change explicitly includes a documented migration.
5. Never commit credentials, personal data, local environments, caches, or unlicensed datasets.
6. Do not weaken lint, coverage, security, or dependency gates merely to obtain a passing run.
7. After code or architecture changes, run `graphify update .` and review the graph diff; do not commit cache-only churn.
8. Do not stage, commit, push, or publish changes unless the maintainer explicitly requests that action.

## Required validation

Run commands from the repository root. Use a clean environment when dependency or packaging behaviour changes.

### Standard development gate

```text
python -m venv .venv
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m ruff format --check .
python -m pytest
python -m compileall app app.py scripts src tests wsgi.py
```

### Repository-specific CI-equivalent checks

```text
python -m pip check
python -m pip_audit --local
python -m bandit -r app scripts -q -ll
python scripts/check_project.py --ci
python -c "from wsgi import app; response = app.test_client().get('/'); assert response.status_code == 200"
docker build --tag estatescope-ai:ci .
```

Also run `git diff --check`. Record exactly which commands ran, their outcomes, and any documented prerequisite that prevented a check. Do not describe an unexecuted check as passing.

### Conditional or integration setup

- Use the documented Docker or Compose commands only for container changes and retain the non-root user, health check, and HTTP smoke behavior.
- OpenAI-backed chat is optional and requires `OPENAI_API_KEY`; core tests and startup must not depend on it.

## Managed artifacts and data

- Treat committed large model artifacts as compatibility-sensitive and checksum-pinned by `artifacts-manifest.json`; require real loading or inference plus manifest review after changes.
- Keep `.env`, local `instance` databases, uploads, caches, and generated coverage files out of Git.
- Use Git LFS, release assets, or external storage for new large artifacts rather than bypassing the repository size gate.

## Integration and claim boundaries

- The default SQLite configuration is for local development; production-style use should use an external database such as Postgres.
- Full Colab tabular/CNN training is not reproducible from the committed data and dependencies.
- Optional NLP/image signals and chat integrations must fail explicitly or degrade only as documented; they do not prove multimodal lift or production deployment.
- **Verified local capability:** Non-root Linux image, health check, HTTP 200, and three model-load smokes.
- **Known boundary:** Full Colab training is not reproducible from committed data/dependencies.

## Code, documentation, and evidence standards

- For Python, follow PEP 8, PEP 257, and Google-style docstrings in the production scopes configured by Ruff.
- Prefer small, typed, testable units and explicit error handling. Avoid silent fallbacks that hide invalid data, missing models, credentials, or services.
- Update README, architecture, data/model documentation, examples, and changelog material when interfaces or behaviour change.
- Every quantitative claim must retain its denominator, dataset/version, split, date range, run/configuration, and calculation method.
- Distinguish measured results from targets, examples, heuristics, previews, and prior runs. Do not infer deployment, accuracy, security, causality, or impact from tests or scaffolding alone.
- Update generated files through their source script. If no generator exists, document the manual process and verify semantic equivalence.

## Review and handoff

Before handing off a change, provide a concise summary, list changed files, report validation evidence, and call out residual risks or unavailable integrations. Keep commits imperative and scoped; do not add tool-attribution or automated co-author trailers.
