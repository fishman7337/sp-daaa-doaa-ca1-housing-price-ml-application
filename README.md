# EstateScope AI

[![CI](https://github.com/fishman7337/sp-daaa-doaa-ca1-housing-price-ml-application/actions/workflows/ci.yml/badge.svg)](https://github.com/fishman7337/sp-daaa-doaa-ca1-housing-price-ml-application/actions/workflows/ci.yml)

EstateScope AI is a Flask-based multimodal housing valuation platform for US real-estate listings. It combines structured tabular inputs, optional listing-description NLP, optional property-image CNN signals, authenticated prediction history, and a housing-focused chat assistant.

This project was completed for Singapore Polytechnic, School of Computing, Diploma in Applied AI & Analytics, under DevOps & Automation for AI (ST1516), CA1. It was prepared by Goh Kun Ming, DAAA student P2415691, AY25/26 Year 2 Semester 2. Lecturer: Ryan Chia Xueyi.

## Evidence and interpretation

| Evidence-backed measure | Current repository evidence |
| --- | --- |
| Modalities | The Flask prototype can combine **3 signal types**: structured attributes, optional text features, and optional image-CNN features. |
| Verification surface | **61 tests** pass at **55.60%** coverage; the non-root Linux image is healthy, serves HTTP 200, and was reduced by **523,760,592 bytes (37.0%)** to **892,109,052 bytes**. |
| Reproducible held-out result | The committed NLP evaluator records **1,517 samples**, MAE **$157,378.25**, RMSE **$311,910.03**, and R² **0.4404**. |

The qualitative outcome is a multimodal property-price prototype with authentication, history, trend APIs, and Docker/local operation. The NLP result is component-specific; uncommitted tabular/CNN held-out sets prevent a valid claim that multimodality improves accuracy or that the system is production-deployed.

## What It Does

- Predicts listing prices from structured property features.
- Adds NLP and image-based predictions when text or images are supplied.
- Stores per-user prediction and chat history in SQLite by default.
- Provides market trend and distribution APIs from pre-aggregated CSV files.
- Runs locally, in Docker, or on a platform such as Render.
- Ships with pytest coverage, CI, MLOps documentation, and governance files.

## Repository Map

```text
.
├── app/                  # Flask package: routes, forms, models, services
├── data/                 # Aggregated market CSV files used by dashboard charts
├── docs/                 # Architecture, MLOps, runbooks, assessment notes, wireframes
├── instance/             # Local SQLite/runtime files; not for committed secrets
├── models/               # Model artifacts and model metadata
├── nlp_data/             # NLP raw/processed data snapshots for notebook workflows
├── notebooks/            # Numbered EDA, cleaning, feature engineering, and modelling notebooks
├── reports/              # Metrics, generated reports, and figures
├── scripts/              # Utility scripts for project checks and operations
├── src/                  # Reproducible pipeline scripts from notebooks
├── static/               # CSS and web assets
├── templates/            # Jinja templates
└── tests/                # Pytest suite
```

Every meaningful folder has its own `README.md` so reviewers can enter from any directory and understand what belongs there.

## Quick Start

Use Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
copy .env.example .env
flask --app app:create_app --debug run
```

On macOS/Linux, replace the activate/copy commands with:

```bash
source .venv/bin/activate
cp .env.example .env
```

Open http://localhost:5000.

## Environment

Start from `.env.example`. Required production values:

- `SECRET_KEY`: long random string.
- `DATABASE_URL`: production database URI, preferably Postgres.
- `MODEL_DIR`: folder containing model artifacts.
- `UPLOAD_FOLDER`: runtime upload directory.

Optional values:

- `OPENAI_API_KEY`: enables OpenAI-backed chat replies.
- `OPENAI_MODEL`: model used by the housing chat assistant.
- `TREND_TIMESERIES_PATH`, `TREND_HISTOGRAM_PATH`, `TREND_DATA_URL`: market data sources.
- `WEB_CONCURRENCY`, `WEB_TIMEOUT`, `PORT`: Gunicorn/runtime tuning.

## Testing

```bash
python -m ruff check .
python -m ruff format --check .
python -m bandit -r app scripts -q -ll
python -m pip_audit --local
python scripts/check_project.py --ci
python -m pytest
```

The pytest suite covers authentication, prediction validation, history isolation, chat guardrails, route behavior, serialization, and compatibility of all three deployed model artifacts. The current suite contains 61 tests and enforces at least 55% statement coverage for the Flask package.

The committed NLP model was re-evaluated on 1,517 held-out examples with TensorFlow CPU 2.21.0: MAE **$157,378.25**, RMSE **$311,910.03**, and R² **0.4404**. Reproduce the result with `python scripts/evaluate_nlp_model.py`; full provenance is in `reports/metrics/nlp-metrics.json`.

## Docker

```bash
docker build -t estatescope-ai .
docker run --env-file .env -p 5000:5000 estatescope-ai
```

Local Compose:

```bash
docker compose up --build
```

## CI And MLOps

GitHub Actions lives in `.github/workflows/ci.yml`. It installs constrained dependencies, verifies the dependency graph, audits known vulnerabilities, enforces Ruff/Google-style docstrings, scans application code with Bandit, compiles Python files, validates artifact integrity and repository size, runs pytest with coverage, smoke-tests the WSGI entry point, and builds and health-checks the production container.

GitHub community files are included:

- Issue templates for bugs, features, and MLOps tasks.
- Pull request template with validation checklist.
- Dependabot configuration for Python and GitHub Actions dependencies.
- `CITATION.cff`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`, `GOVERNANCE.md`, and `SUPPORT.md`.
- `.gitattributes` and `.gitignore` for cleaner cross-platform commits.

MLOps guidance is split under `docs/mlops/`:

- Model lifecycle and promotion process.
- Model card for intended use and limitations.
- Data card for dataset risks and lineage.
- Monitoring plan for drift, quality, and service health.

## Important Notes

- `.env` is intentionally ignored. Use `.env.example` as the template.
- Do not upload `.env`, local `instance/*.db` files, cache folders, or generated coverage files to GitHub.
- The hardened repository contains 154 tracked files totalling about 313.83 MiB; seven declared artifacts are at least 10 MiB. They are checksum-pinned in `artifacts-manifest.json`, and CI rejects any tracked file over 95 MiB. New large artifacts should use Git LFS, release assets, or external object storage.
- The default SQLite setup is for local development and CA demonstration; use Postgres for production-style deployment.
- TensorFlow is loaded lazily so basic tests and Flask startup are not blocked when optional NLP/CNN dependencies are absent.

## License

MIT License. See `LICENSE`.
