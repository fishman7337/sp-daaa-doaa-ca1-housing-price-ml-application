# Testing

## Full Test Suite

```bash
python -m pytest
```

The suite currently contains 61 tests and must retain at least 55% statement coverage across `app/`.

## Quality And Security Gates

```bash
python -m ruff check .
python -m ruff format --check .
python -m bandit -r app scripts -q -ll
python -m pip_audit --local
python -m compileall app src tests scripts
python scripts/check_project.py --ci
```

Ruff excludes stateful notebooks but validates application code, reusable pipelines, scripts, and tests. Bandit fails on medium/high findings; low findings remain visible during a full local scan.

## Structure Check

```bash
python scripts/check_project.py
```

## What The Tests Cover

- Signup and login flows.
- Required prediction fields and numeric validation.
- Successful prediction path with stubbed model service.
- User-scoped prediction history.
- User-scoped chat history.
- Chat domain guardrails.
- API fallback behavior.
- Serialization of non-finite model outputs.

## Dependency Note

TensorFlow CPU 2.21 is verified against both committed Keras artifacts. NumPy remains below 2.0 for model compatibility. On Windows without long-path support, create the virtual environment at a short path (for example, `C:\venvs\estatescope`) before installing TensorFlow.

## Reproduce The NLP Metric

```bash
python scripts/evaluate_nlp_model.py
```

This evaluates 1,517 committed held-out samples. The reviewed result and runtime versions are stored in `reports/metrics/nlp-metrics.json`.
