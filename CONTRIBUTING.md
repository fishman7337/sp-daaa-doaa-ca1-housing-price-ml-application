# Contributing

## Development Flow

1. Create a focused branch for one change.
2. Install development dependencies with `python -m pip install -r requirements-dev.txt`.
3. Copy `.env.example` to `.env` and fill only local values.
4. Run `python -m pytest` before submitting changes.
5. Update docs when behavior, setup, model artifacts, data paths, or deployment steps change.

## Quality Bar

- Keep code changes scoped and readable.
- Do not commit `.env`, local databases, coverage files, cache folders, or generated uploads.
- Prefer deterministic scripts for data and model workflows.
- Record model metrics and assumptions in `reports/metrics/` and `docs/mlops/`.
- Keep user-facing validation and server-side validation aligned.

## Pull Request Checklist

- Tests pass locally.
- New or changed environment variables are reflected in `.env.example`.
- Model/data artifact changes are documented.
- Security or privacy implications are called out.
- README files remain accurate for touched folders.
