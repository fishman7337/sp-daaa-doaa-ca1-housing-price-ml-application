# Testing

## Full Test Suite

```bash
python -m pytest
```

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

TensorFlow builds used by this project require NumPy below 2.0. Keep the pin in `requirements.txt` unless the TensorFlow dependency is upgraded and verified.
