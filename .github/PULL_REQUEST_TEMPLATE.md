## Summary

- 

## Type Of Change

- [ ] Documentation
- [ ] Bug fix
- [ ] Feature
- [ ] Test/CI
- [ ] MLOps/model/data update

## Validation

- [ ] `python scripts/check_project.py --ci`
- [ ] `python -m ruff check .`
- [ ] `python -m ruff format --check .`
- [ ] `python -m bandit -r app scripts -q -ll`
- [ ] `python -m pip_audit --local`
- [ ] `python -m compileall app src tests scripts`
- [ ] `python -m pytest`

## Notes

Mention any changed environment variables, model artifacts, data files, or deployment assumptions.
