# MLOps

EstateScope AI treats notebooks as exploration and `src/` scripts as the reproducible path. Model artifacts are promoted only when they are traceable to a dataset version, training script or notebook, metrics, and documented limitations.

Key files:

- `mlops-playbook.md`: lifecycle, promotion, rollback, and release expectations.
- `model-card.md`: intended use, limitations, and evaluation notes.
- `data-card.md`: dataset lineage, quality risks, and privacy notes.
- `monitoring.md`: service, data, and model monitoring plan.
