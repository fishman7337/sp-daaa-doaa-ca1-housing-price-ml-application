# MLOps Playbook

## Lifecycle

1. Explore in notebooks using numbered, reproducible steps.
2. Promote stable logic into `src/` pipeline scripts.
3. Save metrics into `reports/metrics/`.
4. Save generated plots into `reports/figures/`.
5. Export production artifacts into `models/`.
6. Update the model card and data card.
7. Run CI before deployment.

## Artifact Promotion

A model artifact should not be promoted unless it has:

- Training data source and date.
- Feature set description.
- Evaluation split and metrics.
- Known limitations.
- Compatible dependency versions.
- Rollback artifact or previous known-good version.

## Rollback

Keep the previous model artifact available until the new artifact is validated in a deployed environment. If predictions fail, restore the previous artifact and redeploy with the same environment variables.

## Reproducibility

Use deterministic random seeds where possible. Record package versions with `python -m pip freeze > reports/metrics/environment-freeze.txt` when finalizing a model run.
