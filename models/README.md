# Models

Model artifacts and small model metadata files live here.

Expected artifacts:

- `best_regressor.joblib`: tabular model (37.21 MiB).
- `best_rnn_model_full.keras`: NLP model (53.12 MiB, input length 154).
- `nlp_tokenizer.json`: tokenizer for NLP model.
- `tuned_cnn_best.keras`: image model (80.17 MiB, input shape 224 × 224 × 3).
- `city_listing_summary.csv`, `state_listing_summary.csv`, or `city_state_counts.json`: lookup metadata.

Exact byte sizes and SHA-256 checksums are stored in `artifacts-manifest.json` and verified by CI. Record promoted model details in `docs/mlops/model-card.md` and metrics in `reports/metrics/`.
