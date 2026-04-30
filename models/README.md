# Models

Model artifacts and small model metadata files live here.

Expected artifacts:

- `best_regressor.joblib`: tabular model.
- `best_rnn_model_full.keras`: NLP model.
- `nlp_tokenizer.json`: tokenizer for NLP model.
- `tuned_cnn_best.keras`: image model.
- `city_listing_summary.csv`, `state_listing_summary.csv`, or `city_state_counts.json`: lookup metadata.

Record promoted model details in `docs/mlops/model-card.md` and metrics in `reports/metrics/`.
