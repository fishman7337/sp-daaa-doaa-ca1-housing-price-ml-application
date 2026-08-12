# Model Card

## Model Family

EstateScope AI uses an ensemble of available modalities:

- Tabular structured regressor from engineered property fields.
- NLP model for listing descriptions.
- CNN model for uploaded property imagery.

The final prediction is the mean of available modality predictions.

## Intended Use

The model supports academic demonstration of DevOps, automation, validation, deployment, and MLOps practices for an AI-enabled Flask application. Predictions are estimates and should not be treated as financial, legal, or professional appraisal advice.

## Inputs

- City and state.
- Listing status.
- Bedrooms, bathrooms, living area, and lot size.
- Optional text description.
- Optional property images.

## Outputs

- Per-modality predicted prices.
- Final ensemble price.
- Persisted prediction history for the logged-in user.

## Limitations

- Dataset coverage and recency constrain prediction quality.
- US market dynamics vary by region and time.
- Uploaded images may not represent property condition accurately.
- Text descriptions can be biased, incomplete, or promotional.
- The simple ensemble does not calibrate modality confidence.

## Evaluation Expectations

Record MAE, RMSE, R2, data split, feature list, and artifact checksum for each promoted tabular model. For NLP/CNN models, include validation loss or MAE and the dataset size.

## Verified Evaluation Snapshot

The committed NLP artifact was evaluated on 11 August 2026 against the committed 1,517-example test split using TensorFlow CPU 2.21.0:

- MAE: $157,378.25.
- RMSE: $311,910.03.
- R²: 0.4404.
- Target range: $52,500 to $6,000,000.
- Prediction range: $218,860.30 to $1,463,931.38.

See `reports/metrics/nlp-metrics.json` and reproduce with `python scripts/evaluate_nlp_model.py`. Comparable held-out tabular and CNN evaluation datasets are not committed, so this repository does not claim verified MAE/RMSE/R² for those artifacts.
