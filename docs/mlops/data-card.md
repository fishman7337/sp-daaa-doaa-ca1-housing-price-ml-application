# Data Card

## Data Sources

The project includes aggregated housing market CSVs under `data/` and NLP snapshots under `nlp_data/`. Larger raw data and generated artifacts should be stored outside source control unless required for assessment packaging.

## Data Uses

- Training and evaluation in notebooks and `src/` scripts.
- City/state counts for dropdown choices and feature engineering.
- Aggregated chart data for market trend and distribution APIs.

## Quality Checks

Recommended checks before modelling:

- Missing values by column.
- Duplicate listings.
- Invalid prices, lot sizes, and living areas.
- State and city normalization.
- Outlier handling policy.
- Train/validation/test split leakage review.

## Privacy And Licensing

Do not include private user records, credentials, or data that violates source licensing. If a dataset license changes, update this file and the root README before publishing or deploying.
