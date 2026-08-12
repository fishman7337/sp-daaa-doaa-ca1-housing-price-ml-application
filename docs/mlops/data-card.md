# Data Card

## Data Sources

The project includes aggregated housing market CSVs under `data/` and NLP snapshots under `nlp_data/`. Larger raw data and generated artifacts should be stored outside source control unless required for assessment packaging.

## Data Uses

- Training and evaluation in notebooks and `src/` scripts.
- City/state counts for dropdown choices and feature engineering.
- Aggregated chart data for market trend and distribution APIs.

## Committed Snapshot

- Raw Austin housing data: 15,171 rows × 47 columns across 9 named cities.
- Cleaned NLP data: 15,169 rows × 3 columns.
- NLP split: 12,135 training, 1,517 validation, and 1,517 test examples; each token sequence has length 154.
- Dashboard time series: 816 monthly aggregate rows from January 1901 through April 2026.
- Dashboard histogram source: 200,000 price observations.
- Location lookup summaries: 20,071 city rows and 56 state/territory rows.

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
