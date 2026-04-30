# Monitoring

## Service Health

Track:

- HTTP status codes.
- Prediction endpoint latency.
- Chat endpoint latency.
- Database errors.
- Upload failures.

## Data Quality

Track:

- Missing required fields.
- Invalid numeric values.
- State/city choice misses.
- Upload file type rejections.
- Distribution shift in bedrooms, bathrooms, living area, and lot size.

## Model Quality

When ground truth becomes available, compare predicted price against actual sale price and record MAE/RMSE by region and property type. Flag large sustained error increases for retraining review.

## Alerting

For a production-style deployment, alert on:

- Repeated prediction failures.
- Database connection failures.
- High 5xx rate.
- Missing model artifacts.
- Unexpected increase in fallback-only predictions.
