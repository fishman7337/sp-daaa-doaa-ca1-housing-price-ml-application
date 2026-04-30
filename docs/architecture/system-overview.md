# System Overview

## Runtime Components

- Flask application factory in `app/__init__.py`.
- Routes and API handlers in `app/routes.py`.
- SQLAlchemy models in `app/models.py`.
- WTForms validation in `app/forms.py`.
- Model orchestration in `app/services/model_service.py`.
- Feature preprocessing helpers in `app/services/preprocessing.py`.
- Jinja templates in `templates/`.
- Static CSS and media assets in `static/`.

## Request Flow

1. User signs up or logs in through Flask-Login.
2. Dashboard loads state/city choices from model metadata.
3. Prediction form validates structured fields server-side and client-side.
4. Uploaded images are stored under the configured upload folder.
5. `ModelService` runs available modalities: tabular, NLP, and/or image.
6. The ensemble price is persisted to `Prediction`.
7. History and chart APIs return user-scoped or aggregate data to the frontend.

## Data Flow

Training and preparation work begins in notebooks and is mirrored by scripts in `src/`. Deployment consumes curated artifacts from `models/` and aggregated chart CSVs from `data/`.

## Operational Boundaries

- SQLite is the local default and is appropriate for demonstration.
- Production-style deployment should use Postgres through `DATABASE_URL`.
- TensorFlow is optional at import time and loaded only when NLP/CNN paths are used.
- OpenAI chat is optional and falls back to local template responses when no API key is present.
