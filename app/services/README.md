# Services

Service modules isolate model and preprocessing logic from Flask routes.

- `model_service.py`: lazy-loads model artifacts and runs tabular, NLP, image, and ensemble predictions.
- `preprocessing.py`: normalizes city/state values and creates the deployed tabular feature row.

Keep route handlers thin; add reusable business logic here.
