# Deployment

## Docker

```bash
docker build -t estatescope-ai .
docker run --env-file .env -p 5000:5000 estatescope-ai
```

## Compose

```bash
docker compose up --build
```

## Platform Deployment

Set these environment variables in the platform dashboard:

- `SECRET_KEY`
- `DATABASE_URL`
- `MODEL_DIR`
- `UPLOAD_FOLDER`
- `OPENAI_API_KEY` when chat should call OpenAI
- `PORT` when the platform provides a dynamic port

## Pre-Deployment Checklist

- CI passes.
- `.env` secrets are configured outside source control.
- Model artifacts exist in `models/` or are mounted at `MODEL_DIR`.
- Database is migrated or initialized.
- Upload directory is writable.
- Monitoring and rollback plan are documented.
- `docker inspect --format='{{.State.Health.Status}}' <container>` reaches `healthy`.

The image runs as the non-root `appuser` (UID 10001) and declares a health check against the public landing route. The `instance/` and `static/uploads/` directories remain writable for SQLite and user uploads.
