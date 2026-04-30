# Local Development

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
copy .env.example .env
```

## Run

```bash
flask --app app:create_app --debug run
```

Open http://localhost:5000.

## Notes

- Local SQLite files live in `instance/`.
- Model artifacts are loaded from `MODEL_DIR`.
- OpenAI-backed chat is disabled until `OPENAI_API_KEY` is set.
