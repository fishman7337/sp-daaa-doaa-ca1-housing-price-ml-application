"""WSGI entrypoint for production servers (e.g., gunicorn, Render)."""

from __future__ import annotations

from app import create_app

app = create_app()

# Optional: enable app to run directly for quick sanity checks.
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
