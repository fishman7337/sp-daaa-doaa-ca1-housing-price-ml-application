"""Entrypoint for running the Flask application."""

from __future__ import annotations

from app import create_app

app = create_app()


def main() -> None:
    """Run the local server without exposing Flask's debugger."""
    app.run(host="127.0.0.1", port=int(app.config["PORT"]), debug=False)


if __name__ == "__main__":
    main()
