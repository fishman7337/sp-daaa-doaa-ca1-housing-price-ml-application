"""Shared pytest fixtures for the Flask application tests."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Tuple

import pytest

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app
from app.extensions import db


@pytest.fixture(scope="function")
def app(tmp_path: Path) -> Generator:
    """Create a Flask app instance configured for testing."""

    config = {
        "TESTING": True,
        "WTF_CSRF_ENABLED": False,
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.db'}",
        "MODEL_DIR": str(tmp_path / "models"),
        "UPLOAD_FOLDER": str(tmp_path / "uploads"),
    }
    test_app = create_app(config_override=config)
    with test_app.app_context():
        db.drop_all()
        db.create_all()
    yield test_app


@pytest.fixture(scope="function")
def client(app) -> Generator:
    """Provide a Flask test client with application context."""

    with app.test_client() as client:
        yield client


@pytest.fixture(autouse=True)
def _clear_openai_env(monkeypatch) -> None:
    """Remove OPENAI_API_KEY for tests to avoid network calls."""

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Ensure consistent locale for predictable formatting if needed.
    os.environ["LC_ALL"] = "C"
