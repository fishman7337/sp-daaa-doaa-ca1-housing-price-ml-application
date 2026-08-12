"""Lightweight API and auth tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import create_app
from app.extensions import db


@pytest.fixture()
def client(tmp_path: Path):
    config = {
        "TESTING": True,
        "WTF_CSRF_ENABLED": False,
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.db'}",
        "MODEL_DIR": str(tmp_path / "models"),
        "UPLOAD_FOLDER": str(tmp_path / "uploads"),
    }
    app = create_app(config_override=config)
    with app.app_context():
        db.create_all()
    with app.test_client() as client:
        yield client, app


def _register_and_login(client, username: str = "alice") -> dict[str, str]:
    client.post(
        "/signup",
        data={
            "username": username,
            "email": f"{username}@example.com",
            "password": "secret123",
            "terms": "y",
        },
        follow_redirects=True,
    )
    client.post(
        "/login",
        data={"username": username, "password": "secret123"},
        follow_redirects=True,
    )
    return {"username": username}


def test_dashboard_requires_login(client):
    client, _app = client
    response = client.get("/dashboard")
    assert response.status_code in (301, 302)


def test_signup_and_login_flow(client):
    client, _app = client
    _register_and_login(client, "bob")
    response = client.get("/dashboard", follow_redirects=True)
    assert response.status_code == 200
    assert b"Multimodal prediction" in response.data


def test_api_predict_validation_error(client):
    client, _app = client
    _register_and_login(client, "carol")
    resp = client.post("/api/predict", data={})
    assert resp.status_code == 400


def test_api_predict_success_with_stub(client, monkeypatch):
    client, _app = client
    _register_and_login(client, "dave")

    class DummyService:
        def __init__(self):
            self.listing_counts = {
                "state_to_cities": {"tx": {"austin": 10}},
                "state_listing_count": {"tx": 1},
            }

        def predict(self, structured_payload=None, description=None, image_paths=None):
            del structured_payload, description, image_paths
            return {"tabular": 123456.0}, 123456.0

    monkeypatch.setattr("app.routes._get_model_service", lambda: DummyService())

    resp = client.post(
        "/api/predict",
        data={
            "city": "austin",
            "state": "tx",
            "status": "for_sale",
            "bedrooms": 3,
            "bathrooms": 2,
            "living_area": 1500,
            "lot_size": 0.2,
            "year_built": 2010,
        },
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["final_price"] == 123456.0
