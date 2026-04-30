"""Additional prediction field coverage."""

from __future__ import annotations

from io import BytesIO

import pytest


def _login_with_counts(client, monkeypatch):
    """Register/login and stub model service with listing counts."""

    client.post(
        "/signup",
        data={
            "username": "preduser",
            "email": "preduser@example.com",
            "password": "secret123",
            "terms": "y",
        },
        follow_redirects=True,
    )
    client.post(
        "/login",
        data={"username": "preduser", "password": "secret123"},
        follow_redirects=True,
    )

    class DummyService:
        def __init__(self):
            self.listing_counts = {
                "state_to_cities": {"tx": {"austin": 5}},
                "state_listing_count": {"tx": 1},
            }

        def predict(self, structured_payload=None, description=None, image_paths=None):
            return {"tabular": 111000.0}, 111000.0

    monkeypatch.setattr("app.routes._get_model_service", lambda: DummyService())


def test_predict_requires_all_structured_fields(client, monkeypatch):
    """Missing required structured fields should yield 400."""

    _login_with_counts(client, monkeypatch)
    resp = client.post(
        "/api/predict",
        data={
            "city": "austin",
            "state": "tx",
            "status": "for_sale",
            "bedrooms": 3,
            # bathrooms missing
            "living_area": 1200,
            "lot_size": 5000,
        },
    )
    assert resp.status_code == 400


def test_predict_with_optional_description_and_image(client, monkeypatch, tmp_path):
    """Prediction should accept optional description and image uploads."""

    _login_with_counts(client, monkeypatch)

    fake_image = (BytesIO(b"fakepng"), "house.png")
    resp = client.post(
        "/api/predict",
        data={
            "city": "austin",
            "state": "tx",
            "status": "for_sale",
            "bedrooms": 3,
            "bathrooms": 2,
            "living_area": 1500,
            "lot_size": 6000,
            "description": "Nice home with upgrades.",
            "image": [fake_image],
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["final_price"] == 111000.0
    assert payload["predictions"]["tabular"] == 111000.0
