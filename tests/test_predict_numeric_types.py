"""Numeric field validation tests for prediction."""

from __future__ import annotations

import pytest


def _login_with_stub(client, monkeypatch):
    client.post(
        "/signup",
        data={
            "username": "numuser",
            "email": "numuser@example.com",
            "password": "secret123",
            "terms": "y",
        },
        follow_redirects=True,
    )
    client.post(
        "/login",
        data={"username": "numuser", "password": "secret123"},
        follow_redirects=True,
    )

    class DummyService:
        def __init__(self):
            self.listing_counts = {
                "state_to_cities": {"tx": {"austin": 1}},
                "state_listing_count": {"tx": 1},
            }

        def predict(self, structured_payload=None, description=None, image_paths=None):
            return {"tabular": 101000.0}, 101000.0

    monkeypatch.setattr("app.routes._get_model_service", lambda: DummyService())


@pytest.mark.parametrize(
    "bad_field,bad_value",
    [
        ("bedrooms", "abc"),
        ("bathrooms", "abc"),
        ("living_area", "abc"),
        ("lot_size", "abc"),
        ("bedrooms", ""),
        ("bathrooms", ""),
        ("living_area", ""),
        ("lot_size", ""),
    ],
)
def test_predict_rejects_non_numeric_fields(client, monkeypatch, bad_field, bad_value):
    """All numeric fields must be numbers; non-numeric should 400."""

    _login_with_stub(client, monkeypatch)
    payload = {
        "city": "austin",
        "state": "tx",
        "status": "for_sale",
        "bedrooms": 3,
        "bathrooms": 2,
        "living_area": 1200,
        "lot_size": 4000,
    }
    payload[bad_field] = bad_value
    resp = client.post("/api/predict", data=payload)
    assert resp.status_code == 400


def test_predict_accepts_numeric_strings(client, monkeypatch):
    """Numeric strings that can coerce should pass (WTForms handles coercion)."""

    _login_with_stub(client, monkeypatch)
    resp = client.post(
        "/api/predict",
        data={
            "city": "austin",
            "state": "tx",
            "status": "for_sale",
            "bedrooms": "3",
            "bathrooms": "2.5",
            "living_area": "1500",
            "lot_size": "5000",
        },
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["final_price"] == 101000.0
