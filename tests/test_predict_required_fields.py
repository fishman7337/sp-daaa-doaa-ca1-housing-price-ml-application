"""Prediction required-field validation and status tests."""

from __future__ import annotations

import pytest


def _login_with_stub(client, monkeypatch):
    client.post(
        "/signup",
        data={
            "username": "requser",
            "email": "requser@example.com",
            "password": "secret123",
            "terms": "y",
        },
        follow_redirects=True,
    )
    client.post(
        "/login",
        data={"username": "requser", "password": "secret123"},
        follow_redirects=True,
    )

    class DummyService:
        def __init__(self):
            self.listing_counts = {
                "state_to_cities": {"tx": {"austin": 1}},
                "state_listing_count": {"tx": 1},
            }

        def predict(self, structured_payload=None, description=None, image_paths=None):
            return {"tabular": 202000.0}, 202000.0

    monkeypatch.setattr("app.routes._get_model_service", lambda: DummyService())


@pytest.mark.parametrize(
    "missing_field",
    ["city", "state", "status", "bedrooms", "bathrooms", "living_area", "lot_size"],
)
def test_predict_missing_required_field(client, monkeypatch, missing_field):
    """Each required field missing should yield a 400 response."""

    _login_with_stub(client, monkeypatch)
    payload = {
        "city": "austin",
        "state": "tx",
        "status": "for_sale",
        "bedrooms": 3,
        "bathrooms": 2,
        "living_area": 1200,
        "lot_size": 5000,
    }
    payload.pop(missing_field)
    resp = client.post("/api/predict", data=payload)
    assert resp.status_code == 400


def test_predict_invalid_status_choice(client, monkeypatch):
    """Invalid status should be rejected by the SelectField choices."""

    _login_with_stub(client, monkeypatch)
    resp = client.post(
        "/api/predict",
        data={
            "city": "austin",
            "state": "tx",
            "status": "invalid_status",
            "bedrooms": 3,
            "bathrooms": 2,
            "living_area": 1200,
            "lot_size": 5000,
        },
    )
    assert resp.status_code == 400


def test_predict_accepts_large_numeric_values(client, monkeypatch):
    """Large but positive numeric values should validate."""

    _login_with_stub(client, monkeypatch)
    resp = client.post(
        "/api/predict",
        data={
            "city": "austin",
            "state": "tx",
            "status": "for_sale",
            "bedrooms": 10,
            "bathrooms": 8,
            "living_area": 12000,
            "lot_size": 200000,
        },
    )
    assert resp.status_code == 200
    assert resp.get_json()["final_price"] == 202000.0
