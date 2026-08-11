"""Broad validation, range, and failure-mode tests."""

from __future__ import annotations

from io import BytesIO


def _login_and_stub_service(client, monkeypatch, username: str = "matrix"):
    """Register/login and provide a stub model service with listing counts."""

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

    class DummyService:
        def __init__(self):
            self.listing_counts = {
                "state_to_cities": {"tx": {"austin": 3}},
                "state_listing_count": {"tx": 1},
            }

        def predict(self, structured_payload=None, description=None, image_paths=None):
            return {"tabular": 999000.0}, 999000.0

    monkeypatch.setattr("app.routes._get_model_service", lambda: DummyService())


def test_history_requires_login(client):
    """History endpoint should redirect unauthenticated users (auth guard)."""

    resp = client.get("/api/history")
    assert resp.status_code in (301, 302)
    assert "/login" in resp.headers.get("Location", "")


def test_predict_negative_range_rejected(client, monkeypatch):
    """Range testing: negative numeric fields should be rejected (400)."""

    _login_and_stub_service(client, monkeypatch, "rangeuser")
    resp = client.post(
        "/api/predict",
        data={
            "city": "austin",
            "state": "tx",
            "status": "for_sale",
            "bedrooms": -1,  # invalid
            "bathrooms": 2,
            "living_area": 1200,
            "lot_size": 4000,
        },
    )
    assert resp.status_code == 400


def test_predict_city_not_in_choices(client, monkeypatch):
    """Consistency testing: city must be one of the allowed choices for the state."""

    _login_and_stub_service(client, monkeypatch, "choiceuser")
    resp = client.post(
        "/api/predict",
        data={
            "city": "houston",  # not in stubbed choices
            "state": "tx",
            "status": "for_sale",
            "bedrooms": 3,
            "bathrooms": 2,
            "living_area": 1200,
            "lot_size": 4000,
        },
    )
    assert resp.status_code == 400


def test_predict_unexpected_model_error(client, monkeypatch):
    """Unexpected failure testing: model exceptions should surface as 500."""

    client.post(
        "/signup",
        data={
            "username": "crash",
            "email": "crash@example.com",
            "password": "secret123",
            "terms": "y",
        },
        follow_redirects=True,
    )
    client.post(
        "/login",
        data={"username": "crash", "password": "secret123"},
        follow_redirects=True,
    )

    class BoomService:
        def __init__(self):
            self.listing_counts = {
                "state_to_cities": {"tx": {"austin": 1}},
                "state_listing_count": {"tx": 1},
            }

        def predict(self, *args, **kwargs):
            raise RuntimeError("kaboom")

    monkeypatch.setattr("app.routes._get_model_service", lambda: BoomService())
    resp = client.post(
        "/api/predict",
        data={
            "city": "austin",
            "state": "tx",
            "status": "for_sale",
            "bedrooms": 3,
            "bathrooms": 2,
            "living_area": 1200,
            "lot_size": 4000,
        },
    )
    assert resp.status_code == 500


def test_predict_rejects_non_image_upload(client, monkeypatch):
    """Expected failure testing: non-image upload should be rejected by validators."""

    _login_and_stub_service(client, monkeypatch, "fileuser")
    fake_txt = (BytesIO(b"not an image"), "note.txt")
    resp = client.post(
        "/api/predict",
        data={
            "city": "austin",
            "state": "tx",
            "status": "for_sale",
            "bedrooms": 3,
            "bathrooms": 2,
            "living_area": 1200,
            "lot_size": 4000,
            "image": [fake_txt],
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
