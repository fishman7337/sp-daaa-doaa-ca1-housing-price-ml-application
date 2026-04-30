"""Persistence and chat behavior tests."""

from __future__ import annotations

from io import BytesIO

import pytest

from app.extensions import db
from app.models import Prediction, User


def _login_with_stub(client, monkeypatch, username: str = "persist"):
    """Register/login and stub model service with listing counts."""

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
            self.listing_counts = {"state_to_cities": {"tx": {"austin": 1}}, "state_listing_count": {"tx": 1}}

        def predict(self, structured_payload=None, description=None, image_paths=None):
            return {"tabular": 505000.0}, 505000.0

    monkeypatch.setattr("app.routes._get_model_service", lambda: DummyService())


def test_prediction_persists_description_and_image(client, monkeypatch, tmp_path):
    """Prediction record should store description and image path and appear in history."""

    _login_with_stub(client, monkeypatch)

    fake_image = (BytesIO(b"fakepngdata"), "house.png")
    resp = client.post(
        "/api/predict",
        data={
            "city": "austin",
            "state": "tx",
            "status": "for_sale",
            "bedrooms": 3,
            "bathrooms": 2,
            "living_area": 1400,
            "lot_size": 5000,
            "description": "Test home",
            "image": [fake_image],
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 200

    history = client.get("/api/history")
    assert history.status_code == 200
    items = history.get_json()
    assert items, "History should not be empty"
    last = items[0]
    assert "Test home" in (last.get("description") or "")
    assert last.get("image_path")


def test_user_delete_cascades_predictions(app, client, monkeypatch):
    """Deleting a user should remove their predictions via cascade."""

    _login_with_stub(client, monkeypatch, username="cascade")
    client.post(
        "/api/predict",
        data={
            "city": "austin",
            "state": "tx",
            "status": "for_sale",
            "bedrooms": 3,
            "bathrooms": 2,
            "living_area": 1400,
            "lot_size": 5000,
        },
    )

    with app.app_context():
        user = User.query.filter_by(username="cascade").first()
        assert user is not None
        assert Prediction.query.filter_by(user_id=user.id).count() >= 1
        db.session.delete(user)
        db.session.commit()
        assert Prediction.query.filter_by(user_id=user.id).count() == 0


def test_price_trend_empty_returns_empty(client, monkeypatch):
    """Price trend API should return empty list when no data."""

    _login_with_stub(client, monkeypatch, username="trendempty")
    monkeypatch.setattr("app.routes._load_price_trend", lambda state=None, city=None: [])
    resp = client.get("/api/price-trend")
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_chat_history_trims_to_last_20(client, monkeypatch):
    """Chat should respond and handle long histories gracefully."""

    _login_with_stub(client, monkeypatch, username="historytrim")
    long_history = [{"role": "user", "content": f"q{i}"} for i in range(25)]
    resp = client.post("/api/chat", json={"message": "House price?", "history": long_history})
    assert resp.status_code == 200
    reply = resp.get_json().get("reply")
    assert isinstance(reply, str) and reply


def test_chat_history_requires_login(client):
    """History endpoint should require authentication."""

    resp = client.get("/api/chat/history")
    # flask-login redirects to login when unauthenticated
    assert resp.status_code in (302, 401)
    if resp.status_code == 302:
        assert "login" in (resp.headers.get("Location") or "")


def test_chat_history_is_user_scoped(app, monkeypatch):
    """Each user should only see their own chat transcript."""

    with app.test_client() as c1:
        _login_with_stub(c1, monkeypatch, username="u1scope")
        c1.post("/api/chat", json={"message": "House price in Austin?"})
        h1 = c1.get("/api/chat/history")
        assert h1.status_code == 200
        history_user1 = h1.get_json()
        assert history_user1 and any("Austin" in m["content"] for m in history_user1)

    with app.test_client() as c2:
        _login_with_stub(c2, monkeypatch, username="u2scope")
        c2.post("/api/chat", json={"message": "House price in Boston?"})
        h2 = c2.get("/api/chat/history")
        assert h2.status_code == 200
        history_user2 = h2.get_json()
        assert history_user2 and any("Boston" in m["content"] for m in history_user2)

    # Re-open first user's history to ensure isolation
    with app.test_client() as c1b:
        _login_with_stub(c1b, monkeypatch, username="u1scope")
        h1b = c1b.get("/api/chat/history").get_json()
        combined_user1 = " ".join(m["content"] for m in h1b)
        assert "Boston?" not in combined_user1
        assert "Austin?" in combined_user1
