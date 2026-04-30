"""High-coverage route and form tests for the housing app.

These tests focus on:
1. Authentication flows (signup, login, access control).
2. Prediction API validation and success path with stubbed model service.
3. Auxiliary APIs (cities, price trend/distribution) including error fallbacks.
4. Chat API behavior (housing-only gating, fallback replies without OpenAI).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pytest
from flask import Flask

from app.extensions import db


def _register(client, username: str = "user1") -> None:
    """Helper to create a user via signup."""

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


def _login(client, username: str = "user1") -> None:
    """Helper to log in an existing user."""

    client.post(
        "/login",
        data={"username": username, "password": "secret123"},
        follow_redirects=True,
    )


def test_app_factory_uses_override(app: Flask) -> None:
    """Ensure create_app respects testing configuration."""

    assert app.config["TESTING"] is True
    assert app.config["WTF_CSRF_ENABLED"] is False
    assert "sqlite" in app.config["SQLALCHEMY_DATABASE_URI"]


def test_signup_requires_terms(client) -> None:
    """Signup should fail when terms are not accepted."""

    response = client.post(
        "/signup",
        data={
            "username": "no_terms",
            "email": "no_terms@example.com",
            "password": "secret123",
            # terms omitted
        },
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Please accept the terms" in response.data


def test_login_invalid_credentials(client) -> None:
    """Login should show an error for bad credentials."""

    response = client.post(
        "/login",
        data={"username": "ghost", "password": "badpass"},
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"Invalid credentials" in response.data


def test_dashboard_requires_auth(client) -> None:
    """Dashboard should redirect unauthenticated users to login."""

    response = client.get("/dashboard")
    assert response.status_code in (301, 302)
    assert "/login" in response.headers.get("Location", "")


def test_cities_api_returns_state_filtered(client, monkeypatch) -> None:
    """Cities API should respect state filtering and fall back to all when missing."""

    _register(client)
    _login(client)

    listing_counts = {
        "state_to_cities": {"texas": {"austin": 10, "dallas": 5}},
        "state_listing_count": {"texas": 2},
    }

    class DummyService:
        def __init__(self) -> None:
            self.listing_counts = listing_counts

    monkeypatch.setattr("app.routes._get_model_service", lambda: DummyService())

    resp = client.get("/api/cities?state=texas")
    assert resp.status_code == 200
    cities = resp.get_json()
    assert "austin" in cities
    assert "dallas" in cities


def test_price_trend_api_success_and_error(client, monkeypatch) -> None:
    """Price trend endpoint should return data and degrade gracefully on errors."""

    _register(client)
    _login(client)

    monkeypatch.setattr(
        "app.routes._load_price_trend",
        lambda state=None, city=None: [{"date": "2024-01-01", "mean": 100.0, "median": 90.0, "count": 1}],
    )
    ok = client.get("/api/price-trend")
    assert ok.status_code == 200
    assert ok.get_json()[0]["mean"] == 100.0

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("app.routes._load_price_trend", _boom)
    err = client.get("/api/price-trend")
    assert err.status_code == 200
    assert err.get_json() == []


def test_price_distribution_api_success_and_error(client, monkeypatch) -> None:
    """Price distribution endpoint should return histogram data or empty on failure."""

    _register(client)
    _login(client)

    monkeypatch.setattr(
        "app.routes._load_price_distribution",
        lambda: {"edges": [0, 1], "counts": [1]},
    )
    ok = client.get("/api/price-distribution")
    assert ok.status_code == 200
    assert ok.get_json()["counts"] == [1]

    monkeypatch.setattr("app.routes._load_price_distribution", lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    err = client.get("/api/price-distribution")
    assert err.status_code == 200
    assert err.get_json() == {"edges": [], "counts": []}


def test_predict_validation_and_success(client, monkeypatch) -> None:
    """Prediction API should validate required fields and succeed with a stubbed model."""

    _register(client)
    _login(client)

    # With missing fields, expect validation error.
    bad = client.post("/api/predict", data={})
    assert bad.status_code == 400

    listing_counts = {
        "state_to_cities": {"texas": {"austin": 3}},
        "state_listing_count": {"texas": 1},
    }

    class DummyService:
        def __init__(self) -> None:
            self.listing_counts = listing_counts

        def predict(self, structured_payload=None, description=None, image_paths=None):
            return {"tabular": 321000.0}, 321000.0

    monkeypatch.setattr("app.routes._get_model_service", lambda: DummyService())

    good = client.post(
        "/api/predict",
        data={
            "city": "austin",
            "state": "texas",
            "status": "for_sale",
            "bedrooms": 3,
            "bathrooms": 2,
            "living_area": 1500,
            "lot_size": 6000,
        },
    )
    assert good.status_code == 200
    payload = good.get_json()
    assert payload["final_price"] == 321000.0
    assert payload["predictions"]["tabular"] == 321000.0


def test_chat_housing_guard_and_fallback(client) -> None:
    """Chat API should gate non-housing queries and respond to housing prompts without OpenAI."""

    # Must be logged in to use chat
    client.post(
        "/signup",
        data={"username": "chatuser", "email": "chat@example.com", "password": "secret123", "terms": "y"},
        follow_redirects=True,
    )
    client.post("/login", data={"username": "chatuser", "password": "secret123"}, follow_redirects=True)

    # Non-housing question should be rejected politely.
    guard = client.post("/api/chat", json={"message": "Tell me a joke"})
    assert guard.status_code == 200
    assert "housing" in guard.get_json()["reply"].lower()

    # Housing question should return a useful reply (fallback, no OpenAI).
    resp = client.post("/api/chat", json={"message": "What is a fair price for a 3 bed house?"})
    assert resp.status_code == 200
    reply = resp.get_json()["reply"]
    assert reply
    assert isinstance(reply, str)
