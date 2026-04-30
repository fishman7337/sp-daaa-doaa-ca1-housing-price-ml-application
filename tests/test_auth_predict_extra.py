"""Additional authentication and prediction tests."""

from __future__ import annotations

import pytest

from app.extensions import db
from app.models import User


def _signup(client, username: str, email: str) -> None:
    client.post(
        "/signup",
        data={
            "username": username,
            "email": email,
            "password": "secret123",
            "terms": "y",
        },
        follow_redirects=True,
    )


def test_signup_duplicate_username(client, app) -> None:
    """Signup should reject duplicate usernames."""

    _signup(client, "dup", "dup1@example.com")
    response = client.post(
        "/signup",
        data={
            "username": "dup",
            "email": "dup2@example.com",
            "password": "secret123",
            "terms": "y",
        },
        follow_redirects=True,
    )
    assert b"Username already exists" in response.data


def test_signup_duplicate_email(client, app) -> None:
    """Signup should reject duplicate emails."""

    _signup(client, "userA", "dup@example.com")
    response = client.post(
        "/signup",
        data={
            "username": "userB",
            "email": "dup@example.com",
            "password": "secret123",
            "terms": "y",
        },
        follow_redirects=True,
    )
    assert b"Email already registered" in response.data


def test_login_success_redirects_dashboard(client) -> None:
    """Login should redirect to dashboard on success."""

    _signup(client, "logme", "logme@example.com")
    resp = client.post(
        "/login",
        data={"username": "logme", "password": "secret123"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Multimodal prediction" in resp.data


def test_logout_requires_login(client) -> None:
    """Logout should redirect when not logged in."""

    resp = client.get("/logout", follow_redirects=False)
    assert resp.status_code in (301, 302)


def test_dashboard_after_login(client) -> None:
    """Dashboard should load after successful login."""

    _signup(client, "dash", "dash@example.com")
    client.post(
        "/login",
        data={"username": "dash", "password": "secret123"},
        follow_redirects=True,
    )
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert b"Prediction ready" in resp.data or b"Multimodal prediction" in resp.data

