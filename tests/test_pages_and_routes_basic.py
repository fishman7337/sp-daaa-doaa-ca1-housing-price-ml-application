"""Basic page route tests."""

from __future__ import annotations


def test_public_pages_accessible(client):
    """Public pages should return 200."""

    for path in ("/", "/login", "/signup"):
        resp = client.get(path)
        assert resp.status_code == 200


def test_dashboard_after_login_shows_form(client):
    """Dashboard should render after login and include the prediction form."""

    client.post(
        "/signup",
        data={
            "username": "pageuser",
            "email": "pageuser@example.com",
            "password": "secret123",
            "terms": "y",
        },
        follow_redirects=True,
    )
    client.post(
        "/login",
        data={"username": "pageuser", "password": "secret123"},
        follow_redirects=True,
    )
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert b"Predict" in resp.data
