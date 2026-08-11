"""Additional tests for history endpoints and signup validation."""

from __future__ import annotations


def _signup_and_login(client, username: str = "histuser") -> None:
    """Create an account and log in."""

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


def test_signup_rejects_invalid_email(client) -> None:
    """Signup should fail with malformed email address."""

    resp = client.post(
        "/signup",
        data={
            "username": "bademail",
            "email": "not-an-email",
            "password": "secret123",
            "terms": "y",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Invalid email address" in resp.data


def test_signup_rejects_short_password(client) -> None:
    """Signup should fail when password is too short."""

    resp = client.post(
        "/signup",
        data={
            "username": "shortpass",
            "email": "short@example.com",
            "password": "123",
            "terms": "y",
        },
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"Field must be at least 6 characters long" in resp.data


def test_history_list_and_delete(client, monkeypatch) -> None:
    """History endpoints should list and delete user records."""

    _signup_and_login(client, "hist1")

    class DummyService:
        def __init__(self):
            self.listing_counts = {
                "state_to_cities": {"tx": {"austin": 1}},
                "state_listing_count": {"tx": 1},
            }

        def predict(self, structured_payload=None, description=None, image_paths=None):
            return {"tabular": 222000.0}, 222000.0

    monkeypatch.setattr("app.routes._get_model_service", lambda: DummyService())

    resp = client.post(
        "/api/predict",
        data={
            "city": "austin",
            "state": "tx",
            "status": "for_sale",
            "bedrooms": 3,
            "bathrooms": 2,
            "living_area": 1400,
            "lot_size": 5500,
        },
    )
    assert resp.status_code == 200
    history_id = resp.get_json()["history_id"]

    listed = client.get("/api/history")
    assert listed.status_code == 200
    hist_items = listed.get_json()
    assert any(item["id"] == history_id for item in hist_items)

    deleted = client.delete(f"/api/history/{history_id}")
    assert deleted.status_code == 200

    # Deleting again should return 404
    deleted_again = client.delete(f"/api/history/{history_id}")
    assert deleted_again.status_code == 404


def test_history_is_user_isolated(client, monkeypatch) -> None:
    """Users should not see or delete each other's history."""

    # User one creates a record
    _signup_and_login(client, "alice")

    class DummyService:
        def __init__(self):
            self.listing_counts = {
                "state_to_cities": {"tx": {"austin": 1}},
                "state_listing_count": {"tx": 1},
            }

        def predict(self, structured_payload=None, description=None, image_paths=None):
            return {"tabular": 210000.0}, 210000.0

    monkeypatch.setattr("app.routes._get_model_service", lambda: DummyService())
    resp = client.post(
        "/api/predict",
        data={
            "city": "austin",
            "state": "tx",
            "status": "for_sale",
            "bedrooms": 3,
            "bathrooms": 2,
            "living_area": 1400,
            "lot_size": 5500,
        },
    )
    history_id = resp.get_json()["history_id"]

    # Log out and sign up a different user
    client.get("/logout")
    _signup_and_login(client, "bob")

    listed = client.get("/api/history")
    ids = [item["id"] for item in listed.get_json()]
    assert history_id not in ids

    denied = client.delete(f"/api/history/{history_id}")
    assert denied.status_code == 404
