"""Model-layer behavior tests."""

from __future__ import annotations

from datetime import UTC, datetime
from math import inf, nan

from app.extensions import db
from app.models import Prediction, User


def test_user_password_hash_and_check(app):
    """User password hashing and verification should behave correctly."""

    with app.app_context():
        user = User(username="hash", email="hash@example.com")
        user.set_password("secret123")
        assert user.check_password("secret123") is True
        assert user.check_password("wrong") is False


def test_user_loader_bad_id_returns_none(app):
    """User loader should return None for invalid ids."""

    from app.models import load_user

    with app.app_context():
        assert load_user(None) is None
        assert load_user("notanint") is None or load_user("9999999") is None


def test_prediction_to_dict_handles_nan_inf(app):
    """Prediction serialization should drop NaN/inf values."""

    with app.app_context():
        pred = Prediction(
            user_id=1,
            tabular_price=nan,
            nlp_price=inf,
            cnn_price=123.0,
            final_price=456.0,
        )
        pred.created_at = datetime.now(UTC)
        data = pred.to_dict()
        assert data["tabular_price"] is None
        assert data["nlp_price"] is None
        assert data["cnn_price"] == 123.0
        assert data["final_price"] == 456.0


def test_prediction_created_at_timezone(app):
    """Timestamps should be timezone-aware."""

    with app.app_context():
        pred = Prediction(user_id=1, final_price=1.0)
        db.session.add(pred)
        db.session.flush()
        assert pred.created_at.tzinfo is not None
