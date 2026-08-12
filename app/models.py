"""Database models for authentication and prediction history."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from flask import current_app, has_app_context
from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db, login_manager


class User(UserMixin, db.Model):
    """User account for authentication."""

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(UTC))

    predictions = db.relationship(
        "Prediction", backref="user", lazy=True, cascade="all, delete-orphan"
    )
    chat_messages = db.relationship(
        "ChatMessage", backref="user", lazy=True, cascade="all, delete-orphan"
    )

    def set_password(self, password: str) -> None:
        """Hash and store a password."""
        method = "scrypt"
        if has_app_context() and current_app.config.get("TESTING"):
            method = "pbkdf2:sha256:1000"
        self.password_hash = generate_password_hash(password, method=method)

    def check_password(self, password: str) -> bool:
        """Verify a password against the stored hash."""
        return check_password_hash(self.password_hash, password)


class Prediction(db.Model):
    """Model to store per-user prediction history."""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(UTC))

    structured_payload = db.Column(db.JSON, nullable=True)
    description = db.Column(db.Text, nullable=True)
    image_path = db.Column(db.String(255), nullable=True)

    tabular_price = db.Column(db.Float, nullable=True)
    nlp_price = db.Column(db.Float, nullable=True)
    cnn_price = db.Column(db.Float, nullable=True)
    final_price = db.Column(db.Float, nullable=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize prediction for API responses."""

        def _clean_num(val: Any) -> float | None:
            try:
                f = float(val)
                if math.isnan(f) or math.isinf(f):
                    return None
                return f
            except (TypeError, ValueError):
                return None

        return {
            "id": self.id,
            "created_at": self.created_at.isoformat(),
            "tabular_price": _clean_num(self.tabular_price),
            "nlp_price": _clean_num(self.nlp_price),
            "cnn_price": _clean_num(self.cnn_price),
            "final_price": _clean_num(self.final_price),
            "structured_payload": self.structured_payload or {},
            "description": self.description or "",
            "image_path": self.image_path or "",
        }


class ChatMessage(db.Model):
    """Per-user chat transcripts for the housing assistant."""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    role = db.Column(db.String(16), nullable=False)  # "user" or "assistant"
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize chat message."""
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat() if self.created_at else "",
        }


@login_manager.user_loader
def load_user(user_id: str) -> User | None:
    """flask-login loader."""
    if user_id is None:
        return None
    try:
        return db.session.get(User, int(user_id))
    except Exception:
        return None


login_manager.login_view = "main.login"
login_manager.login_message_category = "warning"

__all__ = ["User", "Prediction", "ChatMessage", "load_user"]
