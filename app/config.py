"""Application configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Type

from sqlalchemy.pool import NullPool

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional during minimal installs
    load_dotenv = None

if load_dotenv:
    load_dotenv()


BASE_DIR = Path(__file__).resolve().parent.parent


def _path_from_env(name: str, default: Path) -> str:
    """Resolve path-like environment variables relative to the project root."""

    configured = os.environ.get(name)
    if not configured:
        return str(default)

    path = Path(configured)
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path)


def _sqlite_uri(db_name: str) -> str:
    """Build a cross-platform SQLite URI."""

    db_path = (BASE_DIR / "instance" / db_name).as_posix()
    return f"sqlite:///{db_path}"


class Config:
    """Base configuration class."""

    BASE_DIR = BASE_DIR
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-me")
    SQLITE_DB_NAME = os.environ.get("SQLITE_DB_NAME", "local_db.db")
    SQLALCHEMY_DATABASE_URI = (
        os.environ.get("DATABASE_URL")
        or os.environ.get("SQLALCHEMY_DATABASE_URI")
        or _sqlite_uri(SQLITE_DB_NAME)
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    _base_engine_options = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }
    if SQLALCHEMY_DATABASE_URI.startswith("sqlite"):
        SQLALCHEMY_ENGINE_OPTIONS = {
            **_base_engine_options,
            "poolclass": NullPool,
            "connect_args": {"check_same_thread": False, "timeout": 60},
        }
    else:
        SQLALCHEMY_ENGINE_OPTIONS = _base_engine_options

    UPLOAD_FOLDER = _path_from_env("UPLOAD_FOLDER", BASE_DIR / "static" / "uploads")
    MODEL_DIR = _path_from_env("MODEL_DIR", BASE_DIR / "models")
    TREND_TIMESERIES_PATH = _path_from_env(
        "TREND_TIMESERIES_PATH",
        BASE_DIR / "data" / "usa_real_estate_price_time_series.csv",
    )
    TREND_HISTOGRAM_PATH = _path_from_env(
        "TREND_HISTOGRAM_PATH",
        BASE_DIR / "data" / "usa_real_estate_price_histogram.csv",
    )
    TREND_DATA_PATH = _path_from_env(
        "TREND_DATA_PATH",
        BASE_DIR / "data" / "processed" / "usa_real_estate_clean.csv",
    )
    TREND_DATA_URL = os.environ.get("TREND_DATA_URL", "")
    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    PORT = int(os.environ.get("PORT", "5000"))
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH_MB", "16")) * 1024 * 1024
    WTF_CSRF_TIME_LIMIT = None


def get_config() -> Type[Config]:
    """Return the active configuration class."""

    return Config


__all__ = ["Config", "get_config"]
