"""Flask application factory for the multimodal housing price predictor."""

from __future__ import annotations

from pathlib import Path

from flask import Flask
from sqlalchemy import event

from .config import get_config
from .extensions import csrf, db, login_manager
from .routes import main_bp


def create_app(config_override: dict | None = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        config_override: Optional runtime configuration overrides
            (useful for testing).

    Returns:
        Configured Flask application instance.
    """
    base_dir = Path(__file__).resolve().parent.parent
    templates = base_dir / "templates"
    static_dir = base_dir / "static"

    app = Flask(
        __name__,
        template_folder=str(templates),
        static_folder=str(static_dir),
    )
    app.config.from_object(get_config())
    # Optionally load config.cfg (e.g., to override SQLALCHEMY_DATABASE_URI)
    cfg_path = base_dir / "config.cfg"
    if cfg_path.exists():
        app.config.from_pyfile(str(cfg_path), silent=True)

    if config_override:
        app.config.update(config_override)

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    login_manager.init_app(app)
    csrf.init_app(app)

    with app.app_context():
        # Enable more resilient SQLite locking behavior when using the dev SQLite DB.
        def _set_sqlite_pragma(dbapi_connection, _):
            import sqlite3

            if isinstance(dbapi_connection, sqlite3.Connection):
                cursor = dbapi_connection.cursor()
                try:
                    cursor.execute("PRAGMA journal_mode=WAL;")
                    cursor.execute("PRAGMA synchronous=NORMAL;")
                    cursor.execute("PRAGMA busy_timeout=10000;")
                except sqlite3.OperationalError:
                    # If the DB is already locked during startup, skip pragma changes.
                    pass
                finally:
                    cursor.close()

        event.listen(db.engine, "connect", _set_sqlite_pragma)  # type: ignore[arg-type]
        db.create_all()

    app.register_blueprint(main_bp)
    return app


__all__ = ["create_app"]
