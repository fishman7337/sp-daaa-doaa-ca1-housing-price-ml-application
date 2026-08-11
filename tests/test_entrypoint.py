"""Tests for the local Flask entry point."""

from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any


def test_local_entrypoint_uses_safe_server_defaults(monkeypatch: Any) -> None:
    """Keep the development debugger private and disabled by default."""
    namespace = runpy.run_path(
        str(Path(__file__).parents[1] / "app.py"),
        run_name="estatescope_entrypoint",
    )
    application = namespace["app"]
    observed: dict[str, object] = {}
    monkeypatch.setattr(application, "run", lambda **kwargs: observed.update(kwargs))

    namespace["main"]()

    assert observed == {
        "host": "127.0.0.1",
        "port": int(application.config["PORT"]),
        "debug": False,
    }
