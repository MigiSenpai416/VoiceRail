import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Every test gets its own empty data directory."""
    monkeypatch.setenv("VOICERAIL_HOME", str(tmp_path / "home"))
    yield tmp_path / "home"


def assets_home() -> Path | None:
    """A data directory with downloaded base models, for integration tests."""
    path = os.environ.get("VOICERAIL_TEST_ASSETS")
    return Path(path) if path else None
