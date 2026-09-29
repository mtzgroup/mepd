"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _private_web_state(tmp_path, monkeypatch):
    """Every test gets its own mepd web state folder (the "recent sessions"
    list, ~/.config/mepd by default). A test that creates a web app would
    otherwise add its temporary workspace to the user's real list: every run
    of the suite left more "ws" entries in the Open-session dialog."""
    monkeypatch.setenv("MEPD_WEB_STATE_DIR", str(tmp_path / "_mepd_web_state"))
