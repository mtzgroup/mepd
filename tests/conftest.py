"""Shared test setup."""

import pytest


@pytest.fixture(autouse=True)
def _private_web_state(tmp_path, monkeypatch):
    """Every test gets its own mepd web state folder (the "recent sessions"
    list, ~/.config/mepd by default). A test that creates a web app would
    otherwise add its temporary workspace to the user's real list: every run
    of the suite left more "ws" entries in the Open-session dialog."""
    monkeypatch.setenv("MEPD_WEB_STATE_DIR", str(tmp_path / "_mepd_web_state"))


def pytest_configure(config):
    config.addinivalue_line("markers", "real_programs: use mepd.programs' real lookup (and allow its download)")


@pytest.fixture(autouse=True)
def _no_real_programs(request, monkeypatch):
    """Tests never pick up this machine's g-xTB (its PATH entries, ~/.local/opt)
    and never download one: they see g-xTB only through $GXTB_EXECUTABLE, as
    before mepd.programs existed. Many fake `subprocess.run` for the plain
    `gxtb` command; a resolved real path would be wrapped (setpriv) past them.
    Tests of mepd.programs itself opt out with @pytest.mark.real_programs."""
    if request.node.get_closest_marker("real_programs"):
        return
    import os

    import mepd.programs as programs

    monkeypatch.setenv("MEPD_NO_DOWNLOAD", "1")
    monkeypatch.setattr(programs, "find_gxtb", lambda: os.environ.get("GXTB_EXECUTABLE") or None)
